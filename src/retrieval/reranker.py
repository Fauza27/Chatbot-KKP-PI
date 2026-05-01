import os
import re

from config.settings import get_settings

settings = get_settings()
if settings.hf_token:
    os.environ.setdefault("HF_TOKEN", settings.hf_token)
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", settings.hf_token)

from sentence_transformers import CrossEncoder
from loguru import logger


def _extract_keywords(text: str) -> set[str]:
    """Extract keywords from text (lowercase, alphanumeric only)"""
    text = text.lower()
    # Remove punctuation, keep alphanumeric and spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    words = text.split()
    # Filter out common stopwords
    stopwords = {'dan', 'atau', 'yang', 'di', 'ke', 'dari', 'untuk', 'pada', 'dengan', 
                 'adalah', 'ini', 'itu', 'dalam', 'tidak', 'akan', 'bisa', 'ada', 'juga',
                 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with'}
    keywords = {w for w in words if len(w) > 2 and w not in stopwords}
    return keywords


def _calculate_keyword_boost(query: str, content: str) -> float:
    """
    Calculate keyword matching boost score.
    Returns a score between 0.0 and 1.0 based on keyword overlap.
    """
    query_keywords = _extract_keywords(query)
    content_keywords = _extract_keywords(content)
    
    if not query_keywords:
        return 0.0
    
    # Calculate overlap
    overlap = query_keywords & content_keywords
    overlap_ratio = len(overlap) / len(query_keywords)
    
    # Boost for exact phrase matches
    query_lower = query.lower()
    content_lower = content.lower()
    
    # Check for important exact matches
    exact_matches = 0
    important_phrases = [
        'kemeja putih', 'almamater', 'celana hitam', 'rok hitam', 'jilbab hitam',
        '40 halaman', 'minimal halaman', '3-5 kata kunci', '300 kata',
        'maksimal kata', 'daftar pustaka', 'minimal referensi'
    ]
    
    for phrase in important_phrases:
        if phrase in query_lower and phrase in content_lower:
            exact_matches += 1
    
    # Combine overlap ratio with exact match bonus
    exact_match_bonus = min(exact_matches * 0.2, 0.5)  # Max 0.5 bonus
    
    return min(overlap_ratio + exact_match_bonus, 1.0)


class CrossEncoderReranker:
    """
    Re-rank document using Cross-Encoder model.
    """

    _shared_model: CrossEncoder | None = None
    _shared_model_name: str | None = None

    def __init__(self, model_name: str | None = None):
        """
        Args:
            model_name: model name of cross-encoder from HuggingFace.
                       Default from settings (cross-encoder/ms-marco-MiniLM-L-6-v2)
        """
        settings = get_settings()
        self.model_name = model_name or settings.cross_encoder_model
        self.top_n = settings.rerank_top_n
        self._model: CrossEncoder | None = None

    def _get_model(self) -> CrossEncoder:
        """Lazy load cross-encoder model."""
        cls = type(self)
        if (
            cls._shared_model is None
            or cls._shared_model_name != self.model_name
        ):
            logger.info(f"Loading cross-encoder model: {self.model_name}...")
            cls._shared_model = CrossEncoder(self.model_name)
            cls._shared_model_name = self.model_name
            logger.info("Cross-encoder model loaded.")

        return cls._shared_model

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_n: int | None = None,
        content_key: str = "content",
        enable_keyword_boost: bool = True,
        keyword_boost_weight: float = 0.3,
    ) -> list[dict]:
        """
        Re-rank document based on relevance with query using Cross-Encoder.
        
        Args:
            query: Search query
            documents: List of documents to rerank
            top_n: Number of top documents to return
            content_key: Key for document content
            enable_keyword_boost: Enable keyword matching boost
            keyword_boost_weight: Weight for keyword boost (0.0-1.0)
        
        Returns:
            Reranked documents with cross_encoder_score
        """
        if not documents:
            return []

        top_n = top_n or self.top_n
        model = self._get_model()

        pairs = []
        for doc in documents:
            content = doc.get(content_key, "")
            truncated = content[:2000] if len(content) > 2000 else content
            pairs.append([query, truncated])

        logger.info(f"Cross-encoder scoring {len(pairs)} pairs...")
        scores = model.predict(pairs)

        # Apply hybrid scoring: semantic + keyword matching
        for i, (doc, semantic_score) in enumerate(zip(documents, scores)):
            if enable_keyword_boost:
                content = doc.get(content_key, "")
                keyword_score = _calculate_keyword_boost(query, content)
                
                # Combine scores: weighted average
                # Normalize semantic score to 0-1 range (assuming typical range -10 to 10)
                normalized_semantic = (float(semantic_score) + 10) / 20
                normalized_semantic = max(0.0, min(1.0, normalized_semantic))
                
                # Weighted combination
                final_score = (
                    (1 - keyword_boost_weight) * normalized_semantic +
                    keyword_boost_weight * keyword_score
                )
                
                # Scale back to semantic score range for consistency
                final_score = (final_score * 20) - 10
                
                doc["cross_encoder_score"] = float(final_score)
                doc["semantic_score"] = float(semantic_score)
                doc["keyword_score"] = float(keyword_score)
                
                if keyword_score > 0.5:
                    logger.debug(
                        f"Keyword boost applied: semantic={semantic_score:.4f}, "
                        f"keyword={keyword_score:.4f}, final={final_score:.4f}"
                    )
            else:
                doc["cross_encoder_score"] = float(semantic_score)

        documents.sort(key=lambda x: x["cross_encoder_score"], reverse=True)

        reranked = documents[:top_n]

        if reranked:
            logger.info(
                f"Reranking done: {len(documents)} → {len(reranked)} documents. "
                f"Top score: {reranked[0]['cross_encoder_score']:.4f}, "
                f"Bottom score: {reranked[-1]['cross_encoder_score']:.4f}"
            )
        else:
            logger.info(f"Reranking done: {len(documents)} → 0 documents.")

        return reranked