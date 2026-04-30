import os

from config.settings import get_settings

settings = get_settings()
if settings.hf_token:
    os.environ.setdefault("HF_TOKEN", settings.hf_token)
    os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", settings.hf_token)

from sentence_transformers import CrossEncoder
from loguru import logger


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
    ) -> list[dict]:
        """
        Re-rank document based on relevance with query using Cross-Encoder.
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

        for doc, score in zip(documents, scores):
            doc["cross_encoder_score"] = float(score)

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