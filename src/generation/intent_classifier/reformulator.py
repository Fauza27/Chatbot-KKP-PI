"""Query reformulation utilities."""

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from loguru import logger

from config.settings import get_settings
from src.generation.memory import ConversationMemory
from .constants import IMPLICIT_REFERENCE_SIGNALS, REFORMULATION_PROMPT


class QueryReformulator:
    """Reformulates queries with implicit references."""
    
    def __init__(self, llm: ChatOpenAI = None):
        if llm is None:
            settings = get_settings()
            self._llm = ChatOpenAI(
                model=settings.llm_model,
                temperature=0,
                api_key=settings.open_api_key,  
                max_tokens=100,
            )
        else:
            self._llm = llm
    
    def has_implicit_references(self, message: str) -> bool:
        """Check if message contains implicit references."""
        message_lower = message.lower()
        return any(ref in message_lower for ref in IMPLICIT_REFERENCE_SIGNALS)
    
    def reformulate_query(self, message: str, memory: ConversationMemory) -> str:
        """Reformulate query to be self-contained."""
        if memory.is_empty:
            return message
        
        if not self.has_implicit_references(message):
            return message
        
        history_text = memory.get_conversation_summary()
        
        prompt = REFORMULATION_PROMPT.format(
            history=history_text,
            question=message,
        )
        
        try:
            response = self._llm.invoke([HumanMessage(content=prompt)])
            reformulated = response.content.strip()
            
            if reformulated and reformulated != message:
                logger.info(f"🔄 Query reformulated: '{message}' → '{reformulated}'")
            
            return reformulated or message
        
        except Exception as e:
            logger.warning(f"Reformulation failed: {e} → using original query")
            return message


# Backward compatibility function
def reformulate_query(
    message: str,
    memory: ConversationMemory,
    llm: ChatOpenAI | None = None,
) -> str:
    """Legacy function for backward compatibility."""
    reformulator = QueryReformulator(llm)
    return reformulator.reformulate_query(message, memory)