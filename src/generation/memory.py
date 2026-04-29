# src/generation/memory.py
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

class IntentType(str, Enum):
    """
    three type of intent. used by the intent classifier for choosing what rag needs
    """
    NEEDS_RETRIEVAL = "needs_retrieval"
    CONVERSATIONAL = "conversational"
    CLARIFICATION = "clarification"

@dataclass
class Turn:
    """
    A turn in the conversation, which can be either a user input or a assistant response.
    """
    role: Literal["user", "assistant"]
    content: str
    intent: IntentType | None = None
    retrieved_doc_contents: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_lc_message(self) -> dict:
        """convert to langchain message format"""
        return {"role": self.role, "content": self.content}

class ConversationMemory:
    """
    save and manage conversation history with sliding window
    """
    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self._turns: list[Turn] = []
    
    def add_user_turn(
        self,
        content: str,
        intent: IntentType | None = None,
    ) -> None:
        """add user turn to history"""
        self._turns.append(Turn(role="user", content=content, intent=intent))
        self._enforce_window()
    
    def add_assistant_turn(
        self,
        content: str,
        retrieved_doc_contents: list[str] | None = None,
    ) -> None:
        """add assistant turn to history"""
        self._turns.append(Turn(
            role="assistant",
            content=content,
            retrieved_doc_contents=retrieved_doc_contents or [],
        ))
        self._enforce_window()
    
    def _enforce_window(self) -> None:
        """keep only max_turns x 2 last messages"""
        max_messages = self.max_turns * 2
        if len(self._turns) > max_messages:
            self._turns = self._turns[-max_messages:]
    
    def get_history_for_llm(self) -> list[dict]:
        """
        turn back the conversation history in a format suitable for LLM input.
        Excludes the last message (current user turn belum dijawab).
        """
        if len(self._turns) <= 1:
            return []
        return [t.to_lc_message() for t in self._turns[:-1]]
    
    def get_last_retrieved_docs(self) -> list[str]:
        """get retrieved docs from the last assistant turn"""
        for turn in reversed(self._turns):
            if turn.role == "assistant" and turn.retrieved_doc_contents:
                return turn.retrieved_doc_contents
        return []
    
    def get_conversation_summary(self) -> str:
        """
        make a short summary for llm context
        """
        if not self._turns:
            return ""
        
        lines = []
        for turn in self._turns[:-1]:
            prefix = "User: " if turn.role == "user" else "Assistant: "
            content = turn.content[:200] + "..." if len(turn.content) > 200 else turn.content
            lines.append(f"{prefix}: {content}")
        return "\n".join(lines)
        
    def get_last_question(self) -> str | None:  
        """get the last user question"""
        for turn in reversed(self._turns):
            if turn.role == "user":
                return turn.content
        return None
    
    def get_last_answer(self) -> str | None:
        """get the last assistant answer"""
        for turn in reversed(self._turns):
            if turn.role == "assistant":
                return turn.content
        return None
    
    @property
    def turn_count(self) -> int:
        """get the number of turns in the conversation"""
        return len(self._turns) // 2
    
    @property
    def is_empty(self) -> bool:
        """check if the conversation history is empty"""
        return len(self._turns) == 0
    
    def reset(self) -> None:
        """clear the conversation history"""
        self._turns = []
    
    def __repr__(self) -> str:
        return (
            f"ConversationMemory("
            f"turns={self.turn_count}, "
            f"max={self.max_turns}, "
            f"messages={len(self._turns)})"
        )