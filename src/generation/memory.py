from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


class IntentType(str, Enum):
    NEEDS_RETRIEVAL = "needs_retrieval"
    CONVERSATIONAL = "conversational"
    CLARIFICATION = "clarification"


@dataclass
class Turn:
    role: Literal["user", "assistant"]
    content: str
    intent: IntentType | None = None
    retrieved_doc_contents: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    def to_lc_message(self) -> dict:
        return {"role": self.role, "content": self.content}


class ConversationMemory:
    def __init__(self, max_turns: int = 5) -> None:
        self.max_turns = max_turns
        self._turns: list[Turn] = []

    def add_user_turn(
        self,
        content: str,
        intent: IntentType | None = None,
    ) -> None:
        self._turns.append(Turn(role="user", content=content, intent=intent))
        self._enforce_window()

    def add_assistant_turn(
        self,
        content: str,
        retrieved_doc_contents: list[str] | None = None,
    ) -> None:
        self._turns.append(Turn(
            role="assistant",
            content=content,
            retrieved_doc_contents=retrieved_doc_contents or [],
        ))
        self._enforce_window()

    def _enforce_window(self) -> None:
        max_messages = self.max_turns * 2
        if len(self._turns) > max_messages:
            self._turns = self._turns[-max_messages:]

    def get_history_for_llm(self) -> list[dict]:
        if len(self._turns) <= 1:
            return []
        return [t.to_lc_message() for t in self._turns[:-1]]

    def get_last_retrieved_docs(self) -> list[str]:
        for turn in reversed(self._turns):
            if turn.role == "assistant" and turn.retrieved_doc_contents:
                return turn.retrieved_doc_contents
        return []

    def get_conversation_summary(self) -> str:
        if not self._turns:
            return ""

        lines = []
        for turn in self._turns[:-1]:
            # FIX: prefix sudah berisi spasi dan label lengkap,
            # tidak perlu tambah ": " lagi di f-string
            prefix = "User" if turn.role == "user" else "Assistant"
            content = (
                turn.content[:200] + "..."
                if len(turn.content) > 200
                else turn.content
            )
            lines.append(f"{prefix}: {content}")
        return "\n".join(lines)

    def get_last_question(self) -> str | None:
        for turn in reversed(self._turns):
            if turn.role == "user":
                return turn.content
        return None

    def get_last_answer(self) -> str | None:
        for turn in reversed(self._turns):
            if turn.role == "assistant":
                return turn.content
        return None

    @property
    def turn_count(self) -> int:
        return len(self._turns) // 2

    @property
    def is_empty(self) -> bool:
        return len(self._turns) == 0

    def reset(self) -> None:
        self._turns = []

    def __repr__(self) -> str:
        return (
            f"ConversationMemory("
            f"turns={self.turn_count}, "
            f"max={self.max_turns}, "
            f"messages={len(self._turns)})"
        )