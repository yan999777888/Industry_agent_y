"""Structured in-memory session store for multi-turn dialogue."""

from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class SessionState:
    """Persistent state tracked for one chat session."""

    session_id: str
    current_product: str = ""
    current_models: list[str] = field(default_factory=list)
    current_route: str = ""
    current_service_topics: list[str] = field(default_factory=list)
    recent_questions: list[str] = field(default_factory=list)
    recent_sub_questions: list[str] = field(default_factory=list)
    recent_image_ids: list[str] = field(default_factory=list)
    recent_user_image_summaries: list[str] = field(default_factory=list)
    dialog_summary: str = ""
    history: list[dict[str, str]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    last_accessed_at: float = field(default_factory=time.time)


class InMemorySessionStore:
    """Simple in-memory store with trimming and expiry."""

    def __init__(
        self,
        *,
        max_history_turns: int = 5,
        ttl_seconds: int = 3600,
        max_recent_questions: int = 6,
        max_recent_sub_questions: int = 10,
        max_recent_images: int = 10,
        max_recent_user_image_summaries: int = 4,
    ) -> None:
        self.max_history_turns = max_history_turns
        self.ttl_seconds = ttl_seconds
        self.max_recent_questions = max_recent_questions
        self.max_recent_sub_questions = max_recent_sub_questions
        self.max_recent_images = max_recent_images
        self.max_recent_user_image_summaries = max_recent_user_image_summaries
        self._sessions: dict[str, SessionState] = {}

    def get(self, session_id: str) -> SessionState | None:
        if not session_id:
            return None
        self.cleanup_expired()
        session = self._sessions.get(session_id)
        if session is None:
            return None
        self._touch(session)
        return session

    def get_or_create(self, session_id: str) -> SessionState:
        if not session_id:
            raise ValueError("session_id must not be empty")
        session = self.get(session_id)
        if session is not None:
            return session
        now = time.time()
        session = SessionState(
            session_id=session_id,
            created_at=now,
            updated_at=now,
            last_accessed_at=now,
        )
        self._sessions[session_id] = session
        return session

    def save(self, session: SessionState) -> SessionState:
        self._trim(session)
        self._touch(session, update_timestamp=True)
        self._sessions[session.session_id] = session
        return session

    def clear(self, session_id: str) -> None:
        if not session_id:
            return
        self._sessions.pop(session_id, None)

    def append_turn(self, session: SessionState, *, user_question: str, assistant_answer: str) -> SessionState:
        session.history.append({"role": "user", "content": user_question})
        session.history.append({"role": "assistant", "content": assistant_answer})
        return self.save(session)

    def cleanup_expired(self) -> None:
        now = time.time()
        expired_ids = [
            session_id
            for session_id, session in self._sessions.items()
            if now - session.last_accessed_at > self.ttl_seconds
        ]
        for session_id in expired_ids:
            self._sessions.pop(session_id, None)

    def _touch(self, session: SessionState, *, update_timestamp: bool = False) -> None:
        now = time.time()
        session.last_accessed_at = now
        if update_timestamp:
            session.updated_at = now

    def _trim(self, session: SessionState) -> None:
        max_history_items = self.max_history_turns * 2
        if len(session.history) > max_history_items:
            session.history = session.history[-max_history_items:]
        if len(session.recent_questions) > self.max_recent_questions:
            session.recent_questions = session.recent_questions[-self.max_recent_questions:]
        if len(session.recent_sub_questions) > self.max_recent_sub_questions:
            session.recent_sub_questions = session.recent_sub_questions[-self.max_recent_sub_questions:]
        if len(session.recent_image_ids) > self.max_recent_images:
            session.recent_image_ids = session.recent_image_ids[-self.max_recent_images:]
        if len(session.recent_user_image_summaries) > self.max_recent_user_image_summaries:
            session.recent_user_image_summaries = session.recent_user_image_summaries[-self.max_recent_user_image_summaries:]
