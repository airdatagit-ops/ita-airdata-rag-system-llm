"""
Session Manager for Chatbot Conversations.

This module manages chat sessions with conversation history, automatic cleanup,
and scalable in-memory storage with TTL (Time To Live).

Features:
- Thread-safe session storage
- Automatic session expiry (TTL)
- Background cleanup of expired sessions
- Conversation history management
- Configurable session limits

Usage:
    from api.session_manager import SessionManager

    manager = SessionManager()
    session_id = manager.create_session()
    manager.add_message(session_id, "user", "Hello!")
    messages = manager.get_messages(session_id)
"""

import time
import threading
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
from loguru import logger


class ChatSession:
    """Represents a single chat session with message history."""

    def __init__(self, session_id: str, max_messages: int = 50):
        """
        Initialize chat session.

        Args:
            session_id: Unique session identifier
            max_messages: Maximum messages to keep in history
        """
        self.session_id = session_id
        self.messages: List[Dict[str, str]] = []
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.max_messages = max_messages
        self.metadata: Dict = {}

    def add_message(self, role: str, content: str) -> None:
        """
        Add message to session history.

        Args:
            role: Message role ('user' or 'assistant')
            content: Message content
        """
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

        # Keep only last max_messages to prevent memory bloat
        if len(self.messages) > self.max_messages:
            # Remove oldest messages but keep context
            self.messages = self.messages[-self.max_messages:]

        self.last_accessed = datetime.now()

    def get_messages(self, include_timestamps: bool = False) -> List[Dict[str, str]]:
        """
        Get all messages in session.

        Args:
            include_timestamps: Whether to include timestamps

        Returns:
            List of message dictionaries
        """
        self.last_accessed = datetime.now()

        if include_timestamps:
            return self.messages.copy()
        else:
            # Return only role and content (for LLM)
            return [
                {"role": msg["role"], "content": msg["content"]}
                for msg in self.messages
            ]

    def get_context_window(self, last_n: int = 10) -> List[Dict[str, str]]:
        """
        Get recent messages for context window.

        Args:
            last_n: Number of recent messages to include

        Returns:
            List of recent messages
        """
        self.last_accessed = datetime.now()
        return self.get_messages()[-last_n:]

    def clear_history(self) -> None:
        """Clear all messages in session."""
        self.messages = []
        self.last_accessed = datetime.now()

    def is_expired(self, ttl_minutes: int) -> bool:
        """
        Check if session has expired.

        Args:
            ttl_minutes: Time to live in minutes

        Returns:
            True if session is expired
        """
        expiry_time = self.last_accessed + timedelta(minutes=ttl_minutes)
        return datetime.now() > expiry_time

    def get_message_count(self) -> int:
        """Get number of messages in session."""
        return len(self.messages)

    def get_age_minutes(self) -> float:
        """Get session age in minutes."""
        return (datetime.now() - self.created_at).total_seconds() / 60

    def to_dict(self) -> Dict:
        """Convert session to dictionary."""
        return {
            "session_id": self.session_id,
            "message_count": len(self.messages),
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "age_minutes": self.get_age_minutes(),
            "metadata": self.metadata
        }


class SessionManager:
    """
    Thread-safe manager for multiple chat sessions.

    Features:
    - Automatic session creation
    - TTL-based expiration
    - Background cleanup
    - Thread-safe operations
    """

    def __init__(
        self,
        ttl_minutes: int = 60,
        cleanup_interval_seconds: int = 300,
        max_messages_per_session: int = 50,
        max_sessions: int = 10000
    ):
        """
        Initialize session manager.

        Args:
            ttl_minutes: Session time-to-live in minutes
            cleanup_interval_seconds: How often to run cleanup
            max_messages_per_session: Max messages per session
            max_sessions: Maximum number of concurrent sessions
        """
        self.sessions: Dict[str, ChatSession] = {}
        self.ttl_minutes = ttl_minutes
        self.cleanup_interval = cleanup_interval_seconds
        self.max_messages_per_session = max_messages_per_session
        self.max_sessions = max_sessions
        self.lock = threading.RLock()

        # Statistics
        self.stats = {
            "sessions_created": 0,
            "sessions_expired": 0,
            "messages_total": 0
        }

        # Start background cleanup thread
        self._start_cleanup_thread()

        logger.info(
            f"SessionManager initialized (TTL={ttl_minutes}min, "
            f"cleanup={cleanup_interval_seconds}s, max_sessions={max_sessions})"
        )

    def create_session(self, session_id: Optional[str] = None) -> str:
        """
        Create new chat session.

        Args:
            session_id: Optional custom session ID

        Returns:
            Session ID

        Raises:
            ValueError: If max sessions reached
        """
        with self.lock:
            # Check if at capacity
            if len(self.sessions) >= self.max_sessions:
                # Try cleanup first
                self._cleanup_expired_sessions()

                if len(self.sessions) >= self.max_sessions:
                    raise ValueError(
                        f"Maximum sessions ({self.max_sessions}) reached. "
                        "Please try again later."
                    )

            # Generate session ID if not provided
            if session_id is None:
                session_id = self._generate_session_id()

            # Create session
            if session_id in self.sessions:
                logger.warning(f"Session {session_id} already exists, returning existing")
                return session_id

            session = ChatSession(
                session_id=session_id,
                max_messages=self.max_messages_per_session
            )
            self.sessions[session_id] = session
            self.stats["sessions_created"] += 1

            logger.debug(f"Created session: {session_id}")
            return session_id

    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """
        Get session by ID.

        Args:
            session_id: Session identifier

        Returns:
            ChatSession or None if not found
        """
        with self.lock:
            session = self.sessions.get(session_id)

            if session and session.is_expired(self.ttl_minutes):
                logger.debug(f"Session {session_id} expired, removing")
                del self.sessions[session_id]
                self.stats["sessions_expired"] += 1
                return None

            return session

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        create_if_missing: bool = True
    ) -> bool:
        """
        Add message to session.

        Args:
            session_id: Session identifier
            role: Message role ('user' or 'assistant')
            content: Message content
            create_if_missing: Create session if doesn't exist

        Returns:
            True if message added successfully
        """
        with self.lock:
            session = self.get_session(session_id)

            if session is None:
                if create_if_missing:
                    self.create_session(session_id)
                    session = self.sessions[session_id]
                else:
                    logger.warning(f"Session {session_id} not found")
                    return False

            session.add_message(role, content)
            self.stats["messages_total"] += 1
            return True

    def get_messages(
        self,
        session_id: str,
        include_timestamps: bool = False
    ) -> List[Dict[str, str]]:
        """
        Get all messages from session.

        Args:
            session_id: Session identifier
            include_timestamps: Include timestamps

        Returns:
            List of messages
        """
        session = self.get_session(session_id)
        if session is None:
            return []

        return session.get_messages(include_timestamps=include_timestamps)

    def get_context_window(
        self,
        session_id: str,
        last_n: int = 10
    ) -> List[Dict[str, str]]:
        """
        Get recent messages for context.

        Args:
            session_id: Session identifier
            last_n: Number of recent messages

        Returns:
            List of recent messages
        """
        session = self.get_session(session_id)
        if session is None:
            return []

        return session.get_context_window(last_n=last_n)

    def clear_session(self, session_id: str) -> bool:
        """
        Clear session history.

        Args:
            session_id: Session identifier

        Returns:
            True if cleared successfully
        """
        session = self.get_session(session_id)
        if session is None:
            return False

        session.clear_history()
        logger.debug(f"Cleared session: {session_id}")
        return True

    def delete_session(self, session_id: str) -> bool:
        """
        Delete session completely.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted successfully
        """
        with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.debug(f"Deleted session: {session_id}")
                return True
            return False

    def get_active_sessions_count(self) -> int:
        """Get number of active sessions."""
        with self.lock:
            return len(self.sessions)

    def get_stats(self) -> Dict:
        """Get session statistics."""
        with self.lock:
            return {
                "active_sessions": len(self.sessions),
                "total_messages": self.stats["messages_total"],
                "sessions_created": self.stats["sessions_created"],
                "sessions_expired": self.stats["sessions_expired"],
                "ttl_minutes": self.ttl_minutes,
                "max_sessions": self.max_sessions
            }

    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        import uuid
        return str(uuid.uuid4())

    def _cleanup_expired_sessions(self) -> int:
        """
        Remove expired sessions.

        Returns:
            Number of sessions cleaned up
        """
        with self.lock:
            expired_sessions = [
                sid for sid, session in self.sessions.items()
                if session.is_expired(self.ttl_minutes)
            ]

            for session_id in expired_sessions:
                del self.sessions[session_id]
                self.stats["sessions_expired"] += 1

            if expired_sessions:
                logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

            return len(expired_sessions)

    def _cleanup_loop(self):
        """Background cleanup loop."""
        while True:
            try:
                time.sleep(self.cleanup_interval)
                self._cleanup_expired_sessions()
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")

    def _start_cleanup_thread(self):
        """Start background cleanup thread."""
        cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True,
            name="SessionCleanup"
        )
        cleanup_thread.start()
        logger.info("Background cleanup thread started")


# ========================================
# Global Session Manager Instance
# ========================================

# Single global instance for the entire application
session_manager = SessionManager(
    ttl_minutes=60,           # 1 hour session timeout
    cleanup_interval_seconds=300,  # Cleanup every 5 minutes
    max_messages_per_session=50,   # Keep last 50 messages
    max_sessions=10000        # Support up to 10k concurrent sessions
)


# ========================================
# Example Usage
# ========================================

if __name__ == "__main__":
    """Example usage of SessionManager."""

    print("Creating session manager...")
    manager = SessionManager(ttl_minutes=5, cleanup_interval_seconds=10)

    # Create session
    session_id = manager.create_session()
    print(f"Created session: {session_id}")

    # Add messages
    manager.add_message(session_id, "user", "Hello, how are you?")
    manager.add_message(session_id, "assistant", "I'm doing well, thank you!")
    manager.add_message(session_id, "user", "What can you help me with?")

    # Get messages
    messages = manager.get_messages(session_id)
    print(f"\nMessages ({len(messages)}):")
    for msg in messages:
        print(f"  {msg['role']}: {msg['content']}")

    # Get stats
    stats = manager.get_stats()
    print(f"\nStats: {stats}")

    # Test context window
    context = manager.get_context_window(session_id, last_n=2)
    print(f"\nContext window (last 2): {len(context)} messages")

    print("\n✓ SessionManager example completed!")
