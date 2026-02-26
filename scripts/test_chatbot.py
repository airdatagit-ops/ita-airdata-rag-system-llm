"""
Test script for the Chatbot API.

This script demonstrates how to use the chatbot endpoints:
- Basic chat without RAG
- Chat with RAG (retrieval-augmented generation)
- Session management
- Streaming chat

Run this after starting the API server.
"""

import requests
import json
import time
from typing import Optional


class ChatbotClient:
    """Client for interacting with the Aviation RAG Chatbot API."""

    def __init__(self, base_url: str = "http://127.0.0.1:8083", api_key: str = None):
        """
        Initialize chatbot client.

        Args:
            base_url: Base URL of the API
            api_key: API key for authentication
        """
        self.base_url = base_url
        self.api_key = api_key or "L2B7zzrla7pV0Ro2Bc5ipwxf3-H8EKwxWz0kX1qMsa0"
        self.headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }

    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        use_rag: bool = False,
        rag_date: Optional[str] = None,
        context_window: int = 10,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> dict:
        """
        Send a chat message.

        Args:
            message: User message
            session_id: Session ID for conversation continuity
            use_rag: Whether to use RAG for context
            rag_date: Date for RAG temporal search (YYYY-MM-DD)
            context_window: Number of previous messages to include
            temperature: LLM temperature override
            max_tokens: Max tokens override

        Returns:
            Response dictionary
        """
        url = f"{self.base_url}/api/chat"

        payload = {
            "message": message,
            "use_rag": use_rag,
            "context_window": context_window
        }

        if session_id:
            payload["session_id"] = session_id
        if rag_date:
            payload["rag_date"] = rag_date
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        response = requests.post(url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()

    def chat_stream(
        self,
        message: str,
        session_id: Optional[str] = None,
        use_rag: bool = False,
        rag_date: Optional[str] = None,
        context_window: int = 10,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Send a chat message with streaming response.

        Args:
            message: User message
            session_id: Session ID for conversation continuity
            use_rag: Whether to use RAG for context
            rag_date: Date for RAG temporal search
            context_window: Number of previous messages
            temperature: LLM temperature
            max_tokens: Max tokens

        Yields:
            Response chunks as they arrive
        """
        url = f"{self.base_url}/api/chat/stream"

        payload = {
            "message": message,
            "use_rag": use_rag,
            "context_window": context_window
        }

        if session_id:
            payload["session_id"] = session_id
        if rag_date:
            payload["rag_date"] = rag_date
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        response = requests.post(
            url,
            headers=self.headers,
            json=payload,
            stream=True
        )
        response.raise_for_status()

        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = json.loads(line[6:])
                    yield data

    def get_session_info(self, session_id: str) -> dict:
        """Get information about a session."""
        url = f"{self.base_url}/api/chat/session/{session_id}"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def delete_session(self, session_id: str) -> dict:
        """Delete a session."""
        url = f"{self.base_url}/api/chat/session/{session_id}"
        response = requests.delete(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def clear_session(self, session_id: str) -> dict:
        """Clear session history."""
        url = f"{self.base_url}/api/chat/session/{session_id}/clear"
        response = requests.post(url, headers=self.headers)
        response.raise_for_status()
        return response.json()

    def get_session_stats(self) -> dict:
        """Get statistics about all sessions."""
        url = f"{self.base_url}/api/chat/sessions/stats"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()


def example_basic_chat():
    """Example: Basic chat without RAG."""
    print("\n" + "="*60)
    print("Example 1: Basic Chat (No RAG)")
    print("="*60)

    client = ChatbotClient()

    # First message (creates new session)
    print("\nUser: Olá! Como você pode me ajudar?")
    response = client.chat(
        message="Olá! Como você pode me ajudar?",
        temperature=0.7
    )

    print(f"Assistant: {response['message']}")
    print(f"Session ID: {response['session_id']}")
    session_id = response['session_id']

    # Follow-up message (uses same session)
    print("\nUser: O que você sabe sobre aviação?")
    response = client.chat(
        message="O que você sabe sobre aviação?",
        session_id=session_id,
        temperature=0.7
    )

    print(f"Assistant: {response['message']}")
    print(f"Message count: {response['message_count']}")


def example_rag_chat():
    """Example: Chat with RAG for regulations."""
    print("\n" + "="*60)
    print("Example 2: Chat with RAG (Regulations)")
    print("="*60)

    client = ChatbotClient()

    # Ask about specific regulation
    print("\nUser: Quais são os requisitos para manutenção de aeronaves?")
    response = client.chat(
        message="Quais são os requisitos para manutenção de aeronaves?",
        use_rag=True,
        temperature=0.3
    )

    print(f"Assistant: {response['message']}")
    
    if response.get('sources'):
        print(f"\nSources ({len(response['sources'])}):")
        for i, source in enumerate(response['sources'][:3], 1):
            print(f"  {i}. {source['regulation_id']} (score: {source['score']:.3f})")


def example_temporal_search():
    """Example: Temporal search for regulations on a specific date."""
    print("\n" + "="*60)
    print("Example 3: Temporal RAG Search")
    print("="*60)

    client = ChatbotClient()

    # Ask about regulations valid on a specific date
    print("\nUser: Quais eram as normas sobre tripulação em maio de 2022?")
    response = client.chat(
        message="Quais eram as normas sobre tripulação em maio de 2022?",
        use_rag=True,
        rag_date="2022-05-15",
        temperature=0.3
    )

    print(f"Assistant: {response['message']}")
    print(f"Processing time: {response['processing_time_ms']}ms")


def example_streaming_chat():
    """Example: Streaming chat for real-time responses."""
    print("\n" + "="*60)
    print("Example 4: Streaming Chat")
    print("="*60)

    client = ChatbotClient()

    print("\nUser: Explique o que é um RAG system em detalhes.")
    print("Assistant: ", end='', flush=True)

    session_id = None
    sources = None

    for chunk in client.chat_stream(
        message="Explique o que é um RAG system em detalhes.",
        temperature=0.7,
        max_tokens=300
    ):
        chunk_type = chunk.get('type')
        
        if chunk_type == 'session':
            session_id = chunk.get('session_id')
        elif chunk_type == 'chunk':
            print(chunk.get('content', ''), end='', flush=True)
        elif chunk_type == 'sources':
            sources = chunk.get('sources')
        elif chunk_type == 'done':
            print("\n[Stream completed]")
        elif chunk_type == 'error':
            print(f"\n[Error: {chunk.get('error')}]")

    print(f"\nSession ID: {session_id}")


def example_session_management():
    """Example: Session management operations."""
    print("\n" + "="*60)
    print("Example 5: Session Management")
    print("="*60)

    client = ChatbotClient()

    # Create conversation
    print("\nCreating conversation...")
    response = client.chat(message="Olá!")
    session_id = response['session_id']
    print(f"Session created: {session_id}")

    # Add more messages
    client.chat(message="Como você está?", session_id=session_id)
    client.chat(message="Me fale sobre aviação", session_id=session_id)

    # Get session info
    print("\nGetting session info...")
    info = client.get_session_info(session_id)
    print(f"Message count: {info['message_count']}")
    print(f"Age: {info['age_minutes']:.2f} minutes")

    # Get global stats
    print("\nGlobal session statistics:")
    stats = client.get_session_stats()
    print(f"Active sessions: {stats['active_sessions']}")
    print(f"Total messages: {stats['total_messages']}")

    # Clear session
    print("\nClearing session history...")
    client.clear_session(session_id)
    info = client.get_session_info(session_id)
    print(f"Messages after clear: {info['message_count']}")

    # Delete session
    print("\nDeleting session...")
    client.delete_session(session_id)
    print("Session deleted successfully")


def example_conversation_flow():
    """Example: Multi-turn conversation with context."""
    print("\n" + "="*60)
    print("Example 6: Multi-turn Conversation")
    print("="*60)

    client = ChatbotClient()
    session_id = None

    conversation = [
        "Olá! Preciso de informações sobre licenças de piloto.",
        "Quais são os tipos de licença disponíveis?",
        "E quais são os requisitos para a licença de piloto comercial?",
        "Quanto tempo de voo é necessário?"
    ]

    for i, message in enumerate(conversation, 1):
        print(f"\n[Turn {i}]")
        print(f"User: {message}")

        response = client.chat(
            message=message,
            session_id=session_id,
            use_rag=(i > 1),  # Use RAG for follow-up questions
            temperature=0.5
        )

        print(f"Assistant: {response['message'][:200]}..." if len(response['message']) > 200 else f"Assistant: {response['message']}")
        
        if session_id is None:
            session_id = response['session_id']
        
        print(f"[Message count: {response['message_count']}, Time: {response['processing_time_ms']}ms]")

        time.sleep(0.5)  # Be nice to the API


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Aviation RAG Chatbot API - Test Examples")
    print("="*60)
    print("\nMake sure the API server is running at http://127.0.0.1:8083")
    print("Start with: uvicorn api.server:app --reload --host 127.0.0.1 --port 8083")

    try:
        # Run examples
        example_basic_chat()
        example_rag_chat()
        example_temporal_search()
        example_streaming_chat()
        example_session_management()
        example_conversation_flow()

        print("\n" + "="*60)
        print("✓ All examples completed successfully!")
        print("="*60)

    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API server.")
        print("Please make sure the API is running at http://127.0.0.1:8083")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
