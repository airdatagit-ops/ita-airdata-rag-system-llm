# Chatbot API Documentation

## Overview

The Aviation RAG System includes a powerful chatbot API that supports:

- **Stateful Conversations**: Maintains conversation history per session
- **RAG Integration**: Optional retrieval-augmented generation for fact-based responses
- **Temporal Search**: Query regulations valid on specific dates
- **Streaming Responses**: Real-time streaming for better UX
- **Scalable Architecture**: Session management with automatic cleanup

## Quick Start

### 1. Start the API Server

```bash
uvicorn api.server:app --reload --host 127.0.0.1 --port 8083
```

### 2. Basic Chat Request

```python
import requests

headers = {"X-API-Key": "your-api-key-here"}

response = requests.post(
    "http://127.0.0.1:8083/api/chat",
    headers=headers,
    json={
        "message": "Olá! Como você pode me ajudar?",
        "use_rag": False
    }
)

data = response.json()
print(f"Assistant: {data['message']}")
print(f"Session ID: {data['session_id']}")
```

## API Endpoints

### POST /api/chat

Main chat endpoint with conversation history.

**Request Body:**
```json
{
    "message": "string (required, max 5000 chars)",
    "session_id": "string (optional, UUID)",
    "use_rag": "boolean (default: false)",
    "rag_date": "string (optional, YYYY-MM-DD)",
    "context_window": "integer (1-50, default: 10)",
    "temperature": "float (0.0-2.0, optional)",
    "max_tokens": "integer (50-2000, optional)"
}
```

**Response:**
```json
{
    "message": "Assistant's response",
    "session_id": "UUID of the session",
    "message_count": 5,
    "sources": [
        {
            "regulation_id": "lei-8666-art-42",
            "text": "Article text...",
            "score": 0.89,
            "version": "2022-01-15"
        }
    ],
    "processing_time_ms": 3245,
    "model_used": "llama3.1:8b"
}
```

**Features:**
- Creates new session if no `session_id` provided
- Maintains conversation history
- Optional RAG for fact-based answers
- Cites sources when using RAG

**Example (Python):**
```python
# First message
response1 = requests.post(
    "http://127.0.0.1:8083/api/chat",
    headers={"X-API-Key": "your-key"},
    json={"message": "Olá!"}
)
session_id = response1.json()['session_id']

# Follow-up (maintains context)
response2 = requests.post(
    "http://127.0.0.1:8083/api/chat",
    headers={"X-API-Key": "your-key"},
    json={
        "message": "Quais são os requisitos para pilotos?",
        "session_id": session_id,
        "use_rag": True
    }
)
```

**Example (JavaScript):**
```javascript
const response = await fetch('http://127.0.0.1:8083/api/chat', {
    method: 'POST',
    headers: {
        'X-API-Key': 'your-key',
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        message: 'Quais normas sobre manutenção?',
        use_rag: true,
        temperature: 0.3
    })
});

const data = await response.json();
console.log('Assistant:', data.message);
console.log('Sources:', data.sources);
```

---

### POST /api/chat/stream

Streaming chat endpoint for real-time responses.

**Request Body:**
Same as `/api/chat`

**Response:**
Server-Sent Events (SSE) stream with the following event types:

```
data: {"session_id": "uuid", "type": "session"}

data: {"content": "text chunk", "type": "chunk"}

data: {"sources": [...], "type": "sources"}

data: {"type": "done"}

data: {"error": "error message", "type": "error"}
```

**Example (Python):**
```python
import requests
import json

response = requests.post(
    "http://127.0.0.1:8083/api/chat/stream",
    headers={"X-API-Key": "your-key"},
    json={
        "message": "Explique o que é um RAG system",
        "temperature": 0.7
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data = json.loads(line[6:])
            
            if data['type'] == 'chunk':
                print(data['content'], end='', flush=True)
            elif data['type'] == 'done':
                print('\n[Completed]')
```

**Example (JavaScript with EventSource):**
```javascript
// Note: For POST with EventSource, you need a library like eventsource
// Or use fetch with streaming:

const response = await fetch('http://127.0.0.1:8083/api/chat/stream', {
    method: 'POST',
    headers: {
        'X-API-Key': 'your-key',
        'Content-Type': 'application/json'
    },
    body: JSON.stringify({
        message: 'Explain RAG systems',
        temperature: 0.7
    })
});

const reader = response.body.getReader();
const decoder = new TextDecoder();

while (true) {
    const {value, done} = await reader.read();
    if (done) break;
    
    const chunk = decoder.decode(value);
    const lines = chunk.split('\n');
    
    for (const line of lines) {
        if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6));
            
            if (data.type === 'chunk') {
                process.stdout.write(data.content);
            }
        }
    }
}
```

---

### GET /api/chat/session/{session_id}

Get information about a session.

**Response:**
```json
{
    "session_id": "uuid",
    "message_count": 10,
    "created_at": "2026-01-25T10:30:00",
    "last_accessed": "2026-01-25T10:35:00",
    "age_minutes": 5.2,
    "metadata": {}
}
```

**Example:**
```python
response = requests.get(
    f"http://127.0.0.1:8083/api/chat/session/{session_id}",
    headers={"X-API-Key": "your-key"}
)
print(response.json())
```

---

### DELETE /api/chat/session/{session_id}

Delete a session completely.

**Response:**
```json
{
    "message": "Session deleted successfully",
    "session_id": "uuid"
}
```

**Example:**
```python
requests.delete(
    f"http://127.0.0.1:8083/api/chat/session/{session_id}",
    headers={"X-API-Key": "your-key"}
)
```

---

### POST /api/chat/session/{session_id}/clear

Clear session history without deleting the session.

**Response:**
```json
{
    "message": "Session history cleared",
    "session_id": "uuid"
}
```

**Example:**
```python
requests.post(
    f"http://127.0.0.1:8083/api/chat/session/{session_id}/clear",
    headers={"X-API-Key": "your-key"}
)
```

---

### GET /api/chat/sessions/stats

Get global session statistics.

**Response:**
```json
{
    "active_sessions": 42,
    "total_messages": 1523,
    "sessions_created": 156,
    "sessions_expired": 114,
    "ttl_minutes": 60,
    "max_sessions": 10000
}
```

**Example:**
```python
response = requests.get(
    "http://127.0.0.1:8083/api/chat/sessions/stats",
    headers={"X-API-Key": "your-key"}
)
print(response.json())
```

---

## Architecture & Scalability

### Session Management

**Features:**
- **In-memory storage** with automatic TTL (Time To Live)
- **Thread-safe** operations with RLock
- **Background cleanup** thread removes expired sessions
- **Configurable limits**: max sessions, max messages per session
- **Session timeout**: 60 minutes by default

**Configuration (in session_manager.py):**
```python
session_manager = SessionManager(
    ttl_minutes=60,                    # Session expires after 60 min
    cleanup_interval_seconds=300,       # Cleanup every 5 min
    max_messages_per_session=50,        # Keep last 50 messages
    max_sessions=10000                  # Support 10k concurrent sessions
)
```

**Memory Usage:**
- Average session: ~5-10 KB (with 50 messages)
- 10,000 sessions: ~50-100 MB RAM
- Scales horizontally with load balancer + Redis (future upgrade)

### Performance Characteristics

| Metric | Value |
|--------|-------|
| Response time (no RAG) | 2-5 seconds |
| Response time (with RAG) | 3-8 seconds |
| Streaming latency | 50-200ms first token |
| Concurrent sessions | 10,000+ |
| Messages/second | 50-100 (single instance) |

### Horizontal Scaling

For production with high load:

1. **Use Redis for session storage:**
   ```python
   # Replace in-memory with Redis
   import redis
   session_store = redis.Redis(host='localhost', port=6379)
   ```

2. **Load balancer with sticky sessions:**
   ```nginx
   upstream api_servers {
       ip_hash;  # Sticky sessions
       server api1:8083;
       server api2:8083;
       server api3:8083;
   }
   ```

3. **Separate LLM service:**
   - API servers → Message queue → LLM workers
   - Use Celery or RabbitMQ for async processing

---

## Use Cases

### 1. General Chat Assistant

```python
response = client.chat(
    message="Como você pode me ajudar?",
    temperature=0.7  # More creative
)
```

### 2. Regulation Lookup (RAG)

```python
response = client.chat(
    message="Quais são os requisitos de manutenção para aeronaves comerciais?",
    use_rag=True,
    temperature=0.3  # More factual
)

# Check sources
for source in response['sources']:
    print(f"- {source['regulation_id']}: {source['text'][:100]}...")
```

### 3. Temporal Compliance Check

```python
response = client.chat(
    message="Quais eram as normas sobre tripulação em março de 2020?",
    use_rag=True,
    rag_date="2020-03-15"
)
```

### 4. Multi-turn Conversation

```python
# Turn 1
r1 = client.chat("Preciso de informações sobre licenças de piloto")
session = r1['session_id']

# Turn 2 (context maintained)
r2 = client.chat("Quais são os tipos?", session_id=session, use_rag=True)

# Turn 3
r3 = client.chat("E os requisitos para cada tipo?", session_id=session, use_rag=True)
```

### 5. Real-time Streaming UI

```python
# Stream response for better UX
for chunk in client.chat_stream(
    message="Explique o processo de certificação de aeronaves em detalhes",
    use_rag=True
):
    if chunk['type'] == 'chunk':
        update_ui(chunk['content'])  # Update UI in real-time
```

---

## Error Handling

### Common Errors

**401 Unauthorized:**
```json
{"detail": "Invalid API key"}
```
→ Check your `X-API-Key` header

**404 Not Found:**
```json
{"detail": "Session not found or expired"}
```
→ Session expired (60 min TTL) or invalid session_id

**400 Bad Request:**
```json
{"detail": "Maximum sessions (10000) reached. Please try again later."}
```
→ Server at capacity, wait for cleanup

**429 Too Many Requests:**
```json
{"detail": "Rate limit exceeded"}
```
→ Exceeded rate limit (100 req/min), slow down

**Example Error Handling:**
```python
try:
    response = client.chat(message="Hello", session_id="invalid-uuid")
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 404:
        print("Session expired, creating new one...")
        response = client.chat(message="Hello")  # No session_id
    else:
        raise
```

---

## Integration Examples

### React Frontend

```javascript
import React, { useState } from 'react';

function ChatBot() {
    const [messages, setMessages] = useState([]);
    const [sessionId, setSessionId] = useState(null);
    const [input, setInput] = useState('');

    const sendMessage = async () => {
        // Add user message to UI
        setMessages([...messages, { role: 'user', content: input }]);

        // Send to API
        const response = await fetch('http://127.0.0.1:8083/api/chat', {
            method: 'POST',
            headers: {
                'X-API-Key': 'your-key',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: input,
                session_id: sessionId,
                use_rag: true
            })
        });

        const data = await response.json();

        // Update state
        setMessages([...messages, 
            { role: 'user', content: input },
            { role: 'assistant', content: data.message }
        ]);
        setSessionId(data.session_id);
        setInput('');
    };

    return (
        <div>
            <div className="messages">
                {messages.map((msg, i) => (
                    <div key={i} className={msg.role}>
                        {msg.content}
                    </div>
                ))}
            </div>
            <input 
                value={input} 
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
            />
            <button onClick={sendMessage}>Send</button>
        </div>
    );
}
```

### Python Desktop App

```python
import tkinter as tk
from scripts.test_chatbot import ChatbotClient

class ChatApp:
    def __init__(self):
        self.client = ChatbotClient()
        self.session_id = None
        
        self.root = tk.Tk()
        self.root.title("Aviation Chat Assistant")
        
        # Chat display
        self.chat_display = tk.Text(self.root, height=20, width=60)
        self.chat_display.pack()
        
        # Input
        self.input_field = tk.Entry(self.root, width=50)
        self.input_field.pack()
        self.input_field.bind('<Return>', self.send_message)
        
        # RAG toggle
        self.use_rag = tk.BooleanVar()
        tk.Checkbutton(self.root, text="Use RAG", variable=self.use_rag).pack()
        
        # Send button
        tk.Button(self.root, text="Send", command=self.send_message).pack()
        
    def send_message(self, event=None):
        message = self.input_field.get()
        if not message:
            return
            
        self.chat_display.insert(tk.END, f"You: {message}\n")
        
        response = self.client.chat(
            message=message,
            session_id=self.session_id,
            use_rag=self.use_rag.get()
        )
        
        self.session_id = response['session_id']
        self.chat_display.insert(tk.END, f"Assistant: {response['message']}\n\n")
        self.input_field.delete(0, tk.END)
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = ChatApp()
    app.run()
```

---

## Testing

Run the comprehensive test suite:

```bash
python scripts/test_chatbot.py
```

This will test:
- Basic chat without RAG
- Chat with RAG
- Temporal search
- Streaming
- Session management
- Multi-turn conversations

---

## Best Practices

1. **Reuse sessions**: Always pass `session_id` for follow-up questions
2. **Use RAG selectively**: Enable `use_rag=True` only for fact-based questions
3. **Set appropriate temperature**: 
   - 0.1-0.3 for factual/regulatory questions
   - 0.6-0.9 for creative/conversational responses
4. **Handle expired sessions**: Catch 404 errors and create new sessions
5. **Use streaming for long responses**: Better UX, lower perceived latency
6. **Monitor session stats**: Call `/api/chat/sessions/stats` periodically
7. **Clean up sessions**: Delete old sessions to free memory

---

## Future Enhancements

- [ ] Redis backend for distributed sessions
- [ ] WebSocket support for bi-directional streaming
- [ ] Multi-language support
- [ ] User feedback mechanism (thumbs up/down)
- [ ] Conversation export/import
- [ ] Integration with external knowledge bases
- [ ] Fine-tuning on aviation-specific conversations
