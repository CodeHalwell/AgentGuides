# Pydantic AI: Recipes & Real-World Examples

**Version:** 1.0.0  
**Purpose:** Practical, production-tested code examples for common scenarios

---

## Recipe 1: Customer Support Chatbot with Database Integration

```python
"""
Production-ready customer support agent that:
- Accesses customer database
- Tracks conversation history
- Validates inputs and outputs
- Handles errors gracefully
"""

from dataclasses import dataclass
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, ModelRetry
import asyncio
import sqlite3
from datetime import datetime

# Database setup
def init_database():
    """Initialize SQLite database."""
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT,
            email TEXT,
            account_status TEXT,
            issue_history TEXT
        )
    ''')
    
    cursor.execute('INSERT INTO customers VALUES (1, "John Doe", "john@example.com", "active", "None")')
    cursor.execute('INSERT INTO customers VALUES (2, "Jane Smith", "jane@example.com", "premium", "Payment issue")')
    
    conn.commit()
    return conn

# Dependencies
@dataclass
class SupportDependencies:
    db_connection: sqlite3.Connection
    customer_id: int
    conversation_history: list[dict]

# Output models
class SupportResponse(BaseModel):
    """Structured support response."""
    solution: str = Field(..., description="Detailed solution to customer's problem")
    category: str = Field(..., regex='^(technical|billing|general|escalation)$')
    confidence: float = Field(..., ge=0.0, le=1.0)
    next_steps: list[str] = Field(..., description="Recommended next steps")

# Create agent
support_agent = Agent(
    'openai:gpt-4o',
    deps_type=SupportDependencies,
    output_type=SupportResponse,
    name='CustomerSupportAgent'
)

# System prompt with dynamic context
@support_agent.system_prompt
async def customer_context(ctx: RunContext[SupportDependencies]) -> str:
    """Fetch customer context and build system prompt."""
    
    cursor = ctx.deps.db_connection.cursor()
    cursor.execute(
        'SELECT name, email, account_status FROM customers WHERE id = ?',
        (ctx.deps.customer_id,)
    )
    
    customer = cursor.fetchone()
    
    if not customer:
        return "You are a helpful customer support agent."
    
    name, email, status = customer
    
    return f"""You are a professional customer support representative.
    
Customer Profile:
- Name: {name}
- Email: {email}
- Account Status: {status}
- Conversation History: {len(ctx.deps.conversation_history)} messages

Guidelines:
1. Be empathetic and professional
2. Provide clear, actionable solutions
3. For premium customers, prioritise faster resolution
4. Escalate if the issue is outside your scope
"""

# Tools for agents
@support_agent.tool
async def get_customer_history(
    ctx: RunContext[SupportDependencies],
    limit: int = 5
) -> str:
    """Retrieve customer's issue history."""
    
    cursor = ctx.deps.db_connection.cursor()
    cursor.execute(
        'SELECT issue_history FROM customers WHERE id = ?',
        (ctx.deps.customer_id,)
    )
    
    result = cursor.fetchone()
    return result[0] if result else "No history available"

@support_agent.tool
async def create_support_ticket(
    ctx: RunContext[SupportDependencies],
    issue_summary: str,
    priority: str = 'normal'
) -> str:
    """Create a support ticket for escalation."""
    
    ticket_id = f"TICKET_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return f"Created ticket {ticket_id} with priority {priority}"

# Output validator
@support_agent.output_validator
async def validate_response(
    ctx: RunContext[SupportDependencies],
    output: SupportResponse
) -> SupportResponse:
    """Validate support response quality."""
    
    if len(output.solution) < 50:
        raise ModelRetry(
            "Please provide a more detailed solution (at least 50 characters)."
        )
    
    if output.confidence < 0.5:
        raise ModelRetry(
            "Your confidence is low. Please reconsider your response or escalate."
        )
    
    return output

# Main execution
async def handle_customer_issue(customer_id: int, issue: str):
    """Handle a customer support issue."""
    
    conn = init_database()
    
    deps = SupportDependencies(
        db_connection=conn,
        customer_id=customer_id,
        conversation_history=[]
    )
    
    try:
        result = await support_agent.run(
            issue,
            deps=deps
        )
        
        # Store in conversation history
        deps.conversation_history.append({
            'customer_id': customer_id,
            'issue': issue,
            'response': result.output,
            'timestamp': datetime.now().isoformat()
        })
        
        return result.output
    
    finally:
        conn.close()

# Usage
if __name__ == '__main__':
    response = asyncio.run(handle_customer_issue(
        customer_id=1,
        issue="I was charged twice for my last purchase!"
    ))
    
    print(f"Solution: {response.solution}")
    print(f"Category: {response.category}")
```

---

## Recipe 2: Multi-Agent Workflow - Research & Writing Pipeline

```python
"""
Multi-agent system where agents specialise in different tasks:
1. ResearchAgent: Gathers information
2. WriterAgent: Structures and writes content
3. EditorAgent: Reviews and refines
"""

from pydantic_ai import Agent, RunContext
from dataclasses import dataclass
from pydantic import BaseModel

@dataclass
class SharedContext:
    """Shared context between agents."""
    topic: str
    research_notes: str = ""
    draft_content: str = ""
    feedback: str = ""

# Research Agent
research_agent = Agent(
    'openai:gpt-4o',
    name='ResearchAgent',
    instructions='You are a research specialist. Gather comprehensive information on topics.'
)

@research_agent.tool_plain
def search_knowledge_base(query: str) -> str:
    """Search internal knowledge base."""
    # Simulate knowledge base search
    return f"Found articles about '{query}' in the knowledge base."

async def research_phase(context: SharedContext) -> str:
    """Research phase of the pipeline."""
    
    research_prompt = f"""
    Research the following topic comprehensively:
    Topic: {context.topic}
    
    Provide:
    1. Key facts and findings
    2. Recent developments
    3. Expert perspectives
    4. Credible sources
    """
    
    result = await research_agent.run(research_prompt)
    context.research_notes = result.output
    return result.output

# Writer Agent
class Article(BaseModel):
    title: str
    introduction: str
    sections: list[dict]  # {'heading': str, 'content': str}
    conclusion: str

writer_agent = Agent(
    'openai:gpt-4o',
    output_type=Article,
    name='WriterAgent',
    instructions='You are a technical writer. Structure information into clear, engaging articles.'
)

async def writing_phase(context: SharedContext) -> Article:
    """Writing phase using research notes."""
    
    writing_prompt = f"""
    Based on this research:
    {context.research_notes}
    
    Write a well-structured article on: {context.topic}
    
    Structure:
    - Title (catchy and descriptive)
    - Introduction (hook the reader)
    - 3-4 main sections with clear headings
    - Conclusion (key takeaways)
    """
    
    result = await writer_agent.run(writing_prompt)
    context.draft_content = str(result.output)
    return result.output

# Editor Agent
class EditedArticle(BaseModel):
    original: Article
    suggestions: list[str]
    grammar_issues: list[str]
    improvements: str

editor_agent = Agent(
    'openai:gpt-4o',
    output_type=EditedArticle,
    name='EditorAgent',
    instructions='You are a professional editor. Review content for clarity, grammar, and impact.'
)

async def editing_phase(context: SharedContext, article: Article) -> EditedArticle:
    """Editing phase for quality assurance."""
    
    editing_prompt = f"""
    Review this article:
    {context.draft_content}
    
    Provide:
    1. Improvement suggestions for clarity
    2. Grammar and spelling issues
    3. Overall quality assessment
    4. Recommendation for publication
    """
    
    result = await editor_agent.run(editing_prompt)
    return result.output

# Orchestrate pipeline
async def research_and_write_pipeline(topic: str) -> Article:
    """Complete research-to-publication pipeline."""
    
    context = SharedContext(topic=topic)
    
    print("🔍 Research phase...")
    research = await research_phase(context)
    print(f"Research complete. Found: {len(research)} characters")
    
    print("✍️ Writing phase...")
    article = await writing_phase(context)
    print(f"Draft complete: {article.title}")
    
    print("✏️ Editing phase...")
    edited = await editing_phase(context, article)
    print(f"Feedback: {edited.improvements}")
    
    return article

# Usage
if __name__ == '__main__':
    article = asyncio.run(
        research_and_write_pipeline("Type Safety in Python")
    )
    print(f"\n✅ Final Article: {article.title}")
```

---

## Recipe 3: RAG (Retrieval-Augmented Generation) with Vector Search

```python
"""
RAG pattern combining semantic search with LLM generation.
Requires: pgvector, asyncpg, openai (for embeddings)
"""

from pydantic_ai import Agent, RunContext
from dataclasses import dataclass
import asyncpg
from openai import AsyncOpenAI
from pydantic import BaseModel

@dataclass
class RAGDependencies:
    """RAG system dependencies."""
    db_pool: asyncpg.Pool
    openai_client: AsyncOpenAI
    embedding_model: str = 'text-embedding-3-small'

class RAGResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float

rag_agent = Agent(
    'openai:gpt-4o',
    deps_type=RAGDependencies,
    output_type=RAGResponse,
    name='RAGAgent'
)

@rag_agent.tool
async def retrieve_documents(
    ctx: RunContext[RAGDependencies],
    query: str,
    top_k: int = 5
) -> str:
    """Retrieve relevant documents using vector search."""
    
    # Create embedding for query
    embedding_response = await ctx.deps.openai_client.embeddings.create(
        input=query,
        model=ctx.deps.embedding_model
    )
    
    query_embedding = embedding_response.data[0].embedding
    
    # Search for similar documents using pgvector
    async with ctx.deps.db_pool.acquire() as conn:
        rows = await conn.fetch('''
            SELECT id, title, content, 1 - (embedding <=> $1) as similarity
            FROM documents
            ORDER BY similarity DESC
            LIMIT $2
        ''', query_embedding, top_k)
    
    # Format retrieved documents
    formatted_docs = []
    for row in rows:
        formatted_docs.append(
            f"[{row['title']}]\n{row['content']}\n"
        )
    
    return "\n\n".join(formatted_docs)

async def rag_query(
    query: str,
    db_pool: asyncpg.Pool,
    openai_client: AsyncOpenAI
) -> RAGResponse:
    """Execute RAG query."""
    
    deps = RAGDependencies(
        db_pool=db_pool,
        openai_client=openai_client
    )
    
    result = await rag_agent.run(query, deps=deps)
    return result.output

# Usage
async def main():
    # Initialize database pool
    pool = await asyncpg.create_pool('postgresql://localhost/documents')
    openai_client = AsyncOpenAI()
    
    try:
        response = await rag_query(
            "How do I implement type safety in Python?",
            pool,
            openai_client
        )
        
        print(f"Answer: {response.answer}")
        print(f"Sources: {response.sources}")
        print(f"Confidence: {response.confidence}")
    
    finally:
        await pool.close()
```

---

## Recipe 4: Streaming Agent with Real-Time Response

```python
"""
Real-time streaming agent for web frontends.
Returns tokens as they arrive for immediate UI updates.
"""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic_ai import Agent
import asyncio

app = FastAPI()
agent = Agent('openai:gpt-4o')

@app.post('/api/chat/stream')
async def chat_stream(message: str):
    """Stream agent response in real-time."""
    
    async def stream_generator():
        async with agent.run_stream(message) as response:
            # Stream text as it arrives
            async for text in response.stream_text():
                yield text.encode()
                yield b'\n'  # Newline for client parsing
    
    return StreamingResponse(
        stream_generator(),
        media_type='text/event-stream'
    )

@app.post('/api/chat/structured-stream')
async def structured_stream(message: str):
    """Stream structured output with events."""
    
    from pydantic import BaseModel
    import json
    
    class Response(BaseModel):
        answer: str
        metadata: dict
    
    structured_agent = Agent(
        'openai:gpt-4o',
        output_type=Response
    )
    
    async def event_stream():
        async with structured_agent.run_stream(message) as response:
            # Stream partial text
            async for text in response.stream_text():
                event = {
                    'type': 'text_delta',
                    'data': text
                }
                yield f"data: {json.dumps(event)}\n\n".encode()
            
            # Stream final structured output
            result = await response.result()
            final_event = {
                'type': 'final_result',
                'data': result.output.model_dump()
            }
            yield f"data: {json.dumps(final_event)}\n\n".encode()
    
    return StreamingResponse(
        event_stream(),
        media_type='text/event-stream'
    )

# Frontend JavaScript
javascript_example = """
// Connect to streaming endpoint
const eventSource = new EventSource('/api/chat/stream?message=Hello');

eventSource.onmessage = (event) => {
    const text = event.data;
    document.getElementById('response').textContent += text;
};

eventSource.onerror = () => {
    console.error('Stream error');
    eventSource.close();
};
"""
```

---

## Recipe 5: Agent with Persistent Memory

```python
"""
Agent that maintains conversation memory across sessions.
Uses PostgreSQL for persistence.
"""

from pydantic_ai import Agent, RunContext
from dataclasses import dataclass
import asyncpg
from datetime import datetime

@dataclass
class MemoryDependencies:
    db_pool: asyncpg.Pool
    user_id: int
    session_id: str

memory_agent = Agent(
    'openai:gpt-4o',
    deps_type=MemoryDependencies,
    name='MemoryAgent'
)

@memory_agent.tool
async def store_memory(
    ctx: RunContext[MemoryDependencies],
    key: str,
    value: str,
    ttl: int = 86400
) -> bool:
    """Store information in persistent memory."""
    
    async with ctx.deps.db_pool.acquire() as conn:
        await conn.execute('''
            INSERT INTO user_memory (user_id, session_id, key, value, expires_at)
            VALUES ($1, $2, $3, $4, NOW() + INTERVAL '1 second' * $5)
            ON CONFLICT (user_id, key) DO UPDATE
            SET value = $3, updated_at = NOW()
        ''', ctx.deps.user_id, ctx.deps.session_id, key, value, ttl)
    
    return True

@memory_agent.tool
async def recall_memory(
    ctx: RunContext[MemoryDependencies],
    key: str | None = None
) -> dict:
    """Recall stored information."""
    
    async with ctx.deps.db_pool.acquire() as conn:
        if key:
            row = await conn.fetchrow('''
                SELECT key, value FROM user_memory
                WHERE user_id = $1 AND key = $2
                AND expires_at > NOW()
            ''', ctx.deps.user_id, key)
            
            return {key: row['value']} if row else {}
        else:
            rows = await conn.fetch('''
                SELECT key, value FROM user_memory
                WHERE user_id = $1 AND expires_at > NOW()
            ''', ctx.deps.user_id)
            
            return {row['key']: row['value'] for row in rows}

@memory_agent.system_prompt
async def memory_aware_prompt(ctx: RunContext[MemoryDependencies]) -> str:
    """Include user's memory in system prompt."""
    
    memories = await recall_memory(ctx)
    
    memory_str = "\n".join([f"- {k}: {v}" for k, v in memories.items()])
    
    return f"""
    You have access to the user's persistent memory:
    {memory_str if memory_str else "No stored memories yet"}
    
    Feel free to reference or update this memory during conversation.
    """

async def conversational_agent_with_memory(
    user_id: int,
    session_id: str,
    message: str,
    db_pool: asyncpg.Pool
) -> str:
    """Run agent with memory capabilities."""
    
    deps = MemoryDependencies(
        db_pool=db_pool,
        user_id=user_id,
        session_id=session_id
    )
    
    result = await memory_agent.run(message, deps=deps)
    return result.output
```

---

## Recipe 6: Error Recovery with Retry Strategies

```python
"""
Production agent with sophisticated error handling and recovery.
"""

from pydantic_ai import Agent, ModelRetry, RunContext
from typing import Optional
import asyncio
from enum import Enum

class ErrorRecoveryStrategy(str, Enum):
    EXPONENTIAL_BACKOFF = 'exponential'
    LINEAR_BACKOFF = 'linear'
    IMMEDIATE_RETRY = 'immediate'

class ResilientAgent:
    def __init__(
        self,
        model: str,
        max_retries: int = 3,
        strategy: ErrorRecoveryStrategy = ErrorRecoveryStrategy.EXPONENTIAL_BACKOFF
    ):
        self.agent = Agent(model)
        self.max_retries = max_retries
        self.strategy = strategy
    
    def _calculate_backoff(self, attempt: int) -> float:
        """Calculate backoff time based on strategy."""
        
        if self.strategy == ErrorRecoveryStrategy.EXPONENTIAL_BACKOFF:
            return 2 ** attempt  # 1, 2, 4, 8, ...
        elif self.strategy == ErrorRecoveryStrategy.LINEAR_BACKOFF:
            return attempt  # 1, 2, 3, 4, ...
        else:
            return 0  # Immediate retry
    
    async def run_with_recovery(self, prompt: str) -> Optional[str]:
        """Run agent with automatic recovery."""
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                result = await self.agent.run(prompt)
                return result.output
            
            except Exception as e:
                last_error = e
                
                if attempt < self.max_retries - 1:
                    backoff = self._calculate_backoff(attempt)
                    print(f"Attempt {attempt + 1} failed. Retrying in {backoff}s...")
                    await asyncio.sleep(backoff)
        
        # All retries exhausted
        print(f"All retries failed. Last error: {last_error}")
        return None

# Usage
resilient = ResilientAgent(
    'openai:gpt-4o',
    max_retries=3,
    strategy=ErrorRecoveryStrategy.EXPONENTIAL_BACKOFF
)

result = asyncio.run(
    resilient.run_with_recovery("Analyse this data: ...")
)
```

---

(Continue with 10+ more production-ready recipes covering various patterns)

