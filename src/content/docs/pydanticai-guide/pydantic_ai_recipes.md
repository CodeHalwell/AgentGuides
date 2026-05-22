---
title: "Pydantic AI: Recipes & Real-World Examples"
description: "Version: 1.101.0 Purpose: Practical, production-tested code examples for common scenarios — customer support, multi-agent pipelines, RAG, streaming, memory, error recovery, research agents, batch processing, node inspection, and XML prompt enrichment."
framework: pydanticai
---

# Pydantic AI: Recipes & Real-World Examples

**Version:** 1.101.0 (May 2026)
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

## Recipe 7: Research Agent with `common_tools` (DuckDuckGo + web fetch)

```python
"""
Research agent that searches the web then fetches and summarises the top result.
Requires: pip install "pydantic-ai-slim[duckduckgo,web-fetch]"
"""

import asyncio
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai.common_tools.web_fetch import web_fetch_tool

class ResearchSummary(BaseModel):
    topic: str
    key_findings: list[str] = Field(..., min_length=2, description='At least 2 bullet findings')
    sources_consulted: list[str] = Field(..., description='URLs fetched during research')
    confidence: float = Field(..., ge=0.0, le=1.0)

research_agent = Agent(
    'openai:gpt-4o',
    output_type=ResearchSummary,
    tools=[
        duckduckgo_search_tool(max_results=3),
        web_fetch_tool(max_content_length=4000, timeout=15),
    ],
    instructions=(
        'You are a research assistant. '
        '1. Use duckduckgo_search to find relevant URLs for the topic. '
        '2. Use web_fetch on the top 1-2 results to read the actual content. '
        '3. Synthesise findings into the structured output.'
    ),
)

async def research(topic: str) -> ResearchSummary:
    result = await research_agent.run(f'Research: {topic}')
    return result.output

if __name__ == '__main__':
    summary = asyncio.run(research('PydanticAI latest features 2026'))
    print(f'Topic: {summary.topic}')
    for finding in summary.key_findings:
        print(f'  • {finding}')
    print(f'Sources: {summary.sources_consulted}')
    print(f'Confidence: {summary.confidence:.0%}')
```

---

## Recipe 8: Concurrent batch processing with `ConcurrencyLimiter`

```python
"""
Process a large batch of items in parallel while capping model concurrency.
Useful when you have many tasks but want to avoid hitting rate limits.
"""

import asyncio
from pydantic import BaseModel
from pydantic_ai import Agent, ConcurrencyLimiter, limit_model_concurrency
from pydantic_ai.exceptions import ConcurrencyLimitExceeded

class ItemAnalysis(BaseModel):
    item_id: str
    category: str
    sentiment: str  # positive | negative | neutral
    score: float

# Shared limiter: max 5 parallel model calls, queue up to 15 more
shared_limiter = ConcurrencyLimiter(max_running=5, max_queued=15, name='batch-pool')
model = limit_model_concurrency('openai:gpt-4o', limiter=shared_limiter)
agent = Agent(model, output_type=ItemAnalysis)

async def analyse_item(item_id: str, text: str) -> ItemAnalysis | None:
    try:
        result = await agent.run(
            f'Analyse this customer feedback (item_id={item_id}): {text}'
        )
        return result.output
    except ConcurrencyLimitExceeded:
        print(f'[{item_id}] Dropped — queue full')
        return None

async def process_batch(items: list[dict]) -> list[ItemAnalysis]:
    tasks = [analyse_item(item['id'], item['text']) for item in items]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]

async def main():
    items = [
        {'id': f'item-{i}', 'text': f'Sample feedback number {i}. Great product!'}
        for i in range(30)
    ]
    analyses = await process_batch(items)
    positives = [a for a in analyses if a.sentiment == 'positive']
    print(f'Processed {len(analyses)}/30 items, {len(positives)} positive')
    print(f'Pool status — running: {shared_limiter.running_count}, waiting: {shared_limiter.waiting_count}')

if __name__ == '__main__':
    asyncio.run(main())
```

---

## Recipe 9: Node-level inspection with `AgentRun.iter()`

```python
"""
Use agent.iter() to record every graph node, measure per-step latency,
and inspect tool call arguments before they execute.
"""

import asyncio
import time
from pydantic_ai import Agent, RunContext
from pydantic_graph import End
from pydantic_ai.run import AgentRun

agent = Agent('openai:gpt-4o')

@agent.tool_plain
def get_temperature(city: str) -> float:
    """Return the current temperature for a city (simulated)."""
    return 22.0

async def run_with_inspection(prompt: str) -> dict:
    """Run the agent and collect execution telemetry."""
    telemetry = {
        'nodes': [],
        'step_times_ms': [],
        'run_id': None,
        'output': None,
    }

    async with agent.iter(prompt) as run:
        telemetry['run_id'] = run.run_id
        node = run.next_node

        while not isinstance(node, End):
            node_name = type(node).__name__
            t0 = time.monotonic()

            # Inspect the node before it executes
            if hasattr(node, 'request'):
                telemetry['nodes'].append({'type': node_name, 'has_request': True})
            else:
                telemetry['nodes'].append({'type': node_name})

            node = await run.next(node)
            elapsed_ms = (time.monotonic() - t0) * 1000
            telemetry['step_times_ms'].append(round(elapsed_ms, 1))

        telemetry['output'] = run.result.output
        telemetry['total_messages'] = len(run.all_messages())

    return telemetry

async def main():
    telemetry = await run_with_inspection('What is the temperature in London?')
    print('run_id:', telemetry['run_id'])
    for i, (node, ms) in enumerate(zip(telemetry['nodes'], telemetry['step_times_ms'])):
        print(f'  step {i+1}: {node["type"]} — {ms} ms')
    print('output:', telemetry['output'])
    print('messages:', telemetry['total_messages'])

if __name__ == '__main__':
    asyncio.run(main())
```

---

## Recipe 10: Prompt enrichment with `format_as_xml`

```python
"""
Use format_as_xml to inject structured context (product catalogue, user profile)
into the system prompt so the model can reason over it without JSON parsing issues.
"""

import asyncio
from dataclasses import dataclass, field
from pydantic import BaseModel
from pydantic_ai import Agent, RunContext, format_as_xml

@dataclass
class Product:
    id: str
    name: str
    price: float
    category: str
    in_stock: bool

@dataclass
class ShoppingContext:
    user_name: str
    budget: float
    products: list[Product] = field(default_factory=list)

class Recommendation(BaseModel):
    recommended_products: list[str]   # product IDs
    reasoning: str
    total_cost: float

agent = Agent(
    'openai:gpt-4o',
    deps_type=ShoppingContext,
    output_type=Recommendation,
)

@agent.system_prompt
async def inject_catalogue(ctx: RunContext[ShoppingContext]) -> str:
    catalogue_xml = format_as_xml(
        ctx.deps.products,
        root_tag='catalogue',
        item_tag='product',
    )
    return (
        f'You are a shopping assistant for {ctx.deps.user_name}. '
        f'Their budget is ${ctx.deps.budget:.2f}.\n\n'
        f'Available products:\n{catalogue_xml}\n\n'
        'Recommend products within the budget and explain your choices.'
    )

async def main():
    context = ShoppingContext(
        user_name='Alice',
        budget=50.0,
        products=[
            Product('p1', 'Python Book', 35.0, 'books', True),
            Product('p2', 'Keyboard', 85.0, 'electronics', True),
            Product('p3', 'USB Hub', 25.0, 'electronics', True),
            Product('p4', 'Mouse Pad', 12.0, 'accessories', True),
        ],
    )

    result = await agent.run(
        'What should I buy today?',
        deps=context,
    )
    rec = result.output
    print('Recommended:', rec.recommended_products)
    print('Total cost:', rec.total_cost)
    print('Reasoning:', rec.reasoning)

if __name__ == '__main__':
    asyncio.run(main())
```


