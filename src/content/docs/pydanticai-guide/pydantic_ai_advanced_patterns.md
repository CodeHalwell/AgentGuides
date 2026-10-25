---
title: "PydanticAI: Advanced Patterns"
description: "Patterns and architectural techniques for PydanticAI V2: reflection, hierarchies, dynamic tools, caching, routing, composition, rate limiting, custom models, middleware and AgentRun iteration."
framework: "pydanticai"
language: "python"
sidebar:
  label: "Advanced patterns"
  order: 19
---

Verified against **pydantic-ai %%version=pydantic-ai%%** (checked %%versions-checked%%). First written for the 1.x line; the examples have been updated for the V2 API.

**Focus:** patterns, optimisations and architectural techniques. Several patterns below are plain-Python orchestration around `Agent`; where PydanticAI has a built-in equivalent (capabilities, `Hooks`, `ConcurrencyLimiter`, toolsets) it is pointed out.

---

## Advanced Pattern 1: Self-Correcting Agents with Reflection

A second agent reviews the first one's output, and the review is fed back as a `ModelRetry` from an output validator. The writer revises its answer in the same run, with the reviewer's objections in front of it; the output retry budget caps the number of revision rounds.

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext


class Draft(BaseModel):
    answer: str
    sources: list[str] = Field(description='URLs or document titles the answer relies on')


class Critique(BaseModel):
    acceptable: bool
    problems: list[str] = Field(description='Concrete problems a reviewer would raise; empty if acceptable')


writer = Agent(
    'openai:%%model=openai%%',
    output_type=Draft,
    instructions='Answer the question accurately and list your sources.',
    retries={'output': 2},     # up to two rounds of revision
)
critic = Agent(
    'anthropic:%%model=anthropic%%',   # a different model family catches different mistakes
    output_type=Critique,
    instructions='Review the draft answer strictly. Reject unsupported claims and missing sources.',
)


@writer.output_validator
async def review(ctx: RunContext[None], draft: Draft) -> Draft:
    verdict = await critic.run(
        f'Question: {ctx.prompt}\n\nDraft answer: {draft.answer}\nSources: {draft.sources}',
        usage=ctx.usage,          # count the critic's tokens in the same run's usage
    )
    if not verdict.output.acceptable:
        raise ModelRetry('A reviewer rejected the draft:\n- ' + '\n- '.join(verdict.output.problems))
    return draft


result = writer.run_sync('When did the Python 2 line reach end of life?')
print(result.output)
print(result.usage.requests)   # writer and critic requests together
```

If the critic still rejects the draft after the last retry, the run raises `UnexpectedModelBehavior`. Each round costs a writer request and a critic request, so reserve this for outputs where an error is expensive.

---

## Advanced Pattern 2: Hierarchical Agent Structures

The usual way to build a coordinator with specialists is *agent delegation*: the coordinator gets one tool per specialist, and each tool runs the specialist agent. Passing `usage=ctx.usage` makes every specialist's requests count towards the coordinator's run, so one `UsageLimits` caps the whole tree.

```python
from pydantic_ai import Agent, RunContext, UsageLimits

researcher = Agent('openai:%%model=openai%%', instructions='Find facts. Reply with short bullet points.')
writer = Agent('anthropic:%%model=anthropic%%', instructions='Turn notes into a clear paragraph.')

coordinator = Agent(
    'openai:%%model=openai%%',
    instructions='Use research for facts and write_up for prose, then reply with the final text.',
)


@coordinator.tool
async def research(ctx: RunContext[None], topic: str) -> str:
    """Ask the research specialist for facts about a topic."""
    result = await researcher.run(topic, usage=ctx.usage)
    return result.output


@coordinator.tool
async def write_up(ctx: RunContext[None], notes: str) -> str:
    """Ask the writing specialist to turn notes into prose."""
    result = await writer.run(notes, usage=ctx.usage)
    return result.output


result = coordinator.run_sync('Explain why the sky is blue.', usage_limits=UsageLimits(request_limit=10))
print(result.output)
print(result.usage.requests)   # coordinator + specialists
```

Each specialist keeps its own instructions, model and tools, and can itself delegate further. If the specialists need the coordinator's dependencies, pass `deps=ctx.deps` as well. When the order of steps is fixed rather than chosen by the model, plain Python (`await a.run(...)` then `await b.run(...)`) or a [graph](/pydanticai-guide/pydantic_ai_graph_support/) is simpler and cheaper than a coordinator.

---

## Advanced Pattern 3: Dynamic Tool Generation

```python
"""
Generate tools dynamically based on context or capabilities.
Useful for plugin systems and extensible applications.
"""

from typing import Any, Callable

from pydantic_ai import Agent, Tool
from pydantic_ai.toolsets import FunctionToolset


class DynamicToolRegistry:
    """Registry for dynamically created tools, exposed to agents as a toolset."""

    def __init__(self) -> None:
        self.toolset: FunctionToolset = FunctionToolset()

    def register_tool_function(
        self,
        func: Callable[..., Any],
        name: str | None = None,
        description: str | None = None,
    ) -> Tool:
        """Register a plain function (no RunContext) as a tool.

        The JSON schema is generated from the signature and docstring, exactly as
        with @agent.tool_plain.
        """
        tool = Tool(func, takes_ctx=False, name=name, description=description)
        self.toolset.add_tool(tool)
        return tool

    def get_tool_schema(self, tool: Tool) -> dict:
        """The ToolDefinition the model will see."""
        td = tool.tool_def
        return {'name': td.name, 'description': td.description, 'parameters': td.parameters_json_schema}


# Usage
registry = DynamicToolRegistry()


def calculate(a: int, b: int, operation: str) -> int:
    """Perform arithmetic operations ('+' or '-')."""
    ops = {'+': lambda x, y: x + y, '-': lambda x, y: x - y}
    return ops[operation](a, b)


def search_database(query: str) -> list[dict]:
    """Search the database."""
    return [{'id': 1, 'match': query}]


calc_tool = registry.register_tool_function(calculate)
registry.register_tool_function(search_database)
print(registry.get_tool_schema(calc_tool))

# The toolset is read on every run step, so tools registered later are picked up too.
agent = Agent('openai:%%model=openai%%', toolsets=[registry.toolset])


async def use_dynamic_tools(query: str) -> str:
    result = await agent.run(query)
    return result.output
```

For tools that depend on per-run state (the user's permissions, a feature flag), prefer a dynamic toolset (`@agent.toolset` / `DynamicToolset`) or a `prepare` function, which build the tool list for each run rather than mutating a shared registry.

---

## Advanced Pattern 4: Semantic Caching for Cost Reduction

```python
"""
Use semantic similarity for intelligent caching.
Cache similar queries together to reduce API calls and costs.
"""

import math
from typing import Optional

from pydantic_ai import Agent, Embedder

class SemanticCache:
    """Cache based on semantic similarity, not exact matches."""
    
    def __init__(self, similarity_threshold: float = 0.95):
        self.cache: list[tuple[str, str, list[float]]] = []
        self.similarity_threshold = similarity_threshold
        self.embedder = Embedder('openai:text-embedding-3-small')
    
    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding for text."""
        result = await self.embedder.embed_query(text)
        return list(result.embeddings[0])
    
    def cosine_similarity(
        self,
        vec1: list[float],
        vec2: list[float]
    ) -> float:
        """Calculate cosine similarity between vectors."""
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm = math.sqrt(sum(a * a for a in vec1)) * math.sqrt(sum(b * b for b in vec2))
        return dot / norm if norm else 0.0
    
    async def get(self, query: str) -> Optional[str]:
        """Get cached response if semantically similar."""
        
        query_embedding = await self.get_embedding(query)
        
        for cached_query, cached_response, cached_embedding in self.cache:
            similarity = self.cosine_similarity(
                query_embedding,
                cached_embedding
            )
            
            if similarity >= self.similarity_threshold:
                # Found similar cached result
                return cached_response
        
        return None
    
    async def set(self, query: str, response: str):
        """Cache query-response pair."""
        
        embedding = await self.get_embedding(query)
        self.cache.append((query, response, embedding))

# Usage
semantic_cache = SemanticCache(similarity_threshold=0.90)
agent = Agent('openai:%%model=openai%%')   # build once, reuse for every request

async def cached_agent_run(query: str):
    # Check semantic cache
    cached = await semantic_cache.get(query)
    if cached:
        print(f"Semantic cache hit for: {query}")
        return cached
    
    # Run agent
    result = await agent.run(query)
    
    # Cache result
    await semantic_cache.set(query, result.output)
    
    return result.output
```

The linear scan is fine for a few thousand entries; beyond that, store the vectors in a vector database. Pick the threshold carefully: two questions can be close in embedding space and still need different answers ("refund policy for the UK" vs "for the US").

---

## Advanced Pattern 5: Conditional Tool Execution Strategy

```python
"""
Dynamically choose between early exit and exhaustive tool calling
based on query complexity.
"""

from pydantic_ai import Agent
from pydantic_ai.agent import EndStrategy
import asyncio

class AdaptiveAgentOrchestrator:
    """Adaptively choose execution strategy."""

    def __init__(self) -> None:
        # end_strategy is fixed per agent, so build one agent per strategy, once
        self.agents: dict[EndStrategy, Agent] = {
            'early': Agent('openai:%%model=openai%%', end_strategy='early', instructions='Answer quickly and concisely.'),
            'graceful': Agent('openai:%%model=openai%%', end_strategy='graceful', instructions='Use tools only if needed.'),
            'exhaustive': Agent(
                'openai:%%model=openai%%',
                end_strategy='exhaustive',
                instructions='Use all available tools to thoroughly answer.',
            ),
        }
    
    async def estimate_complexity(self, query: str) -> float:
        """Estimate query complexity (0-1)."""
        
        # Simple heuristics
        complexity_factors = {
            'multi_step': 1 if 'then' in query else 0,
            'aggregation': 1 if 'all' in query else 0,
            'comparison': 1 if 'vs' in query or 'compare' in query else 0,
            'length': min(len(query) / 100, 1),  # Longer queries more complex
        }
        
        return sum(complexity_factors.values()) / len(complexity_factors)
    
    async def run_adaptive(self, query: str) -> str:
        """Run agent with adaptive strategy."""
        
        complexity = await self.estimate_complexity(query)
        
        # EndStrategy is a Literal['early', 'graceful', 'exhaustive'] — plain strings.
        # Since 2.0 the default is 'graceful': function tools requested alongside a
        # successful output tool still run. 'early' skips them; 'exhaustive' runs every
        # tool call, output tools included.
        strategy: EndStrategy
        if complexity < 0.3:
            strategy = 'early'        # simple query: stop as soon as a final output is available
        elif complexity < 0.7:
            strategy = 'graceful'
        else:
            strategy = 'exhaustive'   # complex query: run every tool call the model makes
        
        result = await self.agents[strategy].run(query)
        
        return result.output

# Usage
orchestrator = AdaptiveAgentOrchestrator()

# Simple query - exits early
result1 = asyncio.run(
    orchestrator.run_adaptive("What is Python?")
)

# Complex query - uses all tools
result2 = asyncio.run(
    orchestrator.run_adaptive(
        "Compare Python vs JavaScript for web development, then suggest best practice tools"
    )
)
```

---

## Advanced Pattern 6: Agent Function Composition

```python
"""
Compose multiple agent operations into complex workflows.
Functional programming approach to agent orchestration.
"""

import asyncio
from typing import Awaitable, Callable, TypeVar

from pydantic_ai import Agent

T = TypeVar('T')

class AgentComposer:
    """Compose agent operations functionally."""
    
    @staticmethod
    async def pipe(
        initial_input: str,
        *operations: Callable[[str], Awaitable[str]]
    ) -> str:
        """
        Pipe input through multiple operations.
        output = op3(op2(op1(input)))
        """
        
        result = initial_input
        for operation in operations:
            result = await operation(result)
        return result
    
    @staticmethod
    async def map(
        inputs: list[str],
        operation: Callable[[str], Awaitable[str]]
    ) -> list[str]:
        """Apply operation to each input in parallel."""
        
        return await asyncio.gather(*[operation(inp) for inp in inputs])
    
    @staticmethod
    async def reduce(
        inputs: list[str],
        operation: Callable[[str, str], Awaitable[str]],
        initial: str = ""
    ) -> str:
        """Reduce multiple inputs to single output."""
        
        result = initial
        for inp in inputs:
            result = await operation(result, inp)
        return result

# Usage
researcher = Agent('openai:%%model=openai%%', name='Researcher')
summariser = Agent('openai:%%model=openai%%', name='Summariser')
translator = Agent('openai:%%model=openai%%', name='Translator')


async def complex_workflow():
    # Define operations
    async def research(query: str) -> str:
        return (await researcher.run(f"Research: {query}")).output
    
    async def summarise(content: str) -> str:
        return (await summariser.run(f"Summarise: {content}")).output
    
    async def translate(text: str) -> str:
        return (await translator.run(f"Translate to Spanish: {text}")).output
    
    # Compose: research → summarise → translate
    composer = AgentComposer()
    
    final_result = await composer.pipe(
        "Type safety in Python",
        research,
        summarise,
        translate
    )
    
    return final_result


print(asyncio.run(complex_workflow()))
```

---

## Advanced Pattern 7: Agent Rate Limiting & Queue Management

For a cap on concurrent requests, PydanticAI's own [`ConcurrencyLimiter`](/pydanticai-guide/pydantic_ai_concurrency/) (with `max_queued` for back-pressure) is enough. The queue below adds what it does not do: request-per-minute/hour budgets and priorities.

```python
"""
Manage agent requests with rate limiting and queue handling
for production-grade reliability.
"""

from asyncio import PriorityQueue, Semaphore
from datetime import datetime, timedelta
from dataclasses import dataclass
import time

@dataclass
class QueuedRequest:
    query: str
    user_id: str
    priority: int = 0
    timestamp: float | None = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def __lt__(self, other):
        # For priority queue sorting (higher priority first)
        return self.priority > other.priority

class RateLimitedAgentQueue:
    """Queue with rate limiting and priority."""
    
    def __init__(
        self,
        max_concurrent: int = 10,
        max_per_minute: int = 60,
        max_per_hour: int = 1000
    ):
        self.queue: PriorityQueue[QueuedRequest] = PriorityQueue()   # uses QueuedRequest.__lt__
        self.semaphore = Semaphore(max_concurrent)
        
        self.max_per_minute = max_per_minute
        self.max_per_hour = max_per_hour
        
        self.requests_per_minute: list[float] = []
        self.requests_per_hour: list[float] = []
    
    def check_rate_limits(self) -> bool:
        """Check if within rate limits."""
        
        now = time.time()
        
        # Remove old entries
        self.requests_per_minute = [
            t for t in self.requests_per_minute
            if now - t < 60
        ]
        
        self.requests_per_hour = [
            t for t in self.requests_per_hour
            if now - t < 3600
        ]
        
        # Check limits
        if len(self.requests_per_minute) >= self.max_per_minute:
            return False
        
        if len(self.requests_per_hour) >= self.max_per_hour:
            return False
        
        return True
    
    async def submit_request(self, request: QueuedRequest) -> bool:
        """Submit request if rate limits allow."""
        
        if not self.check_rate_limits():
            return False
        
        await self.queue.put(request)
        return True
    
    async def process_queue(self, agent_func):
        """Process queued requests."""
        
        while True:
            request = await self.queue.get()
            
            # Wait for semaphore
            async with self.semaphore:
                try:
                    result = await agent_func(request.query)
                    
                    # Record successful request
                    now = time.time()
                    self.requests_per_minute.append(now)
                    self.requests_per_hour.append(now)
                    
                    yield {
                        'user_id': request.user_id,
                        'result': result,
                        'timestamp': datetime.now()
                    }
                
                except Exception as e:
                    yield {
                        'user_id': request.user_id,
                        'error': str(e),
                        'timestamp': datetime.now()
                    }
            
            self.queue.task_done()

# Usage
queue = RateLimitedAgentQueue(
    max_concurrent=10,
    max_per_minute=60,
    max_per_hour=1000
)

async def handle_request(user_query: str, user_id: str, priority: int = 0) -> bool:
    """Try to queue user request."""
    
    request = QueuedRequest(
        query=user_query,
        user_id=user_id,
        priority=priority
    )
    
    success = await queue.submit_request(request)
    
    if not success:
        return False  # Rate limited
    
    return True
```

---

## Advanced Pattern 8: Custom Model Adapters

There are three levels, from least to most work:

1. **OpenAI-compatible endpoint** (vLLM, LiteLLM proxy, an in-house gateway): point `OpenAIChatModel` at it with a custom `OpenAIProvider`. No adapter code at all.
2. **Arbitrary HTTP API, quick and local**: wrap it in a `FunctionModel`, which turns a Python function into a `Model`.
3. **Production adapter**: subclass `pydantic_ai.models.Model` and implement `request()` (and `request_stream()` if you need streaming), as the built-in providers do.

```python
import httpx2
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

# 1. OpenAI-compatible server
compatible = OpenAIChatModel(
    'my-finetune',
    provider=OpenAIProvider(base_url='https://llm.internal.example.com/v1', api_key='your-api-key'),
)
agent_a = Agent(compatible)

# 2. Any HTTP endpoint, via FunctionModel
CUSTOM_ENDPOINT = 'https://api.custom.example.com/generate'


async def call_custom_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    prompt = next(
        part.content
        for message in reversed(messages)
        for part in message.parts
        if isinstance(part, UserPromptPart)
    )
    async with httpx2.AsyncClient() as client:
        r = await client.post(
            CUSTOM_ENDPOINT,
            json={'system': info.instructions or '', 'prompt': prompt},
            headers={'Authorization': 'Bearer your-api-key'},
        )
        r.raise_for_status()
    return ModelResponse(parts=[TextPart(r.json()['result'])], model_name='custom')


agent_b = Agent(FunctionModel(call_custom_model), instructions='You are a helpful assistant.')
# result = await agent_b.run('What is type safety?')
```

`FunctionModel` gives you message history, instructions and tool definitions (`info.function_tools`, `info.output_tools`), so it can support tool calling too — return a `ToolCallPart` instead of a `TextPart`. `httpx2` is the HTTP client PydanticAI itself depends on; plain `httpx` works equally well here.

---

## Advanced Pattern 9: Agent Middleware System

Inside an agent run, the [`Hooks`](/pydanticai-guide/pydantic_ai_hooks/) capability is the built-in middleware: `before_run`, `after_run`, `model_request`, `tool_execute` and the rest, attachable per agent or per run. The hand-written chain below sits *outside* `agent.run()`, which suits concerns that belong to your application rather than the agent (authentication of the caller, per-endpoint metrics) and wraps any async callable, agent or not.

```python
"""
Middleware pattern for cross-cutting concerns like logging,
authentication, and metrics collection.
"""

from typing import Callable, Any
from abc import ABC, abstractmethod

class AgentMiddleware(ABC):
    """Base middleware class."""
    
    @abstractmethod
    async def before_run(self, query: str, metadata: dict) -> dict:
        """Called before agent.run()"""
        pass
    
    @abstractmethod
    async def after_run(self, result: str, metadata: dict) -> str:
        """Called after agent.run()"""
        pass

class LoggingMiddleware(AgentMiddleware):
    """Log all agent executions."""
    
    async def before_run(self, query: str, metadata: dict) -> dict:
        print(f"[LOG] Running query: {query[:50]}...")
        return metadata
    
    async def after_run(self, result: str, metadata: dict) -> str:
        print(f"[LOG] Result length: {len(result)}")
        return result

class AuthenticationMiddleware(AgentMiddleware):
    """Check authentication before running."""
    
    async def before_run(self, query: str, metadata: dict) -> dict:
        if not metadata.get('user_id'):
            raise PermissionError("Not authenticated")
        return metadata
    
    async def after_run(self, result: str, metadata: dict) -> str:
        return result

class MetricsMiddleware(AgentMiddleware):
    """Collect metrics on agent execution."""
    
    def __init__(self):
        self.metrics = {'total_runs': 0, 'total_tokens': 0}
    
    async def before_run(self, query: str, metadata: dict) -> dict:
        self.metrics['total_runs'] += 1
        return metadata
    
    async def after_run(self, result: str, metadata: dict) -> str:
        tokens = metadata.get('tokens_used', 0)
        self.metrics['total_tokens'] += tokens
        return result

class MiddlewareChain:
    """Chain multiple middlewares."""
    
    def __init__(self, middlewares: list[AgentMiddleware]):
        self.middlewares = middlewares
    
    async def execute(
        self,
        query: str,
        agent_func: Callable,
        metadata: dict
    ) -> str:
        """Execute query through middleware chain."""
        
        # Before hooks
        for middleware in self.middlewares:
            metadata = await middleware.before_run(query, metadata)
        
        # Execute agent
        result = await agent_func(query)
        
        # After hooks
        for middleware in reversed(self.middlewares):
            result = await middleware.after_run(result, metadata)
        
        return result

# Usage
middlewares = [
    LoggingMiddleware(),
    AuthenticationMiddleware(),
    MetricsMiddleware()
]

chain = MiddlewareChain(middlewares)

from pydantic_ai import Agent

agent = Agent('openai:%%model=openai%%')


async def run_with_middleware(query: str, user_id: str):
    async def agent_func(q):
        result = await agent.run(q)
        return result.output
    
    result = await chain.execute(
        query=query,
        agent_func=agent_func,
        metadata={'user_id': user_id}
    )
    
    return result
```

---

## Advanced Pattern: `AgentRun` — granular iteration with `agent.iter()`

`agent.iter()` returns an `AgentRun` async context manager. Iterate node-by-node with `async for` or drive manually with `.next()` for fine-grained control, early termination, or node-level observability.

### Basic node-by-node iteration

```python
import asyncio
from pydantic_ai import Agent, CallToolsNode, ModelRequestNode, UserPromptNode
from pydantic_graph import End

agent = Agent('openai:%%model=openai%%')

async def main():
    nodes_visited = []

    async with agent.iter('What is the capital of France?') as run:
        async for node in run:
            nodes_visited.append(type(node).__name__)

    print(nodes_visited)
    # ['UserPromptNode', 'ModelRequestNode', 'CallToolsNode', 'End']
    print(run.result.output)
    # 'The capital of France is Paris.'

asyncio.run(main())
```

### Manual `.next()` driving for early exit

```python
import asyncio
from pydantic_ai import Agent, CallToolsNode
from pydantic_graph import End

agent = Agent('openai:%%model=openai%%')

async def run_with_budget(prompt: str, max_steps: int = 10):
    async with agent.iter(prompt) as run:
        node = run.next_node   # the first node (a UserPromptNode), not yet run
        for _ in range(max_steps):
            if isinstance(node, End):
                break
            if isinstance(node, CallToolsNode):
                # Peek at what tools the model called before executing
                for part in node.model_response.parts:
                    if hasattr(part, 'tool_name'):
                        print(f'  → calling tool: {part.tool_name}')
            node = await run.next(node)
        else:
            print('WARNING: max steps reached — run may be incomplete')
    return run.result

asyncio.run(run_with_budget('Search for recent AI papers and summarise three of them.'))
```

### Streaming node events to a client

Forward agent progress to a WebSocket in real-time:

```python
import asyncio
import json
from pydantic_ai import Agent, CallToolsNode, ModelRequestNode
from pydantic_graph import End

agent = Agent('openai:%%model=openai%%')

async def stream_to_websocket(ws, prompt: str):
    async with agent.iter(prompt) as run:
        async for node in run:
            if isinstance(node, ModelRequestNode):
                await ws.send(json.dumps({'event': 'thinking'}))
            elif isinstance(node, CallToolsNode):
                tool_names = [
                    p.tool_name for p in node.model_response.parts
                    if hasattr(p, 'tool_name')
                ]
                await ws.send(json.dumps({'event': 'tool_calls', 'tools': tool_names}))
            elif isinstance(node, End):
                await ws.send(json.dumps({'event': 'done', 'output': run.result.output}))
```

### Capturing per-node timings

```python
import asyncio
import time
from pydantic_ai import Agent
from pydantic_graph import End

agent = Agent('openai:%%model=openai%%')

async def timed_run(prompt: str):
    timings = []
    async with agent.iter(prompt) as run:
        node = run.next_node
        while not isinstance(node, End):
            t0 = time.perf_counter()
            step_name = type(node).__name__
            node = await run.next(node)   # actually executes the node
            timings.append((step_name, time.perf_counter() - t0))

    for name, elapsed in timings:
        print(f'{name}: {elapsed:.3f}s')
    return run.result

asyncio.run(timed_run('What is the speed of light?'))
```

### Resuming from a specific node (graph-level control)

```python
import asyncio
from pydantic_ai import Agent, ModelRequestNode
from pydantic_graph import End

agent = Agent('openai:%%model=openai%%')

async def run_with_model_override(prompt: str):
    """Override model parameters on a specific request node before execution."""
    async with agent.iter(prompt) as run:
        node = run.next_node
        while not isinstance(node, End):
            if isinstance(node, ModelRequestNode):
                # Inspect before calling the model — could log, trace, or modify
                print(f'About to make model request with {len(node.request.parts)} parts')
            node = await run.next(node)
    return run.result

asyncio.run(run_with_model_override('Generate a haiku about Python.'))
```

### `AgentRun` properties

```python
import asyncio

from pydantic_ai import Agent

agent = Agent('openai:%%model=openai%%')


async def main():
    async with agent.iter('prompt') as run:
        async for node in run:
            pass

    print(run.result.output)        # AgentRunResult — available after the loop completes
    print(run.usage)                # RunUsage — totals across all steps
    print(len(run.all_messages()))  # list[ModelMessage] — the full conversation
    print(run.run_id, run.conversation_id)


asyncio.run(main())
```

Inside the loop, `run.next_node` is the node about to run, `run.ctx` exposes the graph state and deps, and `run.cancel()` stops the run with `RunCancelled`.
