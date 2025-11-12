# Pydantic AI: Advanced Patterns & Deep Dives

**Version:** 1.0.0  
**Focus:** Expert-level patterns, optimisations, and architectural deep dives

---

## Advanced Pattern 1: Self-Correcting Agents with Reflection

```python
"""
Agent that reflects on its output and self-corrects before returning.
Useful for ensuring quality without external validation.
"""

from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic import BaseModel

class ReflectionResponse(BaseModel):
    initial_response: str
    reflection: str
    corrected_response: str
    confidence: float

reflection_agent = Agent(
    'openai:gpt-4o',
    output_type=ReflectionResponse,
    name='ReflectiveAgent'
)

@reflection_agent.tool
async def reflect_and_correct(
    ctx: RunContext,
    initial_response: str
) -> dict:
    """
    Self-correction loop:
    1. Generate response
    2. Reflect on quality
    3. Correct if needed
    4. Return final output
    """
    
    # Use same agent for reflection (meta-reasoning)
    reflection_prompt = f"""
    Analyse this response for accuracy and quality:
    "{initial_response}"
    
    Provide:
    1. Assessment of correctness (score 0-1)
    2. Any issues or improvements needed
    3. A corrected version if needed
    """
    
    return {
        'assessment': 'Good response with minor improvements possible',
        'corrected': initial_response + " (with additional context)"
    }

# Advanced: Multi-layer reflection
async def multi_layer_reflection(query: str):
    """Agent that reflects at multiple levels."""
    
    # Layer 1: Generate response
    agent_1 = Agent('openai:gpt-4o', name='Generator')
    result_1 = await agent_1.run(f"Generate: {query}")
    
    # Layer 2: Critique response
    agent_2 = Agent('openai:gpt-4o', name='Critic')
    result_2 = await agent_2.run(
        f"Critique this: {result_1.output}"
    )
    
    # Layer 3: Synthesise improved response
    agent_3 = Agent('openai:gpt-4o', name='Synthesizer')
    result_3 = await agent_3.run(
        f"Original: {result_1.output}\n"
        f"Critique: {result_2.output}\n"
        f"Synthesise an improved version"
    )
    
    return result_3.output
```

---

## Advanced Pattern 2: Hierarchical Agent Structures

```python
"""
Hierarchical multi-agent system with delegation and specialisation.
Coordinator → Specialists → Tools
"""

from pydantic_ai import Agent, RunContext
from dataclasses import dataclass
from enum import Enum

class TaskType(str, Enum):
    ANALYSIS = "analysis"
    WRITING = "writing"
    CODING = "coding"
    RESEARCH = "research"

@dataclass
class HierarchicalContext:
    """Shared context through hierarchy."""
    task_type: TaskType
    depth_level: int  # For tracking recursion
    max_depth: int
    results_cache: dict

class HierarchicalCoordinator:
    """Manages agent hierarchy."""
    
    def __init__(self):
        self.coordinator = Agent(
            'openai:gpt-4o',
            name='Coordinator',
            instructions='Delegate tasks to specialists'
        )
        
        self.specialists = {
            TaskType.ANALYSIS: Agent('openai:gpt-4o', name='Analyst'),
            TaskType.WRITING: Agent('anthropic:claude-3-5-sonnet', name='Writer'),
            TaskType.CODING: Agent('openai:gpt-4o', name='Coder'),
            TaskType.RESEARCH: Agent('openai:gpt-4o', name='Researcher'),
        }
    
    async def process(
        self,
        query: str,
        task_type: TaskType,
        context: HierarchicalContext
    ) -> str:
        """Process query through hierarchy."""
        
        # Check recursion depth
        if context.depth_level >= context.max_depth:
            return "Max depth reached"
        
        # Check cache
        cache_key = f"{task_type}_{query}"
        if cache_key in context.results_cache:
            return context.results_cache[cache_key]
        
        # Get specialist
        specialist = self.specialists.get(
            task_type,
            self.specialists[TaskType.ANALYSIS]
        )
        
        # Process with specialist
        result = await specialist.run(query)
        
        # Cache result
        context.results_cache[cache_key] = result.output
        
        return result.output

# Usage
async def hierarchical_workflow(query: str):
    coordinator = HierarchicalCoordinator()
    context = HierarchicalContext(
        task_type=TaskType.ANALYSIS,
        depth_level=0,
        max_depth=3,
        results_cache={}
    )
    
    return await coordinator.process(query, TaskType.ANALYSIS, context)
```

---

## Advanced Pattern 3: Dynamic Tool Generation

```python
"""
Generate tools dynamically based on context or capabilities.
Useful for plugin systems and extensible applications.
"""

from pydantic_ai import Agent, Tool
from typing import Callable, Any
import inspect

class DynamicToolRegistry:
    """Registry for dynamically created tools."""
    
    def __init__(self):
        self.tools: dict[str, Tool] = {}
        self.agent = Agent('openai:gpt-4o')
    
    def register_tool_function(
        self,
        func: Callable,
        name: str | None = None,
        description: str | None = None
    ) -> Tool:
        """
        Register a function as a tool dynamically.
        
        Args:
            func: Function to register
            name: Tool name (defaults to function name)
            description: Tool description (uses docstring if not provided)
        """
        
        tool_name = name or func.__name__
        tool_description = description or func.__doc__ or "No description"
        
        # Create Tool instance
        tool = Tool(
            func=func,
            name=tool_name,
            description=tool_description
        )
        
        self.tools[tool_name] = tool
        
        # Add to agent (requires toolset approach)
        return tool
    
    def get_tool_schema(self, tool_name: str) -> dict:
        """Get JSON schema for a tool."""
        
        tool = self.tools.get(tool_name)
        if not tool:
            return {}
        
        # Extract function signature
        sig = inspect.signature(tool.func)
        
        parameters = {}
        for param_name, param in sig.parameters.items():
            if param_name in ('self', 'ctx'):
                continue
            
            annotation = param.annotation
            
            parameters[param_name] = {
                'type': 'string' if annotation == str else 'number',
                'description': f"Parameter: {param_name}"
            }
        
        return {
            'name': tool_name,
            'description': tool.description,
            'parameters': parameters
        }
    
    def get_available_tools_description(self) -> str:
        """Get description of all available tools."""
        
        descriptions = []
        for name, tool in self.tools.items():
            descriptions.append(f"- {name}: {tool.description}")
        
        return "\n".join(descriptions)

# Usage
registry = DynamicToolRegistry()

# Register tools dynamically
def calculate(a: int, b: int, operation: str) -> int:
    """Perform arithmetic operations."""
    ops = {'+': lambda x, y: x + y, '-': lambda x, y: x - y}
    return ops[operation](a, b)

def search_database(query: str) -> list[dict]:
    """Search the database."""
    return [{'id': 1, 'match': query}]

registry.register_tool_function(calculate)
registry.register_tool_function(search_database)

# Agent can now use dynamically registered tools
async def use_dynamic_tools(query: str):
    tools_desc = registry.get_available_tools_description()
    
    agent = Agent(
        'openai:gpt-4o',
        system_prompt=f"Available tools:\n{tools_desc}"
    )
    
    result = await agent.run(query)
    return result.output
```

---

## Advanced Pattern 4: Semantic Caching for Cost Reduction

```python
"""
Use semantic similarity for intelligent caching.
Cache similar queries together to reduce API calls and costs.
"""

from typing import Optional
import numpy as np
from openai import AsyncOpenAI

class SemanticCache:
    """Cache based on semantic similarity, not exact matches."""
    
    def __init__(self, similarity_threshold: float = 0.95):
        self.cache: list[tuple[str, str, list[float]]] = []
        self.similarity_threshold = similarity_threshold
        self.embeddings_client = AsyncOpenAI()
    
    async def get_embedding(self, text: str) -> list[float]:
        """Get embedding for text."""
        
        response = await self.embeddings_client.embeddings.create(
            model='text-embedding-3-small',
            input=text
        )
        
        return response.data[0].embedding
    
    def cosine_similarity(
        self,
        vec1: list[float],
        vec2: list[float]
    ) -> float:
        """Calculate cosine similarity between vectors."""
        
        arr1 = np.array(vec1)
        arr2 = np.array(vec2)
        
        return float(np.dot(arr1, arr2) / (np.linalg.norm(arr1) * np.linalg.norm(arr2)))
    
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

async def cached_agent_run(query: str):
    from pydantic_ai import Agent
    
    # Check semantic cache
    cached = await semantic_cache.get(query)
    if cached:
        print(f"Semantic cache hit for: {query}")
        return cached
    
    # Run agent
    agent = Agent('openai:gpt-4o')
    result = await agent.run(query)
    
    # Cache result
    await semantic_cache.set(query, result.output)
    
    return result.output
```

---

## Advanced Pattern 5: Conditional Tool Execution Strategy

```python
"""
Dynamically choose between early exit and exhaustive tool calling
based on query complexity.
"""

from pydantic_ai import Agent, EndStrategy
import asyncio

class AdaptiveAgentOrchestrator:
    """Adaptively choose execution strategy."""
    
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
        
        # Choose strategy based on complexity
        if complexity < 0.3:
            # Simple query - exit early when possible
            strategy = EndStrategy.EARLY
            instructions = "Answer quickly and concisely."
        elif complexity < 0.7:
            # Medium complexity - use some tools
            strategy = EndStrategy.EARLY
            instructions = "Use tools only if needed."
        else:
            # Complex query - use all available tools
            strategy = EndStrategy.EXHAUSTIVE
            instructions = "Use all available tools to thoroughly answer."
        
        agent = Agent(
            'openai:gpt-4o',
            end_strategy=strategy,
            instructions=instructions
        )
        
        result = await agent.run(query)
        
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

from typing import Awaitable, Callable, TypeVar

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
async def complex_workflow():
    from pydantic_ai import Agent
    
    # Define operations
    async def research(query: str) -> str:
        agent = Agent('openai:gpt-4o', name='Researcher')
        result = await agent.run(f"Research: {query}")
        return result.output
    
    async def summarise(content: str) -> str:
        agent = Agent('openai:gpt-4o', name='Summariser')
        result = await agent.run(f"Summarise: {content}")
        return result.output
    
    async def translate(text: str) -> str:
        agent = Agent('openai:gpt-4o', name='Translator')
        result = await agent.run(f"Translate to Spanish: {text}")
        return result.output
    
    # Compose: research → summarise → translate
    composer = AgentComposer()
    
    final_result = await composer.pipe(
        "Type safety in Python",
        research,
        summarise,
        translate
    )
    
    return final_result
```

---

## Advanced Pattern 7: Agent Rate Limiting & Queue Management

```python
"""
Manage agent requests with rate limiting and queue handling
for production-grade reliability.
"""

from asyncio import Queue, Semaphore
from datetime import datetime, timedelta
from dataclasses import dataclass
import time

@dataclass
class QueuedRequest:
    query: str
    user_id: str
    priority: int = 0
    timestamp: float = None
    
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
        self.queue: Queue[QueuedRequest] = Queue()
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

```python
"""
Create custom adapters for unsupported models or specialised use cases.
"""

from pydantic_ai import Agent
from typing import Optional

class CustomModelAdapter:
    """Adapter for custom or proprietary models."""
    
    def __init__(self, model_endpoint: str, api_key: str):
        self.model_endpoint = model_endpoint
        self.api_key = api_key
    
    async def call_model(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7
    ) -> str:
        """Call custom model endpoint."""
        
        import httpx
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.model_endpoint,
                json={
                    'system': system_prompt,
                    'messages': [{'role': 'user', 'content': user_message}],
                    'temperature': temperature,
                },
                headers={'Authorization': f'Bearer {self.api_key}'}
            )
            
            return response.json()['result']

# Create agent with custom model
class CustomAgent:
    def __init__(self, adapter: CustomModelAdapter):
        self.adapter = adapter
    
    async def run(self, query: str, system_prompt: str = "") -> str:
        """Run query against custom model."""
        
        return await self.adapter.call_model(
            system_prompt=system_prompt or "You are a helpful assistant.",
            user_message=query
        )

# Usage
adapter = CustomModelAdapter(
    model_endpoint='https://api.custom.com/models/gpt',
    api_key='your-api-key'
)

custom_agent = CustomAgent(adapter)

result = asyncio.run(
    custom_agent.run('What is type safety?')
)
```

---

## Advanced Pattern 9: Agent Middleware System

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

async def run_with_middleware(query: str, user_id: str):
    from pydantic_ai import Agent
    
    agent = Agent('openai:gpt-4o')
    
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

(Advanced patterns continue with more sophisticated techniques...)

These patterns demonstrate expert-level usage of Pydantic AI for complex, production-grade applications.

