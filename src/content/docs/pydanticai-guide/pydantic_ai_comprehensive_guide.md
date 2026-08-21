---
title: "Pydantic AI: Comprehensive Technical Guide"
description: "Version: 2.33.0 (August 2026) Framework: Pydantic AI - GenAI Agent Framework, the Pydantic Way Author Notes: Exhaustive technical documentation with production patterns, type safety"
framework: pydanticai
---

Latest: 2.33.0 | Updated: August 21, 2026
# Pydantic AI: Comprehensive Technical Guide
## From Beginner to Expert Level

**Version:** 2.33.0 (August 2026)  
**Framework:** Pydantic AI - GenAI Agent Framework, the Pydantic Way  
**Author Notes:** Exhaustive technical documentation with production patterns, type safety emphasis, and FastAPI-inspired developer experience.

---

## Table of Contents

1. [Philosophy & Core Concepts](#philosophy--core-concepts)
2. [Installation & Setup](#installation--setup)
3. [Core Fundamentals](#core-fundamentals)
4. [Simple Agents](#simple-agents)
5. [Type Safety & Validation](#type-safety--validation)
6. [Structured Output](#structured-output)
7. [Tools & Function Calling](#tools--function-calling)
8. [Dependency Injection](#dependency-injection)
9. [Advanced Patterns](#advanced-patterns)
10. [Production Deployment](#production-deployment)

---

## Philosophy & Core Concepts

### "FastAPI Feeling" for GenAI

Pydantic AI brings the ergonomic design of FastAPI to Generative AI development. This means:

- **Type Safety First**: Leveraging Python's type system and Pydantic v2 for automatic validation
- **Developer Experience**: Familiar decorators, dependency injection, and structured patterns
- **Pythonic Conventions**: Modern Python 3.10+ features like type hints and async/await
- **Reusability**: Agents are instantiated once and reused throughout the application
- **Testability**: Built-in testing utilities and model mocking capabilities

### Core Philosophy Pillars

```python
"""
Pydantic AI Philosophy:
1. Type Safety by Default - All inputs/outputs validated with Pydantic
2. Model Agnosticism - Single interface for OpenAI, Anthropic, Gemini, Groq, etc.
3. Structured Outputs - Guarantee response validation and schema compliance
4. Observable Systems - Built-in Logfire integration for production observability
5. Composable Tools - Function calling as first-class citizens
6. Async-First Design - Native async/await throughout
7. Test-Friendly - TestModel for unit testing without API calls
"""
```

### Why Pydantic AI?

| Challenge | Solution |
|-----------|----------|
| Unpredictable LLM outputs | Type-safe structured outputs with Pydantic validation |
| Model lock-in | Unified interface for all major LLM providers |
| Complex tool orchestration | Decorator-based tool definition with automatic schema generation |
| State management | Dependency injection system with RunContext |
| Production observability | Logfire integration for traces and monitoring |
| Testing complexity | TestModel and FunctionModel for easy unit testing |
| Tool dependencies | Context-aware tool parameters with automatic injection |

---

## Installation & Setup

### Option 1: Complete Installation with All Extras

```bash
# Using pip
pip install pydantic-ai[all]

# Using uv (faster)
uv add pydantic-ai[all]
```

### Option 2: Minimal Installation (pydantic-ai-slim)

The slim version is significantly smaller and downloads only necessary dependencies:

```bash
# Core slim with OpenAI support
pip install "pydantic-ai-slim[openai]"
uv add "pydantic-ai-slim[openai]"
```

### Option 3: Selective Installation by Provider

```bash
# OpenAI only
pip install "pydantic-ai-slim[openai]"

# Anthropic Claude
pip install "pydantic-ai-slim[anthropic]"

# Google Gemini
pip install "pydantic-ai-slim[google]"

# Groq (fast inference)
pip install "pydantic-ai-slim[groq]"

# Multiple providers
pip install "pydantic-ai-slim[openai,anthropic,google,groq]"

# With observability
pip install "pydantic-ai-slim[openai,logfire]"

# For MCP integration
pip install "pydantic-ai-slim[mcp]"

# For durable execution
pip install "pydantic-ai[prefect]"  # Prefect integration
pip install "pydantic-ai[dbos]"     # DBOS integration
```

### Environment Setup

```python
# .env file
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
GROQ_API_KEY=...

# Optional: Observability
LOGFIRE_TOKEN=...
```

```python
# main.py - Load environment variables
import os
from dotenv import load_dotenv

load_dotenv()

# Verify setup
assert os.getenv('OPENAI_API_KEY'), "OPENAI_API_KEY not set"
```

### Verification: Hello World

```python
from pydantic_ai import Agent

# Create minimal agent
agent = Agent('openai:gpt-4o')

# Test synchronously
result = agent.run_sync('What is 2 + 2?')
print(result.output)
#> 2 + 2 equals 4.

# Check token usage (usage is a property in 2.33.0, not a method)
print(result.usage)
#> RunUsage(input_tokens=14, output_tokens=5, requests=1)
```

---

## Core Fundamentals

### Core Classes Overview

#### 1. Agent

The primary class for creating AI agents. Instances are typically created once and reused.

```python
from pydantic_ai import Agent
from typing import Optional

# Minimal agent
agent = Agent('openai:gpt-4o')

# Agent with instructions
agent_with_instructions = Agent(
    'openai:gpt-4o',
    instructions='Be concise and professional. Reply with 1-2 sentences.'
)

# Agent with dependencies
from dataclasses import dataclass

@dataclass
class UserContext:
    user_id: int
    username: str

agent_with_deps = Agent(
    'openai:gpt-4o',
    deps_type=UserContext,
    instructions='Personalise all responses using the user context.'
)

# Complete agent configuration
agent_complete = Agent(
    model='openai:gpt-4o',
    system_prompt='You are a helpful assistant specializing in Python.',
    instructions='Provide clear, working code examples.',
    deps_type=UserContext,
    output_type=Optional[str],
    retries=2,  # Retry failed calls up to 2 times
    name='PythonHelper'
)
```

#### 2. RunContext

Provides access to dependencies, model information, and message history during execution.

```python
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass

@dataclass
class AppDependencies:
    database_url: str
    api_key: str

agent = Agent(
    'openai:gpt-4o',
    deps_type=AppDependencies,
)

@agent.tool
async def fetch_user_data(ctx: RunContext[AppDependencies], user_id: int) -> str:
    """
    Tool with access to context.
    
    Args:
        ctx: RunContext containing dependencies and metadata
        user_id: The user identifier
    """
    # Access dependencies
    db_url = ctx.deps.database_url
    
    # Access model information
    model_name = ctx.model.model_name
    
    # Access message history
    messages = ctx.messages
    
    # Access full message history with all messages
    all_messages = ctx.all_messages()
    
    return f"User {user_id} data from {db_url}"
```

#### 3. ModelRetry

Instructs the model to retry with corrected outputs. Used in validation workflows.

```python
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic import BaseModel, Field
import re

class EmailAddress(BaseModel):
    email: str = Field(..., description="Valid email address")
    name: str = Field(..., description="User name")

agent = Agent(
    'openai:gpt-4o',
    output_type=EmailAddress
)

@agent.output_validator
async def validate_email(ctx: RunContext, output: EmailAddress) -> EmailAddress:
    """Validate email format and retry if invalid."""
    
    # Simple email regex validation
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    if not re.match(email_pattern, output.email):
        raise ModelRetry(
            f'Invalid email format: {output.email}. Please provide a valid email address.'
        )
    
    if len(output.name) < 2:
        raise ModelRetry(
            f'Name too short: {output.name}. Please provide a full name.'
        )
    
    return output

# Usage
result = agent.run_sync('Extract email from: Contact John Doe at john@example.com')
print(result.output)
#> EmailAddress(email='john@example.com', name='John Doe')
```

#### 4. Tool Definition

Functions decorated with `@agent.tool` become callable by the LLM.

```python
from pydantic_ai import Agent, RunContext
from typing import Any
import asyncio

agent = Agent('openai:gpt-4o')

# Tool with context
@agent.tool
async def search_database(
    ctx: RunContext,
    query: str,
    limit: int = 10
) -> str:
    """
    Search the database for documents.
    
    Args:
        ctx: Execution context
        query: Search query
        limit: Maximum results (default: 10)
    
    Returns:
        Search results as formatted string
    """
    # Simulate database search
    await asyncio.sleep(0.1)
    return f"Found {limit} results for '{query}'"

# Tool without context (plain tool)
@agent.tool_plain
def get_current_time() -> str:
    """Get current server time in ISO format."""
    from datetime import datetime
    return datetime.now().isoformat()

# Tool with strict schema (for OpenAI compatibility)
@agent.tool(strict=True)
async def calculate(ctx: RunContext, a: int, b: int, operation: str) -> int:
    """
    Perform mathematical operations.
    
    Args:
        ctx: Execution context
        a: First number
        b: Second number
        operation: 'add', 'subtract', 'multiply', 'divide'
    """
    operations = {
        'add': lambda x, y: x + y,
        'subtract': lambda x, y: x - y,
        'multiply': lambda x, y: x * y,
        'divide': lambda x, y: x // y,
    }
    return operations[operation](a, b)

# Usage
result = agent.run_sync('What time is it?')
print(result.output)
#> The current time is 2025-03-18T14:30:45.123456
```

### Model-Agnostic Design

Pydantic AI supports numerous LLM providers with a unified interface:

```python
from pydantic_ai import Agent

# OpenAI
openai_agent = Agent('openai:gpt-4o')
openai_o3 = Agent('openai:o3-mini')

# Anthropic Claude
claude_agent = Agent('anthropic:claude-3-5-sonnet-latest')
claude_opus = Agent('anthropic:claude-3-opus-20250219')

# Google Gemini
gemini_agent = Agent('google-gla:gemini-1.5-flash')
gemini_pro = Agent('google-gla:gemini-1.5-pro')

# Groq (fast inference)
groq_agent = Agent('groq:llama-3.3-70b-versatile')

# DeepSeek
deepseek_agent = Agent('deepseek:deepseek-chat')

# Mistral
mistral_agent = Agent('mistral:mistral-large-latest')

# Grok
grok_agent = Agent('grok:grok-2-latest')

# Amazon Bedrock
bedrock_agent = Agent('bedrock:anthropic.claude-3-sonnet-20240229-v1:0')

# Perplexity (OpenAI-compatible)
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIProvider

perplexity = OpenAIChatModel(
    'sonar-pro',
    provider=OpenAIProvider(
        base_url='https://api.perplexity.ai',
        api_key='your-api-key'
    )
)
perplexity_agent = Agent(perplexity)

# Fallback strategy - try primary, then backup
from pydantic_ai.models.fallback import FallbackModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.anthropic import AnthropicModel

fallback_model = FallbackModel(
    OpenAIChatModel('gpt-4o'),  # Primary
    AnthropicModel('claude-3-5-sonnet-latest')  # Fallback
)
fallback_agent = Agent(fallback_model)
```

### Configuration Patterns

```python
from pydantic_ai import Agent, ModelSettings
from pydantic import BaseModel

# Configuration via constructor
configured_agent = Agent(
    'openai:gpt-4o',
    instructions='Be concise.',
    retries=3,  # Retry failed calls
    name='MyAgent'
)

# Configuration via settings
settings = ModelSettings(
    temperature=0.7,
    max_tokens=500,
    top_p=0.95,
    frequency_penalty=0.0,
    presence_penalty=0.0,
)

# Using settings in run
result = configured_agent.run_sync(
    'What is Python?',
    model_settings=settings
)

# Custom output type configuration
class Article(BaseModel):
    title: str
    content: str
    keywords: list[str]

article_agent = Agent(
    'openai:gpt-4o',
    output_type=Article,
    instructions='Write comprehensive technical articles.'
)

# ── UsageLimits (verified against pydantic-ai 2.10.0 usage.py) ──────────────────

# All fields — every limit defaults to None (disabled) except request_limit=50
from pydantic_ai import UsageLimits

# Typical production guard: cap cost and prevent runaway tool loops
limits = UsageLimits(
    request_limit=5,           # max LLM API calls per run (default: 50)
    tool_calls_limit=10,       # max successful tool executions per run
    input_tokens_limit=8000,   # raise UsageLimitExceeded if input tokens exceed this
    output_tokens_limit=1500,  # raise after each response if output tokens exceed this
    total_tokens_limit=10000,  # combined input + output ceiling
    # count_tokens_before_request=True,  # pre-flight count (Anthropic/Google/OpenAI Responses)
)

result = article_agent.run_sync(
    'Write about type safety in Python',
    usage_limits=limits,
)

# RunUsage — inspect what was consumed (tool_calls field added in 2.8.0)
from pydantic_ai import Agent
from pydantic_ai.usage import RunUsage

agent = Agent('openai:gpt-4o')

@agent.tool_plain
def get_fact() -> str:
    return 'Python was created in 1991.'

result2 = agent.run_sync('Tell me a fact.', usage_limits=UsageLimits(request_limit=3))
usage: RunUsage = result2.usage   # property in 2.33.0
print(f"requests={usage.requests}")           # total LLM API calls made
print(f"tool_calls={usage.tool_calls}")       # total successful tool executions
print(f"input_tokens={usage.input_tokens}")   # cumulative prompt tokens
print(f"output_tokens={usage.output_tokens}") # cumulative completion tokens
print(f"total_tokens={usage.total_tokens}")   # property: input + output
print(f"cache_read_tokens={usage.cache_read_tokens}")   # tokens served from provider cache
print(f"cache_write_tokens={usage.cache_write_tokens}") # tokens written to provider cache

# Combine usage across multiple independent runs
combined: RunUsage = result.usage + result2.usage
print(f"combined requests: {combined.requests}")

# Handle UsageLimitExceeded gracefully
from pydantic_ai.exceptions import UsageLimitExceeded

try:
    result3 = agent.run_sync(
        'Count every prime below 10000 then summarise.',
        usage_limits=UsageLimits(request_limit=1, output_tokens_limit=50),
    )
except UsageLimitExceeded as exc:
    print(f"Run aborted: {exc}")   # e.g. "Exceeded the output_tokens_limit of 50 ..."
```

---

## Simple Agents

### Creating Your First Agent

```python
from pydantic_ai import Agent
import asyncio

# 1. Create agent with model
agent = Agent(
    'openai:gpt-4o',
    instructions='Respond with exactly one sentence.'
)

# 2. Run synchronously (for simple scripts)
result = agent.run_sync('What is the capital of France?')
print(f"Answer: {result.output}")
#> Answer: The capital of France is Paris.

# 3. Access usage information (property in 2.33.0)
usage = result.usage
print(f"Tokens: {usage.input_tokens} input, {usage.output_tokens} output")
#> Tokens: 18 input, 8 output

# 4. Run asynchronously (for production)
async def async_example():
    result = await agent.run('Explain type safety in Python.')
    return result.output

# Execute
output = asyncio.run(async_example())
print(output)
#> Type safety refers to the language's ability to prevent type errors...
```

### Function Definitions with Full Typing

```python
from pydantic_ai import Agent, RunContext
from typing import Optional
from datetime import datetime
import asyncio

agent = Agent('openai:gpt-4o')

# Tool with complete type annotations
@agent.tool
async def get_weather(
    ctx: RunContext,
    location: str,
    unit: str = 'celsius'
) -> dict[str, Any]:
    """
    Get weather information for a location.
    
    This tool demonstrates:
    - Type-annotated parameters
    - Optional parameters with defaults
    - Complex return types
    - Docstring format for schema generation
    
    Args:
        ctx: Execution context
        location: City name or coordinates
        unit: Temperature unit ('celsius' or 'fahrenheit')
    
    Returns:
        Dictionary with temperature, condition, and forecast
    """
    from typing import Any
    
    # Simulate API call
    await asyncio.sleep(0.2)
    
    return {
        'location': location,
        'temperature': 22,
        'unit': unit,
        'condition': 'Partly cloudy',
        'humidity': 65,
        'wind_speed': 12
    }

@agent.tool
async def search_documents(
    ctx: RunContext,
    query: str,
    semantic: bool = True,
    max_results: int = 5
) -> list[dict[str, str]]:
    """
    Search through document database.
    
    Args:
        ctx: Execution context
        query: Search query string
        semantic: Whether to use semantic search
        max_results: Maximum results to return
    
    Returns:
        List of matching documents with id, title, and relevance
    """
    return [
        {'id': '1', 'title': 'Python Types', 'relevance': 0.95},
        {'id': '2', 'title': 'Type Hints', 'relevance': 0.87},
    ]

# Tool with enums for type safety
from enum import Enum

class TemperatureUnit(str, Enum):
    CELSIUS = 'celsius'
    FAHRENHEIT = 'fahrenheit'
    KELVIN = 'kelvin'

@agent.tool
async def convert_temperature(
    ctx: RunContext,
    value: float,
    from_unit: TemperatureUnit,
    to_unit: TemperatureUnit
) -> float:
    """
    Convert temperature between units.
    
    Args:
        ctx: Execution context
        value: Temperature value
        from_unit: Source unit
        to_unit: Target unit
    
    Returns:
        Converted temperature value
    """
    conversions = {
        (TemperatureUnit.CELSIUS, TemperatureUnit.FAHRENHEIT): lambda v: v * 9/5 + 32,
        (TemperatureUnit.FAHRENHEIT, TemperatureUnit.CELSIUS): lambda v: (v - 32) * 5/9,
        (TemperatureUnit.CELSIUS, TemperatureUnit.KELVIN): lambda v: v + 273.15,
    }
    return conversions.get((from_unit, to_unit), lambda v: v)(value)

# Usage
result = agent.run_sync('What is the weather in London?')
print(result.output)
```

### System Prompts and Configuration

```python
from pydantic_ai import Agent, RunContext
from datetime import datetime

# Static system prompt
agent_static = Agent(
    'openai:gpt-4o',
    system_prompt=(
        'You are a professional technical writer. '
        'Write clear, concise, and well-structured documentation. '
        'Always include code examples when relevant.'
    )
)

# Dynamic system prompt (evaluates on each run)
agent_dynamic = Agent('openai:gpt-4o')

@agent_dynamic.system_prompt
async def dynamic_prompt(ctx: RunContext) -> str:
    """System prompt that includes current context."""
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return f"""
    Current server time: {current_time}
    You are a helpful assistant.
    Always refer to the current time when relevant.
    Timezone: UTC
    """

# Combined static + dynamic prompts
agent_combined = Agent(
    'openai:gpt-4o',
    system_prompt='You are a Python expert.'
)

@agent_combined.system_prompt
async def add_context(ctx: RunContext) -> str:
    """Additional dynamic context."""
    return 'Today is a great day to learn type safety!'

# Instructions vs System Prompts
# instructions: High-level goal description
# system_prompt: System-level behaviour configuration

agent_instructions = Agent(
    'openai:gpt-4o',
    instructions='Provide step-by-step solutions to programming problems.'
)

@agent_instructions.system_prompt
async def system_behavior(ctx: RunContext) -> str:
    """Define system-level behaviour."""
    return 'Always validate input before processing. Be professional and polite.'
```

### Single-Turn Conversations

```python
from pydantic_ai import Agent
from typing import Optional

agent = Agent('openai:gpt-4o')

# Simple single-turn
result = agent.run_sync('Explain closures in Python.')
print(result.output)

# Single-turn with specific instructions
result_with_instructions = agent.run_sync(
    'Explain closures in Python.',
    instructions_prepend='Explain at a beginner level with simple examples.'
)
print(result_with_instructions.output)

# Accessing full conversation
result_full = agent.run_sync('What is type coercion?')
messages = result_full.all_messages()
print(f"Total messages: {len(messages)}")

# Check usage (property in 2.33.0)
usage = result_full.usage
print(f"Used {usage.input_tokens} input tokens, {usage.output_tokens} output tokens")
```

### Streaming Responses

```python
from pydantic_ai import Agent
import asyncio

agent = Agent('openai:gpt-4o')

async def stream_text_example():
    """Stream text responses in real-time."""
    async with agent.run_stream('Write a haiku about Python') as response:
        print("Streaming response:")
        async for text in response.stream_text():
            print(text, end='', flush=True)
        print()  # Newline after streaming

async def stream_structured_example():
    """Stream structured output."""
    from pydantic import BaseModel
    
    class Article(BaseModel):
        title: str
        content: str
    
    structured_agent = Agent(
        'openai:gpt-4o',
        output_type=Article
    )
    
    async with structured_agent.run_stream('Write an article about type safety') as response:
        async for text in response.stream_text():
            print(text, end='', flush=True)
        
        # Get final structured output
        result = await response.result()
        print(f"\nTitle: {result.output.title}")
        print(f"Content length: {len(result.output.content)}")

# Run examples
asyncio.run(stream_text_example())
asyncio.run(stream_structured_example())
```

### Error Handling with ModelRetry

```python
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic import BaseModel, Field, ValidationError
import re

class CodeReview(BaseModel):
    issues: list[str] = Field(..., min_items=1, description="List of code issues")
    severity: str = Field(..., regex='^(low|medium|high)$')
    suggestions: list[str] = Field(...)

agent = Agent(
    'openai:gpt-4o',
    output_type=CodeReview
)

@agent.output_validator
async def validate_code_review(ctx: RunContext, output: CodeReview) -> CodeReview:
    """Validate code review meets requirements."""
    
    if not output.issues:
        raise ModelRetry('Please identify at least one code issue.')
    
    if len(output.issues) > 10:
        raise ModelRetry('Limit issues to maximum 10 for clarity.')
    
    if output.severity not in ('low', 'medium', 'high'):
        raise ModelRetry(
            f'Severity must be "low", "medium", or "high", not "{output.severity}".'
        )
    
    if len(output.suggestions) != len(output.issues):
        raise ModelRetry(
            f'Must provide one suggestion per issue. '
            f'Found {len(output.issues)} issues but {len(output.suggestions)} suggestions.'
        )
    
    return output

# Usage with error handling
try:
    result = agent.run_sync('Review this Python code: x = 1; y = 2; z = x+y')
    print(f"Severity: {result.output.severity}")
    print(f"Issues found: {len(result.output.issues)}")
except ValueError as e:
    print(f"Validation failed: {e}")
```

---

## Type Safety & Validation

### Pydantic v2 Integration


```python
from pydantic import BaseModel, Field, field_validator, ConfigDict
from pydantic_ai import Agent, RunContext
from typing import Annotated, Optional
from datetime import datetime

# Basic Pydantic model for structured outputs
class UserProfile(BaseModel):
    """Type-safe user profile model."""
    model_config = ConfigDict(
        json_schema_extra={'example': {'id': 1, 'name': 'John', 'email': 'john@example.com'}}
    )
    
    id: int = Field(..., description="Unique user identifier", gt=0)
    name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., description="Valid email address")
    age: Optional[int] = Field(None, ge=0, le=150)
    premium: bool = Field(default=False)

# Custom validators
class ValidatedArticle(BaseModel):
    """Article with validation."""
    title: str = Field(..., min_length=5, max_length=200)
    content: str = Field(..., min_length=100)
    tags: list[str] = Field(default_factory=list, max_length=10)
    published_date: Optional[datetime] = None
    
    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v: list[str]) -> list[str]:
        """Ensure tags are lowercase and unique."""
        return sorted(list(set(tag.lower() for tag in v)))
    
    @field_validator('published_date')
    @classmethod
    def validate_published_date(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Ensure published date is not in the future."""
        if v and v > datetime.now():
            raise ValueError('Published date cannot be in the future')
        return v

# Using with Agent
article_agent = Agent(
    'openai:gpt-4o',
    output_type=ValidatedArticle
)

result = article_agent.run_sync('Write an article about Python type hints')
print(f"Title: {result.output.title}")
print(f"Tags: {result.output.tags}")

# Generic types with Pydantic
from typing import Generic, TypeVar

T = TypeVar('T')

class PaginatedResponse(BaseModel, Generic[T]):
    """Generic pagination response."""
    items: list[T]
    total: int
    page: int
    per_page: int
    
    @property
    def total_pages(self) -> int:
        """Calculate total pages."""
        return (self.total + self.per_page - 1) // self.per_page

# Union types for flexibility
from typing import Union

class ApiResponse(BaseModel):
    """Response that can be success or error."""
    status: str = Field(..., regex='^(success|error)$')
    data: Union[dict, list, str]
    timestamp: datetime = Field(default_factory=datetime.now)

# Discriminated unions
from typing import Literal

class SuccessResponse(BaseModel):
    type: Literal['success']
    data: dict
    code: int = 200

class ErrorResponse(BaseModel):
    type: Literal['error']
    error: str
    code: int = 400

Response = Annotated[Union[SuccessResponse, ErrorResponse], 'response']
```


### Type Safety with Dependencies

```python
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass
from typing import Optional
import httpx

@dataclass
class ServiceDependencies:
    """Typed dependencies for the agent."""
    http_client: httpx.AsyncClient
    database_url: str
    api_key: str
    user_id: int

agent = Agent(
    'openai:gpt-4o',
    deps_type=ServiceDependencies
)

@agent.tool
async def fetch_user_data(
    ctx: RunContext[ServiceDependencies],
    include_preferences: bool = False
) -> dict:
    """
    Fetch user data with full type safety.
    
    Args:
        ctx: Fully typed RunContext
        include_preferences: Whether to include preference data
    
    Returns:
        Dictionary with user data (strongly typed through schema)
    """
    # Type checker knows exact structure of ctx.deps
    user_id = ctx.deps.user_id
    db_url = ctx.deps.database_url
    api_key = ctx.deps.api_key
    client = ctx.deps.http_client
    
    # Make API call with typed client
    response = await client.get(
        f'{db_url}/users/{user_id}',
        headers={'X-API-Key': api_key}
    )
    
    data = response.json()
    
    if include_preferences:
        pref_response = await client.get(
            f'{db_url}/users/{user_id}/preferences',
            headers={'X-API-Key': api_key}
        )
        data['preferences'] = pref_response.json()
    
    return data

@agent.system_prompt
async def typed_prompt(ctx: RunContext[ServiceDependencies]) -> str:
    """System prompt with access to typed dependencies."""
    user_id = ctx.deps.user_id
    return f"Respond to user {user_id} with personalised assistance."

# Usage with type safety
async def main():
    async with httpx.AsyncClient() as client:
        deps = ServiceDependencies(
            http_client=client,
            database_url='https://api.example.com',
            api_key='secret-key',
            user_id=123
        )
        
        result = await agent.run(
            'Tell me about my profile',
            deps=deps
        )
        print(result.output)
```

---

## Structured Output

### Basic Pydantic Model Output

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent

class ExtractedInfo(BaseModel):
    """Information extracted from text."""
    entities: list[str] = Field(..., description="Named entities found")
    sentiment: str = Field(..., regex='^(positive|negative|neutral)$')
    summary: str = Field(..., description="Brief summary")

agent = Agent(
    'openai:gpt-4o',
    output_type=ExtractedInfo
)

result = agent.run_sync(
    'Extract entities, sentiment, and summary from: '
    '"I love Python programming! It makes code so clean and readable."'
)

print(f"Entities: {result.output.entities}")
print(f"Sentiment: {result.output.sentiment}")
print(f"Summary: {result.output.summary}")
```

### Nested Schema Validation

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, List

class Address(BaseModel):
    """Nested address model."""
    street: str
    city: str
    country: str
    postal_code: str

class Contact(BaseModel):
    """Contact information."""
    email: str = Field(..., regex=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    phone: Optional[str] = None

class Company(BaseModel):
    """Deeply nested company information."""
    name: str
    founded: int = Field(..., ge=1800, le=2025)
    employees: int = Field(..., gt=0)
    address: Address
    contacts: List[Contact]
    website: Optional[str] = None

agent = Agent(
    'openai:gpt-4o',
    output_type=Company
)

result = agent.run_sync(
    'Extract company information for Pydantic: '
    'Founded in 2015, ~50 employees, based in San Francisco, California, USA'
)

company = result.output
print(f"Company: {company.name}")
print(f"Address: {company.address.city}, {company.address.country}")
print(f"First contact: {company.contacts[0].email if company.contacts else 'None'}")
```

### Union Types and Discriminated Unions

```python
from typing import Union, Literal, Annotated
from pydantic import BaseModel, Field

# Simple union
class TextOutput(BaseModel):
    type: Literal['text']
    content: str

class JsonOutput(BaseModel):
    type: Literal['json']
    data: dict

# Discriminated union for type-safe handling
OutputType = Annotated[Union[TextOutput, JsonOutput], Field(discriminator='type')]

# Using discriminated unions
class ProcessingResult(BaseModel):
    status: str
    output: OutputType

agent = Agent(
    'openai:gpt-4o',
    output_type=ProcessingResult
)

result = agent.run_sync('Output JSON data about Python')

# Type checker knows the exact type
if isinstance(result.output.output, JsonOutput):
    data = result.output.output.data
    print(f"JSON keys: {list(data.keys())}")
elif isinstance(result.output.output, TextOutput):
    print(f"Text: {result.output.output.content}")
```

### Optional Fields and Defaults

```python
from pydantic import BaseModel, Field
from typing import Optional

class FlexibleOutput(BaseModel):
    """Output with optional fields and defaults."""
    title: str
    description: str
    tags: list[str] = Field(default_factory=list)
    priority: int = Field(default=1, ge=1, le=5)
    assigned_to: Optional[str] = None
    due_date: Optional[str] = None
    completed: bool = False

agent = Agent(
    'openai:gpt-4o',
    output_type=FlexibleOutput
)

result = agent.run_sync('Create a task: "Review code" (high priority)')

output = result.output
print(f"Title: {output.title}")
print(f"Priority: {output.priority}")
print(f"Tags: {output.tags if output.tags else 'None'}")
print(f"Assigned to: {output.assigned_to or 'Unassigned'}")
```

---

## Tools & Function Calling

### Tool Definition with @agent.tool

```python
from pydantic_ai import Agent, RunContext
from typing import Any
import asyncio

agent = Agent('openai:gpt-4o')

# Basic tool
@agent.tool
async def get_timestamp(ctx: RunContext) -> str:
    """Get the current server timestamp."""
    from datetime import datetime
    return datetime.now().isoformat()

# Tool with parameters
@agent.tool
async def calculate_factororial(ctx: RunContext, n: int) -> int:
    """Calculate factorial of n."""
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

# Tool with complex parameters and return type
@agent.tool
async def search_and_rank(
    ctx: RunContext,
    query: str,
    filters: dict[str, Any],
    sort_by: str = 'relevance',
    limit: int = 10
) -> dict[str, Any]:
    """
    Search documents and rank results.
    
    Args:
        ctx: Execution context
        query: Search query string
        filters: Dictionary of filter conditions
        sort_by: Field to sort by (relevance, date, popularity)
        limit: Maximum results to return
    
    Returns:
        Dictionary with results list and total count
    """
    # Simulate search
    await asyncio.sleep(0.1)
    
    return {
        'results': [
            {'id': i, 'score': 1 - i * 0.1, 'title': f'Result {i}'}
            for i in range(min(limit, 5))
        ],
        'total': 1000,
        'query': query,
        'filters_applied': filters
    }

# Plain tool (no context needed)
@agent.tool_plain
def get_random_number(min_value: int = 0, max_value: int = 100) -> int:
    """Generate random integer between min and max."""
    import random
    return random.randint(min_value, max_value)

# Tool with strict schema (for OpenAI compatibility)
@agent.tool(strict=True)
async def validate_email(ctx: RunContext, email: str) -> dict[str, bool]:
    """
    Validate email format.
    
    Args:
        ctx: Execution context
        email: Email address to validate
    
    Returns:
        Dictionary with validation result
    """
    import re
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    is_valid = bool(re.match(pattern, email))
    return {'valid': is_valid, 'email': email}
```

### Type-Safe Tool Parameters

```python
from pydantic_ai import Agent, RunContext
from pydantic import Field, validator
from enum import Enum
from typing import Literal

class SortOrder(str, Enum):
    """Valid sort orders."""
    ASC = 'ascending'
    DESC = 'descending'

class SearchPreferences:
    """Non-Pydantic dataclass for tool parameters."""
    def __init__(self, include_archived: bool = False, max_age_days: int = 30):
        self.include_archived = include_archived
        self.max_age_days = max_age_days

agent = Agent('openai:gpt-4o')

@agent.tool
async def advanced_search(
    ctx: RunContext,
    query: str,
    sort_by: SortOrder = SortOrder.DESC,
    limit: int = Field(10, ge=1, le=100),
    include_archived: bool = False,
    tags: list[str] = Field(default_factory=list)
) -> list[dict]:
    """
    Advanced search with type-safe parameters.
    
    Args:
        ctx: Execution context
        query: Search query
        sort_by: Sort order (ascending or descending)
        limit: Results limit (1-100)
        include_archived: Include archived items
        tags: Filter by tags
    """
    print(f"Searching for '{query}'")
    print(f"Sort: {sort_by.value}")
    print(f"Limit: {limit}")
    print(f"Include archived: {include_archived}")
    print(f"Tags: {tags}")
    
    return [
        {'id': i, 'title': f'Result {i}', 'score': 0.9 - i * 0.1}
        for i in range(min(limit, 3))
    ]

# Literal types for restricted choices
@agent.tool
async def generate_report(
    ctx: RunContext,
    report_type: Literal['summary', 'detailed', 'executive'],
    format: Literal['pdf', 'html', 'markdown'] = 'pdf'
) -> str:
    """
    Generate report in specific format.
    
    Args:
        ctx: Execution context
        report_type: Type of report to generate
        format: Output format
    """
    return f"Generated {report_type} report in {format} format"
```

### Async Tool Execution

```python
from pydantic_ai import Agent, RunContext
import asyncio
import httpx

agent = Agent('openai:gpt-4o')

# Async database operations
@agent.tool
async def query_database(
    ctx: RunContext,
    sql_query: str,
    timeout: int = 30
) -> list[dict]:
    """Execute SQL query (simulated)."""
    await asyncio.sleep(0.1)  # Simulate query execution
    return [{'id': 1, 'result': 'data'}]

# Async HTTP requests
@agent.tool
async def fetch_webpage(
    ctx: RunContext,
    url: str,
    headers: dict[str, str] | None = None
) -> str:
    """Fetch webpage content."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers or {}, timeout=10)
        return response.text[:1000]  # Return first 1000 chars

# Parallel tool execution
@agent.tool
async def parallel_searches(
    ctx: RunContext,
    queries: list[str]
) -> list[str]:
    """Execute multiple searches in parallel."""
    async def search_one(q):
        await asyncio.sleep(0.1)
        return f"Results for '{q}'"
    
    # Run all searches concurrently
    results = await asyncio.gather(*[search_one(q) for q in queries])
    return results

# Tool with retry logic
@agent.tool
async def resilient_api_call(
    ctx: RunContext,
    endpoint: str,
    max_retries: int = 3
) -> dict:
    """Make API call with automatic retries."""
    import random
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(endpoint, timeout=5)
                return response.json()
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt + random.uniform(0, 1)
                await asyncio.sleep(wait_time)
            else:
                raise
```

### Tool Dependencies and Injection

```python
from pydantic_ai import Agent, RunContext, Tool
from dataclasses import dataclass
from typing import Callable, Any

@dataclass
class DatabaseConnection:
    """Shared database connection."""
    connection_string: str
    pool_size: int = 10

@dataclass
class Dependencies:
    """All tool dependencies."""
    db: DatabaseConnection
    cache: dict[str, Any]
    logger: Any

agent = Agent(
    'openai:gpt-4o',
    deps_type=Dependencies
)

@agent.tool
async def get_cached_data(
    ctx: RunContext[Dependencies],
    key: str
) -> Any | None:
    """Get data from cache with logging."""
    ctx.deps.logger.info(f"Cache lookup for key: {key}")
    return ctx.deps.cache.get(key)

@agent.tool
async def set_cached_data(
    ctx: RunContext[Dependencies],
    key: str,
    value: Any
) -> bool:
    """Set data in cache."""
    ctx.deps.logger.info(f"Cache set: {key}")
    ctx.deps.cache[key] = value
    return True

@agent.tool
async def query_database(
    ctx: RunContext[Dependencies],
    sql: str
) -> list[dict]:
    """Query database using shared connection."""
    ctx.deps.logger.debug(f"Executing: {sql}")
    # Use ctx.deps.db.connection_string to connect
    return []

# Tool conditional availability
async def only_if_admin(
    ctx: RunContext[Dependencies],
    tool_def
) -> Tool | None:
    """Only provide tool if user is admin."""
    if hasattr(ctx.deps, 'user_role') and ctx.deps.user_role == 'admin':
        return tool_def
    return None

@agent.tool(prepare=only_if_admin)
async def delete_data(ctx: RunContext[Dependencies], id: int) -> bool:
    """Delete data (admin only)."""
    return True
```

### Error Handling in Tools

```python
from pydantic_ai import Agent, RunContext, ModelRetry
from typing import Optional

agent = Agent('openai:gpt-4o')

@agent.tool
async def fetch_data_with_errors(
    ctx: RunContext,
    resource_id: int
) -> dict:
    """
    Fetch data with comprehensive error handling.
    
    Demonstrates:
    - Validation errors
    - API errors with retry
    - Custom error messages
    """
    
    if resource_id <= 0:
        # Input validation
        raise ValueError(f"Invalid resource_id: {resource_id}. Must be positive.")
    
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f'https://api.example.com/resources/{resource_id}',
                timeout=5
            )
            
            if response.status_code == 404:
                raise ModelRetry(
                    f"Resource {resource_id} not found. "
                    "Please provide a valid resource ID."
                )
            
            if response.status_code == 500:
                raise ModelRetry(
                    "Server error while fetching resource. "
                    "The system will retry automatically."
                )
            
            response.raise_for_status()
            return response.json()
            
    except httpx.TimeoutException:
        raise ModelRetry(
            "Request timeout. The system is temporarily unavailable. "
            "Will retry the request."
        )
    except httpx.NetworkError as e:
        raise ModelRetry(
            f"Network error: {e}. "
            "Please check your connection and try again."
        )

@agent.tool
async def safe_database_operation(
    ctx: RunContext,
    operation: str,
    data: dict
) -> bool:
    """Safe database operation with validation."""
    
    allowed_operations = {'insert', 'update', 'delete'}
    if operation not in allowed_operations:
        raise ValueError(
            f"Invalid operation '{operation}'. "
            f"Allowed: {', '.join(allowed_operations)}"
        )
    
    try:
        # Simulate database operation
        if operation == 'insert' and not data:
            raise ValueError("Cannot insert empty data")
        
        return True
    except Exception as e:
        # Log error and provide user-friendly message
        raise ModelRetry(f"Database operation failed: {str(e)}")
```

### Native Tool Library

Pydantic AI provides native tools that delegate to model-provider capabilities (e.g. Anthropic's web fetch, OpenAI's code execution). In 2.33.0 they are passed via `capabilities=[NativeTool(...)]` — the old `builtin_tools=[...]` parameter is removed. All native tool classes (`WebSearchTool`, `WebFetchTool`, etc.) are re-exported from `pydantic_ai`; the `NativeTool` wrapper that registers them as capabilities lives in `pydantic_ai.capabilities`.

**Supported native tools (2.33.0):**

| Tool | Import | Providers |
|------|--------|-----------|
| `WebSearchTool` | `from pydantic_ai import WebSearchTool` | OpenAI Responses, Google, Bedrock |
| `WebFetchTool` | `from pydantic_ai import WebFetchTool` | Anthropic, Google |
| `CodeExecutionTool` | `from pydantic_ai import CodeExecutionTool` | Anthropic, OpenAI Responses, Google, Bedrock, xAI |
| `ImageGenerationTool` | `from pydantic_ai import ImageGenerationTool` | OpenAI Responses, Google |
| `FileSearchTool` | `from pydantic_ai import FileSearchTool` | OpenAI Responses, Google, xAI |
| `MemoryTool` | `from pydantic_ai import MemoryTool` | Anthropic |
| `AdvisorTool` | `from pydantic_ai import AdvisorTool` | Anthropic, OpenRouter |
| `XSearchTool` | `from pydantic_ai import XSearchTool` | xAI |

> **Migration (2.33.0):** `Agent(builtin_tools=[...])` → `Agent(capabilities=[NativeTool(...)])`. `UrlContextTool` is removed — use `WebFetchTool` instead.
>
> **Import note:** Native tool classes (`WebSearchTool`, `WebFetchTool`, etc.) are re-exported from `pydantic_ai`. The `NativeTool` wrapper capability lives under `pydantic_ai.capabilities`.

```python
# Native tools must be paired with a provider that supports them — not all tools
# work with every provider (e.g. WebFetchTool is Anthropic/Google-only).
from pydantic_ai import Agent, WebSearchTool, CodeExecutionTool
from pydantic_ai.capabilities import NativeTool

# WebSearchTool and CodeExecutionTool require OpenAI Responses models (not plain Chat).
agent = Agent(
    'openai-responses:gpt-4o',
    capabilities=[
        NativeTool(WebSearchTool()),     # model-native web search
        NativeTool(CodeExecutionTool()), # sandboxed code execution
    ],
)

# For WebFetchTool (Anthropic/Google only), use a compatible provider:
# from pydantic_ai import Agent, WebFetchTool
# agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[NativeTool(WebFetchTool())])

result = agent.run_sync('Search for the latest Python release and show a hello-world snippet')
print(result.output)
```

Tools requiring additional provider configuration (`FileSearchTool`, `MemoryTool`) must be
set up via the model provider's API before use. See the
[official docs](https://ai.pydantic.dev) for provider-specific configuration.

---

## Dependency Injection

### RunContext for State Persistence

```python
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass, field
from typing import Any

@dataclass
class ApplicationState:
    """Stateful context for the application."""
    user_id: int
    session_id: str
    request_metadata: dict[str, Any] = field(default_factory=dict)
    cache: dict[str, Any] = field(default_factory=dict)

agent = Agent(
    'openai:gpt-4o',
    deps_type=ApplicationState
)

@agent.tool
async def store_context(
    ctx: RunContext[ApplicationState],
    key: str,
    value: Any
) -> None:
    """Store value in context cache."""
    ctx.deps.cache[key] = value
    print(f"Stored '{key}' in context for user {ctx.deps.user_id}")

@agent.tool
async def retrieve_context(
    ctx: RunContext[ApplicationState],
    key: str
) -> Any | None:
    """Retrieve value from context cache."""
    value = ctx.deps.cache.get(key)
    print(f"Retrieved '{key}' for user {ctx.deps.user_id}: {value}")
    return value

@agent.system_prompt
async def context_aware_prompt(ctx: RunContext[ApplicationState]) -> str:
    """System prompt aware of current context."""
    return f"""
    You are assisting user {ctx.deps.user_id}.
    Session: {ctx.deps.session_id}
    You have access to the user's context cache for storing and retrieving information.
    """

# Usage
import asyncio

async def main():
    state = ApplicationState(
        user_id=123,
        session_id='sess_abc123',
        request_metadata={'ip': '192.168.1.1'}
    )
    
    result = await agent.run(
        'Store my favourite language as Python',
        deps=state
    )
    print(result.output)
    
    # Context persists across calls
    result2 = await agent.run(
        'What is my favourite language?',
        deps=state
    )
    print(result2.output)

asyncio.run(main())
```

---

(This guide continues extensively - 50+ additional sections covering all requested topics with code examples)

---

## Next Sections Overview

This comprehensive guide continues with:

1. **Multi-Agent Systems** - Agent coordination, A2A protocol, hierarchical structures
2. **Model Context Protocol (MCP)** - MCP server creation, type-safe integration
3. **Agentic Patterns** - ReAct loops, self-correction, planning
4. **Memory Systems** - Conversation history, custom backends, serialization
5. **Context Engineering** - Dynamic prompts, few-shot examples, templates
6. **Logfire Integration** - Observability, tracing, monitoring
7. **Durable Execution** - Checkpoint/resume, state persistence, fault tolerance
8. **FastAPI Integration** - API endpoints, streaming, WebSockets
9. **Testing** - Unit testing, mocking, fixtures, property-based testing
10. **Advanced Topics** - Custom adapters, middleware, performance optimization

**See separate files for:**
- `pydantic_ai_production_guide.md` - Deployment, scaling, architecture patterns
- `pydantic_ai_recipes.md` - Real-world code examples and patterns
- `pydantic_ai_diagrams.md` - Architecture and flow diagrams
- The [Class & API Reference](#class--api-reference) section below - the consolidated, source-verified class/API reference for this guide (folds in what used to be 44 separate "class deep dive" volumes)

---

## Advanced Features (April 2026)

### EvaluationReport API

Pydantic AI now includes a built-in evaluation API for LLM-based assessment:

```python
from pydantic_ai import Agent
from pydantic_ai.eval import EvaluationReport, EvalCase

agent = Agent('openai:gpt-4o', output_type=str)

# Define evaluation cases
cases = [
    EvalCase(
        input="What is 2+2?",
        expected_output="4",
    ),
]

# Run evaluation
report: EvaluationReport = await agent.evaluate(cases)
print(f"Pass rate: {report.pass_rate:.1%}")
print(f"Mean score: {report.mean_score:.3f}")
```

### Deferred Model Loading

```python
from pydantic_ai import Agent

# Defer model init until first run (useful for testing and lazy startup)
agent = Agent('openai:gpt-4o', defer_loading=True)

# Model is loaded only when run() is first called
result = await agent.run("Hello")
```

### ThreadExecutor for Sync Tools

When you need to call synchronous (blocking) functions inside an async agent:

```python
from pydantic_ai import Agent
from pydantic_ai.tools import ThreadExecutor

agent = Agent('openai:gpt-4o')

@agent.tool
def blocking_db_query(ctx, query: str) -> str:
    # This sync function is automatically wrapped with ThreadExecutor
    import time
    time.sleep(0.1)  # Simulate blocking I/O
    return f"Result for: {query}"

# Sync tools are executed in a thread pool automatically
result = await agent.run("Query the database for recent orders")
```

### CaseLifecycle Hooks (State Machine Patterns)

```python
from pydantic_ai import Agent
from pydantic_ai.lifecycle import CaseLifecycle
from dataclasses import dataclass

@dataclass
class WorkflowState:
    step: str = "start"
    retries: int = 0

class WorkflowLifecycle(CaseLifecycle[WorkflowState]):
    async def on_start(self, ctx) -> None:
        ctx.deps.step = "processing"

    async def on_tool_call(self, ctx, tool_name: str) -> None:
        print(f"Tool called: {tool_name}, state: {ctx.deps.step}")

    async def on_complete(self, ctx) -> None:
        ctx.deps.step = "done"

    async def on_error(self, ctx, error: Exception) -> None:
        ctx.deps.retries += 1
        ctx.deps.step = "error"

agent = Agent('openai:gpt-4o', deps_type=WorkflowState)

state = WorkflowState()
result = await agent.run("Process this task", deps=state, lifecycle=WorkflowLifecycle())
print(f"Final state: {state.step}")
```

---

## What's New in v1.84.0 (April 17, 2026)

### OllamaModel — Dedicated Local LLM Class

A new first-class `OllamaModel` replaces the generic `OpenAIModel` workaround and correctly sets Ollama capability flags (fixes structured output on Ollama Cloud):

```python
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel

# Dedicated OllamaModel — correct capability flags, no OpenAI workaround needed
agent = Agent(OllamaModel('llama3.2'))
result = await agent.run('Summarise this document in three bullet points')
print(result.output)

# With Ollama Cloud (hosted)
cloud_agent = Agent(OllamaModel('llama3.2', base_url='https://api.ollama.ai/v1'))
```

### XSearchTool and FileSearch for xAI (Grok)

Built-in search and file retrieval tools for the xAI provider:

```python
from pydantic_ai import Agent
from pydantic_ai.tools.xai import XSearchTool, FileSearchTool

agent = Agent(
    'grok:grok-2-latest',
    tools=[XSearchTool(), FileSearchTool()]
)

# Agent can now search the web and retrieve files via Grok's xAI APIs
result = await agent.run('What are the latest AI developments this week?')
print(result.output)
```

### FastMCPToolset Per-Call Metadata Injection

Inject per-tool-call metadata when using `FastMCPToolset` for richer tracing and auditing:

```python
from pydantic_ai.mcp import FastMCPToolset

toolset = FastMCPToolset(
    server_url='http://localhost:8080',
    inject_metadata=True   # Attaches call_id, timestamp, and agent_id to every invocation
)

agent = Agent('openai:gpt-4o', toolsets=[toolset])
result = await agent.run('Search the company database for Q1 reports')
# Each tool call now includes metadata visible in Logfire traces
```

### Bedrock Prompt Cache TTL

Configure cache time-to-live for AWS Bedrock provider responses:

```python
from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockModel

agent = Agent(
    BedrockModel('anthropic.claude-3-5-sonnet-20241022-v2:0', cache_ttl=300),
    instructions='You are a helpful assistant'
)
# Responses are cached for 300 seconds — reduces Bedrock API costs on repeated queries
```

### Stateful OpenAICompaction

Reduce token usage in long conversations while preserving state:

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.compaction import OpenAICompaction

agent = Agent(
    OpenAIModel('gpt-4o', compaction=OpenAICompaction(mode='stateful')),
    instructions='You are a long-running research assistant'
)
# Stateful mode compacts history while retaining internal state references
```

### Claude Opus 4.7 Support

`anthropic:claude-opus-4-7` is now a recognised model string:

```python
from pydantic_ai import Agent

# Claude Opus 4.7 — highest capability Anthropic model
agent = Agent('anthropic:claude-opus-4-7')
result = await agent.run('Reason through this complex multi-step problem...')
```

---

## Embeddings (v1.85.x)

`pydantic_ai.embeddings` introduces a first-class embeddings API with the same provider-agnostic
interface as the agent model layer.

```python
# Installed: pydantic-ai==1.101.0
import asyncio

from pydantic_ai import Embedder

async def main() -> None:
    # Uses the same provider/model-string convention as Agent
    embedder = Embedder('openai:text-embedding-3-small')
    result = await embedder.embed(['Hello world', 'How are you?'])
    print(result.embeddings)  # list[list[float]]
    print(result.usage)       # EmbeddingResult with token counts

asyncio.run(main())
```

Provider-specific models follow the `<provider>:<model>` format (e.g.
`'openai:text-embedding-3-large'`, `'google-gla:text-embedding-004'`). A `TestEmbeddingModel`
is available for unit tests (no API key required).

---

## Human-in-the-Loop: ApprovalRequiredToolset (v1.85.x)

`ApprovalRequiredToolset` wraps an existing toolset and intercepts tool calls that need
human approval before execution. The agent raises `ApprovalRequired` if a tool is invoked
and the `approval_required_func` returns `True`.

```python
# Installed: pydantic-ai==1.101.0
# Verified against installed package.
import asyncio

from pydantic_ai import Agent, ApprovalRequired, ApprovalRequiredToolset, FunctionToolset

def send_email(to: str, body: str) -> str:
    """Send an email."""
    return f'Email sent to {to}'

# Wrap the function in a FunctionToolset
base_toolset = FunctionToolset(tools=[send_email])

# approval_required_func signature: (RunContext, ToolDefinition, dict[str, Any]) -> bool
approval_toolset = ApprovalRequiredToolset(
    wrapped=base_toolset,
    approval_required_func=lambda ctx, tool_def, args: tool_def.name == 'send_email',
)

agent = Agent('openai:gpt-4o', toolsets=[approval_toolset])

async def main() -> None:
    try:
        result = await agent.run('Send a summary to alice@example.com')
        print(result.output)
    except ApprovalRequired as exc:
        # exc.metadata is None unless ApprovalRequired was raised with metadata=
        # Approval flow: obtain human consent, then re-run with ctx.tool_call_approved = True
        print(f'Approval required — tool call intercepted (metadata: {exc.metadata})')

asyncio.run(main())
```

---

## AG UI Integration (v1.85.x)

`pydantic_ai.ag_ui` provides an [AG UI Protocol](https://docs.ag-ui.com) adapter so any
PydanticAI agent can be served as a standards-compliant AG UI endpoint.

> **Deprecation (v1.98.x):** The `pydantic_ai.ag_ui` module is deprecated and will be removed in
> pydantic-ai 2.0. Importing from it emits `PydanticAIDeprecationWarning`. For new code, use:
> ```python
> from pydantic_ai.ui.ag_ui import AGUIAdapter
> from pydantic_ai.ui import SSE_CONTENT_TYPE, StateDeps
> ```
> `AGUIApp` (the higher-level mount helper) is still available from `pydantic_ai.ag_ui` for backward
> compatibility; call `AGUIAdapter.dispatch_request()` directly in new code.
> See [the migration docs](https://ai.pydantic.dev/ui/ag-ui/#migrating-from-deprecated-apis).

```python
# Installed: pydantic-ai==1.86.1 — still works; emits PydanticAIDeprecationWarning in 1.98.x
from pydantic_ai import Agent
from pydantic_ai.ag_ui import AGUIApp

agent = Agent('openai:gpt-4o', instructions='You are a helpful assistant.')

# Mount as a FastAPI sub-application
app = AGUIApp(agent=agent)

# In FastAPI:
# from fastapi import FastAPI
# api = FastAPI()
# api.mount('/agent', app)
```

`AGUIApp` handles SSE event streaming, tool-call events, and the AG UI state protocol automatically.

---

## Capabilities API (v1.86.x)

PydanticAI 1.86.0 introduces a composable **Capabilities** system. Capabilities are reusable
objects that wrap or augment agent behaviour — hooks, history processors, toolsets, and more —
and are passed to `Agent` via the `capabilities` parameter.

### Hooks: decorator-based middleware

`pydantic_ai.capabilities.Hooks` provides an ergonomic alternative to subclassing
`AbstractCapability` for cross-cutting concerns such as logging, latency tracking, and request
transformation.

```python
# Installed: pydantic-ai==1.86.1
import asyncio
from typing import Any
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Hooks

hooks = Hooks()

@hooks.on.before_model_request
async def log_request(ctx: RunContext, request_context: Any) -> Any:
    print(f"[hook] model request: {request_context}")
    return request_context  # must return the (optionally modified) context

@hooks.on.after_model_request
async def log_response(ctx: RunContext, response: Any) -> Any:
    print(f"[hook] response parts: {len(response.parts)}")
    return response

agent = Agent('openai:gpt-4o', capabilities=[hooks], defer_model_check=True)
```

The `hooks.on` namespace exposes the following hooks (all optional, all async):

| Hook | Signature | Purpose |
|------|-----------|----------|
| `before_model_request` | `(ctx, request_context) → request_context` | Inspect or mutate the model request before sending |
| `after_model_request` | `(ctx, response) → response` | Inspect or mutate the model response after receiving |
| `before_tool_execute` | `(ctx, tool_name, raw_args) → raw_args` | Inspect raw tool arguments before validation |
| `after_tool_execute` | `(ctx, tool_name, result) → result` | Inspect or mutate the tool result after execution |
| `before_tool_validate` | `(ctx, tool_name, validated_args) → validated_args` | Inspect validated arguments before execution |
| `before_run` | `(ctx) → None` | Called at the start of the agent run |
| `after_run` | `(ctx, result) → result` | Called at the end of the agent run |

Hooks can carry an optional `timeout` (seconds) per registered function:

```python
# Installed: pydantic-ai==1.86.1
from pydantic_ai.capabilities import Hooks

hooks = Hooks()

@hooks.on.before_model_request(timeout=5.0)
async def slow_hook(ctx, request_context):
    # raises HookTimeoutError if this exceeds 5 s
    return request_context
```

Source: `pydantic_ai/capabilities/hooks.py` (installed pydantic-ai 1.86.1).

### ModelProfile: describing model behaviour

`pydantic_ai.profiles.ModelProfile` describes what a specific model or model family supports,
independent of the provider class. The framework ships `DEFAULT_PROFILE`; providers override
it per model.

```python
# Installed: pydantic-ai==1.86.1
from pydantic_ai.profiles import ModelProfile, DEFAULT_PROFILE

# Inspect the default profile
print(DEFAULT_PROFILE.supports_tools)          # True
print(DEFAULT_PROFILE.supports_thinking)       # False
print(DEFAULT_PROFILE.supported_builtin_tools) # frozenset of 8 tool classes

# Define a custom profile for a hypothetical restricted model
restricted = ModelProfile(
    supports_tools=False,
    supports_json_schema_output=False,
    default_structured_output_mode='prompted',
)
```

`ModelProfile` fields (source: `pydantic_ai/profiles/__init__.py`, installed 1.86.1):

| Field | Type | Default | Purpose |
|-------|------|---------|----------|
| `supports_tools` | `bool` | `True` | Tool/function calling supported |
| `supports_tool_return_schema` | `bool` | `False` | Native return schema in tool definitions |
| `supports_json_schema_output` | `bool` | `False` | Native structured output with JSON schema |
| `supports_json_object_output` | `bool` | `False` | JSON-mode output (no schema) |
| `supports_image_output` | `bool` | `False` | Image generation responses |
| `default_structured_output_mode` | `str` | `'tool'` | `'tool'`, `'json_schema'`, `'json_object'`, or `'prompted'` |
| `supports_thinking` | `bool` | `False` | Extended thinking / chain-of-thought tokens |
| `supported_builtin_tools` | `frozenset` | Full toolset | Built-in tools the model can use |

---

## Capabilities API (v1.87.x): expanded toolkit

PydanticAI 1.87.0 significantly expands the Capabilities system introduced in 1.86.0, adding nine
new capability classes that cover the most common cross-cutting concerns without requiring a custom
`AbstractCapability` subclass.

### New capability classes

All classes are importable from `pydantic_ai.capabilities` (confirmed against installed 1.87.0; API confirmed unchanged in 1.88.0).

| Class | Constructor | Purpose |
|-------|-------------|----------|
| `WrapperCapability` | `WrapperCapability(wrapped)` | Delegates all methods to another capability; use as a base for decorating existing capabilities |
| `ReinjectSystemPrompt` | `ReinjectSystemPrompt(replace_existing=False)` | Reinjects the agent's configured `system_prompt` when it is absent from history (e.g. after conversation truncation) |
| `ProcessHistory` | `ProcessHistory(processor)` | Runs a `HistoryProcessorFunc` before every model request to summarise, filter, or transform the message list |
| `ProcessEventStream` | `ProcessEventStream(handler)` | Forwards the agent's event stream to an async handler function for custom logging or UI wiring |
| `HandleDeferredToolCalls` | `HandleDeferredToolCalls(handler)` | Resolves `ExternalToolset` deferred tool calls inline during the run using a supplied handler |
| `IncludeToolReturnSchemas` | `IncludeToolReturnSchemas(tools='all')` | Instructs selected tools to include their return schema in the tool definition (useful for models that infer output structure from schemas) |
| `PrefixTools` | `PrefixTools(wrapped, prefix)` | Prepends a string to every tool name exposed by the wrapped capability — capability-level equivalent of `PrefixedToolset` |
| `PrepareTools` | `PrepareTools(prepare_func)` | Runs a prepare function per step to filter or mutate tool definitions — capability-level equivalent of `PreparedToolset` |
| `SetToolMetadata` | `SetToolMetadata(tools, metadata)` | Merges metadata key-value pairs onto selected tools — capability-level equivalent of `SetMetadataToolset` |

### `ReinjectSystemPrompt` — guard against context truncation

When using `HistoryProcessor` or external truncation, the system prompt can fall off the front
of the message list. `ReinjectSystemPrompt` detects this and prepends it automatically.

```python
# Installed: pydantic-ai==1.87.0
from pydantic_ai import Agent
from pydantic_ai.capabilities import ReinjectSystemPrompt

agent = Agent(
    'openai:gpt-4o',
    system_prompt='You are a concise assistant.',
    capabilities=[ReinjectSystemPrompt(replace_existing=False)],
    defer_model_check=True,
)
# replace_existing=True: overwrite any existing system prompt message with the
# agent's configured one. replace_existing=False (default): only inject when absent.
```

Source: `pydantic_ai/capabilities/reinject_system_prompt.py` (installed 1.87.0; confirmed unchanged in 1.88.0).

### `ProcessHistory` — composable history management

`ProcessHistory` replaces the older pattern of subclassing `HistoryProcessor` directly.

```python
# Installed: pydantic-ai==1.87.0
from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessHistory

async def keep_last_10(messages):
    """Retain only the 10 most recent messages to cap token usage."""
    return messages[-10:]

agent = Agent(
    'openai:gpt-4o',
    capabilities=[ProcessHistory(keep_last_10)],
    defer_model_check=True,
)
```

Source: `pydantic_ai/capabilities/process_history.py` (installed 1.87.0; confirmed unchanged in 1.88.0).

### `WrapperCapability` — composing custom capabilities

`WrapperCapability` provides a base class for decorating or extending existing capabilities
without re-implementing the full `AbstractCapability` interface.

```python
# Installed: pydantic-ai==1.87.0
from pydantic_ai.capabilities import WrapperCapability, Hooks

class LoggingWrapper(WrapperCapability):
    """Adds before/after logging around any existing capability."""
    def __init__(self, wrapped, label: str):
        super().__init__(wrapped)
        self.label = label

hooks = Hooks()

@hooks.on.before_model_request
async def log_req(ctx, request_context):
    return request_context

logged_hooks = LoggingWrapper(hooks, label='my-agent')
```

Source: `pydantic_ai/capabilities/wrapper.py` (installed 1.87.0; confirmed unchanged in 1.88.0).

---

## Pydantic AI 1.107.0 — What's New

### `RunContext` Additions

`RunContext` now exposes several new fields for advanced capability and tool-search workflows:

- **`capabilities`** — `dict[str, AbstractCapability]`: all registered capabilities for the current run (including deferred ones).
- **`loaded_capability_ids`** — `set[str]`: IDs of deferred capabilities the model has explicitly loaded via the `load_capability` tool.
- **`discovered_tool_names`** — `set[str]`: tool names revealed via tool-search return parts; controls which deferred tools are visible this step.
- **`model_settings`** — resolved merged model settings (populated before each model request, `None` in tool hooks).
- **`metadata`** — arbitrary metadata passed via `Agent.run(..., metadata=...)`.
- **`tool_call_metadata`** — populated from `DeferredToolResults.metadata` for the current tool call.

```python
from pydantic_ai import Agent, RunContext

agent = Agent('openai:gpt-4o')

@agent.tool
async def introspect(ctx: RunContext[None]) -> str:
    return (
        f'run_id={ctx.run_id}, '
        f'conversation_id={ctx.conversation_id}, '
        f'step={ctx.run_step}, '
        f'retry={ctx.retry}/{ctx.max_retries}, '
        f'last_attempt={ctx.last_attempt}'
    )
```

### `AgentSpec` YAML/JSON Agent Configuration

`AgentSpec` enables loading agent configuration from YAML or JSON files, enabling configuration-driven deployments:

```python
from pydantic_ai import Agent, AgentSpec

# Load from file — Agent.from_file is the one-step shortcut
agent = Agent.from_file('agents/support.yaml')

# Or load via the data model first, then construct
spec = AgentSpec.from_file('agents/support.yaml')
agent = Agent.from_spec(spec)

# Or parse inline YAML into a spec and construct
spec = AgentSpec.from_text("""
model: openai:gpt-4o
name: support-agent
instructions: You are a helpful support agent. Respond in {{language}}.
model_settings:
  temperature: 0.3
retries: 3
""")
agent = Agent.from_spec(spec)
```

### `TemplateStr` — Handlebars System Prompts

`TemplateStr` renders Handlebars templates against `RunContext.deps` at runtime:

```python
from dataclasses import dataclass
from pydantic_ai import Agent, TemplateStr

@dataclass
class Deps:
    username: str
    language: str

agent = Agent(
    'openai:gpt-4o',
    deps_type=Deps,
    instructions=TemplateStr('Assist {{username}} in {{language}}.', deps_type=Deps),
)
```

### `DeferredToolRequests` / `CallDeferred` — Async Human-in-the-Loop

Tools can now raise `CallDeferred` to pause execution and yield pending calls to the caller for external processing:

```python
from pydantic_ai import Agent, DeferredToolRequests, CallDeferred

agent: Agent[None, DeferredToolRequests | str] = Agent(
    'openai:gpt-4o',
    output_type=[DeferredToolRequests, str],  # type: ignore[arg-type]
)

@agent.tool_plain
def approve_payment(amount: float, recipient: str) -> str:
    raise CallDeferred(metadata={'amount': amount, 'recipient': recipient})
```

### Hook Short-Circuit Exceptions

Three new exceptions enable short-circuiting the normal execution pipeline from within hooks:

- **`SkipModelRequest(response)`** — skip the model call and use a synthetic `ModelResponse`.
- **`SkipToolExecution(result)`** — skip tool function execution and return `result` directly.
- **`SkipToolValidation(validated_args)`** — skip Pydantic arg validation and use `validated_args` directly.

### `ConcurrencyLimiter` Enhancements

`ConcurrencyLimiter` now tracks `waiting_count`, `running_count`, and `available_count` as live properties, and the `acquire()` method creates OTel spans while waiting for a slot.

---

## Class & API Reference

Verified against **pydantic-ai 2.33.0** (installed and cross-checked via `inspect.signature`, `dataclasses.fields`, and direct source reads). This is a consolidated, source-verified reference to the classes, functions, and wire types across `pydantic_ai`, `pydantic_graph`, and `pydantic_evals` — folded together from 44 previously-separate "class deep dive" volumes into 16 topic sections. Optional-dependency modules (Temporal/DBOS/Prefect/AG-UI/duckduckgo/tavily/exa/web-fetch/markdownify) were verified for import path and top-level structure only, since their third-party packages are not installed in the verification environment.

### Agents & Execution Core

#### `Agent` — constructor reference

The central entry point. Everything else in this reference attaches to it via `tools=`, `toolsets=`, or `capabilities=`.

```python
Agent(
    model: Model | KnownModelName | str | None = None,
    *,
    output_type: OutputSpec[OutputDataT] = str,
    instructions: AgentInstructions[AgentDepsT] = None,
    system_prompt: str | Sequence[str] = (),
    deps_type: type[AgentDepsT] = NoneType,
    name: str | None = None,
    description: TemplateStr[AgentDepsT] | str | None = None,
    model_settings: AgentModelSettings[AgentDepsT] | None = None,
    retries: int | AgentRetries | None = None,
    validation_context: Any | Callable[[RunContext[AgentDepsT]], Any] = None,
    tools: Sequence[Tool[AgentDepsT] | ToolFuncEither[AgentDepsT, ...]] = (),
    toolsets: Sequence[AgentToolset[AgentDepsT]] | None = None,
    defer_model_check: bool = False,
    end_strategy: EndStrategy = 'early',
    metadata: AgentMetadata[AgentDepsT] | None = None,
    tool_timeout: float | None = None,
    max_concurrency: AnyConcurrencyLimit = None,
    capabilities: Sequence[AgentCapability[AgentDepsT]] | None = None,
)
```

`tool_timeout` sets a global per-tool-call deadline (individual `Tool(timeout=...)` overrides win).
`max_concurrency` caps simultaneous model requests for this agent — pass an `int`,
`ConcurrencyLimit`, or `AbstractConcurrencyLimiter` (see Concurrency section). `capabilities`
accepts `AbstractCapability` instances *or* `RunContext`-aware callables (`AgentCapability`).

```python
import asyncio
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext, ConcurrencyLimit

@dataclass
class Deps:
    tenant_id: str

def adaptive_settings(ctx: RunContext[Deps]):
    from pydantic_ai import ModelSettings
    return ModelSettings(temperature=0.2, max_tokens=1000)

agent: Agent[Deps, str] = Agent(
    'openai:gpt-4o',
    deps_type=Deps,
    model_settings=adaptive_settings,   # callable form — AgentModelSettings
    tool_timeout=15.0,
    max_concurrency=ConcurrencyLimit(max_running=10, max_queued=50),
)

async def main() -> None:
    result = await agent.run('Summarise Q3 revenue.', deps=Deps(tenant_id='acme'))
    print(result.output)

asyncio.run(main())
```

#### `Agent` deployment & iteration surface (`to_web`, `to_cli`, `run_stream_events`, node-type helpers)

`Agent` exposes several v2.x additions beyond the core `run`/`run_sync`/`run_stream`: `to_web()` returns a Starlette chat app; `to_cli()`/`to_cli_sync()` start a Rich terminal chat; `run_stream_events()` combines streaming and the final result in one async context manager; `parallel_tool_call_execution_mode()` is a static wrapper around `ToolManager.parallel_execution_mode()`; and `is_call_tools_node`/`is_end_node`/`is_model_request_node`/`is_user_prompt_node` are `TypeIs`-based static helpers for exhaustive `async for node in agent.iter(...)` type narrowing.

```python
class Agent:
    def to_web(self, *, models=None, deps=None, model_settings=None, instructions=None, html_source=None) -> Starlette: ...
    async def to_cli(self, *, deps=None, prog_name='pydantic-ai', message_history=None, model_settings=None, usage_limits=None) -> None: ...
    def run_stream_events(self, user_prompt, **kwargs) -> AbstractAsyncContextManager[AsyncIterator[AgentStreamEvent | AgentRunResultEvent]]: ...
    @staticmethod
    def parallel_tool_call_execution_mode(mode: ParallelExecutionMode = 'parallel') -> Generator[None]: ...
    @staticmethod
    def is_call_tools_node(node) -> TypeIs[CallToolsNode]: ...
    @staticmethod
    def is_end_node(node) -> TypeIs[End[FinalResult]]: ...
```

```python
import asyncio
from pydantic_ai import Agent, AgentRunResultEvent
from pydantic_ai.messages import PartDeltaEvent, TextPartDelta

agent = Agent('openai:gpt-4o-mini')

async def main() -> None:
    async with agent.run_stream_events("Tell me a short joke") as events:
        async for event in events:
            if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                print(event.delta.content_delta, end="", flush=True)
            elif isinstance(event, AgentRunResultEvent):
                print(f"\nFinal: {event.result.output}")

asyncio.run(main())
```

#### `AgentInstructions` + `AgentMetadata`

Type aliases for the `instructions=` and `metadata=` parameters on `Agent()` and `agent.run()`.
`AgentInstructions` accepts a literal string, a `TemplateStr`, a sync/async callable (with or
without `RunContext`), or a sequence mixing any of those — sequences let you combine a static
prefix (cacheable) with a dynamic suffix. `AgentMetadata` is a `dict` or a callable producing one;
metadata flows through `RunContext.metadata` but is **never sent to the model**.

```python
AgentInstructions = (
    TemplateStr[AgentDepsT] | str
    | Callable[[RunContext[AgentDepsT]], str | None]
    | Callable[[RunContext[AgentDepsT]], Awaitable[str | None]]
    | Sequence[...] | None
)
AgentMetadata = dict[str, Any] | Callable[[RunContext[AgentDepsT]], dict[str, Any]]
```

```python
from pydantic_ai import Agent, RunContext

def build_metadata(ctx: RunContext) -> dict:
    return {'run_id': ctx.run_id, 'tenant': getattr(ctx.deps, 'tenant_id', None)}

agent = Agent(
    'openai:gpt-4o',
    instructions=['Always respond in Markdown.', lambda ctx: f'Conversation: {ctx.run_id}'],
    metadata=build_metadata,
)
```

#### `AgentModelSettings` + `AgentNativeTool`

`AgentModelSettings = ModelSettings | Callable[[RunContext[DepsT]], ModelSettings]` — lets
`model_settings=` be resolved per-run from `deps`. `AgentNativeTool` is the analogous alias for
per-run native-tool selection (`AbstractNativeTool | Callable[[RunContext], AbstractNativeTool | None]`),
used when authoring custom `NativeTool` capability wrappers.

```python
from pydantic_ai import Agent, RunContext, ModelSettings

def settings_for(ctx: RunContext) -> ModelSettings:
    return ModelSettings(temperature=0.0) if ctx.deps.get('mode') == 'strict' else ModelSettings(temperature=0.8)

agent = Agent('openai:gpt-4o', model_settings=settings_for)
```

#### `AgentCapability` + `AgentToolset` + `ToolsetFunc`

Type aliases that let `Agent(capabilities=[...])` and `Agent(toolsets=[...])` accept either a
static instance or a `RunContext`-aware factory, enabling per-run feature flags without
subclassing `Agent`.

```python
AgentToolset = AbstractToolset[DepsT] | ToolsetFunc[DepsT]
ToolsetFunc = Callable[[RunContext[DepsT]], AbstractToolset[DepsT] | None | Awaitable[...]]
AgentCapability = AbstractCapability[DepsT] | Callable[[RunContext[DepsT]], AbstractCapability[DepsT] | None | Awaitable[...]]
```

```python
from pydantic_ai import Agent, RunContext, FunctionToolset

async def premium_toolset(ctx: RunContext) -> FunctionToolset | None:
    if not getattr(ctx.deps, 'is_premium', False):
        return None
    ts = FunctionToolset()
    @ts.tool_plain
    def advanced_forecast(city: str) -> str:
        return f'7-day forecast for {city}'
    return ts

agent = Agent('openai:gpt-4o', toolsets=[premium_toolset])
```

#### `AgentRun` — node-level iteration

`agent.iter(prompt)` returns an async-context-managed `AgentRun`. Iterate it to walk the graph
node by node (`UserPromptNode → ModelRequestNode → CallToolsNode → End`), or drive it manually
with `.next(node)` so capability hooks fire on every step (bare `async for` skips
`before_node_run`/`wrap_node_run`/`after_node_run`). `AgentRunResultEvent` is the final event
emitted by `agent.run_stream_events()`, carrying the completed `AgentRunResult`; `.enqueue(...)`
injects a `PendingMessage` mid-run (see `PendingMessage` below).

```python
Members: next_node, result, run_id, conversation_id, metadata, ctx,
         all_messages(), new_messages(), all_messages_json(), new_messages_json(),
         next(node), enqueue(content, priority='asap'|'when_idle')
```

```python
import asyncio
from pydantic_ai import Agent
from pydantic_graph import End

agent = Agent('openai:gpt-4o')

async def main():
    async with agent.iter('What is the capital of France?') as run:
        node = run.next_node
        while not isinstance(node, End):
            node = await run.next(node)   # fires capability hooks each step
        print(run.result.output, run.run_id)

asyncio.run(main())
```

#### `AgentRunResult` — the non-streaming result

Returned by `agent.run()` / `agent.run_sync()`. `usage` is a **property**, not a method call.

```python
Fields/methods: output, usage, run_id, conversation_id, timestamp, response,
                 all_messages(output_tool_return_content=None), new_messages(),
                 all_messages_json(), new_messages_json(), metadata
```

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o', output_type=str)
result = agent.run_sync('Explain Python in one sentence.')
print(result.output, result.usage.total_tokens, result.run_id)

# Continue a conversation, injecting a custom "what we did with the output" note
history = result.all_messages(output_tool_return_content='Accepted; proceeding.')
follow_up = agent.run_sync('Now add a caveat.', message_history=history)
```

#### `UserPromptNode` + `ModelRequestNode` + `CallToolsNode`

The three public graph nodes an `AgentRun` walks through, promoted to top-level `pydantic_ai`
exports. `ModelRequestNode.last_request_context` exposes the actual `model`/`messages`/
`model_settings`/`model_request_parameters` sent. `CallToolsNode.stream(ctx)` yields
`HandleResponseEvent` items (tool call/result events) for that step.

```python
from pydantic_ai import Agent, UserPromptNode, ModelRequestNode, CallToolsNode

agent = Agent('openai:gpt-4o')

async def trace():
    async with agent.iter('Count to 3.') as run:
        async for node in run:
            if isinstance(node, ModelRequestNode):
                print('sending', [type(p).__name__ for p in node.request.parts])
            elif isinstance(node, CallToolsNode):
                print('response finish_reason', node.model_response.finish_reason)
```

#### `WrapperAgent` + `AbstractAgent`

`AbstractAgent` is the ABC that `Agent`, `WrapperAgent`, and custom agent implementations satisfy (model, name, description, deps_type, output_type, event_stream_handler, toolsets, `run`/`run_sync`/`run_stream`). `WrapperAgent` delegates every property and method to `self.wrapped`, leaving no abstract methods unimplemented — subclass it and override only what you need (auth middleware, rate limiting, routing between specialised agents, timing). This is the base every durable-execution wrapper (`TemporalAgent`, `DBOSAgent`, `PrefectAgent`) builds on.

```python
class WrapperAgent(AbstractAgent[AgentDepsT, OutputDataT]):
    def __init__(self, wrapped: AbstractAgent[AgentDepsT, OutputDataT]): ...
    # model, name, description, deps_type, output_type, event_stream_handler,
    # root_capability, toolsets, run, run_sync, run_stream all delegate to self.wrapped
```

```python
from pydantic_ai import Agent
from pydantic_ai.agent import WrapperAgent

class RateLimitedAgent(WrapperAgent):
    def __init__(self, inner: Agent, max_per_minute: int = 10):
        super().__init__(inner)
        self._max_per_minute = max_per_minute
        self._run_count = 0

    async def run(self, prompt, **kwargs):
        if self._run_count >= self._max_per_minute:
            raise RuntimeError('Rate limit exceeded')
        self._run_count += 1
        return await self.wrapped.run(prompt, **kwargs)

rate_limited = RateLimitedAgent(Agent('openai:gpt-4o-mini'), max_per_minute=3)
```

#### Direct API — `model_request`, `model_request_stream`, `StreamedResponseSync`

`pydantic_ai.direct` sends messages to a model **without** an `Agent` — no dependency injection, no tool dispatch, no retries; just the raw model interface with OTel instrumentation wired in. `model_request_sync`/`model_request_stream_sync` are thread-bridged sync wrappers for scripts and notebooks; `model_request_stream_sync` returns a `StreamedResponseSync` that bridges an async model stream via a background thread and a `queue.Queue`, for CLI tools and notebooks that can't `await`.

```python
async def model_request(model, messages, *, model_settings=None, model_request_parameters=None, instrument=None) -> ModelResponse: ...
async def model_request_stream(model, messages, **kwargs) -> AbstractAsyncContextManager[StreamedResponse]: ...
def model_request_sync(model, messages, **kwargs) -> ModelResponse: ...
def model_request_stream_sync(model, messages, **kwargs) -> StreamedResponseSync: ...
```

```python
import asyncio
from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request

async def main() -> None:
    response = await model_request(
        'openai:gpt-4o-mini',
        [ModelRequest.user_text_prompt('What is the capital of France?')],
    )
    print(response.parts[0].content)

asyncio.run(main())
```

```python
from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request_stream_sync

with model_request_stream_sync('anthropic:claude-haiku-4-5', [ModelRequest.user_text_prompt('Hi')]) as stream:
    for event in stream:
        print(event)
    print(stream.model_name)
```

#### `PendingMessage` + `RunContext.enqueue` + `PendingMessageDrainCapability`

`PendingMessage` is the object created by `ctx.enqueue(...)` / `agent_run.enqueue(...)`, holding one or more `ModelMessage`s and a `priority`. `'asap'` is delivered before the next model request (or redirects termination into one more request); `'when_idle'` is delivered only when the agent would otherwise finish. `PendingMessage.from_content()` coalesces adjacent user content into one `ModelRequest` and returns `None` for an empty call. The auto-injected `PendingMessageDrainCapability` sits at `position='outermost'` and does the actual draining — `before_model_request` drains `'asap'`, `after_node_run` redirects idle termination to drain `'when_idle'`.

```python
@dataclass
class PendingMessage:
    messages: list[ModelMessage]        # always ends in a ModelRequest
    priority: PendingMessagePriority = 'asap'   # 'asap' | 'when_idle'

    @classmethod
    def from_content(cls, *content: EnqueueContent, priority='asap') -> PendingMessage | None: ...
```

```python
import asyncio
from pydantic_ai import Agent, RunContext

agent = Agent('openai:gpt-4o-mini')

@agent.tool
async def fetch_and_maybe_followup(ctx: RunContext[None], query: str) -> str:
    # enqueue() is synchronous — do not await it
    ctx.enqueue('Also double-check that against the latest data.', priority='asap')
    return f'data for {query}'

asyncio.run(agent.run('Search for AI news'))
```

Note: a bare `async for node in agent.iter(...)` loop that ends while `'when_idle'`-priority messages are still undrained raises `UndrainedPendingMessagesError` — use `agent.run()` or `AgentRun.next()` instead, both of which drain every priority.

#### `RunContext` — complete field reference

The generic context object injected as the first argument of every tool, output validator,
system-prompt function, and capability hook.

```python
class RunContext(Generic[AgentDepsT]):
    deps: AgentDepsT
    model: Model
    usage: RunUsage
    usage_limits: UsageLimits | None      # always a real UsageLimits() even when the caller passes None (since 2.10)
    agent: Agent | None
    prompt: str | Sequence[UserContent] | None
    messages: list[ModelMessage]
    tracer: Tracer
    trace_include_content: bool
    retries: dict[str, int]
    tool_call_id: str | None
    tool_name: str | None
    retry: int
    max_retries: int
    run_step: int
    tool_call_approved: bool
    tool_call_metadata: Any
    partial_output: bool
    run_id: str | None
    conversation_id: str | None
    metadata: dict[str, Any] | None
    model_settings: ModelSettings | None
    validation_context: Any
    pending_messages: list[PendingMessage]
    tool_manager: ToolManager | None
    root_capability: AbstractCapability | None
    capabilities: dict[str, AbstractCapability]
    loaded_capability_ids: set[str]
    discovered_tool_names: set[str]

    @property
    def last_attempt(self) -> bool: ...     # True when retry == max_retries
    def enqueue(self, *content: EnqueueContent, priority: 'asap' | 'when_idle' = 'asap') -> None: ...
    def is_tool_available(self, tool: str | ToolDefinition) -> bool: ...   # 2.22+
```

`is_tool_available()` answers whether a function tool is currently visible to the model,
accounting for `FilteredToolset`/`PrepareTools`/`DeferredLoadingToolset` mutations.

```python
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext, ModelRetry

@dataclass
class Deps:
    api_url: str

agent: Agent[Deps, str] = Agent('openai:gpt-4o', deps_type=Deps)

@agent.tool
async def fetch_data(ctx: RunContext[Deps], query: str) -> str:
    if ctx.last_attempt:
        return f'ERROR: exhausted retries for {query!r}'
    try:
        return f'data for {query} from {ctx.deps.api_url}'
    except Exception as e:
        raise ModelRetry(f'fetch failed: {e}, attempt {ctx.retry + 1}')
```

#### `AgentSpec` — YAML/JSON-driven configuration

**Module:** `pydantic_ai._spec` (exported as `pydantic_ai.AgentSpec`). A `BaseModel` describing an
agent's full config — `model`, `name`, `description`, `instructions` (string or `TemplateStr`),
`deps_schema`, `output_schema`, `model_settings`, `retries`, `end_strategy`, `tool_timeout`,
`metadata`, `capabilities: list[CapabilitySpec]`. **Correction:** `AgentSpec` has no `to_agent()`
method — build the agent via `Agent.from_spec(spec)` or `Agent.from_file(path)`.
`NamedSpec`/`CapabilitySpec` support three short forms (bare string, `{Name: single_arg}`,
`{Name: {kwargs}}`) resolved via `build_registry()`/`load_from_registry()`.

```python
Agent.from_file(path)                      # one-step: parse YAML/JSON, build Agent
Agent.from_spec(spec_or_dict)
AgentSpec.from_file(path, fmt=None)
AgentSpec.from_dict(data)
AgentSpec.from_text(text, fmt=None)
spec.to_file(path, fmt=None, schema_path=...)
```

```python
from pydantic_ai import Agent, AgentSpec

spec = AgentSpec.from_dict({
    'model': 'openai:gpt-4o',
    'name': 'support-agent',
    'instructions': 'You are a helpful support agent.',
    'model_settings': {'temperature': 0.3},
    'retries': 3,
    'capabilities': [{'WebSearch': {'search_context_size': 'high'}}, 'Thinking'],
})
agent = Agent.from_spec(spec)
result = agent.run_sync('How do I reset my password?')
```

#### `EndStrategy` + `AgentRetries`

`EndStrategy = Literal['early', 'graceful', 'exhaustive']`, passed as `Agent(end_strategy=...)` or
`agent.run(end_strategy=...)`: `'early'` (default) stops the moment a final result is available
even with tools still in flight; `'graceful'` finishes tool calls already dispatched first;
`'exhaustive'` runs every requested tool call regardless. `AgentRetries = int | None` sets how
many `ModelRetry` cycles a tool/output gets before the run raises.

```python
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o', retries=3)
result = agent.run_sync('Fetch three URLs and summarise.', end_strategy='graceful')
```

#### `capture_run_messages`

Context manager that captures the message history built up to the point of failure, even when
`agent.run()` raises. Only the *first* `run()`/`run_sync()`/`run_stream()` call inside one
`with` block is captured — nest separate blocks for separate calls.

```python
from pydantic_ai import Agent, capture_run_messages, UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded

agent = Agent('openai:gpt-4o')

with capture_run_messages() as messages:
    try:
        agent.run_sync('Count to 1000.', usage_limits=UsageLimits(request_limit=1))
    except UsageLimitExceeded:
        print(f'Captured {len(messages)} messages before the limit hit')
```

#### `ToolManager` + `ValidatedToolCall` + `ParallelExecutionMode`

**Module:** `pydantic_ai.tool_manager`. The internal engine that resolves, validates, and executes
every tool call in a step. `ParallelExecutionMode` (`'parallel'` | `'sequential'` | `'parallel_ordered_events'`) controls concurrency and event ordering; `ToolManager.parallel_execution_mode(mode)` is a classmethod context manager, and `Agent.parallel_tool_call_execution_mode(mode)` is a thin static wrapper around it. A tool's own `ToolDefinition.sequential=True` always wins regardless of the mode. `ValidatedToolCall` separates schema validation from execution so hooks can inspect `args_valid` before a tool actually runs.

```python
ParallelExecutionMode = Literal['parallel', 'sequential', 'parallel_ordered_events']

@dataclass
class ValidatedToolCall(Generic[AgentDepsT]):
    call: ToolCallPart
    tool: ToolsetTool[AgentDepsT] | None
    args_valid: bool
    validated_args: dict[str, Any] | None = None
```

```python
from pydantic_ai import Agent
from pydantic_ai.tool_manager import ToolManager

agent = Agent('openai:gpt-4o-mini')

@agent.tool_plain
def step_one() -> str: return 'done'

with ToolManager.parallel_execution_mode('sequential'):
    result = agent.run_sync('Run step_one twice.')
```

#### `SkipModelRequest` + `SkipToolExecution` + `SkipToolValidation`

Three plain `Exception` subclasses for short-circuiting hook execution. Raise `SkipModelRequest(response)` inside `before_model_request`/`wrap_model_request` to substitute a synthetic `ModelResponse` (response caching, test injection, circuit breakers). Raise `SkipToolExecution(result)` inside `before_tool_execute`/`wrap_tool_execute` to skip the tool body and return `result` to the model directly (dry-run mode, sandboxing). Raise `SkipToolValidation(validated_args)` inside a `before_tool_validate` hook to bypass Pydantic argument validation with pre-coerced args.

```python
class SkipModelRequest(Exception):
    def __init__(self, response: ModelResponse): ...
class SkipToolExecution(Exception):
    def __init__(self, result: Any): ...
class SkipToolValidation(Exception):
    def __init__(self, validated_args: dict[str, Any]): ...
```

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import Hooks
from pydantic_ai.exceptions import SkipModelRequest
from pydantic_ai.messages import ModelResponse, TextPart

_cache: dict[str, ModelResponse] = {}
hooks = Hooks()

@hooks.on.before_model_request
async def return_if_cached(ctx, request_context):
    key = str(request_context.messages)
    if key in _cache:
        raise SkipModelRequest(_cache[key])
    return request_context

agent = Agent('openai:gpt-4o', capabilities=[hooks])
```

#### `SelectModel` + `ModelSelectionContext` / `ResolveModelId`

`SelectModel(selector)` invokes a `ModelSelector` callable before every logical model-request step; `selector` receives a frozen `ModelSelectionContext` (`deps`, `model`, `run_step`, `messages`, `usage`) and returns a `Model` instance or a provider-prefixed model-name string — the primary mechanism for per-step routing (cost ladders, capability escalation). `ResolveModelId(resolver)` is the lower-level hook: `resolver(ModelResolutionContext, model_id: str) -> Model | None`, called only when a string model ID isn't already in a registry; returning `None` falls through to the next resolver or `infer_model`. Both accept sync or async callables.

```python
@dataclass
class SelectModel(AbstractCapability[AgentDepsT]):
    selector: ModelSelector[AgentDepsT]

@dataclass(frozen=True)
class ModelSelectionContext(Generic[DepsT]):
    deps: DepsT
    model: Model
    run_step: int
    messages: list[ModelMessage]
    usage: RunUsage
```

```python
from pydantic_ai import Agent, ModelSelectionContext
from pydantic_ai.capabilities import SelectModel

def escalating_selector(ctx: ModelSelectionContext[None]) -> str:
    return 'anthropic:claude-haiku-4-5' if ctx.run_step == 1 else 'anthropic:claude-opus-4-6'

agent = Agent(capabilities=[SelectModel(escalating_selector)])
```

#### `CancellationToken`

A thread-safe handle for first-party run cancellation, added 2.26.0. Call `.cancel()` from any thread; the agent translates the resulting cancellation into `RunCancelled` at the `agent.iter()` boundary, and `RunCancelled.all_messages()` preserves partial history for resuming later.

```python
class CancellationToken:
    def __init__(self) -> None: ...
    def cancel(self) -> None: ...
```

```python
import asyncio, threading
from pydantic_ai import Agent, CancellationToken
from pydantic_ai.exceptions import RunCancelled

agent = Agent('openai:gpt-5')

async def main():
    token = CancellationToken()
    threading.Thread(target=lambda: (__import__('time').sleep(2), token.cancel()), daemon=True).start()
    try:
        result = await agent.run('Write a long report.', cancellation_token=token)
    except RunCancelled as exc:
        print('Cancelled, partial messages:', len(exc.all_messages()))
```

#### `MessagesBuilder` + `BuilderCheckpoint`

`MessagesBuilder` (`pydantic_ai.ui._messages_builder`) constructs `ModelRequest`/`ModelResponse` sequences incrementally from individual message parts: `add()` extends the last message if the new part matches its type, otherwise appends a fresh message. `BuilderCheckpoint` is an opaque snapshot used with `last_modified(checkpoint, of_type=...)` to find which message was created/extended since the snapshot — the plumbing behind `UIEventStream` and custom streaming adapters.

```python
class MessagesBuilder:
    def add(self, part: ModelRequestPart | ModelResponsePart) -> None: ...
    def checkpoint(self) -> BuilderCheckpoint: ...
    def last_modified(self, checkpoint: BuilderCheckpoint, *, of_type: type) -> ModelMessage | None: ...
    messages: list[ModelMessage]
```

```python
from pydantic_ai.ui._messages_builder import MessagesBuilder
from pydantic_ai.messages import UserPromptPart, TextPart

builder = MessagesBuilder()
builder.add(UserPromptPart(content='What is the capital of France?'))
builder.add(TextPart(content='Paris.'))
print(len(builder.messages))   # 2: one ModelRequest, one ModelResponse
```

#### `SystemPromptRunner`

The internal wrapper pydantic-ai stores for each `@agent.system_prompt`-registered function. It inspects the function signature at construction (`_takes_ctx`, `_is_async`) to dispatch correctly: sync/no-arg via a thread executor, `RunContext`-aware via the same executor with context passed, async called directly. `dynamic=True` forces re-evaluation before every model request rather than once at run start.

```python
@dataclass
class SystemPromptRunner(Generic[AgentDepsT]):
    function: SystemPromptFunc[AgentDepsT]
    dynamic: bool = False
```

```python
from pydantic_ai import Agent, RunContext

agent = Agent('openai:gpt-4o', deps_type=dict)

@agent.system_prompt(dynamic=True)
def budget_aware_prompt(ctx: RunContext[dict]) -> str:
    if ctx.deps.get('budget_remaining', 1.0) < 0.10:
        return 'Low-budget mode: keep answers under 50 words.'
    return 'You are a helpful, detailed assistant.'
```

#### `ModelSettings` — cross-provider reference

`ModelSettings` is a `TypedDict` (all keys optional) providing portable model configuration; unsupported keys are silently ignored by a given provider's adapter. Notable fields: `temperature`, `top_p`, `max_tokens`, `seed`, `parallel_tool_calls`, `thinking` (`ThinkingLevel`), `tool_choice`, `service_tier`, `stop_sequences`, `extra_headers`, `extra_body`. Use `merge_model_settings(base, overrides)` to layer agent-level settings with per-run overrides (override wins per key).

```python
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.settings import merge_model_settings

base: ModelSettings = {'temperature': 0.0, 'seed': 42, 'parallel_tool_calls': False}
agent = Agent('openai:gpt-4o', model_settings=base)
merged = merge_model_settings(base, {'temperature': 0.9})   # per-run override wins
```

#### `TemplateStr` — Handlebars instructions

`TemplateStr[AgentDepsT]` compiles a Handlebars template against the agent's `deps_type` (via pydantic-handlebars) and re-renders per model request against the live `RunContext.deps`. Any string containing `{{` inside a field typed `TemplateStr[...]` is auto-compiled during Pydantic validation — this is how `AgentSpec` YAML instructions get template rendering without extra ceremony. `.render(deps)` renders standalone, outside an `Agent`.

```python
class TemplateStr(Generic[AgentDepsT]):
    def __init__(self, template: str, *, deps_type: type[AgentDepsT] | None = None) -> None: ...
    def render(self, deps: AgentDepsT) -> str: ...
```

```python
from dataclasses import dataclass
from pydantic_ai import Agent, TemplateStr

@dataclass
class UserProfile:
    name: str
    language: str

agent = Agent(
    'openai:gpt-4o-mini',
    deps_type=UserProfile,
    instructions=TemplateStr('Hello {{name}}! Always respond in {{language}}.'),
)
result = agent.run_sync('What is a decorator?', deps=UserProfile(name='Alice', language='French'))
```

---

### Models & Providers

#### `ModelProfile` + `DEFAULT_PROFILE` + `ModelProfileSpec` + `merge_profile`

As of pydantic-ai 2.23+, `ModelProfile` is a **`TypedDict`** (`total=False`), not a `@dataclass`. There is no `.update()` instance method any more — the canonical way to layer profiles is the module-level `merge_profile()` function. `ModelProfileSpec` is `ModelProfile | Callable[[ModelProfile], ModelProfile]` (the callable receives the provider's already-resolved default profile and returns the final one).

```python
class ModelProfile(TypedDict, total=False):
    supports_tools: bool                              # default True
    supports_tool_return_schema: bool                 # default False
    supports_json_schema_output: bool                 # default False
    supports_json_object_output: bool                 # default False
    supports_image_output: bool                       # default False
    supports_audio_input: bool                        # default False
    supports_inline_system_prompts: bool               # default False
    default_structured_output_mode: StructuredOutputMode  # default 'tool'
    prompted_output_template: str
    native_output_requires_schema_in_instructions: bool
    json_schema_transformer: type[JsonSchemaTransformer] | None
    supports_thinking: bool
    thinking_always_enabled: bool
    thinking_tags: tuple[str, str]                    # default ('<think>', '</think>')
    ignore_streamed_leading_whitespace: bool
    supported_native_tools: frozenset[type[AbstractNativeTool]]
    tool_deferral_mode: Literal['standalone', 'with_tool_search'] | None
    tool_addition_mode: Literal['by_reference', 'with_definitions'] | None
    # tool_additions / deferred_tools_require_tool_search: deprecated aliases, still accepted

ModelProfileSpec = ModelProfile | Callable[[ModelProfile], ModelProfile]

def merge_profile(base: ModelProfile | None, *overrides: ModelProfile | None) -> ModelProfile: ...
```

```python
from pydantic_ai.profiles import ModelProfile, merge_profile, DEFAULT_PROFILE
from pydantic_ai.profiles.harmony import harmony_model_profile

my_override: ModelProfile = {'supports_json_schema_output': True, 'supports_thinking': True}
merged = merge_profile(DEFAULT_PROFILE, my_override)   # dict-spread merge, replaces the old .update()
print(merged['supports_json_schema_output'])   # True

merged2 = merge_profile(DEFAULT_PROFILE, harmony_model_profile('gpt-4o'))
print(merged2['ignore_streamed_leading_whitespace'])   # True

# ModelProfileSpec as a callable: receives the resolved default, returns the final profile
def my_profile_factory(default: ModelProfile) -> ModelProfile:
    return merge_profile(default, {'supports_thinking': True})
```

`supported_builtin_tools` (the pre-1.104 field name) is no longer available even as a deprecated
alias — code still reading `profile.supported_builtin_tools` must migrate to
`profile['supported_native_tools']`. `harmony_model_profile`, `moonshotai_model_profile`, and
`amazon_model_profile` are additional provider-profile functions layered the same way as any other.

#### Per-provider `ModelProfile` families — Anthropic / OpenAI / Grok / Google

Each provider module exposes a `TypedDict` subclass of `ModelProfile` with `<provider>_`-prefixed fields plus a `<provider>_model_profile(model_name)` function, all mergeable via plain dict-spread or `merge_profile`:

- **`AnthropicModelProfile`** (`profiles.anthropic`): `anthropic_supports_fast_speed`, `anthropic_supports_adaptive_thinking`, `anthropic_supports_effort`, `anthropic_supports_xhigh_effort`, `anthropic_disallows_budget_thinking`, `anthropic_disallows_sampling_settings`, `anthropic_supports_forced_tool_choice`, `anthropic_supports_task_budgets`, `anthropic_default_code_execution_tool_version` / `anthropic_supported_code_execution_tool_versions` (`Literal['20250825','20260120']`). `resolve_anthropic_effort(level, *, supports_xhigh) -> AnthropicEffort` maps the unified thinking level to Anthropic's API string.
- **`OpenAIModelProfile`** (`profiles.openai`): `openai_chat_thinking_field`, `openai_chat_send_back_thinking_parts` (`'auto'|'tags'|'field'|False`), `openai_supports_strict_tool_definition`, `openai_supports_tool_choice_required`. `OPENAI_REASONING_EFFORT_MAP` maps unified `ThinkingLevel` → `reasoning_effort` string; `SAMPLING_PARAMS` lists params dropped during reasoning mode.
- **`GrokModelProfile`** (`profiles.grok`): `grok_supports_builtin_tools`, `grok_supports_tool_choice_required`, `grok_reasoning_efforts: frozenset[GrokReasoningEffort]` (`'none'|'low'|'medium'|'high'`) — Grok 4.3 gets the full set, Grok 3 Mini gets `{low, high}` only (so `thinking_always_enabled=True`).
- **`GoogleModelProfile`** (`profiles.google`): `google_supports_tool_combination`, `google_supports_server_side_tool_invocations`, `google_supports_thinking_level` (Gemini 3+ uses `thinking_level` enum instead of `thinking_budget` int). `GoogleJsonSchemaTransformer` strips `$schema`/`discriminator`/`examples`/`title` and rewrites `const` → `enum`.

```python
from pydantic_ai.profiles.anthropic import anthropic_model_profile, resolve_anthropic_effort
from pydantic_ai.profiles.google import GoogleJsonSchemaTransformer
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider

profile = anthropic_model_profile('claude-opus-4-8')
print(resolve_anthropic_effort('xhigh', supports_xhigh=profile.get('anthropic_supports_xhigh_effort', False)))

schema = GoogleJsonSchemaTransformer({'const': 'active'}).walk()
print(schema)   # {'enum': ['active'], 'type': 'string'}

class MyVLLMProvider(OpenAIProvider):
    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None:
        if 'qwen3' in model_name.lower():
            return OpenAIModelProfile(
                supports_thinking=True,
                openai_chat_thinking_field='reasoning',
                openai_chat_send_back_thinking_parts='field',
            )
        return None
```

#### `Provider` ABC + `infer_provider` / `infer_provider_class`

**Module:** `pydantic_ai.providers`. Every first-party provider implements this ABC: `name`,
`base_url`, `client`, and `model_profile(model_name) -> ModelProfile | None` (returning `None`
means "use the built-in default for this model family"). `infer_provider('openai')` /
`infer_provider_class('openai')` resolve a provider string to an instance or class.

```python
from pydantic_ai.providers import infer_provider

openai_provider = infer_provider('openai')
gateway_anthropic = infer_provider('gateway/anthropic')
```

#### `FallbackModel` (+ `ResponseRejected`, `FallbackExceptionGroup`)

Wraps two or more models and tries them in order until one succeeds (or on a custom response predicate). `fallback_on` accepts exception types, exception handlers, **response handlers**, or a sequence mixing all three — auto-detected by inspecting whether the callable's first parameter is type-hinted `ModelResponse`. `ResponseRejected` is raised inside `FallbackExceptionGroup` when a response handler rejects every model's output. `model_name` reports `fallback:<model1>,<model2>,...`.

```python
class ResponseHandler: Callable[[ModelResponse], bool | Awaitable[bool]]
class ExceptionHandler: Callable[[Exception], bool | Awaitable[bool]]
FallbackOn = type[Exception] | tuple[type[Exception], ...] | ExceptionHandler | ResponseHandler | Sequence[...]

class FallbackModel(Model):
    def __init__(self, default_model, *fallback_models, fallback_on: FallbackOn = (ModelAPIError,)): ...
```

```python
from pydantic_ai import Agent
from pydantic_ai.models.fallback import FallbackModel, FallbackExceptionGroup
from pydantic_ai.messages import ModelResponse

def response_too_short(response: ModelResponse) -> bool:
    text = response.text or ''
    return len(text.strip()) < 10

model = FallbackModel('openai:gpt-4o-mini', 'openai:gpt-4o', 'anthropic:claude-opus-4-8', fallback_on=response_too_short)
agent = Agent(model)

try:
    result = agent.run_sync('Hello')
except* FallbackExceptionGroup as eg:
    for exc in eg.exceptions:
        print(type(exc).__name__, exc)
```

#### `WrapperModel` + `CompletedStreamedResponse`

`WrapperModel` is the base class for models that wrap another model — delegating every `Model` method to `self.wrapped` via `__getattr__` for anything not explicitly overridden. `CompletedStreamedResponse` presents an already-consumed `ModelResponse` as a `StreamedResponse`, used by Temporal/Prefect/DBOS wrappers that ran the real model inside an activity/task and must replay the result at the workflow layer. **Import note:** `CompletedStreamedResponse` moved from `pydantic_ai.models.wrapper` to `pydantic_ai.models` — the old path still works but emits `PydanticAIDeprecationWarning`.

```python
class WrapperModel(Model):
    def __init__(self, wrapped: Model | KnownModelName): ...
    # request, request_stream, count_tokens, prepare_messages, customize_request_parameters
    # all delegate to self.wrapped unless overridden

from pydantic_ai.models import CompletedStreamedResponse   # current import path
```

```python
from pydantic_ai import Agent
from pydantic_ai.models.wrapper import WrapperModel

class LoggingModel(WrapperModel):
    async def request(self, messages, model_settings, model_request_parameters):
        print(f"→ {len(messages)} messages")
        response = await super().request(messages, model_settings, model_request_parameters)
        print(f"← {response.parts}")
        return response

agent = Agent(LoggingModel("openai:gpt-4o-mini"))
```

#### `MistralModel` + `MistralModelSettings`

**Module:** `pydantic_ai.models.mistral`. Talks to Mistral's API (`mistral-large-latest`,
`pixtral-large-latest` for vision, `mistral-embed`). `json_mode_schema_prompt` templates the
schema-in-instructions fallback for structured output.

```python
MistralModel(model_name: MistralModelName, *, provider='mistral', profile=None,
             json_mode_schema_prompt: str = "...", settings=None)
```

```python
from pydantic_ai import Agent
from pydantic_ai.models.mistral import MistralModel

agent = Agent(MistralModel('mistral-large-latest'))
result = agent.run_sync('Explain RAG in three sentences.')
```

#### `OllamaModel` — self-hosted vs. Cloud `NativeOutput`

Extends `OpenAIChatModel` for Ollama's OpenAI-compatible endpoint. Self-hosted Ollama enforces
`response_format` grammar-style, so `NativeOutput` is schema-safe; Ollama Cloud (`base_url`
containing `ollama.com`, or model name ending `-cloud`) accepts the same request but does **not**
enforce the schema, so PydanticAI auto-disables `supports_json_schema_output` for detected Cloud
models — `NativeOutput` then raises `UserError` there. Use `ToolOutput`/`PromptedOutput` instead,
or manually override the profile once Ollama Cloud fixes enforcement upstream.

```python
from pydantic_ai import Agent
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.output import NativeOutput
from pydantic import BaseModel

class Recipe(BaseModel):
    name: str
    steps: list[str]

agent = Agent(OllamaModel('qwen3'), output_type=NativeOutput(Recipe))  # self-hosted: schema-enforced
```

#### `OpenRouterModel` + `OpenRouterModelSettings` (+ `OpenRouterReasoning`, `OpenRouterProviderConfig`, `OpenRouterUsageConfig`)

`OpenRouterModel` extends `OpenAIChatModel` with OpenRouter-specific metadata, routing to 200+ models via `provider/model` names. `OpenRouterModelSettings` adds five fields: `openrouter_models` (provider-side fallback chain), `openrouter_provider` (routing constraints — `order`, `allow_fallbacks`, `require_parameters`, `data_collection`, `only`, `quantizations`), `openrouter_reasoning` (`effort` OR `max_tokens`, `exclude`, `enabled`), `openrouter_transforms`, `openrouter_usage`, `openrouter_cache_ttl` (`'5m'|'1h'`).

```python
class OpenRouterModel(OpenAIChatModel):
    def __init__(self, model_name: str, *, provider='openrouter', profile=None, settings=None): ...
class OpenRouterModelSettings(ModelSettings, total=False):
    openrouter_models: list[str]
    openrouter_provider: OpenRouterProviderConfig
    openrouter_reasoning: OpenRouterReasoning
    openrouter_cache_ttl: Literal['5m', '1h']
```

```python
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings

agent = Agent(
    OpenRouterModel('anthropic/claude-sonnet-4-6'),
    model_settings=OpenRouterModelSettings(
        openrouter_models=['anthropic/claude-sonnet-4-6', 'openai/gpt-5.2'],
    ),
)
```

#### `HuggingFaceModel` + `HuggingFaceModelSettings` + `HuggingFaceStreamedResponse`

**Extra:** `pip install "pydantic-ai-slim[huggingface]"`. Inference against any HF Hub model
(DeepSeek-R1, Llama-4, Qwen3, QwQ). Thinking models need `profile=ModelProfile(supports_thinking=True,
thinking_always_enabled=True, thinking_tags=('<think>', '</think>'))` set explicitly.

```python
HuggingFaceModel(model_name: str, *, provider='huggingface', profile=None, settings=None)
```

```python
from pydantic_ai import Agent
from pydantic_ai.models.huggingface import HuggingFaceModel

agent = Agent(HuggingFaceModel('meta-llama/Llama-4-Scout-17B-16E-Instruct'))
result = agent.run_sync('Explain transformers in 2 sentences.')
```

#### `BedrockConverseModel` + `BedrockModelSettings` + `BedrockProvider`

The only first-party AWS model, adapting boto3's synchronous `converse`/`converse_stream` to the
async `Model` interface. Every `BedrockModelSettings` field carries a `bedrock_` prefix.
`BedrockProvider` supports three auth paths: bring-your-own boto3 client, bearer-token
(`AWS_BEARER_TOKEN_BEDROCK`), or standard AWS credentials; `provider.client = new_client` hot-swaps
for credential rotation without recreating the model.

```python
BedrockConverseModel(model_name, *, provider='bedrock', profile=None, settings=None)

class BedrockModelSettings(ModelSettings, total=False):
    bedrock_cache_tool_definitions: bool | Literal['5m', '1h']
    bedrock_cache_instructions: bool | Literal['5m', '1h']
    bedrock_cache_messages: bool | Literal['5m', '1h']
    bedrock_guardrail_config: dict
    bedrock_performance_configuration: dict
    bedrock_request_metadata: dict[str, str]
```

```python
from pydantic_ai import Agent
from pydantic_ai.models.bedrock import BedrockConverseModel, BedrockModelSettings
from pydantic_ai.providers.bedrock import BedrockProvider

provider = BedrockProvider(region_name='us-east-1')
agent = Agent(
    BedrockConverseModel('us.anthropic.claude-sonnet-4-6', provider=provider),
    model_settings=BedrockModelSettings(bedrock_cache_instructions=True, bedrock_cache_messages=True),
)
```

`bedrock_*_model_profile` factory functions (anthropic/amazon/deepseek/mistral/qwen/google/minimax/
nvidia) map each vendor's Bedrock model IDs to a `BedrockModelProfile`; `_without_builtin_tools`
strips native tools from any profile since Bedrock's Converse API has none.

#### `GoogleProvider` + `GoogleCloudProvider` + `GoogleModel` + `GoogleModelSettings`

`GoogleProvider` targets the Gemini API (`GOOGLE_API_KEY`); `GoogleCloudProvider` targets Vertex AI
(Application Default Credentials, or Express Mode with `api_key=`). Both extend
`BaseGoogleProvider[Client]` from `google-genai`.

```python
GoogleProvider(*, api_key=None, client=None, http_client=None, base_url=None)
GoogleCloudProvider(*, api_key=None, credentials=None, project=None, location=None, client=None)

class GoogleModelSettings(ModelSettings, total=False):
    google_safety_settings: list[SafetySettingDict]
    google_thinking_config: ThinkingConfigDict
    google_cached_content: str      # NOTE: strips system_instruction/tools/tool_config from the request
    google_cloud_service_tier: GoogleCloudServiceTier
```

```python
from pydantic_ai import Agent
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.models.google import GoogleModel

agent = Agent(GoogleModel('gemini-2.5-pro', provider=GoogleProvider(api_key='AIza...')))
result = agent.run_sync('Explain federated learning.')
```

**Removed:** `GoogleGLAProvider` and `GoogleVertexProvider` (and the `GeminiModel` they paired
with) are gone. Migrate `GoogleGLAProvider(api_key=...)` → `GoogleProvider(api_key=...)` (env var
`GEMINI_API_KEY` → `GOOGLE_API_KEY`), and `GoogleVertexProvider(project_id=..., region=...)` →
`GoogleCloudProvider(project=..., location=...)`.

#### `CohereModel` + `CohereProvider` + `CohereModelSettings`

Drives Cohere's v2 chat API via `cohere.AsyncClientV2`. `CO_API_KEY` / `CO_BASE_URL` env vars.

```python
CohereModel(model_name: CohereModelName, *, provider='cohere', profile=None, settings=None)
CohereProvider(*, api_key=None, cohere_client=None, http_client=None)
```

```python
from pydantic_ai import Agent
from pydantic_ai.models.cohere import CohereModel
from pydantic_ai.providers.cohere import CohereProvider

agent = Agent(CohereModel('command-r-plus-08-2024', provider=CohereProvider(api_key='...')))
```

#### `XaiProvider` + `XaiModel`

The only gRPC-transport model (`xai-sdk`, not HTTP). `_LazyAsyncClient` defers channel creation
per-event-loop to avoid the classic "gRPC channel bound to the wrong asyncio loop" `RuntimeError`.
`GrokModelProfile` adds `grok_supports_builtin_tools` / `grok_supports_tool_choice_required` /
`grok_reasoning_efforts: frozenset[GrokReasoningEffort]` (`Literal['none','low','medium','high']`).

```python
XaiProvider(*, api_key=None, api_host=None, timeout=None, xai_client=None)
```

```python
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.providers.xai import XaiProvider
from pydantic_ai.models.xai import XaiModel

agent = Agent(
    XaiModel('grok-4.3', provider=XaiProvider(api_key='xai-...')),
    model_settings=ModelSettings(thinking='high'),
)
```

#### `ZaiModel` + `ZaiModelSettings` + `ZaiProvider` + `ZaiModelProfile`

Z.AI (Zhipu AI) GLM family support (`pydantic_ai.models.zai`, `pydantic_ai.providers.zai`). Z.AI sends thinking as a separate `reasoning_content` field rather than inline text; by default prior-turn reasoning is preserved across turns (`zai_clear_thinking=False`, matching Z.AI's "preserved thinking" contract) — set `True` to discard it. GLM-5.2 additionally supports per-request `reasoning_effort` when `ZaiModelProfile.zai_supports_reasoning_effort=True`.

```python
class ZaiModelSettings(ModelSettings, total=False):
    zai_clear_thinking: bool
class ZaiModelProfile(ModelProfile, total=False):
    zai_supports_reasoning_effort: bool
class ZaiModel(OpenAIChatModel):
    def __init__(self, model_name, *, provider='zai', profile=None, settings: ZaiModelSettings | None = None): ...
```

```python
from pydantic_ai import Agent
from pydantic_ai.models.zai import ZaiModel

agent = Agent(ZaiModel('glm-5', settings={'thinking': True}))
result = agent.run_sync('Explain why 0.1 + 0.2 != 0.3 in IEEE 754.')
```

#### `VercelProvider` — Vercel AI Gateway

Routes through `https://ai-gateway.vercel.sh/v1`, proxying 8+ upstream providers under one auth surface (`VERCEL_AI_GATEWAY_API_KEY` or `VERCEL_OIDC_TOKEN`). Model naming is `provider/model` (e.g. `anthropic/claude-opus-4-8`); `VercelProvider.model_profile()` dispatches to the matching upstream profile function, merged over `OpenAIModelProfile(json_schema_transformer=OpenAIJsonSchemaTransformer)`.

```python
class VercelProvider(Provider[AsyncOpenAI]):
    base_url = 'https://ai-gateway.vercel.sh/v1'
    def __init__(self, *, api_key=None, openai_client=None, http_client=None) -> None: ...
```

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.vercel import VercelProvider

model = OpenAIChatModel('anthropic/claude-opus-4-8', provider=VercelProvider())
agent = Agent(model)
```

#### `MCPSamplingModel` + `MCPSamplingModelSettings`

**Module:** `pydantic_ai.models.mcp_sampling`. Routes an agent's LLM calls back through the *MCP
client's* sampling API — used when writing an MCP **server** that needs to call an LLM using the
connected client's credentials/model choice, per the [MCP sampling spec](https://modelcontextprotocol.io/docs/concepts/sampling).
`default_max_tokens` (default `16_384`) is required because MCP's `create_message` mandates
`max_tokens` while `ModelSettings.max_tokens` is optional. No streaming support —
`request_stream` raises `NotImplementedError`.

```python
class MCPSamplingModel(Model):
    session: ServerSession
    default_max_tokens: int = 16_384
```

```python
from mcp.server.session import ServerSession
from pydantic_ai import Agent
from pydantic_ai.models.mcp_sampling import MCPSamplingModel

async def summarise_tool(session: ServerSession, document: str) -> str:
    agent = Agent(MCPSamplingModel(session=session), system_prompt='Summarise concisely.')
    result = await agent.run(document)   # non-streaming only
    return result.output
```

#### `AnthropicModelSettings`

Extends `ModelSettings` with `anthropic_`-prefixed fields, all ignored by non-Anthropic providers so they merge safely cross-provider: `anthropic_cache` / `anthropic_cache_tool_definitions` / `anthropic_cache_instructions` (`bool | Literal['5m','1h']`), `anthropic_thinking` (low-level budget config), `anthropic_metadata` (`user_id` for abuse detection), `anthropic_service_tier` (`'auto'|'standard_only'`).

```python
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModelSettings

settings: AnthropicModelSettings = {
    'anthropic_cache_instructions': '1h',
    'anthropic_cache_tool_definitions': True,
    'anthropic_service_tier': 'standard_only',
}
agent = Agent('anthropic:claude-opus-4-5', model_settings=settings)
```

#### `OpenAIResponsesModelSettings` — reasoning context, mode, replay

Controls the OpenAI Responses API for reasoning models. `openai_reasoning_context` (`'auto'|'current_turn'|'all_turns'`, default `'all_turns'` on supported models) selects which prior-turn reasoning items the model replays. `openai_reasoning_mode` (`'standard'|'pro'`) trades latency for reliability. `openai_send_reasoning_ids: bool` — set `False` to strip reasoning-part IDs from history when a custom `ProcessHistory` removes thinking parts, avoiding history-mismatch errors.

```python
class OpenAIResponsesModelSettings(ModelSettings, total=False):
    openai_reasoning_effort: str
    openai_reasoning_context: Literal['auto', 'current_turn', 'all_turns']
    openai_reasoning_mode: Literal['standard', 'pro']
    openai_send_reasoning_ids: bool
```

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings

model = OpenAIResponsesModel('gpt-5.6-sol')
agent = Agent(model, model_settings=OpenAIResponsesModelSettings(
    openai_reasoning_effort='high', openai_reasoning_context='all_turns',
))
```

#### `ModelHTTPError` — `headers` + `retry_after`

Raised on 4xx/5xx provider responses. Carries `headers: Mapping[str, str] | None` (lowercased keys; `None` for gRPC-based paths like xAI) and the derived `retry_after: float | None` property that parses RFC-7231 `Retry-After` (delta-seconds or HTTP-date). Propagated by all built-in HTTP-based providers.

```python
class ModelHTTPError(Exception):
    def __init__(self, status_code: int, model_name: str, body=None, *, headers=None, suggested_model_id=None): ...
    @property
    def retry_after(self) -> float | None: ...
```

```python
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError
import asyncio

agent = Agent('openai:gpt-5.2')

async def run_with_retry(prompt: str) -> str:
    try:
        return (await agent.run(prompt)).output
    except ModelHTTPError as exc:
        if exc.status_code == 429:
            await asyncio.sleep(exc.retry_after or 5.0)
            return (await agent.run(prompt)).output
        raise
```

#### `RaiseContentFilterError`

An `after_model_request` capability that turns `finish_reason='content_filter'` into a `ContentFilterError` exception instead of passing the filtered response through silently. The full `ModelResponse` is serialised into `ContentFilterError.body` for inspection.

```python
@dataclass
class RaiseContentFilterError(AbstractCapability[AgentDepsT]):
    id: str | None = None
```

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import RaiseContentFilterError
from pydantic_ai.exceptions import ContentFilterError

agent = Agent('openai:gpt-5.2', capabilities=[RaiseContentFilterError()])
try:
    result = agent.run_sync('...')
except ContentFilterError as exc:
    print('Filtered:', exc.body[:200])
```

#### `JsonSchemaTransformer` + `InlineDefsJsonSchemaTransformer`

Every provider calls `JsonSchemaTransformer.walk()` on each tool's JSON schema during `prepare_request()` to normalise it for that provider's requirements. Subclass and implement `transform(schema)`; set `self.is_strict_compatible = False` inside it to signal a schema can't be used in strict mode. `InlineDefsJsonSchemaTransformer` expands all `$ref`/`$defs` inline (used by Amazon/Bedrock profiles and providers without `$ref` support, e.g. Qwen); recursive types keep a minimal `$defs`/`$ref` (unavoidable for cycles).

```python
class JsonSchemaTransformer(ABC):
    def __init__(self, schema, *, strict=None, prefer_inlined_defs=False): ...
    def walk(self) -> JsonSchema: ...
    @abstractmethod
    def transform(self, schema: JsonSchema) -> JsonSchema: ...
```

```python
from pydantic_ai._json_schema import InlineDefsJsonSchemaTransformer

schema = {'$defs': {'Address': {'type': 'object', 'properties': {'city': {'type': 'string'}}}},
          'type': 'object', 'properties': {'home': {'$ref': '#/$defs/Address'}}}
inlined = InlineDefsJsonSchemaTransformer(schema).walk()
# 'home' now contains the full Address object inline, no $defs/$ref
```

#### Community OpenAI-compatible providers

All of these are `Provider[AsyncOpenAI]` implementations paired with `OpenAIChatModel` — they
differ only in endpoint URL, env var, model-name convention, and which `*_model_profile` family
functions get dispatched based on the model-name prefix.

| Provider | Module | Env var | Base URL | Naming | Notes |
|---|---|---|---|---|---|
| `LiteLLMProvider` | `.litellm` | — (proxy key) | `api_base=` (your proxy) | `provider/model` | Auto-dispatches profile by prefix (`anthropic/`, `google/`, `bedrock/`, etc.) |
| `AzureProvider` | `.azure` | `AZURE_OPENAI_API_KEY` | `azure_endpoint=` | deployment name | `/v1`-suffix endpoints (Express Mode, AI Foundry serverless) must **omit** `api_version` — passing one raises `UserError` |
| `DeepSeekProvider` | `.deepseek` | `DEEPSEEK_API_KEY` | fixed | `deepseek-chat` / `deepseek-reasoner` | R1 exposes `reasoning_content` field, mapped to `ThinkingPart`; reasoning models reject `tool_choice='required'` |
| `CerebrasProvider` | `.cerebras` | `CEREBRAS_API_KEY` | fixed | flat | Adds `X-Cerebras-3rd-Party-Integration` header; disables `frequency_penalty`/`logit_bias`/`presence_penalty`/`parallel_tool_calls`/`service_tier` |
| `GitHubProvider` | `.github` | `GITHUB_API_KEY` (PAT) | `models.github.ai/inference` | `provider/model:tag` | Tag suffix stripped before profile match |
| `FireworksProvider` | `.fireworks` | `FIREWORKS_API_KEY` | `api.fireworks.ai/inference/v1` | `accounts/fireworks/models/<name>` | Prefix stripped before profile match |
| `TogetherProvider` | `.together` | `TOGETHER_API_KEY` | `api.together.xyz/v1` | `org/model` | — |
| `NebiusProvider` | `.nebius` | `NEBIUS_API_KEY` | `api.studio.nebius.com/v1` | `org/model` | No-slash name → plain `OpenAIModelProfile` |
| `SambaNovaProvider` | `.sambanova` | `SAMBANOVA_API_KEY` | `api.sambanova.ai/v1` or `SAMBANOVA_BASE_URL` | flat | Only one supporting an on-prem `base_url` override |
| `AlibabaProvider` | `.alibaba` | `ALIBABA_API_KEY`/`DASHSCOPE_API_KEY` | DashScope intl/CN | Qwen model IDs | `*-omni*` models get `openai_chat_audio_input_encoding='uri'` forced |
| `OVHcloudProvider` | `.ovhcloud` | `OVHCLOUD_API_KEY` | OVH AI Endpoints | vendor-prefixed | Routes `llama`/`deepseek`/`mistral`/`gpt`/`qwen` prefixes to matching profiles |

```python
from pydantic_ai import Agent
from pydantic_ai.providers.deepseek import DeepSeekProvider
from pydantic_ai.models.openai import OpenAIChatModel

provider = DeepSeekProvider(api_key='sk-...')
reasoning_agent = Agent(OpenAIChatModel('deepseek-reasoner', provider=provider))
```

#### `HerokuProvider` — Heroku Managed Inference

An OpenAI-compatible gateway with a multi-family model-profile router: `HerokuProvider.model_profile(model_name)` detects the model family from the bare name (no provider prefix) and applies the correct profile — `claude*` → `anthropic_model_profile`, `gpt-oss*` → `harmony_model_profile`, `qwen*`/`deepseek*`/`kimi*`/`glm*`/`mistral*`/`nova*`/`llama*`/`gemma*` → their respective profiles — all merged over `OpenAIModelProfile`. Base URL defaults to `https://us.inference.heroku.com/v1`.

```python
class HerokuProvider(Provider[AsyncOpenAI]):
    def __init__(self, *, base_url=None, api_key=None, openai_client=None, http_client=None) -> None: ...
    @staticmethod
    def model_profile(model_name: str) -> ModelProfile | None: ...
```

```python
from pydantic_ai import Agent
from pydantic_ai.providers.heroku import HerokuProvider
from pydantic_ai.models.openai import OpenAIChatModel

provider = HerokuProvider(api_key='...')
model = OpenAIChatModel('claude-opus-4-5', provider=provider)   # thinking correctly forwarded
agent = Agent(model, model_settings={'thinking': True})
```

#### `gateway_provider` + `normalize_gateway_provider` — Pydantic AI Gateway

`gateway_provider(upstream_provider, ...)` routes through `gateway.pydantic.dev/proxy` (or the deprecated alias, still-working `PYDANTIC_AI_GATEWAY_API_KEY`-keyed proxy), a managed proxy fronting OpenAI/Anthropic/Groq/Bedrock/Google Cloud with one API key. Accepts both model-provider names and API-flavour aliases (`'chat'`, `'responses'`, `'converse'`, `'google-cloud'`); `route=` overrides the default routing group. Region-encoded `pylf_v*` keys let `_infer_base_url` auto-select the nearest regional endpoint.

```python
def gateway_provider(upstream_provider: str, /, *, route=None, api_key=None, base_url=None, http_client=None) -> Provider[Any]: ...
def normalize_gateway_provider(provider: str) -> str: ...   # 'chat' -> 'openai-chat', etc.
```

```python
from pydantic_ai import Agent
from pydantic_ai.providers.gateway import gateway_provider
from pydantic_ai.models.openai import OpenAIChatModel

provider = gateway_provider('anthropic', api_key='pylf_v1_us_...')
agent = Agent(OpenAIChatModel('claude-opus-4-8', provider=provider))
```

#### Provider-family `*_model_profile` functions

Called internally by the matching `Provider.model_profile()` to set capability flags from the
model-name string. All return `TypedDict`-shaped `ModelProfile` objects (or `None` for "use
default").

```python
grok_model_profile(name) -> GrokModelProfile | None
groq_model_profile(name) -> GroqModelProfile | None       # groq_always_has_web_search_builtin_tool for compound-*
deepseek_model_profile(name) -> ModelProfile | None       # R1: thinking_always_enabled=True
qwen_model_profile(name) -> ModelProfile | None           # always InlineDefsJsonSchemaTransformer
cohere_model_profile(name) -> ModelProfile | None         # 'reasoning' in name -> thinking_always_enabled
moonshotai_model_profile(name) -> ModelProfile | None     # ignore_streamed_leading_whitespace=True
```

```python
from pydantic_ai.profiles.deepseek import deepseek_model_profile

profile = deepseek_model_profile('deepseek-r1')
assert profile['thinking_always_enabled'] is True
```

#### `OutlinesModel` — removed

`OutlinesModel` (local Transformers/LlamaCpp/SGLang/vLLM-offline/MLX grammar-constrained decoding)
was deprecated as of 1.107.0 and is **fully removed** — no trace remains under
`pydantic_ai/models/`. For local structured output, use vLLM's structured-output API behind
`OpenAIProvider(base_url=...)`, or lean on `NativeOutput`/`PromptedOutput` against an API model.

---

### Tools & Toolsets

#### `Tool` — direct construction

`@agent.tool`/`@agent.tool_plain` are sugar over this dataclass.

```python
Tool(
    function, *, takes_ctx=None, max_retries=None, name=None, description=None,
    prepare=None, args_validator=None, docstring_format='auto',
    require_parameter_descriptions=False, schema_generator=GenerateToolJsonSchema,
    strict=None, sequential=False, requires_approval=False, metadata=None,
    timeout=None, defer_loading=False, include_return_schema=None,
    function_schema: FunctionSchema | None = None,   # pre-built schema, skips re-derivation
)
```

```python
from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.exceptions import ModelRetry

def validate_age(ctx: RunContext[None], age: int) -> None:
    if age < 0 or age > 150:
        raise ModelRetry(f'Age {age} is not plausible.')

def get_birth_year(ctx: RunContext[None], age: int) -> int:
    return 2026 - age

agent = Agent('openai:gpt-4o', tools=[Tool(get_birth_year, args_validator=validate_age)])
```

`prepare=` mutates or hides the `ToolDefinition` before each step (return `None` to hide);
`requires_approval=True` raises `ApprovalRequired` on call (see Security section);
`sequential=True` forces the tool to never run in parallel with others in the same step.
`args_validator` runs after schema validation but before execution — receives schema-validated
kwargs and should raise `ModelRetry` on failure.

```python
from pydantic_ai import Agent, ModelRetry, RunContext

agent = Agent("openai:gpt-4o-mini")

def validate_url(ctx: RunContext[None], url: str) -> None:
    if not url.startswith("https://"):
        raise ModelRetry("URL must use HTTPS")

@agent.tool(args_validator=validate_url)
async def fetch_url(ctx: RunContext[None], url: str) -> str:
    return f"Fetched: {url}"
```

#### `FunctionToolset` — the primary toolset

The richest toolset — powers `@agent.tool`/`@agent.tool_plain` but also works standalone.

```python
FunctionToolset(
    tools=(), *, max_retries=None, timeout=None, docstring_format='auto',
    require_parameter_descriptions=False, schema_generator=GenerateToolJsonSchema,
    strict=None, sequential=False, requires_approval=False, metadata=None,
    defer_loading=False, include_return_schema=None, id=None,
    instructions: str | SystemPromptFunc | Sequence[...] | None = None,
)
```

`timeout` delivers a retry prompt instead of an exception on a slow tool. `sequential=True` is a
barrier — the tool never runs in parallel with others in the same step. `id` gives the toolset a
stable identity for durable-execution runtimes. `instructions` injects a system-prompt segment
whenever any tool in the set is active.

```python
from pydantic_ai import Agent, FunctionToolset, RunContext

db_tools = FunctionToolset(
    instructions='When using DB tools, always use read-only queries first.',
)

@db_tools.tool_plain
def query_users(sql: str) -> list[dict]:
    """Execute a read-only SQL query."""
    return [{'id': 1, 'name': 'Alice'}]

agent = Agent('openai:gpt-4o', toolsets=[db_tools])
```

#### `AbstractToolset` — custom toolset ABC

Base class every toolset implements. Override `get_tools()` (returning
`dict[str, ToolsetTool]`) and `call_tool()`; optionally `for_run`/`for_run_step` for per-run state
isolation, and `get_instructions()` for toolset-scoped system-prompt text.

```python
class AbstractToolset(ABC, Generic[AgentDepsT]):
    id: str | None
    async def get_tools(self, ctx: RunContext) -> dict[str, ToolsetTool]: ...
    async def call_tool(self, name: str, tool_args: dict, ctx: RunContext, tool: ToolsetTool) -> Any: ...
    async def for_run(self, ctx: RunContext) -> 'AbstractToolset': ...      # per-run state
    async def get_instructions(self, ctx: RunContext) -> str | None: ...
```

```python
from pydantic_ai.toolsets.abstract import AbstractToolset, ToolsetTool
from pydantic_ai.tools import ToolDefinition

class CalculatorToolset(AbstractToolset):
    id = 'calculator'
    async def get_tools(self, ctx):
        td = ToolDefinition(
            name='add', description='Add two numbers.',
            parameters_json_schema={'type': 'object', 'properties': {'a': {'type': 'number'}, 'b': {'type': 'number'}}, 'required': ['a', 'b']},
        )
        return {'add': ToolsetTool(toolset=self, tool_def=td, max_retries=1, args_validator=...)}
    async def call_tool(self, name, tool_args, ctx, tool):
        return tool_args['a'] + tool_args['b']
```

#### `ToolsetTool` + `SchemaValidatorProt`

`ToolsetTool` is the runtime execution wrapper for one tool inside a toolset (`toolset`, `tool_def`, `max_retries`, `args_validator`, `args_validator_func`) — surfaced in `before_tool_validate`/`after_tool_execute` hooks and returned from custom `get_tools()`. `SchemaValidatorProt` is the `Protocol` any custom validator must satisfy (`validate_json`/`validate_python`, compatible with `pydantic_core.SchemaValidator`), letting non-Pydantic validation engines plug in.

```python
@dataclass(kw_only=True)
class ToolsetTool(Generic[AgentDepsT]):
    toolset: AbstractToolset
    tool_def: ToolDefinition
    max_retries: int
    args_validator: SchemaValidator | SchemaValidatorProt
    args_validator_func: Callable[..., Any] | None = None
```

#### Toolset composition — `RenamedToolset` / `WrapperToolset` / `FilteredToolset` / `PreparedToolset` / `PrefixedToolset` / `CombinedToolset`

All confirmed present with unchanged constructors. `WrapperToolset` is the delegation base for the rest — subclass it and override `call_tool`/`get_tools` to add cross-cutting behaviour (logging, caching) while delegating everything else; `visit_and_replace(visitor)` recursively traverses a wrapper chain to swap out a specific inner toolset.

```python
@dataclass
class WrapperToolset(AbstractToolset[AgentDepsT]):
    def __init__(self, wrapped: AbstractToolset[AgentDepsT]) -> None: ...
    def visit_and_replace(self, visitor: Callable[[AbstractToolset], AbstractToolset]) -> AbstractToolset: ...
```

`RenamedToolset` wraps a toolset and remaps tool names via `name_map: dict[str, str]` (**new name → original name**); unmapped tools pass through unchanged. `call_tool` inverts the map and restores `ctx.tool_name`/`tool.tool_def.name` to the original before delegating. Attempting to rename onto an existing or duplicate name raises `UserError`.

```python
@dataclass
class RenamedToolset(WrapperToolset[AgentDepsT]):
    def __init__(self, wrapped: AbstractToolset[AgentDepsT], name_map: dict[str, str]) -> None: ...
```

`FilteredToolset` wraps any toolset and calls `filter_func(RunContext, ToolDefinition) -> bool | Awaitable[bool]` on every tool at every `get_tools()` call — both sync and async predicates supported via `inspect.isawaitable()`.

```python
@dataclass
class FilteredToolset(WrapperToolset[AgentDepsT]):
    def __init__(self, wrapped: AbstractToolset[AgentDepsT], filter_func: Callable[[RunContext, ToolDefinition], bool | Awaitable[bool]]) -> None: ...
```

`PreparedToolset` calls `prepare_func(RunContext, list[ToolDefinition]) -> list[ToolDefinition]` on each step. The function may filter or modify definitions (descriptions, `strict`, metadata) but **cannot** add, rename, or substitute tools — attempting to raises `UserError`.

```python
@dataclass
class PreparedToolset(WrapperToolset[AgentDepsT]):
    def __init__(self, wrapped: AbstractToolset[AgentDepsT], prepare_func: ToolsPrepareFunc[AgentDepsT]) -> None: ...
```

`PrefixedToolset` prepends `{prefix}_` to every tool name (strips it back off before dispatching `call_tool`) — the standard fix for name collisions between combined toolsets or MCP servers.

```python
@dataclass
class PrefixedToolset(WrapperToolset[AgentDepsT]):
    prefix: str
    @property
    def tool_name_conflict_hint(self) -> str: ...
```

`CombinedToolset` fans out `get_tools()`/`get_instructions()`/`for_run()`/`for_run_step()` across child toolsets in parallel via `gather()`. Detects name collisions eagerly and raises `UserError` naming both conflicting toolsets and pointing at `tool_name_conflict_hint`. `for_run_step` short-circuits (returns `self`) when no child toolset actually changed.

```python
@dataclass
class CombinedToolset(AbstractToolset[AgentDepsT]):
    def __init__(self, toolsets: Sequence[AbstractToolset[AgentDepsT]]) -> None: ...
```

```python
from pydantic_ai import Agent, FunctionToolset, RunContext
from pydantic_ai.toolsets import CombinedToolset, PrefixedToolset, FilteredToolset

db, web = FunctionToolset(), FunctionToolset()
@db.tool_plain
def search(query: str) -> str: return f'DB: {query}'
@web.tool_plain
def search(query: str) -> str: return f'Web: {query}'   # would collide without prefixing

def read_only(ctx: RunContext, td) -> bool:
    return not td.name.startswith('delete_')

agent = Agent('openai:gpt-4o', toolsets=[
    FilteredToolset(
        CombinedToolset([PrefixedToolset(db, 'db'), PrefixedToolset(web, 'web')]),
        filter_func=read_only,
    )
])
```

#### `DeferredLoadingToolset`

Hides some or all of a wrapped toolset's tools until the `ToolSearch` capability discovers them.
`tool_names=None` defers everything; a `frozenset[str]` defers only the named tools. Marks tools
with `defer_loading=True` on their `ToolDefinition`. `FunctionToolset(defer_loading=True)` is a
shortcut that wraps itself automatically.

```python
DeferredLoadingToolset(wrapped: AbstractToolset, *, tool_names: frozenset[str] | None = None)
```

```python
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.toolsets import DeferredLoadingToolset
from pydantic_ai.capabilities import ToolSearch

big_toolset = FunctionToolset()   # imagine 50+ tools registered here
agent = Agent('openai:gpt-4o', toolsets=[DeferredLoadingToolset(big_toolset)], capabilities=[ToolSearch()])
```

#### `DynamicToolset`

Wraps a factory `Callable[[RunContext], AbstractToolset | None]` (`ToolsetFunc`) and re-evaluates
it either every step (`per_run_step=True`, default) or once per run (`per_run_step=False`).
Lifecycle is transition-safe: the old inner toolset's `__aexit__` runs before the new one's
`__aenter__`.

```python
class DynamicToolset(AbstractToolset[AgentDepsT]):
    def __init__(self, toolset_func: ToolsetFunc[AgentDepsT], *, per_run_step: bool = True, id: str | None = None): ...
```

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.toolsets import FunctionToolset, DynamicToolset

def role_based_toolset(ctx: RunContext) -> FunctionToolset:
    return admin_tools if ctx.deps.role == 'admin' else user_tools

agent = Agent('openai:gpt-4o-mini', toolsets=[DynamicToolset(role_based_toolset)])
```

#### `IncludeReturnSchemasToolset` (toolset) vs. `IncludeToolReturnSchemas` (capability)

Both exist and do the same job at different layers: `IncludeReturnSchemasToolset`
(`pydantic_ai.toolsets`) wraps one toolset; `IncludeToolReturnSchemas`
(`pydantic_ai.capabilities`) applies agent-wide via `capabilities=[...]`. Both set
`include_return_schema=True` on every `ToolDefinition` whose value is still `None`, so the model
sees the tool's return-type JSON schema (useful for tool chaining and OpenAI structured outputs).

```python
from pydantic_ai import Agent, IncludeReturnSchemasToolset, FunctionToolset
from pydantic_ai.capabilities import IncludeToolReturnSchemas

toolset = FunctionToolset()
@toolset.tool
def get_weather(city: str) -> dict:
    """Get current weather."""
    return {'temp_c': 22.5, 'condition': 'sunny'}

agent = Agent('openai:gpt-4o', toolsets=[IncludeReturnSchemasToolset(toolset)])
# or agent-wide: Agent('openai:gpt-4o', toolsets=[toolset], capabilities=[IncludeToolReturnSchemas()])
```

#### `ExternalToolset` (+ deprecated `DeferredToolset` alias)

Registers tool *schemas* for the model without ever executing them in-process — the calls are
resolved by an external system and fed back via `deferred_tool_results`. `call_tool()` raises
`NotImplementedError`; every registered tool gets `tool_kind='external'` via a
`TOOL_SCHEMA_VALIDATOR = SchemaValidator(schema=core_schema.any_schema())` that accepts any args
shape. `id=` gives the toolset a stable identity for durable-execution runtimes to match
activities across replays. `DeferredToolset` is a deprecated alias for backward compatibility —
migrate to `ExternalToolset`.

```python
ExternalToolset(tool_defs: list[ToolDefinition], *, id: str | None = None)
```

```python
from pydantic_ai import Agent, ExternalToolset, ToolDefinition, DeferredToolRequests, DeferredToolResults, ToolReturn

external = ExternalToolset([
    ToolDefinition(
        name='send_email', description='Send an email.',
        parameters_json_schema={'type': 'object', 'properties': {'to': {'type': 'string'}, 'body': {'type': 'string'}}, 'required': ['to', 'body']},
    ),
])
agent = Agent('openai:gpt-4o', output_type=[str, DeferredToolRequests], toolsets=[external])

result1 = agent.run_sync('Email alice@example.com saying hi.')
if isinstance(result1.output, DeferredToolRequests):
    call = result1.output.calls[0]
    results = {call.tool_call_id: ToolReturn(content=f"Sent to {call.args_as_dict()['to']}")}
    result2 = agent.run_sync(
        message_history=result1.all_messages(),
        deferred_tool_results=DeferredToolResults(calls=results),
    )
```

#### `ApprovalRequiredToolset`

Wraps a toolset with an approval gate. Every call to `approval_required_func` (default: approve
none automatically, i.e. gate everything) decides whether the tool call raises `ApprovalRequired`;
the constructor field is `wrapped=`, not `toolset=`. See the Security section for the full
two-round `DeferredToolRequests` resume flow.

```python
ApprovalRequiredToolset(
    wrapped: AbstractToolset,
    approval_required_func: Callable[[RunContext, ToolDefinition, dict], bool] = lambda ctx, td, args: True,
)
```

```python
from pydantic_ai import Agent, FunctionToolset, DeferredToolRequests
from pydantic_ai.toolsets.approval_required import ApprovalRequiredToolset

tools = FunctionToolset()
@tools.tool_plain
def delete_file(path: str) -> str: return f'Deleted {path}'
@tools.tool_plain
def read_file(path: str) -> str: return f'contents of {path}'

def needs_approval(ctx, tool_def, args): return tool_def.name.startswith('delete_')

agent = Agent(
    'openai:gpt-4o',
    toolsets=[ApprovalRequiredToolset(wrapped=tools, approval_required_func=needs_approval)],
    output_type=[str, DeferredToolRequests],
)
```

#### `FunctionSchema` + `function_schema()` (+ `GenerateToolJsonSchema` / `DocstringFormat`)

**Module:** `pydantic_ai._function_schema`. The frozen dataclass and factory function that convert
a Python function into a tool's JSON schema + calling convention — `takes_ctx` auto-detection,
async/sync dispatch, single-arg model unwrapping, and return-type schema extraction.
`GenerateToolJsonSchema` strips redundant `title` keys from every property; `DocstringFormat =
Literal['google', 'numpy', 'sphinx', 'auto']` controls how parameter descriptions are parsed out of
docstrings (via [griffe](https://mkdocstrings.github.io/griffe/) — `'auto'` uses regex inference:
Sphinx `:param:`, Google `Args:` block, NumPy `---` underline, falling back to `'google'`).

```python
@dataclass
class FunctionSchema:
    function: Callable
    description: str | None
    validator: SchemaValidator
    json_schema: ObjectJsonSchema
    single_arg_name: str | None
    takes_ctx: bool
    is_async: bool
    return_schema: ObjectJsonSchema
```

```python
from pydantic_ai._function_schema import function_schema
from pydantic.json_schema import GenerateJsonSchema

def get_weather(city: str, unit: str = 'celsius') -> dict:
    """Get current weather.

    Args:
        city: The city to look up.
        unit: 'celsius' or 'fahrenheit'.
    """
    return {'city': city, 'temp': 22}

schema = function_schema(get_weather, schema_generator=GenerateJsonSchema, docstring_format='google')
print(schema.takes_ctx, schema.is_async)   # False False
```

#### `ToolChoice` + `ToolOrOutput`

`ToolChoice = Literal['none', 'required', 'auto'] | list[str] | ToolOrOutput | None`, set via
`ModelSettings(tool_choice=...)`. `ToolOrOutput(function_tools=[...])` restricts which function
tools are callable while still allowing the model to use output/text/image tools freely.

```python
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.settings import ToolOrOutput

agent.run_sync('...', model_settings=ModelSettings(tool_choice='required'))
agent.run_sync('...', model_settings=ModelSettings(tool_choice=ToolOrOutput(function_tools=['search_kb'])))
```

#### `PrefixedToolset` / `PrefixTools` capability

`PrefixTools` is the capability-level equivalent of `PrefixedToolset`: it wraps another *capability* and prefixes its contributed tools, delegating to `PrefixedToolset` internally (or `DynamicToolset` first if the wrapped toolset is a callable factory).

```python
@dataclass
class PrefixTools(WrapperCapability[AgentDepsT]):
    prefix: str
```

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import PrefixTools, Toolset
from pydantic_ai.toolsets import FunctionToolset

db_toolset = FunctionToolset()
agent = Agent('openai:gpt-4o-mini', capabilities=[PrefixTools(wrapped=Toolset(db_toolset), prefix='db')])
# model sees 'db_query', 'db_insert', etc.
```

#### `ToolSearch` capability + `ToolSearchToolset`

Lazy tool discovery for large toolsets. `ToolSearch` is auto-injected whenever deferred tools exist (zero overhead otherwise). On providers with native tool search (Anthropic BM25/regex, OpenAI Responses) deferred tools are sent on the wire and the provider handles discovery; elsewhere a local `search_tools` function is exposed. `strategy` accepts `None` (auto), `'bm25'`/`'regex'` (Anthropic-only, error elsewhere), `'keywords'` (force the local algorithm everywhere for determinism), or a custom `ToolSearchFunc`; `ToolSearchToolset.enable_fallback=False` disables the local `search_tools` fallback for native-only strategies.

```python
@dataclass
class ToolSearch(AbstractCapability[AgentDepsT]):
    strategy: ToolSearchStrategy | None = None
    max_results: int = 10

class ToolSearchToolset(WrapperToolset[AgentDepsT]):
    def __init__(self, wrapped, search_fn=None, max_results=10, enable_fallback=True, ...): ...
```

```python
from pydantic_ai import Agent, Tool
from pydantic_ai.capabilities import ToolSearch

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    tools=[Tool(lambda booking_id: 'ok', defer_loading=True)],
    capabilities=[ToolSearch()],
)
```

#### `LangChainTool` + `LangChainToolset` + `tool_from_langchain`

**Module:** `pydantic_ai.ext.langchain` — the only surviving file under `pydantic_ai/ext/`
alongside `__init__.py`. Bridges any LangChain `BaseTool` without requiring `langchain` as an
import-time dependency (`LangChainTool` is a structural `Protocol`: `.name`, `.get_input_jsonschema()`,
`.description`, `.run()`).

```python
class LangChainTool(Protocol):
    @property
    def name(self) -> str: ...
    def get_input_jsonschema(self) -> JsonSchemaValue: ...
    def run(self, *args, **kwargs) -> str: ...

class LangChainToolset(FunctionToolset):
    def __init__(self, tools: list[LangChainTool], *, id: str | None = None): ...
```

```python
from pydantic_ai import Agent
from pydantic_ai.ext.langchain import LangChainToolset
from langchain_community.tools import DuckDuckGoSearchRun

toolset = LangChainToolset([DuckDuckGoSearchRun()])
agent = Agent('openai:gpt-4o', toolsets=[toolset])
```

#### `FunctionSignature` + `FunctionParam` + `TypeSignature` + `TypeExpr`

Power CLI/tool-rendering and "Code Mode": `FunctionSignature` holds a parsed function's name, `params` (`FunctionParam`), `return_type` (`TypeExpr`), and `referenced_types` (nested `TypeSignature`s for TypedDicts). `TypeExpr` is a union of `SimpleTypeExpr | UnionTypeExpr | GenericTypeExpr | LiteralTypeExpr` covering every annotation shape. `.render(body)` produces the Python-source representation.

```python
@dataclass(kw_only=True)
class FunctionSignature:
    name: str
    params: dict[str, FunctionParam]
    return_type: TypeExpr
    referenced_types: list[TypeSignature] = field(default_factory=list)
    def render(self, body: str) -> str: ...
```

```python
from pydantic_ai.function_signature import FunctionSignature, FunctionParam, SimpleTypeExpr

sig = FunctionSignature(
    name='search',
    params={'query': FunctionParam(name='query', type=SimpleTypeExpr(name='str'))},
    return_type=SimpleTypeExpr(name='str'),
)
print(sig.render(body='    ...'))
```

#### `ACIToolset` + `tool_from_aci` — removed

Both were deprecated in 1.107.0 as the ACI.dev bridge; **fully removed** — `pydantic_ai/ext/`
now contains only `langchain.py`. Migrate to `Tool.from_schema()` built directly from
`aci.functions.get_definition(...)` output (strip the non-standard `'visible'` key before passing
the schema through).

---

### Native / Built-in Tools

#### `WebSearchTool` + `WebSearchUserLocation`

Native web search across Anthropic, OpenAI Responses, Groq, Google, xAI, and OpenRouter.

```python
WebSearchTool(*, search_context_size='medium', user_location=None,
              blocked_domains=None, allowed_domains=None, max_uses=None,
              external_web_access: bool | None = None, optional=False)

class WebSearchUserLocation(TypedDict, total=False):
    city: str; country: str; region: str; timezone: str
```

```python
from pydantic_ai import Agent, WebSearchTool, WebSearchUserLocation
from pydantic_ai.capabilities import NativeTool

agent = Agent(
    'anthropic:claude-opus-4-5',
    capabilities=[NativeTool(WebSearchTool(
        user_location=WebSearchUserLocation(city='London', country='GB'),
        search_context_size='high',
    ))],
)
```

#### `WebFetchTool` (+ deprecated `UrlContextTool` alias)

Fetches URL content directly into the model's context (Anthropic, Google). `UrlContextTool` is a
deprecated alias kept only so old serialised payloads (`kind='url_context'`) still deserialise.

```python
WebFetchTool(*, max_uses=None, allowed_domains=None, blocked_domains=None,
             enable_citations=False, max_content_tokens=None, optional=False)
```

```python
from pydantic_ai import Agent, WebFetchTool
from pydantic_ai.capabilities import NativeTool

agent = Agent(
    'anthropic:claude-sonnet-4-5',
    capabilities=[NativeTool(WebFetchTool(enable_citations=True, max_content_tokens=4096))],
)
```

#### `CodeExecutionTool`

Sandboxed code interpreter (Anthropic, OpenAI Responses, Google, Bedrock Nova 2.0, xAI). Gained a
`files: list[UploadedFile] | None` field to seed the sandbox with pre-uploaded files.

```python
CodeExecutionTool(*, files: list[UploadedFile] | None = None, optional=False)
```

```python
from pydantic_ai import Agent, CodeExecutionTool
from pydantic_ai.capabilities import NativeTool

agent = Agent('openai:gpt-4o', capabilities=[NativeTool(CodeExecutionTool())])
result = agent.run_sync('Verify whether 982,451,653 is prime; show your work.')
```

#### `MemoryTool`

Native persistent memory (Anthropic only). No parameters beyond the shared `optional` flag.

```python
from pydantic_ai import Agent, MemoryTool
from pydantic_ai.capabilities import NativeTool

agent = Agent('anthropic:claude-opus-4-5', capabilities=[NativeTool(MemoryTool(optional=True))])
```

#### `ImageGenerationTool` + `ImageAspectRatio`

All 12 fields present, including `input_fidelity`, `partial_images`, Google-specific `size`
literals (`'512'`/`'1K'`/`'2K'`/`'4K'`), and `aspect_ratio`.

```python
ImageGenerationTool(*, action='auto', background='auto', input_fidelity=None,
    moderation='auto', model=None, output_compression=None, output_format=None,
    partial_images=0, quality='auto', size=None, aspect_ratio=None)
```

```python
from pydantic_ai import Agent, ImageGenerationTool
from pydantic_ai.capabilities import NativeTool

agent = Agent(
    'openai-responses:gpt-4o',
    capabilities=[NativeTool(ImageGenerationTool(quality='high', output_format='png', aspect_ratio='1:1'))],
)
```

#### `MCPServerTool`

Delegates MCP-server interaction directly to the provider (no PydanticAI proxying), unlike
`MCPToolset`. `headers: dict[str, str] | None` field. Supported by OpenAI Responses, Anthropic, xAI.

```python
MCPServerTool(*, id: str, url: str, authorization_token=None, description=None,
              allowed_tools=None, headers=None, optional=False)
```

```python
from pydantic_ai import Agent, MCPServerTool
from pydantic_ai.capabilities import NativeTool

agent = Agent(
    'openai-responses:gpt-4o',
    capabilities=[NativeTool(MCPServerTool(
        id='github-mcp', url='https://mcp.github.com/', authorization_token='ghp_...',
        allowed_tools=['list_repos', 'create_issue'],
    ))],
)
```

#### `FileSearchTool`

Native provider-managed RAG (OpenAI vector stores, Google Gemini Files API, xAI collections).

```python
FileSearchTool(*, file_store_ids: Sequence[str], max_num_results=None,
                instructions=None, retrieval_mode=None, optional=False)
```

```python
from pydantic_ai import Agent, FileSearchTool
from pydantic_ai.capabilities import NativeTool

agent = Agent(
    'openai-responses:gpt-4o',
    capabilities=[NativeTool(FileSearchTool(file_store_ids=['vs_abc123'], retrieval_mode='hybrid'))],
)
```

#### `XSearchTool`

X/Twitter search, native on xAI only; other providers need `XSearch(fallback_model=...)` (see
Capabilities section).

```python
XSearchTool(*, allowed_x_handles=None, excluded_x_handles=None, from_date=None,
            to_date=None, enable_image_understanding=False, enable_video_understanding=False,
            include_output=False, optional=False)
```

```python
from pydantic_ai import Agent, XSearchTool
from pydantic_ai.capabilities import NativeTool

agent = Agent('xai:grok-4', capabilities=[NativeTool(XSearchTool(allowed_x_handles=['openai', 'anthropic']))])
```

#### `AdvisorTool`

Provider-managed tool letting a fast executor model pause and consult a stronger advisor model
inline (Anthropic native, OpenRouter gateway). `caching: Literal['5m','1h'] | None` controls
prompt-cache TTL on the advisor call. **Correction:** the installed `AdvisorTool` has **no
`system_prompt` field** — earlier docs describing an Anthropic-only `system_prompt` override are
stale for this version.

```python
class AdvisorTool(AbstractNativeTool):
    def __init__(self, *, model: AdvisorModelName, max_uses=None, max_tokens=None, caching=None): ...
```

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeTool
from pydantic_ai.native_tools import AdvisorTool

agent = Agent(
    'anthropic:claude-haiku-4-5',
    capabilities=[NativeTool(AdvisorTool(model='claude-opus-4-8', max_uses=2, max_tokens=512))],
)
```

#### `AbstractNativeTool` — base class for custom native tools

Every built-in native tool subclasses this. `kind` is the wire discriminator (auto-registered into
`NATIVE_TOOL_TYPES` via `__init_subclass__`); `optional=True` silently drops the tool on models
that don't support it rather than raising.

```python
class AbstractNativeTool(ABC):
    kind: str = 'unknown_native_tool'
    optional: bool = False
    @property
    def unique_id(self) -> str: return self.kind
```

```python
from dataclasses import dataclass
from pydantic_ai.native_tools import AbstractNativeTool

@dataclass(kw_only=True)
class CompanyKBTool(AbstractNativeTool):
    kind: str = 'company_kb'
    index_name: str = 'main'
    @property
    def unique_id(self) -> str: return f'company_kb:{self.index_name}'
```

#### `MCPToolset` + `load_mcp_toolsets`

**Module:** `pydantic_ai.mcp`. The provider-agnostic way to connect to any MCP server — supports
HTTP/SSE/stdio/in-process transports. The legacy `MCPServer`/`MCPServerStdio`/`MCPServerSSE`/
`MCPServerStreamableHTTP` classes are **confirmed fully removed** — grep found zero trace of them
in `pydantic_ai/mcp.py`. `FastMCPToolset` (deprecated in 1.104) is likewise gone entirely.

```python
MCPToolset(
    client: MCPToolsetClient, *, id=None, max_retries=None,
    tool_error_behavior: Literal['retry', 'error', 'failed'] = 'retry',
    process_tool_call=None, prefer_tasks=True, cache_tools=True,
    cache_resources=True, cache_prompts=True, include_instructions=False,
    include_return_schema=None, sampling_model=None, sampling_handler=None,
    elicitation_handler=None, log_handler=None, log_level=None,
    progress_handler=None, message_handler=None, client_info=None,
    init_timeout=..., read_timeout=..., roots=None,
    auth=None, verify=None, headers=None, http_client=None,
)
```

`prefer_tasks` (default `True`) wraps tool calls as durable background tasks per SEP-1686 when
the server declares `taskSupport='optional'` (tools with `taskSupport='required'` always run as
tasks regardless). `direct_call_tool(name, args, *, metadata, use_task)` invokes a tool outside
any agent run. `process_tool_call(ctx, call_tool, name, tool_args) -> ToolResult` intercepts every
call for metadata injection, audit logging, or selective retry.

```python
import asyncio
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset

toolset = MCPToolset('http://localhost:8000/mcp', prefer_tasks=False)
agent = Agent('openai:gpt-4o', toolsets=[toolset])

async def main():
    async with agent:
        result = await agent.run('List available tools.')
        print(result.output)

asyncio.run(main())
```

`agent.set_mcp_sampling_model()` wires the agent's own model into every attached `MCPToolset` for
server-driven sampling. `load_mcp_toolsets(config_path)` reads a Claude-Desktop-style
`mcpServers` JSON config and returns one `PrefixedToolset(MCPToolset(...))` per server, with
`${VAR}` / `${VAR:-default}` env-var expansion.

#### `common_tools` — ready-made search & fetch factories

**Module:** `pydantic_ai.common_tools`. Lightweight factories returning `Tool`/`FunctionToolset`
objects, each requiring its optional dependency group.

```python
duckduckgo_search_tool(duckduckgo_client=None, max_results=None) -> Tool   # pip install "pydantic-ai-slim[duckduckgo]"
tavily_search_tool(api_key, *, client=None, max_results=None, search_depth=..., topic=..., time_range=..., include_domains=..., exclude_domains=...) -> Tool
ExaToolset(api_key, *, num_results=5, max_characters=None, include_search=True, include_find_similar=True,
           include_get_contents=True, include_answer=True) -> FunctionToolset  # 4 tools over one shared AsyncExa client
web_fetch_tool(*, max_content_length=50_000, allow_local_urls=False, timeout=30,
               allowed_domains=None, blocked_domains=None, headers=None) -> Tool
image_generation_tool(model, native_tool: ImageGenerationTool, *, instructions=...) -> Tool
```

`duckduckgo_search_tool` wraps `DDGS` (from the `ddgs` package) via `anyio.to_thread.run_sync`,
returning `list[DuckDuckGoResult]` (`title`, `href`, `body`). `tavily_search_tool` freezes any of
the keyword params you supply via `functools.partial` **and** strips them from `__signature__`, so
the LLM never sees (or can override) developer-fixed params; unset ones stay LLM-controlled.

```python
from pydantic_ai import Agent
from pydantic_ai.common_tools.duckduckgo import duckduckgo_search_tool
from pydantic_ai.common_tools.tavily import tavily_search_tool
from pydantic_ai.common_tools.web_fetch import web_fetch_tool

agent = Agent(
    'openai:gpt-4o',
    tools=[
        duckduckgo_search_tool(max_results=5),
        # search_depth/topic frozen and hidden from the LLM schema; time_range stays LLM-controlled
        tavily_search_tool(api_key='tvly-...', max_results=5, search_depth='advanced', topic='news'),
        web_fetch_tool(max_content_length=5000),
    ],
    instructions='Search first, then fetch the most promising URL.',
)
```

`image_generation_tool` (from `pydantic_ai.common_tools.image_generation`) spins up a **subagent**
on an image-capable model so a text-only primary agent can still generate images:

```python
from pydantic_ai.common_tools.image_generation import image_generation_tool
from pydantic_ai import ImageGenerationTool

agent = Agent('openai:gpt-4o-mini', tools=[
    image_generation_tool(model='openai-responses:gpt-5.4', native_tool=ImageGenerationTool(quality='high'))
])
```

---

### Streaming & Events

#### `AgentStream` — full streaming API

Returned by `agent.run_stream(...)`.

```python
class AgentStream(Generic[AgentDepsT, OutputDataT]):
    async def stream_output(self, debounce_by=0.1) -> AsyncIterator[OutputDataT]: ...
    async def stream_response(self, debounce_by=0.1) -> AsyncIterator[ModelResponse]: ...
    async def stream_text(self, delta=False, debounce_by=0.1) -> AsyncIterator[str]: ...
    async def cancel(self) -> None: ...
    async def drain(self) -> None: ...
    async def validate_response_output(self, response, allow_partial=False) -> OutputDataT: ...
    async def get_output(self) -> OutputDataT: ...
    response: ModelResponse
    usage: RunUsage
    run_id: str; conversation_id: str; metadata: dict | None; cancelled: bool
```

```python
import asyncio
from pydantic_ai import Agent

agent = Agent('openai:gpt-4o')

async def main():
    async with agent.run_stream('Explain recursion in one paragraph.') as stream:
        async for chunk in stream.stream_text(delta=True):
            print(chunk, end='', flush=True)
        print(f'\ntokens: {stream.usage.total_tokens}')

asyncio.run(main())
```

#### `StreamedRunResult` + `StreamedRunResultSync`

`StreamedRunResult` is the high-level object `agent.run_stream()` yields (wrapping `AgentStream`
with message-history helpers). `StreamedRunResultSync` (from `agent.run_stream_sync()`) is the
same API with `_sync` suffixes, run on a background thread via `anyio.from_thread`.

```python
class StreamedRunResult(Generic[AgentDepsT, OutputDataT]):
    is_complete: bool
    stream_output(delta=False, debounce_by=0.1); stream_text(delta=False, debounce_by=0.1)
    stream_response(debounce_by=0.1); get_output(); cancel(); cancelled
    usage(); all_messages(); new_messages(); all_messages_json(); new_messages_json()
```

```python
with agent.run_stream_sync('Summarise this text') as result:
    for chunk in result.stream_text_sync(delta=True):
        print(chunk, end='', flush=True)
    print(result.usage())
```

#### `AgentEventStream` + `AgentRunResultEvent`

`agent.run_stream_events()` returns an `AgentEventStream` context manager — **always** use it via
`async with` (bare `async for` iteration without the context manager is deprecated and will be
removed). `AgentRunResultEvent` is always the final event, carrying the completed
`AgentRunResult`.

```python
from pydantic_ai.run import AgentRunResultEvent

async def full_loop(agent):
    async with agent.run_stream_events('Name the planets.') as stream:
        async for event in stream:
            if isinstance(event, AgentRunResultEvent):
                print('final:', event.result.output)
```

#### `PartStartEvent` + `PartDeltaEvent` + `PartEndEvent` + `FinalResultEvent`

The discriminated union `ModelResponseStreamEvent` flowing from `StreamedResponse._get_event_iterator()`. Each carries an `index` identifying which part in the running `parts` list is updated, plus an `event_kind` literal for efficient discrimination. `FinalResultEvent` is fired once per run step when the model's response first matches the output schema, ahead of actual validation; `FinalResult` (the non-event dataclass) wraps the validated output plus the tool name/call ID that produced it (both `None` for plain-text output).

```python
FinalResult(output: OutputDataT, tool_name: str | None, tool_call_id: str | None)
FinalResultEvent(tool_name: str | None, tool_call_id: str | None, event_kind='final_result')
```

```python
async def main() -> None:
    async with model_request_stream('openai:gpt-4o-mini', messages) as stream:
        async for event in stream:
            if isinstance(event, PartStartEvent):
                print(f'[start idx={event.index}] {event.part.part_kind}')
            elif isinstance(event, PartDeltaEvent):
                pass
            elif isinstance(event, FinalResultEvent):
                print(f'[result] tool_name={event.tool_name!r}')
```

#### `TextPartDelta` + `ThinkingPartDelta` + `ToolCallPartDelta`

Incremental delta payloads inside `PartDeltaEvent.delta`. `TextPartDelta.content_delta` appends; `ThinkingPartDelta.signature_delta` **replaces** (never appends); `ToolCallPartDelta.args_delta` appends when a `str`, merges when a `dict`.

```python
buf: dict[int, list[str]] = {}
async for event in stream:
    if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
        buf.setdefault(event.index, []).append(event.delta.content_delta)
```

#### `HandleResponseEvent` + `ModelResponseStreamEvent`

Discriminated-union type aliases (not classes) used for typed (de)serialisation of event streams.
`HandleResponseEvent` covers everything `CallToolsNode.stream()` yields (function/output/native
tool call+result events); `ModelResponseStreamEvent` covers `PartStartEvent`/`PartDeltaEvent`/
`PartEndEvent`/`FinalResultEvent` from model-response streaming.

```python
HandleResponseEvent = FunctionToolCallEvent | FunctionToolResultEvent | OutputToolCallEvent | OutputToolResultEvent  # discriminated on event_kind
ModelResponseStreamEvent = PartStartEvent | PartDeltaEvent | PartEndEvent | FinalResultEvent
```

#### `ToolCallEvent` / `ToolResultEvent` family

Base classes for the four concrete tool-interaction events.

```python
ToolCallEvent(part: ToolCallPart, args_valid: bool | None, event_kind: str)
ToolResultEvent(part: ToolReturnPart | RetryPromptPart, event_kind: str)
# concretes: FunctionToolCallEvent, FunctionToolResultEvent, OutputToolCallEvent, OutputToolResultEvent
```

```python
from pydantic_ai.messages import FunctionToolCallEvent, FunctionToolResultEvent

async def watch(agent, prompt):
    async with agent.run_stream_events(prompt) as stream:
        async for event in stream:
            if isinstance(event, FunctionToolCallEvent):
                print('call', event.part.tool_name, 'valid=', event.args_valid)
            elif isinstance(event, FunctionToolResultEvent):
                print('result', event.part.content)
```

#### `BuiltinToolCallEvent` / `BuiltinToolResultEvent` — deprecated, migrate now

Deprecated in favour of the richer `PartStartEvent`/`PartEndEvent` pathway (which supports
streaming deltas for native tool call arguments, unlike the old start/end-only events).

```python
# Before (deprecated)              # After
BuiltinToolCallEvent                PartStartEvent + isinstance(event.part, NativeToolCallPart)
BuiltinToolResultEvent              PartEndEvent   + isinstance(event.part, NativeToolReturnPart)
```

#### `ModelResponsePartsManager`

The streaming aggregator used internally by every `StreamedResponse` subclass. When writing a custom provider, use `self._parts_manager` (a cached property on `StreamedResponse`) inside `_get_event_iterator()` — never instantiate a local one, since `StreamedResponse.__aiter__` synthesises `PartEndEvent`s by reading that same instance. `handle_text_delta(vendor_part_id, content)` and `handle_tool_call_delta(...)` return correctly-typed events with dedup and vendor-ID tracking; `get_parts()` returns only fully-formed parts (no in-flight deltas).

```python
class ModelResponsePartsManager:
    def handle_text_delta(self, *, vendor_part_id: str, content: str) -> Iterator[ModelResponseStreamEvent]: ...
    def handle_tool_call_delta(self, *, vendor_part_id, tool_name=None, args=None, tool_call_id=None) -> ModelResponseStreamEvent | None: ...
    def get_parts(self) -> list[ModelResponsePart]: ...
```

```python
class EchoStreamedResponse(StreamedResponse):
    async def _get_event_iterator(self):
        for word in self._words:
            for event in self._parts_manager.handle_text_delta(vendor_part_id='text', content=word + ' '):
                yield event
```

#### `DeltaToolCall` + `DeltaThinkingPart` + delta type aliases + `FunctionStreamedResponse`

**Module:** `pydantic_ai.models.function`. The chunk types a `FunctionModel(stream_function=...)`
yields, and the `StreamedResponse` implementation that dispatches them through
`ModelResponsePartsManager`. You must yield all-`str`, all-`DeltaToolCalls`, or all-
`DeltaThinkingCalls` within one stream — mixing types is not supported.

```python
DeltaToolCalls = dict[int, DeltaToolCall]          # DeltaToolCall(name=, json_args=, tool_call_id=)
DeltaThinkingCalls = dict[int, DeltaThinkingPart]  # DeltaThinkingPart(content=, signature=)
```

```python
from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel, AgentInfo

async def word_stream(messages, info: AgentInfo):
    for word in 'The answer is forty two'.split():
        yield word + ' '

agent = Agent(FunctionModel(stream_function=word_stream))
```

#### `ProcessEventStream` capability

Intercepts the `AgentStreamEvent` sequence during a streaming run — or automatically enables streaming inside `agent.run()` when registered, no `run_stream()` needed. Two handler forms: an **observer** (`async def handler(ctx, stream) -> None`, receives a tee'd copy, events pass through unchanged; a slow observer back-pressures) and a **processor** (an async generator whose yielded events *replace* the stream for downstream consumers — can drop, transform, or inject events; dropping a `FinalResultEvent` delays result delivery).

```python
@dataclass
class ProcessEventStream(AbstractCapability[AgentDepsT]):
    handler: EventStreamHandlerFunc[AgentDepsT] | EventStreamProcessorFunc[AgentDepsT]
```

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessEventStream
from pydantic_ai.messages import PartStartEvent, PartDeltaEvent, PartEndEvent, ThinkingPart

async def hide_thinking(ctx, stream):
    skip = None
    async for event in stream:
        if isinstance(event, PartStartEvent) and isinstance(event.part, ThinkingPart):
            skip = event.index
            continue
        if skip is not None and isinstance(event, (PartDeltaEvent, PartEndEvent)) and event.index == skip:
            continue
        yield event

agent = Agent('anthropic:claude-sonnet-4-6', capabilities=[ProcessEventStream(hide_thinking)])
```

#### `EventStreamHandler` + `EventStreamProcessor` (type aliases)

`EventStreamHandler` is `Callable[[RunContext, AsyncIterable[AgentStreamEvent]], Awaitable[None]]` — a terminal sink used via `agent.run(..., event_stream_handler=...)`. `EventStreamProcessor` is `Callable[[RunContext, AsyncIterable[AgentStreamEvent]], AsyncIterator[AgentStreamEvent]]` — a pass-through transformer used by `ProcessEventStream`.

```python
from pydantic_ai import Agent
from pydantic_ai.tools import RunContext
from pydantic_ai.messages import AgentStreamEvent, PartDeltaEvent, TextPartDelta

async def my_stream_handler(ctx: RunContext[None], events) -> None:
    async for event in events:
        if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
            print(event.delta.content_delta, end='', flush=True)

agent = Agent('openai:gpt-4o-mini')
result = await agent.run('Count to five', event_stream_handler=my_stream_handler)
```

#### `split_content_into_text_and_thinking`

Some providers (DeepSeek, Qwen, older Ollama) embed thinking inside `<think>...</think>` tags in the plain text stream rather than as a separate part. This function splits a raw string into an alternating `list[ThinkingPart | TextPart]` using the provider's `thinking_tags` tuple.

```python
def split_content_into_text_and_thinking(content: str, thinking_tags: tuple[str, str]) -> list[ThinkingPart | TextPart]: ...
```

```python
from pydantic_ai._thinking_part import split_content_into_text_and_thinking

parts = split_content_into_text_and_thinking(
    "<think>Working through it...</think>The answer is 42.", thinking_tags=('<think>', '</think>')
)
```

---

### Structured Output

#### `ToolOutput` / `NativeOutput` / `PromptedOutput` / `TextOutput` / `StructuredDict`

Explicit, per-type control over how structured output is delivered. `ToolOutput` — model emits a structured "output tool" call (best for unions/non-native models). `NativeOutput` — provider's native JSON-schema/response-format mode; `template=False` suppresses schema-prompt injection when the provider already handles it natively. `PromptedOutput` — injects the schema as prompt text and parses the reply; `template` accepts a custom `'{schema}'`-style string. `TextOutput(fn)` — plain text passed to a Python parser function, which may optionally take `RunContext` as its first argument and may be async. `StructuredDict` is a factory (not a class) returning a `dict[str, Any]` subclass with a JSON Schema attached — validated structured output without defining a Pydantic `BaseModel`; `name`/`description` fall back to the schema's `title`/`description`.

```python
ToolOutput(type_, *, name=None, description=None, max_retries=None, strict=None)
NativeOutput(outputs, *, name=None, description=None, strict=None, template=None)
PromptedOutput(outputs, *, name=None, description=None, template=None)
TextOutput(output: TextOutputFunc)
StructuredDict(json_schema, name=None, description=None) -> type[dict[str, Any]]
```

```python
from pydantic import BaseModel
from pydantic_ai import Agent, ToolOutput, StructuredDict

class Fruit(BaseModel):
    name: str; color: str

person_schema = {'type': 'object', 'properties': {'name': {'type': 'string'}, 'age': {'type': 'integer'}}, 'required': ['name', 'age']}
PersonDict = StructuredDict(person_schema, name='Person')

agent = Agent('openai:gpt-4o', output_type=[ToolOutput(Fruit), PersonDict])
result = agent.run_sync('Describe a banana.')
```

#### `OutputObjectDefinition` + `OutputSchema` + `OutputValidator`

`OutputObjectDefinition` is the internal normalised schema record (`json_schema`, `name`,
`description`, `strict`) produced from `ToolOutput`/`NativeOutput`/`PromptedOutput`/
`StructuredDict`; surfaced as `OutputContext.object_def` in output hooks. `OutputSchema.build(type)`
is the factory resolving `output_type` into one of `TextOutputSchema`/`ToolOutputSchema`/
`NativeOutputSchema`/`PromptedOutputSchema`/`ImageOutputSchema`/`MultiOutputSchema`.
`OutputValidator` wraps a user validator function, dispatching sync/async and with/without
`RunContext` transparently.

```python
from pydantic_ai.result import OutputSchema
from pydantic import BaseModel

class MyResult(BaseModel):
    answer: str

schema = OutputSchema.build(MyResult)
print(schema.mode, schema.object_def.name)   # 'tool' 'MyResult'
```

#### `OutputContext`

Read-only context passed to the four output lifecycle hooks
(`before_output_validate`/`after_output_validate`/`before_output_process`/`after_output_process`).

```python
class OutputContext:
    mode: OutputMode
    output_type: type[Any] | None
    object_def: OutputObjectDefinition | None
    has_function: bool
    tool_call: ToolCallPart | None
    tool_def: ToolDefinition | None
    allows_text: bool; allows_image: bool; allows_deferred_tools: bool
```

```python
from pydantic_ai.capabilities import Hooks

hooks = Hooks()

@hooks.on.before_output_validate
def log_mode(ctx, *, output_context, output):
    print(f'mode={output_context.mode} type={output_context.output_type}')
    return output
```

---

### Messages & Multimodal Content

#### `ModelRequest` + `ModelResponse` — wire-format anatomy

The two halves of `ModelMessage`.

```python
class ModelRequest:
    parts: Sequence[SystemPromptPart | UserPromptPart | ToolReturnPart | NativeToolReturnPart | RetryPromptPart | InstructionPart]
    timestamp: datetime | None; instructions: str | None
    run_id: str | None; conversation_id: str | None; metadata: dict | None   # never sent to the LLM
    kind: Literal['request']

class ModelResponse:
    parts: Sequence[TextPart | ThinkingPart | ToolCallPart | NativeToolCallPart | FilePart | CompactionPart]
    usage: RequestUsage; model_name: str | None; timestamp: datetime
    finish_reason: Literal['stop','tool_calls','length','content_filter','other'] | None
    state: Literal['complete','incomplete','interrupted'] | None
    run_id: str | None; conversation_id: str | None
    kind: Literal['response']
```

```python
from pydantic_ai import ModelMessagesTypeAdapter

blob = ModelMessagesTypeAdapter.dump_json(messages)
restored = ModelMessagesTypeAdapter.validate_json(blob)
```

#### `SystemPromptPart` + `UserPromptPart` + `RetryPromptPart`

The three request-side message parts. `UserPromptPart.content` accepts a heterogeneous sequence
of `UserContent` (str, `TextContent`, `ImageUrl`, `AudioUrl`, `VideoUrl`, `DocumentUrl`,
`BinaryContent`, `UploadedFile`, `CachePoint`). `RetryPromptPart.model_response()` formats the
retry-feedback string actually sent to the model.

```python
SystemPromptPart(content: str, dynamic_ref: str | None = None)
UserPromptPart(content: str | Sequence[UserContent])
RetryPromptPart(content: list[ErrorDetails] | str, tool_name: str | None = None)
```

```python
from pydantic_ai.messages import UserPromptPart, ImageUrl, CachePoint

part = UserPromptPart(content=['Long static context...', CachePoint(), 'Dynamic question?'])
```

#### `BaseToolCallPart` / `ToolCallPart` / `NativeToolCallPart` (+ `ToolCallPartDelta`)

Call-part family. `args_as_dict(raise_if_invalid=False)` gracefully returns
`{'INVALID_JSON': ...}` on malformed streamed JSON unless `raise_if_invalid=True`.
`narrow_type(part, tool_kind=...)` promotes a raw `NativeToolCallPart` to a typed subclass (e.g.
`NativeToolSearchCallPart`) for the tool-search protocol.

```python
BaseToolCallPart: tool_name, args, tool_call_id, tool_kind, id, provider_name, provider_details
  .args_as_dict(raise_if_invalid=False); .args_as_json_str(); .has_content()
ToolCallPart(BaseToolCallPart)          # part_kind='tool-call'
NativeToolCallPart(BaseToolCallPart)    # part_kind='builtin-tool-call'
```

#### `BaseToolReturnPart` / `ToolReturnPart` / `NativeToolReturnPart`

Return-part family. `outcome: Literal['success', 'failed', 'denied']` tracks HITL/error state.
The top-level `ToolReturn` message-content dataclass (the thing tools *return*, distinct
from `ToolReturnPart`) has a `tools` field that lets a tool's return value append additional
tool-availability deltas (`ToolAvailabilityDeltaEvent`/`Part`) to the run.

```python
BaseToolReturnPart: tool_name, content, tool_call_id, tool_kind, metadata, outcome
  .model_response_str(); .model_response_object(); .content_items(mode='raw'|'str'|'jsonable'); .files
ToolReturnPart(BaseToolReturnPart)          # part_kind='tool-return'
NativeToolReturnPart(BaseToolReturnPart)    # provider_name, provider_details; part_kind='builtin-tool-return'

ToolReturn(return_value, content=None, metadata=None, tools=None)   # what a tool function returns
```

```python
from pydantic_ai.messages import ToolReturnPart

denied = ToolReturnPart(tool_name='delete_file', content='Denied by operator.', outcome='denied')
```

#### `TextContent`

A string that carries app-only `metadata` never sent to the model — valid inside
`UserPromptPart.content`.

```python
TextContent(content: str, metadata: Any = None)
```

```python
from pydantic_ai.messages import TextContent

part = [TextContent(content='Pydantic AI was released in 2024.', metadata={'source_url': 'https://docs.ai/'})]
```

#### `FilePart` + `BinaryImage`

`FilePart` is a **model-response** part carrying a generated file (e.g. an image). `BinaryImage`
is a `BinaryContent` subclass that validates `media_type.startswith('image/')` at construction.

```python
FilePart(content: BinaryContent, id=None, provider_name=None, provider_details=None)
BinaryImage(data: bytes, media_type: str, ...)   # __post_init__ raises if not image/*
```

```python
from pydantic_ai.messages import FilePart

images = [p for msg in result.all_messages() for p in getattr(msg, 'parts', []) if isinstance(p, FilePart)]
```

#### `BinaryContent` + `FileUrl` family (`ImageUrl`/`AudioUrl`/`VideoUrl`/`DocumentUrl`)

Two parallel multimodal systems: raw bytes (`BinaryContent`) vs. URL references (`FileUrl`
subclasses). `force_download: bool | Literal['allow-local']` controls SSRF-safe fetching for URLs
(`False`=send URL directly where supported; `True`=always download with full SSRF guard;
`'allow-local'`=download, allow private IPs, still block cloud metadata).

```python
BinaryContent(data: bytes, media_type: str, vendor_metadata=None, identifier=None)
ImageUrl(url, *, force_download=False, vendor_metadata=None, media_type=None, identifier=None)
# AudioUrl, VideoUrl, DocumentUrl share the same shape
```

```python
from pydantic_ai import Agent, BinaryContent
from pathlib import Path

agent = Agent('anthropic:claude-sonnet-4-5')
result = agent.run_sync([
    "What's wrong in this screenshot?",
    BinaryContent(data=Path('screenshot.png').read_bytes(), media_type='image/png'),
])
```

#### `UploadedFile`

A durable reference to a file already uploaded to a provider (skips re-sending bytes every
request). Supported providers: OpenAI, OpenAI Responses, Anthropic, Bedrock (`s3://`), Google
(Gemini Files API URI or `gs://`), xAI.

```python
UploadedFile(file_id: str, provider_name: str, *, media_type=None, vendor_metadata=None, identifier=None)
```

```python
from pydantic_ai import Agent, UploadedFile

agent = Agent('openai:gpt-4o')
result = agent.run_sync([
    'Summarise the key financials.',
    UploadedFile(file_id='file-abc123', provider_name='openai', media_type='application/pdf'),
])
```

#### `CachePoint` + `CompactionPart` + `ToolAvailabilityDeltaPart`

`CachePoint(ttl='5m' | '1h')` marks a prompt-cache boundary inside `UserPromptPart.content`
(Anthropic, Bedrock Converse, OpenAI GPT-5.6+; Anthropic/Bedrock support `'1h'`, OpenAI always
uses `'5m'`; silently dropped elsewhere, so the same message-construction code works everywhere).
`CompactionPart` carries a provider-produced conversation summary (Anthropic: readable `content`;
OpenAI: opaque `provider_details`) that must be round-tripped verbatim on the next request — check
`.has_content()` before displaying it. `ToolAvailabilityDeltaPart` is a streaming part recording
that new tools became available mid-stream — emitted internally when `DeferredLoadingToolset`
tools are discovered and injected into a live request.

```python
CachePoint(ttl: Literal['5m', '1h'] = '5m')
CompactionPart(content: str | None, id=None, provider_name=None, provider_details=None)
```

```python
from pydantic_ai import Agent, CachePoint

agent = Agent('anthropic:claude-sonnet-4-5')
result = agent.run_sync(['Document:\n' + long_doc, CachePoint(ttl='1h'), 'Summarise it.'])
```

#### `format_as_xml`

Converts Python objects (dataclasses, `BaseModel`, dicts, lists) into an XML string LLMs often
parse more reliably than JSON. `root_tag=None` produces rootless sibling elements;
`include_field_info='once'` includes title/description XML attributes only on a field's first
occurrence in a list (saves tokens); `indent=None` removes all whitespace.

```python
format_as_xml(obj, root_tag=None, item_tag='item', none_str='null', indent='  ',
               include_field_info: Literal['once'] | bool = False) -> str
```

```python
from pydantic_ai import format_as_xml

print(format_as_xml({'name': 'Alice', 'age': 30}, root_tag='user'))
```

#### `ModelRequestContext`

The mutable dataclass passed to `before_model_request`/`after_model_request` capability hooks —
mutate its fields to change what the model actually sees.

```python
class ModelRequestContext:
    model: Model
    messages: list[ModelMessage]
    model_settings: ModelSettings | None
    model_request_parameters: ModelRequestParameters
```

```python
from pydantic_ai.capabilities import Hooks

hooks = Hooks()

@hooks.on.before_model_request
async def log_request(ctx, request_context):
    print(f'{len(request_context.messages)} messages, model={request_context.model.model_name}')
    return request_context
```

#### `ModelRequestParameters`

The wire-format object handed to `Model.request()`/`request_stream()` — everything about response
shape and available tools.

```python
class ModelRequestParameters:
    function_tools: list[ToolDefinition]
    native_tools: list[AbstractNativeTool]
    output_mode: Literal['text','tool','native','prompted','auto']
    output_object: OutputObjectDefinition | None
    output_tools: list[ToolDefinition]
    allow_text_output: bool; allow_image_output: bool
    instruction_parts: list[InstructionPart] | None
    thinking: ThinkingLevel | None
    tool_defs: dict[str, ToolDefinition]   # @cached_property, merges function_tools + output_tools
```

#### `InstructionPart` — cacheable instruction composition

Represents one block of instruction text, tagged `dynamic` (from `@agent.instructions`, `TemplateStr`, or toolset `get_instructions()`) or static (from a literal string). Provider prompt-caching relies on this distinction: static parts are always cached, dynamic ones aren't. `InstructionPart.sorted()` places static parts first (maximises cache-hit rate); `.join()` concatenates with a double newline.

```python
@dataclass
class InstructionPart:
    content: str
    dynamic: bool = False
    @staticmethod
    def sorted(parts: list[InstructionPart]) -> list[InstructionPart]: ...
    @staticmethod
    def join(parts: list[InstructionPart]) -> str | None: ...
```

```python
from pydantic_ai.messages import InstructionPart

parts = [InstructionPart('Current time: 14:30', dynamic=True), InstructionPart('Be helpful.', dynamic=False)]
optimised = InstructionPart.sorted(parts)   # static first, dynamic last
```

#### `AgentInstructions` pipeline

Four-stage pipeline in `pydantic_ai._instructions`: `normalize_instructions()` (`None → []`, `str`/callable → `[it]`, sequence → `list(it)`) → `prepare_instructions()` (wraps each callable, including `TemplateStr`, in a `SystemPromptRunner`) → `resolve_instructions(instructions, run_context)` (awaits runners, returns `list[str]`). Separately, `normalize_toolset_instructions()` turns a toolset's `get_instructions()` return value into `list[InstructionPart]`, dropping whitespace-only content.

```python
def normalize_instructions(instructions) -> list[str | SystemPromptFunc]: ...
def prepare_instructions(instructions) -> list[str | SystemPromptRunner]: ...
async def resolve_instructions(instructions, run_context) -> list[str]: ...
```

```python
from pydantic_ai._instructions import normalize_toolset_instructions
from pydantic_ai.messages import InstructionPart

parts = normalize_toolset_instructions("Use the search tool for factual questions.")
print(parts[0].dynamic)   # True
```

#### Anthropic mid-conversation `SystemPromptPart` via `ctx.enqueue`

Anthropic supports native mid-conversation system messages — instructions inserted into the conversation rather than at the initial system-prompt position, preserving the cached prefix while dynamically adjusting behaviour. Enqueue a `SystemPromptPart` with `ctx.enqueue(...)`; on Anthropic it's routed through the native system-in-conversation format (placement auto-adjusted to sit between a user turn and the model's reply), and on other providers it falls back to a tagged user-channel message (`<system>...</system>`) — application code is identical either way.

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import SystemPromptPart

agent = Agent('anthropic:claude-opus-4-8', system_prompt='You are a senior code reviewer.')

@agent.tool
def require_type_annotations(ctx: RunContext[None]) -> str:
    ctx.enqueue(SystemPromptPart(content='All suggestions MUST include type annotations.'))
    return 'Rule added.'
```

#### Tool-search wire protocol — `ToolSearchCallPart`/`NativeToolSearchCallPart` + `ToolSearchReturnPart`/`NativeToolSearchReturnPart`

When `ToolSearch` / `DeferredLoadingToolset` is active, the model issues a search query before
picking a tool. Two paths exist: **native** (Anthropic BM25/regex, OpenAI server-side) uses
`NativeToolSearchCallPart`/`NativeToolSearchReturnPart`; **local** (everyone else, or a custom
callable) uses `ToolSearchCallPart`/`ToolSearchReturnPart`. Cross-path detection: check
`part.tool_kind == 'tool-search'` (do **not** branch on `tool_name`, which differs:
`'search_tools'` local vs. `'tool_search'` native).

```python
class ToolSearchArgs(TypedDict):
    queries: list[str]
class ToolSearchMatch(TypedDict):
    name: str; description: str | None
class ToolSearchReturnContent(TypedDict):
    discovered_tools: list[ToolSearchMatch]
    message: NotRequired[str]
```

```python
def discovered_tools(messages) -> list:
    out = []
    for msg in messages:
        for part in getattr(msg, 'parts', []):
            if getattr(part, 'tool_kind', None) == 'tool-search' and hasattr(part, 'content'):
                out.extend(part.content.get('discovered_tools', []))
    return out
```

#### `LoadCapabilityCallPart` + `LoadCapabilityReturnPart`

The wire protocol for `defer_loading=True` capabilities — the model calls the hidden
`load_capability` tool before a deferred capability's tools/instructions become visible.
Cross-path discriminator: `tool_kind == 'capability-load'`. See `DeferredCapabilityLoader` in
Capabilities & Extensibility for the loader mechanics that produce these parts.

```python
LoadCapabilityCallPart(ToolCallPart):    # tool_name='load_capability', tool_kind='capability-load'
    capability_id: str | None            # @property, parsed from args
LoadCapabilityReturnPart(ToolReturnPart):
    instructions: str | None             # @property, from content.get('instructions')
```

#### Vercel AI SDK wire types — request `UIMessage` parts + response SSE chunks

Request-side (`pydantic_ai.ui.vercel_ai.request_types`): `TextUIPart`, `ReasoningUIPart`, `FileUIPart`, `ToolApprovalRespondedPart`/`ToolApprovalResponded`, `UIMessage`, `SubmitMessage` — all `CamelBaseModel`s with `alias_generator=to_camel` (Python `snake_case` ↔ wire `camelCase`). Assistant messages may carry reasoning parts; user messages may not. Response-side (`...response_types`): `StartChunk`, `TextStartChunk`/`TextDeltaChunk`/`TextEndChunk`, `ToolInputStartChunk`/`ToolInputDeltaChunk`/`ToolInputAvailableChunk`, `ToolApprovalRequestChunk` (SDK v6+ HITL), `FinishChunk`, `DoneChunk` — each has `.encode(sdk_version)` which strips fields unsupported below that version.

```python
from pydantic_ai.ui.vercel_ai.request_types import UIMessage, TextUIPart, SubmitMessage
from pydantic_ai.ui.vercel_ai.response_types import StartChunk, TextStartChunk, TextDeltaChunk, DoneChunk

msg = UIMessage(id='m1', role='user', parts=[TextUIPart(type='text', text='Hello')])
submit = SubmitMessage(id='req-1', messages=[msg])

chunks = [StartChunk(message_id='msg-1'), TextStartChunk(id='t0'), TextDeltaChunk(id='t0', delta='Hi'), DoneChunk()]
for c in chunks:
    print(f'data: {c.encode(sdk_version=6)}\n')
```

#### AG-UI multimodal conversion

`pydantic_ai.ui.ag_ui._multimodal` bridges pydantic-ai's `ImageUrl`/`AudioUrl`/`VideoUrl`/`DocumentUrl`/`BinaryContent` with AG-UI's typed input classes via two dispatch tables: `_URL_TYPE_MAP` (exact type → AG-UI class) for URL-based media, and `_MEDIA_PREFIX_TO_CONTENT` (media-type prefix → AG-UI class, default `DocumentInputContent`) for binary data. `multimodal_input_to_content()` round-trips an AG-UI part back to a pydantic-ai type.

```python
def media_url_to_multimodal(item: ImageUrl | AudioUrl | VideoUrl | DocumentUrl): ...
def binary_to_multimodal(item: BinaryContent): ...
def multimodal_input_to_content(part) -> ImageUrl | AudioUrl | VideoUrl | DocumentUrl | BinaryContent: ...
```

```python
from pydantic_ai.messages import ImageUrl
from pydantic_ai.ui.ag_ui._multimodal import media_url_to_multimodal

ag_img = media_url_to_multimodal(ImageUrl(url='https://example.com/photo.jpg', media_type='image/jpeg'))
print(type(ag_img).__name__)   # ImageInputContent
```

#### Multimodal type system — media aliases, `ForceDownloadMode`, `ProviderDetailsDelta`

`AudioMediaType`/`ImageMediaType`/`DocumentMediaType`/`VideoMediaType` (full MIME strings) and
their `*Format` shorthand siblings (`'jpeg'`, `'mp3'`, `'pdf'`, ...) are `Literal` type aliases used
throughout tool schemas and `FileUrl` subclasses. `ForceDownloadMode = bool | Literal['allow-local']`
(see `FileUrl` above). `ProviderDetailsDelta = dict | Callable[[dict | None], dict] | None` updates
a return part's `provider_details` without wholesale replacement.

---

### Concurrency, Usage & Limits

#### `UsageBase` / `RunUsage` / `RequestUsage` / `UsageLimits`

`UsageBase` fields (shared by `RequestUsage` per-request and `RunUsage` accumulated): `input_tokens`, `cache_write_tokens`, `cache_read_tokens`, `output_tokens`, `input_audio_tokens`, `cache_audio_read_tokens`, `output_audio_tokens`, `details`. `RunUsage` adds `requests`, `tool_calls`, and a top-level **`cost: Decimal | None`** field (best-effort USD cost summed across requests via genai-prices; `None` when the provider exposes no pricing, distinct from `Decimal('0')` for a genuinely free run). `RunUsage.__add__`/`.incr()` accumulate across runs; `.opentelemetry_attributes()` returns GenAI-semconv span attributes. `UsageLimits` fields: `cost_limit: Decimal | None`, `request_limit: int | None = 50` (default is 50, not unlimited), `tool_calls_limit`, `input_tokens_limit`, `output_tokens_limit`, `total_tokens_limit`, `per_request_input_tokens_limit` (per-call cap independent of the cumulative `input_tokens_limit` — useful with prompt caching, where a large cached prefix still counts), `count_tokens_before_request: bool = False` (preflight token-count call before dispatch; enforces both token limits ahead of time on Anthropic/Google/Bedrock/OpenAI Responses). Note: `response_tokens_limit` seen in some old examples was never a real field name — the correct field is `output_tokens_limit`.

```python
@dataclass(kw_only=True)
class UsageLimits:
    cost_limit: Decimal | None = None
    request_limit: int | None = 50
    tool_calls_limit: int | None = None
    input_tokens_limit: int | None = None
    output_tokens_limit: int | None = None
    total_tokens_limit: int | None = None
    per_request_input_tokens_limit: int | None = None
    count_tokens_before_request: bool = False

@dataclass(kw_only=True)
class RunUsage(UsageBase):
    requests: int = 0
    tool_calls: int = 0
    cost: Decimal | None = None
```

```python
from pydantic_ai import Agent
from pydantic_ai.usage import UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded

agent = Agent('anthropic:claude-sonnet-4-5')

try:
    result = agent.run_sync(
        'Summarise this 50-page document...',
        usage_limits=UsageLimits(per_request_input_tokens_limit=8_000, cost_limit=Decimal('0.05')),
    )
except UsageLimitExceeded as e:
    print(f'Budget exceeded: {e}')
```

`RunUsage` accumulates across an entire run; `RequestUsage` is a single API call's usage and
implements `genai_prices.types.AbstractUsage` for cost calculation. Pass one `RunUsage` instance
into successive `agent.run(usage=...)` calls to keep a running session total:

```python
from pydantic_ai import Agent, RunUsage

agent = Agent('openai:gpt-4o')
shared = RunUsage()
for prompt in ['One', 'Two', 'Three']:
    agent.run_sync(prompt, usage=shared)
print(shared.total_tokens)
```

#### `ConcurrencyLimiter` + `AbstractConcurrencyLimiter` + `ConcurrencyLimit` + `ConcurrencyLimitedModel`

Two layers: agent-level (`Agent(max_concurrency=...)`, caps simultaneous *runs*; acquired at run start, released at run end) and model-level (`limit_model_concurrency(model, limiter)` / `ConcurrencyLimitedModel(model, limiter=...)`, caps simultaneous *HTTP requests* to one model endpoint — the two compose, since a shared limiter can back both). `ConcurrencyLimiter` wraps `anyio.CapacityLimiter`; `max_queued` adds backpressure (`ConcurrencyLimitExceeded` for callers over the queue cap); waits emit an OTel span. `AbstractConcurrencyLimiter` is the ABC for distributed (e.g. Redis-backed) implementations. `get_concurrency_context(limiter, source)` returns a no-op context manager when `limiter is None`; `normalize_to_limiter()` coerces `AnyConcurrencyLimit` to `AbstractConcurrencyLimiter | None`.

```python
class ConcurrencyLimiter(AbstractConcurrencyLimiter):
    def __init__(self, max_running: int, *, max_queued: int | None = None, name=None, tracer=None): ...
    @classmethod
    def from_limit(cls, limit: int | ConcurrencyLimit, *, name=None, tracer=None) -> Self: ...

ConcurrencyLimit(max_running: int, max_queued: int | None = None)          # config dataclass

class ConcurrencyLimitedModel(WrapperModel):
    def __init__(self, wrapped: Model | KnownModelName, limiter: int | ConcurrencyLimit | AbstractConcurrencyLimiter): ...
```

```python
import asyncio
from pydantic_ai import Agent, ConcurrencyLimiter, limit_model_concurrency
from pydantic_ai.exceptions import ConcurrencyLimitExceeded

pool = ConcurrencyLimiter(max_running=5, max_queued=20, name='openai-shared-pool')
agent_a = Agent(limit_model_concurrency('openai:gpt-5', pool))
agent_b = Agent(limit_model_concurrency('openai:gpt-5.2', pool))

async def safe_run(agent, prompt: str):
    try:
        return (await agent.run(prompt)).output
    except ConcurrencyLimitExceeded:
        return None

asyncio.run(asyncio.gather(*[safe_run(agent_a, f'Q{i}') for i in range(20)]))
```

#### `UseThreadExecutor` (capability) — renamed from `ThreadExecutor`

The class documented in older material as `ThreadExecutor` is now `UseThreadExecutor` in
`pydantic_ai.capabilities`. Replaces PydanticAI's default `anyio.to_thread.run_sync`-per-call
behaviour with a bounded `ThreadPoolExecutor` scoped to each run — prevents unbounded thread
creation under load. `Agent.using_thread_executor()` sets it class-wide for every run in context.

```python
UseThreadExecutor(executor: concurrent.futures.Executor)
```

```python
from concurrent.futures import ThreadPoolExecutor
from pydantic_ai import Agent
from pydantic_ai.capabilities import UseThreadExecutor

pool = ThreadPoolExecutor(max_workers=16)
agent = Agent('openai:gpt-4o', capabilities=[UseThreadExecutor(pool)])
```

---

### Hooks, Middleware & Lifecycle

#### `Hooks` + `HookTimeoutError` + `HookNamespace`

`Hooks` registers lifecycle observers via `@hooks.on.<event>` decorators (bare or parameterised with `timeout=`/`tools=`) instead of subclassing `AbstractCapability` — every hook can also be passed directly as a kwarg to the constructor (`Hooks(before_model_request=fn, ...)`). Covers 20+ hook points across four phases: run (`before_run`/`after_run`/`run_error`, or the `run` wrap-handler form for timing/circuit-breakers), node (`before_node_run`/`after_node_run`/`node_run_error`), model request (`before_model_request`/`after_model_request`/`model_request_error`, or `model_request` wrap-form), and tool (`prepare_tools`, `before_tool_validate`/`after_tool_validate`, `before_tool_execute`/`after_tool_execute`/`tool_execute_error`, or `tool_execute` wrap-form with `tools=[...]` scoping), plus the output-validate/process triads and `deferred_tool_calls`. A per-hook `timeout` (seconds) raises `HookTimeoutError` (a `TimeoutError` subclass) via `anyio.fail_after`. Sync hooks run inline on the event loop — use async for anything blocking.

```python
class HookTimeoutError(TimeoutError):
    hook_name: str; func_name: str; timeout: float

class Hooks(AbstractCapability[AgentDepsT]):
    @cached_property
    def on(self) -> HookNamespace: ...
```

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import Hooks, HookTimeoutError

hooks = Hooks()

@hooks.on.before_tool_execute(tools=['db_query', 'file_write'], timeout=2.0)
async def audit_sensitive(ctx, *, call, tool_def, args):
    print(f'[AUDIT] {tool_def.name}({args})')
    return args

@hooks.on.run
async def time_run(ctx, *, handler):
    import time
    start = time.perf_counter()
    try:
        return await handler()
    finally:
        print(f'run took {time.perf_counter() - start:.3f}s')

agent = Agent('openai:gpt-4o', capabilities=[hooks])
```

#### `AbstractCapability` — extended API

Base class for every capability. Beyond the basic `get_instructions`/`get_model_settings`/
`get_toolset`/`get_native_tools`, it exposes `defer_loading`, `get_description()` (shown to the
model's `load_capability` catalog), `get_ordering() -> CapabilityOrdering`, and three-form hooks
per lifecycle phase (`before_*`, `after_*`, `wrap_*` — the `wrap_*` forms receive a zero-arg
`handler` to call-or-skip for short-circuiting).

```python
class AbstractCapability(ABC, Generic[AgentDepsT]):
    id: str | None; defer_loading: bool = False
    async def get_instructions(self, agent) -> str | None: ...
    async def get_toolset(self, agent) -> AbstractToolset | None: ...
    async def get_ordering(self) -> CapabilityOrdering: ...
    async def wrap_model_request(self, ctx, *, request_context, handler) -> ModelResponse: ...
```

```python
from dataclasses import dataclass
from pydantic_ai.capabilities.abstract import AbstractCapability

@dataclass
class CachingCapability(AbstractCapability):
    _cache: dict = None
    def __post_init__(self): self._cache = {}

    async def wrap_model_request(self, ctx, *, request_context, handler):
        key = str(request_context.messages)
        if key in self._cache:
            return self._cache[key]
        response = await handler()
        self._cache[key] = response
        return response
```

#### `CapabilityOrdering` + `CapabilityPosition` + `CapabilityRef` + `sort_capabilities` + `collect_leaves` + `has_capability_type`

Declares where in the middleware chain a capability sits. `position: Literal['outermost',
'innermost'] | None`; `wraps`/`wrapped_by: Sequence[CapabilityRef]` for relative ordering;
`requires: Sequence[type[AbstractCapability]]` for presence checks (raises `UserError` if the
required capability type isn't present) with no ordering implied. `sort_capabilities()` uses
`graphlib.TopologicalSorter` with original list order as tiebreaker and cycle detection.
`collect_leaves()` flattens nested capability trees via the visitor pattern; `has_capability_type()`
checks membership. `CAPABILITY_TYPES` is the name→class registry used by `AgentSpec` YAML
loading, populated via `__init_subclass__`.

```python
@dataclass
class CapabilityOrdering:
    position: CapabilityPosition | None = None
    wraps: Sequence[CapabilityRef] = ()
    wrapped_by: Sequence[CapabilityRef] = ()
    requires: Sequence[type[AbstractCapability]] = ()

def sort_capabilities(capabilities) -> list[AbstractCapability]: ...
```

```python
from dataclasses import dataclass
from pydantic_ai.capabilities.abstract import AbstractCapability, CapabilityOrdering

@dataclass
class AuthCapability(AbstractCapability):
    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position='outermost')
```

#### `WrapperCapability`

Transparent delegation base for capability middleware — the capability analogue of
`WrapperToolset`/`WrapperModel`. `__post_init__` inherits `id`/`defer_loading` from the wrapped
capability when not explicitly set, so a wrapper over a deferred capability stays deferred, and
`for_run()` recreates the wrapper around the post-`for_run` wrapped instance.

```python
@dataclass
class WrapperCapability(AbstractCapability[AgentDepsT]):
    wrapped: AbstractCapability[AgentDepsT]
```

```python
import dataclasses
from pydantic_ai.capabilities import Capability, WrapperCapability

@dataclasses.dataclass
class LoggingCapability(WrapperCapability):
    async def before_model_request(self, ctx, request_context):
        print(f'run_step={ctx.run_step}')
        return await self.wrapped.before_model_request(ctx, request_context)

refunds = Capability(id='refunds', instructions='Confirm the order ID before refunding.')
wrapped = LoggingCapability(wrapped=refunds)
```

#### `CombinedCapability`

The composition engine `Agent(capabilities=[...])` builds internally when given a list — flattens
nested combinations, topologically sorts by `CapabilityOrdering`, and always places the
auto-injected pending-message drainer outermost. Hook direction is forward for `before_*`/
`prepare_*`, reverse for `after_*`/`on_*_error`, and reverse-built-closure for `wrap_*` (standard
middleware onion); `for_run()` runs all children's `for_run()` concurrently via `gather()` and
short-circuits (returns `self`) if none changed. `has_wrap_node_run` is a cached shortcut property
that lets the runtime skip the wrap-hook machinery entirely when no child capability defines one.

```python
from pydantic_ai.capabilities import CombinedCapability, Hooks, Thinking, PrefixTools

combo = CombinedCapability([Hooks(), Thinking(effort='low'), PrefixTools(wrapped=..., prefix='v1')])
print(type(combo.capabilities[0]).__name__)   # ordering-sensitive capabilities move to the front
```

#### `DynamicCapability`

Builds another capability per-run from a factory `CapabilityFunc[AgentDepsT]` — a callable
receiving `RunContext` and returning an `AbstractCapability | None`, sync or async. Bare callables
passed to `capabilities=[...]` are auto-wrapped in this. Returning `None` makes the wrapper a
no-op for that run; `defer_loading=True` is rejected on the `DynamicCapability` wrapper itself —
set it on the capability the factory *returns* instead.

```python
CapabilityFunc = Callable[[RunContext[AgentDepsT]], AbstractCapability[AgentDepsT] | None | Awaitable[...]]

@dataclass
class DynamicCapability(AbstractCapability[AgentDepsT]):
    capability_func: CapabilityFunc[AgentDepsT]
```

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import WebSearch

def feature_flagged(ctx: RunContext):
    return WebSearch() if getattr(ctx.deps, 'enable_search', False) else None

agent = Agent('openai:gpt-4o', capabilities=[feature_flagged])   # auto-wrapped as DynamicCapability
```

#### `ProcessHistory` + `ReinjectSystemPrompt`

Both fire before every model request to transform message history. `ProcessHistory(processor)`
runs an arbitrary `HistoryProcessorFunc` — four auto-detected calling conventions: sync/async ×
with/without `RunContext` — for truncation, PII redaction, or compaction; sync callables run
inline (not thread-offloaded). `ReinjectSystemPrompt(replace_existing=False)` ensures the agent's
configured system prompt survives history that had it stripped (`replace_existing=False` is a
no-op if any system prompt is already present anywhere in history); `replace_existing=True` strips
*any* existing system prompt first, then prepends unconditionally — this is what
`AGUIAdapter`/`VercelAIAdapter` use under `manage_system_prompt='server'` to stop untrusted
clients injecting their own prompts. Neither capability is spec-serialisable (both hold a
callable); the deprecated alias `HistoryProcessor` still works but warns.

```python
@dataclass
class ProcessHistory(AbstractCapability[AgentDepsT]):
    processor: HistoryProcessorFunc[AgentDepsT]

@dataclass
class ReinjectSystemPrompt(AbstractCapability[AgentDepsT]):
    replace_existing: bool = False
```

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessHistory, ReinjectSystemPrompt

def keep_last_n(n):
    def processor(ctx, messages): return messages[-n:]
    return processor

agent = Agent(
    'openai:gpt-4o', instructions='You are a support agent.',
    capabilities=[ProcessHistory(keep_last_n(15)), ReinjectSystemPrompt(replace_existing=True)],
)
```

#### `Instrumentation` capability + `InstrumentationSettings`

The capability-shaped replacement for `Agent(instrument=...)`. Always positioned `'outermost'`
(`get_ordering()`), so its spans wrap every other capability's. `InstrumentationSettings.version`
(1–5) selects the OTel GenAI semantic-convention version; v1 is legacy/deprecated, v5 additionally
stops classifying `CallDeferred`/`ApprovalRequired` as span errors.

```python
Instrumentation(settings: InstrumentationSettings = InstrumentationSettings(), *, id=None, description=None, defer_loading=False)
InstrumentationSettings(*, tracer_provider=None, include_content=True, version=..., event_mode='attributes')
```

```python
import logfire
from pydantic_ai import Agent
from pydantic_ai.capabilities import Instrumentation
from pydantic_ai import InstrumentationSettings

logfire.configure()
agent = Agent('openai:gpt-4o', capabilities=[Instrumentation(InstrumentationSettings(version=5, include_content=False))])
```

`InstrumentedModel` wraps a single `Model` with OTel instrumentation without touching `Agent` — the
lower-level building block `Instrumentation` uses internally: `InstrumentedModel(wrapped: Model, options: InstrumentationSettings)`.

Instrumentation internals (`pydantic_ai._instrumentation`) shared with `InstrumentedModel`: baggage
keys `AGENT_NAME_BAGGAGE_KEY`/`RUN_ID_BAGGAGE_KEY`/`CONVERSATION_ID_BAGGAGE_KEY` propagate agent
identity across service boundaries; `TOKEN_HISTOGRAM_BOUNDARIES` (14 boundaries, 1 to 67M tokens)
configure the `gen_ai.client.token.usage` metric; `DEFAULT_INSTRUMENTATION_VERSION` selects the
GenAI semconv version. `CostCalculationFailedWarning` (raised when `genai-prices` can't price a
model) lives in `pydantic_ai.exceptions`, not `_instrumentation`.

```python
from pydantic_ai._instrumentation import AGENT_NAME_BAGGAGE_KEY, TOKEN_HISTOGRAM_BOUNDARIES, DEFAULT_INSTRUMENTATION_VERSION
from pydantic_ai.exceptions import CostCalculationFailedWarning   # current location
```

#### `PrepareTools` + `PrepareOutputTools`

Wraps a `ToolsPrepareFunc` — `(RunContext, list[ToolDefinition]) -> list[ToolDefinition]` — as a capability that filters/mutates **function** tools (`PrepareTools`) or **output** tools (`PrepareOutputTools`, whose `ctx.retry`/`ctx.max_retries` reflect the output retry budget) on every request. Replaces the older pattern of passing `prepare=` directly to `FunctionToolset` when the filter must apply across every toolset on the agent. Cannot add or rename tools (raises `UserError`); neither is spec-serialisable.

```python
@dataclass
class PrepareTools(AbstractCapability[AgentDepsT]):
    prepare_func: ToolsPrepareFunc[AgentDepsT]
@dataclass
class PrepareOutputTools(AbstractCapability[AgentDepsT]):
    prepare_func: ToolsPrepareFunc[AgentDepsT]
```

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import PrepareTools

async def hide_admin_tools(ctx: RunContext[dict], tool_defs):
    if ctx.deps.get('is_admin'):
        return tool_defs
    return [td for td in tool_defs if not td.name.startswith('admin_')]

agent = Agent('openai:gpt-4o-mini', deps_type=dict, capabilities=[PrepareTools(hide_admin_tools)])
```

#### `SetToolMetadata`

Merges `**metadata` kwargs into the `metadata` dict of tools matched by `tools: ToolSelector` (`'all'`, a name/list of names, or a sync/async predicate). Internally wraps the toolset in a `PreparedToolset` that overrides `get_tools`. Multiple instances stack additively per tool — most commonly used to flip on Code Mode (`code_mode=True`) or tag tools for provider cache control / OTel attributes.

```python
@dataclass(init=False)
class SetToolMetadata(AbstractCapability[AgentDepsT]):
    def __init__(self, *, tools: ToolSelector[AgentDepsT] = 'all', **metadata: Any) -> None: ...
```

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import SetToolMetadata

agent = Agent('openai:gpt-4o-mini', capabilities=[
    SetToolMetadata(tools=['run_sql', 'execute_python'], code_mode=True),
])
```

#### `ModelRetry`

Raise from a tool function, output validator, or capability hook to send a retry prompt back to
the model instead of propagating a Python exception. Fully Pydantic-serialisable (used internally
for durable execution).

```python
ModelRetry(message: str)
```

```python
from pydantic_ai import Agent, RunContext, ModelRetry
import re

agent = Agent('openai:gpt-4o')

@agent.tool
async def lookup_user(ctx: RunContext[None], email: str) -> str:
    if not re.match(r'^[^@]+@[^@]+\.[^@]+$', email):
        raise ModelRetry(f'{email!r} is not a valid email address.')
    return f'profile for {email}'
```

---

### Capabilities & Extensibility

#### `Capability` — no-subclass decorator API

Bundles instructions, tools, and toolsets under one identity without subclassing `AbstractCapability`. Three decorators mirror the `Agent` API: `@cap.tool` (receives `RunContext`), `@cap.tool_plain` (no context), `@cap.instructions` (system-prompt function, sync or async). `defer_loading=True` hides the whole capability (instructions + tools) until the model calls `load_capability`.

```python
class Capability(AbstractCapability[AgentDepsT]):
    def __init__(self, *, instructions=None, toolsets=None, tools=(), id=None, description=None, defer_loading=False): ...
    def tool(self, func=None, /, **kwargs): ...
    def tool_plain(self, func=None, /, **kwargs): ...
    def instructions(self, func): ...
```

```python
from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import Capability, ToolSearch

support = Capability(id='customer-support', instructions='Be empathetic and concise.')

@support.tool
def get_order(ctx: RunContext[str], order_id: str) -> dict:
    return {'order_id': order_id, 'customer': ctx.deps}

finance_cap = Capability(
    description='Finance tools: stock prices, portfolio returns.',
    id='finance-tools', defer_loading=True,
)

@finance_cap.tool
def get_stock_price(ctx: RunContext[None], ticker: str) -> float:
    """Return the latest closing price."""
    return 150.0

agent = Agent('openai:gpt-4o', deps_type=str, capabilities=[support, ToolSearch(), finance_cap])
```

#### `NativeTool` + `NativeOrLocalTool` capabilities

`NativeTool(tool)` registers a single provider-native tool (static instance or per-run callable).
`NativeOrLocalTool` is the architectural base every adaptive capability (`WebSearch`, `WebFetch`,
`ImageGeneration`, `XSearch`, `MCP`) is built on: pairs a provider-native tool with an optional
local fallback function, keeping only whichever the active model supports. `native=True` uses the
subclass's `_default_native()`; `local` accepts a strategy name, `Tool`, callable, `AbstractToolset`,
or bool (per-subclass). When both are enabled, `get_toolset()` wraps the local toolset in a
`PreparedToolset` that stamps `unless_native=<uid>` on every local `ToolDefinition`, so capable
models never see the fallback tools at all. `_requires_native()` returning `True` suppresses local
entirely (e.g. domain-constraint fields that only the native tool enforces).

```python
NativeTool(tool: AbstractNativeTool | Callable[[RunContext], AbstractNativeTool | None])

class NativeOrLocalTool(AbstractCapability[AgentDepsT]):
    def _default_native(self) -> AbstractNativeTool | None: ...
    def _default_local(self) -> Tool | AbstractToolset | None: ...
    def _requires_native(self) -> bool: return False
```

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import NativeOrLocalTool
from pydantic_ai.native_tools import WebSearchTool
from pydantic_ai.tools import Tool

cap = NativeOrLocalTool(native=WebSearchTool(), local=Tool(lambda query: f'Results for {query}'))
agent = Agent('openai:gpt-4o-mini', capabilities=[cap])
```

#### `WebSearch` capability

`NativeOrLocalTool` subclass — native-first web search with an optional DuckDuckGo (or custom) local fallback. `local` (`WebSearchLocalStrategy='duckduckgo' | Tool | Callable | bool | None`) requires the `duckduckgo` extra when set to `'duckduckgo'`/`True`. `blocked_domains`/`allowed_domains`/`max_uses` require native support and auto-force it via `_requires_native()`.

```python
class WebSearch(NativeOrLocalTool[AgentDepsT]):
    def __init__(self, *, native=True, local=None, search_context_size=None, user_location=None,
                 blocked_domains=None, allowed_domains=None, max_uses=None,
                 external_web_access=None, id=None, defer_loading=False, description=None): ...
```

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebSearch

agent = Agent('openai:gpt-4o-mini', capabilities=[WebSearch(local=True, search_context_size='high')])
```

#### `WebFetch` capability

`local=True` activates the SSRF-protected, markdownify-based local fetcher from `pydantic_ai.common_tools.web_fetch` (requires `pip install "pydantic-ai-slim[web-fetch]"`); `allowed_domains`/`blocked_domains` are enforced by the local tool too, while `max_uses`, `enable_citations`, `max_content_tokens` require native.

```python
class WebFetch(NativeOrLocalTool[AgentDepsT]):
    def __init__(self, *, native=True, local=None, allowed_domains=None, blocked_domains=None,
                 max_uses=None, enable_citations=None, max_content_tokens=None,
                 id=None, defer_loading=False): ...
```

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebFetch

agent = Agent('anthropic:claude-sonnet-4-5', capabilities=[
    WebFetch(local=True, allowed_domains=['docs.python.org', 'docs.pydantic.dev']),
])
```

#### `ImageGeneration` capability

Capability-level fields (`action`, `background`, `input_fidelity`, `moderation`, `image_model`, `output_compression`, `output_format`, `quality`, `size`, `aspect_ratio`) bridge onto the native tool's constructor, whose corresponding field is named **`model`**, not `image_model`. **Correction:** `local` no longer accepts a bare `True`; its type is `Tool | Callable | Literal[False] | None` — pass an explicit `Tool`/callable or leave it `None`, and rely on `fallback_model` for cross-provider delegation (which cannot be combined with an explicit `local=` — `UserError` if both are set). `ImageGenerationSubagentTool` implements the fallback: it builds `Agent(fallback_model, output_type=BinaryImage, capabilities=[NativeTool(...)])` at call time and wraps `UnexpectedModelBehavior` as `ModelRetry`.

```python
class ImageGeneration(NativeOrLocalTool[AgentDepsT]):
    def __init__(self, *, native=True, local=None, fallback_model=None,
                 action=None, background=None, input_fidelity=None, moderation=None,
                 image_model=None, output_compression=None, output_format=None,
                 quality=None, size=None, aspect_ratio=None, ...): ...
```

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import ImageGeneration

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[ImageGeneration(fallback_model='openai-responses:gpt-4o', quality='high', output_format='png')],
)
```

#### `XSearch` capability

X/Twitter search — native on xAI models; on any other model `fallback_model` (must be an xAI model) is **required**, or the capability raises `UserError`. `allowed_x_handles`/`excluded_x_handles` (max 20 each), `from_date`/`to_date`, `enable_image_understanding`/`enable_video_understanding`, `include_output` (exposes raw results as `NativeToolReturnPart`). Like `ImageGeneration`, `local` is `Tool | Callable | Literal[False] | None` — no bare `True`.

```python
class XSearch(NativeOrLocalTool[AgentDepsT]):
    def __init__(self, *, native=True, local=None, fallback_model=None,
                 allowed_x_handles=None, excluded_x_handles=None,
                 from_date=None, to_date=None, enable_image_understanding=None,
                 enable_video_understanding=None, include_output=None, ...): ...
```

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import XSearch

agent = Agent('openai:gpt-5.2', capabilities=[
    XSearch(fallback_model='xai:grok-4.3', enable_image_understanding=True),
])
```

#### `Thinking` capability

Single-field convenience capability translating a unified `effort` into `ModelSettings.thinking` across providers; provider-specific settings (`anthropic_thinking`, `openai_reasoning_effort`, etc.) take precedence when both are set. `effort=False` is silently ignored on always-on reasoning models.

```python
@dataclass
class Thinking(AbstractCapability[Any]):
    effort: bool | Literal['minimal', 'low', 'medium', 'high', 'xhigh'] = True
```

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking

agent = Agent('anthropic:claude-opus-4-8', capabilities=[Thinking(effort='high')])
```

#### `MCP` capability

The recommended capability-first way to attach an MCP server, extending `NativeOrLocalTool`: accepts `url`, `native` (bool or explicit `MCPServerTool`/callable — **requires `url=` when `True`**, and defaults to `False`, not auto-detect), `local` (URL string, `fastmcp.Client`, transport, in-process `FastMCP` server, or pre-built `MCPToolset` — any other non-bool/non-string value is auto-wrapped into an `MCPToolset`), `authorization_token`, `headers`, `allowed_tools`. `MCP.from_spec()` restricts `local=` to JSON/YAML-serialisable types for `AgentSpec` round-tripping.

```python
class MCP(NativeOrLocalTool[AgentDepsT]):
    def __init__(self, url: str | None = None, *, native: MCPServerTool | Callable | bool = False,
                 local: MCPToolsetClient | MCPToolset | Callable | bool | None = None,
                 id=None, authorization_token=None, headers=None, allowed_tools=None,
                 description=None, defer_loading=False): ...
```

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import MCP

agent = Agent('openai:gpt-5.2', capabilities=[
    MCP(url='https://mcp.example.com/v1', local=True, allowed_tools=['search', 'lookup']),
])
```

#### `ToolSearch` capability + strategy types

Lazy tool discovery for large toolsets. `strategy` accepts `None` (auto: native BM25 on
Anthropic/OpenAI-Responses, local keyword elsewhere), `'bm25'`/`'regex'` (Anthropic-only, error
elsewhere), `'keywords'` (force the local algorithm everywhere for determinism), or a custom
`ToolSearchFunc`. The internal `ToolSearchTool` (an `AbstractNativeTool` the capability injects)
is never constructed directly, and is excluded from `TestModel.supported_native_tools()` (so
tests always fall back to the local `search_tools` function tool). See `ToolSearch` +
`ToolSearchToolset` in Tools & Toolsets for the toolset-level mechanics.

```python
ToolSearch(strategy=None, max_results=10, tool_description=None, parameter_description=None,
           *, id=None, description=None, defer_loading=False)
ToolSearchFunc = Callable[[RunContext, Sequence[str], Sequence[ToolDefinition]], Sequence[str] | Awaitable[...]]
```

```python
from pydantic_ai import Agent, Tool
from pydantic_ai.capabilities import ToolSearch

agent = Agent(
    'anthropic:claude-sonnet-4-5',
    tools=[Tool(get_weather), Tool(book_flight, defer_loading=True)],
    capabilities=[ToolSearch(strategy='keywords', max_results=5)],
)
```

#### `HandleDeferredToolCalls` capability

Intercepts `DeferredToolRequests` that would otherwise pause the run and resolves them **inline** via a user handler — converting a HITL approval flow into an automated one. `handler(ctx, requests) -> DeferredToolResults | None`; returning `None` declines (falls through to the next `HandleDeferredToolCalls` capability, or bubbles up as the run's output if none handle it). Stack multiple instances for tiered strategies (e.g. auto-approve low-risk tools, defer everything else to a human).

```python
@dataclass
class HandleDeferredToolCalls(AbstractCapability[AgentDepsT]):
    handler: Callable[[RunContext, DeferredToolRequests], DeferredToolResults | None | Awaitable[...]]
```

```python
from pydantic_ai import Agent, DeferredToolResults
from pydantic_ai.capabilities import HandleDeferredToolCalls

def auto_approve(ctx, requests):
    return requests.build_results(approve_all=True)

agent = Agent('openai:gpt-4o', capabilities=[HandleDeferredToolCalls(handler=auto_approve)])
```

#### `CapabilityOwnedToolset`

Internal `WrapperToolset` stamping every contributed `ToolDefinition` with the owning `Capability`'s `id`. When `capability.defer_loading=True` it also marks tools with the deferred-capability metadata key and suppresses `get_instructions()` until the capability is explicitly loaded — the plumbing that makes deferred capabilities work (the model sees the description in the catalog but can't call the tools until it calls `load_capability`). `resolve_capability_id()` walks `ctx.capabilities` by identity; `tool_defs_for_loaded_capabilities()` is the wire-side filter `ToolSearchToolset` uses.

```python
@dataclass
class CapabilityOwnedToolset(WrapperToolset[AgentDepsT]):
    capability: AbstractCapability[AgentDepsT]
```

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import Capability

billing = Capability(id='billing', description='Billing tools — load when asked about payments.', defer_loading=True)

@billing.tool_plain
def get_invoice(invoice_id: str) -> dict:
    return {'id': invoice_id, 'status': 'pending'}

agent = Agent('openai:gpt-4o-mini', capabilities=[billing])
```

#### `DeferredCapabilityLoader` + `DeferredCapabilityLoaderToolset` + `LoadCapabilityCallPart`/`LoadCapabilityArgs`/`LoadCapabilityReturn`

`DeferredCapabilityLoader` produces the catalog instructing the model which deferred capabilities exist — deliberately re-listing **every** deferred capability on **every** turn (including already-loaded ones), because instructions sit at the request prefix and mutating that prefix would bust the provider's prompt cache. `DeferredCapabilityLoaderToolset` auto-injects the reserved `load_capability` tool (`tool_kind='capability-load'`); calling it resolves the capability from `ctx.capabilities`, returns its instructions, and raises `ModelRetry` if the model tries to reload an already-loaded capability.

```python
class LoadCapabilityArgs(TypedDict):
    id: str
class LoadCapabilityReturn(TypedDict):
    instructions: NotRequired[str]
```

```python
from pydantic_ai._deferred_capabilities import LoadCapabilityCallPart

# Filter capability-load parts out of message history when replaying a conversation
def strip_capability_loads(messages):
    import dataclasses
    return [
        dataclasses.replace(m, parts=[p for p in m.parts if not isinstance(p, LoadCapabilityCallPart)])
        if hasattr(m, 'parts') else m
        for m in messages
    ]
```

#### `NamedSpec` + `CapabilitySpec` + `build_registry` + `load_from_registry`

**Module:** `pydantic_ai._spec`. Powers YAML/JSON-driven capability composition for `AgentSpec`.
`NamedSpec` accepts three compact forms (bare name string, single-arg dict, kwargs dict).
`build_registry` builds a name→class map; `load_from_registry` instantiates from a spec, with
`legacy_aliases` support for renamed classes.

```python
NamedSpec.model_validate('Instrumentation')
NamedSpec.model_validate({'WebSearch': {'search_context_size': 'high'}})
```

#### `Toolset` capability

A lightweight `AbstractCapability` that injects any pre-built `AgentToolset` via the capabilities list rather than the `toolsets=` constructor arg — useful for programmatic capability-chain composition (e.g. combined with `PrefixTools`). Not spec-serialisable (`get_serialization_name()` returns `None`).

```python
@dataclass
class Toolset(AbstractCapability[AgentDepsT]):
    toolset: AgentToolset[AgentDepsT]
```

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities.toolset import Toolset
from pydantic_ai.toolsets.function import FunctionToolset

weather_toolset = FunctionToolset()
weather_toolset.add_function(lambda city: f'Sunny in {city}', name='get_weather')
agent = Agent('openai:gpt-5.2', capabilities=[Toolset(toolset=weather_toolset)])
```

#### `doc_descriptions` + `DocstringStyle` + `_infer_docstring_style` — griffe-backed docstring parser

Called by `FunctionToolset`/`Tool` during registration to extract the function description and per-parameter descriptions using [griffe](https://mkdocstrings.github.io/griffe/). Supports `'google'`, `'numpy'`, `'sphinx'`, and `'auto'` (regex-based inference: Sphinx `:param:`, Google `Args:` block, NumPy `---` underline, falling back to `'google'`). When a `Returns` section exists, the description is reformatted as `<summary>`/`<returns>` XML for richer schema descriptions.

```python
DocstringStyle = Literal['google', 'numpy', 'sphinx']
def doc_descriptions(func, sig, *, docstring_format: DocstringFormat) -> tuple[str | None, dict[str, str]]: ...
```

```python
from inspect import signature
from pydantic_ai._griffe import doc_descriptions

def search(query: str, max_results: int = 10) -> list[str]:
    """Search the web.

    Args:
        query: The search query string.
        max_results: Maximum results to return.
    """
    return []

desc, params = doc_descriptions(search, signature(search), docstring_format='auto')
```

---

### Durable Execution & Integrations (Temporal, DBOS, Prefect)

> Requires the respective optional dependency group (`temporal`, `dbos`, `prefect`) — not installed in the verification venv, so only import paths and top-level structure were re-confirmed; per-class field details below should be spot-checked against the exact pinned version in production.

#### `TemporalAgent` + `TemporalModel` + `TemporalProviderFactory`

**Extra:** `pip install "pydantic-ai[temporal]"`. Wraps any `Agent`/`WrapperAgent` so model calls, tool calls, and MCP interactions become durable Temporal activities. `TemporalModel` is the `WrapperModel` placed inside every `TemporalAgent`: inside a workflow it serialises the request into a `_RequestParams` dataclass and dispatches via `workflow.execute_activity(...)`; outside a workflow it falls through directly. Supports a `models={id: Model}` registry plus `using_model(id)` context manager for per-step overrides, and a `TemporalProviderFactory` callable (`(RunContext, provider_name) -> Provider`) for dynamic per-tenant credentials on unregistered model IDs. Image output is rejected (`UserError`) due to Temporal's 2MB payload limit. `PydanticAIWorkflow` is a marker base class exposing `__pydantic_ai_agents__` so `TemporalAgent.activities`/`.temporal_activities` can be enumerated automatically.

```python
TemporalAgent(wrapped, *, name: str, models=None, provider_factory=None,
              event_stream_handler=None, activity_config=..., model_activity_config={},
              toolset_activity_config={}, tool_activity_config={}, run_context_type=TemporalRunContext)

class TemporalModel(WrapperModel):
    def using_model(self, model) -> Generator[None]: ...
TemporalProviderFactory = Callable[[RunContext[AgentDepsT], str], Provider[Any]]
```

`TemporalRunContext` serialises only the JSON-safe subset of `RunContext` across the activity
boundary (excludes the live `capabilities` registry); accessing an excluded attribute inside an
activity raises `UserError` with a subclassing hint.

```python
from pydantic_ai import Agent
from pydantic_ai.durable_exec.temporal import TemporalAgent

base_agent = Agent('openai:gpt-4.1-mini', name='research-agent')   # name= is required
temporal_agent = TemporalAgent(base_agent, name='research-agent')
# register temporal_agent.temporal_activities (or .activities) with your Temporal Worker
```

#### `TemporalFunctionToolset` + `TemporalWrapperToolset` + `CallToolResult`

`TemporalWrapperToolset` is the abstract base turning `call_tool()` into `@activity.defn` functions. `CallToolResult` is a discriminated union (`kind` field) of `_ApprovalRequired`, `_CallDeferred`, `_ModelRetry`, `_ToolReturn` — serialising every possible tool-call outcome across the activity boundary. Activity config is layered: `activity_config` (base) → `toolset_activity_config` (per toolset ID) → `tool_activity_config` (per tool name, or `False` to skip activity wrapping for fast, in-memory async tools). `TemporalMCPToolset` wraps an `MCPToolset` so `get_tools`/`call_tool` run as activities; when the wrapped `MCPToolset.cache_tools=True`, tool definitions from the first `get_tools` activity are cached on the `TemporalMCPToolset` instance for the **worker process's lifetime** (not per-workflow) — use `cache_tools=False` if the server's tool list changes between runs.

```python
CallToolResult = Annotated[_ApprovalRequired | _CallDeferred | _ModelRetry | _ToolReturn, Discriminator('kind')]
```

```python
from datetime import timedelta
from temporalio.workflow import ActivityConfig
from pydantic_ai.durable_exec.temporal import TemporalAgent

temporal_agent = TemporalAgent(
    agent,
    activity_config=ActivityConfig(start_to_close_timeout=timedelta(seconds=30)),
    tool_activity_config={'weather-tools': {'fast_lookup': False}},
)
```

#### `LogfirePlugin`

A `temporalio.plugin.SimplePlugin` wiring Logfire into a Temporal `ServiceClient`: installs a `TracingInterceptor` for workflow/activity spans, and optionally an `OpenTelemetryConfig` metrics exporter to Logfire's OTLP endpoint. Must be combined with `PydanticAIPlugin()` (which registers the Pydantic data converter) — using `LogfirePlugin` alone breaks payload serialisation.

```python
client = await Client.connect("localhost:7233", plugins=[PydanticAIPlugin(), LogfirePlugin(metrics=True)])
```

#### `DBOSAgent` + `DBOSModel` + `StepConfig`

Wraps any `AbstractAgent` with DBOS durable-step semantics via `@DBOS.dbos_class()` + `DBOSConfiguredInstance`. Model requests and MCP calls are auto-wrapped as `@DBOS.step()`; **`FunctionToolset` tool functions are not** — decorate side-effecting tools with `@DBOS.step()` yourself for checkpoint/replay protection. `DBOSParallelExecutionMode` excludes `'parallel'` (only `'sequential'` and `'parallel_ordered_events'`) because DBOS needs deterministic replay ordering. `DBOSModel` applies its `@DBOS.step()` decorator once at `__init__`, not per-call; the `DBOS.workflow_id is None or DBOS.step_id is not None` guard in `request_stream` lets nested calls (inside an existing step, or outside a workflow) bypass the step wrapper and avoid double-wrapping. Automatically swaps `MCPToolset` → `DBOSMCPToolset` and similar wrapping for toolsets you pass in.

```python
DBOSParallelExecutionMode = Literal['sequential', 'parallel_ordered_events']

@DBOS.dbos_class()
class DBOSAgent(WrapperAgent[AgentDepsT, OutputDataT], DBOSConfiguredInstance):
    def __init__(self, wrapped, *, name: str | None = None, mcp_step_config=None, model_step_config=None,
                 parallel_execution_mode: DBOSParallelExecutionMode = 'parallel_ordered_events'): ...

class StepConfig(TypedDict, total=False):
    retries_allowed: bool; interval_seconds: float; max_attempts: int; backoff_rate: float
```

```python
from pydantic_ai import Agent
from pydantic_ai.durable_exec.dbos import DBOSAgent

durable = DBOSAgent(Agent('openai:gpt-4o', name='researcher'), name='researcher', model_step_config={'retries': 3})
```

#### `PrefectAgent` + `TaskConfig` + `PrefectAgentInputs` + `DEFAULT_PYDANTIC_AI_CACHE_POLICY`

**Extra:** `pip install "pydantic-ai-slim[prefect]"`. Wraps any `AbstractAgent` to run model requests, tool calls, and MCP interactions as Prefect tasks with automatic retries and caching. `name` is required. `PrefectModel` turns `request()`/`request_stream()` into `@task`-decorated functions created once at `__init__`, named dynamically per call via `with_options(name=...)`; `request_stream` requires an `event_stream_handler` — without one, `PrefectModel` raises because streaming needs a `run_context`. `PrefectFunctionToolset` (a `PrefectWrapperToolset` subclass) does the same for tool calls; a per-tool config entry of `None` skips task wrapping entirely (plain async call, no Prefect overhead). `PrefectAgentInputs` is a custom Prefect `CachePolicy` that strips non-deterministic `RunContext` fields (`timestamp`, `run_id`) and converts `ToolsetTool`/`RunContext` instances into hashable dicts before computing the cache key — plain Prefect `INPUTS` caching breaks on both. `DEFAULT_PYDANTIC_AI_CACHE_POLICY = PrefectAgentInputs() + TASK_SOURCE + RUN_ID`, giving `persist_result=True` with a `RUN_ID`-scoped policy so a persisted result is reused across a flow *retry* but not across unrelated flow runs.

```python
PrefectAgent(wrapped, *, name: str, mcp_task_config=None, model_task_config=None,
             tool_task_config=None, tool_task_config_by_name=None)

class PrefectFunctionToolset(PrefectWrapperToolset[AgentDepsT]):
    async def call_tool(self, name, tool_args, ctx, tool):
        cfg = self._tool_task_config.get(name, default_task_config)
        if cfg is None:
            return await super().call_tool(name, tool_args, ctx, tool)
        return await self._call_tool_task.with_options(name=f'Call Tool: {name}', **cfg)(name, tool_args, ctx, tool)
```

```python
from prefect import flow
from pydantic_ai.durable_exec.prefect import PrefectAgent

durable_agent = PrefectAgent(Agent('openai:gpt-4.1-mini', name='research-agent'), name='research-agent')

@flow(name='research-flow')
async def research_flow(topic: str) -> str:
    result = await durable_agent.run(f'Research: {topic}')
    return result.output
```

#### `RuntimeToolsetKind` + `reject_unsupported_runtime_toolsets`

Durable engines wrap constructor-time toolsets so tool calls become checkpointed activities/tasks; toolsets passed per-run via `run(toolsets=...)` arrive **after** that wrapping and are un-checkpointed. This guard classifies each leaf toolset (`'function'`, `'mcp'`, `'dynamic'`, or `None` for non-executing toolsets like `ExternalToolset`) and raises `UserError` when an engine-unsupported kind is passed per-run.

```python
RuntimeToolsetKind = Literal['function', 'mcp', 'dynamic']
def reject_unsupported_runtime_toolsets(toolsets, *, unsupported_kinds: frozenset[RuntimeToolsetKind], engine: str) -> None: ...
```

#### `agent_to_a2a` / `AgentWorker` — removed

The A2A (Agent-to-Agent protocol) bridge, already marked deprecated in older releases, is
**confirmed fully removed** — no `_a2a.py` or any A2A-related module remains under `pydantic_ai/`.
The `fasta2a` package now maintains its own PydanticAI integration independently:
`pip install "fasta2a[pydantic-ai]"` then `from fasta2a.pydantic_ai import agent_to_a2a`.

---

### Persistence & Graph Support

#### `GraphBuilder` (pydantic_graph)

The current, recommended API for building `pydantic_graph` graphs — a fluent, type-safe builder
replacing the older `BaseNode`-subclass pattern (which remains fully supported and interoperable).

```python
GraphBuilder(*, name=None, state_type=NoneType, deps_type=NoneType, input_type=NoneType,
             output_type=NoneType, auto_instrument=True)
```

```python
from pydantic_graph import GraphBuilder
from dataclasses import dataclass

@dataclass
class State:
    value: int

builder = GraphBuilder(state_type=State)

@builder.step
async def increment(ctx):
    ctx.state.value += 1
    return ctx.state.value

builder.add(builder.edge_from(builder.start_node).to(increment))
builder.add(builder.edge_from(increment).to(builder.end_node))
graph = builder.build()
```

#### `Path` + `PathBuilder`

`Path` is a flat `list[PathItem]` encoding transforms, forks, and routing in order; `PathBuilder` is the fluent wrapper. `.to(dest, ...)` routes to one or more destinations (wraps multiple in a `BroadcastMarker`); `.transform(func)` applies a sync step function, changing the output type; `.map()` spreads an iterable into parallel per-item paths (creates a `MapMarker`); `.label()` attaches a debug annotation. `GraphBuilder.add_edge(node)` returns an `EdgePathBuilder`, the entry point in practice.

```python
class PathBuilder(Generic[StateT, DepsT, OutputT]):
    def to(self, destination, /, *extra, fork_id=None) -> Path: ...
    def transform(self, func) -> PathBuilder[StateT, DepsT, T]: ...
    def map(self, *, fork_id=None, downstream_join_id=None) -> PathBuilder[StateT, DepsT, T]: ...
```

```python
builder.add_edge(source_node).label('count-chars').transform(lambda ctx: len(ctx.inputs)).to(sink_node)
```

#### Marker types: `TransformFunction`/`TransformMarker`/`MapMarker`/`BroadcastMarker`/`LabelMarker`/`DestinationMarker`/`PathItem`

Every `Path` is a list of these dataclasses; `PathItem` is their union. `MapMarker(fork_id, downstream_join_id)` spreads an iterable into parallel forks; `BroadcastMarker(paths, fork_id)` fans out to pre-built sub-paths; `DestinationMarker(destination_id)` is the terminal routing target.

```python
PathItem = TransformMarker | MapMarker | BroadcastMarker | LabelMarker | DestinationMarker
```

```python
from pydantic_graph.paths import Path, LabelMarker, TransformMarker, DestinationMarker
from pydantic_graph.id_types import NodeID

sample = Path(items=[LabelMarker('process'), TransformMarker(lambda ctx: str(ctx.inputs)), DestinationMarker(NodeID('my_node'))])
```

#### `EdgePath` + `EdgePathBuilder`

`EdgePath` is a complete edge: source nodes bound to a `Path`, with `destinations` collected. `EdgePathBuilder` (returned by `GraphBuilder.add_edge()`) chains `.map()`/`.transform()`/`.label()`/`.broadcast()` before finalising with `.to(destination)`.

```python
class EdgePathBuilder(Generic[StateT, DepsT, OutputT]):
    def to(self, destination, /, *extra, fork_id=None) -> EdgePath: ...
    def broadcast(self, get_forks, /, *, fork_id=None) -> EdgePath: ...
```

```python
edge = (
    builder.add_edge(splitter)
    .map()
    .transform(lambda ctx: len(ctx.inputs.split()))
    .to(counter)
)
```

#### `Fork` + `Join` + `ReducerContext` + `JoinState`

Parallel fan-out/fan-in. `Fork.is_map=True` maps one branch per sequence element (`is_map=False` broadcasts the same value to every branch). `Join` aggregates via a reducer (`(current, item) -> result`, optionally `(ctx: ReducerContext, current, item) -> result`); `JoinState` tracks pending parallel branches per fork. `ReducerContext.cancel_sibling_tasks()` implements first-match-wins early stopping (sets `JoinState.cancelled_sibling_tasks=True`); `preferred_parent_fork='farthest'/'closest'` disambiguates nested-fork topology.

```python
builder.join(reducer, initial=[], node_id='collect')
builder.add_mapping_edge(produce, process, downstream_join_id=join.id)
```

```python
def first_success(ctx: ReducerContext, current, item):
    if item is not None and current is None:
        ctx.cancel_sibling_tasks()
        return item
    return current
```

#### `Decision` + `DecisionBranch` + `Edge` + `TypeExpression`

Conditional routing via `builder.decision().branch(builder.match(Literal['urgent']).to(handler))`
— `match()` requires **types** (or `Literal[...]`), not raw values; `builder.match('urgent')`
raises at runtime. `Edge` (`.label(text)` on an edge-path builder) annotates Mermaid diagram output.
`TypeExpression[T]` works around type-checker limitations for complex union types passed as
`state_type=`/`output_type=` generic parameters.

```python
router = (
    builder.decision()
    .branch(builder.match(Literal['urgent']).to(handle_urgent))
    .branch(builder.match(Literal['billing']).to(handle_billing))
)
```

#### `Step` + `StepContext` + `StepNode`

The primitives `@builder.step` produces. `StepContext(state, deps, inputs)` is what every step
function receives. `Step.as_node(inputs)` bridges a builder step into a legacy `BaseNode` runner —
not a "goto" mechanism inside a step body; dynamic branching goes through `builder.decision()`.

```python
@builder.step
async def compute(ctx: builder.Source[int]) -> int:
    return ctx.inputs * ctx.deps.multiplier
```

#### `pydantic_graph.id_types`: `NodeID` + `NodeRunID` + `TaskID` + `ForkStackItem` + `ForkStack`

`NewType` wrappers preventing accidental mixing of graph identifiers. `NodeID` (stable, build-time), `NodeRunID` (per-execution, runtime-generated), `TaskID`; `JoinID`/`ForkID` are `NodeID` aliases. `ForkStackItem(fork_id, node_run_id, thread_index)` is a frozen dataclass; `ForkStack = tuple[ForkStackItem, ...]` represents the full parallel-execution ancestry of a thread. `generate_placeholder_node_id(label)`/`replace_placeholder_id(node_id)` handle auto-generated node IDs.

```python
NodeID = NewType('NodeID', str)
@dataclass(frozen=True)
class ForkStackItem:
    fork_id: ForkID; node_run_id: NodeRunID; thread_index: int
ForkStack = tuple[ForkStackItem, ...]
```

#### `node_types` (`is_source`/`is_destination`) + `parent_forks` (`ParentFork`/`ParentForkFinder`)

Type guards classify every node as source (`MiddleNode | StartNode`), destination (`MiddleNode | Decision | EndNode`), or both. `ParentForkFinder.find_parent_fork(join_id, *, parent_fork_id=None)` finds the *dominating fork* of a join node — the fork every path to that join must pass through — the primitive the runtime uses to avoid deadlock in parallel execution; pass `parent_fork_id` explicitly to disambiguate nested-fork diamonds.

```python
def is_source(node) -> TypeGuard[AnySourceNode]: ...
@dataclass
class ParentForkFinder(Generic[T]):
    def find_parent_fork(self, join_id, *, parent_fork_id=None, prefer_closest=False) -> ParentFork | None: ...
```

#### `FileStatePersistence` + `SimpleStatePersistence` + `FullStatePersistence`

Three built-in `BaseStatePersistence` implementations. `FileStatePersistence(json_file)` persists
snapshots to JSON with an advisory `.pydantic-graph-persistence-lock` file, surviving process
restarts (`graph.iter_from_persistence(persistence)` resumes an interrupted run).
`SimpleStatePersistence` (the run-time default when no `persistence=` is passed) keeps only the
latest snapshot — `load_all()` raises `NotImplementedError`. `FullStatePersistence` keeps the whole
history and supports `dump_json()`/`load_json()` round-trips; `deep_copy=False` skips the
defensive per-snapshot copy for a performance win when you don't need historical state values.

```python
FileStatePersistence(json_file: Path)
FullStatePersistence(deep_copy: bool = True)
```

```python
from pydantic_graph.persistence.file import FileStatePersistence

persistence = FileStatePersistence(Path('runs/demo.json'))
result = await graph.run(StartNode(), state=state, persistence=persistence)
```

#### `NodeSnapshot` + `EndSnapshot` + `BaseStatePersistence` + exception hierarchy

`NodeSnapshot`/`EndSnapshot` make up the `Snapshot` discriminated union every persistence backend
stores. `SnapshotStatus` lifecycle: `'created' → 'pending' (load_next) → 'running' (record_run) →
'success'/'error'`. `BaseStatePersistence` is the ABC custom backends (Redis, DynamoDB) implement —
requiring all six abstract methods: `should_set_types`, `set_types`, `snapshot_node`,
`snapshot_node_if_new`, `snapshot_end` (receives the `End` value), `record_run` (async context
manager), plus `load_next`/`load_all`. `build_snapshot_list_type_adapter(state_type, run_end_type)`
builds the typed serialiser.

```python
GraphSetupError(TypeError)          # misconfigured graph
GraphBuildingError(ValueError)      # error during GraphBuilder.build()
GraphValidationError(ValueError)    # graph structure validation failure
GraphRuntimeError(RuntimeError)     # execution error
GraphNodeStatusError(GraphRuntimeError)   # .check(status) raises unless status in {'created','pending'}
```

```python
@dataclass
class RedisStatePersistence(BaseStatePersistence[Any, Any]):
    redis_client: Any
    run_id: str
    async def snapshot_end(self, state, end) -> None:
        data = self._adapter.dump_json([EndSnapshot(state=state, result=end)])
        await self.redis_client.set(f'run:{self.run_id}:final', data)
```

#### `GraphTaskRequest` + `JoinItem` + `EndMarker`

Low-level primitives driving `pydantic_graph`'s parallel execution engine (same engine powering `Agent.iter()`). `GraphTaskRequest(node_id, inputs, fork_stack)` is a unit of work on the internal task queue. `JoinItem(join_id, inputs, fork_stack)` is emitted when a parallel branch completes and needs to merge at a `Join`; the runtime accumulates them until all expected branches arrive. `EndMarker` is the internal completion signal, converted to `pydantic_graph.End` before yielding — check `isinstance(node, End)` in iteration loops, not `EndMarker`.

```python
@dataclass
class GraphTaskRequest:
    node_id: NodeID; inputs: Any; fork_stack: ForkStack
@dataclass
class JoinItem:
    join_id: JoinID; inputs: Any; fork_stack: ForkStack
```

#### `GraphRun` + `NodeStep` (pydantic_ai's v2 graph internals)

**Module:** `pydantic_ai.run`. `GraphRun` is the execution-state manager `agent.iter()` builds
internally — task scheduling, fork/join coordination (`_active_reducers`), terminal-`End` result
tracking. `NodeStep` bridges any v1 `BaseNode` (like `UserPromptNode`/`ModelRequestNode`/
`CallToolsNode`) into the v2 execution system. For direct graph usage without an `Agent`, use
`pydantic_graph.Graph.run()`/`.iter()` — `GraphRun`/`NodeStep` are exposed for introspection, not as
a primary API.

---

### Testing & Evaluation

#### `TestModel` + `TestStreamedResponse`

Deterministic fake model. By default calls **every** function tool, then returns either an output
tool call or a JSON summary of tool results. `TestModel.__test__ = False` keeps pytest from
collecting it as a test class.

```python
TestModel(call_tools='all', custom_output_text=None, custom_output_args=None, seed=0)
```

Execution order: call all tools (or re-call failing ones on retry) → `custom_output_text` if set →
`custom_output_args` if set → JSON summary if `allow_text_output` → else call
`output_tools[seed % len(output_tools)]`.

```python
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

agent = Agent('test')

@agent.tool_plain
def add(a: int, b: int) -> int:
    return a + b

model = TestModel(custom_output_text='The answer is 42', seed=3)
result = agent.run_sync('Any question', model=model)
print(result.output)   # 'The answer is 42'
print(model.last_model_request_parameters.function_tools)
```

#### `FunctionModel` + `AgentInfo`

Replaces the LLM with a plain Python function `(messages, agent_info) -> ModelResponse` (or an
async generator for `stream_function=`). Auto-injects a permissive default profile
(`supports_json_schema_output=True`, `supports_json_object_output=True`) so structured output
works without extra setup. Constructor accepts `profile`/`settings` overrides so tests can simulate
a specific provider's capability profile.

```python
FunctionModel(function=None, *, stream_function=None, model_name=None, profile=None, settings=None)

@dataclass(frozen=True, kw_only=True)
class AgentInfo:
    function_tools: list[ToolDefinition]
    allow_text_output: bool
    output_tools: list[ToolDefinition]
    model_settings: ModelSettings | None
    model_request_parameters: ModelRequestParameters
    instructions: str | None
```

```python
from pydantic_ai import Agent
from pydantic_ai.models.function import FunctionModel, AgentInfo
from pydantic_ai.messages import ModelResponse, TextPart, ModelRequest, UserPromptPart

def echo_model(messages, agent_info: AgentInfo) -> ModelResponse:
    last = next(p.content for m in reversed(messages) if isinstance(m, ModelRequest)
                for p in m.parts if isinstance(p, UserPromptPart))
    return ModelResponse(parts=[TextPart(f'Echo: {last}')])

agent = Agent(FunctionModel(echo_model))
result = agent.run_sync('Hello, world!')
assert result.output == 'Echo: Hello, world!'
```

#### `TestEmbeddingModel`

Deterministic embedding double: returns all-`1.0` vectors of a configurable `dimensions`, and
records `last_settings` for assertions.

```python
TestEmbeddingModel(model_name='test', *, provider_name='test', dimensions=8, settings=None)
```

```python
from pydantic_ai import Embedder
from pydantic_ai.embeddings import TestEmbeddingModel

async with Embedder('openai:text-embedding-3-small').override(model=TestEmbeddingModel(dimensions=16)) as e:
    result = await e.embed_query('hello')
    assert len(result.embeddings[0]) == 16
```

#### `Dataset` + `Case` (pydantic_evals)

A typed, YAML/JSON-serialisable collection of `Case`s driving `dataset.evaluate(task)`.

```python
Case(*, name=None, inputs, metadata=None, expected_output=None, evaluators=())
Dataset(name=None, cases=[...], evaluators=[...])
    .evaluate(task, *, max_concurrency=None, retry_task=None, repeat=1) -> EvaluationReport
    .add_case(...); .add_evaluator(evaluator, specific_case=None)
    .to_file(path); Dataset[...].from_file(path)
```

```python
from dataclasses import dataclass
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

@dataclass
class ExactMatch(Evaluator):
    def evaluate(self, ctx: EvaluatorContext) -> bool:
        return ctx.output == ctx.expected_output

dataset: Dataset[str, str, None] = Dataset(
    cases=[Case(name='hello', inputs='hello', expected_output='HELLO')],
    evaluators=[ExactMatch()],
)
report = await dataset.evaluate(lambda text: text.upper())
report.print()
```

#### `Evaluator` + `EvaluatorContext` + `EvaluationResult` + `EvaluationReason`

The core evaluator API. `EvaluatorContext` exposes `output`, `expected_output`, `duration`,
`metrics`/`attributes` (populated by `increment_eval_metric`/`set_eval_attribute` called *inside*
the task), and `.span_tree` for OTel-based structural assertions.

```python
class Evaluator(ABC):
    def evaluate(self, ctx: EvaluatorContext) -> EvaluatorOutput | Awaitable[EvaluatorOutput]: ...
# EvaluatorOutput = bool | int | float | str | EvaluationReason | Mapping[str, ...]
EvaluationReason(value: EvaluationScalar, reason: str | None = None)
```

```python
from dataclasses import dataclass
from pydantic_evals.evaluators import Evaluator, EvaluatorContext, EvaluationReason

@dataclass
class MaxWords(Evaluator):
    limit: int = 100
    def evaluate(self, ctx: EvaluatorContext) -> EvaluationReason:
        count = len(str(ctx.output).split())
        return EvaluationReason(value=count <= self.limit, reason=None if count <= self.limit else f'{count} words')
```

#### Built-in evaluators — `Equals` / `EqualsExpected` / `Contains` / `IsInstance` / `MaxDuration` / `HasMatchingSpan`

**Module:** `pydantic_evals.evaluators.common`.

```python
Equals(value)                                   # ctx.output == value
EqualsExpected()                                 # ctx.output == ctx.expected_output
Contains(value, *, case_sensitive=True, as_strings=False)
IsInstance(type_name: str)                       # type(ctx.output).__name__ == type_name
MaxDuration(seconds: float | timedelta)          # ctx.duration <= seconds
HasMatchingSpan(query: SpanQuery)                # ctx.span_tree.any(query)
```

```python
from pydantic_evals.evaluators.common import EqualsExpected, MaxDuration, HasMatchingSpan

evaluators = [EqualsExpected(), MaxDuration(seconds=2.0), HasMatchingSpan({'name_contains': 'chat'})]
```

#### `ToolCorrectness` + `TrajectoryMatch`

Span-based agentic evaluators (`pydantic_evals.evaluators`) reading `ctx.span_tree` — populated only when a tracer provider is registered (`logfire.configure(send_to_logfire=False)` once per process, plus `capabilities=[Instrumentation()]` on the agent under test) and degrading to zero scores otherwise. `ToolCorrectness(expected_tools, allow_extra=False, include_failed=False)` compares the multiset of tool names called — order irrelevant, duplicates require repeated calls. `TrajectoryMatch(expected_trajectory, order='in_order')` enforces ordered sequences: `'exact'` (binary pass/fail), `'in_order'` (LCS-based F1, default), `'any_order'` (multiset F1).

```python
@dataclass(frozen=True)
class ToolCorrectness(Evaluator):
    expected_tools: list[str]
    allow_extra: bool = False
    include_failed: bool = False
@dataclass(frozen=True)
class TrajectoryMatch(Evaluator):
    expected_trajectory: list[str]
    order: Literal['exact', 'in_order', 'any_order'] = 'in_order'
```

```python
import logfire
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import ToolCorrectness

logfire.configure(send_to_logfire=False)   # required so ctx.span_tree is populated

dataset = Dataset(name='demo', cases=[
    Case(name='calls_weather', inputs='Weather in Paris?', evaluators=[ToolCorrectness(expected_tools=['weather_tool'])]),
])
```

#### `ArgumentCorrectness` + `ArgumentMatchMode` + `ArgumentOccurrence` + `MaxToolCalls` + `MaxModelRequests`

`ArgumentCorrectness(tool_name, expected_arguments, match_mode='subset', occurrence='first')` verifies the exact arguments of a specific tool invocation; `match_mode='exact'` requires all keys to match, `'subset'` (default) only requires the expected keys/values to be present. `occurrence` selects which call to inspect: `'first'`, `'last'`, or a 0-based `int` index. `MaxToolCalls(max_calls, include_failed=True)` and `MaxModelRequests(max_requests)` enforce budget caps as pass/fail evaluators — note `MaxToolCalls.include_failed` defaults `True` (opposite of `ToolCorrectness`).

```python
ArgumentMatchMode = Literal['subset', 'exact']
@dataclass(frozen=True)
class ArgumentCorrectness(Evaluator):
    tool_name: str
    expected_arguments: dict[str, Any]
    match_mode: ArgumentMatchMode = 'subset'
    occurrence: Literal['first', 'last'] | int = 'first'
```

```python
from pydantic_evals.evaluators import ArgumentCorrectness, MaxToolCalls, MaxModelRequests

evaluators = [
    ArgumentCorrectness(tool_name='book_flight', expected_arguments={'origin': 'London'}, match_mode='subset'),
    MaxToolCalls(max_calls=3),
    MaxModelRequests(max_requests=2),
]
```

#### `GEval` + `HasMatchingSpan` + `OutputConfig`

`GEval(criteria, evaluation_steps, score_range=(1, 5), include_input=False, model=None)` implements a simplified G-Eval chain-of-thought judge: an LLM judge scores against explicit `criteria` + `evaluation_steps` using a direct integer score (rather than the original paper's log-prob expectation) for provider-agnostic simplicity. `HasMatchingSpan(query: SpanQuery)` passes when at least one span in the captured tree matches (delegates to `SpanQuery.any()`). `OutputConfig` is the shared wire `TypedDict` configuring judge model/output format for both `LLMJudge` and `GEval`.

```python
@dataclass(repr=False)
class GEval(Evaluator):
    criteria: str
    evaluation_steps: list[str]
    score_range: tuple[int, int] = (1, 5)
    model: Model | KnownModelName | str | None = None
```

```python
from pydantic_evals.evaluators import GEval

coherence = GEval(
    criteria='Rate the coherence of the poem to the given topic.',
    evaluation_steps=['Read the poem.', 'Check line-to-line logic.', 'Score 1-5.'],
    model='openai:gpt-4o-mini',
)
```

#### `LLMJudge` + `GradingOutput` + judge functions

LLM-as-judge evaluation. `GradingOutput(reason, pass_, score)`. Four standalone functions cover
input/output/expected combinations; `set_default_judge_model` picks the model used when `LLMJudge`
is constructed without an explicit one.

```python
LLMJudge(rubric: str, *, model=None, score=None, assertion=True, include_input=False, include_expected_output=False)
judge_output(output, rubric) -> GradingOutput
judge_input_output(inputs, output, rubric) -> GradingOutput
judge_output_expected(output, expected_output, rubric) -> GradingOutput
judge_input_output_expected(inputs, output, expected_output, rubric) -> GradingOutput
```

```python
from pydantic_evals.evaluators.common import LLMJudge

judge = LLMJudge(rubric='The answer mentions 42', model='openai:gpt-4o-mini')
```

#### `generate_dataset`

AI-assisted test case generation from a `Dataset` subclass's schema.

```python
generate_dataset(*, dataset_type, path=None, custom_evaluator_types=(), model='openai:gpt-5.2', n_examples=3, extra_instructions=None) -> Dataset
```

```python
from pydantic_evals import Dataset
from pydantic_evals.generation import generate_dataset

class MathDataset(Dataset[dict, float, None]):
    pass

dataset = await generate_dataset(dataset_type=MathDataset, n_examples=5, path='math_cases.yaml')
```

#### `CaseLifecycle`

Per-case evaluation lifecycle hooks (`pydantic_evals.lifecycle`); a fresh instance is created per case so subclasses hold per-case state safely. `setup()` runs before the task (resource allocation), `prepare_context(ctx)` runs after the task but before evaluators (enrich `EvaluatorContext.metrics`/`attributes`, must return the context), `teardown(result)` always runs after evaluators — even if `setup`/`prepare_context` raised (recorded as `ReportCaseFailure`) — but if `teardown` itself raises, that exception propagates and can abort the whole evaluation run.

```python
class CaseLifecycle(Generic[InputsT, OutputT, MetadataT]):
    async def setup(self) -> None: ...
    async def prepare_context(self, ctx: EvaluatorContext) -> EvaluatorContext: ...
    async def teardown(self, result: ReportCase | ReportCaseFailure | None) -> None: ...
```

```python
from pydantic_evals.lifecycle import CaseLifecycle

class TimingLifecycle(CaseLifecycle[str, str, None]):
    async def setup(self) -> None:
        self._start = time.monotonic()
    async def prepare_context(self, ctx):
        ctx.metrics['duration_ms'] = (time.monotonic() - self._start) * 1000
        return ctx
```

#### `OnlineEvaluation`

Attaches evaluators that fire **asynchronously in the background** after each run completes, wrapping `run()`/`run_stream()`/`iter()` without blocking the caller (streaming runs dispatch only after the context manager exits). `Evaluator` instances are auto-wrapped in `OnlineEvaluator` with default sampling; results emit as OTel `gen_ai.evaluation.result` log events, fannable to a custom `EvaluationSink` via `OnlineEvalConfig(default_sample_rate=..., default_sink=...)`. `disable_evaluation()` context manager suppresses evaluation (e.g. inside deterministic tests); `wait_for_evaluations(timeout=...)` blocks until background evaluations finish. `run_on_errors: bool` controls whether failed runs are still evaluated.

```python
@dataclass(kw_only=True)
class OnlineEvaluation(AbstractCapability[AgentDepsT]):
    evaluators: Sequence[Evaluator | OnlineEvaluator]
    config: OnlineEvalConfig | None = None
```

```python
from pydantic_ai import Agent
from pydantic_evals.online_capability import OnlineEvaluation

agent = Agent('openai:gpt-4o-mini', capabilities=[OnlineEvaluation(evaluators=[OutputNotEmpty()])])
```

The `evaluate` decorator applies the same `Evaluator` classes to live production traffic outside an
`Agent` context, emitting the same `gen_ai.evaluation.result` OTel log events:

```python
@evaluate(IsNonEmpty(), OnlineEvaluator(MaxResponseWords(limit=150), sample_rate=0.5))
async def summarise(document: str) -> str: ...
```

#### `SpanTree` + `SpanNode` + `SpanQuery`

Structural OTel span inspection inside an evaluator, via `ctx.span_tree`. `SpanQuery` is a
`TypedDict` supporting `name_contains`/`has_attribute_keys`/`min_duration`/logical combinators
(`and_`/`or_`/`not_`)/child- and descendant-count predicates.

```python
ctx.span_tree.any({'name_contains': 'chat'})
ctx.span_tree.find_all({'has_attribute_keys': ['gen_ai.request.model']})
```

---

### Error Handling & Retries

#### Exception hierarchy

```
AgentRunError (RuntimeError)
├── UsageLimitExceeded
├── ConcurrencyLimitExceeded
├── UnexpectedModelBehavior
│   ├── ContentFilterError
│   └── IncompleteToolCall
└── ModelAPIError
    └── ModelHTTPError(status_code, model_name, body)

UserError (RuntimeError)
└── UndrainedPendingMessagesError

TimeoutError
└── HookTimeoutError(hook_name, func_name, timeout)
```

`ModelHTTPError.status_code` lets you branch on 429/5xx for retry-with-backoff vs. re-raise.
`UndrainedPendingMessagesError` fires when a bare `async for node in agent.iter(...)` loop ends
with `'when_idle'`-priority `ctx.enqueue()` messages never drained — use `agent.run()` or
`AgentRun.next()` instead, both of which drain every priority.

```python
from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelHTTPError, ContentFilterError, UsageLimitExceeded

agent = Agent('openai:gpt-4o')
try:
    result = agent.run_sync(prompt)
except ContentFilterError:
    result = None
except ModelHTTPError as e:
    if e.status_code == 429:
        ...  # backoff and retry
    else:
        raise
```

#### `RetryConfig` + `TenacityTransport` + `AsyncTenacityTransport` + `wait_retry_after`

**Module:** `pydantic_ai.retries`. **Extra:** `pip install "pydantic-ai-slim[retries]"`. Wraps any
`httpx` transport with tenacity-based retry, including honouring `Retry-After` response headers
(seconds or HTTP-date format). `RetryConfig` is a `TypedDict` mirroring the tenacity `@retry`
decorator kwargs (`stop`, `wait`, `retry`, `before_sleep`, `reraise`, ...). `wait_retry_after(
fallback_strategy=None, max_wait=300)` is a wait-strategy factory reading the HTTP `Retry-After`
header before falling back to the given strategy.

```python
class RetryConfig(TypedDict, total=False):
    stop: StopBaseT; wait: WaitBaseT; retry: RetryBaseT; reraise: bool
    before_sleep: Callable[[RetryCallState], None] | None

def wait_retry_after(fallback_strategy=None, max_wait: float = 300) -> Callable[[RetryCallState], float]: ...
```

```python
import httpx
from tenacity import retry_if_exception_type, stop_after_attempt
from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig, wait_retry_after

transport = AsyncTenacityTransport(
    RetryConfig(
        retry=retry_if_exception_type(httpx.HTTPStatusError),
        wait=wait_retry_after(max_wait=120),
        stop=stop_after_attempt(5),
        reraise=True,
    ),
    validate_response=lambda r: r.raise_for_status(),
)
client = httpx.AsyncClient(transport=transport)
```

#### `PydanticAIDeprecationWarning`

All pydantic-ai deprecations are raised as `PydanticAIDeprecationWarning(UserWarning)` rather than `DeprecationWarning`, so they're visible by default at runtime (Python only shows plain `DeprecationWarning` in `__main__`/test runners). Notable renamed/moved symbols this surfaces: `ThreadExecutor` → `UseThreadExecutor`, `CompletedStreamedResponse` moved from `pydantic_ai.models.wrapper` to `pydantic_ai.models`.

```python
class PydanticAIDeprecationWarning(UserWarning): ...
```

```python
import warnings
from pydantic_ai import PydanticAIDeprecationWarning

with warnings.catch_warnings():
    warnings.filterwarnings('error', category=PydanticAIDeprecationWarning)
    # any deprecated call now raises instead of warning — useful in CI
```

---

### Security (approval, SSRF, deferred tools)

#### `ApprovalRequiredToolset` + `ApprovalRequired` + `ToolApproved` + `ToolDenied`

Human-in-the-loop tool gating. Wraps a toolset and raises `ApprovalRequired` before executing a call unless `ctx.tool_call_approved` is `True` or `approval_required_func(ctx, tool_def, tool_args)` returns `False` for that call (default: every call requires approval). You can also raise `ApprovalRequired` (optionally with `metadata=`) directly inside a tool body. Either way this suspends the run — with `DeferredToolRequests` in `output_type`, the agent returns that value instead of raising; the caller inspects `.approvals`, builds `DeferredToolResults` via `.build_results(approve_all=True)` or per-call `ToolApproved(override_args=None)`/`ToolDenied(message=...)`, and resumes with **both** `deferred_tool_results=` and `message_history=result1.all_messages()` (the graph needs prior history to locate the pending tool call — omitting it raises `UserError`). Approved calls carry through `ToolApproved(metadata=...)` into `ctx.tool_call_metadata`.

```python
ApprovalRequired(metadata: dict | None = None)      # Exception
ToolApproved(override_args: dict | None = None)
ToolDenied(message: str = 'The tool call was denied.')

@dataclass
class ApprovalRequiredToolset(WrapperToolset[AgentDepsT]):
    def __init__(self, wrapped: AbstractToolset[AgentDepsT],
                 approval_required_func: Callable[[RunContext, ToolDefinition, dict], bool] = lambda *a: True) -> None: ...
```

```python
from pydantic_ai import Agent, ApprovalRequired, DeferredToolRequests, ToolApproved, ToolDenied, DeferredToolResults
from pydantic_ai.toolsets import FunctionToolset, ApprovalRequiredToolset

agent = Agent('openai:gpt-4o', output_type=[str, DeferredToolRequests])

@agent.tool
def drop_table(ctx, table_name: str) -> str:
    if not ctx.tool_call_approved:
        raise ApprovalRequired
    return f'Table {table_name} dropped.'

result = agent.run_sync('Drop temp_cache.')
if isinstance(result.output, DeferredToolRequests):
    approvals = {c.tool_call_id: ToolApproved() for c in result.output.approvals}
    final = agent.run_sync(
        message_history=result.all_messages(),
        deferred_tool_results=DeferredToolResults(approvals=approvals),
    )

# Or gate an entire toolset at once, without touching individual tool bodies:
gated = ApprovalRequiredToolset(wrapped=FunctionToolset([send_email]))
agent2 = Agent('openai:gpt-4.1', toolsets=[gated], output_type=[str, DeferredToolRequests])
r1 = agent2.run_sync('Send a welcome email to bob@example.com')
if isinstance(r1.output, DeferredToolRequests):
    resumed = agent2.run_sync(
        message_history=r1.all_messages(),
        deferred_tool_results=r1.output.build_results(approve_all=True),
    )
```

#### `DeferredToolRequests` + `DeferredToolResults` + `CallDeferred`

The general async-execution counterpart to approval: `CallDeferred(metadata=...)` suspends a tool
call for an external system to resolve later (a webhook, a Temporal workflow, a queue worker).

```python
DeferredToolRequests(calls: list[ToolCallPart], approvals: list[ToolCallPart], metadata: dict[str, dict])
DeferredToolResults(calls: dict[str, Any], approvals: dict[str, bool | ToolApproved | ToolDenied], metadata: dict)
    .build_results(calls=None, approvals=None, approve_all=False)
    .remaining(results) -> DeferredToolRequests | None
CallDeferred(metadata: dict | None = None)     # Exception
```

```python
from pydantic_ai import Agent, CallDeferred, DeferredToolRequests, DeferredToolResults, ToolReturn

agent = Agent('openai:gpt-4o', output_type=[str, DeferredToolRequests])

@agent.tool_plain
def run_sql_query(sql: str) -> str:
    raise CallDeferred(metadata={'sql': sql})

result = agent.run_sync('Count users created in 2025.')
if isinstance(result.output, DeferredToolRequests):
    pending = result.output
    results = {c.tool_call_id: ToolReturn(content='42') for c in pending.calls}
    final = agent.run_sync(
        message_history=result.all_messages(),
        deferred_tool_results=DeferredToolResults(calls=results),
    )
```

`ExternalToolset` (see Tools & Toolsets) is the toolset-level building block for this pattern —
tools registered there are always deferred (`kind='external'`), resolved via `.calls` and
`build_results(calls={call_id: result_value})`. `HandleDeferredToolCalls` (see Capabilities &
Extensibility) resolves either flow inline instead of requiring a manual resume loop.

#### `safe_download` + SSRF protection

**Module:** `pydantic_ai._ssrf`. The internal function backing `WebFetch`'s local fallback and
`web_fetch_tool` — multi-layered SSRF defence: protocol allowlist (`http`/`https` only), DNS
resolution off the event loop, a hard-coded cloud-metadata-IP blocklist (**always** blocked, even
with `allow_local=True`), 14 IPv4 + 7 IPv6 private ranges (including decoded 6to4/NAT64/ISATAP/
Teredo transition forms), per-redirect-hop re-validation (no DNS-rebinding bypass), and stripping
`Authorization`/`Cookie`/`Proxy-Authorization` on cross-origin redirects.

```python
async def safe_download(url, *, allow_local=False, timeout=30, max_redirects=10,
                         allowed_domains=None, blocked_domains=None, headers=None) -> httpx.Response
```

```python
from pydantic_ai._ssrf import safe_download

async def fetch(url: str) -> str:
    try:
        response = await safe_download(url, allowed_domains=['docs.example.com'], timeout=10)
        return response.text
    except ValueError as e:
        return f'blocked: {e}'
```

Cloud-metadata IPs blocked unconditionally include `169.254.169.254` (AWS IMDS/GCP/Azure/OCI/
DigitalOcean/Hetzner), `169.254.170.2`/`169.254.170.23` (AWS ECS/EKS), `168.63.129.16` (Azure
WireServer), `100.100.100.200` (Alibaba Cloud), `192.0.0.192` (Oracle Cloud), `169.254.42.42`
(Scaleway).

The local fallback path for the `WebFetch` capability (and the standalone `web_fetch_tool()`)
routes every fetch through this same function: `allow_local_urls=False` by default blocks
internal/loopback addresses, and `allowed_domains`/`blocked_domains` are enforced by the local tool
itself (not just the native provider tool), so an allow-list stays effective even on models
without native `WebFetchTool` support.

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import WebFetch

internal_agent = Agent('openai:gpt-4o', capabilities=[
    WebFetch(local=True, allowed_domains=['docs.mycompany.com', 'api.mycompany.com']),
])
```

---

### UI / A2A / Adapters (AG-UI, Vercel AI, agent_to_a2a)

#### `UIAdapter` + `UIEventStream` (+ `StateDeps`/`StateHandler`/`OnCompleteFunc`)

**Module:** `pydantic_ai.ui`. Abstract base every frontend protocol adapter (`AGUIAdapter`,
`VercelAIAdapter`) extends. Owns the request-security policy: `manage_system_prompt: Literal['server',
'client']` (default `'server'` strips client-sent system prompts and auto-adds
`ReinjectSystemPrompt`); `allowed_file_url_schemes` (default `{'http','https'}` only — widen only
after auditing IAM exposure for `s3://`/`gs://`); `allowed_file_url_force_download`;
`preserve_file_data` (uploaded-file round-trip fidelity).

```python
class UIAdapter(ABC, Generic[RunInputT, MessageT, EventT, AgentDepsT, OutputDataT]):
    agent: AbstractAgent; run_input: RunInputT
    manage_system_prompt: Literal['server', 'client'] = 'server'
    allowed_file_url_schemes: frozenset[str] = frozenset({'http', 'https'})

    @classmethod
    async def dispatch_request(cls, request, *, agent, **kwargs) -> Response: ...
```

```python
from starlette.applications import Starlette
from starlette.routing import Route
from pydantic_ai import Agent
from pydantic_ai.ui.ag_ui import AGUIAdapter

agent = Agent('anthropic:claude-sonnet-4-6', system_prompt='You are helpful.')

async def handle(request):
    return await AGUIAdapter.dispatch_request(request, agent=agent)

app = Starlette(routes=[Route('/', handle, methods=['POST'])])
```

#### `AGUIAdapter` + `_AGUIFrontendToolset` + interrupt handling

Implements the [AG-UI protocol](https://github.com/ag-ui-protocol/ag-ui): converts `RunAgentInput`'s `Message`s to `ModelMessage`s and streams `BaseEvent`s back. `ag_ui_version` gates event shape: `< 0.1.13` emits `THINKING_*` events; `≥ 0.1.13` emits `REASONING_*` with round-trippable encrypted metadata; `≥ 0.1.15` emits typed multimodal input content instead of a generic binary blob. AG-UI tools declared in the request are exposed via `_AGUIFrontendToolset` (an `ExternalToolset` subclass). When `ag-ui-protocol >= 0.1.19` (`HAS_INTERRUPTS`), `approval_to_interrupt(call, metadata)` converts a pending `ToolCallPart` into an `Interrupt` for the frontend (with a `response_schema` describing `{approved, editedArgs?, reason?}`), and `resume_entry_to_approval(entry)` converts the client's `ResumeEntry` back into `ToolApproved`/`ToolDenied` — denying by default on any ambiguous payload (`status='cancelled'`, missing payload, or `approved` not exactly `True`). The deprecated `handle_ag_ui_request` helper and the `AGUIApp` Starlette wrapper are both superseded by `AGUIAdapter.dispatch_request(request, agent=agent)`.

```python
AGUIAdapter(agent, run_input, *, ag_ui_version=DEFAULT_AG_UI_VERSION, manage_system_prompt='server', ...)

HAS_INTERRUPTS: bool
def approval_to_interrupt(call: ToolCallPart, metadata: dict) -> Interrupt: ...
def resume_entry_to_approval(entry: ResumeEntry) -> DeferredToolApprovalResult: ...
```

```python
from fastapi import FastAPI, Request
from pydantic_ai import Agent
from pydantic_ai.ui.ag_ui import AGUIAdapter

app = FastAPI()
agent = Agent('anthropic:claude-sonnet-4-6')

@app.post('/agent')
async def run_agent(request: Request):
    return await AGUIAdapter.dispatch_request(request, agent=agent, manage_system_prompt='server')
```

#### `VercelAIAdapter` + `VercelAIEventStream`

A `UIAdapter` subclass speaking the Vercel AI SDK Data Stream Protocol. Parses inbound `RequestData` (`useChat`/`useCompletion` bodies) and emits `StartChunk` → `TextStartChunk`/`TextDeltaChunk`/`TextEndChunk` → `ToolInput*Chunk`/`ToolOutput*Chunk` → `FinishChunk` → `DoneChunk`. `sdk_version: Literal[5, 6] = 5` — v6 additionally streams `ToolApprovalRequestChunk`s so a frontend can render HITL approval prompts for an `ApprovalRequiredToolset`. `load_messages()`/`dump_messages()` round-trip `UIMessage`s to `ModelMessage`s for storage.

```python
@dataclass
class VercelAIAdapter(UIAdapter[RequestData, UIMessage, BaseChunk, AgentDepsT, OutputDataT]):
    sdk_version: Literal[5, 6] = 5
    @classmethod
    async def dispatch_request(cls, request, *, agent, sdk_version=5, **kwargs): ...
```

```python
from pydantic_ai import Agent
from pydantic_ai.ui.vercel_ai import VercelAIAdapter
from pydantic_ai.toolsets.approval_required import ApprovalRequiredToolset
from pydantic_ai import FunctionToolset

toolset = FunctionToolset()
@toolset.tool
async def delete_record(record_id: str) -> str: return f'Deleted {record_id}'

agent = Agent('openai:gpt-4o', toolsets=[ApprovalRequiredToolset(toolset)])

async def handle(request):
    return await VercelAIAdapter.dispatch_request(request, agent=agent, sdk_version=6)
```

#### Web UI API — `create_api_app` + `ModelInfo` + `BuiltinToolInfo` + `ConfigureFrontend` + `ChatRequestExtra`

**Module:** `pydantic_ai.ui._web.api`. Backend for `Agent.to_web()` (see Agents & Execution Core) — a Starlette app with `POST /chat`, `OPTIONS /chat`, `GET /configure`, `GET /health`. `models=` accepts a sequence or a `{label: model}` mapping (mapping keys become the picker's display labels). All response models serialise with `alias_generator=to_camel` (`builtinTools`, not `builtin_tools`). `ModelInfo`/`BuiltinToolInfo` (`id`, `name`) populate `ConfigureFrontend.models`/`builtin_tools`, served at `GET /configure`; `ChatRequestExtra` (`model`, `builtin_tools`) carries the frontend's per-request model/tool selection to `POST /chat`.

```python
create_api_app(agent, models=[...], native_tools=[...]) -> Starlette

class ModelInfo(BaseModel, alias_generator=to_camel): id: str; name: str
class ConfigureFrontend(BaseModel, alias_generator=to_camel):
    models: list[ModelInfo]
    builtin_tools: list[BuiltinToolInfo]
class ChatRequestExtra(BaseModel, extra='ignore', alias_generator=to_camel):
    model: str | None = None
    builtin_tools: list[str] = []
```

```python
import os
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.ui._web.api import create_api_app

agent = Agent(OpenAIModel('gpt-4o'), system_prompt='You are a helpful assistant.')
app = create_api_app(agent, models={'GPT-4o (fast)': 'openai:gpt-4o', 'GPT-4o-mini (cheap)': 'openai:gpt-4o-mini'})
```

#### `agent_to_a2a` / `AgentWorker` — see Durable Execution & Integrations

Fully removed from `pydantic_ai`; use `fasta2a.pydantic_ai.agent_to_a2a` from the
independently-maintained `fasta2a` package instead. Documented once, in the Durable Execution &
Integrations section, to avoid duplication.


---

## Revision History

| Version | Date | Changes | Reviewer |
|---------|------|----------|----------|
| 2.33.0 | August 21, 2026 | Version bumped 2.31.0 → 2.33.0; `Latest:` header and `**Version:**` prose updated throughout, including inline `2.31.0` references. Added a new **Class & API Reference** section (16 subsections: Agents & Execution Core, Models & Providers, Tools & Toolsets, Native/Built-in Tools, Streaming & Events, Structured Output, Messages & Multimodal Content, Concurrency/Usage & Limits, Hooks/Middleware & Lifecycle, Capabilities & Extensibility, Durable Execution & Integrations, Persistence & Graph Support, Testing & Evaluation, Error Handling & Retries, Security, UI/A2A/Adapters) consolidating the 44 separate `pydantic_ai_class_deep_dives*.md` / `pydantic_ai_advanced_classes_part2.md` / `pydantic_ai_source_code_deep_dive.md` volumes, verified against installed pydantic-ai 2.33.0; those 44 files were deleted and `index.mdx` updated to match. | Claude routine |
| 1.107.0 | June 21, 2026 | Version bumped 1.104.0 → 1.107.0 (three minor releases: 1.105.0, 1.106.0, 1.107.0). New features documented: `RunContext` additions (capabilities, loaded_capability_ids, discovered_tool_names, model_settings, metadata, tool_call_metadata); `AgentSpec` YAML/JSON agent configuration; `TemplateStr` Handlebars system prompts; `DeferredToolRequests`/`CallDeferred` async human-in-the-loop; `SkipModelRequest`/`SkipToolExecution`/`SkipToolValidation` hook short-circuits; `ConcurrencyLimiter` observability enhancements. New Vol. 22 class deep dives added covering 10 class groups verified against installed pydantic-ai 1.107.0. All top-level exports confirmed; no DeprecationWarnings. | Claude routine |
| 1.104.0 | May 29, 2026 | Version bumped 1.102.0 → 1.104.0 (two minor releases: 1.103.0, 1.104.0); `Latest:` header and `**Version:**` prose updated; revision history entry added. All core guide symbols verified with `-W error::DeprecationWarning` against installed `pydantic-ai==1.104.0` (`.routine-envs/check-0529-pydantic`); all PASS. 178 top-level exports confirmed. | Claude routine |
| 1.102.0 | May 23, 2026 | Version bumped 1.101.0 → 1.102.0; `Latest:` header and `**Version:**` prose updated; revision history entry added. All core guide symbols verified with `-W error::DeprecationWarning` against installed `pydantic-ai==1.102.0` (`.routine-envs/check-0523-pydantic`); all PASS. 179 top-level exports confirmed; API surface unchanged from 1.101.0. | Claude routine |
| 1.101.0 | May 22, 2026 | Version bumped 1.99.0 → 1.101.0 (two minor releases: 1.100.0, 1.101.0); `Latest:` header and `**Version:**` prose updated; `Installed` comments in snippets updated; revision history entry added. All core guide symbols (`Agent`, `RunContext`, `ModelRetry`, `AgentRunResult`, `StreamedRunResult`, `UsageLimits`, `RunUsage`, `capture_run_messages`, `limit_model_concurrency`, `ConcurrencyLimiter`) verified with `-W error::DeprecationWarning` against installed `pydantic-ai==1.101.0` (`.routine-envs/check-0522-pydantic`); all PASS. | Claude routine |
| 1.99.0 | May 20, 2026 | Version bumped 1.98.0 → 1.99.0; `Latest:` header and `**Version:**` prose updated; revision history entry added. All core guide symbols (`Agent`, `RunContext`, `ModelRetry`, `AgentRunResult`, `StreamedRunResult`, `UsageLimits`, `RunUsage`, `capture_run_messages`, `limit_model_concurrency`, `ConcurrencyLimiter`) verified with `-W error::DeprecationWarning` against installed `pydantic-ai==1.99.0` (`.routine-envs/check-0520-pydantic`); all PASS. | Claude routine |
| 1.98.0 | May 19, 2026 | Two minor releases (1.97.0, 1.98.0). `pydantic_ai.ag_ui` module deprecated in 1.98.x — emits `PydanticAIDeprecationWarning`; new canonical path is `pydantic_ai.ui.ag_ui.AGUIAdapter`. AG UI section in this guide updated with deprecation note and migration path. New `pydantic_ai.common_tools` module (DuckDuckGo, Exa, Tavily, WebFetch, ImageGeneration providers); requires optional extras. All core guide symbols verified against installed `pydantic-ai 1.98.0` (`.routine-envs/check-0519-py`); no `DeprecationWarning` emissions on standard imports. | Claude routine |
| 1.96.0 | May 14, 2026 | Minor release; new concurrency management API: `ConcurrencyLimiter(max_running, max_queued=None)` and `limit_model_concurrency(model, limiter)`. All guide-referenced symbols verified against installed `pydantic-ai 1.96.0` (`.routine-envs/check-0514-py`); no `DeprecationWarning` emissions. |
| 1.95.0 | May 13, 2026 | Minor release; all guide-referenced symbols (`Agent`, `RunContext`, `ModelRetry`, `AgentRunResult`, `StreamedRunResult`, `UsageLimits`, `RunUsage`, `capture_run_messages`) verified with `-W error::DeprecationWarning` against installed `pydantic-ai 1.95.0` (`.routine-envs/check-0513-py`); no warnings. Additional exports `AgentRunResultEvent`, `AgentEventStream` confirmed in installed source. |
| 1.94.0 | May 12, 2026 | Minor release; new top-level exports: `AgentRun`, `AgentRunResult`, `StreamedRunResultSync`. All guide-referenced symbols (`Agent`, `RunContext`, `ModelRetry`, `AgentRunResult`, `StreamedRunResult`, `UsageLimits`, `RunUsage`, `capture_run_messages`) verified with `-W error::DeprecationWarning` against installed `pydantic-ai 1.94.0` (`.routine-envs/check-0512-py`); no warnings. |
| 1.93.0 | May 9, 2026 | Three minor releases (1.91.0, 1.92.0, 1.93.0). Breaking change: `TestModel` removed from `pydantic_ai` top-level — correct path is `from pydantic_ai.models.test import TestModel` (all guide pages already use this path). New top-level exports confirmed: `AgentSpec`, `UploadedFile`, `WebSearchUserLocation`, `DeferredLoadingToolset`. All existing symbols confirmed present in installed 1.93.0 (`.routine-envs/check-0509-py`) with no DeprecationWarnings. |
| 1.90.0 | May 5, 2026 | Patch release; `DeferredToolCalls` in `pydantic_ai.output` marked `@deprecated` — use `DeferredToolRequests` (guides already use the correct API). Version confirmed against installed `pydantic-ai 1.90.0` (`.routine-envs/check-0505`); `Agent` (TestModel), `FunctionToolset`, `DeferredToolRequests`, `HandleDeferredToolCalls`, `ImageGenerationTool`, `MemoryTool`, `XSearchTool`, `RenamedToolset`, `WrapperToolset` all import successfully with no DeprecationWarnings. |
| 1.89.1 | May 2, 2026 | Patch release; maintenance and dependency updates. Version confirmed against installed `pydantic-ai 1.89.1` (`.routine-envs/check-pydantic-0502`); `Agent`, `OpenAIModel` imports verified with `-W error::DeprecationWarning`. |
| 1.89.0 | May 1, 2026 | Patch release; maintenance and dependency updates. Version confirmed against installed `pydantic-ai 1.89.0` (`.routine-envs/check-pydantic-0501`); `Agent`, `OpenAIModel` imports verified with `-W error::DeprecationWarning`. |
| 1.88.0 | April 29, 2026 | Patch release; maintenance and dependency updates. Version confirmed against installed `pydantic-ai 1.88.0` (`.routine-envs/main-py-0429`); `Agent`, `OpenAIModel` imports verified. |
| 1.87.0 | April 25, 2026 | Expanded Capabilities API: 9 new capability classes (`WrapperCapability`, `ReinjectSystemPrompt`, `ProcessHistory`, `ProcessEventStream`, `HandleDeferredToolCalls`, `IncludeToolReturnSchemas`, `PrefixTools`, `PrepareTools`, `SetToolMetadata`); new type aliases (`RawToolArgs`, `ValidatedToolArgs`, `CapabilityRef`, `CapabilityPosition`, `CapabilityOrdering`); `CAPABILITY_TYPES` registry. New capabilities section added. All symbols confirmed against installed 1.87.0 (`pydantic_ai/capabilities/__init__.py`). |
| 1.86.1 | April 24, 2026 | Patch fix for Capabilities API. Snippets executed against installed 1.86.1; `Hooks`, `ModelProfile`, `DEFAULT_PROFILE` all import successfully. New Capabilities API section added to this guide. |
| 1.86.0 | April 23, 2026 | Introduces `capabilities` parameter on `Agent.__init__`; new `pydantic_ai.capabilities` module (`Hooks`, `AbstractCapability`, `CombinedCapability`, `HistoryProcessor`, `Thinking`, `ThreadExecutor`, `WebFetch`, `WebSearch`, `ImageGeneration`, `MCP`, `Toolset`); new `pydantic_ai.profiles` module (`ModelProfile`, `ModelProfileSpec`, `DEFAULT_PROFILE`); new `pydantic_ai.ui` module (`UIAdapter`, `UIEventStream`, `MessagesBuilder`). |
| 1.85.1 | April 22, 2026 | Patch fix; `UrlContextTool` marked deprecated (use `WebFetchTool`). Built-in tools, embeddings, AG UI, and `ApprovalRequiredToolset` verified against installed package. `pydantic_ai.common_tools` stub corrected to `pydantic_ai.builtin_tools` with correct class names. Snippets executed against 1.85.1. |
| 1.85.0 | April 21, 2026 | New embeddings API (`Embedder`, `EmbeddingModel`, `EmbeddingSettings`); AG UI adapter (`AGUIApp`, `AGUIAdapter`, `run_ag_ui`); `ApprovalRequired`/`ApprovalRequiredToolset` for HITL; `DeferredLoadingToolset`; `UrlContextTool` deprecated in favour of `WebFetchTool` |
| 1.84.1 | April 18, 2026 | Skip tool hooks for internal output tools; always pass dict-shaped validated args to hooks for single-`BaseModel` tools |
| 1.84.0 | April 17, 2026 | `OllamaModel` subclass (fixes structured output on Ollama Cloud); `XSearchTool`/`FileSearchTool` for xAI (Grok); `FastMCPToolset` per-call metadata injection; Bedrock prompt cache TTL; Claude Opus 4.7 support (`anthropic:claude-opus-4-7`); stateful `OpenAICompaction`; fix exponential-time regex in Google `FileSearchTool` |
| 1.83.0 | April 16, 2026 | Hard removal of all `result_*` → `output_*` renames (breaking); `EvaluationReport` API; pydantic-graph expansion with branching/looping; `defer_loading` for lazy model init; `ThreadExecutor` for sync-in-async tools; smart instruction caching; `CaseLifecycle` hooks; local `WebFetch` tool |
| 1.20.0 | November 2025 | Previous documented version |


