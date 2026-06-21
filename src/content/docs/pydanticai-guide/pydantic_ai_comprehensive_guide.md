---
title: "Pydantic AI: Comprehensive Technical Guide"
description: "Version: 1.107.0 (June 2026) Framework: Pydantic AI - GenAI Agent Framework, the Pydantic Way Author Notes: Exhaustive technical documentation with production patterns, type safety"
framework: pydanticai
---

Latest: 1.107.0 | Updated: June 21, 2026
# Pydantic AI: Comprehensive Technical Guide
## From Beginner to Expert Level

**Version:** 1.107.0 (June 2026)  
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

# Check token usage
print(result.usage())
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

# Usage limits
from pydantic_ai import UsageLimits

result = article_agent.run_sync(
    'Write about type safety in Python',
    usage_limits=UsageLimits(
        request_limit=2,  # Max 2 API calls
        total_tokens_limit=2000  # Max 2000 tokens total
    )
)
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

# 3. Access usage information
usage = result.usage()
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

# Check usage
usage = result_full.usage()
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

### Built-in Tool Library

Pydantic AI provides built-in tools that delegate to model-native capabilities (e.g. OpenAI's built-in tools).
They are passed via the `builtin_tools` parameter on `Agent`. All are importable directly from `pydantic_ai`.

**Supported built-in tools (v1.85.x):**

| Tool | Import | Notes |
|------|--------|-------|
| `WebSearchTool` | `from pydantic_ai import WebSearchTool` | Model-native web search |
| `WebFetchTool` | `from pydantic_ai import WebFetchTool` | Fetch and read URL content |
| `CodeExecutionTool` | `from pydantic_ai import CodeExecutionTool` | Sandboxed code execution |
| `ImageGenerationTool` | `from pydantic_ai import ImageGenerationTool` | Image generation |
| `FileSearchTool` | `from pydantic_ai import FileSearchTool` | File/vector-store search (requires config) |
| `MemoryTool` | `from pydantic_ai import MemoryTool` | Persistent memory (requires config) |
| `MCPServerTool` | `from pydantic_ai import MCPServerTool` | MCP server integration (requires config) |
| `XSearchTool` | `from pydantic_ai import XSearchTool` | xAI (Grok) web search |

> **Deprecation (v1.85.0):** `UrlContextTool` is deprecated — use `WebFetchTool` instead.

```python
# Installed: pydantic-ai==1.101.0
# Verified against installed package — run with uv pip install pydantic-ai==1.99.0
from pydantic_ai import Agent, WebSearchTool, WebFetchTool, CodeExecutionTool

agent = Agent(
    'openai:gpt-4o',
    builtin_tools=[
        WebSearchTool(),                    # model-native web search
        WebFetchTool(enable_citations=True), # fetch URL content with citation metadata
        CodeExecutionTool(),                 # sandboxed code execution
    ]
)

# The agent can now search the web, fetch pages, and execute code
result = agent.run_sync('Search for the latest Python release and show a hello-world snippet')
print(result.output)
```

Tools requiring additional provider configuration (`FileSearchTool`, `MemoryTool`, `MCPServerTool`) must be
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
- `pydantic_ai_class_deep_dives.md` - Source-verified deep dives: AgentRun.iter(), AgentRunResult, ConcurrencyLimiter, Direct API, capture_run_messages, format_as_xml, common_tools, ExternalToolset, FilteredToolset, FunctionToolset+instructions

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
from pydantic_ai import AgentSpec

# Load from file
spec = AgentSpec.from_file('agents/support.yaml')
agent = spec.to_agent()

# Or parse inline YAML
spec = AgentSpec.from_text("""
model: openai:gpt-4o
name: support-agent
instructions: You are a helpful support agent. Respond in {{language}}.
model_settings:
  temperature: 0.3
retries: 3
""")
agent = spec.to_agent()
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

## Revision History

| Version | Date | Changes | Reviewer |
|---------|------|----------|----------|
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


