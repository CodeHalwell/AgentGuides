# OpenAI Agents SDK: Critical 2025 Features and Updates

**Status:** Production-Ready
**Last Updated:** January 2025
**SDK Version:** v0.2.9+
**Language:** Python

---

## Overview

The OpenAI Agents SDK is the **official production-ready replacement** for the experimental Swarm framework. This guide covers all critical 2025 features that make the Agents SDK the definitive choice for building agentic AI applications.

---

## Table of Contents

1. [Swarm Replacement](#swarm-replacement)
2. [Core Primitives (2025)](#core-primitives-2025)
3. [Built-in Tracing and Visualization](#built-in-tracing-and-visualization)
4. [Function Tools with Automatic Schema Generation](#function-tools-with-automatic-schema-generation)
5. [Provider-Agnostic Support](#provider-agnostic-support)
6. [Guardrails System](#guardrails-system)
7. [Session Management](#session-management)
8. [MCP Integration](#mcp-integration)
9. [Evaluation and Fine-tuning Integration](#evaluation-and-fine-tuning-integration)
10. [Production-Ready Features](#production-ready-features)

---

## Swarm Replacement

### Official Statement

**The Agents SDK is the production-ready evolution of Swarm.** OpenAI has deprecated Swarm in favour of the Agents SDK, which provides:

- ✅ **Production Stability**: Stable API with semantic versioning
- ✅ **Active Maintenance**: Regular updates and security patches
- ✅ **Comprehensive Features**: Guardrails, tracing, sessions, MCP
- ✅ **Enterprise Support**: Documentation, examples, community

### Migration Path

```python
# BEFORE: Swarm (Experimental)
from swarm import Swarm, Agent as SwarmAgent

client = Swarm()
agent = SwarmAgent(name="Assistant", instructions="Help users")
result = client.run(agent, messages=[{"role": "user", "content": "Hello"}])

# AFTER: Agents SDK (Production)
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="Help users")
result = await Runner.run(agent, "Hello")
print(result.final_output)
```

**See:** [Complete Swarm Migration Guide](openai_agents_sdk_swarm_migration_guide.md)

---

## Core Primitives (2025)

The Agents SDK provides six lightweight primitives that compose to build complex systems:

### 1. Agent

An LLM configured with instructions, tools, and guardrails.

```python
from agents import Agent

agent = Agent(
    name="Research Assistant",
    instructions="""You are a thorough research assistant.
        - Provide detailed, well-sourced information
        - Cite sources when available
        - Acknowledge uncertainty appropriately""",
    model="gpt-4o",
    temperature=0.7,
    max_tokens=2048
)
```

### 2. Handoff

Mechanism for transferring control between agents.

```python
from agents import Agent

specialist_agent = Agent(
    name="Technical Specialist",
    handoff_description="Handles technical support queries",
    instructions="Resolve technical issues with step-by-step guidance"
)

triage_agent = Agent(
    name="Triage",
    instructions="Route customers to appropriate specialist",
    handoffs=[specialist_agent]  # Automatic delegation
)
```

### 3. Guardrail

Input/output validation for safety and compliance.

```python
from agents import Agent, input_guardrail, GuardrailFunctionOutput

@input_guardrail
async def content_safety_check(ctx, agent, input_data):
    """Validate input safety."""
    unsafe_patterns = ["harmful", "dangerous", "illegal"]

    is_unsafe = any(
        pattern in str(input_data).lower()
        for pattern in unsafe_patterns
    )

    return GuardrailFunctionOutput(
        output_info={"safety_check": "fail" if is_unsafe else "pass"},
        tripwire_triggered=is_unsafe
    )

agent = Agent(
    name="Safe Assistant",
    input_guardrails=[content_safety_check]
)
```

### 4. Tool

Functions agents can call with automatic schema generation.

```python
from agents import Agent, function_tool
from pydantic import BaseModel, Field

class BookingData(BaseModel):
    customer_name: str = Field(..., description="Full name")
    email: str = Field(..., pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")

@function_tool
def book_appointment(data: BookingData) -> dict:
    """Book appointment with automatic Pydantic validation."""
    return {
        "status": "confirmed",
        "booking_id": f"BK-{hash(data.email)}",
        "customer": data.customer_name
    }

agent = Agent(
    name="Booking Agent",
    tools=[book_appointment]
)
```

### 5. Session

Automatic conversation history management.

```python
from agents import Agent, Runner, SQLiteSession

agent = Agent(name="Assistant", instructions="Remember conversation context")
session = SQLiteSession("user_123", "conversations.db")

# Turn 1
result1 = await Runner.run(agent, "My name is Alice", session=session)

# Turn 2 - automatic context retention
result2 = await Runner.run(agent, "What's my name?", session=session)
# Response: "Your name is Alice"
```

### 6. Runner

Orchestrator executing the agent loop.

```python
from agents import Agent, Runner

agent = Agent(name="Assistant")

# Synchronous
result = Runner.run_sync(agent, "What is AI?")

# Asynchronous
result = await Runner.run(agent, "What is AI?")

# Streaming
result = Runner.run_streamed(agent, "Write a poem")
async for event in result.stream_events():
    print(event)
```

---

## Built-in Tracing and Visualization

### Workflow Tracing

Track and visualize agent execution for debugging and optimization.

```python
from agents import Agent, Runner, trace

agent = Agent(name="Data Processor")

# Group related runs
with trace(
    workflow_name="Data Processing Pipeline",
    group_id="batch_2025_01",
    metadata={
        "environment": "production",
        "version": "2.0.0",
        "batch_size": 1000
    }
):
    result1 = await Runner.run(agent, "Process dataset A")
    result2 = await Runner.run(agent, "Process dataset B")
    result3 = await Runner.run(agent, "Generate report")

# View traces at: https://platform.openai.com/traces
```

### Integration with Evaluation Tools

```python
from agents import Agent, Runner, trace

# Automatic integration with OpenAI evaluation tools
with trace("Evaluation Run", metadata={"eval_id": "eval_123"}):
    results = []

    for test_case in test_cases:
        result = await Runner.run(agent, test_case["input"])
        results.append({
            "input": test_case["input"],
            "output": result.final_output,
            "expected": test_case["expected"]
        })

# Traces automatically appear in OpenAI evaluation dashboard
```

### Fine-tuning Integration

```python
from agents import trace

# Collect data for fine-tuning
with trace("Fine-tune Data Collection", metadata={"purpose": "training"}):
    for example in training_examples:
        result = await Runner.run(agent, example["prompt"])

        # Data automatically logged for fine-tuning
        # Access via OpenAI fine-tuning interface
```

### Distillation Tools

```python
from agents import Agent, Runner, trace

# Use tracing for model distillation
expensive_model = Agent(name="GPT-4", model="gpt-4o", temperature=0)
cheap_model = Agent(name="GPT-3.5", model="gpt-3.5-turbo", temperature=0)

with trace("Distillation Comparison"):
    # Generate teacher outputs
    teacher_result = await Runner.run(expensive_model, query)

    # Train student model on teacher outputs
    student_result = await Runner.run(cheap_model, query)

    # Compare results in trace visualization
```

---

## Function Tools with Automatic Schema Generation

### Pydantic-Powered Validation

```python
from agents import Agent, function_tool
from pydantic import BaseModel, Field, validator
from typing import Literal

class FlightBooking(BaseModel):
    origin: str = Field(..., min_length=3, max_length=3, description="Origin airport code")
    destination: str = Field(..., min_length=3, max_length=3)
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    passengers: int = Field(ge=1, le=9)
    class_type: Literal["economy", "business", "first"] = "economy"

    @validator("origin", "destination")
    def validate_airport_code(cls, v):
        if not v.isupper():
            raise ValueError("Airport code must be uppercase")
        return v

@function_tool
def book_flight(booking: FlightBooking) -> dict:
    """Book flight with comprehensive validation."""
    return {
        "confirmation": f"FLIGHT-{booking.origin}-{booking.destination}",
        "status": "confirmed",
        "passengers": booking.passengers
    }

agent = Agent(
    name="Travel Agent",
    instructions="Help book flights",
    tools=[book_flight]
)

# Automatic validation and error handling
result = await Runner.run(agent, "Book flight from LAX to JFK on 2025-03-15 for 2 passengers")
```

### Async Tool Execution

```python
from agents import function_tool
import asyncio

@function_tool
async def fetch_user_data(user_id: str) -> dict:
    """Async tool execution for non-blocking operations."""
    await asyncio.sleep(1)  # Simulate API call
    return {"user_id": user_id, "name": "John Doe"}

@function_tool
async def process_payment(amount: float) -> dict:
    """Async payment processing."""
    await asyncio.sleep(2)  # Simulate payment gateway
    return {"status": "success", "amount": amount}

agent = Agent(
    name="Async Agent",
    tools=[fetch_user_data, process_payment]
)
```

---

## Provider-Agnostic Support

### 100+ LLM Providers via LiteLLM

```bash
pip install 'openai-agents[litellm]'
```

```python
from agents import Agent, Runner
import os

# Set API keys
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
os.environ["GOOGLE_API_KEY"] = "..."

# Use Claude 3.5 Sonnet
claude_agent = Agent(
    name="Claude Assistant",
    model="litellm/anthropic/claude-3-5-sonnet-20240620",
    instructions="You are Claude, an AI assistant created by Anthropic"
)

# Use Gemini 2.0 Flash
gemini_agent = Agent(
    name="Gemini Assistant",
    model="litellm/gemini/gemini-2.0-flash",
    instructions="You are Gemini, Google's AI model"
)

# Use Llama 2
llama_agent = Agent(
    name="Llama Assistant",
    model="litellm/replicate/meta-llama/llama-2-70b-chat",
    instructions="You are Llama 2, Meta's open-source model"
)

# Use Mistral
mistral_agent = Agent(
    name="Mistral Assistant",
    model="litellm/mistral/mistral-large-latest",
    instructions="You are Mistral, a powerful European AI model"
)

# Seamless provider switching
async def compare_models():
    query = "Explain quantum computing in simple terms"

    claude_result = await Runner.run(claude_agent, query)
    gemini_result = await Runner.run(gemini_agent, query)
    llama_result = await Runner.run(llama_agent, query)

    return {
        "claude": claude_result.final_output,
        "gemini": gemini_result.final_output,
        "llama": llama_result.final_output
    }
```

### OpenAI Responses API Integration

```python
from agents import Agent, Runner, ModelSettings
from openai.types.shared import Reasoning

agent = Agent(
    name="Research Agent",
    instructions="Conduct thorough research with reasoning",
    model_settings=ModelSettings(
        reasoning=Reasoning(effort="high")  # Enable reasoning
    )
)

result = await Runner.run(
    agent,
    "Analyze the implications of quantum computing on cryptography"
)

# Access reasoning trace
print("Reasoning Process:")
print(result.output_reasoning)

print("\nFinal Analysis:")
print(result.final_output)
```

### Chat Completions API Support

```python
from agents import Agent, ModelSettings

# Use standard Chat Completions API
agent = Agent(
    name="Assistant",
    model="gpt-4o",
    model_settings=ModelSettings(
        temperature=0.7,
        max_tokens=1000,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.5
    )
)
```

---

## Guardrails System

### Input Guardrails

```python
from agents import Agent, input_guardrail, GuardrailFunctionOutput, InputGuardrailTripwireTriggered
import re

@input_guardrail
async def validate_no_pii(ctx, agent, input_data):
    """Block inputs containing PII."""
    input_text = str(input_data)

    # Check for common PII patterns
    pii_patterns = [
        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
        r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",  # Credit card
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"  # Email
    ]

    for pattern in pii_patterns:
        if re.search(pattern, input_text, re.IGNORECASE):
            return GuardrailFunctionOutput(
                output_info={"pii_detected": True},
                tripwire_triggered=True
            )

    return GuardrailFunctionOutput(
        output_info={"pii_detected": False},
        tripwire_triggered=False
    )

agent = Agent(
    name="Secure Agent",
    instructions="Help users while protecting privacy",
    input_guardrails=[validate_no_pii]
)

try:
    result = await Runner.run(agent, "My SSN is 123-45-6789")
except InputGuardrailTripwireTriggered:
    print("Input blocked: PII detected")
```

### Output Guardrails

```python
from agents import Agent, output_guardrail, GuardrailFunctionOutput, OutputGuardrailTripwireTriggered
from pydantic import BaseModel

class Response(BaseModel):
    content: str

@output_guardrail
async def filter_inappropriate_content(ctx, agent, output):
    """Filter inappropriate output."""
    blocked_terms = ["inappropriate1", "inappropriate2", "offensive"]

    content_lower = output.content.lower()

    for term in blocked_terms:
        if term in content_lower:
            return GuardrailFunctionOutput(
                output_info={"blocked_term": term},
                tripwire_triggered=True
            )

    return GuardrailFunctionOutput(
        output_info={"content_clean": True},
        tripwire_triggered=False
    )

agent = Agent(
    name="Family-Friendly Agent",
    instructions="Provide helpful, appropriate responses",
    output_guardrails=[filter_inappropriate_content],
    output_type=Response
)
```

---

## Session Management

### Multiple Backend Support

```python
from agents import Agent, Runner, SQLiteSession, InMemorySession
from agents.extensions.memory import RedisSession

# SQLite (local, persistent)
sqlite_session = SQLiteSession("user_123", "sessions.db")

# Redis (distributed, high-performance)
redis_session = RedisSession.from_url(
    "user_123",
    url="redis://localhost:6379"
)

# In-Memory (testing, temporary)
memory_session = InMemorySession()

# OpenAI Backend (managed)
from agents import OpenAIConversationsSession
openai_session = OpenAIConversationsSession()

# Use any backend interchangeably
agent = Agent(name="Assistant")
result = await Runner.run(agent, "Hello", session=sqlite_session)
```

### Session Operations

```python
from agents import SQLiteSession

session = SQLiteSession("user_456", "conversations.db")

# Add custom messages
await session.add_items([
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Custom message"}
])

# Get conversation history
items = await session.get_items()
print(f"Total messages: {len(items)}")

# Remove last message (for corrections)
await session.pop_item()

# Clear entire session
await session.clear_session()
```

---

## MCP Integration

### Filesystem MCP

```python
from agents import Agent, Runner
from agents.mcp import MCPServerStdio
from pathlib import Path

async def main():
    async with MCPServerStdio(
        name="Filesystem",
        params={
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                str(Path("./project"))
            ]
        }
    ) as server:
        agent = Agent(
            name="File Assistant",
            instructions="Help manage project files",
            mcp_servers=[server]
        )

        result = await Runner.run(agent, "List all Python files in the project")
        print(result.final_output)
```

### Git MCP

```python
from agents import Agent, Runner
from agents.mcp import MCPServerStdio
from pathlib import Path

async def main():
    async with MCPServerStdio(
        name="Git",
        params={
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-git",
                str(Path("./repository"))
            ]
        }
    ) as server:
        agent = Agent(
            name="Code Reviewer",
            instructions="Review git changes and commits",
            mcp_servers=[server]
        )

        result = await Runner.run(agent, "Show recent commits and their changes")
        print(result.final_output)
```

### Hosted MCP with HTTP

```python
from agents import Agent, Runner
from agents.mcp import MCPServerStreamableHttp
import os

async def main():
    async with MCPServerStreamableHttp(
        name="GitHub API",
        params={
            "url": "https://api.github.com/mcp",
            "headers": {"Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}"},
            "timeout": 30
        }
    ) as server:
        agent = Agent(
            name="GitHub Assistant",
            instructions="Help manage GitHub repositories",
            mcp_servers=[server]
        )

        result = await Runner.run(agent, "List top Python repositories")
        print(result.final_output)
```

---

## Evaluation and Fine-tuning Integration

### Evaluation Workflow

```python
from agents import Agent, Runner, trace

evaluation_agent = Agent(
    name="Evaluation Agent",
    instructions="Answer questions accurately",
    model="gpt-4o"
)

# Collect evaluation data
test_cases = [
    {"input": "What is Python?", "expected": "programming language"},
    {"input": "Explain ML", "expected": "machine learning"},
]

with trace("Evaluation Run", metadata={"eval_type": "accuracy"}):
    results = []

    for case in test_cases:
        result = await Runner.run(evaluation_agent, case["input"])

        results.append({
            "input": case["input"],
            "output": result.final_output,
            "expected": case["expected"],
            "match": case["expected"].lower() in result.final_output.lower()
        })

    accuracy = sum(1 for r in results if r["match"]) / len(results)
    print(f"Accuracy: {accuracy:.2%}")

# Access results in OpenAI evaluation tools
```

### Fine-tuning Data Collection

```python
from agents import Agent, Runner, trace

# Collect high-quality examples for fine-tuning
with trace("Fine-tune Training Data", metadata={"purpose": "training_data"}):
    for prompt in training_prompts:
        result = await Runner.run(agent, prompt)

        # Data automatically logged with trace
        # Export via OpenAI fine-tuning interface
        # Format: {prompt: ..., completion: result.final_output}
```

---

## Production-Ready Features

### Error Handling

```python
from agents import Agent, Runner, MaxTurnsExceeded, ModelBehaviorError

agent = Agent(name="Robust Agent")

try:
    result = await Runner.run(agent, query, max_turns=5)
except MaxTurnsExceeded:
    print("Agent exceeded maximum turns - infinite loop detected")
except ModelBehaviorError as e:
    print(f"Model behavior error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Token Usage Tracking

```python
from agents import Agent, Runner

agent = Agent(name="Assistant")
result = await Runner.run(agent, "Explain AI")

# Access usage statistics
if hasattr(result, "usage"):
    print(f"Input tokens: {result.usage.input_tokens}")
    print(f"Output tokens: {result.usage.output_tokens}")
    print(f"Total tokens: {result.usage.input_tokens + result.usage.output_tokens}")
```

### Cost Optimization

```python
from agents import Agent, ModelSettings

# Use cheaper model for simple queries
fast_agent = Agent(
    name="Fast Agent",
    model="gpt-4o-mini",
    model_settings=ModelSettings(temperature=0, max_tokens=200)
)

# Use premium model for complex queries
premium_agent = Agent(
    name="Premium Agent",
    model="gpt-4o",
    model_settings=ModelSettings(temperature=0.7, max_tokens=2000)
)

# Route based on complexity
def select_agent(query: str):
    if len(query) < 50 and "simple" in query.lower():
        return fast_agent
    return premium_agent
```

---

## Summary

The OpenAI Agents SDK (2025) provides:

✅ **Production-Ready** replacement for Swarm
✅ **Core Primitives** (Agents, Handoffs, Guardrails, Sessions, Tools, Runner)
✅ **Built-in Tracing** with visualization and evaluation integration
✅ **Function Tools** with Pydantic-powered validation
✅ **Provider-Agnostic** support for 100+ LLMs
✅ **Guardrails System** for input/output validation
✅ **Session Management** with multiple backends
✅ **MCP Integration** for filesystem, git, and hosted services
✅ **Evaluation & Fine-tuning** integration
✅ **Production Features** (error handling, cost optimization, observability)

**Recommendation**: Use OpenAI Agents SDK for all production agentic applications.

---

**Last Updated:** January 2025
**SDK Version:** v0.2.9+
**Next:** [Swarm Migration Guide](openai_agents_sdk_swarm_migration_guide.md)
