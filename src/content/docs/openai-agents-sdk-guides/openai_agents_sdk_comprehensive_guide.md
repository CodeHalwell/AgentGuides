---
title: "OpenAI Agents SDK: Comprehensive Technical Guide"
description: "Single-page reference for the Python OpenAI Agents SDK: agents, runner, handoffs, tools, MCP, structured output, guardrails, sessions, tracing, realtime and voice, sandbox agents and testing."
sidebar:
  label: "Comprehensive guide"
  order: 1
framework: "openai-agents-sdk"
language: "python"
---

> Verified against `openai-agents` %%version=openai-agents%% (checked %%versions-checked%%). Python 3.10+, `openai` v3.

A single-page reference for the Python SDK, from installation to sandbox agents. Lifecycle hooks, tool guardrails and approvals have their own page: [Hooks, guardrails and approvals](/openai-agents-sdk-guides/openai_agents_sdk_middleware_python/). Running in production is covered in the [production guide](/openai-agents-sdk-guides/openai_agents_sdk_production_guide/).

## Table of Contents

1. [Installation and Environment Setup](#installation-and-environment-setup)
2. [Design Philosophy](#design-philosophy)
3. [Core Primitives](#core-primitives)
4. [Simple Agents](#simple-agents)
5. [Multi-Agent Systems](#multi-agent-systems)
6. [Tools Integration](#tools-integration)
7. [Structured Outputs](#structured-outputs)
8. [Model Context Protocol](#model-context-protocol)
9. [Agentic Patterns](#agentic-patterns)
10. [Guardrails](#guardrails)
11. [Memory Systems](#memory-systems)
12. [Context Engineering](#context-engineering)
13. [Responses API Integration](#responses-api-integration)
14. [Tracing and Observability](#tracing-and-observability)
15. [Realtime and Voice Agents](#realtime-and-voice-agents)
16. [Model Providers](#model-providers)
17. [Advanced Topics](#advanced-topics)

---

## Installation and Environment Setup

### Basic Installation

The OpenAI Agents SDK requires Python 3.10 or newer and can be installed via pip:

```bash
pip install "openai-agents==%%version=openai-agents%%"
```

> **Requirements:** Python 3.10 or newer. Since 0.21.0 the SDK requires `openai>=3.0.0,<4` (the `openai` v3 client uses HTTPX2); pip resolves this for you. See [Breaking changes by version](#breaking-changes-by-version) below.

For development with optional features, install with extras:

```bash
# Voice support for real-time applications
pip install 'openai-agents[voice]'

# Redis session support for distributed deployments
pip install 'openai-agents[redis]'

# SQLAlchemy-backed sessions (Postgres, MySQL, SQLite via async drivers)
pip install 'openai-agents[sqlalchemy]'

# LiteLLM support for multi-provider model access
pip install 'openai-agents[litellm]'

# Sandbox backends (Docker shown; also e2b, modal, daytona, vercel, cloudflare, blaxel, runloop)
pip install 'openai-agents[docker]'
```

Other extras: `[realtime]` (Realtime agents), `[encrypt]` (`EncryptedSession`), `[mongodb]` and `[dapr]` (session backends), `[temporal]` (durable execution with Temporal), `[any-llm]` (the any-llm provider adapter), `[viz]` (`draw_graph` agent diagrams) and `[s3]`. There is no `[all]` extra; install the ones you need. Hosted tools such as `CodeInterpreterTool` and `WebSearchTool` need no extra.

### Using uv Package Manager

The recommended approach for modern Python projects uses uv:

```bash
uv init
uv add openai-agents
uv add 'openai-agents[voice]'
uv add 'openai-agents[redis]'
```

### Environment Configuration

The SDK requires the OpenAI API key configured as an environment variable:

```bash
export OPENAI_API_KEY=sk-your-actual-key-here
```

For Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="sk-your-actual-key-here"
```

Create a `.env` file for development (do not commit to version control):

```bash
OPENAI_API_KEY=sk-your-actual-key-here
OPENAI_ORG_ID=org-your-org-id
```

Load the `.env` file in your application:

```python
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
org_id = os.getenv("OPENAI_ORG_ID")
```

### Breaking changes by version

The SDK uses `0.Y.Z` versioning: a minor (`Y`) bump may break public interfaces, so pin a minor range in production. The release table, with the breaking changes from 0.4 onwards, is on the [overview page](/openai-agents-sdk-guides/#whats-new).

---

## Design Philosophy

### Lightweight Primitives vs. Heavy Abstractions

The OpenAI Agents SDK intentionally favours simplicity and minimalism. Rather than providing a kitchen-sink framework, it offers lightweight primitives that compose naturally:

**Philosophy Core Principles:**

1. **Few primitives**: agents, tools, handoffs, guardrails, sessions and the runner
2. **Composition**: larger systems are built by combining those primitives, or by plain Python code that calls `Runner.run`
3. **Python-native**: async/await, decorators, Pydantic models and type hints instead of a configuration language
4. **Observable by default**: every run is traced unless you turn tracing off

### Coming from Swarm

The SDK replaced the experimental Swarm library. The agent and handoff ideas carry over; what is new is the runner loop, guardrails, sessions, tracing, MCP and structured output. The [Swarm migration guide](/openai-agents-sdk-guides/openai_agents_sdk_swarm_migration_guide/) has a side-by-side mapping.

---

## Core Primitives

### Agent

The foundational primitive representing an LLM configured with instructions, tools, and guardrails.

**Basic Structure:**

```python
from agents import Agent, ModelSettings

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant that provides accurate information",
    model="%%model=openai%%",
    model_settings=ModelSettings(temperature=0.7),
)
```

Sampling parameters such as `temperature` and `max_tokens` live on `ModelSettings`, not on `Agent` itself. If `model` is omitted, the SDK uses its default (`gpt-5.6-luna` at the time of writing) or the `OPENAI_DEFAULT_MODEL` environment variable.

**Complete Agent Definition:**

```python
from agents import Agent, ModelSettings, function_tool
from pydantic import BaseModel

class TaskResult(BaseModel):
    status: str
    output: str

@function_tool
def execute_task(task_name: str) -> str:
    """Execute a named task."""
    return f"Executed {task_name}"

agent = Agent(
    name="Task Manager",
    instructions="""You manage and execute tasks. 
    When given a task, use the execute_task tool to run it.
    Provide updates on completion status.""",
    model="%%model=openai%%",
    tools=[execute_task],
    output_type=TaskResult,
    model_settings=ModelSettings(temperature=0.5, max_tokens=1000),
)
```

**Agent Parameters:**

- `name` (str): Unique identifier for the agent within the system
- `instructions` (str | Callable): Agent's system prompt or dynamic instruction generator
- `model` (str | Model): Model identifier (e.g. `"%%model=openai%%"`, `"%%model=openai-small%%"`) or a `Model` instance
- `tools` (list): Function tools available to the agent
- `handoffs` (list): Other agents this agent can delegate to
- `input_guardrails` (list): Validation functions for inputs
- `output_guardrails` (list): Validation functions for outputs
- `output_type` (type): Pydantic model (or other type) for structured output
- `model_settings` (ModelSettings): Sampling and model configuration — `temperature`, `max_tokens`, `reasoning`, `tool_choice`, etc.
- `handoff_description` (str): Description used when another agent can hand off to this one
- `tool_use_behavior`: Whether tool results go back to the model or end the run (`"run_llm_again"`, `"stop_on_first_tool"`, `StopAtTools(...)`)
- `mcp_servers` (list): MCP servers whose tools the agent can use

Agents are generic over the context type: `Agent[MyContext](...)`.

### Runner

The orchestration engine that executes agents, managing the agent loop and handling tool calls.

**Basic Execution:**

```python
from agents import Agent, Runner
import asyncio

agent = Agent(
    name="Assistant",
    instructions="Answer questions concisely"
)

# Synchronous execution
result = Runner.run_sync(agent, "What is Python?")
print(result.final_output)

# Asynchronous execution
async def main():
    result = await Runner.run(agent, "What is Python?")
    print(result.final_output)

asyncio.run(main())
```

**Runner Parameters:**

- `agent`: Agent instance to execute
- `input`: User input (string or list of message items)
- `session`: Optional session for conversation memory
- `context`: Optional context object passed to agents and tools
- `max_turns`: Maximum agent loop iterations (default: 10; `None` disables the limit)
- `run_config`: `RunConfig` for run-wide settings — model override, model settings, tracing, guardrails, handoff history
- `hooks`: `RunHooks` lifecycle callbacks
- `error_handlers`: Run error handlers, e.g. `{"max_turns": ..., "model_refusal": ...}`
- `previous_response_id` / `conversation_id`: Server-side conversation state for the Responses API

**Runner Result:**

```python
from agents import Agent, Runner

agent = Agent(name="Assistant", instructions="Answer concisely")

async def main():
    result = await Runner.run(agent, "message")

    # Access results
    print(result.final_output)               # Final output (str or your output_type)
    print(result.final_output_as(str))       # Typed cast of the final output
    print(result.new_items)                  # Items generated during this run
    print(result.to_input_list())            # Input for the next turn (manual history)
    print(result.last_agent)                 # Agent that produced the final output
    print(result.raw_responses)              # Raw model responses
    print(result.context_wrapper.usage)      # Token usage for the run
    print(result.last_response_id)           # For previous_response_id chaining

asyncio.run(main())
```

### Handoff

A mechanism for transferring control between agents, essential for multi-agent systems.

**Simple Handoff:**

```python
from agents import Agent, Runner
import asyncio

support_agent = Agent(
    name="Support",
    handoff_description="Handles customer support requests",
    instructions="Help customers with their issues"
)

sales_agent = Agent(
    name="Sales",
    handoff_description="Handles sales inquiries",
    instructions="Discuss product offerings and pricing"
)

triage_agent = Agent(
    name="Triage",
    instructions="Route customer requests to appropriate agent",
    handoffs=[support_agent, sales_agent]
)

async def main():
    result = await Runner.run(triage_agent, "I want to buy your product")
    print(result.final_output)

asyncio.run(main())
```

**Handoff with Structured Input:**

```python
from agents import Agent, Runner, handoff, RunContextWrapper
from pydantic import BaseModel
import asyncio

class EscalationData(BaseModel):
    reason: str
    priority: str

async def on_escalation(
    ctx: RunContextWrapper[None],
    data: EscalationData
):
    print(f"Escalated: {data.reason} (Priority: {data.priority})")

escalation_agent = Agent(
    name="Manager",
    instructions="Handle escalated issues with priority"
)

support_agent = Agent(
    name="Support",
    instructions="Help customers",
    handoffs=[
        handoff(
            agent=escalation_agent,
            on_handoff=on_escalation,
            input_type=EscalationData,
            tool_description_override="Escalate urgent issue to manager"
        )
    ]
)

async def main():
    result = await Runner.run(
        support_agent,
        "This issue is critical and affecting production!"
    )
    print(result.final_output)

asyncio.run(main())
```

### Guardrail

Safety mechanisms validating inputs and outputs to ensure compliance and safety.

**Input Guardrail:**

```python
from agents import (
    Agent, Runner, input_guardrail, GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered, RunContextWrapper, TResponseInputItem
)
import asyncio

@input_guardrail
async def safety_check(
    ctx: RunContextWrapper[None],
    agent: Agent,
    input_data: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    # Simple safety check
    unsafe_words = ["harm", "illegal", "abuse"]
    input_text = input_data if isinstance(input_data, str) else str(input_data)
    
    is_unsafe = any(word in input_text.lower() for word in unsafe_words)
    
    return GuardrailFunctionOutput(
        output_info={"safety": "pass" if not is_unsafe else "fail"},
        tripwire_triggered=is_unsafe
    )

agent = Agent(
    name="Assistant",
    instructions="Help users safely",
    input_guardrails=[safety_check]
)

async def main():
    try:
        result = await Runner.run(agent, "How can I hurt someone?")
    except InputGuardrailTripwireTriggered:
        print("Unsafe input blocked")

asyncio.run(main())
```

**Output Guardrail:**

```python
from agents import (
    Agent, Runner, output_guardrail, GuardrailFunctionOutput,
    OutputGuardrailTripwireTriggered, RunContextWrapper
)
from pydantic import BaseModel
import asyncio

class ResponseMessage(BaseModel):
    content: str

@output_guardrail
async def content_filter(
    ctx: RunContextWrapper[None],
    agent: Agent,
    output: ResponseMessage
) -> GuardrailFunctionOutput:
    # Check for inappropriate content
    inappropriate_words = ["slur1", "slur2", "offensive"]
    
    has_inappropriate = any(
        word in output.content.lower()
        for word in inappropriate_words
    )
    
    return GuardrailFunctionOutput(
        output_info={"filtered": has_inappropriate},
        tripwire_triggered=has_inappropriate
    )

agent = Agent(
    name="Assistant",
    instructions="Respond helpfully",
    output_guardrails=[content_filter],
    output_type=ResponseMessage
)

async def main():
    try:
        result = await Runner.run(agent, "Tell me a joke")
        print(result.final_output.content)
    except OutputGuardrailTripwireTriggered:
        print("Output blocked by content filter")

asyncio.run(main())
```

### Session

Automatic conversation history management across multiple agent runs.

**SQLite Session:**

```python
from agents import Agent, Runner, SQLiteSession
import asyncio

async def main():
    agent = Agent(
        name="Assistant",
        instructions="Remember conversation context"
    )

    # Create session with unique ID
    session = SQLiteSession("user_123", "conversations.db")

    # First turn
    result = await Runner.run(
        agent,
        "My name is Alice",
        session=session
    )
    print(result.final_output)

    # Second turn - agent remembers context
    result = await Runner.run(
        agent,
        "What's my name?",
        session=session
    )
    print(result.final_output)  # "Alice"

asyncio.run(main())
```

**In-Memory Session:**

`SQLiteSession` defaults to an in-memory database when no path is given, which is handy for tests:

```python
from agents import Agent, Runner, SQLiteSession
import asyncio

async def main():
    session = SQLiteSession("test_session")  # db_path defaults to ":memory:"
    agent = Agent(name="Assistant")
    
    # Temporary session for testing
    result = await Runner.run(agent, "Hello", session=session)
    print(result.final_output)

asyncio.run(main())
```

---

## Simple Agents

### Creating Your First Agent

The simplest agent requires only a name and instructions:

```python
from agents import Agent, Runner
import asyncio

async def main():
    agent = Agent(
        name="Math Helper",
        instructions="Provide clear explanations of mathematical concepts"
    )

    result = await Runner.run(agent, "Explain derivatives")
    print(result.final_output)

asyncio.run(main())
```

### Synchronous and Asynchronous Execution

**Synchronous (Blocking):**

```python
from agents import Agent, Runner

agent = Agent(name="Assistant")
result = Runner.run_sync(agent, "What is AI?")
print(result.final_output)
```

**Asynchronous (Non-Blocking):**

```python
from agents import Agent, Runner
import asyncio

async def main():
    agent = Agent(name="Assistant")
    result = await Runner.run(agent, "What is AI?")
    print(result.final_output)

asyncio.run(main())
```

**Handling Multiple Concurrent Requests:**

```python
import asyncio
from agents import Agent, Runner

async def handle_request(user_id: str, query: str):
    agent = Agent(name="Assistant")
    result = await Runner.run(agent, query)
    return {"user": user_id, "response": result.final_output}

async def main():
    # Process multiple requests concurrently
    requests = [
        handle_request("user1", "What is Python?"),
        handle_request("user2", "Explain recursion"),
        handle_request("user3", "What is machine learning?")
    ]
    
    results = await asyncio.gather(*requests)
    for result in results:
        print(f"{result['user']}: {result['response']}")

asyncio.run(main())
```

### Dynamic System Prompts

Generate instructions dynamically based on context:

```python
from agents import Agent, Runner, RunContextWrapper
from dataclasses import dataclass
import asyncio

@dataclass
class UserContext:
    username: str
    expertise_level: str
    language: str

def create_instructions(
    ctx: RunContextWrapper[UserContext],
    agent: Agent[UserContext]
) -> str:
    user = ctx.context
    
    if user.expertise_level == "beginner":
        complexity = "simple and beginner-friendly"
    elif user.expertise_level == "advanced":
        complexity = "detailed and technical"
    else:
        complexity = "intermediate"
    
    return f"""You are helping {user.username}.
    Explain concepts in a {complexity} manner.
    Respond in {user.language}."""

agent = Agent[UserContext](
    name="Tutor",
    instructions=create_instructions
)

async def main():
    beginner_context = UserContext(
        username="Alice",
        expertise_level="beginner",
        language="English"
    )
    
    advanced_context = UserContext(
        username="Bob",
        expertise_level="advanced",
        language="Spanish"
    )
    
    result1 = await Runner.run(
        agent,
        "Explain machine learning",
        context=beginner_context
    )
    print("Beginner:", result1.final_output[:100])
    
    result2 = await Runner.run(
        agent,
        "Explain neural networks",
        context=advanced_context
    )
    print("Advanced:", result2.final_output[:100])

asyncio.run(main())
```

### Single-Turn Conversations

Agents process queries without maintaining state between turns:

```python
from agents import Agent, Runner
import asyncio

async def main():
    agent = Agent(
        name="Calculator",
        instructions="Perform mathematical calculations"
    )

    queries = [
        "What is 15 * 23?",
        "Square root of 144",
        "10 factorial"
    ]
    
    for query in queries:
        result = await Runner.run(agent, query)
        print(f"Q: {query}\nA: {result.final_output}\n")

asyncio.run(main())
```

### Streaming Outputs

Stream responses token-by-token or at item level:

**Token-Level Streaming:**

```python
from agents import Agent, Runner
from openai.types.responses import ResponseTextDeltaEvent
import asyncio

async def main():
    agent = Agent(
        name="Writer",
        instructions="Write creative content"
    )

    result = Runner.run_streamed(
        agent,
        "Write a short poem about autumn"
    )

    print("Streaming response:\n", end="", flush=True)
    
    async for event in result.stream_events():
        if event.type == "raw_response_event":
            if isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)

    print(f"\n\nFinal output: {result.final_output}")

asyncio.run(main())
```

**Item-Level Streaming:**

```python
from agents import Agent, Runner, function_tool, ItemHelpers
import asyncio
import random

@function_tool
def get_random_number(max_value: int = 100) -> int:
    """Get a random number up to max_value."""
    return random.randint(1, max_value)

async def main():
    agent = Agent(
        name="NumberGenerator",
        instructions="Generate random numbers when asked",
        tools=[get_random_number]
    )

    result = Runner.run_streamed(agent, "Generate 3 random numbers")

    print("Agent stream events:")
    async for event in result.stream_events():
        if event.type == "raw_response_event":
            continue
        elif event.type == "run_item_stream_event":
            item = event.item
            if item.type == "tool_call_item":
                print(f"Tool called: {item.tool_name}")
            elif item.type == "tool_call_output_item":
                print(f"Result: {item.output}")
            elif item.type == "message_output_item":
                text = ItemHelpers.text_message_output(item)
                print(f"Response: {text[:100]}")

asyncio.run(main())
```

### Error Handling and Response Parsing

Handle exceptions and parse responses robustly:

```python
from agents import (
    Agent, Runner, function_tool,
    MaxTurnsExceeded, InputGuardrailTripwireTriggered, ModelBehaviorError
)
from pydantic import BaseModel
import asyncio

class ParsedResponse(BaseModel):
    success: bool
    data: str
    error: str | None = None

@function_tool
def unstable_operation() -> str:
    """An operation that might fail."""
    import random
    if random.random() < 0.5:
        raise ValueError("Operation failed randomly")
    return "Success!"

async def main():
    agent = Agent(
        name="TaskRunner",
        instructions="Execute operations and report results",
        tools=[unstable_operation],
        output_type=ParsedResponse
    )

    try:
        result = await Runner.run(
            agent,
            "Try to run the unstable operation",
            max_turns=3
        )
        
        response = result.final_output_as(ParsedResponse)
        if response.success:
            print(f"Success: {response.data}")
        else:
            print(f"Error: {response.error}")

    except MaxTurnsExceeded:
        print("Agent exceeded maximum turns")
    except InputGuardrailTripwireTriggered as e:
        print(f"Input blocked: {e}")
    except ModelBehaviorError as e:
        print(f"Model error: {e}")
    except Exception as e:
        print(f"Unexpected error: {type(e).__name__}: {e}")

asyncio.run(main())
```

---

## Multi-Agent Systems

### Basic Multi-Agent Handoff

Multiple specialised agents with automatic delegation:

```python
from agents import Agent, Runner
import asyncio

billing_agent = Agent(
    name="Billing Specialist",
    handoff_description="Handles billing, invoices, and payment issues",
    instructions="""You are a billing specialist. Handle all billing-related questions.
    You can process refunds, check invoice status, and update payment methods."""
)

technical_agent = Agent(
    name="Technical Support",
    handoff_description="Handles technical issues and troubleshooting",
    instructions="""You are a technical support specialist.
    Help users troubleshoot problems and resolve technical issues."""
)

triage_agent = Agent(
    name="Customer Service Triage",
    instructions="""Determine the type of customer request and handoff to appropriate specialist.
    - Billing issues -> Billing Specialist
    - Technical problems -> Technical Support
    Be professional and helpful.""",
    handoffs=[billing_agent, technical_agent]
)

async def main():
    test_queries = [
        "I was charged twice for my subscription",
        "The app keeps crashing when I upload files",
        "Can I get an invoice for my purchase?"
    ]
    
    for query in test_queries:
        print(f"\nCustomer: {query}")
        result = await Runner.run(triage_agent, query)
        print(f"Agent: {result.last_agent.name}")
        print(f"Response: {result.final_output[:200]}")

asyncio.run(main())
```

### Filtering What a Specialist Sees

By default the receiving agent sees the whole conversation, including earlier tool calls. `handoff(..., input_filter=...)` rewrites that history before the handoff; `agents.extensions.handoff_filters` has ready-made filters such as `remove_all_tools`:

```python
from agents import Agent, Runner, handoff
from agents.extensions import handoff_filters
import asyncio

manager_agent = Agent(
    name="Manager",
    instructions="Review and handle escalated support tickets with priority",
)

support_agent = Agent(
    name="Support",
    instructions="Handle customer support requests. Escalate account security issues.",
    handoffs=[
        handoff(
            agent=manager_agent,
            input_filter=handoff_filters.remove_all_tools,  # drop earlier tool calls/results
            tool_description_override="Escalate to manager for urgent issues",
        )
    ],
)

async def main():
    result = await Runner.run(
        support_agent,
        "I need immediate help - my account has been hacked!"
    )
    print(result.last_agent.name, result.final_output)

asyncio.run(main())
```

A filter is any function that takes and returns a `HandoffInputData` (`input_history`, `pre_handoff_items`, `new_items`), so you can also write your own, for example to redact personal data. `RunConfig(handoff_input_filter=...)` applies one filter to every handoff in a run.

### Routing Logic and Conditional Handoffs

Implement sophisticated routing based on request characteristics:

```python
from agents import Agent, Runner, function_tool
import asyncio

# Specialised agents
product_agent = Agent(
    name="Product Expert",
    handoff_description="Product features and capabilities",
    instructions="Explain product features and capabilities"
)

pricing_agent = Agent(
    name="Pricing Specialist",
    handoff_description="Pricing and subscription options",
    instructions="Discuss pricing, plans, and subscriptions"
)

demo_agent = Agent(
    name="Demo Coordinator",
    handoff_description="Demo scheduling",
    instructions="Schedule and coordinate product demonstrations"
)

@function_tool
def classify_inquiry(user_query: str) -> str:
    """Classify inquiry to route to correct specialist."""
    query_lower = user_query.lower()
    
    if any(word in query_lower for word in ["price", "cost", "plan", "subscription"]):
        return "pricing"
    elif any(word in query_lower for word in ["demo", "trial", "test", "see"]):
        return "demo"
    else:
        return "product"

router_agent = Agent(
    name="Sales Router",
    instructions="""Use the classify_inquiry tool to understand customer needs.
    Route to appropriate specialist based on inquiry type.
    - 'pricing' -> Pricing Specialist
    - 'demo' -> Demo Coordinator
    - 'product' -> Product Expert
    """,
    tools=[classify_inquiry],
    handoffs=[product_agent, pricing_agent, demo_agent]
)

async def main():
    inquiries = [
        "What are the features of your product?",
        "How much does it cost?",
        "Can I schedule a demo?",
        "What's included in the Pro plan?"
    ]
    
    for inquiry in inquiries:
        print(f"\n→ {inquiry}")
        result = await Runner.run(router_agent, inquiry)
        print(f"← {result.final_output[:150]}")

asyncio.run(main())
```

### Parallel Agent Execution

Execute multiple agents concurrently:

```python
from agents import Agent, Runner
import asyncio

analyst_agents = [
    Agent(
        name="Market Analyst",
        instructions="Analyse market trends and opportunities"
    ),
    Agent(
        name="Competitive Analyst",
        instructions="Analyse competitive landscape"
    ),
    Agent(
        name="Financial Analyst",
        instructions="Analyse financial projections"
    )
]

async def analyse_topic(agent: Agent, topic: str) -> str:
    result = await Runner.run(agent, f"Analyse {topic}")
    return result.final_output

async def main():
    topic = "AI Industry Growth"
    
    # Run analysts in parallel
    analyses = await asyncio.gather(
        *[analyse_topic(agent, topic) for agent in analyst_agents]
    )
    
    print("Comprehensive Analysis:")
    for agent, analysis in zip(analyst_agents, analyses):
        print(f"\n{agent.name}:\n{analysis[:200]}")

asyncio.run(main())
```

### Agent as Tools Pattern

Use specialised agents as tools within a coordinator agent:

```python
from agents import Agent, Runner
import asyncio

translator = Agent(
    name="Translator",
    instructions="Translate text to requested language"
)

summariser = Agent(
    name="Summariser",
    instructions="Summarise text concisely"
)

editor = Agent(
    name="Editor",
    instructions="Edit and improve text quality"
)

coordinator = Agent(
    name="Content Coordinator",
    instructions="Use available tools to process documents",
    tools=[
        translator.as_tool(
            tool_name="translate",
            tool_description="Translate content to another language"
        ),
        summariser.as_tool(
            tool_name="summarise",
            tool_description="Summarise content"
        ),
        editor.as_tool(
            tool_name="edit",
            tool_description="Edit and improve text"
        )
    ]
)

async def main():
    result = await Runner.run(
        coordinator,
        """Process this text:
        1. Summarise it
        2. Improve the writing
        3. Translate to Spanish"""
    )
    print(result.final_output)

asyncio.run(main())
```

---

## Tools Integration

### Function Tools with Automatic Schema Generation

Decorate Python functions to create agent tools:

```python
from agents import Agent, Runner, function_tool
import asyncio
from datetime import datetime

@function_tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().isoformat()

@function_tool
def calculate_compound_interest(
    principal: float,
    rate: float,
    time_years: int,
    compounds_per_year: int = 1
) -> float:
    """Calculate compound interest.
    
    Args:
        principal: Initial investment amount
        rate: Annual interest rate (as decimal, e.g., 0.05 for 5%)
        time_years: Number of years
        compounds_per_year: Compounding frequency (default: 1 for annual)
    """
    amount = principal * (1 + rate / compounds_per_year) ** (compounds_per_year * time_years)
    return round(amount, 2)

@function_tool
async def fetch_weather(city: str) -> dict:
    """Fetch weather data for a city."""
    # In production, call actual weather API
    return {
        "city": city,
        "temperature": 72,
        "condition": "Sunny",
        "humidity": 45
    }

agent = Agent(
    name="Financial Assistant",
    instructions="Help with financial calculations and information",
    tools=[get_current_time, calculate_compound_interest, fetch_weather]
)

async def main():
    result = await Runner.run(
        agent,
        "Calculate compound interest on £10,000 at 5% for 10 years, compounded annually"
    )
    print(result.final_output)

asyncio.run(main())
```

### Pydantic-Powered Validation

Use Pydantic models for complex tool parameters:

```python
from agents import Agent, Runner, function_tool
from pydantic import BaseModel, Field
from typing import Literal
import asyncio

class BookingRequest(BaseModel):
    customer_name: str = Field(..., description="Customer's full name")
    email: str = Field(..., description="Customer's email address")
    hotel_name: str = Field(..., description="Hotel to book")
    check_in_date: str = Field(..., description="Check-in date (YYYY-MM-DD)")
    check_out_date: str = Field(..., description="Check-out date (YYYY-MM-DD)")
    room_type: Literal["single", "double", "suite"] = Field(default="double")
    guests: int = Field(ge=1, le=6, description="Number of guests")

class BookingResult(BaseModel):
    success: bool
    confirmation_number: str | None
    message: str

@function_tool
def book_hotel(request: BookingRequest) -> BookingResult:
    """Book a hotel room."""
    # Validation happens automatically via Pydantic
    return BookingResult(
        success=True,
        confirmation_number=f"CONF-{hash(request.customer_name) % 1000000:06d}",
        message=f"Booking confirmed for {request.customer_name}"
    )

@function_tool
def get_available_hotels(
    location: str,
    check_in: str,
    check_out: str,
    guest_count: int
) -> list[dict]:
    """Search available hotels."""
    return [
        {"name": "Luxury Hotel", "rating": 5, "price_per_night": 250},
        {"name": "Budget Inn", "rating": 3, "price_per_night": 80},
        {"name": "Business Hotel", "rating": 4, "price_per_night": 150}
    ]

agent = Agent(
    name="Travel Booking Assistant",
    instructions="Help customers book hotels and travel arrangements",
    tools=[get_available_hotels, book_hotel]
)

async def main():
    result = await Runner.run(
        agent,
        "I need to book a hotel in London for 2 people from 2024-12-20 to 2024-12-25"
    )
    print(result.final_output)

asyncio.run(main())
```

### OAI Hosted Tools

Use OpenAI's built-in tools without implementing them yourself:

**Web Search Tool:**

```python
from agents import Agent, Runner, WebSearchTool
import asyncio

agent = Agent(
    name="Research Assistant",
    instructions="Research topics using web search",
    tools=[WebSearchTool()]
)

async def main():
    result = await Runner.run(
        agent,
        "What are the latest developments in quantum computing?"
    )
    print(result.final_output)

asyncio.run(main())
```

**File Search Tool:**

```python
from agents import Agent, Runner, FileSearchTool
import asyncio

agent = Agent(
    name="Document Analyst",
    instructions="Answer questions about documents using file search",
    tools=[
        FileSearchTool(
            max_num_results=5,
            vector_store_ids=["vs_abc123", "vs_def456"]
        )
    ]
)

async def main():
    result = await Runner.run(
        agent,
        "What are the key findings in the latest research?"
    )
    print(result.final_output)

asyncio.run(main())
```

**Code Interpreter Tool:**

`CodeInterpreterTool` is a hosted OpenAI tool imported directly from `agents`; it needs no extra install, but it runs only with OpenAI Responses models. `tool_config` is required: it names the tool type and the container to run in (`{"type": "auto"}` lets OpenAI create one).

```python
from agents import Agent, Runner, CodeInterpreterTool
import asyncio

agent = Agent(
    name="Data Analyst",
    instructions="Analyse data and generate visualisations",
    tools=[
        CodeInterpreterTool(
            tool_config={"type": "code_interpreter", "container": {"type": "auto"}}
        )
    ]
)

async def main():
    result = await Runner.run(
        agent,
        "Analyse this data and create a chart: [1,2,3,4,5]"
    )
    print(result.final_output)

asyncio.run(main())
```

**Computer Use Tool:**

`ComputerTool` lets the model drive a screen through an environment you supply. You implement `AsyncComputer` (or the synchronous `Computer`): the `environment` and `dimensions` properties plus `screenshot`, `click`, `double_click`, `scroll`, `type`, `wait`, `move`, `keypress` and `drag`. The SDK turns the model's actions into calls on that object. The outline below shows the shape with Playwright; the SDK repository's `examples/tools/computer_use.py` has a complete version.

```python
# pip install playwright && playwright install chromium
import base64
from agents import Agent, AsyncComputer, Button, ComputerTool
from playwright.async_api import Page

class PlaywrightComputer(AsyncComputer):
    def __init__(self, page: Page):
        self.page = page

    @property
    def environment(self):
        return "browser"

    @property
    def dimensions(self):
        return (1024, 768)

    async def screenshot(self) -> str:
        return base64.b64encode(await self.page.screenshot()).decode()

    async def click(self, x: int, y: int, button: Button) -> None:
        await self.page.mouse.click(x, y, button="right" if button == "right" else "left")

    async def double_click(self, x: int, y: int) -> None:
        await self.page.mouse.dblclick(x, y)

    async def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> None:
        await self.page.mouse.move(x, y)
        await self.page.mouse.wheel(scroll_x, scroll_y)

    async def type(self, text: str) -> None:
        await self.page.keyboard.type(text)

    async def wait(self) -> None:
        await self.page.wait_for_timeout(1000)

    async def move(self, x: int, y: int) -> None:
        await self.page.mouse.move(x, y)

    async def keypress(self, keys: list[str]) -> None:
        for key in keys:
            await self.page.keyboard.press(key)

    async def drag(self, path: list[tuple[int, int]]) -> None:
        await self.page.mouse.move(*path[0])
        await self.page.mouse.down()
        for point in path[1:]:
            await self.page.mouse.move(*point)
        await self.page.mouse.up()

def build_agent(page: Page) -> Agent:
    return Agent(
        name="Browser agent",
        instructions="Complete the task in the browser.",
        tools=[ComputerTool(computer=PlaywrightComputer(page))],
    )
```

Computer use needs a model that supports the computer tool; check the OpenAI model list before choosing one.

Two more tools act on a machine you control: `ShellTool(executor=...)` runs the model's shell commands through an executor you provide (or `environment=` selects a hosted container), and `ApplyPatchTool(editor=...)` applies model-written diffs through an `ApplyPatchEditor` you implement. Both take `needs_approval`, which pauses the run for a person in the same way as [function-tool approvals](/openai-agents-sdk-guides/openai_agents_sdk_middleware_python/#human-in-the-loop-approvals).

**Image Generation Tool:**

```python
from agents import Agent, Runner, ImageGenerationTool
import asyncio

agent = Agent(
    name="Creative Designer",
    instructions="Generate images based on descriptions",
    tools=[ImageGenerationTool(tool_config={"type": "image_generation", "quality": "low"})]
)

async def main():
    result = await Runner.run(
        agent,
        "Generate an image of a sunset over mountains"
    )
    print(result.final_output)

asyncio.run(main())
```

### Error Handling in Tools

Handle tool errors gracefully:

```python
from agents import Agent, Runner, function_tool
import asyncio
from typing import Any

@function_tool
def safe_divide(a: float, b: float) -> float:
    """Divide two numbers safely."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

@function_tool
def database_query(query: str) -> list[dict]:
    """Execute database query with error handling."""
    try:
        # Simulate database operation
        if "DROP" in query.upper():
            raise PermissionError("Dangerous query blocked")
        return [{"id": 1, "data": "result"}]
    except Exception as e:
        raise ValueError(f"Query failed: {str(e)}")

agent = Agent(
    name="Calculator",
    instructions="Perform calculations and queries",
    tools=[safe_divide, database_query]
)

async def main():
    try:
        result = await Runner.run(agent, "Divide 10 by 0")
    except Exception as e:
        print(f"Handled error: {e}")

asyncio.run(main())
```

### Async Tool Execution

Non-blocking tool execution:

```python
from agents import Agent, Runner, function_tool
import asyncio
import time

@function_tool
async def fetch_data_from_api(endpoint: str) -> dict:
    """Fetch data from API asynchronously."""
    await asyncio.sleep(2)  # Simulate API call
    return {"endpoint": endpoint, "data": "result"}

@function_tool
async def process_image(image_url: str) -> str:
    """Process image asynchronously."""
    await asyncio.sleep(1)  # Simulate processing
    return f"Processed {image_url}"

@function_tool
async def run_ml_model(input_data: list) -> dict:
    """Run ML model asynchronously."""
    await asyncio.sleep(3)  # Simulate model inference
    return {"prediction": 0.95, "confidence": 0.89}

agent = Agent(
    name="Async Worker",
    instructions="Process multiple tasks efficiently",
    tools=[fetch_data_from_api, process_image, run_ml_model]
)

async def main():
    start = time.time()
    result = await Runner.run(
        agent,
        "Fetch data, process an image, and run model predictions"
    )
    elapsed = time.time() - start
    print(f"Completed in {elapsed:.2f} seconds")
    print(result.final_output)

asyncio.run(main())
```

### Programmatic Tool Calling

Since 0.19, supported Responses models can write a short JavaScript program that calls your tools in loops or in parallel and returns one result, saving a model round trip per call. The program runs in a hosted V8 sandbox with no filesystem or network access, and can call only tools that opt in with `allowed_callers=["programmatic"]`:

```python
from pydantic import BaseModel
from agents import Agent, ModelSettings, ProgrammaticToolCallingTool, Runner, function_tool

class InventoryOutput(BaseModel):
    sku: str
    available_units: int

@function_tool(allowed_callers=["programmatic"])
def get_inventory(sku: str) -> InventoryOutput:
    """Look up stock for one SKU."""
    return InventoryOutput(sku=sku, available_units=42)

agent = Agent(
    name="Inventory planner",
    model="%%model=openai%%",
    model_settings=ModelSettings(tool_choice="programmatic_tool_calling"),
    tools=[get_inventory, ProgrammaticToolCallingTool()],
)

result = Runner.run_sync(agent, "Check inventory for desk-lamp and summarise it.")
print(result.final_output)
```

`@tool` (from `agents.decorators`, also 0.19) is an alias for `@function_tool`.

---

## Structured Outputs

### Pydantic Models for Responses

Define expected output structure:

```python
from agents import Agent, Runner
from pydantic import BaseModel, Field
import asyncio

class NewsArticle(BaseModel):
    title: str = Field(..., description="Article headline")
    summary: str = Field(..., description="3-4 sentence summary")
    sentiment: str = Field(..., description="Positive, Negative, or Neutral")
    key_topics: list[str] = Field(..., description="Main topics covered")
    source_reliability: int = Field(ge=1, le=10, description="Reliability score 1-10")

agent = Agent(
    name="News Analyst",
    instructions="Analyse news articles and extract key information",
    output_type=NewsArticle
)

async def main():
    article_text = """
    Apple announces revolutionary new AI chip. The tech giant revealed its latest 
    artificial intelligence processor, featuring 40% better performance than competitors.
    Industry analysts predict this will reshape the smartphone market.
    """
    
    result = await Runner.run(agent, f"Analyse this article: {article_text}")
    
    # Access structured output
    article = result.final_output_as(NewsArticle)
    print(f"Title: {article.title}")
    print(f"Sentiment: {article.sentiment}")
    print(f"Topics: {', '.join(article.key_topics)}")

asyncio.run(main())
```

### Complex Nested Structures

Model hierarchical data:

```python
from agents import Agent, Runner
from pydantic import BaseModel, Field
from typing import Optional
import asyncio

class Address(BaseModel):
    street: str
    city: str
    postal_code: str
    country: str

class Contact(BaseModel):
    email: str
    phone: Optional[str] = None
    address: Address

class Company(BaseModel):
    name: str
    industry: str
    founded_year: int
    employees: int
    contact: Contact
    website: Optional[str] = None

agent = Agent(
    name="Company Researcher",
    instructions="Extract company information from text",
    output_type=Company
)

async def main():
    text = """
    Acme Corporation is a software company founded in 2010 with 500 employees.
    They work in artificial intelligence. Contact them at contact@acme.com,
    phone 555-0123, at 123 Tech Street, San Francisco, CA 94105, USA.
    Website: www.acme.com
    """
    
    result = await Runner.run(agent, f"Extract company info: {text}")
    company = result.final_output_as(Company)
    
    print(f"Company: {company.name}")
    print(f"Location: {company.contact.address.city}, {company.contact.address.country}")
    print(f"Contact: {company.contact.email}")

asyncio.run(main())
```

### JSON Mode Configuration

Ensure reliable JSON output:

```python
from agents import Agent, Runner, ModelSettings
from pydantic import BaseModel
import asyncio
import json

class TaskBreakdown(BaseModel):
    main_task: str
    subtasks: list[str]
    estimated_hours: float
    priority: str

agent = Agent(
    name="Project Planner",
    instructions="Break down projects into tasks",
    output_type=TaskBreakdown,  # the SDK sends a strict JSON schema for you
    model_settings=ModelSettings(
        temperature=0,  # Ensure consistency
    )
)

async def main():
    result = await Runner.run(
        agent,
        "I need to build a web application"
    )
    
    tasks = result.final_output_as(TaskBreakdown)
    print(json.dumps(tasks.model_dump(), indent=2))

asyncio.run(main())
```

### Non-Strict Output Types

By default the SDK sends a *strict* JSON schema, which the API enforces exactly. Strict schemas cannot contain open-ended types such as `Any` or a bare `dict`; using one raises `UserError: Strict JSON schema is enabled…`. Either model the data explicitly (a `list[KeyValue]` instead of `dict`), or wrap the type in `AgentOutputSchema(..., strict_json_schema=False)`. Non-strict output is validated by Pydantic after the response arrives, so the model can produce invalid JSON and the run then fails with `ModelBehaviorError`.

```python
from agents import Agent, AgentOutputSchema, Runner
from pydantic import BaseModel
from typing import Any
import asyncio

class FlexibleResponse(BaseModel):
    status: str
    data: Any  # Flexible data structure
    metadata: dict

agent = Agent(
    name="Flexible Agent",
    instructions="Respond with flexible structure",
    output_type=AgentOutputSchema(FlexibleResponse, strict_json_schema=False),
)

async def main():
    result = await Runner.run(agent, "Generate some data")
    response = result.final_output_as(FlexibleResponse)
    print(f"Status: {response.status}")
    print(f"Data: {response.data}")

asyncio.run(main())
```

---

## Model Context Protocol

### Building Agents with MCP

Integrate with Model Context Protocol servers:

```python
from agents import Agent, Runner
from agents.mcp import MCPServerStreamableHttp
import asyncio
import os

async def main():
    # Connect to hosted MCP server
    async with MCPServerStreamableHttp(
        name="GitHub MCP",
        params={
            "url": "https://api.githubcopilot.com/mcp/",
            "headers": {"Authorization": f"Bearer {os.getenv('GITHUB_TOKEN')}"},
            "timeout": 30
        }
    ) as server:
        agent = Agent(
            name="Repository Browser",
            instructions="Help explore GitHub repositories",
            mcp_servers=[server]
        )
        
        result = await Runner.run(
            agent,
            "What are the top Python repositories?"
        )
        print(result.final_output)

asyncio.run(main())
```

### Filesystem MCP Examples

Access local filesystem via MCP:

```python
from agents import Agent, Runner
from agents.mcp import MCPServerStdio
from pathlib import Path
import asyncio

async def main():
    project_dir = Path("./my_project")
    
    async with MCPServerStdio(
        name="Filesystem",
        params={
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                str(project_dir)
            ]
        }
    ) as server:
        agent = Agent(
            name="File Assistant",
            instructions="Help with file operations",
            mcp_servers=[server]
        )
        
        result = await Runner.run(
            agent,
            "List all Python files in the project"
        )
        print(result.final_output)

asyncio.run(main())
```

### Git Integration via MCP

Work with Git repositories:

```python
from agents import Agent, Runner
from agents.mcp import MCPServerStdio
from pathlib import Path
import asyncio

async def main():
    repo_path = Path("./my_repo")
    
    async with MCPServerStdio(
        name="Git",
        params={
            "command": "uvx",  # the reference Git server is a Python package
            "args": ["mcp-server-git", "--repository", str(repo_path)]
        }
    ) as server:
        agent = Agent(
            name="Code Reviewer",
            instructions="Review recent changes",
            mcp_servers=[server]
        )
        
        result = await Runner.run(
            agent,
            "Show me the latest commits and their changes"
        )
        print(result.final_output)

asyncio.run(main())
```

### Custom MCP Server Creation

`agents.mcp.MCPServer` is the *client* side: the SDK's own `MCPServerStdio`, `MCPServerSse` and `MCPServerStreamableHttp` implement it. To expose your own tools over MCP, write a server with the official `mcp` package (installed with the SDK) and connect to it like any other. The example uses MCP Python SDK v2, where the high-level server class is `MCPServer`; under `mcp` 1.x the same code imports `FastMCP` from `mcp.server.fastmcp`.

```python
# finance_server.py
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("Financial Tools")

@mcp.tool()
def calculate_roi(investment: float, return_value: float) -> float:
    """Return on investment, as a percentage."""
    return (return_value - investment) / investment * 100

if __name__ == "__main__":
    mcp.run()  # stdio transport
```

```python
import asyncio
import sys
from agents import Agent, Runner
from agents.mcp import MCPServerStdio

async def main():
    async with MCPServerStdio(
        name="Financial Tools",
        params={"command": sys.executable, "args": ["finance_server.py"]},
        cache_tools_list=True,
    ) as server:
        agent = Agent(
            name="Financial Advisor",
            instructions="Provide financial analysis using the tools.",
            mcp_servers=[server],
        )
        result = await Runner.run(agent, "Calculate ROI on a £1000 investment that returns £1500")
        print(result.final_output)

asyncio.run(main())
```

If your tools are plain Python in the same process, `@function_tool` is simpler; MCP pays off when the tools are shared between applications or written in another language.

### Hosted MCP Tools

`HostedMCPTool` hands a remote MCP server to the OpenAI Responses API, which lists and calls its tools on its side; your process never connects to the server. It works only with OpenAI Responses models. `require_approval` controls whether each call needs consent: `"never"`, `"always"`, or a per-tool mapping. Approval requests arrive as run interruptions, handled with the same `RunState` approve/reject flow as [function-tool approvals](/openai-agents-sdk-guides/openai_agents_sdk_middleware_python/#human-in-the-loop-approvals), or answered in code with `on_approval_request`:

```python
import asyncio
from agents import Agent, HostedMCPTool, Runner

agent = Agent(
    name="Docs assistant",
    instructions="Answer questions about the repository using the MCP tools.",
    model="%%model=openai%%",
    tools=[
        HostedMCPTool(
            tool_config={
                "type": "mcp",
                "server_label": "gitmcp",
                "server_url": "https://gitmcp.io/openai/codex",
                "require_approval": "never",
            }
        )
    ],
)

async def main():
    result = await Runner.run(agent, "Which language is this repository written in?")
    print(result.final_output)

asyncio.run(main())
```

Choose by where the server runs: hosted MCP for public HTTPS servers when you want OpenAI to make the calls; `MCPServerStreamableHttp` (or the older `MCPServerSse`) when your process should connect, for example to a server on a private network; `MCPServerStdio` for local subprocess servers.

---

## Agentic Patterns

### Deterministic Workflows

Build predictable agent flows:

```python
from agents import Agent, Runner, function_tool
from pydantic import BaseModel
import asyncio

class DataValidation(BaseModel):
    is_valid: bool
    errors: list[str]

class ProcessingResult(BaseModel):
    status: str
    records_processed: int
    errors: list[str]

@function_tool
def validate_csv(file_path: str) -> DataValidation:
    """Validate CSV file format."""
    return DataValidation(
        is_valid=True,
        errors=[]
    )

@function_tool
def transform_data(file_path: str) -> ProcessingResult:
    """Transform and process data."""
    return ProcessingResult(
        status="completed",
        records_processed=1000,
        errors=[]
    )

@function_tool
def export_results(output_format: str) -> str:
    """Export processed results."""
    return f"Results exported to {output_format}"

agent = Agent(
    name="Data Pipeline",
    instructions="""Execute data processing pipeline:
    1. Validate input CSV
    2. Transform the data
    3. Export results to JSON""",
    tools=[validate_csv, transform_data, export_results]
)

async def main():
    result = await Runner.run(
        agent,
        "Process data/input.csv and export to JSON"
    )
    print(result.final_output)

asyncio.run(main())
```

### Conditional Tool Usage

Dynamically select tools based on context:

```python
from agents import Agent, Runner, function_tool
import asyncio

@function_tool
def quick_analysis(data: str) -> str:
    """Quick surface-level analysis."""
    return f"Quick analysis: {len(data)} characters"

@function_tool
def deep_analysis(data: str) -> str:
    """Comprehensive in-depth analysis."""
    return f"Deep analysis completed on {len(data)} characters"

agent = Agent(
    name="Analyst",
    instructions="""Analyse text based on complexity:
    - If text < 1000 chars: use quick_analysis
    - If text >= 1000 chars: use deep_analysis
    Choose appropriate tool based on input size.""",
    tools=[quick_analysis, deep_analysis]
)

async def main():
    small_text = "Hello world"
    large_text = "Lorem ipsum " * 200
    
    for text in [small_text, large_text]:
        result = await Runner.run(agent, f"Analyse this: {text[:50]}...")
        print(result.final_output)

asyncio.run(main())
```

### LLM as Judge Pattern

Use an agent to evaluate another agent's output:

```python
from agents import Agent, Runner
from pydantic import BaseModel
import asyncio

class EvaluationResult(BaseModel):
    score: int  # 1-10
    reasoning: str
    approved: bool

# Main task agent
content_writer = Agent(
    name="Content Writer",
    instructions="Write engaging blog posts"
)

# Judge agent
quality_judge = Agent(
    name="Quality Judge",
    instructions="Evaluate content quality on a scale of 1-10",
    output_type=EvaluationResult
)

async def main():
    # Generate content
    content_result = await Runner.run(
        content_writer,
        "Write a blog post about machine learning"
    )
    
    content = content_result.final_output
    
    # Evaluate content
    evaluation_result = await Runner.run(
        quality_judge,
        f"Rate this content: {content}"
    )
    
    evaluation = evaluation_result.final_output_as(EvaluationResult)
    print(f"Quality Score: {evaluation.score}/10")
    print(f"Approved: {evaluation.approved}")
    print(f"Feedback: {evaluation.reasoning}")

asyncio.run(main())
```

### Routing Agents

Route requests to specialised agents:

```python
from agents import Agent, Runner, function_tool
import asyncio

# Specialised agents
python_expert = Agent(
    name="Python Expert",
    handoff_description="Python programming questions",
    instructions="Provide expert Python advice"
)

javascript_expert = Agent(
    name="JavaScript Expert",
    handoff_description="JavaScript programming questions",
    instructions="Provide expert JavaScript advice"
)

@function_tool
def detect_language(query: str) -> str:
    """Detect programming language from query."""
    if "python" in query.lower() or ".py" in query.lower():
        return "python"
    elif "javascript" in query.lower() or ".js" in query.lower():
        return "javascript"
    return "general"

router = Agent(
    name="Router",
    instructions="""Detect programming language and route appropriately:
    - Python queries -> Python Expert
    - JavaScript queries -> JavaScript Expert""",
    tools=[detect_language],
    handoffs=[python_expert, javascript_expert]
)

async def main():
    queries = [
        "How do I use list comprehensions in Python?",
        "What's the best way to handle async in JavaScript?",
        "General programming tips?"
    ]
    
    for query in queries:
        print(f"\n→ {query}")
        result = await Runner.run(router, query)
        print(f"← {result.final_output[:150]}")

asyncio.run(main())
```

---

## Guardrails

### Input Guardrails for Validation

Validate and sanitise user inputs:

```python
from agents import (
    Agent, Runner, input_guardrail, GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered, RunContextWrapper, TResponseInputItem
)
import asyncio
import re

@input_guardrail
async def sql_injection_check(
    ctx: RunContextWrapper[None],
    agent: Agent,
    input_data: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    """Detect potential SQL injection attempts."""
    input_text = input_data if isinstance(input_data, str) else str(input_data)
    
    # Patterns indicating SQL injection
    sql_patterns = [
        r"DROP\s+TABLE",
        r"DELETE\s+FROM",
        r"INSERT\s+INTO",
        r"UPDATE\s+\w+\s+SET",
        r"UNION\s+SELECT"
    ]
    
    for pattern in sql_patterns:
        if re.search(pattern, input_text, re.IGNORECASE):
            return GuardrailFunctionOutput(
                output_info={"threat": "SQL injection detected"},
                tripwire_triggered=True
            )
    
    return GuardrailFunctionOutput(
        output_info={"threat": "none"},
        tripwire_triggered=False
    )

agent = Agent(
    name="Database Query Assistant",
    instructions="Help with database queries",
    input_guardrails=[sql_injection_check]
)

async def main():
    safe_query = "What records have email starting with admin?"
    unsafe_query = "DROP TABLE users; --"
    
    for query in [safe_query, unsafe_query]:
        try:
            result = await Runner.run(agent, query)
            print(f"Query allowed: {result.final_output[:100]}")
        except InputGuardrailTripwireTriggered:
            print("Query blocked: potential SQL injection")

asyncio.run(main())
```

### Output Guardrails for Checks

Validate and filter agent responses:

```python
from agents import (
    Agent, Runner, output_guardrail, GuardrailFunctionOutput,
    OutputGuardrailTripwireTriggered, RunContextWrapper
)
from pydantic import BaseModel
import asyncio

class Response(BaseModel):
    message: str

@output_guardrail
async def profanity_filter(
    ctx: RunContextWrapper[None],
    agent: Agent,
    output: Response
) -> GuardrailFunctionOutput:
    """Filter profanity from responses."""
    blocked_words = ["badword1", "badword2", "badword3"]
    
    message_lower = output.message.lower()
    
    for word in blocked_words:
        if word in message_lower:
            return GuardrailFunctionOutput(
                output_info={"filtered": True},
                tripwire_triggered=True
            )
    
    return GuardrailFunctionOutput(
        output_info={"filtered": False},
        tripwire_triggered=False
    )

agent = Agent(
    name="Friendly Assistant",
    instructions="Respond helpfully",
    output_guardrails=[profanity_filter],
    output_type=Response
)

async def main():
    try:
        result = await Runner.run(agent, "Tell me a joke")
        print(result.final_output.message)
    except OutputGuardrailTripwireTriggered:
        print("Response blocked by content filter")

asyncio.run(main())
```

### Guardrails and Streaming

Output guardrails run once the agent has produced its **final** output, in streamed runs as well. They cannot stop text that has already been streamed to the user; a tripwire raises `OutputGuardrailTripwireTriggered` from the stream after the fact. If unchecked text must never reach the user, buffer the stream until the run completes, or check partial text yourself as it arrives.

Input guardrails run in parallel with the first model call by default. Pass `run_in_parallel=False` to make the run wait for the guardrail before any model call:

```python
from agents import Agent, GuardrailFunctionOutput, input_guardrail

@input_guardrail(run_in_parallel=False)
async def blocking_check(ctx, agent, input_data) -> GuardrailFunctionOutput:
    text = input_data if isinstance(input_data, str) else str(input_data)
    return GuardrailFunctionOutput(output_info=None, tripwire_triggered="ignore previous" in text.lower())

agent = Agent(name="Assistant", input_guardrails=[blocking_check])
```

Guardrails can also be attached to every agent in a run with `RunConfig(input_guardrails=[...], output_guardrails=[...])`. For checks on individual tool calls, see [tool guardrails](/openai-agents-sdk-guides/openai_agents_sdk_middleware_python/#tool-guardrails).

---

## Memory Systems

### Session Management

Maintain conversation history across turns:

```python
from agents import Agent, Runner, SQLiteSession
import asyncio

async def main():
    agent = Agent(
        name="Assistant",
        instructions="Remember and build upon previous context"
    )

    session = SQLiteSession("user_abc", "conversations.db")

    # Turn 1
    r1 = await Runner.run(agent, "My favourite colour is blue", session=session)
    print(f"Turn 1: {r1.final_output}")

    # Turn 2
    r2 = await Runner.run(agent, "What's my favourite colour?", session=session)
    print(f"Turn 2: {r2.final_output}")

    # Turn 3
    r3 = await Runner.run(agent, "Remember that too", session=session)
    print(f"Turn 3: {r3.final_output}")

asyncio.run(main())
```

### Advanced SQLite Patterns

Sophisticated session management:

```python
from agents import Agent, Runner, SQLiteSession
import asyncio

async def main():
    session = SQLiteSession("user_123", "chats.db")

    # Get session size
    items = await session.get_items()
    print(f"Conversation history: {len(items)} items")

    # Add custom items
    await session.add_items([
        {"role": "system", "content": "Custom system message"},
        {"role": "user", "content": "Hello"}
    ])

    # Remove last item (for corrections)
    await session.pop_item()

    # Get the most recent item (get_items is async, so await it before indexing)
    recent = await session.get_items(limit=1)
    last_item = recent[-1] if recent else None

    # Clear entire session
    await session.clear_session()

asyncio.run(main())
```

### Redis Session Storage

Scalable session management:

```python
from agents import Agent, Runner
from agents.extensions.memory import RedisSession
import asyncio

async def main():
    agent = Agent(name="Assistant")

    session = RedisSession.from_url(
        "user_xyz",
        url="redis://localhost:6379"
    )

    result = await Runner.run(
        agent,
        "Remember this information",
        session=session
    )

    # Redis automatically persists session
    print(result.final_output)

asyncio.run(main())
```

### SQLAlchemy Session Storage

Database-agnostic session backend:

```python
# pip install 'openai-agents[sqlalchemy]' asyncpg
from agents import Agent, Runner
from agents.extensions.memory import SQLAlchemySession
import asyncio

async def main():
    agent = Agent(name="Assistant")

    # The SDK manages its own tables (agent_sessions / agent_messages).
    # Use an async driver URL; create_tables=True is convenient in development.
    session = SQLAlchemySession.from_url(
        "user_456",
        url="postgresql+asyncpg://user:password@localhost/conversations",
        create_tables=True,
    )

    # Or share an existing AsyncEngine:
    # session = SQLAlchemySession("user_456", engine=async_engine, create_tables=True)

    result = await Runner.run(agent, "Hello", session=session)
    print(result.final_output)

asyncio.run(main())
```

### OpenAI Session Storage Backend

Managed session storage via OpenAI:

```python
from agents import Agent, Runner, OpenAIConversationsSession
import asyncio

async def main():
    agent = Agent(name="Assistant")

    # Create new conversation
    session = OpenAIConversationsSession()

    # Or resume existing conversation
    # session = OpenAIConversationsSession(conversation_id="conv_abc123")

    result = await Runner.run(
        agent,
        "Store this in managed backend",
        session=session
    )

    print(f"Conversation ID: {session.conversation_id}")
    print(f"Response: {result.final_output}")

asyncio.run(main())
```

---

## Context Engineering

### Instructions vs. Input Differentiation

Distinguish between system instructions and user input:

```python
from agents import Agent, Runner
import asyncio

# Hardcoded instructions
agent = Agent(
    name="Tutor",
    instructions="You are a patient math tutor. Explain concepts step-by-step."
)

# User provides input
async def main():
    queries = [
        "Explain fractions",
        "What is calculus?",
        "Teach me trigonometry"
    ]
    
    for query in queries:
        result = await Runner.run(agent, query)
        print(f"\nQ: {query}\n{result.final_output[:200]}")

asyncio.run(main())
```

### Prompt Templates

Create reusable prompt structures:

```python
from agents import Agent, Runner, RunContextWrapper
from dataclasses import dataclass
import asyncio

@dataclass
class AnalysisContext:
    document_type: str
    audience: str
    depth: str

def create_analysis_prompt(
    ctx: RunContextWrapper[AnalysisContext],
    agent: Agent[AnalysisContext]
) -> str:
    doc = ctx.context
    
    templates = {
        "technical": "Provide technical depth suitable for {audience}",
        "summary": "Provide executive summary for {audience}",
        "educational": "Explain concepts simply for {audience}"
    }
    
    template = templates.get(doc.depth, templates["summary"])
    
    return f"""You are analysing a {doc.document_type}.
    {template.format(audience=doc.audience)}
    Focus on clarity and relevance to the {doc.audience} audience."""

agent = Agent[AnalysisContext](
    name="Document Analyser",
    instructions=create_analysis_prompt
)

async def main():
    contexts = [
        AnalysisContext("research paper", "non-technical stakeholders", "summary"),
        AnalysisContext("research paper", "data scientists", "technical")
    ]
    
    for ctx in contexts:
        result = await Runner.run(
            agent,
            "Analyse this paper on machine learning",
            context=ctx
        )
        print(f"\n{ctx.depth} for {ctx.audience}:")
        print(result.final_output[:150])

asyncio.run(main())
```

### Dynamic Context Injection

Pass runtime data to agents:

```python
from agents import Agent, Runner, RunContextWrapper, function_tool
from dataclasses import dataclass
from datetime import datetime
import asyncio

@dataclass
class UserContext:
    user_id: str
    subscription_tier: str
    language: str
    preferences: dict

@function_tool
def get_user_data(ctx: RunContextWrapper[UserContext]) -> dict:
    """Get user-specific data from context."""
    user = ctx.context
    return {
        "user_id": user.user_id,
        "tier": user.subscription_tier,
        "language": user.language
    }

agent = Agent[UserContext](
    name="Personalised Assistant",
    instructions="Provide personalised responses based on user context",
    tools=[get_user_data]
)

async def main():
    user_context = UserContext(
        user_id="user_789",
        subscription_tier="premium",
        language="British English",
        preferences={"verbose": False}
    )
    
    result = await Runner.run(
        agent,
        "What features am I entitled to?",
        context=user_context
    )
    print(result.final_output)

asyncio.run(main())
```

### Few-Shot Examples in Instructions

Include examples for better performance:

```python
from agents import Agent, Runner
import asyncio

examples = """
EXAMPLES OF SENTIMENT ANALYSIS:
- "I love this product!" -> Positive
- "This is terrible" -> Negative
- "It's okay, nothing special" -> Neutral
- "Best purchase ever made!" -> Positive
- "Not worth the money" -> Negative
"""

agent = Agent(
    name="Sentiment Analyser",
    instructions=f"""Analyse sentiment of text.
{examples}

Use these categories: Positive, Negative, Neutral, Mixed"""
)

async def main():
    texts = [
        "This movie was absolutely fantastic!",
        "Not bad, but could be better",
        "Worst experience of my life"
    ]
    
    for text in texts:
        result = await Runner.run(agent, f"Analyse: {text}")
        print(f"{text[:40]}: {result.final_output}")

asyncio.run(main())
```

### File Handling (Local and Remote)

Process external files and images:

```python
from agents import Agent, Runner, function_tool
import asyncio
from pathlib import Path

@function_tool
def read_local_file(file_path: str) -> str:
    """Read contents of local file."""
    return Path(file_path).read_text()

@function_tool
async def fetch_remote_file(url: str) -> str:
    """Fetch contents from remote URL."""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.text()

agent = Agent(
    name="File Processor",
    instructions="Process and analyse files",
    tools=[read_local_file, fetch_remote_file]
)

async def main():
    # Process local file
    result = await Runner.run(agent, "Read and summarise config.json")
    print(result.final_output)

asyncio.run(main())
```

---

## Responses API Integration

### Unified API Combining Chat and Assistant Capabilities

Use the Responses API for advanced features:

```python
from agents import Agent, Runner, ModelSettings, ReasoningItem
from openai.types.shared import Reasoning
import asyncio

agent = Agent(
    name="Research Assistant",
    instructions="Perform thorough research and analysis",
    model_settings=ModelSettings(
        reasoning=Reasoning(effort="high", summary="auto")
    )
)

async def main():
    result = await Runner.run(
        agent,
        "Analyse the implications of quantum computing on cryptography"
    )
    
    # Reasoning summaries arrive as ReasoningItem entries in new_items
    print("Reasoning:")
    for item in result.new_items:
        if isinstance(item, ReasoningItem):
            for part in item.raw_item.summary:
                print(part.text)
    print("\nFinal Response:")
    print(result.final_output)

asyncio.run(main())
```

### Reasoning Effort Configuration

Control reasoning depth:

```python
from agents import Agent, Runner, ModelSettings
from openai.types.shared import Reasoning
import asyncio

agents = [
    Agent(
        name="Quick Thinker",
        instructions="Answer quickly",
        model_settings=ModelSettings(
            reasoning=Reasoning(effort="low")
        )
    ),
    Agent(
        name="Balanced Thinker",
        instructions="Balance speed and depth",
        model_settings=ModelSettings(
            reasoning=Reasoning(effort="medium")
        )
    ),
    Agent(
        name="Deep Thinker",
        instructions="Analyse thoroughly",
        model_settings=ModelSettings(
            reasoning=Reasoning(effort="high")
        )
    )
]

async def main():
    query = "What's the best approach to solve P vs NP problem?"
    
    for agent in agents:
        result = await Runner.run(agent, query)
        print(f"\n{agent.name}:")
        print(f"Response: {result.final_output[:150]}")

asyncio.run(main())
```

### Built-in Web Search

Leverage built-in web search:

```python
from agents import Agent, Runner, WebSearchTool, ModelSettings
from openai.types.shared import Reasoning
import asyncio

agent = Agent(
    name="Search Assistant",
    instructions="Search the web for latest information",
    tools=[WebSearchTool()],
    model_settings=ModelSettings(
        reasoning=Reasoning(effort="high")
    )
)

async def main():
    result = await Runner.run(
        agent,
        "What are the latest developments in AI safety research?"
    )
    
    print("Latest AI Safety Developments:")
    print(result.final_output)

asyncio.run(main())
```

---

## Tracing and Observability

### Built-in Tracing

Debug and monitor agent runs:

```python
from agents import Agent, Runner, trace
import asyncio

async def main():
    agent = Agent(
        name="Analyser",
        instructions="Analyse data"
    )

    # Group related runs together
    with trace(
        workflow_name="Data Analysis Pipeline",
        group_id="batch_001",
        metadata={
            "environment": "production",
            "version": "1.0.0"
        }
    ):
        # Multiple runs are grouped
        result1 = await Runner.run(agent, "Analyse sales data")
        result2 = await Runner.run(agent, "Analyse customer feedback")

    print("Traces available at: https://platform.openai.com/traces")

asyncio.run(main())
```

### Trace Exports and Analysis

Export and analyse traces:

```python
from agents import Agent, Runner, trace, add_trace_processor, TracingProcessor
import asyncio
import json

class JsonFileExporter(TracingProcessor):
    """Collects finished spans and writes them to a JSON file per trace."""

    def __init__(self):
        self.spans: dict[str, list[dict]] = {}

    def on_trace_start(self, trace):
        self.spans[trace.trace_id] = []

    def on_span_start(self, span):
        pass

    def on_span_end(self, span):
        self.spans.setdefault(span.trace_id, []).append(span.export())

    def on_trace_end(self, trace):
        data = {"trace": trace.export(), "spans": self.spans.pop(trace.trace_id, [])}
        with open(f"trace_{trace.trace_id}.json", "w") as f:
            json.dump(data, f, indent=2, default=str)

    def shutdown(self):
        pass

    def force_flush(self):
        pass

# Runs alongside the default OpenAI exporter; use set_trace_processors() to replace it
add_trace_processor(JsonFileExporter())

async def main():
    agent = Agent(name="Assistant")

    with trace("Analysis Run"):
        result = await Runner.run(agent, "Perform analysis")

asyncio.run(main())
```

### Sending Traces to Other Platforms

The tracing system is pluggable: anything that implements `TracingProcessor` receives every trace and span. `add_trace_processor()` adds a destination alongside the OpenAI dashboard; `set_trace_processors()` replaces it. Langfuse, Logfire, Arize Phoenix/OpenInference, AgentOps, Braintrust and other platforms ship processors or instrumentors for this SDK; use the vendor's integration rather than wrapping `Runner.run` in a generic decorator, which records only the outer call and none of the agent, tool and model spans. The upstream [tracing page](https://openai.github.io/openai-agents-python/tracing/) lists the current integrations.

What goes into traces, how to disable them, and flushing in short-lived processes are covered in the [production guide](/openai-agents-sdk-guides/openai_agents_sdk_production_guide/#tracing-and-data-controls).

### Usage Tracking and Token Counting

Monitor costs and usage:

```python
from agents import Agent, Runner
import asyncio

async def main():
    agent = Agent(name="Assistant")

    result = await Runner.run(agent, "Explain quantum computing")

    # Usage is aggregated on the run context
    usage = result.context_wrapper.usage
    print(f"Requests: {usage.requests}")
    print(f"Input tokens: {usage.input_tokens}")
    print(f"Output tokens: {usage.output_tokens}")
    print(f"Total tokens: {usage.total_tokens}")

asyncio.run(main())
```

---

## Realtime and Voice Agents

The SDK has two ways to build voice applications. Both reuse function tools, and both have deterministic test doubles in `agents.realtime.testing` and `agents.voice.testing`.

| | Realtime agents | Voice pipeline |
|---|---|---|
| Model | A speech-to-speech Realtime model (default `gpt-realtime-2.1` since 0.18) | Speech-to-text, then a normal agent, then text-to-speech |
| Transport | A persistent WebSocket session | One request per turn |
| Latency | Lowest; supports interruption (barge-in) | Higher, but every step is inspectable |
| Install | `pip install 'openai-agents[realtime]'` | `pip install 'openai-agents[voice]'` |

### Realtime Agents

A `RealtimeAgent` takes instructions, tools and handoffs (to other realtime agents). `RealtimeRunner.run()` returns a session; you send it audio or text and iterate its events. Audio arrives as PCM16 bytes for you to play or forward to a browser.

```python
import asyncio
from agents import function_tool
from agents.realtime import RealtimeAgent, RealtimeRunner

@function_tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"It is dry and mild in {city}."

agent = RealtimeAgent(
    name="Voice assistant",
    instructions="Speak in short, friendly sentences.",
    tools=[get_weather],
)

runner = RealtimeRunner(
    agent,
    config={
        "model_settings": {
            "output_modalities": ["audio"],
            "voice": "marin",
            "turn_detection": {"type": "semantic_vad"},
        }
    },
)

def play(pcm16: bytes) -> None:
    """Send audio to a speaker, a WebRTC track or a browser socket."""

async def main():
    session = await runner.run()
    async with session:
        # Microphone audio would go in with: await session.send_audio(chunk)
        await session.send_message("What's the weather in Leeds?")
        async for event in session:
            if event.type == "audio":
                play(event.audio.data)
            elif event.type == "audio_interrupted":
                pass  # the user spoke over the agent: stop playback
            elif event.type == "tool_start":
                print("tool:", event.tool.name, event.arguments)
            elif event.type == "audio_end":
                break
            elif event.type == "error":
                print("error:", event.error)
                break

asyncio.run(main())
```

Realtime sessions support output guardrails (checked on the transcript as it is produced), tool approvals (`tool_approval_required` events) and handoffs between realtime agents with `realtime_handoff`. For browser clients, the usual design is that the browser connects to OpenAI over WebRTC with a short-lived key and your server runs only tools; the TypeScript SDK's `RealtimeSession` covers that case.

### Voice Pipeline

`VoicePipeline` wraps an ordinary `Agent` with speech-to-text and text-to-speech. `SingleAgentVoiceWorkflow` runs one agent per turn and keeps history between turns:

```python
import asyncio
import numpy as np
from agents import Agent
from agents.voice import AudioInput, SingleAgentVoiceWorkflow, VoicePipeline

agent = Agent(
    name="Assistant",
    instructions="You are a polite assistant. Keep answers short.",
    model="%%model=openai%%",
)

pipeline = VoicePipeline(workflow=SingleAgentVoiceWorkflow(agent))

async def main():
    # Three seconds of 24 kHz mono PCM16; use your microphone buffer instead
    buffer = np.zeros(24_000 * 3, dtype=np.int16)
    result = await pipeline.run(AudioInput(buffer=buffer))

    audio_chunks = []
    async for event in result.stream():
        if event.type == "voice_stream_event_audio":
            audio_chunks.append(event.data)  # play these as they arrive
        elif event.type == "voice_stream_event_lifecycle":
            print("lifecycle:", event.event)

    print(f"received {len(audio_chunks)} audio chunks")

asyncio.run(main())
```

For continuous microphone input, pass a `StreamedAudioInput` instead of `AudioInput`; the pipeline then detects turns itself. `VoicePipelineConfig` sets the STT and TTS models, voice and tracing options.

---

## Model Providers

### Chat Completions and OpenAI-Compatible Endpoints

OpenAI models use the Responses API by default. `OpenAIChatCompletionsModel` uses Chat Completions instead, which is also how you reach any OpenAI-compatible endpoint (a local server, a gateway) by pointing the client at its `base_url`:

```python
from agents import Agent, ModelSettings, OpenAIChatCompletionsModel, Runner
from openai import AsyncOpenAI

client = AsyncOpenAI()  # or AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="...")

agent = Agent(
    name="Assistant",
    model=OpenAIChatCompletionsModel(model="%%model=openai-small%%", openai_client=client),
    model_settings=ModelSettings(temperature=0.7, max_tokens=1000),
)

print(Runner.run_sync(agent, "Say hello.").final_output)
```

Call `set_default_openai_api("chat_completions")` once at start-up to switch every OpenAI agent. Hosted tools (web search, file search, code interpreter, hosted MCP) need the Responses API.

### Using Non-OpenAI Models via LiteLLM

```bash
pip install 'openai-agents[litellm]'
```

Prefix the model name with `litellm/` followed by LiteLLM's own provider/model string. Each provider reads its usual API key variable (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, …).

```python
from agents import Agent, Runner
import asyncio

claude_agent = Agent(
    name="Claude Assistant",
    instructions="Answer briefly.",
    model="litellm/anthropic/%%model=anthropic%%",
)

gemini_agent = Agent(
    name="Gemini Assistant",
    instructions="Answer briefly.",
    model="litellm/gemini/%%model=gemini%%",
)

async def main():
    for agent in [claude_agent, gemini_agent]:
        result = await Runner.run(agent, "Explain machine learning briefly")
        print(f"\n{agent.name}: {result.final_output[:100]}")

asyncio.run(main())
```

Local models work the same way through LiteLLM's Ollama provider (`litellm/ollama_chat/<model-tag>`). Tracing still goes to OpenAI by default; set a tracing key with `set_tracing_export_api_key(...)` or disable tracing if you have no OpenAI account. The `[any-llm]` extra provides the same kind of adapter for the any-llm library (`model="any-llm/<provider>/<model>"`).

---

## Advanced Topics

### Lifecycle Hooks

To run code when an agent starts or ends, around every model call, or around every tool call, use `RunHooks` (whole run) or `AgentHooks` (one agent). See [Hooks, guardrails and approvals](/openai-agents-sdk-guides/openai_agents_sdk_middleware_python/#run-hooks) for examples, together with tool guardrails and human approval of tool calls.

To draw an agent graph (agents, tools, handoffs) for documentation, install `[viz]` and call `agents.extensions.visualization.draw_graph(agent)`, which returns a Graphviz object.

### Async/Await Patterns

Handle concurrent operations efficiently:

```python
import asyncio
from agents import Agent, Runner

async def process_multiple_agents(queries: list[str]):
    agent = Agent(name="Processor")
    
    # Run concurrently
    tasks = [
        Runner.run(agent, query)
        for query in queries
    ]
    
    results = await asyncio.gather(*tasks)
    return [r.final_output for r in results]

async def main():
    queries = [
        "What is AI?",
        "Explain machine learning",
        "Define deep learning"
    ]
    
    outputs = await process_multiple_agents(queries)
    for query, output in zip(queries, outputs):
        print(f"Q: {query}\nA: {output[:100]}\n")

asyncio.run(main())
```

### Error Recovery Strategies

Retries belong inside the run, not around it. Re-running a whole `Runner.run` after a failure repeats tool calls that may already have had side effects. The SDK retries individual model calls when you opt in:

```python
from agents import Agent, ModelRetrySettings, ModelSettings, retry_policies

agent = Agent(
    name="Resilient Agent",
    model="%%model=openai%%",
    model_settings=ModelSettings(
        timeout=60,  # per model-call attempt, in seconds
        retry=ModelRetrySettings(
            max_retries=3,
            backoff={"initial_delay": 1.0, "max_delay": 20.0, "multiplier": 2.0, "jitter": True},
            policy=retry_policies.any(
                retry_policies.provider_suggested(),
                retry_policies.network_error(),
                retry_policies.http_status([429, 500, 502, 503]),
            ),
        ),
    ),
)
```

Tool failures are returned to the model as an error message by default, so it can try again or explain; pass `failure_error_function=None` to `@function_tool` to raise instead. A hit turn limit, a refusal and an invalid final output can be turned into a normal result with `error_handlers={"max_turns": ..., "model_refusal": ..., "invalid_final_output": ...}`. The [production guide](/openai-agents-sdk-guides/openai_agents_sdk_production_guide/#bound-every-run) covers these together with timeouts.

### Testing Agent Applications

`agents.testing.ScriptedModel` (since 0.21) is a deterministic stand-in for a model. Script the tool calls and messages it should return; tests then run offline and can assert on the tools that ran and on what was sent to the model:

```python
from agents import Agent, Runner, function_tool
from agents.testing import ScriptedModel, assistant_message, function_call

@function_tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def test_calculator_uses_tool():
    model = ScriptedModel([
        [function_call("add", {"a": 5, "b": 3}, call_id="call_1")],
        [assistant_message("5 + 3 = 8")],
    ])
    agent = Agent(name="Calculator", tools=[add], model=model)

    result = Runner.run_sync(agent, "What's 5 + 3?")

    assert result.final_output == "5 + 3 = 8"
    tool_outputs = [i.output for i in result.new_items if i.type == "tool_call_output_item"]
    assert tool_outputs == [8]
    model.assert_complete()

test_calculator_uses_tool()
```

More patterns (retries, streaming, structured output, swapping the model for a whole run) are in the [production guide](/openai-agents-sdk-guides/openai_agents_sdk_production_guide/#testing-without-the-api).

### Cost Optimisation Strategies

Route simple work to a small model and keep the larger model for queries that need it:

```python
from agents import Agent, Runner, ModelSettings
import asyncio

# Small, cheap model for simple queries
fast_agent = Agent(
    name="Budget Agent",
    instructions="Answer quickly",
    model="%%model=openai-small%%",
    model_settings=ModelSettings(max_tokens=200, verbosity="low"),
)

# Frontier model for complex queries
premium_agent = Agent(
    name="Premium Agent",
    instructions="Analyse thoroughly",
    model="%%model=openai%%",
    model_settings=ModelSettings(max_tokens=2000),
)

async def main():
    # Route based on complexity
    queries = [
        ("Simple query", "What's 2+2?", fast_agent),
        ("Complex query", "Analyse quantum computing implications", premium_agent)
    ]
    
    for complexity, query, agent in queries:
        result = await Runner.run(agent, query)
        usage = result.context_wrapper.usage
        print(f"{complexity}: {usage.total_tokens} tokens")

asyncio.run(main())
```

### Sandbox Agents — Isolated Execution Environment

Added in 0.14.0 (the docs no longer label it beta as of 0.23). A `SandboxAgent` is an ordinary agent plus a workspace: a `Manifest` describes the files, directories, repos and mounts to stage, capabilities add sandbox-native tools (filesystem, shell, skills, memory), and `SandboxRunConfig` chooses where the work runs. You still run it with `Runner`.

```python
import asyncio
from pathlib import Path

from agents import Runner
from agents.run import RunConfig
from agents.sandbox import Manifest, SandboxAgent, SandboxRunConfig
from agents.sandbox.capabilities import Capabilities
from agents.sandbox.entries import Dir, LocalDir
from agents.sandbox.sandboxes.unix_local import UnixLocalSandboxClient

agent = SandboxAgent(
    name="Data Analyst",
    model="%%model=openai%%",
    instructions=(
        "Analyse the CSV files in data/. Write summaries and charts to output/ "
        "and mention the exact commands you ran."
    ),
    default_manifest=Manifest(
        entries={
            "data": LocalDir(src=Path("data")),       # copied from the host (must sit under the CWD)
            "output": Dir(description="Generated artefacts."),
        }
    ),
    capabilities=Capabilities.default(),             # filesystem + shell tools
)

async def main():
    result = await Runner.run(
        agent,
        "Calculate summary statistics for data/sales.csv and save a histogram to output/.",
        run_config=RunConfig(
            sandbox=SandboxRunConfig(client=UnixLocalSandboxClient()),
            workflow_name="Sandbox analysis",
        ),
    )
    print(result.final_output)

asyncio.run(main())
```

The `data/` directory must exist next to the script (here with `sales.csv` in it), because `LocalDir` copies it from the host. `UnixLocalSandboxClient` runs commands with your own user's permissions (on Linux it adds no OS-level confinement), so use it only for trusted local development. For untrusted input use `DockerSandboxClient` (`pip install 'openai-agents[docker]'`) or a hosted backend (E2B, Modal, Daytona, Vercel, Cloudflare, Blaxel). Later runs can reconnect to prior work through `SandboxRunConfig(session=..., session_state=..., snapshot=...)` or a saved `RunState`.

### WebSocket Transport for the Responses API

Since 0.10.0 OpenAI Responses models can use a WebSocket transport instead of HTTP, which lowers per-turn overhead in multi-turn and tool-heavy runs. It is opt-in, and it is a model transport — not a server for your own browser clients.

```python
import asyncio
from agents import Agent, responses_websocket_session

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
)

async def main():
    # Shares one websocket-capable provider and RunConfig across turns
    async with responses_websocket_session() as ws:
        result = ws.run_streamed(agent, "Plan a three-day trip to Lisbon.")
        async for event in result.stream_events():
            if event.type == "raw_response_event" and event.data.type == "response.output_text.delta":
                print(event.data.delta, end="", flush=True)

        # Follow-up turn on the same connection
        follow_up = ws.run_streamed(
            agent,
            "Make day two cheaper.",
            previous_response_id=result.last_response_id,
        )
        async for _ in follow_up.stream_events():
            pass
        print("\n", follow_up.final_output)

asyncio.run(main())
```

To switch transport globally instead, call `set_default_openai_responses_transport("websocket")`, or pass `OpenAIProvider(use_responses_websocket=True)` as the run's `model_provider`.

---

## Summary

1. Start with one agent and `Runner.run`; add tools before adding agents.
2. Use handoffs when a specialist should own the reply, and `as_tool()` when a manager should keep control.
3. Give agents an `output_type` when code consumes the result.
4. Use sessions for multi-turn conversations, keyed by conversation.
5. Add guardrails, tool guardrails and approvals where inputs are untrusted or tools have side effects.
6. Bound runs with `max_turns`, timeouts and retries, and test with `ScriptedModel`.
7. Read traces when an agent misbehaves; decide early what may appear in them.

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| %%version=openai-agents%% | 2026-10-24 | Audit: wrapped the run-result snippet in an async function; removed an incorrect claim that the cheap-model example used the SDK default model; noted the sandbox example's `data/` prerequisite. |
| %%version=openai-agents%% | %%versions-checked%% | Fixed hosted-tool examples (`tool_config` is required for code interpreter and image generation), non-strict output types (`AgentOutputSchema`), the MCP Git server and GitHub MCP URL, and a session example that indexed a coroutine. Replaced the abstract custom-MCP-server class with a real MCP server, the placeholder voice and CLI examples with `RealtimeAgent` and `VoicePipeline`, and the hand-rolled retry and live-API tests with `ModelSettings.retry` and `ScriptedModel`. Added handoff input filters, hosted MCP, programmatic tool calling and Chat Completions. Moved the release table to the overview. Earlier in the same revision: re-verified against the current release (first written for 0.6.x, last revised for 0.17.4). Replaced APIs that never existed or were removed (`InMemorySession`, `ComputerControlTool`, `get_trace_context`, `result.output_reasoning`/`.messages`/`.agent`/`.usage`, `temperature`/`max_tokens` on `Agent`, `response_format` on `ModelSettings`, `agents.transports`); rewrote the Sandbox Agents and WebSocket sections against the real API; replaced the breaking-changes table with the upstream changelog (0.4–0.22); updated model names and install extras. |
| 0.17.4 | May 27, 2026 | Version bump; key symbols re-checked. |
| 0.14.0 | April 2026 | Sandbox Agents section added. |
| 0.6.1 | November 2025 | First version of this guide. |
