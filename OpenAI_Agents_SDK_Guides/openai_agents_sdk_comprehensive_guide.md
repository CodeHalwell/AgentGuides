# OpenAI Agents SDK: Comprehensive Technical Guide

A complete reference for building production-ready multi-agent AI applications with the OpenAI Agents SDK. This guide covers everything from installation through advanced patterns.

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
15. [Realtime Experiences](#realtime-experiences)
16. [Model Providers](#model-providers)
17. [Advanced Topics](#advanced-topics)

---

## Installation and Environment Setup

### Basic Installation

The OpenAI Agents SDK requires Python 3.9 or newer and can be installed via pip:

```bash
pip install openai-agents
```

For development with optional features, install with extras:

```bash
# Voice support for real-time applications
pip install 'openai-agents[voice]'

# Redis session support for distributed deployments
pip install 'openai-agents[redis]'

# All features
pip install 'openai-agents[all]'

# LiteLLM support for multi-provider model access
pip install 'openai-agents[litellm]'
```

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

### Project Structure Setup

Recommended project layout for SDK-based applications:

```
my_agent_app/
├── main.py
├── .env
├── .gitignore
├── requirements.txt
├── agents/
│   ├── __init__.py
│   ├── base_agents.py
│   ├── tools.py
│   └── guardrails.py
├── sessions/
│   ├── __init__.py
│   ├── storage.py
│   └── migrations.py
├── utils/
│   ├── __init__.py
│   ├── logging.py
│   └── tracing.py
└── tests/
    ├── __init__.py
    ├── test_agents.py
    └── test_tools.py
```

---

## Design Philosophy

### Lightweight Primitives vs. Heavy Abstractions

The OpenAI Agents SDK intentionally favours simplicity and minimalism. Rather than providing a kitchen-sink framework, it offers lightweight primitives that compose naturally:

**Philosophy Core Principles:**

1. **Simplicity First**: Minimal API surface area, intuitive interfaces
2. **Composition**: Primitives combine to build complex systems
3. **Customisation**: Extensible architecture without enforced patterns
4. **Python-Native**: Leverages Python features (async/await, decorators, context managers)
5. **No Magic**: Explicit is better than implicit; transparent operation

### Comparison with Experimental Swarm Framework

The SDK represents a production-ready evolution from the experimental OpenAI Swarm framework:

| Aspect | Swarm | Agents SDK |
|--------|-------|-----------|
| **Status** | Experimental | Production-Ready |
| **API Stability** | Unstable | Stable |
| **Guardrails** | None | Built-in |
| **Sessions** | Manual handling | Automatic |
| **Tracing** | Basic | Comprehensive |
| **Schema Generation** | Basic | Pydantic-based |
| **Error Handling** | Limited | Extensive |
| **MCP Support** | None | First-class |
| **Model Support** | OpenAI only | 100+ via LiteLLM |
| **Async/Await** | Limited | Full support |
| **Documentation** | Minimal | Comprehensive |

### Upgrade Path from Swarm

For developers migrating from Swarm to Agents SDK:

```python
# Swarm (experimental)
from swarm import Swarm, Agent
client = Swarm()
result = client.run(agent, "message")

# Agents SDK (production)
from agents import Agent, Runner
result = await Runner.run(agent, "message")
```

Key differences:

1. **Import statements**: Use `agents` instead of `swarm`
2. **Async support**: Use `await Runner.run()` for async operations
3. **Sessions**: Explicitly pass session parameter instead of manual memory handling
4. **Error handling**: Comprehensive exception hierarchy with specific error types
5. **Configuration**: Use `ModelSettings` instead of inline parameters

---

## Core Primitives

### Agent

The foundational primitive representing an LLM configured with instructions, tools, and guardrails.

**Basic Structure:**

```python
from agents import Agent

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant that provides accurate information",
    model="gpt-4o",
    temperature=0.7
)
```

**Complete Agent Definition:**

```python
from agents import Agent, function_tool
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
    model="gpt-4o",
    tools=[execute_task],
    output_type=TaskResult,
    temperature=0.5,
    max_tokens=1000
)
```

**Agent Parameters:**

- `name` (str): Unique identifier for the agent within the system
- `instructions` (str | Callable): Agent's system prompt or dynamic instruction generator
- `model` (str): Model identifier (e.g., "gpt-4o", "gpt-5", "gpt-5-nano")
- `tools` (list): Function tools available to the agent
- `handoffs` (list): Other agents this agent can delegate to
- `input_guardrails` (list): Validation functions for inputs
- `output_guardrails` (list): Validation functions for outputs
- `output_type` (type): Pydantic model for structured output
- `temperature` (float): Sampling temperature (0.0 to 2.0)
- `max_tokens` (int): Maximum response tokens
- `model_settings` (ModelSettings): Advanced model configuration
- `context_type` (type): Type hint for injected context

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
- `max_turns`: Maximum agent loop iterations (default: 10)
- `model_settings`: Override agent's model settings
- `debug`: Enable debug output

**Runner Result:**

```python
from agents import Agent, Runner

result = await Runner.run(agent, "message")

# Access results
print(result.final_output)          # Processed output
print(result.output_type)           # Output schema type
print(result.messages)              # Conversation history
print(result.messages_to_input())   # Format for next run
print(result.agent)                 # Final active agent
print(result.custom_output)         # Custom runner output
print(result.output_reasoning)      # Reasoning trace
```

### Handoff

A specialized mechanism for transferring control between agents, essential for multi-agent systems.

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
from agents import Agent, handoff, RunContextWrapper
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

```python
from agents import Agent, Runner, InMemorySession
import asyncio

async def main():
    session = InMemorySession()
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
                print(f"🔧 Tool called: {item.name}")
            elif item.type == "tool_call_output_item":
                print(f"📤 Result: {item.output}")
            elif item.type == "message_output_item":
                text = ItemHelpers.text_message_output(item)
                print(f"💬 Response: {text[:100]}")

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
            print(f"✓ Success: {response.data}")
        else:
            print(f"✗ Error: {response.error}")

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
        print(f"Agent: {result.agent.name}")
        print(f"Response: {result.final_output[:200]}")

asyncio.run(main())
```

### Agent Delegation with Message Filtering

Control what information flows during handoffs:

```python
from agents import Agent, Runner, handoff, RunContextWrapper
from pydantic import BaseModel
import asyncio

class SupportTicket(BaseModel):
    issue_type: str
    urgency: str
    customer_context: str

async def prepare_escalation(
    ctx: RunContextWrapper[None],
    ticket_data: SupportTicket
):
    print(f"Escalating {ticket_data.issue_type} (Urgency: {ticket_data.urgency})")
    # Log to ticket system, notify manager, etc.

manager_agent = Agent(
    name="Manager",
    instructions="Review and handle escalated support tickets with priority"
)

support_agent = Agent(
    name="Support",
    instructions="Handle customer support requests",
    handoffs=[
        handoff(
            agent=manager_agent,
            on_handoff=prepare_escalation,
            input_type=SupportTicket,
            tool_description_override="Escalate to manager for urgent issues"
        )
    ]
)

async def main():
    result = await Runner.run(
        support_agent,
        "I need immediate help - my account has been hacked!"
    )
    print(result.final_output)

asyncio.run(main())
```

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

summarizer = Agent(
    name="Summarizer",
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
        summarizer.as_tool(
            tool_name="summarize",
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

```python
from agents import Agent, Runner, CodeInterpreterTool
import asyncio

agent = Agent(
    name="Data Analyst",
    instructions="Analyse data and generate visualisations",
    tools=[CodeInterpreterTool()]
)

async def main():
    result = await Runner.run(
        agent,
        "Analyse this data and create a visualization: [1,2,3,4,5]"
    )
    print(result.final_output)

asyncio.run(main())
```

**Computer Control Tool (38.1% OSWorld benchmark):**

```python
from agents import Agent, Runner, ComputerControlTool
import asyncio

agent = Agent(
    name="Automation Agent",
    instructions="Automate computer tasks",
    tools=[ComputerControlTool()]
)

async def main():
    result = await Runner.run(
        agent,
        "Open a text editor and write a greeting"
    )
    print(result.final_output)

asyncio.run(main())
```

**Image Generation Tool:**

```python
from agents import Agent, Runner, ImageGenerationTool
import asyncio

agent = Agent(
    name="Creative Designer",
    instructions="Generate images based on descriptions",
    tools=[ImageGenerationTool()]
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
    output_type=TaskBreakdown,
    model_settings=ModelSettings(
        temperature=0,  # Ensure consistency
        response_format="json_object"
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

Allow more flexibility with output validation:

```python
from agents import Agent, Runner
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
    output_type=FlexibleResponse
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
            "url": "https://api.github.com/mcp",
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
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-git",
                str(repo_path)
            ]
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

Build your own MCP server:

```python
from agents import Agent, Runner
from agents.mcp import MCPServer
import asyncio

class CustomToolServer(MCPServer):
    """Custom MCP server with domain-specific tools."""
    
    async def handle_call_tool(self, name: str, arguments: dict):
        if name == "calculate_roi":
            investment = arguments.get("investment", 0)
            return_value = arguments.get("return_value", 0)
            roi = ((return_value - investment) / investment) * 100
            return {"roi": roi}
        
        raise ValueError(f"Unknown tool: {name}")

async def main():
    server = CustomToolServer(name="Financial Tools")
    
    agent = Agent(
        name="Financial Advisor",
        instructions="Provide financial analysis",
        mcp_servers=[server]
    )
    
    result = await Runner.run(
        agent,
        "Calculate ROI on a £1000 investment that returns £1500"
    )
    print(result.final_output)

asyncio.run(main())
```

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
            print(f"✓ Query allowed: {result.final_output[:100]}")
        except InputGuardrailTripwireTriggered:
            print(f"✗ Query blocked: Potential SQL injection")

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

### Streaming Guardrails

Validate streaming responses:

```python
from agents import Agent, Runner, output_guardrail, GuardrailFunctionOutput
from openai.types.responses import ResponseTextDeltaEvent
import asyncio

@output_guardrail
async def stream_safety_check(ctx, agent, output):
    # Check for safety patterns
    if "forbidden_content" in str(output).lower():
        return GuardrailFunctionOutput(
            output_info={"safe": False},
            tripwire_triggered=True
        )
    return GuardrailFunctionOutput(
        output_info={"safe": True},
        tripwire_triggered=False
    )

agent = Agent(
    name="Streamer",
    instructions="Generate content",
    output_guardrails=[stream_safety_check]
)

async def main():
    result = Runner.run_streamed(agent, "Generate a story")
    
    print("Streaming with guardrails:")
    async for event in result.stream_events():
        if event.type == "raw_response_event":
            if isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)

asyncio.run(main())
```

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

    # Get specific item
    last_item = await session.get_items()[-1] if items else None

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
from agents import Agent, Runner
from agents.extensions.memory import SQLAlchemySession
from sqlalchemy import create_engine, Column, String, Text, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import asyncio

Base = declarative_base()

class ConversationRecord(Base):
    __tablename__ = "conversations"
    id = Column(String, primary_key=True)
    session_id = Column(String)
    message_content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

engine = create_engine("postgresql://user:password@localhost/conversations")
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

async def main():
    agent = Agent(name="Assistant")

    session = SQLAlchemySession(
        "user_456",
        Session(),
        ConversationRecord
    )

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
    name="Document Analyzer",
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
    instructions="Provide personalized responses based on user context",
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
    name="Sentiment Analyzer",
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
from agents import Agent, Runner, ModelSettings
from openai.types.shared import Reasoning
import asyncio

agent = Agent(
    name="Research Assistant",
    instructions="Perform thorough research and analysis",
    model_settings=ModelSettings(
        reasoning=Reasoning(effort="high")
    )
)

async def main():
    result = await Runner.run(
        agent,
        "Analyse the implications of quantum computing on cryptography"
    )
    
    print("Reasoning:")
    print(result.output_reasoning)
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
            reasoning=Reasoning(effort="minimal")
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

### Built-in Web Search (90% SimpleQA Accuracy)

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

### Built-in Tracing Visualization

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
from agents import Agent, Runner, trace, get_trace_context
import asyncio
import json

async def main():
    agent = Agent(name="Assistant")

    with trace("Analysis Run"):
        result = await Runner.run(agent, "Perform analysis")

    # Export trace
    trace_context = get_trace_context()
    
    if trace_context:
        trace_data = {
            "trace_id": trace_context.trace_id,
            "spans": trace_context.spans,
            "metadata": trace_context.metadata
        }
        
        with open("trace_export.json", "w") as f:
            json.dump(trace_data, f, indent=2)

asyncio.run(main())
```

### Integration with Langfuse for Evaluation

Connect to Langfuse observability platform:

```python
from agents import Agent, Runner, trace
from langfuse.decorators import observe
import asyncio

@observe()
async def run_with_langfuse():
    agent = Agent(
        name="Evaluatable Agent",
        instructions="Generate evaluatable responses"
    )

    result = await Runner.run(agent, "What is machine learning?")
    return result.final_output

async def main():
    output = await run_with_langfuse()
    print("Response logged to Langfuse:")
    print(output)

asyncio.run(main())
```

### Usage Tracking and Token Counting

Monitor costs and usage:

```python
from agents import Agent, Runner
import asyncio

async def main():
    agent = Agent(name="Assistant")

    result = await Runner.run(agent, "Explain quantum computing")

    # Access usage information
    if hasattr(result, 'usage'):
        print(f"Input tokens: {result.usage.input_tokens}")
        print(f"Output tokens: {result.usage.output_tokens}")
        print(f"Total tokens: {result.usage.input_tokens + result.usage.output_tokens}")

asyncio.run(main())
```

---

## Realtime Experiences

### Command-Line Interface Agents

Build interactive CLI agents:

```python
from agents import Agent, Runner
import asyncio

async def main():
    agent = Agent(
        name="CLI Assistant",
        instructions="Help users with command-line tasks"
    )

    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ["exit", "quit"]:
            break
        
        result = await Runner.run(agent, user_input)
        print(f"Assistant: {result.final_output}\n")

asyncio.run(main())
```

### Voice Agents with TTS and STT

Build conversational voice experiences:

```python
from agents import Agent, Runner
import asyncio

async def main():
    agent = Agent(
        name="Voice Assistant",
        instructions="Respond conversationally to voice input"
    )

    # Example voice input (would normally come from STT)
    voice_transcript = "What's the weather like?"

    result = await Runner.run(agent, voice_transcript)

    # Convert response to speech (would normally use TTS)
    print(f"TTS Output: {result.final_output}")

asyncio.run(main())
```

---

## Model Providers

### Using Non-OpenAI Models via LiteLLM

Access 100+ model providers:

```bash
pip install 'openai-agents[litellm]'
```

```python
from agents import Agent, Runner
import asyncio
import os

os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
os.environ["GOOGLE_API_KEY"] = "..."

# Use Claude via LiteLLM
claude_agent = Agent(
    name="Claude Assistant",
    instructions="You are Claude",
    model="litellm/anthropic/claude-3-5-sonnet-20240620"
)

# Use Gemini via LiteLLM
gemini_agent = Agent(
    name="Gemini Assistant",
    instructions="You are Gemini",
    model="litellm/gemini/gemini-2.0-flash"
)

# Use Llama via LiteLLM
llama_agent = Agent(
    name="Llama Assistant",
    instructions="You are Llama",
    model="litellm/replicate/meta-llama/llama-2-70b-chat"
)

async def main():
    for agent in [claude_agent, gemini_agent, llama_agent]:
        result = await Runner.run(agent, "Explain machine learning briefly")
        print(f"\n{agent.name}: {result.final_output[:100]}")

asyncio.run(main())
```

---

## Advanced Topics

### Agent Lifecycle Management

Control agent startup, shutdown, and cleanup:

```python
from agents import Agent, Runner
import asyncio
from contextlib import asynccontextmanager

@asynccontextmanager
async def managed_agent():
    # Setup
    agent = Agent(name="Managed Agent")
    print("Agent initialised")
    
    try:
        yield agent
    finally:
        # Cleanup
        print("Agent cleaned up")

async def main():
    async with managed_agent() as agent:
        result = await Runner.run(agent, "Do something")
        print(result.final_output)

asyncio.run(main())
```

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

Implement retry logic and fallbacks:

```python
from agents import Agent, Runner
import asyncio
import random

class RetryableRunner:
    def __init__(self, max_retries: int = 3, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
    
    async def run_with_retry(self, agent: Agent, input_text: str):
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                result = await Runner.run(agent, input_text)
                return result
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = self.backoff_factor ** attempt
                    print(f"Retry {attempt + 1}/{self.max_retries} after {wait_time}s")
                    await asyncio.sleep(wait_time)
        
        raise last_error

async def main():
    runner = RetryableRunner(max_retries=3)
    agent = Agent(name="Resilient Agent")
    
    result = await runner.run_with_retry(agent, "Test query")
    print(result.final_output)

asyncio.run(main())
```

### Testing Agent Applications

Comprehensive testing patterns:

```python
from agents import Agent, Runner
import asyncio
import pytest

class TestAgents:
    @pytest.mark.asyncio
    async def test_basic_agent(self):
        agent = Agent(name="Test Agent")
        result = await Runner.run(agent, "2 + 2")
        
        assert result.final_output is not None
        assert "4" in result.final_output

    @pytest.mark.asyncio
    async def test_agent_with_tools(self):
        from agents import function_tool
        
        @function_tool
        def add(a: int, b: int) -> int:
            return a + b
        
        agent = Agent(
            name="Calculator",
            tools=[add]
        )
        
        result = await Runner.run(agent, "What's 5 + 3?")
        assert result.final_output is not None

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### Cost Optimization Strategies

Minimize API costs:

```python
from agents import Agent, Runner, ModelSettings
import asyncio

# Fast, cheap model for simple queries
fast_agent = Agent(
    name="Budget Agent",
    instructions="Answer quickly",
    model="gpt-4o-mini",
    model_settings=ModelSettings(
        temperature=0,
        max_tokens=200
    )
)

# Premium model for complex queries
premium_agent = Agent(
    name="Premium Agent",
    instructions="Analyse thoroughly",
    model="gpt-4o",
    model_settings=ModelSettings(
        temperature=0.7,
        max_tokens=2000
    )
)

async def main():
    # Route based on complexity
    queries = [
        ("Simple query", "What's 2+2?", fast_agent),
        ("Complex query", "Analyse quantum computing implications", premium_agent)
    ]
    
    for complexity, query, agent in queries:
        result = await Runner.run(agent, query)
        print(f"{complexity}: Cost optimised ✓")

asyncio.run(main())
```

---

## Summary and Best Practices

The OpenAI Agents SDK provides a production-ready framework for building sophisticated multi-agent applications. Key takeaways:

1. **Start Simple**: Begin with basic agents and gradually add complexity
2. **Use Sessions**: Always maintain conversation context with sessions
3. **Implement Guardrails**: Validate inputs and outputs for safety
4. **Leverage Handoffs**: Use agent delegation for specialised tasks
5. **Monitor Costs**: Use faster models for simple tasks
6. **Trace Everything**: Enable tracing for debugging and optimisation
7. **Handle Errors**: Implement comprehensive error handling
8. **Test Thoroughly**: Unit test agents and integration test workflows
9. **Compose Tools**: Build sophisticated workflows from simple tools
10. **Iterate Constantly**: Use traces and feedback to improve agents

This comprehensive framework enables developers to build production-grade AI applications with confidence, from simple chatbots to complex multi-agent research systems.

