# OpenAI Agents SDK: Complete Migration Guide from Swarm

**Status:** Production-Ready (2025)
**Swarm Status:** Experimental - Deprecated
**Migration Priority:** HIGH
**Last Updated:** January 2025

---

## Executive Summary

The **OpenAI Agents SDK** is the official, production-ready evolution of the experimental Swarm framework. OpenAI has transitioned from Swarm (experimental) to the Agents SDK (production-ready) with significant improvements in stability, features, and enterprise support.

**Key Decision Factors:**

- ✅ **Agents SDK**: Production-ready, actively maintained, comprehensive features
- ❌ **Swarm**: Experimental, limited support, no production guarantees

---

## Table of Contents

1. [Why Migrate from Swarm?](#why-migrate-from-swarm)
2. [Key Improvements in Agents SDK](#key-improvements-in-agents-sdk)
3. [Side-by-Side Comparison](#side-by-side-comparison)
4. [Migration Checklist](#migration-checklist)
5. [Code Migration Examples](#code-migration-examples)
6. [Breaking Changes and Solutions](#breaking-changes-and-solutions)
7. [Testing Your Migration](#testing-your-migration)
8. [Rollback Strategy](#rollback-strategy)

---

## Why Migrate from Swarm?

### Swarm's Limitations

1. **Experimental Status**: No production guarantees, API instability
2. **Limited Features**: Missing guardrails, tracing, and advanced session management
3. **No Active Development**: Deprecated in favour of Agents SDK
4. **Missing Enterprise Features**: No built-in observability or provider flexibility

### Agents SDK Advantages

1. **Production-Ready**: Stable API, comprehensive testing, battle-tested
2. **Active Maintenance**: Regular updates, security patches, feature additions
3. **Enterprise Features**: Guardrails, tracing, session backends, MCP support
4. **Provider Flexibility**: Support for 100+ LLM providers via LiteLLM
5. **Better Performance**: Optimised execution, efficient token usage
6. **Comprehensive Documentation**: Extensive guides, recipes, production patterns

---

## Key Improvements in Agents SDK

### 1. Built-in Guardrails

**Swarm**: No native guardrail support
**Agents SDK**: Comprehensive input/output validation

```python
# Agents SDK - Input Guardrail
from agents import Agent, input_guardrail, GuardrailFunctionOutput

@input_guardrail
async def safety_check(ctx, agent, input_data):
    unsafe_patterns = ["malicious", "harmful"]
    is_unsafe = any(pattern in str(input_data).lower() for pattern in unsafe_patterns)

    return GuardrailFunctionOutput(
        output_info={"safety": "fail" if is_unsafe else "pass"},
        tripwire_triggered=is_unsafe
    )

agent = Agent(
    name="Safe Assistant",
    instructions="Help users safely",
    input_guardrails=[safety_check]
)
```

### 2. Automatic Session Management

**Swarm**: Manual memory handling required
**Agents SDK**: Multiple session backends with automatic persistence

```python
from agents import Agent, Runner, SQLiteSession, RedisSession

# SQLite for local/small deployments
session = SQLiteSession("user_123", "sessions.db")

# Redis for distributed/production
session = RedisSession.from_url("user_123", url="redis://localhost:6379")

# Automatic conversation history
result = await Runner.run(agent, "Hello", session=session)
```

### 3. Built-in Tracing and Observability

**Swarm**: Basic logging only
**Agents SDK**: Comprehensive tracing and visualization

```python
from agents import Agent, Runner, trace

with trace(
    workflow_name="Customer Support",
    group_id="batch_001",
    metadata={"environment": "production"}
):
    result1 = await Runner.run(agent, "Query 1")
    result2 = await Runner.run(agent, "Query 2")

# View traces at https://platform.openai.com/traces
```

### 4. Pydantic-Powered Tools

**Swarm**: Basic schema generation
**Agents SDK**: Full Pydantic validation with automatic schema extraction

```python
from agents import Agent, function_tool
from pydantic import BaseModel, Field

class BookingRequest(BaseModel):
    customer_name: str = Field(..., description="Customer's full name")
    email: str = Field(..., pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
    check_in: str = Field(..., description="Check-in date (YYYY-MM-DD)")
    guests: int = Field(ge=1, le=10)

@function_tool
def book_hotel(request: BookingRequest) -> dict:
    """Book a hotel with automatic validation."""
    return {"status": "confirmed", "booking_id": "BK123"}

agent = Agent(
    name="Booking Agent",
    tools=[book_hotel]
)
```

### 5. Provider-Agnostic Support

**Swarm**: OpenAI models only
**Agents SDK**: 100+ LLM providers via LiteLLM

```python
# Use Claude
agent = Agent(
    name="Claude Agent",
    model="litellm/anthropic/claude-3-5-sonnet-20240620",
    instructions="You are Claude"
)

# Use Gemini
agent = Agent(
    name="Gemini Agent",
    model="litellm/gemini/gemini-2.0-flash",
    instructions="You are Gemini"
)

# Use Llama
agent = Agent(
    name="Llama Agent",
    model="litellm/replicate/meta-llama/llama-2-70b-chat",
    instructions="You are Llama"
)
```

### 6. Enhanced MCP Integration

**Swarm**: No MCP support
**Agents SDK**: First-class Model Context Protocol integration

```python
from agents import Agent, Runner
from agents.mcp import MCPServerStdio, MCPServerStreamableHttp
from pathlib import Path

# Filesystem MCP
async with MCPServerStdio(
    name="Filesystem",
    params={
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", str(Path("./"))]
    }
) as server:
    agent = Agent(
        name="File Assistant",
        mcp_servers=[server]
    )

    result = await Runner.run(agent, "List all Python files")
```

### 7. Structured Outputs with Validation

**Swarm**: Manual JSON parsing
**Agents SDK**: Automatic Pydantic validation

```python
from agents import Agent, Runner
from pydantic import BaseModel

class AnalysisResult(BaseModel):
    summary: str
    sentiment: str
    key_points: list[str]
    confidence: float

agent = Agent(
    name="Analyser",
    instructions="Analyse content",
    output_type=AnalysisResult
)

result = await Runner.run(agent, "Analyse this text")
analysis = result.final_output_as(AnalysisResult)
```

---

## Side-by-Side Comparison

### Basic Agent Creation

```python
# ============================================
# SWARM (Experimental)
# ============================================
from swarm import Swarm, Agent as SwarmAgent

client = Swarm()

swarm_agent = SwarmAgent(
    name="Assistant",
    instructions="You are a helpful assistant",
    model="gpt-4"
)

result = client.run(
    agent=swarm_agent,
    messages=[{"role": "user", "content": "Hello"}]
)

print(result.messages[-1]["content"])

# ============================================
# AGENTS SDK (Production-Ready)
# ============================================
from agents import Agent, Runner

agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant",
    model="gpt-4o"
)

result = await Runner.run(agent, "Hello")
print(result.final_output)
```

### Agent Handoffs

```python
# ============================================
# SWARM
# ============================================
from swarm import Swarm, Agent as SwarmAgent

def transfer_to_specialist():
    return specialist_agent

triage_agent = SwarmAgent(
    name="Triage",
    instructions="Route to specialist",
    functions=[transfer_to_specialist]
)

specialist_agent = SwarmAgent(
    name="Specialist",
    instructions="Handle specialized queries"
)

client = Swarm()
result = client.run(agent=triage_agent, messages=[{"role": "user", "content": "I need help"}])

# ============================================
# AGENTS SDK
# ============================================
from agents import Agent, Runner

specialist_agent = Agent(
    name="Specialist",
    handoff_description="Handles specialized queries",
    instructions="Handle specialized queries"
)

triage_agent = Agent(
    name="Triage",
    instructions="Route to appropriate specialist",
    handoffs=[specialist_agent]
)

result = await Runner.run(triage_agent, "I need help")
```

### Tools/Functions

```python
# ============================================
# SWARM
# ============================================
from swarm import Swarm, Agent as SwarmAgent

def get_weather(location: str) -> str:
    """Get weather for location."""
    return f"Weather in {location}: Sunny"

swarm_agent = SwarmAgent(
    name="Weather",
    functions=[get_weather]
)

client = Swarm()
result = client.run(agent=swarm_agent, messages=[{"role": "user", "content": "Weather in London?"}])

# ============================================
# AGENTS SDK
# ============================================
from agents import Agent, Runner, function_tool
from pydantic import BaseModel

class WeatherResult(BaseModel):
    location: str
    condition: str
    temperature: float

@function_tool
def get_weather(location: str) -> WeatherResult:
    """Get weather for location with validation."""
    return WeatherResult(
        location=location,
        condition="Sunny",
        temperature=72.5
    )

agent = Agent(
    name="Weather",
    tools=[get_weather]
)

result = await Runner.run(agent, "Weather in London?")
```

### Session/Memory Management

```python
# ============================================
# SWARM
# ============================================
from swarm import Swarm, Agent as SwarmAgent

client = Swarm()
messages = []  # Manual message tracking

# Turn 1
messages.append({"role": "user", "content": "My name is Alice"})
result = client.run(agent=agent, messages=messages)
messages.extend(result.messages)

# Turn 2
messages.append({"role": "user", "content": "What's my name?"})
result = client.run(agent=agent, messages=messages)

# ============================================
# AGENTS SDK
# ============================================
from agents import Agent, Runner, SQLiteSession

agent = Agent(name="Assistant")
session = SQLiteSession("user_123", "sessions.db")

# Turn 1
result1 = await Runner.run(agent, "My name is Alice", session=session)

# Turn 2 - automatic context
result2 = await Runner.run(agent, "What's my name?", session=session)
```

---

## Migration Checklist

### Pre-Migration

- [ ] **Audit Current Swarm Usage**: Document all agents, functions, and workflows
- [ ] **Review Dependencies**: Check for Swarm-specific dependencies
- [ ] **Backup Current Code**: Create version control snapshot
- [ ] **Test Coverage**: Ensure existing tests cover critical paths
- [ ] **Stakeholder Communication**: Inform team about migration plan

### During Migration

- [ ] **Install Agents SDK**: `pip install openai-agents`
- [ ] **Update Imports**: Change from `swarm` to `agents`
- [ ] **Convert Agents**: Update agent creation patterns
- [ ] **Migrate Functions to Tools**: Use `@function_tool` decorator
- [ ] **Implement Session Management**: Add SQLite/Redis sessions
- [ ] **Add Guardrails**: Implement input/output validation
- [ ] **Enable Tracing**: Add observability with `trace()`
- [ ] **Update Tests**: Modify tests for new API
- [ ] **Performance Testing**: Verify response times and costs

### Post-Migration

- [ ] **Monitor Production**: Watch for errors and performance issues
- [ ] **Collect Metrics**: Track token usage, latency, error rates
- [ ] **User Feedback**: Gather feedback on new implementation
- [ ] **Documentation Update**: Update internal documentation
- [ ] **Cleanup**: Remove Swarm dependencies
- [ ] **Team Training**: Educate team on new patterns

---

## Code Migration Examples

### Example 1: Simple Customer Service Agent

**Before (Swarm):**

```python
from swarm import Swarm, Agent as SwarmAgent

def process_refund(booking_id: str) -> str:
    return f"Refund processed for {booking_id}"

refund_agent = SwarmAgent(
    name="Refund Specialist",
    instructions="Process refund requests",
    functions=[process_refund]
)

triage_agent = SwarmAgent(
    name="Triage",
    instructions="Route customer requests",
    functions=[lambda: refund_agent]
)

client = Swarm()
result = client.run(
    agent=triage_agent,
    messages=[{"role": "user", "content": "I need a refund"}]
)
```

**After (Agents SDK):**

```python
from agents import Agent, Runner, function_tool, SQLiteSession
import asyncio

@function_tool
def process_refund(booking_id: str) -> dict:
    """Process refund request."""
    return {
        "status": "approved",
        "booking_id": booking_id,
        "amount": 99.99
    }

refund_agent = Agent(
    name="Refund Specialist",
    handoff_description="Handles refund requests",
    instructions="Process refund requests efficiently",
    tools=[process_refund]
)

triage_agent = Agent(
    name="Triage",
    instructions="Route customer requests to appropriate specialist",
    handoffs=[refund_agent]
)

async def handle_request(user_id: str, query: str):
    session = SQLiteSession(user_id, "support.db")
    result = await Runner.run(triage_agent, query, session=session)
    return result.final_output

asyncio.run(handle_request("user_123", "I need a refund"))
```

### Example 2: Multi-Agent Research System

**Before (Swarm):**

```python
from swarm import Swarm, Agent as SwarmAgent

def search_web(query: str) -> str:
    return f"Search results for {query}"

researcher = SwarmAgent(
    name="Researcher",
    instructions="Research topics",
    functions=[search_web]
)

summariser = SwarmAgent(
    name="Summariser",
    instructions="Summarise research"
)

client = Swarm()

# Research
research_result = client.run(
    agent=researcher,
    messages=[{"role": "user", "content": "Research AI trends"}]
)

# Summarise
summary_result = client.run(
    agent=summariser,
    messages=[{"role": "user", "content": f"Summarise: {research_result.messages[-1]['content']}"}]
)
```

**After (Agents SDK):**

```python
from agents import Agent, Runner, WebSearchTool
from pydantic import BaseModel
import asyncio

class ResearchSummary(BaseModel):
    topic: str
    key_findings: list[str]
    sources: list[str]
    summary: str

researcher = Agent(
    name="Researcher",
    instructions="Research topics thoroughly using web search",
    tools=[WebSearchTool()]
)

summariser = Agent(
    name="Summariser",
    instructions="Create structured summaries",
    output_type=ResearchSummary
)

async def research_workflow(topic: str):
    # Step 1: Research
    research_result = await Runner.run(researcher, f"Research: {topic}")

    # Step 2: Summarise with structured output
    summary_result = await Runner.run(
        summariser,
        f"Summarise this research: {research_result.final_output}"
    )

    return summary_result.final_output_as(ResearchSummary)

summary = asyncio.run(research_workflow("AI trends 2025"))
print(f"Topic: {summary.topic}")
print(f"Key Findings: {summary.key_findings}")
```

### Example 3: Tool Execution with Validation

**Before (Swarm):**

```python
from swarm import Swarm, Agent as SwarmAgent

def book_flight(origin: str, destination: str, date: str) -> str:
    # No validation
    return f"Booked flight from {origin} to {destination} on {date}"

agent = SwarmAgent(
    name="Booking Agent",
    functions=[book_flight]
)

client = Swarm()
result = client.run(
    agent=agent,
    messages=[{"role": "user", "content": "Book flight"}]
)
```

**After (Agents SDK):**

```python
from agents import Agent, Runner, function_tool
from pydantic import BaseModel, Field
import asyncio

class FlightBooking(BaseModel):
    origin: str = Field(..., min_length=3, max_length=3)
    destination: str = Field(..., min_length=3, max_length=3)
    date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    passengers: int = Field(ge=1, le=9)

class BookingResult(BaseModel):
    confirmation_number: str
    status: str

@function_tool
def book_flight(booking: FlightBooking) -> BookingResult:
    """Book flight with automatic validation."""
    return BookingResult(
        confirmation_number=f"CONF-{booking.origin}-{booking.destination}",
        status="confirmed"
    )

agent = Agent(
    name="Booking Agent",
    instructions="Book flights with validation",
    tools=[book_flight],
    output_type=BookingResult
)

async def main():
    result = await Runner.run(agent, "Book flight from LAX to JFK on 2025-03-15 for 2 passengers")
    booking = result.final_output_as(BookingResult)
    print(f"Status: {booking.status}, Confirmation: {booking.confirmation_number}")

asyncio.run(main())
```

---

## Breaking Changes and Solutions

### 1. Import Changes

**Breaking Change:**
```python
from swarm import Swarm, Agent
```

**Solution:**
```python
from agents import Agent, Runner
```

### 2. Client Instantiation

**Breaking Change:**
```python
client = Swarm()
result = client.run(agent=agent, messages=messages)
```

**Solution:**
```python
result = await Runner.run(agent, input_string, session=session)
```

### 3. Message Format

**Breaking Change:**
```python
messages = [{"role": "user", "content": "Hello"}]
```

**Solution:**
```python
# Simple string input
result = await Runner.run(agent, "Hello")

# Or with session for conversation history
session = SQLiteSession("user_id", "db.sqlite")
result = await Runner.run(agent, "Hello", session=session)
```

### 4. Function Definitions

**Breaking Change:**
```python
def my_function(param: str) -> str:
    return "result"

agent = SwarmAgent(functions=[my_function])
```

**Solution:**
```python
from agents import function_tool

@function_tool
def my_function(param: str) -> str:
    """Function description for tool."""
    return "result"

agent = Agent(tools=[my_function])
```

### 5. Agent Handoffs

**Breaking Change:**
```python
def transfer_to_specialist():
    return specialist_agent

agent = SwarmAgent(functions=[transfer_to_specialist])
```

**Solution:**
```python
specialist_agent = Agent(
    name="Specialist",
    handoff_description="Transfer to specialist"
)

agent = Agent(
    name="Triage",
    handoffs=[specialist_agent]
)
```

---

## Testing Your Migration

### Unit Testing Pattern

```python
import pytest
from agents import Agent, Runner
import asyncio

@pytest.mark.asyncio
async def test_migrated_agent():
    """Test migrated agent functionality."""
    agent = Agent(
        name="Test Agent",
        instructions="You are a test agent"
    )

    result = await Runner.run(agent, "What is 2+2?")

    assert result.final_output is not None
    assert "4" in result.final_output

@pytest.mark.asyncio
async def test_agent_with_tools():
    """Test migrated tools."""
    from agents import function_tool

    @function_tool
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    agent = Agent(
        name="Math Agent",
        tools=[add_numbers]
    )

    result = await Runner.run(agent, "What is 5 + 3?")
    assert result.final_output is not None
```

### Integration Testing

```python
@pytest.mark.asyncio
async def test_multi_agent_workflow():
    """Test complete workflow."""
    from agents import Agent, Runner, SQLiteSession

    specialist = Agent(name="Specialist", instructions="Handle requests")
    triage = Agent(name="Triage", handoffs=[specialist])

    session = SQLiteSession("test_user", ":memory:")

    result = await Runner.run(triage, "I need specialist help", session=session)

    assert result.agent.name == "Specialist"
    assert result.final_output is not None
```

---

## Rollback Strategy

If migration issues occur:

### 1. Version Control Rollback

```bash
# Revert to pre-migration state
git checkout migration-backup-branch

# Reinstall Swarm
pip uninstall openai-agents
pip install swarm
```

### 2. Gradual Rollout

```python
# Use feature flags for gradual migration
USE_AGENTS_SDK = os.getenv("USE_AGENTS_SDK", "false").lower() == "true"

if USE_AGENTS_SDK:
    from agents import Agent, Runner
    # New implementation
else:
    from swarm import Swarm, Agent
    # Old implementation
```

### 3. A/B Testing

```python
import random

def route_request(user_id: str):
    # Route 10% to new Agents SDK
    if hash(user_id) % 10 == 0:
        return handle_with_agents_sdk(user_id)
    else:
        return handle_with_swarm(user_id)
```

---

## Migration Timeline Recommendations

### Small Projects (< 5 agents)
- **Planning**: 1 day
- **Migration**: 2-3 days
- **Testing**: 1-2 days
- **Total**: ~1 week

### Medium Projects (5-20 agents)
- **Planning**: 2-3 days
- **Migration**: 1-2 weeks
- **Testing**: 3-5 days
- **Total**: 2-3 weeks

### Large Projects (20+ agents)
- **Planning**: 1 week
- **Migration**: 2-4 weeks
- **Testing**: 1-2 weeks
- **Total**: 4-7 weeks

---

## Support and Resources

### Documentation
- [Agents SDK Comprehensive Guide](openai_agents_sdk_comprehensive_guide.md)
- [Production Guide](openai_agents_sdk_production_guide.md)
- [Recipes Guide](openai_agents_sdk_recipes.md)

### Community
- GitHub Issues: https://github.com/openai/openai-agents-python
- OpenAI Forums: https://community.openai.com

### Official Resources
- Agents SDK Documentation: https://openai.github.io/openai-agents-python/
- API Reference: https://platform.openai.com/docs

---

## Conclusion

Migrating from Swarm to the OpenAI Agents SDK provides:

✅ **Production stability and active maintenance**
✅ **Enterprise features (guardrails, tracing, sessions)**
✅ **Provider flexibility (100+ LLMs)**
✅ **Better performance and reliability**
✅ **Comprehensive documentation and support**

The migration effort is justified by the significant improvements in stability, features, and long-term support. The Agents SDK represents OpenAI's commitment to production-ready agentic frameworks.

**Recommendation**: Migrate all Swarm implementations to Agents SDK for production use.

---

**Last Updated:** January 2025
**Migration Guide Version:** 1.0
**Agents SDK Version:** v0.2.9+
