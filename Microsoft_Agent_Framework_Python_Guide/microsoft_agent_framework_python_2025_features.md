# Microsoft Agent Framework - Python 2025 Features Guide
## October 2025 Release - Advanced Python-Specific Features

**Document Version:** 1.0
**Last Updated:** November 2025
**Target Audience:** Python Developers
**Preview Status:** October 2025 Public Preview
**Python Version:** 3.10+

---

## Table of Contents

1. [Agent2Agent (A2A) Protocol - Python](#agent2agent-a2a-protocol---python)
2. [Graph-Based Workflows - Python](#graph-based-workflows---python)
3. [Declarative Agents with Python](#declarative-agents-with-python)
4. [OpenTelemetry Integration - Python](#opentelemetry-integration---python)
5. [Content Safety - Python](#content-safety---python)
6. [Async Best Practices](#async-best-practices)

---

## Agent2Agent (A2A) Protocol - Python

### Quick Start with A2A

```python
import asyncio
from agent_framework import ChatAgent
from agent_framework.a2a import A2AProtocolAdapter, A2AClient, A2AMessage
from azure.identity.aio import DefaultAzureCredential

async def setup_a2a_agent():
    """Create an agent with A2A capability"""

    async with DefaultAzureCredential() as credential:
        # Create base agent
        agent = ChatAgent(
            instructions="You are a helpful sales data agent.",
            name="SalesAgent"
        )

        # Wrap with A2A adapter
        a2a_adapter = A2AProtocolAdapter(
            agent=agent,
            agent_id="agent_sales_001",
            endpoint="https://api.contoso.com/agents/sales",
            authentication={
                "method": "entra-id",
                "credential": credential
            }
        )

        # Register for incoming A2A messages
        @a2a_adapter.on_message
        async def handle_request(message: A2AMessage):
            """Handle incoming A2A requests"""
            query = message.payload.get('content')
            response = await agent.run(query)

            return {
                "type": "response",
                "status": "success",
                "content": response.text
            }

        await a2a_adapter.register()
        return a2a_adapter

# Call external agent via A2A
async def call_openai_agent():
    """Call an OpenAI SDK agent via A2A protocol"""

    client = A2AClient(
        agent_id="agent_sales_001",
        authentication={
            "method": "oauth2",
            "client_id": "your_client_id",
            "client_secret": "your_client_secret"
        }
    )

    message = A2AMessage(
        recipient_agent_id="agent_openai_analytics",
        recipient_endpoint="https://api.partner.com/agents/analytics",
        recipient_framework="openai-sdk",
        payload={
            "type": "request",
            "action": "query",
            "content": "Analyze Q3 sales trends"
        }
    )

    response = await client.send_message(message)
    print(f"Response: {response.payload['content']}")

# Run
asyncio.run(call_openai_agent())
```

### Type-Safe A2A with Pydantic

```python
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from agent_framework.a2a import A2AMessage, A2AClient

class SalesQuery(BaseModel):
    """Type-safe sales query request"""
    query_type: str = Field(..., description="Type of sales query")
    date_range: str = Field(..., description="Date range for query")
    region: Optional[str] = Field(None, description="Geographic region")
    currency: str = Field(default="USD", description="Currency code")

class SalesResponse(BaseModel):
    """Type-safe sales response"""
    total_sales: float
    period: str
    region: str
    currency: str
    breakdown: Dict[str, float]

async def type_safe_a2a_call(query: SalesQuery) -> SalesResponse:
    """Type-safe A2A communication"""

    client = A2AClient(agent_id="sales_agent")

    message = A2AMessage(
        recipient_agent_id="analytics_agent",
        recipient_endpoint="https://api.partner.com/agents/analytics",
        payload={
            "type": "request",
            "action": "query_sales",
            "content": query.model_dump_json()
        }
    )

    response = await client.send_message(message)

    # Parse response with Pydantic
    sales_data = SalesResponse.model_validate_json(
        response.payload['content']
    )

    return sales_data

# Usage with type safety
async def main():
    query = SalesQuery(
        query_type="regional_sales",
        date_range="2025-Q3",
        region="EMEA",
        currency="EUR"
    )

    result = await type_safe_a2a_call(query)
    print(f"Total Sales: {result.total_sales} {result.currency}")
    print(f"Breakdown: {result.breakdown}")

asyncio.run(main())
```

---

## Graph-Based Workflows - Python

### Building Complex Workflows

```python
from agent_framework.graphs import AgentGraph, GraphNode
from agent_framework import ChatAgent
from typing import Dict, Any

async def create_content_pipeline():
    """Multi-agent content creation pipeline"""

    # Define specialized agents
    research_agent = ChatAgent(
        instructions="You research topics thoroughly using web sources.",
        name="ResearchAgent"
    )

    writer_agent = ChatAgent(
        instructions="You write engaging content based on research.",
        name="WriterAgent"
    )

    editor_agent = ChatAgent(
        instructions="You edit and improve content quality.",
        name="EditorAgent"
    )

    seo_agent = ChatAgent(
        instructions="You optimize content for search engines.",
        name="SEOAgent"
    )

    # Create graph
    graph = AgentGraph(name="ContentPipeline")

    # Add nodes
    graph.add_node("research", agent=research_agent)
    graph.add_node("write", agent=writer_agent)
    graph.add_node("edit", agent=editor_agent)
    graph.add_node("seo", agent=seo_agent)

    # Linear workflow
    graph.add_edge("research", "write")
    graph.add_edge("write", "edit")
    graph.add_edge("edit", "seo")

    graph.set_entry_point("research")

    return graph

# Execute with streaming
async def run_content_pipeline_with_streaming(topic: str):
    """Execute pipeline with real-time streaming"""

    graph = await create_content_pipeline()

    print(f"Creating content for: {topic}\n")

    # Stream execution events
    async for event in graph.stream(
        initial_state={"topic": topic, "target_length": 1500},
        stream_mode="updates"
    ):
        node_id = event.get("node")
        event_type = event.get("type")

        if event_type == "node_start":
            print(f"▶ Starting: {node_id}")
        elif event_type == "node_output":
            output = event.get("data", {}).get("output", "")
            print(f"  Output preview: {output[:100]}...")
        elif event_type == "node_complete":
            print(f"✓ Completed: {node_id}\n")

    # Get final result
    final_result = await graph.get_final_state()
    return final_result

# Conditional routing
async def create_conditional_workflow():
    """Workflow with conditional branching"""

    classifier_agent = ChatAgent(
        instructions="Classify customer queries by urgency and type."
    )

    simple_agent = ChatAgent(
        instructions="Handle simple queries quickly."
    )

    complex_agent = ChatAgent(
        instructions="Handle complex queries with detailed analysis."
    )

    urgent_agent = ChatAgent(
        instructions="Handle urgent queries with priority."
    )

    graph = AgentGraph(name="CustomerSupport")

    graph.add_node("classifier", agent=classifier_agent)
    graph.add_node("simple", agent=simple_agent)
    graph.add_node("complex", agent=complex_agent)
    graph.add_node("urgent", agent=urgent_agent)

    # Conditional routing
    def route_query(state: Dict[str, Any]) -> str:
        """Route based on classification"""
        urgency = state.get("urgency", "normal")
        complexity = state.get("complexity", "simple")

        if urgency == "high":
            return "urgent"
        elif complexity == "high":
            return "complex"
        else:
            return "simple"

    graph.add_conditional_edge(
        source="classifier",
        path_function=route_query,
        path_map={
            "simple": "simple",
            "complex": "complex",
            "urgent": "urgent"
        }
    )

    return graph

# Parallel execution
async def create_parallel_analysis():
    """Run multiple agents in parallel"""

    sentiment_agent = ChatAgent(instructions="Analyze sentiment")
    keyword_agent = ChatAgent(instructions="Extract keywords")
    summary_agent = ChatAgent(instructions="Summarize text")

    graph = AgentGraph(name="ParallelAnalysis")

    graph.add_node("sentiment", agent=sentiment_agent)
    graph.add_node("keywords", agent=keyword_agent)
    graph.add_node("summary", agent=summary_agent)
    graph.add_node("aggregator", agent=ChatAgent(instructions="Combine results"))

    # Parallel fan-out
    for node in ["sentiment", "keywords", "summary"]:
        graph.add_edge("START", node)
        graph.add_edge(node, "aggregator")

    return graph
```

### Checkpointing and Recovery

```python
from agent_framework.graphs import AgentGraph, CheckpointConfig
import os

async def create_checkpointed_workflow():
    """Workflow with automatic checkpointing"""

    graph = AgentGraph(name="DataProcessing")

    # Configure checkpointing
    checkpoint_config = CheckpointConfig(
        backend="azure-storage",
        connection_string=os.getenv("AZURE_STORAGE_CONNECTION"),
        container="agent-checkpoints",
        checkpoint_interval="node_completion",
        retention_days=7
    )

    graph.configure_checkpointing(checkpoint_config)

    # Add nodes...

    return graph

async def resume_workflow_from_checkpoint(checkpoint_id: str):
    """Resume a failed workflow"""

    graph = await create_checkpointed_workflow()

    # Load checkpoint
    checkpoint = await graph.load_checkpoint(checkpoint_id)

    print(f"Resuming from: {checkpoint.current_node}")
    print(f"State: {checkpoint.state}")

    # Resume execution
    result = await graph.resume(checkpoint)

    return result
```

### Human-in-the-Loop

```python
from agent_framework.graphs import AgentGraph, HITLConfig

async def create_approval_workflow():
    """Workflow requiring human approval"""

    graph = AgentGraph(name="ContentApproval")

    # Add agents
    draft_agent = ChatAgent(instructions="Draft content")
    review_agent = ChatAgent(instructions="Review content")
    publish_agent = ChatAgent(instructions="Publish content")

    graph.add_node("draft", agent=draft_agent)
    graph.add_node("review", agent=review_agent)
    graph.add_node("publish", agent=publish_agent)

    # Configure HITL
    hitl_config = HITLConfig(
        enabled=True,
        approval_endpoint="https://approval.contoso.com/api/requests",
        notification_channels=["email", "teams"],
        timeout_seconds=3600
    )

    graph.add_hitl_checkpoint(
        after_node="review",
        config=hitl_config,
        approval_prompt="Review and approve this content"
    )

    graph.add_edge("draft", "review")
    graph.add_edge("review", "publish")

    return graph

# Interactive HITL
async def interactive_approval_workflow():
    """Interactive human approval"""

    graph = await create_approval_workflow()

    async for event in graph.stream(initial_state={"topic": "Product Launch"}):
        if event.get("type") == "hitl_required":
            content = event.get("content")
            request_id = event.get("request_id")

            print("\n" + "="*60)
            print("APPROVAL REQUIRED")
            print("="*60)
            print(f"Content:\n{content}")
            print("="*60)

            decision = input("Approve? (yes/no): ").lower()

            if decision == "yes":
                await graph.approve_hitl(request_id)
            else:
                reason = input("Rejection reason: ")
                await graph.reject_hitl(request_id, reason=reason)
```

---

## Declarative Agents with Python

### YAML Configuration

```yaml
# agent-config.yaml
agent:
  name: CustomerSupportAgent
  version: "1.0.0"

  model:
    provider: azure-openai
    deployment: gpt-4o-mini
    temperature: 0.7
    max_tokens: 1000

  instructions: |
    You are a helpful customer support agent.
    Provide clear, concise answers.

  tools:
    - name: search_kb
      type: function
      function:
        module: tools.knowledge_base
        name: search_knowledge_base

    - name: create_ticket
      type: function
      function:
        module: tools.ticketing
        name: create_support_ticket

  memory:
    type: persistent
    backend: cosmos-db
    connection_string: ${COSMOS_CONNECTION_STRING}

  authentication:
    method: entra-id
    tenant_id: ${AZURE_TENANT_ID}
```

### Loading Declarative Configuration

```python
import yaml
from agent_framework.config import load_agent_from_yaml, AgentConfig
from azure.identity.aio import DefaultAzureCredential
from typing import Dict, Any

async def load_agent_from_config(config_path: str):
    """Load agent from YAML configuration"""

    # Load with automatic credential management
    agent = await load_agent_from_yaml(
        config_path=config_path,
        credential=DefaultAzureCredential()
    )

    return agent

# Usage
async def main():
    agent = await load_agent_from_config("agent-config.yaml")

    # Agent is fully configured
    response = await agent.run("Help me with my order")
    print(response.text)

# Template with environment variables
async def load_with_environment():
    """Load configuration with environment substitution"""

    import os

    # Set environment variables
    os.environ["AGENT_NAME"] = "SalesAgent"
    os.environ["MODEL_DEPLOYMENT"] = "gpt-4o"
    os.environ["TEMPERATURE"] = "0.8"

    agent = await load_agent_from_yaml(
        config_path="agent-template.yaml",
        substitutions={
            "MAX_TOKENS": "1500",
            "REGION": "eastus"
        }
    )

    return agent
```

### Pydantic Models for Configuration

```python
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any
from enum import Enum

class ModelProvider(str, Enum):
    AZURE_OPENAI = "azure-openai"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

class ModelConfig(BaseModel):
    provider: ModelProvider
    deployment: str
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, gt=0)

class ToolConfig(BaseModel):
    name: str
    type: str = "function"
    module: str
    function_name: str

class MemoryConfig(BaseModel):
    type: str
    backend: str
    connection_string: Optional[str] = None

    @field_validator('connection_string')
    @classmethod
    def validate_connection_string(cls, v, info):
        if info.data.get('type') == 'persistent' and not v:
            raise ValueError('connection_string required for persistent memory')
        return v

class AgentConfiguration(BaseModel):
    """Type-safe agent configuration"""
    name: str
    version: str = "1.0.0"
    model: ModelConfig
    instructions: str
    tools: List[ToolConfig] = []
    memory: Optional[MemoryConfig] = None

    def validate_config(self) -> bool:
        """Validate configuration"""
        # Custom validation logic
        return True

# Load and validate
async def load_validated_config(config_dict: Dict[str, Any]):
    """Load configuration with validation"""

    # Parse and validate with Pydantic
    config = AgentConfiguration.model_validate(config_dict)

    # Create agent from validated config
    agent = await create_agent_from_config(config)

    return agent

async def create_agent_from_config(config: AgentConfiguration):
    """Create agent from Pydantic model"""

    from agent_framework import ChatAgent

    agent = ChatAgent(
        name=config.name,
        instructions=config.instructions,
        model=config.model.deployment,
        temperature=config.model.temperature,
        max_tokens=config.model.max_tokens
    )

    return agent
```

---

## OpenTelemetry Integration - Python

### Complete Observability Setup

```python
from opentelemetry import trace, metrics
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from azure.monitor.opentelemetry import configure_azure_monitor
from agent_framework import ChatAgent
from agent_framework.telemetry import configure_telemetry
import os

def setup_opentelemetry():
    """Configure comprehensive OpenTelemetry"""

    # Azure Monitor integration
    configure_azure_monitor(
        connection_string=os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"),
        enable_live_metrics=True
    )

    # Additional OTLP exporter
    tracer_provider = TracerProvider()
    trace.set_tracer_provider(tracer_provider)

    otlp_exporter = OTLPSpanExporter(
        endpoint="https://otel-collector.contoso.com:4317",
        headers={"api-key": os.getenv("OTEL_API_KEY")}
    )

    tracer_provider.add_span_processor(
        BatchSpanProcessor(otlp_exporter)
    )

    # Configure agent framework telemetry
    configure_telemetry(
        service_name="customer-support-agents",
        service_version="1.0.0",
        environment="production",
        enable_auto_instrumentation=True
    )

async def traced_agent_execution():
    """Execute agent with full tracing"""

    setup_opentelemetry()

    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("customer_support_workflow") as span:
        span.set_attribute("customer.id", "cust_12345")
        span.set_attribute("query.type", "order_status")

        agent = ChatAgent(
            instructions="You help customers with orders."
        )

        # Automatically traced
        response = await agent.run("Where is my order #12345?")

        span.set_attribute("response.length", len(response.text))
        span.add_event("response_generated")

        return response
```

### Custom Metrics

```python
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
import time

# Setup metrics
meter_provider = MeterProvider()
metrics.set_meter_provider(meter_provider)
meter = metrics.get_meter(__name__)

# Create metrics
agent_requests_counter = meter.create_counter(
    name="agent.requests.total",
    description="Total agent requests",
    unit="1"
)

agent_latency_histogram = meter.create_histogram(
    name="agent.latency",
    description="Agent request latency",
    unit="ms"
)

agent_active_requests = meter.create_up_down_counter(
    name="agent.requests.active",
    description="Currently active requests",
    unit="1"
)

async def monitored_agent_execution(query: str):
    """Execute agent with custom metrics"""

    # Increment active requests
    agent_active_requests.add(1, {"agent": "support"})

    start_time = time.time()

    try:
        agent = ChatAgent(instructions="You help customers.")
        response = await agent.run(query)

        # Record success
        agent_requests_counter.add(1, {
            "agent": "support",
            "status": "success"
        })

        latency = (time.time() - start_time) * 1000
        agent_latency_histogram.record(latency, {"agent": "support"})

        return response

    except Exception as e:
        # Record failure
        agent_requests_counter.add(1, {
            "agent": "support",
            "status": "error",
            "error_type": type(e).__name__
        })
        raise

    finally:
        # Decrement active requests
        agent_active_requests.add(-1, {"agent": "support"})
```

---

## Content Safety - Python

### Azure AI Content Safety Integration

```python
from agent_framework import ChatAgent
from agent_framework.safety import ContentSafetyConfig
from azure.ai.contentsafety.aio import ContentSafetyClient
from azure.identity.aio import DefaultAzureCredential

async def create_safe_agent():
    """Create agent with Content Safety"""

    # Configure Content Safety
    content_safety = ContentSafetyConfig(
        endpoint=os.getenv("CONTENT_SAFETY_ENDPOINT"),
        credential=DefaultAzureCredential(),
        categories={
            "hate": {"severity_threshold": 2, "block": True},
            "sexual": {"severity_threshold": 2, "block": True},
            "violence": {"severity_threshold": 4, "block": True},
            "self_harm": {"severity_threshold": 2, "block": True}
        },
        check_input=True,
        check_output=True,
        action_on_violation="block_and_log"
    )

    agent = ChatAgent(
        instructions="You are a customer support agent.",
        content_safety=content_safety
    )

    return agent

# Custom violation handler
from agent_framework.safety import SafetyViolationHandler, ContentSafetyViolation

class CustomSafetyHandler(SafetyViolationHandler):
    """Custom content safety violation handler"""

    async def handle_input_violation(self, violation: ContentSafetyViolation):
        """Handle user input violations"""
        # Log to security monitoring
        await self.log_security_event({
            "type": "input_violation",
            "category": violation.category,
            "severity": violation.severity,
            "user_id": violation.context.get("user_id"),
            "timestamp": violation.timestamp
        })

        return {
            "blocked": True,
            "message": "Your message violates content policy.",
            "reference_id": violation.id
        }

    async def handle_output_violation(self, violation: ContentSafetyViolation):
        """Handle agent output violations"""
        # This should rarely happen with proper agent instructions
        await self.log_security_event({
            "type": "output_violation",
            "category": violation.category,
            "agent_id": violation.agent_id
        })

        return {
            "blocked": True,
            "message": "I apologize, but I cannot provide that information.",
            "fallback": True
        }
```

### PII Detection

```python
from agent_framework.safety import PIIDetectionConfig

async def create_pii_protected_agent():
    """Agent with PII detection and redaction"""

    pii_config = PIIDetectionConfig(
        enabled=True,
        detect_categories=[
            "email",
            "phone_number",
            "ssn",
            "credit_card",
            "person_name",
            "address"
        ],
        action="redact",
        redaction_pattern="[REDACTED_{category}]"
    )

    agent = ChatAgent(
        instructions="You handle customer data.",
        pii_detection=pii_config
    )

    return agent

# Usage
async def test_pii_redaction():
    agent = await create_pii_protected_agent()

    # Input with PII
    response = await agent.run(
        "My email is john@example.com and SSN is 123-45-6789"
    )

    # PII automatically redacted in logs and output
    print(response.text)
    # Output: "My email is [REDACTED_email] and SSN is [REDACTED_ssn]"
```

---

## Async Best Practices

### Proper Async Context Management

```python
from contextlib import asynccontextmanager
from agent_framework import ChatAgent
from azure.identity.aio import DefaultAzureCredential

@asynccontextmanager
async def managed_agent(instructions: str):
    """Properly managed async agent lifecycle"""

    credential = DefaultAzureCredential()
    agent = None

    try:
        # Setup
        agent = ChatAgent(
            instructions=instructions,
            credential=credential
        )
        yield agent
    finally:
        # Cleanup
        if agent:
            await agent.close()
        await credential.close()

# Usage
async def main():
    async with managed_agent("You are helpful") as agent:
        response = await agent.run("Hello!")
        print(response.text)
    # Automatic cleanup
```

### Concurrent Agent Execution

```python
import asyncio
from typing import List

async def execute_agents_concurrently(queries: List[str]):
    """Execute multiple agent calls concurrently"""

    async with managed_agent("You are helpful") as agent:
        # Create tasks
        tasks = [agent.run(query) for query in queries]

        # Execute concurrently
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle results
        for query, response in zip(queries, responses):
            if isinstance(response, Exception):
                print(f"Error for '{query}': {response}")
            else:
                print(f"'{query}' -> {response.text}")
```

### Error Handling

```python
from agent_framework.exceptions import (
    AgentExecutionError,
    ContentSafetyViolation,
    RateLimitError
)

async def robust_agent_execution(query: str):
    """Robust error handling"""

    try:
        async with managed_agent("You are helpful") as agent:
            response = await agent.run(query)
            return response.text

    except ContentSafetyViolation as e:
        print(f"Content policy violation: {e.category}")
        return "I cannot process that request."

    except RateLimitError as e:
        print(f"Rate limit exceeded. Retry after {e.retry_after} seconds")
        await asyncio.sleep(e.retry_after)
        return await robust_agent_execution(query)

    except AgentExecutionError as e:
        print(f"Agent execution failed: {e}")
        return "An error occurred. Please try again."

    except Exception as e:
        print(f"Unexpected error: {e}")
        return "An unexpected error occurred."
```

---

## Production Patterns

### Agent Pool Pattern

```python
import asyncio
from typing import List
from agent_framework import ChatAgent

class AgentPool:
    """Pool of agents for high-throughput scenarios"""

    def __init__(self, pool_size: int = 5):
        self.pool_size = pool_size
        self.agents: List[ChatAgent] = []
        self.semaphore = asyncio.Semaphore(pool_size)

    async def __aenter__(self):
        """Initialize agent pool"""
        for _ in range(self.pool_size):
            agent = ChatAgent(instructions="You are helpful.")
            self.agents.append(agent)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup pool"""
        for agent in self.agents:
            await agent.close()

    async def execute(self, query: str):
        """Execute query using available agent from pool"""
        async with self.semaphore:
            # Get an agent (simple round-robin)
            agent = self.agents[hash(query) % self.pool_size]
            response = await agent.run(query)
            return response.text

# Usage
async def main():
    queries = [f"Query {i}" for i in range(100)]

    async with AgentPool(pool_size=10) as pool:
        tasks = [pool.execute(query) for query in queries]
        results = await asyncio.gather(*tasks)

        print(f"Processed {len(results)} queries")
```

---

**Last Updated:** November 2025
**Document Version:** 1.0
**Preview Status:** October 2025 Public Preview
