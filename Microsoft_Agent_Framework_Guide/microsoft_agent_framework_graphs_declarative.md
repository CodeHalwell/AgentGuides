# Microsoft Agent Framework - Graph-Based Workflows & Declarative Definitions
## October 2025 Release - Advanced Agent Orchestration

**Document Version:** 1.0
**Last Updated:** November 2025
**Target Audience:** Architects and Senior Developers
**Preview Status:** October 2025 Public Preview

---

## Table of Contents

1. [Graph-Based Workflows](#graph-based-workflows)
2. [Declarative Agent Definitions](#declarative-agent-definitions)
3. [Streaming in Graphs](#streaming-in-graphs)
4. [Checkpointing and State Management](#checkpointing-and-state-management)
5. [Human-in-the-Loop Integration](#human-in-the-loop-integration)
6. [Time-Travel Debugging](#time-travel-debugging)
7. [Production Patterns](#production-patterns)

---

## Graph-Based Workflows

Graph-based workflows enable complex agent orchestration through directed graphs where nodes represent agents or actions, and edges represent data flow and control flow between them.

### Core Concepts

#### **Graph Structure**

```
┌─────────────┐
│   START     │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ Research Agent  │
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌───────┐  ┌──────────┐
│  LLM  │  │ Web Tool │
└───┬───┘  └────┬─────┘
    │           │
    └─────┬─────┘
          │
          ▼
┌──────────────────┐
│ Analysis Agent   │
└─────────┬────────┘
          │
    ┌─────┴──────┐
    │            │
    ▼            ▼
┌────────┐  ┌─────────┐
│Success │  │ Failure │
└────────┘  └─────────┘
```

### Creating Graphs - Python

#### **Basic Graph Definition**

```python
from agent_framework.graphs import AgentGraph, GraphNode, GraphEdge
from agent_framework import ChatAgent
from azure.identity.aio import DefaultAzureCredential

async def create_research_workflow():
    """Create a graph-based research workflow"""

    async with DefaultAzureCredential() as credential:
        # Define agents
        research_agent = ChatAgent(
            instructions="You research topics and gather information.",
            name="ResearchAgent"
        )

        analysis_agent = ChatAgent(
            instructions="You analyze research findings and extract insights.",
            name="AnalysisAgent"
        )

        summary_agent = ChatAgent(
            instructions="You create concise summaries of analysis.",
            name="SummaryAgent"
        )

        # Create graph
        graph = AgentGraph(name="ResearchWorkflow")

        # Add nodes
        research_node = graph.add_node(
            "research",
            agent=research_agent,
            description="Gather research data"
        )

        analysis_node = graph.add_node(
            "analysis",
            agent=analysis_agent,
            description="Analyze research findings"
        )

        summary_node = graph.add_node(
            "summary",
            agent=summary_agent,
            description="Create summary"
        )

        # Define edges (data flow)
        graph.add_edge(
            source="research",
            target="analysis",
            condition=lambda state: state.get("research_complete", False)
        )

        graph.add_edge(
            source="analysis",
            target="summary",
            condition=lambda state: state.get("confidence_score", 0) > 0.7
        )

        # Set entry point
        graph.set_entry_point("research")

        return graph

# Execute graph
async def main():
    graph = await create_research_workflow()

    # Run workflow
    result = await graph.run(
        initial_state={
            "topic": "AI Agent Frameworks",
            "depth": "comprehensive"
        }
    )

    print(f"Research: {result['research']}")
    print(f"Analysis: {result['analysis']}")
    print(f"Summary: {result['summary']}")
```

#### **Advanced Graph with Branching**

```python
from agent_framework.graphs import AgentGraph, ConditionalEdge
from typing import Dict, Any

async def create_customer_support_graph():
    """Complex customer support workflow with branching"""

    # Create agents
    classifier_agent = ChatAgent(
        instructions="Classify customer queries by type and urgency.",
        name="ClassifierAgent"
    )

    technical_agent = ChatAgent(
        instructions="Handle technical support queries.",
        name="TechnicalAgent"
    )

    billing_agent = ChatAgent(
        instructions="Handle billing and payment queries.",
        name="BillingAgent"
    )

    escalation_agent = ChatAgent(
        instructions="Handle escalated complex cases.",
        name="EscalationAgent"
    )

    # Create graph
    graph = AgentGraph(name="CustomerSupportWorkflow")

    # Add nodes
    graph.add_node("classifier", agent=classifier_agent)
    graph.add_node("technical", agent=technical_agent)
    graph.add_node("billing", agent=billing_agent)
    graph.add_node("escalation", agent=escalation_agent)

    # Conditional routing function
    def route_query(state: Dict[str, Any]) -> str:
        """Route to appropriate agent based on classification"""
        query_type = state.get("query_type", "unknown")
        urgency = state.get("urgency", "normal")

        if urgency == "critical":
            return "escalation"
        elif query_type == "technical":
            return "technical"
        elif query_type == "billing":
            return "billing"
        else:
            return "escalation"

    # Add conditional edges
    graph.add_conditional_edge(
        source="classifier",
        path_function=route_query,
        path_map={
            "technical": "technical",
            "billing": "billing",
            "escalation": "escalation"
        }
    )

    # All paths converge to END
    for node in ["technical", "billing", "escalation"]:
        graph.add_edge(source=node, target="END")

    graph.set_entry_point("classifier")

    return graph

# Usage
async def handle_customer_query(query: str):
    graph = await create_customer_support_graph()

    result = await graph.run(
        initial_state={
            "customer_query": query,
            "customer_id": "cust_12345"
        }
    )

    return result
```

### Creating Graphs - .NET

#### **Basic Graph Definition (.NET)**

```csharp
using Microsoft.Agents.AI.Graphs;
using Microsoft.Agents.AI;
using System;
using System.Threading.Tasks;
using System.Collections.Generic;

public class ResearchWorkflow
{
    public static async Task<AgentGraph> CreateResearchGraphAsync()
    {
        // Define agents
        var researchAgent = new ChatAgent(
            instructions: "You research topics and gather information.",
            name: "ResearchAgent"
        );

        var analysisAgent = new ChatAgent(
            instructions: "You analyze research findings and extract insights.",
            name: "AnalysisAgent"
        );

        var summaryAgent = new ChatAgent(
            instructions: "You create concise summaries.",
            name: "SummaryAgent"
        );

        // Create graph
        var graph = new AgentGraph("ResearchWorkflow");

        // Add nodes
        graph.AddNode(
            id: "research",
            agent: researchAgent,
            description: "Gather research data"
        );

        graph.AddNode(
            id: "analysis",
            agent: analysisAgent,
            description: "Analyze findings"
        );

        graph.AddNode(
            id: "summary",
            agent: summaryAgent,
            description: "Create summary"
        );

        // Define edges
        graph.AddEdge(
            source: "research",
            target: "analysis",
            condition: (state) => (bool)state.GetValueOrDefault("research_complete", false)
        );

        graph.AddEdge(
            source: "analysis",
            target: "summary",
            condition: (state) => (double)state.GetValueOrDefault("confidence_score", 0.0) > 0.7
        );

        // Set entry point
        graph.SetEntryPoint("research");

        return graph;
    }

    public static async Task Main()
    {
        var graph = await CreateResearchGraphAsync();

        // Execute workflow
        var result = await graph.RunAsync(new Dictionary<string, object>
        {
            ["topic"] = "AI Agent Frameworks",
            ["depth"] = "comprehensive"
        });

        Console.WriteLine($"Research: {result["research"]}");
        Console.WriteLine($"Analysis: {result["analysis"]}");
        Console.WriteLine($"Summary: {result["summary"]}");
    }
}
```

#### **Parallel Execution in Graphs**

```csharp
public class ParallelWorkflow
{
    public static async Task<AgentGraph> CreateParallelAnalysisGraphAsync()
    {
        // Create specialized analysis agents
        var sentimentAgent = new ChatAgent(
            instructions: "Analyze sentiment in text.",
            name: "SentimentAgent"
        );

        var keywordAgent = new ChatAgent(
            instructions: "Extract key terms and topics.",
            name: "KeywordAgent"
        );

        var summaryAgent = new ChatAgent(
            instructions: "Summarize text content.",
            name: "SummaryAgent"
        );

        var aggregatorAgent = new ChatAgent(
            instructions: "Aggregate all analysis results.",
            name: "AggregatorAgent"
        );

        var graph = new AgentGraph("ParallelAnalysis");

        // Add parallel analysis nodes
        graph.AddNode("sentiment", sentimentAgent);
        graph.AddNode("keywords", keywordAgent);
        graph.AddNode("summary", summaryAgent);
        graph.AddNode("aggregator", aggregatorAgent);

        // Fan-out: All analysis nodes run in parallel from START
        graph.AddParallelEdges(
            source: "START",
            targets: new[] { "sentiment", "keywords", "summary" }
        );

        // Fan-in: All results converge to aggregator
        graph.AddParallelEdges(
            sources: new[] { "sentiment", "keywords", "summary" },
            target: "aggregator"
        );

        graph.SetEntryPoint("START");

        return graph;
    }
}
```

---

## Declarative Agent Definitions

Declarative agent definitions enable configuration-driven agent creation using YAML or JSON, promoting reusability, version control, and separation of concerns.

### YAML Configuration Format

#### **Simple Agent Definition**

```yaml
# simple-agent.yaml
agent:
  name: CustomerSupportAgent
  version: "1.0.0"
  framework: microsoft-agent-framework

  model:
    provider: azure-openai
    deployment: gpt-4o-mini
    temperature: 0.7
    max_tokens: 1000

  instructions: |
    You are a helpful customer support agent.
    Provide clear, concise answers to customer queries.
    Always be polite and professional.

  tools:
    - name: search_knowledge_base
      description: Search the knowledge base for articles
      type: function
      function:
        name: search_kb
        module: tools.knowledge_base

    - name: create_ticket
      description: Create a support ticket
      type: function
      function:
        name: create_support_ticket
        module: tools.ticketing

  memory:
    type: persistent
    backend: cosmos-db
    connection_string: ${COSMOS_CONNECTION_STRING}
    container: agent-memory

  authentication:
    method: entra-id
    tenant_id: ${AZURE_TENANT_ID}

  observability:
    enabled: true
    application_insights: ${APPINSIGHTS_CONNECTION_STRING}
    log_level: INFO
```

#### **Multi-Agent Workflow Definition**

```yaml
# multi-agent-workflow.yaml
workflow:
  name: ContentCreationPipeline
  version: "2.0.0"
  description: End-to-end content creation workflow

  agents:
    research_agent:
      name: ResearchAgent
      model:
        provider: azure-openai
        deployment: gpt-4o
      instructions: |
        Research topics thoroughly and gather credible sources.
        Provide comprehensive background information.
      tools:
        - web_search
        - academic_search

    writer_agent:
      name: WriterAgent
      model:
        provider: azure-openai
        deployment: gpt-4o
        temperature: 0.8
      instructions: |
        Write engaging, well-structured content based on research.
        Maintain consistent tone and style.

    editor_agent:
      name: EditorAgent
      model:
        provider: azure-openai
        deployment: gpt-4o-mini
      instructions: |
        Review and edit content for grammar, clarity, and style.
        Ensure factual accuracy.

    seo_agent:
      name: SEOAgent
      model:
        provider: azure-openai
        deployment: gpt-4o-mini
      instructions: |
        Optimize content for SEO.
        Add meta descriptions, keywords, and tags.

  graph:
    nodes:
      - id: research
        agent: research_agent
        description: Gather research

      - id: write
        agent: writer_agent
        description: Write content

      - id: edit
        agent: editor_agent
        description: Edit and review

      - id: seo
        agent: seo_agent
        description: SEO optimization

    edges:
      - source: research
        target: write
        condition:
          field: research_quality
          operator: greater_than
          value: 0.8

      - source: write
        target: edit

      - source: edit
        target: seo
        condition:
          field: edit_approved
          operator: equals
          value: true

    entry_point: research

  checkpointing:
    enabled: true
    backend: azure-storage
    checkpoint_interval: node_completion

  human_in_loop:
    enabled: true
    checkpoints:
      - node: edit
        approval_required: true
        timeout_seconds: 3600
```

### JSON Configuration Format

#### **Agent Configuration (JSON)**

```json
{
  "agent": {
    "name": "DataAnalysisAgent",
    "version": "1.0.0",
    "framework": "microsoft-agent-framework",
    "model": {
      "provider": "azure-openai",
      "deployment": "gpt-4o",
      "temperature": 0.3,
      "max_tokens": 2000
    },
    "instructions": "You are a data analysis expert. Analyze datasets and provide actionable insights.",
    "tools": [
      {
        "name": "query_database",
        "description": "Query SQL database",
        "type": "function",
        "function": {
          "name": "execute_sql_query",
          "module": "tools.database"
        }
      },
      {
        "name": "create_visualization",
        "description": "Create data visualizations",
        "type": "function",
        "function": {
          "name": "generate_chart",
          "module": "tools.visualization"
        }
      }
    ],
    "memory": {
      "type": "vector",
      "backend": "azure-ai-search",
      "index_name": "agent-knowledge",
      "embedding_model": "text-embedding-3-large"
    },
    "rate_limiting": {
      "enabled": true,
      "requests_per_minute": 60,
      "burst_size": 10
    },
    "security": {
      "content_safety": {
        "enabled": true,
        "endpoint": "${CONTENT_SAFETY_ENDPOINT}",
        "block_harmful_content": true
      },
      "pii_detection": {
        "enabled": true,
        "redact": true
      }
    }
  }
}
```

### Loading Declarative Configurations

#### **Python - Load from YAML**

```python
import yaml
from agent_framework.config import AgentConfig, load_agent_from_yaml
from azure.identity.aio import DefaultAzureCredential

async def load_declarative_agent():
    """Load agent from YAML configuration"""

    # Load configuration
    with open("agent-config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Create agent from config
    agent = await load_agent_from_yaml(
        config_path="agent-config.yaml",
        credential=DefaultAzureCredential()
    )

    return agent

# Load and run
async def main():
    agent = await load_declarative_agent()

    # Agent is ready to use with all configured tools, memory, etc.
    response = await agent.run("Analyze customer feedback data")
    print(response.text)
```

#### **Python - Load Multi-Agent Workflow**

```python
from agent_framework.config import load_workflow_from_yaml
from agent_framework.graphs import AgentGraph

async def load_workflow():
    """Load complete workflow from YAML"""

    # Load workflow configuration
    workflow = await load_workflow_from_yaml(
        config_path="multi-agent-workflow.yaml",
        credential=DefaultAzureCredential()
    )

    # Workflow is a fully configured AgentGraph
    return workflow

# Execute workflow
async def run_content_pipeline(topic: str):
    workflow = await load_workflow()

    result = await workflow.run(
        initial_state={
            "topic": topic,
            "target_word_count": 1500,
            "target_audience": "technical professionals"
        }
    )

    return result
```

#### **.NET - Load from JSON**

```csharp
using Microsoft.Agents.AI.Config;
using System.Text.Json;
using Azure.Identity;

public class DeclarativeAgentLoader
{
    public static async Task<ChatAgent> LoadAgentFromJsonAsync(string configPath)
    {
        // Read configuration
        var json = await File.ReadAllTextAsync(configPath);
        var config = JsonSerializer.Deserialize<AgentConfig>(json);

        // Load agent from configuration
        var agent = await AgentConfig.LoadAgentAsync(
            config: config,
            credential: new DefaultAzureCredential()
        );

        return agent;
    }

    public static async Task Main()
    {
        // Load agent from JSON
        var agent = await LoadAgentFromJsonAsync("agent-config.json");

        // Agent is fully configured and ready
        var result = await agent.RunAsync("Analyze sales data for Q3 2025");
        Console.WriteLine(result);
    }
}
```

#### **.NET - Load Workflow from YAML**

```csharp
using Microsoft.Agents.AI.Config;
using Microsoft.Agents.AI.Graphs;
using YamlDotNet.Serialization;

public class WorkflowLoader
{
    public static async Task<AgentGraph> LoadWorkflowAsync(string yamlPath)
    {
        // Read YAML
        var yaml = await File.ReadAllTextAsync(yamlPath);

        // Deserialize
        var deserializer = new DeserializerBuilder().Build();
        var config = deserializer.Deserialize<WorkflowConfig>(yaml);

        // Create graph from config
        var workflow = await WorkflowConfig.BuildGraphAsync(
            config: config,
            credential: new DefaultAzureCredential()
        );

        return workflow;
    }

    public static async Task Main()
    {
        var workflow = await LoadWorkflowAsync("content-workflow.yaml");

        var result = await workflow.RunAsync(new Dictionary<string, object>
        {
            ["topic"] = "Cloud Computing Trends 2025",
            ["format"] = "blog_post"
        });

        Console.WriteLine($"Final Content: {result["final_content"]}");
    }
}
```

### Agent Templates

#### **Template with Placeholders**

```yaml
# agent-template.yaml
agent:
  name: ${AGENT_NAME}
  version: "1.0.0"

  model:
    provider: azure-openai
    deployment: ${MODEL_DEPLOYMENT}
    temperature: ${TEMPERATURE:0.7}
    max_tokens: ${MAX_TOKENS:1000}

  instructions: ${AGENT_INSTRUCTIONS}

  tools: ${TOOLS:[]}

  memory:
    type: ${MEMORY_TYPE:ephemeral}
    backend: ${MEMORY_BACKEND}

  authentication:
    method: ${AUTH_METHOD:entra-id}
```

#### **Using Templates (Python)**

```python
from agent_framework.config import load_agent_from_template
import os

# Set environment variables
os.environ["AGENT_NAME"] = "CustomSupportAgent"
os.environ["MODEL_DEPLOYMENT"] = "gpt-4o-mini"
os.environ["AGENT_INSTRUCTIONS"] = "You are a helpful support agent."
os.environ["MEMORY_TYPE"] = "persistent"
os.environ["MEMORY_BACKEND"] = "cosmos-db"

# Load from template with environment substitution
agent = await load_agent_from_template(
    template_path="agent-template.yaml",
    substitutions={
        "TEMPERATURE": "0.8",
        "MAX_TOKENS": "1500"
    }
)
```

### Configuration Validation

#### **Validation Schema**

```python
from agent_framework.config import AgentConfigValidator
from pydantic import ValidationError

async def validate_config():
    """Validate agent configuration before loading"""

    validator = AgentConfigValidator()

    try:
        # Validate YAML config
        validation_result = validator.validate_file("agent-config.yaml")

        if validation_result.is_valid:
            print("✓ Configuration is valid")
            print(f"Agent: {validation_result.agent_name}")
            print(f"Tools: {', '.join(validation_result.tool_names)}")
        else:
            print("✗ Configuration is invalid")
            for error in validation_result.errors:
                print(f"  - {error}")

    except ValidationError as e:
        print(f"Validation error: {e}")
```

---

## Streaming in Graphs

Enable real-time streaming of agent outputs as graphs execute.

### Streaming Graph Execution (Python)

```python
from agent_framework.graphs import AgentGraph

async def stream_graph_execution():
    """Stream outputs from graph execution"""

    graph = await create_research_workflow()

    # Stream execution
    async for event in graph.stream(
        initial_state={"topic": "Quantum Computing"},
        stream_mode="updates"  # Options: updates, values, debug
    ):
        event_type = event.get("type")
        node_id = event.get("node")
        data = event.get("data")

        print(f"[{event_type}] Node: {node_id}")

        if event_type == "node_start":
            print(f"  Starting: {node_id}")
        elif event_type == "node_output":
            print(f"  Output: {data}")
        elif event_type == "node_complete":
            print(f"  Completed: {node_id}")
        elif event_type == "edge_transition":
            print(f"  Transitioning: {data['from']} -> {data['to']}")
```

### Streaming with Token-by-Token Output

```python
async def stream_tokens_from_graph():
    """Stream individual tokens from agent outputs"""

    graph = await create_research_workflow()

    async for event in graph.stream(
        initial_state={"topic": "AI Safety"},
        stream_mode="tokens"
    ):
        node_id = event.get("node")
        token = event.get("token")

        # Print tokens as they arrive
        print(token, end="", flush=True)

    print()  # Newline at end
```

---

## Checkpointing and State Management

Checkpointing enables saving and resuming graph execution, supporting long-running workflows and failure recovery.

### Automatic Checkpointing

```python
from agent_framework.graphs import AgentGraph, CheckpointConfig

async def create_checkpointed_workflow():
    """Create workflow with automatic checkpointing"""

    graph = AgentGraph(name="LongRunningWorkflow")

    # Configure checkpointing
    checkpoint_config = CheckpointConfig(
        backend="azure-storage",
        connection_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
        container="agent-checkpoints",
        checkpoint_interval="node_completion",  # After each node
        retention_days=7
    )

    graph.configure_checkpointing(checkpoint_config)

    # Add nodes...
    # (graph definition here)

    return graph

# Execute with checkpointing
async def run_with_checkpoints():
    graph = await create_checkpointed_workflow()

    # Start execution
    run_id = await graph.run_async(
        initial_state={"data": "..."},
        checkpoint_id="checkpoint_001"
    )

    print(f"Started workflow with run ID: {run_id}")
```

### Resume from Checkpoint

```python
async def resume_from_checkpoint(checkpoint_id: str):
    """Resume workflow from saved checkpoint"""

    graph = await create_checkpointed_workflow()

    # Load checkpoint
    checkpoint = await graph.load_checkpoint(checkpoint_id)

    print(f"Resuming from node: {checkpoint.current_node}")
    print(f"State: {checkpoint.state}")

    # Resume execution
    result = await graph.resume(checkpoint)

    return result
```

### Manual Checkpoints

```python
async def manual_checkpoint_control():
    """Manually control checkpoint creation"""

    graph = await create_checkpointed_workflow()

    async for event in graph.stream(initial_state={"task": "process"}):
        if event.get("type") == "node_complete":
            node_id = event.get("node")

            # Create checkpoint after important nodes
            if node_id in ["data_processing", "analysis"]:
                checkpoint = await graph.create_checkpoint(
                    name=f"checkpoint_{node_id}",
                    metadata={"node": node_id, "timestamp": datetime.utcnow()}
                )
                print(f"Checkpoint created: {checkpoint.id}")
```

---

## Human-in-the-Loop Integration

Enable human oversight and approval at critical points in agent workflows.

### HITL Configuration

```python
from agent_framework.graphs import AgentGraph, HITLConfig

async def create_hitl_workflow():
    """Workflow with human-in-the-loop checkpoints"""

    graph = AgentGraph(name="ContentApprovalWorkflow")

    # Add nodes
    graph.add_node("draft", agent=writer_agent)
    graph.add_node("review", agent=reviewer_agent)
    graph.add_node("publish", agent=publisher_agent)

    # Configure HITL after review node
    hitl_config = HITLConfig(
        enabled=True,
        approval_endpoint="https://approval.contoso.com/api/requests",
        notification_channels=["email", "teams"],
        timeout_seconds=3600,  # 1 hour timeout
        escalation_policy={
            "timeout_action": "reject",
            "escalate_to": "manager@contoso.com"
        }
    )

    # Add HITL checkpoint
    graph.add_hitl_checkpoint(
        after_node="review",
        config=hitl_config,
        approval_prompt="Review this content draft for approval"
    )

    graph.add_edge("draft", "review")
    graph.add_edge("review", "publish")

    return graph

# Execute with HITL
async def run_with_human_approval():
    graph = await create_hitl_workflow()

    result = await graph.run(
        initial_state={"topic": "Product Launch"},
        hitl_callback=handle_approval_request
    )

    return result

async def handle_approval_request(approval_request):
    """Handle approval request"""
    print(f"Approval requested for: {approval_request.content}")
    print(f"Request ID: {approval_request.id}")

    # In production, this would trigger notification
    # and wait for human approval via API/UI

    # For demo, auto-approve
    await approval_request.approve(
        approver="user@contoso.com",
        comments="Looks good!"
    )
```

### Interactive HITL

```python
async def interactive_hitl():
    """Interactive human-in-the-loop workflow"""

    graph = await create_hitl_workflow()

    async for event in graph.stream(initial_state={"task": "create_report"}):
        if event.get("type") == "hitl_required":
            # Present to human
            content = event.get("content")
            print(f"\n{'='*60}")
            print(f"HUMAN APPROVAL REQUIRED")
            print(f"{'='*60}")
            print(f"Content: {content}")
            print(f"{'='*60}")

            # Get human input
            decision = input("Approve? (yes/no/edit): ").lower()

            if decision == "yes":
                await graph.approve_hitl(event.get("request_id"))
            elif decision == "no":
                await graph.reject_hitl(
                    event.get("request_id"),
                    reason="Not approved"
                )
            elif decision == "edit":
                edited = input("Enter edited content: ")
                await graph.approve_hitl(
                    event.get("request_id"),
                    modified_content=edited
                )
```

---

## Time-Travel Debugging

Navigate through graph execution history for debugging and analysis.

### Enable Time-Travel

```python
from agent_framework.graphs import AgentGraph, TimeTravelConfig

async def create_debuggable_workflow():
    """Create workflow with time-travel debugging"""

    graph = AgentGraph(name="DebuggableWorkflow")

    # Configure time-travel
    time_travel_config = TimeTravelConfig(
        enabled=True,
        save_all_states=True,
        save_intermediate_outputs=True,
        retention_period_days=30
    )

    graph.configure_time_travel(time_travel_config)

    # Add nodes...
    return graph
```

### Navigate History

```python
async def debug_workflow_execution(run_id: str):
    """Debug workflow using time-travel"""

    graph = await create_debuggable_workflow()

    # Load execution history
    history = await graph.get_execution_history(run_id)

    print(f"Total steps: {len(history.steps)}")

    # Navigate through states
    for i, step in enumerate(history.steps):
        print(f"\nStep {i}: {step.node_id}")
        print(f"  State: {step.state}")
        print(f"  Output: {step.output}")
        print(f"  Duration: {step.duration_ms}ms")

    # Jump to specific step
    step_5_state = await history.get_state_at_step(5)
    print(f"\nState at step 5: {step_5_state}")

    # Replay from specific step
    print("\nReplaying from step 5...")
    replay_result = await graph.replay_from_step(
        run_id=run_id,
        step_number=5,
        modified_state={"debug": True}
    )
```

### Visual Timeline

```python
async def visualize_execution_timeline(run_id: str):
    """Create visual timeline of execution"""

    graph = await create_debuggable_workflow()
    history = await graph.get_execution_history(run_id)

    # Generate timeline
    timeline = []
    for step in history.steps:
        timeline.append({
            "timestamp": step.timestamp,
            "node": step.node_id,
            "duration_ms": step.duration_ms,
            "status": step.status
        })

    # Export to visualization format
    await history.export_timeline(
        format="html",
        output_path=f"timeline_{run_id}.html"
    )

    print(f"Timeline saved to timeline_{run_id}.html")
```

---

## Production Patterns

### Error Handling in Graphs

```python
from agent_framework.graphs import AgentGraph, ErrorPolicy

async def create_resilient_graph():
    """Graph with comprehensive error handling"""

    graph = AgentGraph(name="ResilientWorkflow")

    # Configure error policy
    error_policy = ErrorPolicy(
        on_node_error="retry",
        retry_attempts=3,
        retry_delay_seconds=5,
        retry_backoff="exponential",
        fallback_node="error_handler",
        save_failed_state=True
    )

    graph.configure_error_policy(error_policy)

    # Add error handler node
    error_handler_agent = ChatAgent(
        instructions="Handle errors and provide fallback responses."
    )

    graph.add_node("error_handler", agent=error_handler_agent)

    return graph
```

### Performance Optimization

```python
async def create_optimized_graph():
    """Optimized graph for performance"""

    graph = AgentGraph(name="OptimizedWorkflow")

    # Enable parallel execution where possible
    graph.configure_parallelization(
        max_parallel_nodes=4,
        enable_batch_processing=True
    )

    # Enable caching
    graph.configure_caching(
        backend="redis",
        connection_string=os.getenv("REDIS_CONNECTION_STRING"),
        ttl_seconds=3600
    )

    return graph
```

### Monitoring and Observability

```python
from agent_framework.graphs import AgentGraph, ObservabilityConfig

async def create_observable_graph():
    """Graph with full observability"""

    graph = AgentGraph(name="ObservableWorkflow")

    # Configure observability
    observability_config = ObservabilityConfig(
        enable_tracing=True,
        enable_metrics=True,
        enable_logging=True,
        export_endpoint="https://otel-collector.contoso.com",
        custom_tags={
            "environment": "production",
            "version": "1.0.0"
        }
    )

    graph.configure_observability(observability_config)

    return graph
```

---

## Best Practices

### 1. Graph Design
- Keep graphs simple and focused
- Use descriptive node and edge names
- Document conditional logic clearly
- Test graphs with various input scenarios

### 2. Declarative Configuration
- Store configurations in version control
- Use environment variables for secrets
- Validate configurations before deployment
- Maintain separate configs for dev/staging/prod

### 3. Checkpointing
- Checkpoint after expensive operations
- Set appropriate retention periods
- Test checkpoint recovery regularly
- Monitor checkpoint storage costs

### 4. Human-in-the-Loop
- Set reasonable timeout values
- Implement escalation policies
- Provide clear approval prompts
- Log all approval decisions

### 5. Debugging
- Enable time-travel in development
- Review execution timelines regularly
- Use step-by-step replay for issues
- Archive debugging data appropriately

---

## Migration Guide

### From Direct Agent Orchestration

```python
# Before: Manual orchestration
research_result = await research_agent.run(query)
analysis_result = await analysis_agent.run(research_result.text)
summary_result = await summary_agent.run(analysis_result.text)

# After: Graph-based
graph = await create_research_workflow()
result = await graph.run({"query": query})
```

### From Configuration Code to Declarative

```python
# Before: Hardcoded configuration
agent = ChatAgent(
    instructions="You are helpful.",
    model="gpt-4o-mini",
    temperature=0.7
)

# After: Declarative YAML
agent = await load_agent_from_yaml("agent-config.yaml")
```

---

## Future Roadmap

- **Dynamic Graph Modification:** Modify graphs during execution
- **Graph Templates:** Reusable graph patterns
- **Advanced Parallelization:** GPU-accelerated graph execution
- **Visual Graph Editor:** UI for designing workflows
- **Cross-Graph Communication:** Graphs calling other graphs

---

**Last Updated:** November 2025
**Document Version:** 1.0
**Preview Status:** October 2025 Public Preview
**Next Review:** Q2 2026
