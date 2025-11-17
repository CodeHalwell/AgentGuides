# Pydantic AI: 2025 Integration Features
## MCP, A2A Protocol, and UI Event Streams

**Version:** 1.0.0 (2025)
**Framework:** Pydantic AI Advanced Integrations
**Focus:** Model Context Protocol, Agent-to-Agent communication, and UI streaming

---

## Table of Contents

1. [Model Context Protocol (MCP)](#model-context-protocol-mcp)
2. [Agent-to-Agent Protocol (A2A)](#agent-to-agent-protocol-a2a)
3. [UI Event Streams](#ui-event-streams)
4. [Enhanced Model Support](#enhanced-model-support)
5. [Production Integration Patterns](#production-integration-patterns)

---

## Model Context Protocol (MCP)

### What is MCP?

The Model Context Protocol (MCP) is an open protocol that enables seamless integration between AI applications and external data sources. It provides a standardized way to:

- Connect to databases, filesystems, APIs
- Share tools across applications
- Maintain consistent context
- Enable modular agent architectures

**Official Specification:** [modelcontextprotocol.io](https://modelcontextprotocol.io)

### MCP Client Integration

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPClient, MCPServer
from mcp import types
import asyncio

# Connect to MCP server
async def create_mcp_client() -> MCPClient:
    """Create MCP client connected to server."""

    client = MCPClient()

    # Connect to local MCP server
    await client.connect_stdio(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/path/to/data"]
    )

    # Or connect to remote MCP server
    # await client.connect_sse("http://localhost:8000/mcp")

    return client

# Use MCP tools in Pydantic AI
async def agent_with_mcp():
    """Agent using MCP tools."""

    # Initialize MCP client
    mcp_client = await create_mcp_client()

    # List available MCP tools
    tools_list = await mcp_client.list_tools()
    print(f"Available MCP tools: {[tool.name for tool in tools_list.tools]}")

    # Create agent with MCP tools
    agent = Agent(
        'openai:gpt-4o',
        instructions='You have access to filesystem tools via MCP'
    )

    # Register MCP tools with agent
    for tool in tools_list.tools:
        @agent.tool
        async def mcp_tool(ctx, **kwargs):
            """Dynamic MCP tool wrapper."""
            result = await mcp_client.call_tool(tool.name, kwargs)
            return result.content

    # Use agent with MCP tools
    result = await agent.run(
        "Read the file 'example.txt' and summarize its contents"
    )

    print(result.output)

    # Cleanup
    await mcp_client.disconnect()

# Run
asyncio.run(agent_with_mcp())
```

### MCP Server Implementation

```python
from pydantic_ai.mcp import MCPServer, Tool, Resource
from pydantic import BaseModel
from typing import Optional, List

class DatabaseMCPServer(MCPServer):
    """Custom MCP server for database access."""

    def __init__(self, db_connection_string: str):
        super().__init__(
            name="database-mcp-server",
            version="1.0.0"
        )
        self.db_url = db_connection_string

    async def initialize(self):
        """Initialize database connection."""
        # Setup database connection pool
        import asyncpg
        self.pool = await asyncpg.create_pool(self.db_url)

    @Tool(
        name="query_database",
        description="Execute SQL query and return results",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "SQL query to execute"},
                "limit": {"type": "integer", "description": "Max results", "default": 100}
            },
            "required": ["query"]
        }
    )
    async def query_database(self, query: str, limit: int = 100) -> dict:
        """Execute database query."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(f"{query} LIMIT {limit}")

            return {
                "rows": [dict(row) for row in rows],
                "count": len(rows)
            }

    @Tool(
        name="get_table_schema",
        description="Get schema information for a table",
        parameters={
            "type": "object",
            "properties": {
                "table_name": {"type": "string", "description": "Name of the table"}
            },
            "required": ["table_name"]
        }
    )
    async def get_table_schema(self, table_name: str) -> dict:
        """Get table schema."""
        async with self.pool.acquire() as conn:
            columns = await conn.fetch("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = $1
            """, table_name)

            return {
                "table": table_name,
                "columns": [dict(col) for col in columns]
            }

    @Resource(
        uri_pattern="db://tables/{table_name}",
        name="Database Table",
        description="Access database table data"
    )
    async def get_table_resource(self, table_name: str) -> str:
        """Get table data as resource."""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(f"SELECT * FROM {table_name} LIMIT 100")
            return str([dict(row) for row in rows])

# Run MCP server
async def run_mcp_server():
    """Start MCP server."""

    server = DatabaseMCPServer("postgresql://localhost/mydb")
    await server.initialize()

    # Start server (SSE transport)
    await server.run_sse(host="0.0.0.0", port=8000)

    # Or stdio transport for local use
    # await server.run_stdio()

if __name__ == '__main__':
    asyncio.run(run_mcp_server())
```

### MCP with Multiple Servers

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPClient, MultiMCPClient
import asyncio

async def agent_with_multiple_mcp_servers():
    """Agent connected to multiple MCP servers."""

    # Create multi-client
    multi_client = MultiMCPClient()

    # Connect to filesystem MCP server
    await multi_client.add_server(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/data"]
    )

    # Connect to database MCP server
    await multi_client.add_server(
        name="database",
        command="python",
        args=["database_mcp_server.py"]
    )

    # Connect to GitHub MCP server
    await multi_client.add_server(
        name="github",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-github"],
        env={"GITHUB_TOKEN": "your-token"}
    )

    # List all available tools across servers
    all_tools = await multi_client.list_all_tools()
    print(f"Total tools available: {len(all_tools)}")

    for server_name, tools in all_tools.items():
        print(f"{server_name}: {[t.name for t in tools]}")

    # Create agent with all MCP tools
    agent = Agent(
        'openai:gpt-4o',
        instructions="""
        You have access to multiple data sources via MCP:
        - Filesystem: Read/write files
        - Database: Query SQL database
        - GitHub: Access repositories and issues
        """
    )

    # Register all tools
    for server_name, tools in all_tools.items():
        for tool in tools:
            # Create wrapped tool that knows which server to call
            async def create_mcp_tool(srv_name, tool_name):
                async def mcp_tool_wrapper(ctx, **kwargs):
                    result = await multi_client.call_tool(
                        server_name=srv_name,
                        tool_name=tool_name,
                        arguments=kwargs
                    )
                    return result.content

                return mcp_tool_wrapper

            tool_func = await create_mcp_tool(server_name, tool.name)
            agent.add_tool(tool_func, name=f"{server_name}_{tool.name}")

    # Use agent
    result = await agent.run("""
        1. Read the project README from filesystem
        2. Check recent GitHub issues
        3. Query database for user statistics
        4. Summarize all findings
    """)

    print(result.output)

    # Cleanup
    await multi_client.disconnect_all()

asyncio.run(agent_with_multiple_mcp_servers())
```

### MCP Resource Subscriptions

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPClient, ResourceSubscription
import asyncio

async def agent_with_mcp_subscriptions():
    """Agent that subscribes to MCP resources for live updates."""

    client = await create_mcp_client()

    # Subscribe to resource updates
    subscription = ResourceSubscription(
        uri="file:///data/logs/app.log",
        on_update=lambda content: print(f"Log updated: {content}")
    )

    await client.subscribe_resource(subscription)

    # Create agent
    agent = Agent('openai:gpt-4o')

    # Agent can now react to resource changes
    @agent.tool
    async def get_latest_logs(ctx) -> str:
        """Get latest log entries."""
        resource = await client.read_resource("file:///data/logs/app.log")
        return resource.contents

    # Monitor and react to changes
    result = await agent.run("Monitor logs and alert on errors")

    # Unsubscribe
    await client.unsubscribe_resource(subscription)
```

---

## Agent-to-Agent Protocol (A2A)

### What is A2A?

The Agent-to-Agent Protocol enables direct communication between AI agents, allowing:

- Distributed agent networks
- Agent discovery and registration
- Message passing and RPC
- Coordinated multi-agent workflows
- Peer-to-peer agent collaboration

### Basic A2A Communication

```python
from pydantic_ai import Agent
from pydantic_ai.a2a import A2AAgent, A2AMessage, A2AProtocol
from pydantic import BaseModel
from typing import Optional

class TaskMessage(BaseModel):
    """Message format for task delegation."""
    task_id: str
    task_type: str
    payload: dict
    priority: int = 1

class ResultMessage(BaseModel):
    """Message format for results."""
    task_id: str
    result: str
    success: bool
    metadata: dict = {}

class CoordinatorAgent(A2AAgent):
    """Coordinator agent that delegates tasks."""

    def __init__(self):
        super().__init__(
            agent_id="coordinator-001",
            capabilities=["task_coordination", "delegation"],
            protocol_version="1.0"
        )

        self.agent = Agent(
            'openai:gpt-4o',
            instructions='You coordinate tasks among worker agents'
        )

    async def handle_request(
        self,
        message: A2AMessage
    ) -> A2AMessage:
        """Handle incoming A2A requests."""

        if message.message_type == "task_request":
            # Decide which worker to delegate to
            task = TaskMessage(**message.payload)

            # Find suitable worker
            workers = await self.discover_agents(capability="worker")

            if not workers:
                return A2AMessage(
                    from_agent=self.agent_id,
                    to_agent=message.from_agent,
                    message_type="error",
                    payload={"error": "No workers available"}
                )

            # Delegate to worker
            worker = workers[0]
            response = await self.send_message(
                to_agent=worker.agent_id,
                message_type="execute_task",
                payload=task.model_dump()
            )

            return response

class WorkerAgent(A2AAgent):
    """Worker agent that executes tasks."""

    def __init__(self, worker_id: str, specialty: str):
        super().__init__(
            agent_id=worker_id,
            capabilities=["worker", specialty],
            protocol_version="1.0"
        )

        self.specialty = specialty
        self.agent = Agent(
            'anthropic:claude-3-5-sonnet-latest',
            instructions=f'You are a specialist in {specialty}'
        )

    async def handle_request(
        self,
        message: A2AMessage
    ) -> A2AMessage:
        """Handle task execution requests."""

        if message.message_type == "execute_task":
            task = TaskMessage(**message.payload)

            # Execute task with AI agent
            result = await self.agent.run(
                f"Execute {task.task_type}: {task.payload}"
            )

            # Return result
            return A2AMessage(
                from_agent=self.agent_id,
                to_agent=message.from_agent,
                message_type="task_result",
                payload=ResultMessage(
                    task_id=task.task_id,
                    result=result.output,
                    success=True,
                    metadata={'tokens_used': result.usage().total_tokens}
                ).model_dump()
            )

# Setup A2A network
async def setup_a2a_network():
    """Create and connect A2A agents."""

    # Create coordinator
    coordinator = CoordinatorAgent()
    await coordinator.start()

    # Create workers
    workers = [
        WorkerAgent("worker-001", "data_analysis"),
        WorkerAgent("worker-002", "text_processing"),
        WorkerAgent("worker-003", "code_generation")
    ]

    for worker in workers:
        await worker.start()

    # Coordinator delegates task
    task = TaskMessage(
        task_id="task-123",
        task_type="analyze_data",
        payload={"data": "sample data"}
    )

    response = await coordinator.send_message(
        to_agent="worker-001",
        message_type="execute_task",
        payload=task.model_dump()
    )

    result = ResultMessage(**response.payload)
    print(f"Task result: {result.result}")

    # Cleanup
    await coordinator.stop()
    for worker in workers:
        await worker.stop()
```

### A2A Discovery and Registry

```python
from pydantic_ai.a2a import A2ARegistry, AgentCapability, AgentMetadata
from typing import List

class A2AAgentRegistry:
    """Central registry for agent discovery."""

    def __init__(self):
        self.agents: dict[str, AgentMetadata] = {}

    async def register_agent(
        self,
        agent_id: str,
        capabilities: List[str],
        endpoint: str,
        metadata: dict = {}
    ):
        """Register an agent."""

        self.agents[agent_id] = AgentMetadata(
            agent_id=agent_id,
            capabilities=capabilities,
            endpoint=endpoint,
            metadata=metadata,
            status="active"
        )

    async def discover_agents(
        self,
        capability: Optional[str] = None,
        metadata_filter: Optional[dict] = None
    ) -> List[AgentMetadata]:
        """Discover agents by capability."""

        results = []

        for agent in self.agents.values():
            if agent.status != "active":
                continue

            if capability and capability not in agent.capabilities:
                continue

            if metadata_filter:
                matches = all(
                    agent.metadata.get(k) == v
                    for k, v in metadata_filter.items()
                )
                if not matches:
                    continue

            results.append(agent)

        return results

    async def heartbeat(self, agent_id: str):
        """Update agent heartbeat."""
        if agent_id in self.agents:
            self.agents[agent_id].last_heartbeat = datetime.now()

# Usage
registry = A2AAgentRegistry()

# Agents register themselves
await registry.register_agent(
    agent_id="researcher-001",
    capabilities=["research", "web_search"],
    endpoint="http://agent1.example.com:8000"
)

await registry.register_agent(
    agent_id="analyst-001",
    capabilities=["analysis", "data_processing"],
    endpoint="http://agent2.example.com:8000"
)

# Discover agents
research_agents = await registry.discover_agents(capability="research")
print(f"Found {len(research_agents)} research agents")
```

### A2A Workflow Coordination

```python
from pydantic_ai.a2a import A2AWorkflow, WorkflowStep, A2AOrchestrator
from pydantic import BaseModel

class DocumentProcessingWorkflow(A2AWorkflow):
    """Multi-agent workflow using A2A protocol."""

    def __init__(self, registry: A2AAgentRegistry):
        super().__init__(
            workflow_id="doc-processing-v1",
            registry=registry
        )

    async def execute(self, document: str) -> dict:
        """Execute multi-agent document processing."""

        results = {}

        # Step 1: Find OCR agent
        ocr_agents = await self.registry.discover_agents(capability="ocr")
        if not ocr_agents:
            raise ValueError("No OCR agents available")

        ocr_result = await self.call_agent(
            ocr_agents[0],
            "extract_text",
            {"document": document}
        )
        results['text'] = ocr_result['text']

        # Step 2: Find translation agent (parallel with step 3)
        translation_agents = await self.registry.discover_agents(
            capability="translation"
        )

        # Step 3: Find summarization agent (parallel with step 2)
        summary_agents = await self.registry.discover_agents(
            capability="summarization"
        )

        # Execute in parallel
        import asyncio

        translation_task = self.call_agent(
            translation_agents[0],
            "translate",
            {"text": results['text'], "target_lang": "en"}
        )

        summary_task = self.call_agent(
            summary_agents[0],
            "summarize",
            {"text": results['text']}
        )

        translation_result, summary_result = await asyncio.gather(
            translation_task,
            summary_task
        )

        results['translation'] = translation_result['translated_text']
        results['summary'] = summary_result['summary']

        # Step 4: Find analysis agent (depends on steps 2 & 3)
        analysis_agents = await self.registry.discover_agents(
            capability="analysis"
        )

        analysis_result = await self.call_agent(
            analysis_agents[0],
            "analyze",
            {
                "text": results['translation'],
                "summary": results['summary']
            }
        )

        results['analysis'] = analysis_result['analysis']

        return results

# Execute workflow
workflow = DocumentProcessingWorkflow(registry)
result = await workflow.execute("path/to/document.pdf")
```

---

## UI Event Streams

### Real-Time UI Updates

```python
from pydantic_ai import Agent
from pydantic_ai.ui import UIEventStream, StreamEvent
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse
from typing import AsyncIterator
import asyncio

app = FastAPI()

class AgentUIStream:
    """Stream agent events to UI."""

    def __init__(self, agent: Agent):
        self.agent = agent
        self.event_stream = UIEventStream()

    async def stream_to_ui(
        self,
        query: str
    ) -> AsyncIterator[StreamEvent]:
        """Stream agent execution events to UI."""

        # Emit start event
        yield StreamEvent(
            type="agent_start",
            data={"query": query, "timestamp": "2025-01-01T00:00:00Z"}
        )

        # Stream agent execution
        async with self.agent.run_stream(query) as response:

            # Stream thinking/reasoning
            yield StreamEvent(
                type="agent_thinking",
                data={"status": "analyzing query"}
            )

            # Stream tool calls
            for tool_call in response.tool_calls():
                yield StreamEvent(
                    type="tool_call",
                    data={
                        "tool_name": tool_call.name,
                        "arguments": tool_call.arguments
                    }
                )

                # Stream tool result
                tool_result = await tool_call.execute()

                yield StreamEvent(
                    type="tool_result",
                    data={
                        "tool_name": tool_call.name,
                        "result": str(tool_result)[:200]
                    }
                )

            # Stream text output
            async for text_chunk in response.stream_text():
                yield StreamEvent(
                    type="text_delta",
                    data={"delta": text_chunk}
                )

            # Get final result
            result = await response.result()

            # Emit completion event
            yield StreamEvent(
                type="agent_complete",
                data={
                    "output": result.output,
                    "tokens_used": result.usage().total_tokens,
                    "duration_ms": 1234
                }
            )

@app.get("/api/agent/stream")
async def stream_agent_response(query: str):
    """SSE endpoint for streaming agent responses."""

    agent = Agent('openai:gpt-4o')
    ui_stream = AgentUIStream(agent)

    async def event_generator():
        async for event in ui_stream.stream_to_ui(query):
            # Format as SSE
            yield f"event: {event.type}\n"
            yield f"data: {event.data_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )

@app.websocket("/ws/agent")
async def websocket_agent(websocket: WebSocket):
    """WebSocket endpoint for bi-directional agent communication."""

    await websocket.accept()

    agent = Agent('openai:gpt-4o')
    ui_stream = AgentUIStream(agent)

    try:
        while True:
            # Receive query from client
            data = await websocket.receive_json()
            query = data['query']

            # Stream response back
            async for event in ui_stream.stream_to_ui(query):
                await websocket.send_json({
                    'type': event.type,
                    'data': event.data
                })

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()
```

### React Frontend Integration

```typescript
// React component for agent UI streaming

import { useEffect, useState } from 'react';

interface AgentEvent {
  type: string;
  data: any;
}

function AgentChat() {
  const [messages, setMessages] = useState<string[]>([]);
  const [currentResponse, setCurrentResponse] = useState<string>('');
  const [toolCalls, setToolCalls] = useState<any[]>([]);
  const [isThinking, setIsThinking] = useState(false);

  const sendQuery = async (query: string) => {
    // Reset state
    setCurrentResponse('');
    setToolCalls([]);
    setIsThinking(false);

    // Connect to SSE endpoint
    const eventSource = new EventSource(
      `/api/agent/stream?query=${encodeURIComponent(query)}`
    );

    eventSource.addEventListener('agent_start', (e) => {
      const data = JSON.parse(e.data);
      setMessages(prev => [...prev, `User: ${query}`]);
    });

    eventSource.addEventListener('agent_thinking', (e) => {
      setIsThinking(true);
    });

    eventSource.addEventListener('tool_call', (e) => {
      const data = JSON.parse(e.data);
      setToolCalls(prev => [...prev, data]);
    });

    eventSource.addEventListener('text_delta', (e) => {
      const data = JSON.parse(e.data);
      setCurrentResponse(prev => prev + data.delta);
      setIsThinking(false);
    });

    eventSource.addEventListener('agent_complete', (e) => {
      const data = JSON.parse(e.data);
      setMessages(prev => [...prev, `Agent: ${data.output}`]);
      setCurrentResponse('');
      eventSource.close();
    });

    eventSource.onerror = () => {
      eventSource.close();
    };
  };

  return (
    <div className="agent-chat">
      <div className="messages">
        {messages.map((msg, i) => (
          <div key={i} className="message">{msg}</div>
        ))}

        {isThinking && <div className="thinking">Agent is thinking...</div>}

        {currentResponse && (
          <div className="current-response">
            Agent: {currentResponse}<span className="cursor">|</span>
          </div>
        )}

        {toolCalls.length > 0 && (
          <div className="tool-calls">
            <h4>Tools Used:</h4>
            {toolCalls.map((call, i) => (
              <div key={i} className="tool-call">
                {call.tool_name}({JSON.stringify(call.arguments)})
              </div>
            ))}
          </div>
        )}
      </div>

      <input
        type="text"
        placeholder="Ask something..."
        onKeyPress={(e) => {
          if (e.key === 'Enter') {
            sendQuery(e.currentTarget.value);
            e.currentTarget.value = '';
          }
        }}
      />
    </div>
  );
}
```

### Structured Event Streaming

```python
from pydantic_ai import Agent
from pydantic_ai.ui import StructuredEventStream
from pydantic import BaseModel
from typing import AsyncIterator, Literal
from enum import Enum

class EventType(str, Enum):
    PLANNING = "planning"
    EXECUTION = "execution"
    VALIDATION = "validation"
    COMPLETION = "completion"

class AgentEvent(BaseModel):
    """Type-safe agent event."""
    event_type: EventType
    stage: str
    message: str
    progress: float  # 0.0 to 1.0
    metadata: dict = {}

class StructuredAgentStream:
    """Stream structured, type-safe events."""

    def __init__(self, agent: Agent):
        self.agent = agent

    async def execute_with_events(
        self,
        query: str
    ) -> AsyncIterator[AgentEvent]:
        """Execute agent with structured event stream."""

        # Planning phase
        yield AgentEvent(
            event_type=EventType.PLANNING,
            stage="analyze_query",
            message="Analyzing query and planning approach",
            progress=0.1
        )

        await asyncio.sleep(0.5)  # Simulate planning

        yield AgentEvent(
            event_type=EventType.PLANNING,
            stage="select_tools",
            message="Selected 3 tools for execution",
            progress=0.2,
            metadata={"tools": ["search", "calculate", "summarize"]}
        )

        # Execution phase
        yield AgentEvent(
            event_type=EventType.EXECUTION,
            stage="running_tools",
            message="Executing tools",
            progress=0.4
        )

        async with self.agent.run_stream(query) as response:
            async for text in response.stream_text():
                yield AgentEvent(
                    event_type=EventType.EXECUTION,
                    stage="generating_response",
                    message=text,
                    progress=0.7
                )

            result = await response.result()

            # Validation phase
            yield AgentEvent(
                event_type=EventType.VALIDATION,
                stage="validating_output",
                message="Validating response quality",
                progress=0.9
            )

            # Completion
            yield AgentEvent(
                event_type=EventType.COMPLETION,
                stage="complete",
                message="Task completed successfully",
                progress=1.0,
                metadata={
                    "tokens_used": result.usage().total_tokens,
                    "output_length": len(result.output)
                }
            )

# FastAPI endpoint
@app.get("/api/agent/structured-stream")
async def structured_stream(query: str):
    """Stream structured events."""

    agent = Agent('openai:gpt-4o')
    stream = StructuredAgentStream(agent)

    async def event_generator():
        async for event in stream.execute_with_events(query):
            yield f"data: {event.model_dump_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )
```

---

## Enhanced Model Support

### All Provider Support (2025)

```python
from pydantic_ai import Agent
from pydantic_ai.models import (
    OpenAIModel,
    AnthropicModel,
    GeminiModel,
    DeepSeekModel,
    GrokModel,
    CohereModel,
    MistralModel,
    PerplexityModel,
    AzureOpenAIModel,
    BedrockModel,
    VertexAIModel,
    OllamaModel
)

# OpenAI (standard and new models)
openai_agent = Agent('openai:gpt-4o')
openai_o3 = Agent('openai:o3-mini')  # Reasoning model

# Anthropic Claude (latest models)
claude_sonnet = Agent('anthropic:claude-3-5-sonnet-20250219')
claude_opus = Agent('anthropic:claude-3-opus-20250219')

# Google Gemini
gemini_pro = Agent('google-gla:gemini-2.0-flash-exp')
gemini_flash = Agent('google-gla:gemini-1.5-flash')

# DeepSeek (cost-effective reasoning)
deepseek = Agent('deepseek:deepseek-chat')
deepseek_reasoner = Agent('deepseek:deepseek-reasoner')

# Grok (xAI)
grok = Agent('grok:grok-2-latest')
grok_vision = Agent('grok:grok-2-vision-latest')

# Cohere
cohere_command = Agent('cohere:command-r-plus')

# Mistral
mistral_large = Agent('mistral:mistral-large-latest')
mistral_small = Agent('mistral:mistral-small-latest')

# Perplexity (with online search)
perplexity = PerplexityModel('sonar-pro')
perplexity_agent = Agent(perplexity)

# Azure OpenAI
azure_model = AzureOpenAIModel(
    deployment_name='gpt-4o',
    azure_endpoint='https://your-resource.openai.azure.com',
    api_version='2024-10-01-preview'
)
azure_agent = Agent(azure_model)

# AWS Bedrock
bedrock_model = BedrockModel(
    model_id='anthropic.claude-3-5-sonnet-20241022-v2:0',
    region='us-east-1'
)
bedrock_agent = Agent(bedrock_model)

# Google Vertex AI
vertex_model = VertexAIModel(
    model='gemini-1.5-pro',
    project='your-project-id',
    location='us-central1'
)
vertex_agent = Agent(vertex_model)

# Ollama (local models)
ollama_agent = Agent('ollama:llama3.1:70b')
ollama_custom = Agent('ollama:custom-model')
```

### Provider-Specific Features

```python
from pydantic_ai import Agent
from pydantic_ai.models import ModelSettings

# OpenAI with structured outputs (guaranteed JSON)
openai_structured = Agent(
    'openai:gpt-4o',
    output_type=MyModel,
    model_settings=ModelSettings(
        strict_schema=True  # OpenAI structured outputs
    )
)

# Anthropic with prompt caching
claude_cached = Agent(
    'anthropic:claude-3-5-sonnet-latest',
    model_settings=ModelSettings(
        cache_system_prompt=True,  # Cache expensive system prompts
        cache_ttl=300  # 5 minute cache
    )
)

# Gemini with thinking mode
gemini_thinking = Agent(
    'google-gla:gemini-2.0-flash-thinking-exp',
    model_settings=ModelSettings(
        thinking_mode=True  # Enable reasoning traces
    )
)

# DeepSeek with extended context
deepseek_long = Agent(
    'deepseek:deepseek-chat',
    model_settings=ModelSettings(
        max_tokens=4096,
        context_length=64000  # 64k context window
    )
)

# Perplexity with online search
perplexity_search = Agent(
    PerplexityModel('sonar-pro'),
    instructions='Always search for latest information',
    model_settings=ModelSettings(
        enable_online_search=True,
        search_domain_filter=['*.edu', '*.gov']
    )
)
```

### Model Fallback and Routing

```python
from pydantic_ai.models import FallbackModel, RouterModel
from pydantic_ai import Agent

# Fallback: Try primary, then backup
fallback_model = FallbackModel(
    primary='openai:gpt-4o',
    fallbacks=[
        'anthropic:claude-3-5-sonnet-latest',
        'google-gla:gemini-1.5-pro',
        'ollama:llama3.1:70b'  # Local fallback
    ]
)

agent = Agent(fallback_model)

# Router: Select model based on query
class SmartRouter(RouterModel):
    """Route to best model for query type."""

    async def select_model(self, query: str) -> str:
        """Select model based on query."""

        if "code" in query.lower():
            return 'anthropic:claude-3-5-sonnet-latest'  # Best for code

        elif "reasoning" in query.lower() or "math" in query.lower():
            return 'openai:o3-mini'  # Reasoning model

        elif "search" in query.lower() or "latest" in query.lower():
            return 'perplexity:sonar-pro'  # Has web search

        elif len(query) > 10000:
            return 'gemini-2.0-flash-exp'  # Long context

        else:
            return 'openai:gpt-4o'  # Default

router = SmartRouter()
agent = Agent(router)
```

---

## Production Integration Patterns

### Complete Integration Example

```python
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPClient
from pydantic_ai.a2a import A2AAgent
from pydantic_ai.ui import UIEventStream
from pydantic_ai.durable.prefect import PrefectDurableAgent
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel

class ProductionAgent(A2AAgent):
    """Production agent with all integrations."""

    def __init__(self, agent_id: str):
        super().__init__(
            agent_id=agent_id,
            capabilities=["data_analysis", "mcp_tools"]
        )

        # Durable Pydantic AI agent
        self.agent = PrefectDurableAgent(
            'openai:gpt-4o',
            checkpoint_every_n_steps=1
        )

        # MCP clients
        self.mcp_filesystem = None
        self.mcp_database = None

        # UI event stream
        self.ui_stream = UIEventStream()

    async def initialize(self):
        """Initialize all connections."""

        # Connect to MCP servers
        self.mcp_filesystem = MCPClient()
        await self.mcp_filesystem.connect_stdio(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/data"]
        )

        self.mcp_database = MCPClient()
        await self.mcp_database.connect_sse("http://localhost:8000/mcp")

        # Register MCP tools with agent
        await self._register_mcp_tools()

        # Start A2A communication
        await self.start()

    async def _register_mcp_tools(self):
        """Register all MCP tools."""

        tools = await self.mcp_filesystem.list_tools()

        for tool in tools:
            @self.agent.tool
            async def mcp_tool(ctx, **kwargs):
                result = await self.mcp_filesystem.call_tool(
                    tool.name,
                    kwargs
                )
                return result.content

    async def execute_task(
        self,
        task: dict,
        stream_to_ui: bool = False
    ) -> dict:
        """Execute task with full integration."""

        # Emit UI events if requested
        if stream_to_ui:
            await self.ui_stream.emit("task_start", task)

        # Execute with durable agent
        result = await self.agent.run(
            str(task),
            flow_run_name=f"task-{task['id']}"
        )

        # Emit completion
        if stream_to_ui:
            await self.ui_stream.emit("task_complete", {
                "result": result.output,
                "tokens": result.usage().total_tokens
            })

        return {
            "result": result.output,
            "metadata": {
                "tokens_used": result.usage().total_tokens,
                "mcp_tools_used": len(result.tool_calls),
                "durable": True
            }
        }

    async def handle_request(self, message):
        """Handle A2A requests."""

        if message.message_type == "execute_task":
            result = await self.execute_task(message.payload)

            return A2AMessage(
                from_agent=self.agent_id,
                to_agent=message.from_agent,
                message_type="task_result",
                payload=result
            )

# FastAPI application
app = FastAPI()

production_agent = ProductionAgent("production-agent-001")

@app.on_event("startup")
async def startup():
    await production_agent.initialize()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for UI streaming."""

    await websocket.accept()

    # Subscribe to UI events
    async for event in production_agent.ui_stream.subscribe():
        await websocket.send_json({
            "type": event.type,
            "data": event.data
        })

@app.post("/api/task")
async def execute_task(task: dict):
    """HTTP endpoint for task execution."""

    result = await production_agent.execute_task(task, stream_to_ui=True)
    return result
```

---

## Summary

**2025 Integration Features:**

**MCP (Model Context Protocol):**
- ✅ Connect to any MCP server
- ✅ Use tools across applications
- ✅ Build custom MCP servers
- ✅ Multiple server support

**A2A (Agent-to-Agent Protocol):**
- ✅ Direct agent communication
- ✅ Agent discovery and registry
- ✅ Multi-agent workflows
- ✅ Distributed agent networks

**UI Event Streams:**
- ✅ Real-time SSE/WebSocket streaming
- ✅ Structured event types
- ✅ React/Vue integration
- ✅ Progress tracking

**Enhanced Models:**
- ✅ 12+ LLM providers supported
- ✅ Provider-specific features
- ✅ Smart routing and fallbacks
- ✅ Cost optimization

---

**Next Steps:**
1. Choose integrations for your use case
2. Implement MCP for data sources
3. Add A2A for multi-agent systems
4. Stream events to UI for better UX
5. Use multiple providers for redundancy

**See Also:**
- `pydantic_ai_durable_execution.md` - Fault-tolerant execution
- `pydantic_ai_graph_support.md` - Graph-based workflows
- `pydantic_ai_production_guide.md` - Production deployment
