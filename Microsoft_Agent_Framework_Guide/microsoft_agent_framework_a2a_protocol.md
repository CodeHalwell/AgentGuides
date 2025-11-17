# Microsoft Agent Framework - Agent2Agent (A2A) Protocol Guide
## October 2025 Release - Cross-Framework Agent Collaboration

**Document Version:** 1.0
**Last Updated:** November 2025
**Target Audience:** Developers building multi-agent systems across frameworks
**Preview Status:** October 2025 Public Preview

---

## Table of Contents

1. [Introduction to A2A Protocol](#introduction-to-a2a-protocol)
2. [Core Concepts](#core-concepts)
3. [Cross-Framework Interoperability](#cross-framework-interoperability)
4. [Authentication Between Agents](#authentication-between-agents)
5. [Message Format Specification](#message-format-specification)
6. [Implementation Examples](#implementation-examples)
7. [Security Considerations](#security-considerations)
8. [Production Patterns](#production-patterns)

---

## Introduction to A2A Protocol

The Agent2Agent (A2A) Protocol is a standardized communication format introduced in the October 2025 preview release of Microsoft Agent Framework. It enables seamless collaboration between agents across different frameworks, platforms, and organizations.

### Key Objectives

- **Cross-Framework Communication:** Agents built with Microsoft Agent Framework can communicate with agents from OpenAI SDK, Claude SDK, LangGraph, Google ADK, and other frameworks
- **Remote Agent Communication:** Enable distributed agent systems across networks and cloud boundaries
- **Common Message Format:** Standardized JSON-based protocol for agent-to-agent messages
- **Secure Authentication:** Built-in authentication and authorization between agents
- **Framework Agnostic:** Protocol works independently of the underlying agent implementation

### Use Cases

1. **Enterprise Integration:** Connect agents across different departments using different frameworks
2. **Multi-Cloud Deployments:** Agents in Azure communicate with agents in AWS/GCP
3. **Partner Collaboration:** Share agent capabilities with external organizations
4. **Legacy System Integration:** Connect modern agents with legacy agent systems
5. **Vendor Diversity:** Avoid vendor lock-in by enabling cross-platform agent collaboration

---

## Core Concepts

### A2A Message Structure

```json
{
  "protocol": "a2a/1.0",
  "message_id": "msg_abc123xyz",
  "conversation_id": "conv_def456uvw",
  "timestamp": "2025-10-15T14:30:00Z",
  "sender": {
    "agent_id": "agent_microsoft_abc123",
    "framework": "microsoft-agent-framework",
    "version": "1.0.0",
    "endpoint": "https://api.contoso.com/agents/abc123"
  },
  "recipient": {
    "agent_id": "agent_openai_xyz789",
    "framework": "openai-sdk",
    "endpoint": "https://api.partner.com/agents/xyz789"
  },
  "authentication": {
    "method": "oauth2",
    "token": "eyJhbGciOiJSUzI1NiIs..."
  },
  "payload": {
    "type": "request",
    "action": "query",
    "content": "What is the current inventory level for product SKU-12345?",
    "context": {
      "user_id": "user_123",
      "session_id": "sess_456"
    },
    "metadata": {
      "priority": "high",
      "timeout_seconds": 30
    }
  }
}
```

### Protocol Components

#### **1. Message Envelope**
- `protocol`: Protocol version identifier (e.g., "a2a/1.0")
- `message_id`: Unique identifier for message tracking
- `conversation_id`: Groups related messages in a conversation thread
- `timestamp`: ISO 8601 formatted timestamp

#### **2. Sender/Recipient Information**
- `agent_id`: Unique identifier for the agent
- `framework`: Framework identifier (microsoft-agent-framework, openai-sdk, etc.)
- `version`: Agent or framework version
- `endpoint`: API endpoint for agent communication

#### **3. Authentication**
- `method`: Authentication method (oauth2, api-key, mutual-tls, entra-id)
- `token`: Authentication credential
- `claims`: Additional authentication claims

#### **4. Payload**
- `type`: Message type (request, response, event, error)
- `action`: Specific action requested (query, command, notify)
- `content`: Message content (text, structured data)
- `context`: Contextual information
- `metadata`: Additional metadata for routing, priority, etc.

---

## Cross-Framework Interoperability

### Supported Frameworks

The A2A protocol enables interoperability with:

1. **OpenAI SDK** - OpenAI Assistants API
2. **Claude SDK (Anthropic)** - Claude agent implementations
3. **LangGraph** - LangChain's graph-based agent framework
4. **Google ADK** - Google Agent Development Kit
5. **Custom Frameworks** - Any framework implementing A2A protocol

### Framework Registration

#### **Microsoft Agent Framework Registration**

```python
from agent_framework import ChatAgent
from agent_framework.a2a import A2AProtocolAdapter
from azure.identity.aio import DefaultAzureCredential

async def register_a2a_agent():
    """Register agent for A2A communication"""
    async with DefaultAzureCredential() as credential:
        # Create agent
        agent = ChatAgent(
            instructions="You are a sales data agent.",
            name="SalesDataAgent"
        )

        # Create A2A adapter
        a2a_adapter = A2AProtocolAdapter(
            agent=agent,
            agent_id="agent_sales_001",
            endpoint="https://api.contoso.com/agents/sales",
            authentication={
                "method": "entra-id",
                "credential": credential
            }
        )

        # Register with A2A registry
        await a2a_adapter.register()

        return a2a_adapter
```

#### **.NET A2A Registration**

```csharp
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.A2A;
using Azure.Identity;

public async Task<A2AProtocolAdapter> RegisterA2AAgent()
{
    // Create agent
    var agent = new ChatAgent(
        instructions: "You are an inventory management agent.",
        name: "InventoryAgent"
    );

    // Create A2A adapter
    var a2aAdapter = new A2AProtocolAdapter(
        agent: agent,
        agentId: "agent_inventory_001",
        endpoint: new Uri("https://api.contoso.com/agents/inventory"),
        authentication: new A2AAuthentication
        {
            Method = A2AAuthMethod.EntraId,
            Credential = new DefaultAzureCredential()
        }
    );

    // Register with A2A registry
    await a2aAdapter.RegisterAsync();

    return a2aAdapter;
}
```

### Cross-Framework Communication Example

#### **Microsoft Agent → OpenAI Agent**

```python
import asyncio
from agent_framework.a2a import A2AClient, A2AMessage

async def communicate_with_openai_agent():
    """Microsoft agent calls OpenAI agent"""

    # Create A2A client
    client = A2AClient(
        agent_id="agent_microsoft_001",
        authentication={
            "method": "oauth2",
            "client_id": "your_client_id",
            "client_secret": "your_client_secret"
        }
    )

    # Create message
    message = A2AMessage(
        recipient_agent_id="agent_openai_002",
        recipient_endpoint="https://api.partner.com/agents/openai/002",
        payload={
            "type": "request",
            "action": "query",
            "content": "Analyze this customer feedback and provide sentiment score",
            "context": {
                "feedback": "The product quality is excellent but delivery was slow."
            }
        }
    )

    # Send message and await response
    response = await client.send_message(message)

    print(f"Response from OpenAI agent: {response.payload['content']}")
    print(f"Sentiment score: {response.payload.get('metadata', {}).get('sentiment_score')}")
```

#### **LangGraph Agent → Microsoft Agent**

```python
from langgraph.prebuilt import create_agent
from agent_framework.a2a import A2AClient, A2AMessage

async def langgraph_to_microsoft():
    """LangGraph agent calls Microsoft agent"""

    # Create A2A client for LangGraph agent
    a2a_client = A2AClient(
        agent_id="agent_langgraph_003",
        framework="langgraph",
        authentication={
            "method": "api-key",
            "api_key": "your_api_key"
        }
    )

    # Call Microsoft agent
    message = A2AMessage(
        recipient_agent_id="agent_microsoft_sales_001",
        recipient_endpoint="https://api.contoso.com/agents/sales",
        recipient_framework="microsoft-agent-framework",
        payload={
            "type": "request",
            "action": "query",
            "content": "Get quarterly sales figures for Q3 2025",
            "metadata": {
                "format": "json",
                "currency": "USD"
            }
        }
    )

    response = await a2a_client.send_message(message)
    sales_data = response.payload['content']

    return sales_data
```

---

## Authentication Between Agents

### Authentication Methods

#### **1. OAuth 2.0 Client Credentials Flow**

```python
from agent_framework.a2a import A2AClient, OAuth2Authentication

# Configure OAuth2 authentication
auth = OAuth2Authentication(
    token_endpoint="https://login.microsoftonline.com/tenant-id/oauth2/v2.0/token",
    client_id="your_client_id",
    client_secret="your_client_secret",
    scope="https://api.contoso.com/.default"
)

client = A2AClient(
    agent_id="agent_001",
    authentication=auth
)
```

#### **2. Microsoft Entra ID (Azure AD)**

```python
from azure.identity.aio import DefaultAzureCredential
from agent_framework.a2a import A2AClient, EntraIdAuthentication

# Use Entra ID authentication
auth = EntraIdAuthentication(
    credential=DefaultAzureCredential(),
    tenant_id="your_tenant_id",
    scope="https://api.contoso.com/.default"
)

client = A2AClient(
    agent_id="agent_001",
    authentication=auth
)
```

#### **3. Mutual TLS (mTLS)**

```python
from agent_framework.a2a import A2AClient, MutualTLSAuthentication

# Configure mTLS
auth = MutualTLSAuthentication(
    client_cert_path="/path/to/client-cert.pem",
    client_key_path="/path/to/client-key.pem",
    ca_cert_path="/path/to/ca-cert.pem"
)

client = A2AClient(
    agent_id="agent_001",
    authentication=auth
)
```

#### **4. API Key Authentication**

```python
from agent_framework.a2a import A2AClient, ApiKeyAuthentication

# Simple API key authentication
auth = ApiKeyAuthentication(
    api_key="your_api_key",
    header_name="X-Agent-API-Key"
)

client = A2AClient(
    agent_id="agent_001",
    authentication=auth
)
```

### Authorization and Access Control

```python
from agent_framework.a2a import A2AProtocolAdapter, AccessPolicy

# Define access policy for incoming A2A requests
access_policy = AccessPolicy(
    allowed_agents=[
        "agent_partner_001",
        "agent_partner_002"
    ],
    allowed_frameworks=[
        "openai-sdk",
        "langgraph",
        "microsoft-agent-framework"
    ],
    allowed_actions=[
        "query",
        "notify"
    ],
    denied_actions=[
        "admin",
        "delete"
    ],
    rate_limit={
        "requests_per_minute": 60,
        "requests_per_hour": 1000
    }
)

# Apply policy to adapter
adapter = A2AProtocolAdapter(
    agent=agent,
    agent_id="agent_001",
    access_policy=access_policy
)
```

---

## Message Format Specification

### Request Message

```json
{
  "protocol": "a2a/1.0",
  "message_id": "msg_req_001",
  "conversation_id": "conv_001",
  "timestamp": "2025-10-15T14:30:00Z",
  "sender": {
    "agent_id": "agent_sender_001",
    "framework": "microsoft-agent-framework",
    "endpoint": "https://api.sender.com/agents/001"
  },
  "recipient": {
    "agent_id": "agent_recipient_002",
    "framework": "openai-sdk",
    "endpoint": "https://api.recipient.com/agents/002"
  },
  "authentication": {
    "method": "oauth2",
    "token": "eyJhbGciOiJSUzI1NiIs..."
  },
  "payload": {
    "type": "request",
    "action": "query",
    "content": "What is the status of order #12345?",
    "context": {
      "user_id": "user_789",
      "order_id": "12345"
    },
    "metadata": {
      "priority": "high",
      "timeout_seconds": 30,
      "require_acknowledgment": true
    }
  }
}
```

### Response Message

```json
{
  "protocol": "a2a/1.0",
  "message_id": "msg_resp_002",
  "conversation_id": "conv_001",
  "in_reply_to": "msg_req_001",
  "timestamp": "2025-10-15T14:30:05Z",
  "sender": {
    "agent_id": "agent_recipient_002",
    "framework": "openai-sdk",
    "endpoint": "https://api.recipient.com/agents/002"
  },
  "recipient": {
    "agent_id": "agent_sender_001",
    "framework": "microsoft-agent-framework",
    "endpoint": "https://api.sender.com/agents/001"
  },
  "payload": {
    "type": "response",
    "status": "success",
    "content": "Order #12345 is currently in transit. Expected delivery: Oct 20, 2025.",
    "data": {
      "order_id": "12345",
      "status": "in_transit",
      "tracking_number": "TRK789XYZ",
      "expected_delivery": "2025-10-20",
      "current_location": "Distribution Center, Chicago"
    },
    "metadata": {
      "processing_time_ms": 1250
    }
  }
}
```

### Event/Notification Message

```json
{
  "protocol": "a2a/1.0",
  "message_id": "msg_event_003",
  "timestamp": "2025-10-15T14:35:00Z",
  "sender": {
    "agent_id": "agent_monitor_003",
    "framework": "microsoft-agent-framework"
  },
  "payload": {
    "type": "event",
    "event_type": "status_change",
    "content": "Order #12345 has been delivered",
    "data": {
      "order_id": "12345",
      "previous_status": "in_transit",
      "new_status": "delivered",
      "delivered_at": "2025-10-15T14:30:00Z"
    }
  }
}
```

### Error Message

```json
{
  "protocol": "a2a/1.0",
  "message_id": "msg_error_004",
  "conversation_id": "conv_001",
  "in_reply_to": "msg_req_001",
  "timestamp": "2025-10-15T14:30:02Z",
  "sender": {
    "agent_id": "agent_recipient_002",
    "framework": "openai-sdk"
  },
  "payload": {
    "type": "error",
    "error_code": "AUTH_FAILED",
    "error_message": "Authentication token has expired",
    "error_details": {
      "expired_at": "2025-10-15T14:00:00Z",
      "retry_after_seconds": 60
    },
    "metadata": {
      "recoverable": true
    }
  }
}
```

---

## Implementation Examples

### Complete Python A2A Implementation

```python
import asyncio
from typing import Dict, Any, Optional
from agent_framework import ChatAgent
from agent_framework.a2a import (
    A2AProtocolAdapter,
    A2AClient,
    A2AMessage,
    A2AMessageHandler
)
from azure.identity.aio import DefaultAzureCredential

class CustomerServiceA2ASystem:
    """Complete A2A implementation for customer service"""

    def __init__(self):
        self.credential = None
        self.agents: Dict[str, A2AProtocolAdapter] = {}
        self.clients: Dict[str, A2AClient] = {}

    async def __aenter__(self):
        self.credential = DefaultAzureCredential()
        await self._setup_agents()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup
        for adapter in self.agents.values():
            await adapter.unregister()
        if self.credential:
            await self.credential.close()

    async def _setup_agents(self):
        """Setup internal agents with A2A capability"""

        # Order status agent
        order_agent = ChatAgent(
            instructions="You handle order status queries. Provide accurate, real-time order information.",
            name="OrderStatusAgent"
        )

        order_adapter = A2AProtocolAdapter(
            agent=order_agent,
            agent_id="agent_order_status",
            endpoint="https://api.contoso.com/agents/order-status",
            authentication={
                "method": "entra-id",
                "credential": self.credential
            }
        )

        # Register message handler
        @order_adapter.on_message
        async def handle_order_query(message: A2AMessage) -> Dict[str, Any]:
            """Handle incoming order queries"""
            order_id = message.payload.get('context', {}).get('order_id')

            # Query order database (simulated)
            order_info = await self._get_order_info(order_id)

            return {
                "type": "response",
                "status": "success",
                "content": f"Order {order_id}: {order_info['status']}",
                "data": order_info
            }

        await order_adapter.register()
        self.agents["order_status"] = order_adapter

        # Customer data agent
        customer_agent = ChatAgent(
            instructions="You handle customer data queries with privacy compliance.",
            name="CustomerDataAgent"
        )

        customer_adapter = A2AProtocolAdapter(
            agent=customer_agent,
            agent_id="agent_customer_data",
            endpoint="https://api.contoso.com/agents/customer-data",
            authentication={
                "method": "entra-id",
                "credential": self.credential
            }
        )

        await customer_adapter.register()
        self.agents["customer_data"] = customer_adapter

    async def _get_order_info(self, order_id: str) -> Dict[str, Any]:
        """Simulated order database query"""
        return {
            "order_id": order_id,
            "status": "in_transit",
            "tracking": "TRK123XYZ",
            "expected_delivery": "2025-10-20"
        }

    async def call_external_agent(
        self,
        agent_id: str,
        endpoint: str,
        framework: str,
        message: str,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Call an external agent via A2A protocol"""

        # Create or get A2A client
        client_key = f"{framework}_{agent_id}"
        if client_key not in self.clients:
            self.clients[client_key] = A2AClient(
                agent_id="agent_order_status",
                authentication={
                    "method": "oauth2",
                    "client_id": "your_client_id",
                    "client_secret": "your_client_secret"
                }
            )

        client = self.clients[client_key]

        # Create A2A message
        a2a_message = A2AMessage(
            recipient_agent_id=agent_id,
            recipient_endpoint=endpoint,
            recipient_framework=framework,
            payload={
                "type": "request",
                "action": "query",
                "content": message,
                "context": context or {}
            }
        )

        # Send and await response
        response = await client.send_message(a2a_message)
        return response.payload

# Usage
async def main():
    async with CustomerServiceA2ASystem() as system:
        # Internal agent communication works automatically

        # Call external OpenAI agent
        response = await system.call_external_agent(
            agent_id="agent_openai_sentiment",
            endpoint="https://api.partner.com/agents/sentiment",
            framework="openai-sdk",
            message="Analyze sentiment: 'Great product but slow shipping'",
            context={"customer_id": "cust_123"}
        )

        print(f"Sentiment analysis: {response['content']}")

        # Call LangGraph agent for complex reasoning
        response = await system.call_external_agent(
            agent_id="agent_langgraph_recommender",
            endpoint="https://api.partner.com/agents/recommender",
            framework="langgraph",
            message="Recommend products based on customer history",
            context={"customer_id": "cust_123", "category": "electronics"}
        )

        print(f"Recommendations: {response['data']}")

if __name__ == "__main__":
    asyncio.run(main())
```

### Complete .NET A2A Implementation

```csharp
using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.A2A;
using Azure.Identity;

public class EnterpriseA2ASystem : IAsyncDisposable
{
    private readonly Dictionary<string, A2AProtocolAdapter> _agents = new();
    private readonly Dictionary<string, A2AClient> _clients = new();
    private readonly DefaultAzureCredential _credential;

    public EnterpriseA2ASystem()
    {
        _credential = new DefaultAzureCredential();
    }

    public async Task InitializeAsync()
    {
        await SetupAgentsAsync();
    }

    private async Task SetupAgentsAsync()
    {
        // Inventory agent
        var inventoryAgent = new ChatAgent(
            instructions: "You manage inventory queries and updates.",
            name: "InventoryAgent"
        );

        var inventoryAdapter = new A2AProtocolAdapter(
            agent: inventoryAgent,
            agentId: "agent_inventory_001",
            endpoint: new Uri("https://api.contoso.com/agents/inventory"),
            authentication: new A2AAuthentication
            {
                Method = A2AAuthMethod.EntraId,
                Credential = _credential
            }
        );

        // Register message handler
        inventoryAdapter.OnMessage(async (message) =>
        {
            var sku = message.Payload.Context?["sku"]?.ToString();
            var inventoryLevel = await GetInventoryLevelAsync(sku);

            return new A2AResponse
            {
                Type = "response",
                Status = "success",
                Content = $"SKU {sku}: {inventoryLevel} units in stock",
                Data = new Dictionary<string, object>
                {
                    ["sku"] = sku,
                    ["quantity"] = inventoryLevel,
                    ["warehouse"] = "Central Warehouse"
                }
            };
        });

        await inventoryAdapter.RegisterAsync();
        _agents["inventory"] = inventoryAdapter;

        // Pricing agent
        var pricingAgent = new ChatAgent(
            instructions: "You handle pricing queries and discount calculations.",
            name: "PricingAgent"
        );

        var pricingAdapter = new A2AProtocolAdapter(
            agent: pricingAgent,
            agentId: "agent_pricing_002",
            endpoint: new Uri("https://api.contoso.com/agents/pricing"),
            authentication: new A2AAuthentication
            {
                Method = A2AAuthMethod.EntraId,
                Credential = _credential
            }
        );

        await pricingAdapter.RegisterAsync();
        _agents["pricing"] = pricingAdapter;
    }

    private async Task<int> GetInventoryLevelAsync(string sku)
    {
        // Simulated inventory query
        await Task.Delay(100);
        return new Random().Next(0, 1000);
    }

    public async Task<A2AResponse> CallExternalAgentAsync(
        string agentId,
        Uri endpoint,
        string framework,
        string message,
        Dictionary<string, object> context = null)
    {
        // Create or get A2A client
        var clientKey = $"{framework}_{agentId}";
        if (!_clients.ContainsKey(clientKey))
        {
            _clients[clientKey] = new A2AClient(
                agentId: "agent_inventory_001",
                authentication: new A2AAuthentication
                {
                    Method = A2AAuthMethod.OAuth2,
                    ClientId = "your_client_id",
                    ClientSecret = "your_client_secret"
                }
            );
        }

        var client = _clients[clientKey];

        // Create A2A message
        var a2aMessage = new A2AMessage
        {
            RecipientAgentId = agentId,
            RecipientEndpoint = endpoint,
            RecipientFramework = framework,
            Payload = new A2APayload
            {
                Type = "request",
                Action = "query",
                Content = message,
                Context = context ?? new Dictionary<string, object>()
            }
        };

        // Send and await response
        var response = await client.SendMessageAsync(a2aMessage);
        return response;
    }

    public async ValueTask DisposeAsync()
    {
        foreach (var adapter in _agents.Values)
        {
            await adapter.UnregisterAsync();
        }
        _credential?.Dispose();
    }
}

// Usage
public class Program
{
    public static async Task Main(string[] args)
    {
        await using var system = new EnterpriseA2ASystem();
        await system.InitializeAsync();

        // Call external Claude agent
        var claudeResponse = await system.CallExternalAgentAsync(
            agentId: "agent_claude_analytics",
            endpoint: new Uri("https://api.partner.com/agents/claude/analytics"),
            framework: "claude-sdk",
            message: "Analyze this sales data and provide insights",
            context: new Dictionary<string, object>
            {
                ["data_source"] = "Q3_2025_sales.csv",
                ["focus_area"] = "revenue_trends"
            }
        );

        Console.WriteLine($"Claude Analysis: {claudeResponse.Content}");

        // Call Google ADK agent
        var googleResponse = await system.CallExternalAgentAsync(
            agentId: "agent_google_forecasting",
            endpoint: new Uri("https://api.partner.com/agents/google/forecast"),
            framework: "google-adk",
            message: "Forecast Q4 2025 inventory needs",
            context: new Dictionary<string, object>
            {
                ["historical_quarters"] = 8,
                ["seasonality"] = "high"
            }
        );

        Console.WriteLine($"Forecast: {googleResponse.Data}");
    }
}
```

---

## Security Considerations

### End-to-End Encryption

```python
from agent_framework.a2a import A2AClient, EncryptionConfig

# Configure end-to-end encryption
encryption_config = EncryptionConfig(
    algorithm="AES-256-GCM",
    key_exchange="ECDH",
    public_key_path="/path/to/public-key.pem",
    private_key_path="/path/to/private-key.pem"
)

client = A2AClient(
    agent_id="agent_001",
    encryption=encryption_config
)
```

### Message Signing and Verification

```python
from agent_framework.a2a import A2AProtocolAdapter, SigningConfig

# Configure message signing
signing_config = SigningConfig(
    algorithm="RS256",
    private_key_path="/path/to/signing-key.pem",
    certificate_path="/path/to/cert.pem"
)

adapter = A2AProtocolAdapter(
    agent=agent,
    agent_id="agent_001",
    signing=signing_config,
    verify_signatures=True  # Verify incoming message signatures
)
```

### Rate Limiting and Throttling

```python
from agent_framework.a2a import A2AProtocolAdapter, RateLimitConfig

# Configure rate limiting
rate_limit = RateLimitConfig(
    requests_per_minute=60,
    requests_per_hour=1000,
    burst_size=10,
    strategy="token_bucket"
)

adapter = A2AProtocolAdapter(
    agent=agent,
    agent_id="agent_001",
    rate_limit=rate_limit
)
```

---

## Production Patterns

### Circuit Breaker Pattern

```python
from agent_framework.a2a import A2AClient, CircuitBreakerConfig

# Configure circuit breaker for resilient communication
circuit_breaker = CircuitBreakerConfig(
    failure_threshold=5,
    timeout_seconds=30,
    reset_timeout_seconds=60
)

client = A2AClient(
    agent_id="agent_001",
    circuit_breaker=circuit_breaker
)
```

### Message Queue Integration

```python
from agent_framework.a2a import A2AProtocolAdapter
from azure.servicebus.aio import ServiceBusClient

# Integrate with Azure Service Bus for async messaging
async def setup_async_a2a():
    service_bus_client = ServiceBusClient.from_connection_string(
        conn_str="your_connection_string"
    )

    adapter = A2AProtocolAdapter(
        agent=agent,
        agent_id="agent_001",
        message_queue=service_bus_client,
        queue_name="a2a-messages"
    )

    await adapter.register()
```

### Monitoring and Observability

```python
from agent_framework.a2a import A2AProtocolAdapter
from opentelemetry import trace

# Configure OpenTelemetry tracing
tracer = trace.get_tracer(__name__)

adapter = A2AProtocolAdapter(
    agent=agent,
    agent_id="agent_001",
    telemetry={
        "enabled": True,
        "tracer": tracer,
        "export_endpoint": "https://otel-collector.contoso.com"
    }
)

# All A2A messages will be traced automatically
```

---

## Best Practices

### 1. Design for Failure
- Implement timeouts for all cross-agent communications
- Use circuit breakers to prevent cascading failures
- Implement retry logic with exponential backoff

### 2. Security First
- Always use authenticated communications
- Encrypt sensitive payload data
- Implement strict access control policies
- Validate and sanitize all incoming messages

### 3. Monitoring and Logging
- Log all A2A messages for audit trails
- Monitor message latency and failure rates
- Set up alerts for authentication failures
- Track cross-framework communication patterns

### 4. Version Management
- Include protocol version in all messages
- Support multiple protocol versions during transitions
- Document breaking changes clearly

### 5. Performance Optimization
- Use message batching for high-volume scenarios
- Implement caching for frequently accessed data
- Consider async messaging for non-critical paths
- Use compression for large payloads

---

## Migration from Existing Solutions

### From Direct API Calls

```python
# Before: Direct API calls
response = requests.post(
    "https://api.partner.com/query",
    json={"query": "Get data"},
    headers={"Authorization": "Bearer token"}
)

# After: A2A Protocol
from agent_framework.a2a import A2AClient, A2AMessage

client = A2AClient(agent_id="agent_001")
response = await client.send_message(A2AMessage(
    recipient_endpoint="https://api.partner.com/agents/query",
    payload={"type": "request", "content": "Get data"}
))
```

### Benefits of Migration

1. **Standardized Communication:** Common format across all agents
2. **Built-in Authentication:** OAuth2, Entra ID, mTLS support
3. **Better Observability:** Automatic tracing and monitoring
4. **Framework Agnostic:** Works with any agent framework
5. **Enterprise Features:** Rate limiting, circuit breakers, encryption

---

## Future Roadmap

The A2A protocol is actively being developed. Upcoming features include:

- **A2A Registry Service:** Centralized agent discovery and registration
- **Protocol Extensions:** Support for streaming, binary data, and attachments
- **Enhanced Security:** Zero-trust architecture, hardware security module integration
- **Performance Improvements:** HTTP/3, QUIC protocol support
- **Federation:** Cross-organization agent collaboration with trust frameworks

---

## Support and Resources

### Documentation
- [A2A Protocol Specification](https://github.com/microsoft/agent-framework/blob/main/docs/a2a-protocol.md)
- [API Reference](https://learn.microsoft.com/en-us/agent-framework/api/a2a)
- [Migration Guide](https://learn.microsoft.com/en-us/agent-framework/migration/a2a)

### Community
- [GitHub Discussions](https://github.com/microsoft/agent-framework/discussions)
- [Stack Overflow](https://stackoverflow.com/questions/tagged/microsoft-agent-framework)
- [Microsoft Q&A](https://docs.microsoft.com/answers/tags/azure-agent-framework)

### Getting Help
- Report issues: [GitHub Issues](https://github.com/microsoft/agent-framework/issues)
- Feature requests: [GitHub Discussions](https://github.com/microsoft/agent-framework/discussions/categories/ideas)
- Security issues: security@microsoft.com

---

**Last Updated:** November 2025
**Document Version:** 1.0
**Preview Status:** October 2025 Public Preview
**Next Review:** Q2 2026
