# Mistral Agents API: Model Context Protocol (MCP) Guide

**Version**: 2.0 (May 2025 Launch Edition)
**Last Updated**: May 27, 2025

## Overview

The May 27, 2025 Agents API launch includes **Model Context Protocol (MCP)** support, implementing Anthropic's standardized protocol for connecting AI agents to third-party tools and data sources. MCP enables seamless integration with external systems through a unified interface.

## Table of Contents

1. [MCP Architecture](#mcp-architecture)
2. [Anthropic MCP Implementation](#anthropic-mcp-implementation)
3. [Custom MCP Servers](#custom-mcp-servers)
4. [Tool Exposure Patterns](#tool-exposure-patterns)
5. [Integration Best Practices](#integration-best-practices)
6. [Security and Authentication](#security-and-authentication)
7. [Debugging MCP Connections](#debugging-mcp-connections)
8. [Production Deployment](#production-deployment)

---

## MCP Architecture

### What is MCP?

Model Context Protocol (MCP) is an **open standard** created by Anthropic for connecting AI assistants to external data sources and tools. Think of it as USB for AI—a universal connector that works across different systems.

### Key Concepts

#### 1. MCP Server

An MCP server exposes tools and resources to AI agents:

```
┌─────────────────────────────────────┐
│         MCP Server                  │
├─────────────────────────────────────┤
│                                     │
│  ┌─────────────────────────────┐   │
│  │  Tool: database_query       │   │
│  │  Tool: file_read            │   │
│  │  Tool: api_call             │   │
│  │  Resource: /docs/*          │   │
│  │  Resource: /data/*          │   │
│  └─────────────────────────────┘   │
│                                     │
│  ┌─────────────────────────────┐   │
│  │  Authentication             │   │
│  │  Rate Limiting              │   │
│  │  Logging                    │   │
│  └─────────────────────────────┘   │
│                                     │
└─────────────────────────────────────┘
         ▲
         │ MCP Protocol
         │
         ▼
┌─────────────────────────────────────┐
│     Mistral Agent (MCP Client)      │
├─────────────────────────────────────┤
│  - Discovers available tools        │
│  - Invokes tools via MCP            │
│  - Receives results                 │
└─────────────────────────────────────┘
```

#### 2. MCP Client

The Mistral agent acts as an MCP client, discovering and invoking tools from MCP servers.

#### 3. Transport Layer

MCP supports multiple transport mechanisms:
- **HTTP/HTTPS**: Standard web protocol
- **WebSocket**: Real-time bidirectional communication
- **stdio**: Standard input/output for local processes

### MCP vs Built-in Connectors

| Feature | Built-in Connectors | MCP |
|---------|-------------------|-----|
| **Setup** | Zero configuration | Requires MCP server setup |
| **Maintenance** | Managed by Mistral | Self-managed |
| **Use Cases** | General purpose (web search, code execution) | Custom integrations (your database, APIs) |
| **Security** | Mistral-managed sandbox | Your security implementation |
| **Best For** | Common tasks | Custom business logic |

**Use both together:** Built-in connectors for standard tasks, MCP for custom integrations.

---

## Anthropic MCP Implementation

### Mistral's MCP Support

Mistral Agents API implements **Anthropic's MCP specification**, ensuring compatibility with the broader MCP ecosystem.

### Supported MCP Features

```python
# Mistral supports these MCP capabilities:

✅ Tool Discovery         # Automatic tool enumeration
✅ Tool Invocation        # Remote tool execution
✅ Resource Access        # Read external resources
✅ JSON-RPC 2.0          # Standard protocol
✅ HTTP/HTTPS Transport  # Web-based connectivity
✅ Authentication        # Bearer tokens, API keys
✅ Error Handling        # Standardized error responses
✅ Streaming             # Progressive results (where supported)
```

### Configuration

```python
import os
from mistralai import Mistral

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

# Create agent with MCP server connection
agent = client.beta.agents.create(
    model="mistral-medium-latest",
    name="MCP-Enabled Agent",
    description="Agent connected to custom MCP server",
    instructions="""
    You have access to custom tools via MCP.
    Available tools: database_query, send_email, create_ticket
    Use these tools when appropriate for user requests.
    """,
    tools=[
        {
            "type": "mcp",
            "config": {
                "server_url": "https://mcp.yourcompany.com",
                "auth": {
                    "type": "bearer",
                    "token": os.environ["MCP_SERVER_TOKEN"]
                },
                "timeout": 30
            }
        }
    ]
)
```

### Tool Discovery

When an agent connects to an MCP server, it automatically discovers available tools:

```python
# Agent connects to MCP server
# Server responds with tool definitions:

{
  "tools": [
    {
      "name": "database_query",
      "description": "Query the company database",
      "input_schema": {
        "type": "object",
        "properties": {
          "query": {"type": "string", "description": "SQL query"},
          "database": {"type": "string", "enum": ["users", "products", "orders"]}
        },
        "required": ["query", "database"]
      }
    },
    {
      "name": "send_email",
      "description": "Send email via company mail server",
      "input_schema": {
        "type": "object",
        "properties": {
          "to": {"type": "string"},
          "subject": {"type": "string"},
          "body": {"type": "string"}
        },
        "required": ["to", "subject", "body"]
      }
    }
  ]
}

# Agent can now use these tools automatically
```

### Using MCP Tools

```python
# Once configured, agents use MCP tools like built-in tools

conversation = client.beta.conversations.start(
    agent_id=agent.id,
    inputs="Query the database for all users created in the last 7 days"
)

# Agent will:
# 1. Recognize this requires database_query tool
# 2. Construct appropriate SQL query
# 3. Call MCP server's database_query tool
# 4. Receive results
# 5. Present results to user
```

---

## Custom MCP Servers

### Building an MCP Server

Here's a complete MCP server implementation:

```python
# mcp_server.py - Custom MCP Server Example

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import sqlite3

app = FastAPI(title="Custom MCP Server")

# Authentication
MCP_SECRET_TOKEN = os.environ["MCP_SECRET_TOKEN"]


def verify_token(authorization: str = Header(None)):
    """Verify MCP client authentication"""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")

    token = authorization.split(" ")[1]
    if token != MCP_SECRET_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")


# Tool definitions
class ToolDefinition(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any]


class ToolInvocation(BaseModel):
    name: str
    arguments: Dict[str, Any]


class ToolResult(BaseModel):
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None


# MCP Endpoints

@app.get("/mcp/v1/tools")
async def list_tools(authorization: str = Header(None)) -> Dict[str, List[ToolDefinition]]:
    """MCP tool discovery endpoint"""
    verify_token(authorization)

    return {
        "tools": [
            {
                "name": "database_query",
                "description": "Query SQLite database",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "SQL SELECT query"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "send_notification",
                "description": "Send notification to user",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                        "message": {"type": "string"}
                    },
                    "required": ["user_id", "message"]
                }
            },
            {
                "name": "create_ticket",
                "description": "Create support ticket",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "priority": {
                            "type": "string",
                            "enum": ["low", "medium", "high", "urgent"]
                        }
                    },
                    "required": ["title", "description"]
                }
            }
        ]
    }


@app.post("/mcp/v1/tools/invoke")
async def invoke_tool(
    invocation: ToolInvocation,
    authorization: str = Header(None)
) -> ToolResult:
    """MCP tool invocation endpoint"""
    verify_token(authorization)

    try:
        if invocation.name == "database_query":
            result = execute_database_query(invocation.arguments)
            return ToolResult(success=True, result=result)

        elif invocation.name == "send_notification":
            result = send_notification(invocation.arguments)
            return ToolResult(success=True, result=result)

        elif invocation.name == "create_ticket":
            result = create_ticket(invocation.arguments)
            return ToolResult(success=True, result=result)

        else:
            return ToolResult(
                success=False,
                error=f"Unknown tool: {invocation.name}"
            )

    except Exception as e:
        return ToolResult(success=False, error=str(e))


# Tool implementations

def execute_database_query(args: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Execute database query"""
    query = args["query"]

    # Validate query (security!)
    if not query.strip().upper().startswith("SELECT"):
        raise ValueError("Only SELECT queries allowed")

    # Connect to database
    conn = sqlite3.connect("company.db")
    cursor = conn.cursor()

    try:
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        results = [
            dict(zip(columns, row))
            for row in rows
        ]

        return results

    finally:
        conn.close()


def send_notification(args: Dict[str, Any]) -> Dict[str, str]:
    """Send notification to user"""
    user_id = args["user_id"]
    message = args["message"]

    # In production: integrate with notification system
    print(f"Sending to {user_id}: {message}")

    return {
        "status": "sent",
        "user_id": user_id,
        "timestamp": "2025-05-27T10:00:00Z"
    }


def create_ticket(args: Dict[str, Any]) -> Dict[str, Any]:
    """Create support ticket"""
    title = args["title"]
    description = args["description"]
    priority = args.get("priority", "medium")

    # In production: integrate with ticketing system
    ticket_id = f"TICK-{hash(title) % 10000:04d}"

    return {
        "ticket_id": ticket_id,
        "title": title,
        "priority": priority,
        "status": "open",
        "created_at": "2025-05-27T10:00:00Z"
    }


# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Deploying MCP Server

```bash
# Install dependencies
pip install fastapi uvicorn pydantic

# Run server
export MCP_SECRET_TOKEN="your-secret-token"
uvicorn mcp_server:app --host 0.0.0.0 --port 8000

# Test tool discovery
curl -H "Authorization: Bearer your-secret-token" \
     http://localhost:8000/mcp/v1/tools

# Test tool invocation
curl -X POST \
     -H "Authorization: Bearer your-secret-token" \
     -H "Content-Type: application/json" \
     -d '{"name":"database_query","arguments":{"query":"SELECT * FROM users LIMIT 5"}}' \
     http://localhost:8000/mcp/v1/tools/invoke
```

### Connecting Agent to Custom Server

```python
# Connect Mistral agent to your MCP server

agent = client.beta.agents.create(
    model="mistral-large-latest",
    name="Company Agent",
    description="Agent with access to company systems",
    instructions="""
    You have access to company tools via MCP:
    - database_query: Query company database
    - send_notification: Send notifications to users
    - create_ticket: Create support tickets

    Use these tools to help with:
    - Looking up customer information
    - Notifying users about updates
    - Creating tickets for issues

    Always verify information before taking action.
    """,
    tools=[
        {
            "type": "mcp",
            "config": {
                "server_url": "https://mcp.yourcompany.com",
                "auth": {
                    "type": "bearer",
                    "token": os.environ["MCP_SERVER_TOKEN"]
                }
            }
        }
    ]
)

# Use the agent
conversation = client.beta.conversations.start(
    agent_id=agent.id,
    inputs="Find all customers from California and send them a notification about our new product"
)

# Agent will:
# 1. Use database_query to find California customers
# 2. Use send_notification for each customer
# 3. Report results
```

---

## Tool Exposure Patterns

### Pattern: Database Access

```python
# Expose safe database access via MCP

@app.post("/mcp/v1/tools/invoke")
async def invoke_tool(invocation: ToolInvocation, authorization: str = Header(None)):
    """MCP tool invocation with security"""

    if invocation.name == "database_query":
        query = invocation.arguments["query"]

        # Security: Whitelist approach
        if not is_safe_query(query):
            return ToolResult(
                success=False,
                error="Query not allowed. Only SELECT on allowed tables."
            )

        # Security: Read-only user
        conn = get_readonly_db_connection()

        try:
            results = execute_query(conn, query)
            return ToolResult(success=True, result=results)
        except Exception as e:
            return ToolResult(success=False, error=str(e))


def is_safe_query(query: str) -> bool:
    """Validate query safety"""
    query_upper = query.upper().strip()

    # Only SELECT
    if not query_upper.startswith("SELECT"):
        return False

    # No dangerous operations
    dangerous = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "CREATE"]
    if any(keyword in query_upper for keyword in dangerous):
        return False

    # Only allowed tables
    allowed_tables = ["users", "products", "orders"]
    # Parse and validate table names (simplified)
    # In production: use SQL parser

    return True
```

### Pattern: API Gateway

```python
# Expose internal APIs via MCP

class APIGatewayMCP:
    """MCP server as API gateway"""

    def __init__(self):
        self.internal_apis = {
            "crm": "http://internal-crm:8000",
            "billing": "http://internal-billing:8000",
            "analytics": "http://internal-analytics:8000"
        }

    async def invoke_api(self, api_name: str, endpoint: str, method: str, data: Dict):
        """Invoke internal API"""
        base_url = self.internal_apis.get(api_name)
        if not base_url:
            raise ValueError(f"Unknown API: {api_name}")

        url = f"{base_url}{endpoint}"

        # Make internal API call
        async with httpx.AsyncClient() as client:
            if method == "GET":
                response = await client.get(url, params=data)
            elif method == "POST":
                response = await client.post(url, json=data)
            else:
                raise ValueError(f"Unsupported method: {method}")

            return response.json()


@app.post("/mcp/v1/tools/invoke")
async def invoke_tool(invocation: ToolInvocation, authorization: str = Header(None)):
    gateway = APIGatewayMCP()

    if invocation.name == "crm_lookup":
        result = await gateway.invoke_api(
            api_name="crm",
            endpoint="/customers/search",
            method="POST",
            data=invocation.arguments
        )
        return ToolResult(success=True, result=result)

    elif invocation.name == "billing_invoice":
        result = await gateway.invoke_api(
            api_name="billing",
            endpoint="/invoices/create",
            method="POST",
            data=invocation.arguments
        )
        return ToolResult(success=True, result=result)
```

### Pattern: Multi-Service Orchestration

```python
# MCP server that orchestrates multiple services

@app.post("/mcp/v1/tools/invoke")
async def invoke_tool(invocation: ToolInvocation, authorization: str = Header(None)):
    """Complex multi-service tool"""

    if invocation.name == "onboard_customer":
        # This tool orchestrates multiple services

        customer_data = invocation.arguments

        try:
            # Step 1: Create in CRM
            crm_result = await create_crm_record(customer_data)

            # Step 2: Setup billing
            billing_result = await setup_billing_account(
                customer_id=crm_result["customer_id"],
                plan=customer_data["plan"]
            )

            # Step 3: Send welcome email
            email_result = await send_welcome_email(
                email=customer_data["email"],
                customer_id=crm_result["customer_id"]
            )

            # Step 4: Create initial ticket
            ticket_result = await create_onboarding_ticket(
                customer_id=crm_result["customer_id"]
            )

            return ToolResult(
                success=True,
                result={
                    "customer_id": crm_result["customer_id"],
                    "billing_id": billing_result["account_id"],
                    "welcome_email_sent": email_result["sent"],
                    "onboarding_ticket": ticket_result["ticket_id"]
                }
            )

        except Exception as e:
            # Rollback on failure
            # In production: implement proper transaction handling
            return ToolResult(success=False, error=str(e))
```

---

## Integration Best Practices

### 1. Tool Design

```python
# Good tool design principles

# ✅ GOOD: Clear, focused tools
{
  "name": "get_customer_by_email",
  "description": "Retrieve customer record by email address",
  "input_schema": {
    "type": "object",
    "properties": {
      "email": {"type": "string", "format": "email"}
    },
    "required": ["email"]
  }
}

# ❌ BAD: Vague, multi-purpose tools
{
  "name": "do_customer_stuff",
  "description": "Does various customer operations",
  "input_schema": {
    "type": "object",
    "properties": {
      "action": {"type": "string"},
      "data": {"type": "object"}
    }
  }
}

# Design principles:
# - One tool, one purpose
# - Clear input validation
# - Descriptive names and descriptions
# - Required vs optional parameters explicit
```

### 2. Error Handling

```python
# Robust error handling in MCP server

@app.post("/mcp/v1/tools/invoke")
async def invoke_tool(invocation: ToolInvocation, authorization: str = Header(None)):
    try:
        # Validate inputs
        if invocation.name not in AVAILABLE_TOOLS:
            return ToolResult(
                success=False,
                error=f"Tool not found: {invocation.name}"
            )

        # Validate arguments against schema
        validate_arguments(invocation.name, invocation.arguments)

        # Execute tool
        result = await execute_tool(invocation.name, invocation.arguments)

        return ToolResult(success=True, result=result)

    except ValidationError as e:
        # Input validation errors
        return ToolResult(
            success=False,
            error=f"Invalid arguments: {str(e)}"
        )

    except TimeoutError as e:
        # Timeout errors
        return ToolResult(
            success=False,
            error="Tool execution timeout. Please try again."
        )

    except PermissionError as e:
        # Authorization errors
        return ToolResult(
            success=False,
            error="Permission denied for this operation"
        )

    except Exception as e:
        # Unexpected errors
        logger.exception("Tool execution failed")
        return ToolResult(
            success=False,
            error="Internal server error. Please contact support."
        )
```

### 3. Versioning

```python
# Version your MCP server for backward compatibility

@app.get("/mcp/v1/tools")
async def list_tools_v1(authorization: str = Header(None)):
    """Version 1 tool definitions"""
    return {"tools": [...]}


@app.get("/mcp/v2/tools")
async def list_tools_v2(authorization: str = Header(None)):
    """Version 2 with enhanced tools"""
    return {"tools": [...]}


# Configure agent with specific version
agent = client.beta.agents.create(
    model="mistral-medium-latest",
    name="Agent",
    tools=[
        {
            "type": "mcp",
            "config": {
                "server_url": "https://mcp.company.com/v2",  # Explicit version
                "auth": {"type": "bearer", "token": token}
            }
        }
    ]
)
```

### 4. Rate Limiting

```python
# Implement rate limiting in MCP server

from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import redis.asyncio as redis

# Initialize limiter
@app.on_event("startup")
async def startup():
    redis_client = redis.from_url("redis://localhost", encoding="utf-8")
    await FastAPILimiter.init(redis_client)


# Apply rate limits
@app.post("/mcp/v1/tools/invoke")
@limiter.limit("100/minute")  # 100 requests per minute
async def invoke_tool(
    invocation: ToolInvocation,
    authorization: str = Header(None)
):
    # Tool invocation logic
    pass


# Or per-tool rate limiting
@app.post("/mcp/v1/tools/invoke")
async def invoke_tool(invocation: ToolInvocation, authorization: str = Header(None)):
    # Check rate limit based on tool
    rate_limits = {
        "expensive_query": "10/minute",
        "send_email": "50/hour",
        "normal_operation": "1000/minute"
    }

    limit = rate_limits.get(invocation.name, "100/minute")
    # Apply limit...
```

---

## Security and Authentication

### Authentication Methods

```python
# 1. Bearer Token (Recommended)
agent = client.beta.agents.create(
    model="mistral-medium-latest",
    name="Secure Agent",
    tools=[
        {
            "type": "mcp",
            "config": {
                "server_url": "https://mcp.company.com",
                "auth": {
                    "type": "bearer",
                    "token": os.environ["MCP_TOKEN"]
                }
            }
        }
    ]
)


# 2. API Key
agent = client.beta.agents.create(
    model="mistral-medium-latest",
    name="API Key Agent",
    tools=[
        {
            "type": "mcp",
            "config": {
                "server_url": "https://mcp.company.com",
                "auth": {
                    "type": "api_key",
                    "key": os.environ["MCP_API_KEY"],
                    "header": "X-API-Key"  # Custom header name
                }
            }
        }
    ]
)


# 3. OAuth 2.0 (for complex scenarios)
agent = client.beta.agents.create(
    model="mistral-medium-latest",
    name="OAuth Agent",
    tools=[
        {
            "type": "mcp",
            "config": {
                "server_url": "https://mcp.company.com",
                "auth": {
                    "type": "oauth2",
                    "client_id": os.environ["OAUTH_CLIENT_ID"],
                    "client_secret": os.environ["OAUTH_CLIENT_SECRET"],
                    "token_url": "https://auth.company.com/token"
                }
            }
        }
    ]
)
```

### Server-Side Security

```python
# Comprehensive security for MCP server

from fastapi import Security, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import hashlib
import secrets

security = HTTPBearer()


# Token validation
def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify bearer token"""
    token = credentials.credentials

    # Check against stored token (hashed)
    expected_hash = os.environ["MCP_TOKEN_HASH"]
    token_hash = hashlib.sha256(token.encode()).hexdigest()

    if not secrets.compare_digest(token_hash, expected_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token"
        )

    return token


# IP whitelisting
ALLOWED_IPS = [
    "35.123.456.789",  # Mistral Agents API IP ranges
    "52.234.567.890",
]


def verify_ip(request: Request):
    """Verify request comes from allowed IP"""
    client_ip = request.client.host

    if client_ip not in ALLOWED_IPS:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="IP not whitelisted"
        )


# Complete endpoint with security
@app.post("/mcp/v1/tools/invoke")
async def invoke_tool(
    invocation: ToolInvocation,
    request: Request,
    token: str = Depends(verify_token)
):
    # Verify IP
    verify_ip(request)

    # Tool invocation logic
    # ...
```

---

## Production Deployment

### Complete Production Setup

```python
# production_mcp_server.py

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import logging
import os
from prometheus_client import Counter, Histogram, make_asgi_app
import time

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Metrics
tool_invocations = Counter(
    'mcp_tool_invocations_total',
    'Total tool invocations',
    ['tool_name', 'status']
)

tool_duration = Histogram(
    'mcp_tool_duration_seconds',
    'Tool execution duration',
    ['tool_name']
)

# App
app = FastAPI(
    title="Production MCP Server",
    version="1.0.0",
    docs_url="/docs" if os.getenv("ENV") == "development" else None
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://api.mistral.ai"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# Tool invocation with production features
@app.post("/mcp/v1/tools/invoke")
async def invoke_tool(
    invocation: ToolInvocation,
    request: Request,
    token: str = Depends(verify_token)
):
    """Production tool invocation"""

    # Request ID for tracing
    request_id = request.headers.get("X-Request-ID", "unknown")

    logger.info(
        f"Tool invocation started",
        extra={
            "request_id": request_id,
            "tool_name": invocation.name,
            "client_ip": request.client.host
        }
    )

    start_time = time.time()

    try:
        # Execute tool
        result = await execute_tool_with_monitoring(
            invocation.name,
            invocation.arguments
        )

        # Record metrics
        duration = time.time() - start_time
        tool_invocations.labels(tool_name=invocation.name, status="success").inc()
        tool_duration.labels(tool_name=invocation.name).observe(duration)

        logger.info(
            f"Tool invocation completed",
            extra={
                "request_id": request_id,
                "tool_name": invocation.name,
                "duration_seconds": duration
            }
        )

        return ToolResult(success=True, result=result)

    except Exception as e:
        # Record failure metrics
        duration = time.time() - start_time
        tool_invocations.labels(tool_name=invocation.name, status="error").inc()

        logger.error(
            f"Tool invocation failed",
            extra={
                "request_id": request_id,
                "tool_name": invocation.name,
                "error": str(e),
                "duration_seconds": duration
            },
            exc_info=True
        )

        return ToolResult(success=False, error="Internal error")


# Health checks
@app.get("/health")
async def health_check():
    """Basic health check"""
    return {"status": "healthy"}


@app.get("/health/ready")
async def readiness_check():
    """Readiness check (checks dependencies)"""
    try:
        # Check database
        await check_database()
        # Check other dependencies
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail="Not ready")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_config="logging_config.yaml"
    )
```

### Docker Deployment

```dockerfile
# Dockerfile

FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Non-root user
RUN useradd -m -u 1000 mcp && chown -R mcp:mcp /app
USER mcp

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "production_mcp_server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Kubernetes Deployment

```yaml
# kubernetes/deployment.yaml

apiVersion: apps/v1
kind: Deployment
metadata:
  name: mcp-server
spec:
  replicas: 3
  selector:
    matchLabels:
      app: mcp-server
  template:
    metadata:
      labels:
        app: mcp-server
    spec:
      containers:
      - name: mcp-server
        image: your-registry/mcp-server:1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: MCP_SECRET_TOKEN
          valueFrom:
            secretKeyRef:
              name: mcp-secrets
              key: token
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: mcp-server
spec:
  selector:
    app: mcp-server
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

---

## Conclusion

Model Context Protocol (MCP) support in Mistral Agents API enables:

1. **Standardized Integration** - Anthropic MCP compatibility
2. **Custom Tools** - Expose your business logic to agents
3. **Secure Access** - Enterprise-grade authentication and authorization
4. **Production Ready** - Scalable, monitored, and reliable

MCP bridges the gap between Mistral's built-in connectors and your custom systems, creating a comprehensive agent platform.

**Next Steps:**
- Review [Connectors Guide](mistral_agents_api_connectors_guide.md) for built-in capabilities
- Explore [Orchestration Guide](mistral_agents_api_orchestration_guide.md) for multi-agent patterns
- Study [Production Guide](mistral_agents_api_production_guide.md) for deployment

---

**Documentation Version**: 2.0 (May 2025 Launch Edition)
**Last Updated**: May 27, 2025
**Mistral AI - Connect Everything**
