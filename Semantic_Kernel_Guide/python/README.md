# Semantic Kernel Python Guide

**A Comprehensive Python-Specific Technical Reference for Microsoft Semantic Kernel**

Last Updated: November 2025
Python Version: 3.9+
Semantic Kernel Python: 1.38.0+

---

## Overview

This directory contains comprehensive, Python-specific documentation for building AI agents and agentic systems with Microsoft's Semantic Kernel framework. All examples, patterns, and best practices are tailored specifically for Python developers.

### What You'll Find Here

- **Production-Ready Patterns:** Enterprise deployment, monitoring, and scaling strategies
- **Multi-Agent Systems:** Advanced orchestration, coordination, and collaboration patterns
- **2025 Features:** Model Context Protocol (MCP), Vector Store v1.34, Google A2A Protocol integration
- **Real-World Examples:** Complete, tested code recipes for common use cases
- **Best Practices:** Error handling, resilience, security, and performance optimization

---

## Quick Start

### Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install "semantic-kernel[openai,azure]" python-dotenv tenacity opentelemetry-sdk
```

### Hello World Agent

```python
import asyncio
import os
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

async def main():
    # Initialize kernel with OpenAI
    kernel = Kernel()
    kernel.add_service(
        OpenAIChatCompletion(
            model_id="gpt-4",
            api_key=os.environ["OPENAI_API_KEY"]
        )
    )

    # Create and invoke a simple function
    prompt = "Explain Semantic Kernel in one sentence."
    function = kernel.create_function_from_prompt(prompt)
    result = await kernel.invoke(function)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Documentation Structure

### Core Guides

1. **[semantic_kernel_comprehensive_python.md](semantic_kernel_comprehensive_python.md)**
   - Complete Python reference from basics to advanced topics
   - Agents, functions, plugins, memory, planners
   - 2025 features: MCP, Vector Store v1.34, A2A Protocol
   - Installation, setup, and configuration
   - Type hints, async patterns, and Python best practices

2. **[semantic_kernel_production_python.md](semantic_kernel_production_python.md)**
   - Production deployment strategies (Docker, Kubernetes, Azure)
   - Performance optimization and caching
   - Monitoring with OpenTelemetry and Application Insights
   - Error handling, retries, and circuit breakers
   - Security best practices and secrets management
   - Cost tracking and optimization
   - Testing strategies (unit, integration, e2e)

3. **[semantic_kernel_recipes_python.md](semantic_kernel_recipes_python.md)**
   - 30+ production-ready code examples
   - Document processing and RAG implementations
   - Multi-agent systems and orchestration
   - Plugin development patterns
   - Streaming responses with FastAPI
   - Integration with external APIs and services

4. **[semantic_kernel_diagrams_python.md](semantic_kernel_diagrams_python.md)**
   - Python-specific architecture diagrams
   - Request flow visualizations
   - Multi-agent orchestration patterns
   - Memory and embedding pipelines
   - Async execution patterns

### Advanced Topics

5. **[semantic_kernel_advanced_multi_agent_python.md](semantic_kernel_advanced_multi_agent_python.md)**
   - AgentGroupChat and termination strategies
   - Role routing and capability-based routing
   - Human-in-the-loop handoffs
   - Resilience patterns for multi-agent systems
   - Advanced coordination and communication

6. **[semantic_kernel_middleware_python.md](semantic_kernel_middleware_python.md)**
   - Request/response middleware patterns
   - Guardrails and policy enforcement
   - Content filtering and PII detection
   - Telemetry and observability hooks
   - Custom middleware development

7. **[semantic_kernel_streaming_server_fastapi.md](semantic_kernel_streaming_server_fastapi.md)**
   - FastAPI server with Server-Sent Events (SSE)
   - Streaming incremental responses
   - Production deployment with Docker/Kubernetes
   - Authentication and rate limiting
   - WebSocket alternatives

### Index and Navigation

8. **[GUIDE_INDEX.md](GUIDE_INDEX.md)**
   - Complete topic index
   - Quick reference by use case
   - Cross-reference to related sections
   - Learning paths for different skill levels

---

## 2025 Features Highlighted

### Model Context Protocol (MCP)
- **Host/Client Implementation:** Connect to MCP servers for tool/resource access
- **MCP Server Creation:** Expose SK functions as MCP tools
- **Python SDK:** Full MCP support in Python (v1.34+)
- See: Comprehensive Guide → Section 12

### Vector Store Overhaul (v1.34)
- **Unified API:** Consistent interface across all vector stores
- **Improved Performance:** Optimized embedding and retrieval
- **New Connectors:** Enhanced Azure AI Search, Qdrant, Weaviate support
- See: Comprehensive Guide → Section 9

### Google A2A Protocol Integration
- **Agent-to-Agent Communication:** Standardized messaging protocol
- **Interoperability:** Connect SK agents with other A2A-compliant systems
- **Message Routing:** Advanced routing and coordination patterns
- See: Comprehensive Guide → Section 13

### Microsoft Agent Framework Integration
- **Unified Agent SDK:** Integration with Microsoft's broader agent ecosystem
- **Cross-Framework Communication:** SK agents working with AutoGen, Agent Framework
- **Enterprise Features:** Enhanced governance, monitoring, and compliance
- See: Comprehensive Guide → Section 14

---

## Learning Paths

### Beginner Path (1-2 weeks)
1. Start with: **Comprehensive Guide → Sections 1-3** (Fundamentals, Simple Agents)
2. Try: **Quick Start** (above) and first examples
3. Explore: **Recipes → Basic Patterns** (Q&A, summarization)
4. Practice: Build a simple chatbot with memory

### Intermediate Path (2-4 weeks)
1. Study: **Comprehensive Guide → Sections 4-8** (Plugins, Memory, Planners)
2. Build: **Recipes → Multi-Agent Examples** (debate system, research assistant)
3. Learn: **Advanced Multi-Agent Guide** (orchestration patterns)
4. Implement: RAG system with document processing

### Advanced Path (4-8 weeks)
1. Master: **Production Guide** (deployment, monitoring, scaling)
2. Deep Dive: **2025 Features** (MCP, A2A Protocol, Vector Store v1.34)
3. Architect: **Middleware Guide** (guardrails, custom middleware)
4. Deploy: Production-grade multi-agent system on Kubernetes

---

## Common Use Cases

### Intelligent Document Processing
**Example:** Upload PDFs → Extract text → Chunk → Embed → Store → Query → Generate answers

**See:**
- Recipes → Document QA Recipe
- Comprehensive Guide → Memory & Vector Stores
- Production Guide → Scaling Strategies

### Multi-Agent Research Assistant
**Example:** User query → Coordinator agent → Research agent (web search) → Writer agent (draft) → Editor agent (refine)

**See:**
- Advanced Multi-Agent Guide
- Recipes → Multi-Agent Collaboration
- Comprehensive Guide → Agent Orchestration

### API Integration Hub
**Example:** Natural language → Parse intent → Call appropriate API → Format response

**See:**
- Comprehensive Guide → Plugins & OpenAPI
- Recipes → API Integration Patterns
- Production Guide → Error Handling

### Streaming Chat Server
**Example:** WebSocket/SSE → Agent processes → Stream tokens → Real-time updates

**See:**
- Streaming Server Guide (FastAPI)
- Production Guide → Performance Optimization
- Recipes → Streaming Patterns

---

## Python-Specific Best Practices

### Async/Await Patterns
```python
# ✅ Good: Use async/await throughout
async def process_query(kernel: Kernel, query: str) -> str:
    result = await kernel.invoke(function, input=query)
    return str(result)

# ❌ Bad: Mixing sync and async
def process_query(kernel: Kernel, query: str) -> str:
    result = asyncio.run(kernel.invoke(function, input=query))  # Don't do this
    return str(result)
```

### Type Hints
```python
from typing import List, Optional
from semantic_kernel import Kernel
from semantic_kernel.functions import KernelFunction

async def create_agent(
    kernel: Kernel,
    name: str,
    instructions: str,
    tools: Optional[List[KernelFunction]] = None
) -> ChatCompletionAgent:
    # Type hints improve IDE support and catch errors early
    pass
```

### Error Handling
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def invoke_with_retry(kernel: Kernel, function: KernelFunction):
    try:
        return await kernel.invoke(function)
    except Exception as e:
        logger.error(f"Function invocation failed: {e}")
        raise
```

### Environment Configuration
```python
# Use python-dotenv for local development
from dotenv import load_dotenv
import os

load_dotenv()

config = {
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "deployment_name": os.getenv("AZURE_OPENAI_DEPLOYMENT"),
}

# Validate required variables
assert config["openai_api_key"], "OPENAI_API_KEY not set"
```

---

## Integration with Azure Services

### Supported Azure Services

| Service | Purpose | Python Package |
|---------|---------|----------------|
| Azure OpenAI | LLM/Chat/Embeddings | `semantic-kernel[azure]` |
| Azure AI Search | Vector Store | `azure-search-documents` |
| Azure Key Vault | Secrets Management | `azure-keyvault-secrets` |
| Application Insights | Monitoring | `opentelemetry-sdk`, `azure-monitor-opentelemetry` |
| Azure Functions | Serverless | `azure-functions` |
| Azure Container Apps | Hosting | - |

### Quick Azure OpenAI Setup

```python
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

kernel = Kernel()
kernel.add_service(
    AzureChatCompletion(
        deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"]
    )
)
```

---

## Performance Optimization

### Key Strategies

1. **Async Everywhere:** Use async/await for all I/O operations
2. **Connection Pooling:** Reuse HTTP clients and connections
3. **Caching:** Cache function results, embeddings, and API responses
4. **Batching:** Process multiple requests together when possible
5. **Streaming:** Stream responses for long-running operations

**See:** Production Guide → Performance Section

---

## Testing

### Test Structure

```
tests/
├── unit/
│   ├── test_functions.py
│   ├── test_plugins.py
│   └── test_agents.py
├── integration/
│   ├── test_azure_integration.py
│   ├── test_memory_stores.py
│   └── test_workflows.py
└── e2e/
    ├── test_chat_scenarios.py
    └── test_multi_agent.py
```

### Example Unit Test

```python
import pytest
from semantic_kernel import Kernel

@pytest.mark.asyncio
async def test_simple_function():
    kernel = Kernel()
    # Mock the AI service
    kernel.add_service(MockChatCompletion())

    function = kernel.create_function_from_prompt("Say hello")
    result = await kernel.invoke(function)

    assert "hello" in str(result).lower()
```

**See:** Production Guide → Testing Strategies

---

## Security Checklist

- [ ] Use Azure Key Vault or environment variables for secrets
- [ ] Never commit API keys or credentials to version control
- [ ] Implement input validation and sanitization
- [ ] Use HTTPS for all external communications
- [ ] Enable rate limiting and request throttling
- [ ] Log security events (authentication, authorization)
- [ ] Regularly update dependencies (`pip install --upgrade`)
- [ ] Use Managed Identity for Azure services (in production)
- [ ] Implement content filtering and guardrails
- [ ] Audit and monitor API usage and costs

**See:** Production Guide → Security Best Practices

---

## Troubleshooting

### Common Issues

**Issue:** `ImportError: No module named 'semantic_kernel'`
```bash
# Solution: Install the package
pip install semantic-kernel
```

**Issue:** `API key not found` or `Authentication failed`
```python
# Solution: Check environment variables
import os
print(os.getenv("OPENAI_API_KEY"))  # Should not be None
```

**Issue:** `Token limit exceeded`
```python
# Solution: Implement chunking or use smaller context
from semantic_kernel.text import TextChunker

chunks = TextChunker.split_plaintext_lines(text, max_tokens=1000)
```

**Issue:** Async functions not working
```python
# Solution: Always use asyncio.run() at top level
import asyncio

async def main():
    # Your async code here
    pass

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Additional Resources

### Official Documentation
- [Semantic Kernel GitHub](https://github.com/microsoft/semantic-kernel)
- [Python Documentation](https://learn.microsoft.com/semantic-kernel/get-started/quick-start-guide?pivots=programming-language-python)
- [API Reference](https://learn.microsoft.com/python/api/overview/azure/semantic-kernel)

### Python Community
- [SK Python Samples](https://github.com/microsoft/semantic-kernel/tree/main/python/samples)
- [Discussion Forum](https://github.com/microsoft/semantic-kernel/discussions)

### Related Frameworks
- [LangGraph Guide](../LangGraph_Guide/)
- [AutoGen Guide](../AutoGen_Guide/)
- [CrewAI Guide](../CrewAI_Guide/)

---

## Version Support

| Component | Minimum Version | Recommended |
|-----------|----------------|-------------|
| Python | 3.9 | 3.11+ |
| semantic-kernel | 1.0.0 | 1.38.0+ |
| openai | 1.0.0 | Latest |
| azure-identity | 1.12.0 | Latest |
| FastAPI | 0.100.0 | Latest (for streaming) |
| uvicorn | 0.23.0 | Latest (for serving) |

---

## Contributing

Found an issue or want to contribute?
- Submit issues with Python version, SK version, and minimal reproduction
- Include full error messages and stack traces
- Provide code examples demonstrating the problem

---

## Quick Navigation

**New to Semantic Kernel?** → Start with [Comprehensive Guide](semantic_kernel_comprehensive_python.md)
**Building for Production?** → See [Production Guide](semantic_kernel_production_python.md)
**Need Examples?** → Check [Recipes](semantic_kernel_recipes_python.md)
**Multi-Agent Systems?** → Read [Advanced Multi-Agent Guide](semantic_kernel_advanced_multi_agent_python.md)
**Complete Index?** → Browse [GUIDE_INDEX.md](GUIDE_INDEX.md)

---

**Start building intelligent Python applications with Semantic Kernel today!**
