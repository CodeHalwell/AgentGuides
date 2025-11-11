# Pydantic AI: Complete Technical Documentation

> **The Pydantic Way for Generative AI** - Type-safe, production-grade agent framework with FastAPI-inspired developer experience.

**Latest Version:** 1.0.0 (September 2025)  
**Framework:** [Pydantic AI](https://ai.pydantic.dev)  
**Python:** 3.10+  
**License:** Comprehensive Educational Guide

---

## 📚 Documentation Overview

This comprehensive guide covers Pydantic AI from beginner fundamentals to advanced production patterns. Whether you're building your first agent or deploying a multi-agent enterprise system, you'll find detailed explanations, type-annotated code examples, and production-ready patterns.

### Guide Files

| File | Purpose | Audience |
|------|---------|----------|
| **`pydantic_ai_comprehensive_guide.md`** | Complete reference from fundamentals to advanced concepts | Everyone |
| **`pydantic_ai_production_guide.md`** | Deployment, scaling, monitoring, and operational patterns | DevOps/Platform engineers |
| **`pydantic_ai_recipes.md`** | Real-world code examples and practical patterns | Developers |
| **`pydantic_ai_diagrams.md`** | Visual architecture and flow diagrams | Visual learners |
| **`README.md`** | This file - Navigation and quick reference | Everyone |

---

## 🚀 Quick Start

### Installation

```bash
# Full installation
pip install pydantic-ai

# Or minimal with specific providers
pip install "pydantic-ai-slim[openai,anthropic]"

# With observability
pip install "pydantic-ai-slim[openai,logfire]"
```

### Hello World (5 minutes)

```python
from pydantic_ai import Agent

# Create agent
agent = Agent('openai:gpt-4o')

# Run query
result = agent.run_sync('What is Pydantic AI?')
print(result.output)
```

### Type-Safe Agent (10 minutes)

```python
from pydantic_ai import Agent
from pydantic import BaseModel

class Article(BaseModel):
    title: str
    summary: str
    keywords: list[str]

agent = Agent(
    'openai:gpt-4o',
    output_type=Article  # Type-safe output!
)

result = agent.run_sync('Write about Python type safety')
print(f"Title: {result.output.title}")
print(f"Keywords: {result.output.keywords}")
```

### With Tools & Dependency Injection (15 minutes)

```python
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass
import asyncio

@dataclass
class AppDeps:
    database_url: str

agent = Agent(
    'openai:gpt-4o',
    deps_type=AppDeps,
    instructions='Answer using database information'
)

@agent.tool
async def search_database(ctx: RunContext[AppDeps], query: str) -> str:
    # Access injected dependencies
    return f"Results from {ctx.deps.database_url}: {query}"

async def main():
    deps = AppDeps(database_url='postgresql://localhost/mydb')
    result = await agent.run('Find recent articles', deps=deps)
    print(result.output)

asyncio.run(main())
```

---

## 📖 Learning Path

### Level 1: Fundamentals (2-3 hours)
- [ ] **Comprehensive Guide:** "Installation & Setup"
- [ ] **Comprehensive Guide:** "Core Fundamentals"
- [ ] **Comprehensive Guide:** "Simple Agents"
- [ ] **Recipes:** Try "Recipe 1: Customer Support Chatbot"

### Level 2: Type Safety & Tools (3-4 hours)
- [ ] **Comprehensive Guide:** "Type Safety & Validation"
- [ ] **Comprehensive Guide:** "Structured Output"
- [ ] **Comprehensive Guide:** "Tools & Function Calling"
- [ ] **Recipes:** Try "Recipe 2: Multi-Agent Pipeline"

### Level 3: Advanced Patterns (4-5 hours)
- [ ] **Comprehensive Guide:** "Dependency Injection"
- [ ] **Recipes:** Try "Recipe 3: RAG System"
- [ ] **Recipes:** Try "Recipe 4: Streaming Agent"
- [ ] **Comprehensive Guide:** "Advanced Patterns"

### Level 4: Production Deployment (5-6 hours)
- [ ] **Production Guide:** "Production Architecture Patterns"
- [ ] **Production Guide:** "Observability & Monitoring"
- [ ] **Production Guide:** "Error Handling & Resilience"
- [ ] **Production Guide:** "Scaling Strategies"

### Level 5: Mastery (Ongoing)
- [ ] **Diagrams:** Study all architecture patterns
- [ ] **Recipes:** Implement all practical examples
- [ ] **Production Guide:** Setup production monitoring
- [ ] Build your first production application

---

## 🎯 Core Concepts

### Philosophy: "FastAPI Feeling" for GenAI

Pydantic AI brings familiar patterns from FastAPI to AI development:

```
FastAPI:                          Pydantic AI:
├─ Type hints                      ├─ Type hints for agent outputs
├─ Validation (Pydantic)           ├─ Validation (Pydantic v2)
├─ Dependency injection            ├─ Dependency injection (RunContext)
├─ Decorators for routes           ├─ Decorators for tools (@agent.tool)
├─ Structured responses            ├─ Structured outputs (BaseModel)
└─ Production observability        └─ Production observability (Logfire)
```

### Key Pillars

1. **Type Safety**: All inputs/outputs validated with Pydantic
2. **Model Agnosticism**: Supports OpenAI, Anthropic, Gemini, Groq, and more
3. **Structured Outputs**: Guarantee response validation and schema compliance
4. **Observable Systems**: Built-in Logfire integration for production observability
5. **Composable Tools**: Function calling as first-class citizens
6. **Async-First**: Native async/await throughout
7. **Test-Friendly**: TestModel for unit testing without API calls

---

## 🏗️ Architecture Patterns

### Single Agent (Simplest)
```python
agent = Agent('openai:gpt-4o')
result = agent.run_sync('Query...')
```
**Use for:** Simple chatbots, single-task helpers

### Agent with Tools & Dependencies
```python
@dataclass
class Deps:
    db: Database

agent = Agent('openai:gpt-4o', deps_type=Deps)

@agent.tool
async def tool(ctx: RunContext[Deps]) -> str:
    return await ctx.deps.db.query()
```
**Use for:** Complex applications with multiple data sources

### Multi-Agent Coordination
```python
research_agent = Agent('openai:gpt-4o')
write_agent = Agent('anthropic:claude-3-5-sonnet')
editor_agent = Agent('openai:gpt-4o')

# Coordinate via tools that call each other
```
**Use for:** Complex workflows requiring specialisation

### Streaming for Real-Time UI
```python
async with agent.run_stream(query) as response:
    async for text in response.stream_text():
        yield text  # Stream to client
```
**Use for:** Web applications needing real-time updates

---

## 📚 Provider Support

Pydantic AI works with all major LLM providers:

| Provider | Models | Status |
|----------|--------|--------|
| **OpenAI** | GPT-4o, GPT-4 Turbo, GPT-4, etc. | ✅ Fully supported |
| **Anthropic** | Claude 3.5 Sonnet, Claude 3 Opus | ✅ Fully supported |
| **Google** | Gemini 1.5 Pro/Flash | ✅ Fully supported |
| **Groq** | Llama 3.3, Mixtral | ✅ Fully supported |
| **Mistral** | Mistral Large, Small | ✅ Fully supported |
| **DeepSeek** | DeepSeek Chat | ✅ Fully supported |
| **Grok** | Grok 2 | ✅ Fully supported |
| **AWS Bedrock** | Multiple models | ✅ Supported |

**Switch providers** without changing your code:
```python
# From OpenAI to Anthropic - one line change!
agent = Agent('anthropic:claude-3-5-sonnet-latest')
```

---

## 🛠️ Tool Categories

### Function Tools
- **@agent.tool**: Async functions with context access
- **@agent.tool_plain**: Synchronous functions without context

### Built-in Tools
- **WebSearchTool**: Search the web in real-time
- **UrlContextTool**: Fetch and parse webpage content
- **MemoryTool**: Persistent user memory
- **CodeExecutionTool**: Execute Python safely (sandboxed)

### MCP Integration
- Connect to Model Context Protocol servers
- Access filesystem, databases, APIs via MCP
- Share tools across applications

---

## 📊 Common Patterns

### Validation with Retry
```python
@agent.output_validator
async def validate(ctx: RunContext, output: MyModel) -> MyModel:
    if not output.valid:
        raise ModelRetry("Please provide valid data")
    return output
```

### Dynamic System Prompts
```python
@agent.system_prompt
async def prompt(ctx: RunContext) -> str:
    return f"Current time: {datetime.now()}"
```

### Tool Conditional Availability
```python
async def only_for_admins(ctx, tool_def):
    if ctx.deps.is_admin:
        return tool_def
    return None

@agent.tool(prepare=only_for_admins)
async def delete_data(ctx):
    pass
```

### Streaming for UI
```python
async with agent.run_stream(query) as response:
    async for text in response.stream_text():
        yield text  # Send to frontend
```

### Caching Responses
```python
cache.get(query_key)  # Check cache first
result = await agent.run(query)  # Or run agent
cache.set(query_key, result)  # Store for future
```

---

## 🔒 Security Best Practices

### API Keys & Secrets
```python
# ❌ DON'T hardcode
agent = Agent('openai:sk-...your-key...')

# ✅ DO use environment variables
import os
api_key = os.getenv('OPENAI_API_KEY')
# Pydantic AI reads from env automatically
```

### Input Validation
```python
# ❌ DON'T trust user input
@agent.tool
async def delete_record(ctx, record_id: str):
    pass

# ✅ DO validate and constrain
@agent.tool
async def delete_record(ctx, record_id: int = Field(..., gt=0)):
    pass
```

### Tool Approval Requirements
```python
@agent.tool(requires_approval=True)
async def delete_account(ctx):
    # Requires user approval before execution
    pass
```

### Rate Limiting
```python
from pydantic_ai import UsageLimits

result = await agent.run(
    query,
    usage_limits=UsageLimits(
        request_limit=5,
        total_tokens_limit=10000
    )
)
```

---

## 📈 Performance & Optimization

### Token Usage Tracking
```python
result = await agent.run(query)
usage = result.usage()
print(f"Tokens: {usage.input_tokens} in, {usage.output_tokens} out")
print(f"Cost: ${usage.total_tokens * 0.00002:.4f}")
```

### Caching for Cost Reduction
```python
# Cache expensive queries
cached_result = redis.get(query_hash)
if cached_result:
    return cached_result

result = await agent.run(query)
redis.setex(query_hash, 3600, result)  # Cache for 1 hour
```

### Parallel Tool Execution
```python
@agent.tool
async def parallel_searches(ctx, queries: list[str]) -> list[str]:
    # Execute all searches concurrently
    return await asyncio.gather(*[search(q) for q in queries])
```

### Model Selection for Cost/Performance
```python
# Fast & cheap
fast_agent = Agent('openai:gpt-4o-mini')

# Slower & expensive but better
quality_agent = Agent('openai:gpt-4o')

# Use fast agent for simple queries, quality for complex
```

---

## 🧪 Testing

### Unit Testing with TestModel
```python
import pytest
from pydantic_ai.models.test import TestModel

@pytest.mark.asyncio
async def test_agent():
    agent = Agent('openai:gpt-4o')
    
    with agent.override(model=TestModel()):
        result = await agent.run('Test query')
        assert result.output is not None
```

### Mocking Dependencies
```python
from unittest.mock import AsyncMock

@pytest.fixture
async def mock_db():
    db = AsyncMock()
    db.query.return_value = [{'id': 1}]
    return db

@pytest.mark.asyncio
async def test_with_mock(mock_db):
    deps = AppDeps(db=mock_db)
    result = await agent.run('Query', deps=deps)
```

---

## 📦 Project Structure

```
my_agent_project/
├── README.md
├── requirements.txt
├── .env.example
├── .env  (gitignored)
│
├── agents/
│   ├── __init__.py
│   ├── support_agent.py
│   ├── analysis_agent.py
│   └── coordinator_agent.py
│
├── tools/
│   ├── __init__.py
│   ├── database.py
│   ├── external_apis.py
│   └── validators.py
│
├── models/
│   ├── __init__.py
│   ├── dependencies.py
│   ├── responses.py
│   └── inputs.py
│
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── routes.py
│   └── middleware.py
│
├── tests/
│   ├── __init__.py
│   ├── test_agents.py
│   ├── test_tools.py
│   └── conftest.py
│
└── docker-compose.yml
```

---

## 🚀 Deployment Checklist

- [ ] Set API keys in environment variables
- [ ] Configure Logfire for observability
- [ ] Set up PostgreSQL/Redis for persistence
- [ ] Configure rate limiting and usage limits
- [ ] Set up error handling and retries
- [ ] Add monitoring and alerting
- [ ] Test with production models (not mini versions)
- [ ] Set up health checks
- [ ] Configure auto-scaling policies
- [ ] Document agent capabilities and limitations
- [ ] Set up backup and disaster recovery
- [ ] Monitor token usage and costs

---

## 💡 Common Pitfalls & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| Unpredictable outputs | No structure defined | Use `output_type=MyModel` |
| Tools not called | Schema not generated properly | Check tool docstrings |
| High costs | Inefficient prompting | Implement caching, use cheaper models |
| Slow responses | Sequential tool calls | Enable `end_strategy='exhaustive'` |
| API failures | No retry logic | Use `ModelRetry` for validation, backoff for API errors |
| Validation errors | Model outputs wrong format | Use `@agent.output_validator` |
| Token limit exceeded | Large contexts | Use context compression, streaming |

---

## 📚 Additional Resources

### Official Documentation
- [Pydantic AI Documentation](https://ai.pydantic.dev)
- [GitHub Repository](https://github.com/pydantic/pydantic-ai)
- [PyPI Package](https://pypi.org/project/pydantic-ai/)

### Related Technologies
- [Pydantic v2](https://docs.pydantic.dev/latest/) - Data validation
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [Logfire](https://logfire.pydantic.dev/) - Observability
- [MCP](https://modelcontextprotocol.io/) - Context protocol

### Learning Resources
- Type safety in Python: [Python typing docs](https://docs.python.org/3/library/typing.html)
- Async/await: [Python asyncio docs](https://docs.python.org/3/library/asyncio.html)
- LLM integration: [OpenAI API Docs](https://platform.openai.com/docs)

---

## 🤝 Contributing & Community

- **Issues & Questions**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Feature Requests**: GitHub Issues with `enhancement` label
- **Documentation**: Contribution guidelines in repo

---

## 📝 License & Attribution

This comprehensive guide is an educational resource created to help developers understand and implement Pydantic AI effectively. It covers:

- Official Pydantic AI documentation
- Best practices from production deployments
- Community patterns and examples
- Security and performance guidelines

---

## ✨ Key Takeaways

1. **Pydantic AI** brings type safety and structure to AI development
2. **Type hints** make your AI systems reliable and testable
3. **Dependency injection** keeps code clean and composable
4. **Tools** extend agent capabilities with structured functions
5. **Streaming** creates responsive user experiences
6. **Caching & optimization** reduce costs significantly
7. **Testing** with TestModel ensures quality without API calls
8. **Production patterns** scale from single to multi-agent systems

---

## 🎓 Next Steps

1. **Read:** Start with `pydantic_ai_comprehensive_guide.md`
2. **Practice:** Implement recipes from `pydantic_ai_recipes.md`
3. **Deploy:** Follow `pydantic_ai_production_guide.md`
4. **Visualise:** Review `pydantic_ai_diagrams.md` for architecture
5. **Build:** Create your first production agent!

---

**Last Updated:** March 2025  
**Version:** 1.0.0  
**Python:** 3.10+  
**Pydantic AI:** v1.0+

Happy agent building! 🚀


