# LlamaIndex Complete Documentation

Comprehensive technical documentation for LlamaIndex—the leading framework for building LLM-powered agents with Retrieval-Augmented Generation (RAG), data connectors, and agentic reasoning over private data.

**Now organized by programming language: Python and TypeScript**

> **Python Version:** 0.14.20 (April 3, 2026) — previously 0.14.8 (November 2025)
> **TypeScript:** `@llamaindex/workflow` 1.1.4 (April 15, 2026)

## ⚠️ Breaking Changes Since v0.14.8

### Legacy Agent Classes Hard-Removed (Python)
The following classes were **hard-removed** (no deprecation warning — they will raise `ImportError`). Migrate to `AgentWorkflow`-based agents:

| Removed Class | Replacement |
|---------------|-------------|
| `FunctionCallingAgent` | `FunctionAgent` (via `AgentWorkflow`) |
| `AgentRunner` | `AgentWorkflow` |
| `AgentWorker` | `AgentWorkflow` |
| `StructuredAgentPlanner` | `AgentWorkflow` |
| `OpenAIAgent` | `FunctionAgent` with OpenAI LLM |
| `LLMPredictor` | Use LLM object directly |

**Migration example:**
```python
# BEFORE (raises ImportError now)
from llama_index.core.agent import ReActAgent
agent = ReActAgent.from_tools(tools, llm=llm)

# AFTER
from llama_index.core.agent import ReActAgent as agent_cls
# Or better, use the new Workflow-first API:
from llama_index.core.workflow import AgentWorkflow
agent = AgentWorkflow.from_tools_or_functions(tools, llm=llm)
result = await agent.run("Your query")
```

### Python 3.9 Dropped
Minimum Python version is now **3.10**. Python 3.9 is no longer supported.

### TypeScript Package Restructuring
The `llamaindex` monorepo package (`npm install llamaindex`) is largely superseded by scoped packages. Migrate to `@llamaindex/workflow`, `@llamaindex/core`, etc.

### `asyncio_module` Parameter Deprecated
Use `get_asyncio_module` instead of the `asyncio_module` parameter.

---

## 🆕 What's New in 2026

- **`AgentWorkflow`** and **`Workflows`** are now the primary agent orchestration primitives
- New workflow-based agent classes: `FunctionAgent`, `CodeActAgent`, `ReActAgent`, `AgentWorkflow`
- **`early_stopping_method`** parameter on agent workflows
- **Agent Client Protocol integration** for cross-framework interoperability
- **Enhanced long-term and short-term memory** for agents
- **`LlamaSheets`**: structured extraction from complex spreadsheets
- **LiteParse** (TypeScript): zero-Python-dependency PDF/document parsing for TypeScript workflows

---

## 🆕 What's New in 2025

### Workflows 2.x
Both Python and TypeScript implementations now feature **Workflows 2.x**, a powerful event-driven orchestration system:

- **Event-Driven Architecture** - Build reactive, loosely-coupled agent systems
- **Async-First Design** - High-performance asynchronous workflows
- **Type-Safe State Management** - Pydantic (Python) / TypeScript interfaces
- **Multi-Agent Coordination** - Event-based agent communication
- **Streaming Support** - Real-time event streaming
- **Production-Ready** - Battle-tested patterns and best practices

---

## 📚 Documentation by Language

### 🐍 [Python Documentation](python/)

Complete guide to LlamaIndex Python with **380+ code examples**:

- **[Python README](python/README.md)** - Quick start and overview
- **[Comprehensive Guide](python/llamaindex_comprehensive_guide.md)** - 350+ examples covering all features
- **[Diagrams](python/llamaindex_diagrams.md)** - 30+ visual architecture diagrams
- **[Production Guide](python/llamaindex_production_guide.md)** - Docker, Kubernetes, monitoring, scaling
- **[Recipes](python/llamaindex_recipes.md)** - 10 production-ready applications
- **[Guide Index](python/GUIDE_INDEX.md)** - Complete searchable index

**Python Quick Start:**
```bash
pip install llama-index llama-index-core
pip install llama-index-workflows  # Workflows 1.0
```

```python
from llama_index.core.workflow import Workflow, StartEvent, StopEvent, step

class MyWorkflow(Workflow):
    @step
    async def process(self, ev: StartEvent) -> StopEvent:
        result = await self.llm.acomplete(ev.query)
        return StopEvent(result=result)

workflow = MyWorkflow()
result = await workflow.run(query="What is AI?")
```

---

### 📘 [TypeScript Documentation](typescript/)

Complete guide to LlamaIndex TypeScript with **300+ code examples**:

- **[TypeScript README](typescript/README.md)** - Quick start and overview
- **[Workflows Comprehensive Guide](typescript/llamaindex_workflows_typescript_comprehensive_guide.md)** - 200+ Workflows 1.0 examples
- **[Production Guide](typescript/llamaindex_typescript_production_guide.md)** - Node.js deployment, Docker, K8s, serverless
- **[Recipes](typescript/llamaindex_typescript_recipes.md)** - 10 production-ready TypeScript applications
- **[Guide Index](typescript/GUIDE_INDEX.md)** - Complete searchable index

**TypeScript Quick Start:**
```bash
npm install llamaindex
npm install llama-index-workflows  # Workflows 1.0 (standalone package)
```

```typescript
import { Workflow, StartEvent, StopEvent, step } from 'llama-index-workflows';

class MyWorkflow extends Workflow {
  @step()
  async process(ev: StartEvent): Promise<StopEvent> {
    const result = await this.llm.complete(ev.query);
    return new StopEvent({ result: result.text });
  }
}

const workflow = new MyWorkflow();
const result = await workflow.run({ query: "What is AI?" });
```

---

## 🎯 Choose Your Language

### When to Use Python

✅ **Best for:**
- Data science and ML workflows
- Extensive library ecosystem
- Jupyter notebooks and research
- Existing Python infrastructure
- Scientific computing

**Python Strengths:**
- Rich data science libraries (NumPy, pandas, scikit-learn)
- Mature LLM tooling ecosystem
- Extensive community resources
- Strong integration with ML frameworks

---

### When to Use TypeScript

✅ **Best for:**
- Web applications and APIs
- Node.js/JavaScript ecosystem
- Type-safe development
- Frontend integration
- Serverless deployments

**TypeScript Strengths:**
- Full type safety with IDE support
- Native async/await patterns
- Seamless web framework integration (Express, NestJS, Fastify)
- Modern JavaScript ecosystem
- Excellent for microservices

---

## 🚀 Quick Comparison

| Feature | Python | TypeScript |
|---------|--------|------------|
| **Workflows 1.0** | ✅ Core package | ✅ Standalone package |
| **Type Safety** | Pydantic models | Native TypeScript |
| **Async Support** | asyncio | Native Promises |
| **Web Frameworks** | FastAPI, Flask | Express, NestJS, Fastify |
| **Deployment** | Docker, K8s, Lambda | Docker, K8s, Vercel, Lambda |
| **Package Manager** | pip, poetry | npm, yarn, pnpm |
| **IDE Support** | VS Code, PyCharm | VS Code, WebStorm |
| **Code Examples** | 380+ | 300+ |

---

## 📊 Documentation Statistics

### Python Documentation
- **Total Examples:** 380+
- **Lines of Code:** 15,000+
- **Topics Covered:** 50+
- **Diagrams:** 30+
- **Recipes:** 10
- **Words:** 50,000+

### TypeScript Documentation
- **Total Examples:** 300+
- **Lines of Code:** 10,000+
- **Topics Covered:** 40+
- **Recipes:** 10
- **Words:** 35,000+

### Combined Total
- **Code Examples:** 680+
- **Production Recipes:** 20
- **Topics:** 90+
- **Total Documentation:** 85,000+ words

---

## 🎓 Learning Paths

### Beginner Path (Both Languages)
1. Read language-specific README
2. Try Quick Start example
3. Build Recipe 1 (Basic RAG Chatbot)
4. Explore Workflows 1.0 fundamentals

**Time:** 2-3 days

### Intermediate Path
1. Study Workflows comprehensive guide
2. Build 3-5 recipes
3. Implement multi-agent patterns
4. Add production features (caching, monitoring)

**Time:** 1-2 weeks

### Advanced Path
1. Complete all recipes
2. Study production deployment guide
3. Implement CI/CD pipeline
4. Deploy to production with monitoring

**Time:** 2-4 weeks

---

## 💡 Common Use Cases

### Document Q&A / RAG Systems
- **Python:** [Python Recipe 1](python/llamaindex_recipes.md#recipe-1)
- **TypeScript:** [TypeScript Recipe 1](typescript/llamaindex_typescript_recipes.md#recipe-1)

### Multi-Agent Workflows
- **Python:** [Python Comprehensive Guide](python/llamaindex_comprehensive_guide.md) (Multi-Agent section)
- **TypeScript:** [TypeScript Workflows Guide](typescript/llamaindex_workflows_typescript_comprehensive_guide.md) (Multi-Agent section)

### Production Deployment
- **Python:** [Python Production Guide](python/llamaindex_production_guide.md)
- **TypeScript:** [TypeScript Production Guide](typescript/llamaindex_typescript_production_guide.md)

### Data Extraction
- **Python:** [Python Recipe 6](python/llamaindex_recipes.md#recipe-6)
- **TypeScript:** [TypeScript Recipe 6](typescript/llamaindex_typescript_recipes.md#recipe-6)

### Customer Support
- **Python:** [Python Recipe 10](python/llamaindex_recipes.md#recipe-10)
- **TypeScript:** [TypeScript Recipe 10](typescript/llamaindex_typescript_recipes.md#recipe-10)

---

## 🔗 External Resources

### Official Documentation
- [LlamaIndex Python Docs](https://docs.llamaindex.ai)
- [LlamaIndex TypeScript Docs](https://ts.llamaindex.ai)
- [GitHub - Python](https://github.com/run-llama/llama_index)
- [GitHub - TypeScript](https://github.com/run-llama/LlamaIndexTS)

---

## 📋 Revision History

| Date | Version | Changes |
|------|---------|---------|
| April 16, 2026 | Python 0.14.20 / TS @llamaindex/workflow 1.1.4 | Hard-removal of legacy agent classes documented; Python 3.9 drop; TypeScript package restructuring; AgentWorkflow migration examples; LiteParse; Agent Client Protocol |
| November 2025 | Python 0.14.8 / TS Workflows 2.11.2 | Workflows 2.x; event-driven architecture; multi-language reorganization |

### Community
- [Discord Community](https://discord.gg/dGcwcsnxhU)
- [Twitter @llama_index](https://twitter.com/llama_index)
- [Blog](https://blog.llamaindex.ai)

### Related Projects
- [LangChain](https://www.langchain.com) - Alternative framework
- [Pinecone](https://www.pinecone.io) - Vector database
- [Chroma](https://www.trychroma.com) - Local vector store
- [Weaviate](https://weaviate.io) - Vector search engine

---

## 📖 Directory Structure

```
LlamaIndex_Guide/
├── README.md (This file)
│
├── python/
│   ├── README.md
│   ├── GUIDE_INDEX.md
│   ├── [llamaindex_comprehensive_guide.md](./python/llamaindex_comprehensive_guide.md) (350+ examples)
│   ├── [llamaindex_diagrams.md](./python/llamaindex_diagrams.md) (30+ diagrams)
│   ├── [llamaindex_production_guide.md](./python/llamaindex_production_guide.md)
│   ├── [llamaindex_recipes.md](./python/llamaindex_recipes.md) (10 recipes)
│   ├── llamaindex_advanced_implementations.md
│   ├── llamaindex_advanced_agents_python.md
│   ├── llamaindex_observability_python.md
│   └── llamaindex_streaming_server_fastapi.md
│
└── typescript/
    ├── README.md
    ├── GUIDE_INDEX.md
    ├── [llamaindex_workflows_typescript_comprehensive_guide.md](./typescript/llamaindex_workflows_typescript_comprehensive_guide.md) (200+ examples)
    ├── [llamaindex_typescript_production_guide.md](./typescript/llamaindex_typescript_production_guide.md)
    └── [llamaindex_typescript_recipes.md](./typescript/llamaindex_typescript_recipes.md) (10 recipes)
```

---

## 🤝 Contributing

This documentation is community-maintained. Contributions welcome!

### Ways to Contribute
- Report issues or unclear sections
- Add examples for both languages
- Improve diagrams and visualizations
- Add production patterns
- Translate to other languages
- Share your use cases

---

## ❓ FAQ

### Q: Which language should I choose?
**A:** Choose Python for data science workflows and existing Python infrastructure. Choose TypeScript for web applications and Node.js ecosystem integration.

### Q: Can I use both languages together?
**A:** Yes! You can build Python backends and TypeScript frontends, or use microservices architecture with both languages.

### Q: Are the features the same in both languages?
**A:** Core features are similar, but each language has ecosystem-specific optimizations and integrations.

### Q: Do I need to learn Workflows 1.0?
**A:** Workflows 1.0 is optional but recommended for complex multi-agent systems and production applications.

### Q: How do I migrate from Python to TypeScript (or vice versa)?
**A:** The concepts are similar. Review the Quick Start in your target language and adapt your workflow patterns.

---

## 🚀 Next Steps

1. **Choose your language** (Python or TypeScript)
2. **Read the language-specific README**
3. **Try the Quick Start example**
4. **Build Recipe 1** (Basic RAG Chatbot)
5. **Explore Workflows 1.0**
6. **Deploy to production** using the Production Guide

---

## 📞 Support

- **Documentation Issues**: Open an issue in the respective GitHub repo
- **Python Questions**: [Python Discord](https://discord.gg/dGcwcsnxhU)
- **TypeScript Questions**: [TypeScript Discord](https://discord.gg/dGcwcsnxhU)
- **General Questions**: Refer to official docs

---

**Happy building with LlamaIndex! 🦙**

*Last updated: November 2025*
*Python version: 0.14.8+*
*TypeScript version: 0.5.0+*
*Workflows 1.0: Available in both languages*

---

## Appendix: Quick Reference

### Python Installation
```bash
pip install llama-index llama-index-core
pip install llama-index-llms-openai
pip install llama-index-embeddings-openai
pip install llama-index-workflows
```

### TypeScript Installation
```bash
npm install llamaindex
npm install llama-index-workflows
npm install -D typescript @types/node
```

### Environment Variables (Both Languages)
```bash
OPENAI_API_KEY=your_api_key_here
LLM_MODEL=gpt-4
LLM_TEMPERATURE=0.7
```

---

**Documentation by: LlamaIndex Community**
**For: Technical Professionals, AI Engineers, Python & TypeScript Developers**
