# SmolAgents: Complete Technical Reference Guide

Welcome to the comprehensive technical guide for **SmolAgents** – the lightweight Python framework for building AI agents that think in code.

## 📚 What's in This Guide

This guide consists of four interconnected documents, each serving a specific purpose:

### 1. **smolagents_comprehensive_guide.md** (MAIN REFERENCE)
   - **Purpose**: Complete technical reference for all SmolAgents concepts
   - **Best for**: Learning the framework from fundamentals to advanced topics
   - **Contains**:
     - Installation & setup procedures
     - Core architecture & design philosophy
     - Detailed explanations of CodeAgent vs ToolCallingAgent
     - Model configuration for 100+ LLM providers
     - Tool creation patterns (@tool decorator & subclass)
     - Multi-agent orchestration strategies
     - Memory & state management
     - Code execution sandboxing
     - Debugging techniques
     - Comparison with other frameworks
   - **Read this**: First, to understand concepts deeply

### 2. **smolagents_diagrams.md** (VISUAL REFERENCE)
   - **Purpose**: ASCII diagrams and visual representations
   - **Best for**: Understanding architecture & workflows visually
   - **Contains**:
     - Framework architecture diagrams
     - Execution flow visualisations
     - CodeAgent vs ToolCallingAgent paradigm comparison
     - Multi-agent orchestration patterns
     - Tool integration architecture
     - Memory management visualisations
     - Performance characteristics
     - Deployment architecture
   - **Read this**: When you need visual understanding

### 3. **smolagents_production_guide.md** (DEPLOYMENT GUIDE)
   - **Purpose**: Production deployment, scaling & operations
   - **Best for**: Building production-grade systems
   - **Contains**:
     - Production readiness checklists
     - Performance optimisation techniques
     - Cost management & token budgeting
     - Monitoring & observability setup
     - Security best practices
     - Error handling & resilience patterns
     - Deployment options (Docker, Kubernetes)
     - Scaling strategies
     - Testing & QA procedures
     - Weights & Biases Weave integration
   - **Read this**: Before deploying to production

### 4. **smolagents_recipes.md** (PRACTICAL EXAMPLES)
   - **Purpose**: Ready-to-use code patterns for common tasks
   - **Best for**: Copy-paste starting points for implementations
   - **Contains**:
     - 20+ complete working code examples
     - Data analysis agents
     - Web research agents
     - Business intelligence workflows
     - Content generation agents
     - Code review agents
     - Multi-agent pipelines
     - API integration patterns
     - Custom tool creation examples
     - Error handling patterns
   - **Read this**: When building specific features

## 🚀 Quick Start Reading Path

### Path 1: Complete Beginner (Recommended)
1. Start with **comprehensive_guide.md** → Installation & Simple Agents sections
2. Run examples from **recipes.md** → Recipe 1-5 (basic patterns)
3. Reference **diagrams.md** when you need visual understanding
4. Move to **comprehensive_guide.md** → Tools & CodeAgent sections

### Path 2: Experienced Developer
1. Skim **comprehensive_guide.md** → Core Concepts section
2. Review **diagrams.md** → Architecture overview
3. Jump to **recipes.md** for implementation examples
4. Reference **production_guide.md** as needed

### Path 3: Production Deployment
1. Read **production_guide.md** → Production Readiness Checklist
2. Reference **comprehensive_guide.md** → specific topics as needed
3. Use **recipes.md** for code patterns
4. Consult **diagrams.md** for architecture validation

## 📖 Document Structure Overview

```
SmolAgents_Guide/
├── README.md (this file)
├── smolagents_comprehensive_guide.md (20 major sections, ~15,000+ lines)
├── smolagents_diagrams.md (9 major diagram sections)
├── smolagents_production_guide.md (12 production topics)
└── smolagents_recipes.md (20 practical code recipes)
```

## 🎯 Key Concepts at a Glance

### CodeAgent: The Revolution
SmolAgents' killer feature is **CodeAgent** – agents that write Python code rather than JSON:

```python
# Traditional agents (JSON-based)
# ❌ Multiple LLM calls needed
# ❌ Limited expressivity
# ❌ Parsing errors

# CodeAgent (SmolAgents)
# ✓ Single LLM call + execution
# ✓ Full Python expressivity (loops, conditionals, functions)
# ✓ 30% more efficient
# ✓ Natural composability

from smolagents import CodeAgent, InferenceClientModel

agent = CodeAgent(model=InferenceClientModel())
result = agent.run("Find Bitcoin price and calculate 10% gain")
# Agent writes Python code, executes it, returns result
```

### The Four Pillars

```
1. Agent Classes
   ├─ CodeAgent (Python code execution)
   └─ ToolCallingAgent (JSON tool calls)

2. Tool System
   ├─ @tool decorator (simple)
   ├─ Tool subclass (complex)
   ├─ MCP integration
   └─ Hugging Face Spaces

3. Model Layer
   ├─ InferenceClientModel (Hugging Face)
   ├─ LiteLLMModel (100+ providers)
   └─ TransformersModel (local)

4. Execution Engines
   ├─ Local Python
   ├─ Docker
   ├─ E2B (cloud sandbox)
   ├─ Modal (serverless)
   └─ WebAssembly (browser)
```

## 📊 Feature Comparison Matrix

| Feature | CodeAgent | ToolCallingAgent |
|---------|-----------|-----------------|
| **Paradigm** | Python code | JSON calls |
| **Efficiency** | 30% faster | Standard |
| **Loops** | ✓ Native | ✗ No |
| **Conditionals** | ✓ Native | ✗ No |
| **State** | ✓ Full | ✗ Limited |
| **Composability** | ✓ Excellent | ○ Good |
| **Legacy Support** | ✗ No | ✓ Yes |
| **When to Use** | Complex logic | Simple workflows |

## 🔧 Installation Quick Reference

```bash
# Basic installation
pip install 'smolagents[toolkit]'

# With LiteLLM (100+ providers)
pip install 'smolagents[toolkit]' litellm

# Production-grade
pip install 'smolagents[toolkit,e2b,modal]' litellm transformers

# Verify installation
python -c "from smolagents import CodeAgent; print('✓ Ready!')"
```

## 🌍 Supported LLM Providers

SmolAgents works with **100+ LLM providers** through LiteLLMModel:

- **OpenAI**: gpt-4, gpt-4-turbo, gpt-3.5-turbo
- **Anthropic**: Claude 3, Claude 3 Sonnet
- **Google**: Gemini, Palm
- **Meta**: Llama via various providers
- **Groq**: Ultra-fast inference
- **Together AI**: Multiple open models
- **Hugging Face**: Inference endpoints
- **Local**: TransformersModel
- And 90+ more...

## 🛠️ Tool Creation Quick Reference

### Simple: @tool Decorator
```python
from smolagents import tool

@tool
def my_tool(param1: str, param2: int) -> str:
    """Tool description for LLM"""
    return f"Result: {param1} {param2}"
```

### Complex: Tool Subclass
```python
from smolagents import Tool

class MyTool(Tool):
    name = "my_tool"
    description = "..."
    inputs = {
        "param1": {"type": "string"}
    }
    output_type = "string"
    
    def forward(self, param1: str) -> str:
        return f"Result: {param1}"
```

## 🎓 Learning Resources

### Official Resources
- **GitHub**: https://github.com/huggingface/smolagents
- **Hugging Face Hub**: Share agents with `agent.push_to_hub()`
- **Community**: Active discussions & examples

### This Guide
- **Comprehensive Guide**: Master all concepts
- **Diagrams**: Visualise architectures
- **Production Guide**: Deploy with confidence
- **Recipes**: Copy-paste implementations

## 🚀 Common Tasks

### Run a Simple Agent
See: **comprehensive_guide.md** → Simple Agents Fundamentals

### Create a Custom Tool
See: **recipes.md** → Custom Tool Creation (Recipes 15-16)

### Build Multi-Agent System
See: **comprehensive_guide.md** → Multi-Agent Systems & **recipes.md** → Recipes 11-12

### Deploy to Production
See: **production_guide.md** → Deployment Options

### Integrate with Specific Service
See: **recipes.md** → API Integration Patterns (Recipes 13-14)

### Debug & Monitor
See: **production_guide.md** → Monitoring & Observability

## 💡 Key Principles

1. **Minimal Abstractions**: ~1,000 lines of core code
2. **Transparency**: Easy to understand and extend
3. **Flexibility**: Works with any LLM provider
4. **Efficiency**: 30% fewer tokens for multi-step tasks
5. **Composability**: Tools stack naturally in code

## 🔒 Security & Best Practices

**Remember**: Always follow these practices:
- Use sandboxing for untrusted code (executor_type="e2b" or "docker")
- Validate all inputs before agent execution
- Implement rate limiting in production
- Monitor token usage for cost control
- Use HTTPS for all API calls
- Audit agent outputs before using them
- Follow principle of least privilege for tool permissions

See: **production_guide.md** → Security Best Practices

## 🤝 Contributing & Community

SmolAgents is open source and community-driven:
- Report issues on GitHub
- Share agents via Hugging Face Hub
- Contribute tools and improvements
- Participate in discussions

## 📞 Getting Help

1. **Concepts unclear?** → Read **comprehensive_guide.md** section
2. **Need a visual?** → Check **diagrams.md**
3. **Want code example?** → Find in **recipes.md**
4. **Deploying?** → Follow **production_guide.md**
5. **Still stuck?** → Check GitHub issues or Hugging Face discussions

---

## 📝 Document Statistics

| Document | Content | Focus |
|----------|---------|-------|
| Comprehensive Guide | ~15,000+ lines | Theory & Concepts |
| Diagrams | 9 sections | Visualisation |
| Production Guide | ~8,000 lines | Operations |
| Recipes | 20 examples | Practical Code |
| **Total** | **~30,000+ lines** | **Complete Reference** |

---

## 🎉 Ready to Get Started?

**Choose your starting point:**

- 🏫 **New to agents?** → Start with [comprehensive_guide.md](./smolagents_comprehensive_guide.md#introduction--philosophy)
- 🔨 **Ready to code?** → Jump to [recipes.md](./smolagents_recipes.md)
- 🚀 **Going to production?** → Read [production_guide.md](./smolagents_production_guide.md)
- 📊 **Want diagrams?** → Browse [diagrams.md](./smolagents_diagrams.md)

---

**SmolAgents**: The framework for building intelligent agents with minimal abstractions and maximum capability. Let's build the future of AI, one line of code at a time.

**Happy agent building! 🤖✨**

