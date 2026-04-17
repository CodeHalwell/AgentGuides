# AG2 (Formerly AutoGen) Guide

> **The Next Generation of AutoGen**

AG2 is the community-driven continuation of the AutoGen framework, designed for building next-generation multi-agent systems.

**Current Version**: 0.11.5 (April 5, 2026) — previously 0.3.2 (November 2025)

## ⚠️ Upcoming Breaking Changes — v1.0 Roadmap

AG2 is executing a formal deprecation roadmap toward v1.0. **Existing code will continue to work until v1.0**, but you should plan for migration:

| Version | Expected Timeline | Status |
|---------|-------------------|--------|
| v0.12 | Current window | Deprecations marked; feedback open |
| v0.13 | ~Q2 2026 | Feedback incorporated; new orchestration |
| v0.14 | ~Q3 2026 | Last release with deprecated features |
| v1.0 | ~Q4 2026 | `autogen.beta` becomes official; old framework → maintenance |

**What to watch:**
- `autogen.beta` is a **parallel, non-breaking** redesign (streaming + event-driven, `MemoryStream` pub/sub) — you can start using it today alongside existing AG2 code
- Model references to `gpt-3.5-turbo`, `gpt-4-vision-preview`, `gpt-4-turbo-preview` are deprecated in v0.12; removed in v0.14

## 🆕 What's New in v0.11.5 (2026)

- **`autogen.beta`**: New streaming/event-driven architecture — every conversation runs on a `MemoryStream` pub/sub event bus; agents safely reusable across concurrent users
- **A2A protocol**: Native Agent-to-Agent protocol support for cross-framework agent communication
- **`A2UIAgent`**: Reference agent combining A2A and A2UI protocols for dynamic agent-driven frontends
- **`AG2 CLI`**: Full CLI for building, running, testing, and deploying multi-agent applications
- **Multi-provider LLM support in beta**: OpenAI, Anthropic, Google Gemini, Alibaba DashScope (Qwen), Ollama
- **`RemyxCodeExecutor`**: Code executor for research paper execution
- **`GroupToolExecutor` async handler**: Improved async workflow support
- **Security fixes**: Multiple CVE patches applied

## 🚀 Quick Start

### Installation

```bash
pip install ag2
```

### Basic Example

```python
from autogen import AssistantAgent, UserProxyAgent

# Create an assistant agent
assistant = AssistantAgent(
    name="assistant",
    llm_config={"config_list": [{"model": "gpt-4", "api_key": "YOUR_API_KEY"}]}
)

# Create a user proxy agent
user_proxy = UserProxyAgent(
    name="user_proxy",
    code_execution_config={"work_dir": "coding"}
)

# Start the conversation
user_proxy.initiate_chat(
    assistant,
    message="Plot a chart of NVDA and TSLA stock price change YTD."
)
```

## 📚 Documentation Structure

- **[Comprehensive Guide](./ag2_comprehensive_guide.md)**: Detailed reference for AG2 concepts and features.
- **[Production Guide](./ag2_production_guide.md)**: Best practices for deploying AG2 systems.
- **[Diagrams](./ag2_diagrams.md)**: Visual architecture of AG2 workflows.
- **[Recipes](./ag2_recipes.md)**: Common patterns and use cases.

## 🔗 Resources

- **PyPI**: [ag2](https://pypi.org/project/ag2/)
- **GitHub**: [ag2ai/ag2](https://github.com/ag2ai/ag2)

---

## 📋 Revision History

| Date | Version | Changes |
|------|---------|---------|
| April 16, 2026 | 0.11.5 | Updated to v0.11.5; added autogen.beta section; documented v1.0 deprecation roadmap; A2A protocol; AG2 CLI; security CVE notes |
| November 2025 | 0.3.2 | Initial guide; multi-agent orchestration; GroupChat patterns |
