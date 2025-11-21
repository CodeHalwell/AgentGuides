# AG2 (Formerly AutoGen) Guide

> **The Next Generation of AutoGen**

AG2 is the community-driven continuation of the AutoGen framework, designed for building next-generation multi-agent systems.

**Current Version**: 0.3.2

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
