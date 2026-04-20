---
title: "AG2 (AutoGen 0.2.x Community Fork) - Complete Documentation Suite"
description: "A Comprehensive, Exhaustive Technical Guide for Building Autonomous Multi-Agent AI Systems"
framework: autogen
language: python
---

# AG2 (AutoGen 0.2.x Community Fork) - Complete Documentation Suite

**A Comprehensive, Exhaustive Technical Guide for Building Autonomous Multi-Agent AI Systems**

## Overview

This documentation suite provides everything needed to master AG2, from absolute beginner to advanced practitioner. AG2 is a vibrant, community-driven fork of AutoGen 0.2.x that streamlines building sophisticated multi-agent AI systems with LLMs, tool integration, and flexible orchestration patterns.

**Key Philosophy:** Learn by doing. Each guide includes conceptual explanations, 3+ working examples per topic, troubleshooting guides, and production-ready patterns.

---

## 📚 Guide Structure

### 1. **[ag2_comprehensive_guide.md](./autogen_comprehensive_guide/)** - Core Knowledge
**Target Audience:** Everyone (start here if new)

This is the main technical reference covering:

- **Core Fundamentals**: Installation, configuration, LLM providers, core classes (ConversableAgent, AssistantAgent, UserProxyAgent, GroupChat)
- **Simple Agents**: Creating your first agent, configuration patterns, code execution environments, function calling, human-in-the-loop workflows
- **Multi-Agent Systems**: Two-agent conversations, group chats, speaker selection strategies, nested chats, sequential workflows, state management
- **Tools Integration**: Function registration, type hints, schema generation, tool execution control, error handling
- **Structured Output**: JSON mode, Pydantic models, schema enforcement
- **Agentic Patterns**: ReAct implementation, self-refinement loops, autonomous workflows
- **Memory Systems**: Conversation history, context preservation, message filtering
- **Context Engineering**: System message design, few-shot prompting, context optimisation

**Format:** Markdown with extensive code blocks, detailed explanations, and step-by-step tutorials

**Estimated Time:** 2-4 hours to read thoroughly; 6-8 hours to follow all code examples

---

### 2. **[ag2_production_guide.md](./autogen_production_guide/)** - Enterprise-Grade Patterns
**Target Audience:** Developers building production systems

Focuses on deploying AG2 at scale:

- **Logging and Debugging**: Comprehensive logging setup, message inspection, custom logging
- **Cost Tracking**: Token monitoring, usage optimisation, message pruning, dynamic context management
- **Error Handling**: Retry strategies, graceful degradation, validation, sanitisation
- **Testing Strategies**: Unit testing, integration testing, mocking frameworks
- **Deployment Patterns**: Local deployment, Docker containers, REST API services (A2A protocol)
- **Performance Optimisation**: Parallel execution, connection pooling, response caching
- **Security**: API key management, input validation, rate limiting
- **Monitoring**: Metrics collection, health checks, observability
- **Integration**: LangChain compatibility, LlamaIndex integration, async patterns
- **Async Execution**: Concurrent task processing, non-blocking patterns

**Format:** Practical code examples focusing on production-ready implementations

**Estimated Time:** 2-3 hours; 4-6 hours if implementing patterns

---

### 3. **[ag2_recipes.md](./autogen_recipes/)** - Working Tutorials
**Target Audience:** Hands-on learners, project builders

Ten complete, self-contained working examples:

1. **Customer Support Bot** - Two-agent triage and resolution system
2. **Code Review Team** - Multi-agent collaborative code review
3. **Data Analysis Pipeline** - Multi-stage data validation and analysis
4. **Content Creation Workflow** - End-to-end blog post creation
5. **Research Team** - Multi-perspective information gathering
6. **Multi-Stage Decision Making** - Hierarchical approval workflow
7. **Document Q&A System** - Information retrieval and synthesis
8. **Collaborative Learning** - Teachers and learners collaborating
9. **API Integration** - Agents calling external APIs
10. **Autonomous Task Scheduler** - Project planning and execution

**Format:** Copy-paste ready Python scripts; can run immediately

**Estimated Time:** 30 mins per recipe; use as templates for your own projects

---

### 4. **[ag2_diagrams.md](./autogen_diagrams/)** - Visual Architecture
**Target Audience:** Visual learners, architects

Comprehensive ASCII diagrams and flowcharts:

- Core framework architecture and component relationships
- Agent communication patterns and message flows
- Multi-agent orchestration and speaker selection
- Nested chats and hierarchical execution
- Sequential chat pipelines
- Tool integration workflows
- Code execution models (local vs Docker)
- State management architecture
- Production deployment patterns
- End-to-end system workflows

**Format:** ASCII art diagrams with detailed annotations

**Estimated Time:** 1-2 hours; excellent as reference material

---

## 🚀 Getting Started

### Installation (2 minutes)

```bash
# Create virtual environment
python -m venv ag2_env
source ag2_env/bin/activate  # On Windows: ag2_env\Scripts\activate

# Install AG2 with OpenAI support
pip install ag2[openai]
```

### Set Up API Keys (2 minutes)

Create `OAI_CONFIG_LIST` file:

```json
[
    {
        "model": "gpt-4",
        "api_key": "sk-your-key-here"
    }
]
```

Or use environment variables:

```bash
export OPENAI_API_KEY="sk-your-key"
```

### Run Your First Agent (5 minutes)

```python
from autogen import ConversableAgent, LLMConfig

llm_config = LLMConfig.from_json("OAI_CONFIG_LIST")

agent = ConversableAgent(
    name="assistant",
    llm_config=llm_config,
    human_input_mode="NEVER"
)

print("AG2 is ready!")
```

---

## 📖 Learning Paths

### Path 1: Complete Beginner (Week 1)

1. **Day 1**: Read Core Fundamentals section (ag2_comprehensive_guide.md)
2. **Day 2**: Follow Recipe 1 (Customer Support Bot) and Recipe 7 (Document Q&A)
3. **Day 3**: Experiment with two-agent conversations (Multi-Agent Systems section)
4. **Day 4**: Build group chat with 3+ agents
5. **Day 5**: Integrate a function using @register_function
6. **Day 6**: Deploy simple REST API using A2A protocol
7. **Day 7**: Review architecture diagrams, plan first project

### Path 2: Intermediate Developer (Week 2-3)

1. **Week 1 recap**: Review nested chats and sequential workflows
2. **Advanced patterns**: Custom speaker selection, state management
3. **Recipes 2-5**: Implement code review team, data pipeline, content workflow
4. **Production setup**: Add logging, cost tracking, error handling (ag2_production_guide.md)
5. **Testing**: Write unit and integration tests
6. **Real project**: Start building your own multi-agent system

### Path 3: Advanced/Enterprise (Week 4+)

1. **Scaling patterns**: Docker deployment, async execution
2. **Integration**: LangChain, LlamaIndex, custom APIs
3. **Recipes 6-10**: Complex workflows, API integration, task scheduling
4. **Performance**: Optimisation, caching, connection pooling
5. **Security**: Input validation, rate limiting, monitoring
6. **Production deployment**: Full monitoring stack, health checks

---

## 🔍 Quick Reference by Use Case

### "I want to..."

**Build a simple chatbot**
→ Section: "Simple Agents" (ag2_comprehensive_guide.md)
→ Recipe: #1 Customer Support Bot

**Create a multi-agent system**
→ Section: "Multi-Agent Systems" (ag2_comprehensive_guide.md)
→ Diagrams: "Multi-Agent Orchestration" (ag2_diagrams.md)

**Integrate external APIs**
→ Recipe: #9 API Integration
→ Section: "Tool Integration" (ag2_comprehensive_guide.md)

**Deploy to production**
→ ag2_production_guide.md (all sections)
→ Recipe: #1, #2, #3 (for deployment examples)

**Understand architecture**
→ ag2_diagrams.md (all sections)
→ Section: "Core Fundamentals" (ag2_comprehensive_guide.md)

**Debug issues**
→ Section: "Troubleshooting Common Issues" (ag2_comprehensive_guide.md)
→ ag2_production_guide.md "Logging and Debugging"

**Optimise costs**
→ ag2_production_guide.md "Cost Tracking and Token Optimisation"
→ Section: "Token Limits and Cost Control" (ag2_comprehensive_guide.md)

**Ensure security**
→ ag2_production_guide.md "Security Considerations"

---

## 💡 Key Concepts at a Glance

| Concept | Definition | Where to Learn |
|---------|-----------|-----------------|
| **ConversableAgent** | Base class for all agents; handles bidirectional messaging | Comprehensive: "Core Classes" |
| **LLMConfig** | Centralised LLM provider configuration | Comprehensive: "Configuring LLM Providers" |
| **GroupChat** | Container for multi-agent conversations | Comprehensive: "Multi-Agent Systems" |
| **Speaker Selection** | Method for choosing next agent to speak | Diagrams: "Group Chat Speaker Selection" |
| **Nested Chats** | Hierarchical conversations triggered conditionally | Diagrams: "Nested Chats Architecture" |
| **Tool Registration** | System for agents to discover and call functions | Comprehensive: "Tool Integration" |
| **Code Execution** | Local or Docker-based code running | Comprehensive: "Code Execution Environments" |
| **State Management** | Shared context variables across agents | Diagrams: "State Management Architecture" |
| **A2A Protocol** | REST API for distributed agent communication | Production: "REST API Deployment" |

---

## 📊 Comparison: AG2 vs Alternatives

| Feature | AG2 | AutoGen 0.4 | LangChain Agents |
|---------|-----|------------|-----------------|
| **Ease of learning** | High | Medium | Medium |
| **Maturity** | High (0.2.x stable) | Medium (new arch) | High |
| **Multi-agent support** | Excellent | Excellent | Limited |
| **Code execution** | Local + Docker | Async model | Via tools only |
| **Structured output** | Native Pydantic | Native | Via tools |
| **Community** | Active fork | Microsoft-led | Large ecosystem |
| **Migration from 0.2.x** | Easy | Breaking changes | N/A |

---

## 🎓 Learning Resources

### Within This Suite

- **Conceptual**: Read first 2 sections of comprehensive_guide.md
- **Practical**: Follow recipes in order (1-10)
- **Visual**: Study diagrams.md before complex topics
- **Reference**: Use production_guide.md for specific problems

### External Resources

- **Official AG2 Repository**: https://github.com/ag2ai/ag2
- **AG2 Discord Community**: https://discord.gg/pAbnFJrkgZ
- **GitHub Discussions**: https://github.com/ag2ai/ag2/discussions
- **Examples Repository**: https://github.com/ag2ai/ag2-examples

---

## 🛠️ Setting Up Your Development Environment

### Recommended Setup

```bash
# 1. Create project directory
mkdir my-ag2-project
cd my-ag2-project

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install ag2[openai]
pip install python-dotenv  # For environment variables
pip install pytest          # For testing
pip install black          # For code formatting

# 4. Create .env file
cat > .env << EOF
OPENAI_API_KEY=sk-your-key
# Add other API keys as needed
EOF

# 5. Create directory structure
mkdir -p agents utils examples notebooks tests

# 6. Create requirements.txt
pip freeze > requirements.txt
```

### Project Structure

```
my-ag2-project/
├── agents/              # Agent definitions
│   ├── base.py
│   ├── assistants.py
│   └── coordinators.py
├── utils/               # Helper utilities
│   ├── config.py
│   ├── logging_config.py
│   └── api_clients.py
├── examples/            # Working examples
│   ├── example_1_simple.py
│   ├── example_2_group_chat.py
│   └── ...
├── notebooks/           # Jupyter notebooks for experimentation
├── tests/               # Unit and integration tests
├── .env                 # Environment variables (add to .gitignore)
├── requirements.txt     # Python dependencies
├── README.md           # Project documentation
└── main.py             # Main entry point
```

---

## ✅ Verification Checklist

After completing this guide, you should be able to:

- [ ] Install and configure AG2
- [ ] Create a simple ConversableAgent
- [ ] Set up LLM configuration (OpenAI, Anthropic, local)
- [ ] Build a two-agent conversation
- [ ] Implement a group chat with 3+ agents
- [ ] Register and use functions as tools
- [ ] Execute Python code locally and in Docker
- [ ] Use nested chats for hierarchical workflows
- [ ] Create sequential multi-stage pipelines
- [ ] Implement custom speaker selection
- [ ] Deploy as REST API using A2A protocol
- [ ] Add comprehensive logging and monitoring
- [ ] Track costs and optimize token usage
- [ ] Write tests for agents and conversations
- [ ] Implement production-grade error handling
- [ ] Integrate with external APIs
- [ ] Use Pydantic models for structured output
- [ ] Manage shared state across agents
- [ ] Implement memory and conversation management
- [ ] Deploy to production (Docker, cloud)

---

## 🐛 Troubleshooting Quick Links

| Issue | Location |
|-------|----------|
| Installation fails | Comprehensive: "Installation and Setup" |
| API key not working | Comprehensive: "Configuring LLM Providers" |
| Agent not responding | Production: "Error Handling and Resilience" |
| High API costs | Production: "Cost Tracking and Token Optimisation" |
| Code execution errors | Comprehensive: "Code Execution Environments" |
| Complex workflows needed | Recipes: #2, #4, #5, #10 |
| Deployment questions | Production: "Deployment Patterns" |
| Performance issues | Production: "Performance Optimisation" |
| Security concerns | Production: "Security Considerations" |

---

## 📝 Contributing

Found an error or have suggestions? Contributions are welcome!

1. Check existing issues
2. Create a pull request with improvements
3. Share examples via GitHub discussions
4. Report bugs with reproduction steps

---

## 📄 License

This documentation suite is provided as-is for educational purposes. AG2 is Apache 2.0 licensed.

---

## 🙏 Acknowledgments

This comprehensive guide draws from:

- Official AG2 documentation and examples
- Community-contributed patterns and best practices
- Production deployment experiences from enterprise users
- Extensive testing and real-world application development

---

## 🔗 Quick Links

| Resource | Link |
|----------|------|
| **AG2 Repository** | https://github.com/ag2ai/ag2 |
| **PyPI Package** | https://pypi.org/project/ag2/ |
| **Discord Community** | https://discord.gg/pAbnFJrkgZ |
| **GitHub Discussions** | https://github.com/ag2ai/ag2/discussions |
| **Issue Tracker** | https://github.com/ag2ai/ag2/issues |

---

## 📞 Support

- **Questions**: Ask in GitHub Discussions or Discord
- **Bugs**: Report in GitHub Issues with reproduction steps
- **General Help**: Check troubleshooting sections in guides

---

## 🎯 Next Steps

1. **Right now**: Install AG2 (5 minutes)
2. **Next**: Read first 3 sections of ag2_comprehensive_guide.md (30 minutes)
3. **Then**: Follow Recipe #1 (Customer Support Bot) (30 minutes)
4. **After**: Build your first project using patterns from recipes

**Estimated time to productivity: 2-3 hours**

---

**Happy building with AG2! 🚀**

*Last updated: April 2026*
*Documentation version: 2.0 (AG2 0.12.0)*

## ⚠️ Breaking Changes in AG2 v0.4+ (Migration from 0.2.x)

AG2 has undergone significant architectural changes since the 0.2.x series this guide originally covered. The package is available as both `pyautogen` and `ag2` on PyPI; the `ag2` package is the recommended install going forward.

```bash
pip install ag2[openai]
```

Key changes since 0.2.x:
- **Pydantic v2** is now required; Pydantic v1 is no longer supported.
- **Python 3.10–3.13** required; Python 3.9 is no longer supported.
- **`ConversableAgent` API** is largely stable but several internal hooks have moved — refer to the [AG2 migration docs](https://ag2ai.github.io/ag2/docs/Migration-Guide) for details.
- **GroupChat speaker selection** now supports LLM-based, round-robin, random, and custom callable selectors via `speaker_selection_method`.
- **Async-first orchestration** (`a_initiate_chat`, `a_run`) is recommended for production workloads.
- **Agent-to-Agent (A2A) protocol** is available via the `ag2.interop` module for cross-framework communication.

> **Current version:** AG2 0.12.0 (April 17, 2026). The guide prose and code examples reference 0.2.x patterns; the breaking changes section above documents the most critical differences. A full rewrite of examples for 0.12.x is deferred — see open items below.

## 📋 Revision History

| Date | Version | Summary of changes | Reviewer |
|------|---------|-------------------|----------|
| 2026-04-20 | 0.12.0 | Version metadata updated to 0.12.0; added breaking-changes section for 0.4+ migration (Pydantic v2, Python 3.10+, async-first, A2A interop); flagged prose/code examples as 0.2.x-era pending full rewrite | Claude routine |
| November 2024 | 0.2.x | Initial guide; ConversableAgent; GroupChat; tools; sequential workflows |

---

### Open Items (Human Review Required)

- Full example rewrite for AG2 0.12.x API is out of scope for this routine pass; prose and code blocks still reference 0.2.x patterns. Recommend a dedicated rewrite sprint once 0.12.x is declared fully stable (currently approaching 1.0).

### Advanced Guides
- autogen_advanced_multi_agent.md
- autogen_observability_python.md

