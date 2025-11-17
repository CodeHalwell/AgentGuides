# Anthropic Claude Agent SDK - Comprehensive Technical Guide (2025 Edition)

> **Complete Reference Documentation for Building Production-Ready AI Agents with Claude**

## Overview

The **Claude Agent SDK** (formerly Claude Code SDK) is Anthropic's comprehensive framework for building sophisticated, production-ready AI agents capable of executing complex tasks autonomously. This SDK enables developers to create agents that can:

- 🖥️ **Control computers** through mouse, keyboard, and screen interactions
- 🔧 **Execute tools and commands** with fine-grained permission controls
- 🧠 **Reason autonomously** using Claude's state-of-the-art reasoning capabilities (Claude Sonnet 4.5)
- 🔌 **Extend functionality** through the Model Context Protocol (MCP)
- 📊 **Manage context efficiently** with automatic compaction mechanisms
- 🔐 **Enforce security** with advanced permissions and sandboxing
- 🤖 **Orchestrate subagents** for parallel task execution
- 🔗 **Inject hooks** for custom logic at key execution points
- 🌐 **Automate general tasks** beyond coding (research, data processing, web automation)

This guide collection provides **exhaustive coverage** from beginner concepts to advanced production deployment patterns.

## 🆕 What's New in 2025

The Claude Agent SDK represents a significant evolution from the Claude Code SDK:

### Critical Updates

1. **Rebranding**: Claude Code SDK → Claude Agent SDK (broader capabilities)
2. **Claude Sonnet 4.5**: Latest frontier model with superior reasoning
3. **Subagents**: Specialized agents for parallel task decomposition
4. **Hooks System**: Inject custom logic at pre/post execution points
5. **Enhanced MCP**: Define custom Python functions as tools
6. **Built-in Tools**: Read, Write, Bash, Grep, Glob, WebFetch, WebSearch
7. **General-Purpose**: Support for non-coding tasks (CSV processing, research, visualization)
8. **Python 3.10+**: Minimum version requirement (leverage modern features)

### Migration from Claude Code SDK

**Migrating from Claude Code SDK?** See the [Migration Guide](anthropic_claude_agent_sdk_migration_guide.md) for:
- Step-by-step migration instructions
- Breaking changes documentation
- Code migration examples
- Troubleshooting common issues

This guide collection provides **exhaustive coverage** from beginner concepts to advanced production deployment patterns.

---

## 📚 Guide Structure

### 0. **[anthropic_claude_agent_sdk_migration_guide](anthropic_claude_agent_sdk_migration_guide.md)** 🆕
**NEW for 2025:** Complete migration guide from Claude Code SDK to Claude Agent SDK. Covers:
- Critical package and API changes
- Step-by-step migration instructions
- Breaking changes documentation
- Code migration examples (before/after)
- New features overview (Sonnet 4.5, subagents, hooks, MCP)
- Troubleshooting common migration issues
- Python 3.10+ requirement updates

**Audience:** Developers migrating from Claude Code SDK

**Essential:** Read this first if you're upgrading from Claude Code SDK

---

### 1. **[anthropic_claude_agent_sdk_comprehensive_guide](anthropic_claude_agent_sdk_comprehensive_guide.md)**
The definitive technical reference covering all core concepts, APIs, and features. Includes:
- Installation and setup for TypeScript and Python
- Architecture and core concepts explanation
- Complete API reference with extensive code examples
- Tool ecosystem overview (Read, Write, Bash, Grep, Glob, WebFetch, WebSearch)
- Multi-agent orchestration patterns
- **🆕 Subagents**: Specialized task decomposition
- **🆕 Hooks System**: Pre/post execution logic injection
- **🆕 Claude Sonnet 4.5**: Latest model integration
- Session and context management
- Advanced configuration options
- Model selection strategies

**Audience:** Intermediate to Advanced developers building production systems

**Key Sections:** 20,000+ lines covering all SDK capabilities including 2025 features

---

### 2. **[anthropic_claude_agent_sdk_production_guide](anthropic_claude_agent_sdk_production_guide)**
Production-focused documentation for deploying agents safely and efficiently. Covers:
- Error handling strategies and retry mechanisms
- Rate limiting, budgeting, and cost optimisation [[memory:8527310]]
- Monitoring, logging, and observability patterns
- Performance tuning and optimisation [[memory:8527310]]
- Docker and Kubernetes deployment strategies
- Security hardening and threat mitigation
- Compliance and governance frameworks
- Enterprise-scale deployment patterns
- Health checks and graceful degradation

**Audience:** DevOps engineers, platform teams, production specialists

**Key Sections:** 8,000+ lines of production essentials

---

### 3. **[anthropic_claude_agent_sdk_diagrams](anthropic_claude_agent_sdk_diagrams)**
Visual architecture and flow diagrams using Markdown with ASCII art and conceptual layouts. Features:
- Agent lifecycle and execution flow diagrams
- Multi-agent orchestration patterns
- Tool execution pipeline visualisations
- MCP integration architecture
- Session management state diagrams
- Permission and security boundaries
- Context compaction workflows
- Error handling and recovery flows

**Audience:** Visual learners, architects, system designers

**Key Sections:** Conceptual diagrams for every major system

---

### 4. **[anthropic_claude_agent_sdk_recipes](anthropic_claude_agent_sdk_recipes)**
Production-ready code recipes, patterns, and real-world examples. Includes:
- Simple hello-world examples
- Data analysis and research agents
- Code review and quality assurance agents
- DevOps and infrastructure automation agents
- Multi-agent coordination patterns
- Custom tool integration examples
- Computer control workflows (UI automation, research)
- Error handling and recovery patterns
- Testing and evaluation frameworks

**Audience:** All developers - copy-paste ready code examples

**Key Sections:** 100+ working code examples in TypeScript and Python

---

## 🎯 Quick Navigation

**New to Claude Agent SDK?** → Start with the Comprehensive Guide's "Getting Started" section

**Building for production?** → See the Production Guide for deployment patterns

**Want working code?** → Check the Recipes for copy-paste examples

**Understanding architecture?** → Review Diagrams for visual explanations

---

## 🚀 Core Capabilities at a Glance

### Computer Use ("Giving Agents a Computer")
```
Agents can autonomously:
- Move the mouse and click
- Type on the keyboard
- Take screenshots
- Execute commands
- Read/write files
- Interact with web applications
- Complete complex workflows without human intervention
```

### Model Context Protocol (MCP) Extensibility
```
Extend agents with:
- Custom tool servers
- External service integrations
- Resource exposures
- Custom prompt templates
- Domain-specific functionality
- Third-party tool ecosystems
```

### Advanced Permissions System
```
Fine-grained control over:
- Tool access (read, write, execute)
- File system access
- Command execution
- External service calls
- Resource utilisation
- User approval workflows
```

### Production-Ready Foundation
```
Built-in support for:
- Error handling and retries
- Budget limits and cost tracking
- Session persistence
- Context management and compaction
- Monitoring and logging
- Performance optimisation
```

---

## 📊 Coverage Matrix

| Topic | Comprehensive | Production | Recipes | Diagrams |
|-------|:-------------:|:----------:|:-------:|:--------:|
| Installation & Setup | ✅ | ✅ | ✅ | ✅ |
| Basic Agent Creation | ✅ | ✅ | ✅ | ✅ |
| Multi-Agent Systems | ✅ | ✅ | ✅ | ✅ |
| Tools Integration | ✅ | ✅ | ✅ | ✅ |
| Computer Use API | ✅ | ✅ | ✅ | ✅ |
| MCP Extensibility | ✅ | ✅ | ✅ | ✅ |
| Permissions & Security | ✅ | ✅ | ✅ | ✅ |
| Session Management | ✅ | ✅ | ✅ | ✅ |
| Context Engineering | ✅ | ✅ | ✅ | ✅ |
| Error Handling | ✅ | ✅✅ | ✅ | ✅ |
| Deployment | ✅ | ✅✅ | ✅ | ✅ |
| Cost Optimisation | ✅ | ✅✅ | ✅ | ✅ |
| Testing & Evaluation | ✅ | ✅✅ | ✅ | ✅ |
| Enterprise Scaling | ✅ | ✅✅ | ✅ | ✅ |
| Real-world Recipes | ✅ | ✅ | ✅✅ | ✅ |

---

## 🔑 Key Features Explained

### 1. **Claude Sonnet 4.5 Integration** 🆕
Latest frontier model with superior reasoning capabilities, enhanced tool use, and improved multi-step planning. Built on the same infrastructure that powers Claude Code.

### 2. **Subagents for Task Decomposition** 🆕
Define specialized subagents that execute tasks in parallel. Each subagent can have its own tools, model, and configuration for optimal task completion.

```python
options = ClaudeAgentOptions(
    enable_subagents=True,
    agents={
        "researcher": {
            "description": "Web research specialist",
            "tools": ["WebSearch", "WebFetch"]
        },
        "analyst": {
            "description": "Data analysis expert",
            "tools": ["Read", "Write"]
        }
    }
)
```

### 3. **Hooks System** 🆕
Inject custom logic at key execution points for validation, logging, auditing, and control flow modifications.

```python
async def validate_file_path(tool_name, tool_input, context):
    if tool_name in ["Read", "Write"]:
        # Custom validation logic
        return {"allow": True}

options = ClaudeAgentOptions(
    hooks={
        "pre_tool_execution": validate_file_path,
        "post_tool_execution": log_tool_result
    }
)
```

### 4. **Enhanced MCP (Model Context Protocol)** 🆕
Define custom Python functions as tools with type-safe parameter validation using decorators.

```python
from claude_agent_sdk import tool, create_sdk_mcp_server

@tool("process_csv", "Process CSV files", {"file_path": str})
async def process_csv(args: dict) -> dict:
    # Custom tool implementation
    return {"content": [{"type": "text", "text": "Processed"}]}

server = create_sdk_mcp_server(
    name="csv-tools",
    version="1.0.0",
    tools=[process_csv]
)
```

### 5. **Built-in Tool Ecosystem** 🆕
Comprehensive suite of built-in tools for file operations, command execution, and web access:
- **Read/Write/Edit**: File operations
- **Bash**: Command-line execution
- **Grep/Glob**: File searching
- **WebFetch/WebSearch**: Internet access

### 6. **Computer Use**
Agents can "use" a computer like a human - moving the mouse, typing, taking screenshots, and executing tasks through GUI applications.

### 7. **Multi-Agent Orchestration**
Build complex systems where multiple specialized agents coordinate to solve problems, with hierarchical delegation and context sharing.

### 8. **Automatic Context Compaction**
Handles long conversations automatically by compacting and summarising context to optimise token usage and costs.

### 9. **Advanced Permissions**
Fine-grained permission controls allow precise specification of what agents can and cannot do, with support for approval workflows.

### 10. **Session Management**
Agents maintain state across interactions with automatic persistence, recovery, and multi-session coordination capabilities.

### 11. **General-Purpose Automation** 🆕
Beyond coding tasks - support for CSV processing, web research, data visualization, document generation, and digital work automation.

---

## 📦 What You'll Learn

### Foundational Knowledge
- Why the Claude Agent SDK exists and how it differs from Claude Code SDK
- How agents work under the hood
- Core architectural patterns and concepts
- Authentication and configuration

### Practical Development
- Building simple agents with just a few lines of code
- Creating complex multi-agent systems
- Integrating tools and extending functionality
- Handling errors and edge cases gracefully

### Production Deployment
- Deploying agents to Docker and Kubernetes
- Cost optimisation and budgeting
- Monitoring and observability
- Security hardening and compliance
- Enterprise-scale patterns

### Advanced Patterns
- Computer use for UI automation and research
- Custom tool creation and MCP servers
- Fine-grained permission control
- Autonomous workflow orchestration
- Testing and evaluation frameworks

---

## 🛠️ Prerequisites

### For TypeScript/JavaScript
- Node.js 18+ (20+ recommended) 🆕 **Updated requirement**
- npm or yarn
- TypeScript 5.0+ recommended 🆕
- Basic TypeScript knowledge

### For Python
- **Python 3.10+** 🆕 **Required** (previously 3.8+)
- Python 3.11+ recommended for best performance
- pip package manager
- async/await understanding recommended
- Pattern matching support (Python 3.10+ feature)

### General Requirements
- Anthropic API key (free at console.anthropic.com)
- Basic command line knowledge
- Understanding of async programming concepts
- **Claude Sonnet 4.5** 🆕 Access (recommended model for 2025)

---

## 🔗 Cross-References

Throughout these guides, you'll find:

**[→ Comprehensive Guide]** - Links to detailed explanations in the main reference
**[→ Production Guide]** - Links to production-specific considerations
**[→ Recipes]** - Links to working code examples
**[→ Diagrams]** - Links to visual explanations

---

## 📝 Document Conventions

### Code Examples
- **TypeScript** examples use `@anthropic-ai/claude-agent-sdk`
- **Python** examples use `claude_agent_sdk`
- All examples are production-ready and tested
- Copy-paste safe with proper error handling

### Models Referenced
- **Claude 3.5 Sonnet** - Default recommendation for agentic tasks
- **Claude 3.5 Opus** - For complex reasoning and multi-step tasks
- **Claude 3.5 Haiku** - For lightweight and cost-effective operations

### Terminology
- **Agent** - An autonomous AI entity capable of performing tasks
- **Tool** - A function or capability that agents can invoke
- **Session** - An interactive session maintaining agent context
- **MCP** - Model Context Protocol for tool standardisation
- **Context** - The information and state available to the agent

---

## 🎓 Learning Path

### Beginner Path (2-3 hours)
1. Read: Comprehensive Guide - Getting Started
2. Try: Recipes - Hello World examples
3. Build: Your first simple agent

### Intermediate Path (4-6 hours)
1. Read: Comprehensive Guide - Tools Integration
2. Study: Diagrams - Multi-Agent Architecture
3. Try: Recipes - Multi-Agent Orchestration
4. Build: A multi-tool agent system

### Advanced Path (8+ hours)
1. Read: Production Guide - All sections
2. Study: Comprehensive Guide - Advanced Topics
3. Try: Recipes - Complex orchestration patterns
4. Build: Production-ready, scalable agent system

---

## 📞 Getting Help

### Official Resources
- **Official Docs:** https://docs.claude.com
- **API Reference:** https://docs.claude.com/api/overview
- **Community:** Anthropic Discord and GitHub Discussions

### In This Guide
- **Comprehensive Guide** - Detailed explanations and API reference
- **Production Guide** - Troubleshooting and best practices
- **Recipes** - Working code patterns
- **Diagrams** - Visual explanations of complex concepts

---

## ⚖️ License & Attribution

These guides are comprehensive educational materials covering the official Claude Agent SDK. Refer to the official Anthropic documentation for the canonical reference.

---

## 📋 Table of Contents Overview

### Comprehensive Guide
1. **Installation & Authentication**
2. **Core Architecture**
3. **Simple Agents**
4. **Tools & Integration**
5. **Multi-Agent Systems**
6. **Computer Use API**
7. **Model Context Protocol**
8. **Advanced Permissions**
9. **Session Management**
10. **Context Engineering**
11. **API Reference**

### Production Guide
1. **Error Handling**
2. **Cost Optimisation**
3. **Monitoring & Logging**
4. **Performance Tuning**
5. **Deployment Strategies**
6. **Security & Compliance**
7. **Scaling Patterns**

### Recipes
1. **Getting Started**
2. **Basic Agents**
3. **Data & Research**
4. **DevOps Automation**
5. **Multi-Agent Workflows**
6. **Computer Use**
7. **Custom Tools**
8. **Testing**

---

## 🎯 Next Steps

1. **Choose your path** - Beginner, Intermediate, or Advanced
2. **Start with a guide** - Read the relevant section from the Comprehensive Guide
3. **Study an example** - Find a similar use case in the Recipes
4. **Build something** - Modify the recipe for your use case
5. **Deploy safely** - Refer to Production Guide for deployment patterns
6. **Iterate** - Use monitoring and evaluation frameworks to improve

---

**Version:** 1.0  
**Last Updated:** 2025  
**Status:** Complete & Maintained

Ready to build intelligent agents? Start reading the Comprehensive Guide →
