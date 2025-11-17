# Migration Guide: Claude Code SDK → Claude Agent SDK (Python)

**Version:** 2025.1
**Target Audience:** Developers migrating from Claude Code SDK to Claude Agent SDK
**Status:** Official Migration Guide

---

## Table of Contents

1. [Overview](#overview)
2. [Critical Changes](#critical-changes)
3. [Step-by-Step Migration](#step-by-step-migration)
4. [Breaking Changes](#breaking-changes)
5. [New Features in 2025](#new-features-in-2025)
6. [Code Migration Examples](#code-migration-examples)
7. [Troubleshooting](#troubleshooting)

---

## Overview

### What Changed?

The **Claude Code SDK** has been rebranded and significantly expanded into the **Claude Agent SDK** to reflect its broader capabilities beyond coding tasks. This evolution represents Anthropic's commitment to building a comprehensive framework for general-purpose autonomous agents.

### Why the Change?

**Claude Code SDK** focused primarily on:
- Code analysis and generation
- Development workflows
- Software engineering tasks

**Claude Agent SDK** now supports:
- General-purpose task automation
- CSV processing and data manipulation
- Web research and information gathering
- Visualization building
- Digital work automation
- Complex multi-agent orchestration
- **All previous coding capabilities**

### Timeline

- **Pre-2025**: Claude Code SDK (coding-focused)
- **2025+**: Claude Agent SDK (general-purpose agents)
- **Support**: Claude Code SDK will be deprecated in Q4 2025

---

## Critical Changes

### 1. Package Name Change

```python
# OLD - Claude Code SDK
from claude_code import query

# NEW - Claude Agent SDK
from claude_agent_sdk import query
```

### 2. Python Version Requirement

- **Minimum**: Python 3.10+ (previously 3.8+)
- **Recommended**: Python 3.11+ for best performance
- **Reason**: Leverage modern async improvements and type hints

```bash
# Check your Python version
python --version

# Should output: Python 3.10.0 or higher
```

### 3. Model Naming Updates

```python
# OLD
model = "claude-3-5-sonnet-20241022"

# NEW - Claude Sonnet 4.5 (recommended)
model = "claude-sonnet-4-5"

# Still supported for compatibility
model = "claude-3-5-sonnet-20241022"
```

### 4. Enhanced Built-in Tools

**New Built-in Tools:**
- `Read` - File reading operations
- `Write` - File writing operations
- `Edit` - File editing operations
- `Bash` - Command-line execution
- `Grep` - Content searching
- `Glob` - File pattern matching
- `WebFetch` - Web content retrieval
- `WebSearch` - Internet search capabilities

**Code Example:**

```python
from claude_agent_sdk import query

# Agent automatically selects appropriate tools
async for message in query(
    prompt="Search the web for the latest TypeScript features and create a summary document"
):
    print(message)
```

---

## Step-by-Step Migration

### Step 1: Update Dependencies

**Remove Old Package:**

```bash
pip uninstall claude-code-sdk
```

**Install New Package:**

```bash
pip install claude-agent-sdk

# Or with all extras
pip install claude-agent-sdk[full]
```

**Update requirements.txt:**

```txt
# OLD
claude-code-sdk>=0.9.0

# NEW
claude-agent-sdk>=1.0.0
python-dotenv>=1.0.0
pydantic>=2.0.0
aiohttp>=3.9.0
```

### Step 2: Update Import Statements

**Find and Replace:**

```python
# OLD
from claude_code import query, ClaudeAgentOptions
from claude_code.tools import tool, create_mcp_server

# NEW
from claude_agent_sdk import query, ClaudeAgentOptions
from claude_agent_sdk.tools import tool, create_sdk_mcp_server
```

**Automated Migration Script:**

```python
# migration_helper.py
import os
import re
from pathlib import Path

def migrate_imports(file_path: Path):
    """Automatically update import statements."""
    with open(file_path, 'r') as f:
        content = f.read()

    # Replace package imports
    content = content.replace('from claude_code', 'from claude_agent_sdk')
    content = content.replace('import claude_code', 'import claude_agent_sdk')

    # Replace specific tool imports
    content = content.replace('create_mcp_server', 'create_sdk_mcp_server')

    with open(file_path, 'w') as f:
        f.write(content)

    print(f"✓ Migrated: {file_path}")

# Usage
for py_file in Path('.').rglob('*.py'):
    if 'venv' not in str(py_file) and 'node_modules' not in str(py_file):
        migrate_imports(py_file)
```

### Step 3: Update Configuration

**Environment Variables:**

```bash
# .env file - NO CHANGES REQUIRED
ANTHROPIC_API_KEY=sk-ant-your-key-here
ANTHROPIC_MODEL=claude-sonnet-4-5
```

**Python Configuration:**

```python
# OLD
from claude_code import ClaudeAgentOptions

options = ClaudeAgentOptions(
    model="claude-3-5-sonnet-20241022"
)

# NEW - Enhanced with 2025 features
from claude_agent_sdk import ClaudeAgentOptions

options = ClaudeAgentOptions(
    model="claude-sonnet-4-5",  # Latest model
    max_budget_usd=5.0,          # Cost controls
    allowed_tools=[              # Fine-grained tool permissions
        "Read", "Write", "WebSearch", "Bash"
    ],
    hooks={                      # NEW: Hooks system
        "pre_tool_execution": validate_tool_input,
        "post_tool_execution": log_tool_result
    },
    enable_subagents=True        # NEW: Subagent support
)
```

### Step 4: Update Code Patterns

**Basic Query Pattern - UNCHANGED:**

```python
# This pattern works in both versions
import asyncio
from claude_agent_sdk import query

async def main():
    async for message in query(prompt="Analyze this code"):
        print(message)

asyncio.run(main())
```

**Advanced Patterns - ENHANCED:**

```python
# NEW: Leverage 2025 features
from claude_agent_sdk import query, ClaudeAgentOptions

async def advanced_agent():
    options = ClaudeAgentOptions(
        model="claude-sonnet-4-5",
        system_prompt="You are a research and analysis specialist.",
        allowed_tools=["WebSearch", "WebFetch", "Read", "Write"],
        enable_subagents=True  # NEW
    )

    async for message in query(
        prompt="""Research the latest trends in AI agents,
        create a summary document, and generate visualizations.""",
        options=options
    ):
        if message.type == "tool_call":
            print(f"Using tool: {message.tool_name}")
        elif message.type == "assistant":
            print(message.content)
```

---

## Breaking Changes

### 1. MCP Server Creation

**BREAKING:** Function renamed for clarity

```python
# OLD
from claude_code.tools import create_mcp_server

server = create_mcp_server(
    name="my-tools",
    tools=[my_tool]
)

# NEW
from claude_agent_sdk.tools import create_sdk_mcp_server

server = create_sdk_mcp_server(
    name="my-tools",
    version="1.0.0",  # NEW: Version required
    tools=[my_tool]
)
```

### 2. Message Types

**ENHANCED:** New message subtypes added

```python
# OLD
if message.type == "system":
    print("System message")

# NEW - More granular
if message.type == "system":
    if message.subtype == "subagent_start":
        print("Subagent started")
    elif message.subtype == "subagent_end":
        print("Subagent completed")
    elif message.subtype == "hook_execution":
        print("Hook executed")
```

### 3. Permission Callbacks

**ENHANCED:** Additional context provided

```python
# OLD
async def can_use_tool(tool_name: str, tool_input: dict):
    return {"behavior": "allow"}

# NEW - Enhanced with context
async def can_use_tool(tool_name: str, tool_input: dict, context: dict):
    # context includes: session_id, previous_tools, cost_so_far, etc.
    if context.get("cost_so_far", 0) > 10.0:
        return {"behavior": "deny", "message": "Budget exceeded"}
    return {"behavior": "allow"}
```

### 4. Python 3.10+ Required Features

**Leverage new Python features:**

```python
# Pattern matching (Python 3.10+)
match message.type:
    case "assistant":
        print(message.content)
    case "tool_call":
        print(f"Calling: {message.tool_name}")
    case "error":
        print(f"Error: {message.error}")
    case _:
        pass

# Union types (Python 3.10+)
from typing import Literal

MessageType = Literal["assistant", "user", "system", "tool_call"]
```

---

## New Features in 2025

### 1. Claude Sonnet 4.5 Integration

**Frontier Model Capabilities:**

```python
from claude_agent_sdk import query, ClaudeAgentOptions

# Use the latest Claude Sonnet 4.5
options = ClaudeAgentOptions(
    model="claude-sonnet-4-5",
    temperature=1.0
)

async for message in query(
    prompt="Design a complete microservices architecture",
    options=options
):
    print(message)
```

**Benefits:**
- Superior reasoning for complex tasks
- Enhanced multi-step planning
- Better context understanding
- Improved tool use accuracy

### 2. Subagents for Task Decomposition

**NEW FEATURE:** Specialized subagents for parallel execution

```python
from claude_agent_sdk import query, ClaudeAgentOptions

options = ClaudeAgentOptions(
    model="claude-sonnet-4-5",
    enable_subagents=True,
    agents={
        "researcher": {
            "description": "Web research and information gathering",
            "tools": ["WebSearch", "WebFetch", "Read"],
            "model": "claude-sonnet-4-5"
        },
        "analyst": {
            "description": "Data analysis and insights",
            "tools": ["Read", "Write"],
            "model": "claude-3-5-sonnet"
        },
        "visualizer": {
            "description": "Create charts and visualizations",
            "tools": ["Write", "Bash"],
            "model": "claude-3-5-haiku"
        }
    }
)

async for message in query(
    prompt="""Research AI safety trends, analyze the data,
    and create visualizations showing key insights.""",
    options=options
):
    if message.subtype == "subagent_start":
        print(f"Starting subagent: {message.agent_name}")
    elif message.subtype == "subagent_end":
        print(f"Completed subagent: {message.agent_name}")
```

### 3. Hooks System

**NEW FEATURE:** Inject logic at key execution points

```python
from claude_agent_sdk import query, ClaudeAgentOptions
from pathlib import Path

# Define hooks
async def validate_file_path(tool_name: str, tool_input: dict, context: dict):
    """Pre-execution hook to validate file paths."""
    if tool_name in ["Read", "Write", "Edit"]:
        file_path = Path(tool_input.get("file_path", ""))

        # Prevent access to sensitive directories
        forbidden_dirs = ["/etc", "/sys", "/root", "C:\\Windows"]
        if any(str(file_path).startswith(d) for d in forbidden_dirs):
            return {
                "allow": False,
                "message": f"Access to {file_path} is forbidden"
            }

    return {"allow": True}

async def log_tool_result(tool_name: str, tool_result: str, context: dict):
    """Post-execution hook to log results."""
    with open("tool_audit.log", "a") as f:
        f.write(f"{context['timestamp']}: {tool_name} -> {len(tool_result)} bytes\n")

# Use hooks
options = ClaudeAgentOptions(
    model="claude-sonnet-4-5",
    hooks={
        "pre_tool_execution": validate_file_path,
        "post_tool_execution": log_tool_result,
        "session_start": lambda ctx: print(f"Session {ctx['session_id']} started"),
        "session_end": lambda ctx: print(f"Session completed in {ctx['duration']}s")
    }
)

async for message in query(
    prompt="Read all configuration files and create a backup",
    options=options
):
    print(message)
```

### 4. Model Context Protocol (MCP) Enhancements

**ENHANCED:** Define custom Python functions as tools

```python
from claude_agent_sdk import tool, create_sdk_mcp_server, query

@tool(
    name="process_csv",
    description="Process CSV file and return statistics",
    parameters={
        "file_path": str,
        "operations": list[str]
    }
)
async def process_csv(args: dict) -> dict:
    """Custom CSV processing tool."""
    import pandas as pd

    df = pd.read_csv(args["file_path"])

    results = {
        "rows": len(df),
        "columns": list(df.columns),
        "statistics": {}
    }

    if "summary" in args["operations"]:
        results["statistics"]["summary"] = df.describe().to_dict()

    if "missing" in args["operations"]:
        results["statistics"]["missing"] = df.isnull().sum().to_dict()

    return {
        "content": [{
            "type": "text",
            "text": f"Processed {results['rows']} rows, {len(results['columns'])} columns"
        }]
    }

# Create MCP server with custom tool
csv_server = create_sdk_mcp_server(
    name="csv-tools",
    version="1.0.0",
    tools=[process_csv]
)

# Use in agent
options = ClaudeAgentOptions(
    mcp_servers={"csv": csv_server},
    allowed_tools=["mcp__csv__process_csv", "Read", "Write"]
)

async for message in query(
    prompt="Process the sales_data.csv file and create a summary report",
    options=options
):
    print(message)
```

### 5. General-Purpose Agent Development

**NEW CAPABILITY:** Beyond coding tasks

```python
from claude_agent_sdk import query, ClaudeAgentOptions

# Example 1: Web Research Agent
async def research_agent(topic: str):
    """Agent that researches a topic and creates a comprehensive report."""
    options = ClaudeAgentOptions(
        model="claude-sonnet-4-5",
        system_prompt="You are a research specialist with access to web search.",
        allowed_tools=["WebSearch", "WebFetch", "Write"]
    )

    async for message in query(
        prompt=f"""Research the topic: {topic}

        Tasks:
        1. Search for recent articles and papers
        2. Gather key findings and statistics
        3. Identify trends and patterns
        4. Create a comprehensive report in Markdown
        5. Save the report as research_report.md
        """,
        options=options
    ):
        print(message)

# Example 2: Data Processing Agent
async def data_processor(csv_file: str):
    """Agent that processes CSV data and generates visualizations."""
    options = ClaudeAgentOptions(
        model="claude-sonnet-4-5",
        system_prompt="You are a data analysis specialist.",
        allowed_tools=["Read", "Write", "Bash"]
    )

    async for message in query(
        prompt=f"""Process the CSV file: {csv_file}

        Tasks:
        1. Read and analyze the data
        2. Calculate statistics (mean, median, std dev)
        3. Identify outliers
        4. Generate a Python script to create visualizations
        5. Run the script to produce charts
        6. Create a summary report
        """,
        options=options
    ):
        print(message)

# Example 3: Document Automation Agent
async def document_automator(template: str, data: dict):
    """Agent that generates documents from templates."""
    options = ClaudeAgentOptions(
        model="claude-3-5-sonnet",
        allowed_tools=["Read", "Write"]
    )

    async for message in query(
        prompt=f"""Generate a document using this template: {template}

        Data:
        {data}

        Tasks:
        1. Read the template
        2. Fill in the data
        3. Format appropriately
        4. Save as output_document.pdf
        """,
        options=options
    ):
        print(message)
```

### 6. Enhanced Built-in Tools

**All Built-in Tools Available:**

```python
from claude_agent_sdk import query, ClaudeAgentOptions

options = ClaudeAgentOptions(
    model="claude-sonnet-4-5",
    allowed_tools=[
        "Read",        # Read file contents
        "Write",       # Create or overwrite files
        "Edit",        # Modify specific sections
        "Bash",        # Execute shell commands
        "Grep",        # Search file contents
        "Glob",        # File pattern matching
        "WebSearch",   # Search the internet
        "WebFetch"     # Fetch web content
    ]
)

async for message in query(
    prompt="""
    1. Search the web for Python best practices
    2. Create a summary document
    3. Find all Python files in the project
    4. Run the test suite
    """,
    options=options
):
    print(message)
```

---

## Code Migration Examples

### Example 1: Simple Agent

**Before (Claude Code SDK):**

```python
from claude_code import query
import asyncio

async def main():
    async for message in query(prompt="Explain async/await"):
        print(message)

asyncio.run(main())
```

**After (Claude Agent SDK):**

```python
from claude_agent_sdk import query, ClaudeAgentOptions
import asyncio

async def main():
    # Enhanced with 2025 features
    options = ClaudeAgentOptions(
        model="claude-sonnet-4-5",  # Latest model
        max_budget_usd=1.0           # Cost control
    )

    async for message in query(
        prompt="Explain async/await with modern Python 3.10+ examples",
        options=options
    ):
        print(message)

asyncio.run(main())
```

### Example 2: Custom Tool

**Before (Claude Code SDK):**

```python
from claude_code.tools import tool, create_mcp_server

@tool("my_tool", "Description")
async def my_tool(args):
    return {"result": "data"}

server = create_mcp_server(
    name="my-server",
    tools=[my_tool]
)
```

**After (Claude Agent SDK):**

```python
from claude_agent_sdk import tool, create_sdk_mcp_server

@tool(
    name="my_tool",
    description="Enhanced tool with Pydantic validation",
    parameters={
        "input_data": str,
        "options": dict
    }
)
async def my_tool(args: dict) -> dict:
    """Enhanced tool with better type safety."""
    return {
        "content": [{
            "type": "text",
            "text": f"Processed: {args['input_data']}"
        }]
    }

server = create_sdk_mcp_server(
    name="my-server",
    version="2.0.0",  # Required in new version
    tools=[my_tool]
)
```

### Example 3: Multi-Agent System

**Before (Claude Code SDK):**

```python
# Sequential execution only
from claude_code import query

async def pipeline():
    # Agent 1
    async for msg in query(prompt="Task 1"):
        result1 = msg

    # Agent 2 (uses result1)
    async for msg in query(prompt=f"Task 2: {result1}"):
        result2 = msg

    return result2
```

**After (Claude Agent SDK with Subagents):**

```python
from claude_agent_sdk import query, ClaudeAgentOptions

async def pipeline():
    options = ClaudeAgentOptions(
        model="claude-sonnet-4-5",
        enable_subagents=True,
        agents={
            "agent1": {
                "description": "First task specialist",
                "tools": ["Read", "WebSearch"]
            },
            "agent2": {
                "description": "Second task specialist",
                "tools": ["Write", "Bash"]
            }
        }
    )

    # Automatic subagent orchestration
    async for message in query(
        prompt="Complete Task 1, then use results for Task 2",
        options=options
    ):
        if message.subtype == "subagent_start":
            print(f"Starting: {message.agent_name}")
        elif message.subtype == "subagent_end":
            print(f"Completed: {message.agent_name}")
```

---

## Troubleshooting

### Issue 1: Import Errors

**Problem:**

```python
ModuleNotFoundError: No module named 'claude_code'
```

**Solution:**

```bash
# Uninstall old package
pip uninstall claude-code-sdk

# Install new package
pip install claude-agent-sdk

# Update all imports
# From: from claude_code import query
# To:   from claude_agent_sdk import query
```

### Issue 2: Python Version Compatibility

**Problem:**

```
SyntaxError: invalid syntax (pattern matching not supported)
```

**Solution:**

```bash
# Check Python version
python --version

# Upgrade if necessary
# Option 1: Use pyenv
pyenv install 3.11.7
pyenv global 3.11.7

# Option 2: Use system package manager
# Ubuntu/Debian:
sudo apt update
sudo apt install python3.11

# macOS:
brew install python@3.11
```

### Issue 3: Tool Function Signature Changed

**Problem:**

```python
TypeError: create_mcp_server() got an unexpected keyword argument 'version'
```

**Solution:**

```python
# OLD (won't work)
from claude_code.tools import create_mcp_server

server = create_mcp_server(name="my-tools", tools=[...])

# NEW (correct)
from claude_agent_sdk.tools import create_sdk_mcp_server

server = create_sdk_mcp_server(
    name="my-tools",
    version="1.0.0",  # Now required
    tools=[...]
)
```

### Issue 4: Model Name Not Recognized

**Problem:**

```
ValueError: Unknown model: claude-code-20241022
```

**Solution:**

```python
# Update model names
options = ClaudeAgentOptions(
    # OLD (may not work)
    # model="claude-code-20241022"

    # NEW (recommended)
    model="claude-sonnet-4-5"

    # Or use previous models
    # model="claude-3-5-sonnet-20241022"
)
```

### Issue 5: Hooks Not Working

**Problem:**

```
TypeError: hooks must be a dictionary of callables
```

**Solution:**

```python
# Ensure hooks are async functions
async def my_hook(tool_name: str, tool_input: dict, context: dict):
    return {"allow": True}

# Correct hook definition
options = ClaudeAgentOptions(
    hooks={
        "pre_tool_execution": my_hook  # async function, not my_hook()
    }
)
```

---

## Migration Checklist

### Pre-Migration

- [ ] Review current Claude Code SDK usage
- [ ] Identify all custom tools and MCP servers
- [ ] Document current agent configurations
- [ ] Check Python version (must be 3.10+)
- [ ] Review deprecated features being used

### Migration Steps

- [ ] Install Claude Agent SDK (`pip install claude-agent-sdk`)
- [ ] Run automated import migration script
- [ ] Update model names to Claude Sonnet 4.5
- [ ] Update `create_mcp_server` to `create_sdk_mcp_server`
- [ ] Add version numbers to MCP servers
- [ ] Test basic agent functionality
- [ ] Test custom tools and MCP integrations
- [ ] Implement hooks if needed
- [ ] Configure subagents for complex workflows
- [ ] Update error handling for new message types

### Post-Migration

- [ ] Run comprehensive test suite
- [ ] Verify all tools function correctly
- [ ] Monitor costs with new models
- [ ] Update documentation
- [ ] Train team on new features
- [ ] Remove Claude Code SDK dependencies
- [ ] Uninstall `claude-code-sdk` package

---

## Additional Resources

### Official Documentation

- **Claude Agent SDK Docs**: https://docs.anthropic.com/agent-sdk
- **API Reference**: https://docs.anthropic.com/api
- **Migration Support**: https://docs.anthropic.com/agent-sdk/migration

### Community Resources

- **GitHub Examples**: https://github.com/anthropics/claude-agent-sdk-examples
- **Discord Community**: https://discord.gg/anthropic
- **Stack Overflow**: Tag questions with `claude-agent-sdk`

### Getting Help

For migration assistance:
1. Check the official documentation
2. Search GitHub issues
3. Ask in the Discord community
4. Contact Anthropic support for enterprise customers

---

## Summary

**Key Takeaways:**

1. **Package Renamed**: `claude-code-sdk` → `claude-agent-sdk`
2. **Python 3.10+ Required**: Upgrade if on older versions
3. **New Model**: Claude Sonnet 4.5 recommended for best performance
4. **Enhanced Features**: Subagents, hooks, and expanded tool ecosystem
5. **General-Purpose**: Now supports non-coding tasks (research, data processing, automation)
6. **Breaking Changes**: MCP server creation function renamed

**Migration Effort:**

- **Simple Projects**: 1-2 hours (mostly find/replace)
- **Medium Projects**: 4-8 hours (custom tools, testing)
- **Complex Projects**: 1-2 days (subagents, hooks, full integration)

**Recommended Approach:**

1. Start with automated import updates
2. Update one agent at a time
3. Test thoroughly before production deployment
4. Leverage new features (subagents, hooks) for complex workflows

---

**Questions or issues?** Consult the comprehensive guide or reach out to the community.

**Ready to unlock the full potential of Claude Agent SDK?** Start migrating today!
