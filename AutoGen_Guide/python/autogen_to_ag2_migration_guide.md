# AutoGen to AG2 Migration Guide
## Complete Guide to Migrating from Microsoft AutoGen to AG2

---

## Table of Contents

1. [Introduction](#introduction)
2. [Understanding AG2](#understanding-ag2)
3. [Why Migrate?](#why-migrate)
4. [Compatibility and Backward Compatibility](#compatibility-and-backward-compatibility)
5. [Migration Steps](#migration-steps)
6. [Code Examples](#code-examples)
7. [New AG2 Features](#new-ag2-features)
8. [Troubleshooting](#troubleshooting)
9. [FAQ](#faq)
10. [Resources](#resources)

---

## Introduction

In November 2024, Microsoft AutoGen was rebranded as **AG2 (AutoGen 2.0)** and transitioned to open governance under the new **AG2AI** organization. This guide provides everything you need to successfully migrate from AutoGen to AG2.

### What You'll Learn

- What AG2 is and why it was created
- Step-by-step migration process
- How to leverage new AG2 features
- Common migration scenarios and solutions
- Best practices for AG2 development

### Who Should Migrate?

**Everyone!** However, migration is **100% optional** because:

- AG2 is **fully backward compatible** with AutoGen
- Your existing AutoGen code will continue to work
- You can migrate gradually at your own pace
- No urgent action required

That said, AG2 represents the future of the framework with:

- Open governance
- Active community development
- Latest features and enhancements
- Long-term sustainability

---

## Understanding AG2

### What is AG2?

**AG2** is the evolution of Microsoft AutoGen, positioned as the **"PyTorch-equivalent for Agentic AI"**. It's:

- **Open Source**: Apache 2.0 license (from v0.3+)
- **Community-Driven**: Open governance model under AG2AI
- **Backward Compatible**: Works with existing AutoGen code
- **Production Ready**: Battle-tested by enterprises worldwide

### Key Changes from AutoGen to AG2

| Aspect | AutoGen | AG2 |
|--------|---------|-----|
| **Organization** | Microsoft Research | AG2AI (community) |
| **Governance** | Microsoft-led | Open governance |
| **License** | MIT (pre-v0.3) | Apache 2.0 (v0.3+) |
| **Package Name** | `autogen-agentchat` | `ag2` |
| **Imports** | `import autogen` | `import autogen` (still works!) or `import ag2` |
| **API** | Stable API | Same stable API |
| **Features** | Core features | Core + 2025 enhancements |
| **Community** | Growing | Larger, more active |

### What Changed?

**Very little!** The core changes are:

1. **Organization**: Now AG2AI instead of Microsoft
2. **Package name**: `ag2` instead of `autogen-agentchat`
3. **Branding**: AG2 instead of AutoGen
4. **Governance**: Open, community-driven

### What Stayed the Same?

**Almost everything!**

- ✅ All agent classes (ConversableAgent, AssistantAgent, UserProxyAgent)
- ✅ All APIs and methods
- ✅ Configuration files (OAI_CONFIG_LIST.json)
- ✅ Group chats, sequential chats, nested chats
- ✅ Function registration and tool use
- ✅ Code execution
- ✅ AutoGen Studio

---

## Why Migrate?

### Benefits of Migrating to AG2

1. **Future-Proofing**
   - AG2 is the actively developed future of the framework
   - New features will be released for AG2
   - Long-term support and maintenance

2. **Community Ownership**
   - Open governance means community-driven development
   - Transparent decision-making
   - More contributors, faster innovation

3. **Latest Features**
   - Enhanced ConversableAgent with better context management
   - Improved telemetry and monitoring
   - Cost management features
   - AutoGen Studio 2025 updates

4. **Better Support**
   - Active, growing community
   - More resources and tutorials
   - Better documentation

5. **Enterprise Confidence**
   - Apache 2.0 license for commercial use
   - Open governance for stability
   - Industry backing

### When to Migrate?

**Recommended Timeline:**

- **Immediate**: New projects should start with AG2
- **Short-term (1-3 months)**: Migrate actively developed projects
- **Medium-term (3-6 months)**: Migrate stable production systems
- **Long-term**: Eventually migrate all projects

**No rush!** AutoGen will continue to work, but AG2 is the future.

---

## Compatibility and Backward Compatibility

### Full Backward Compatibility

AG2 is **100% backward compatible** with AutoGen:

✅ **All code works as-is**
✅ **No breaking changes**
✅ **Same APIs and interfaces**
✅ **Drop-in replacement**

### Compatibility Matrix

| Feature | AutoGen | AG2 | Migration Needed? |
|---------|---------|-----|-------------------|
| **Agent Classes** | ✅ | ✅ | ❌ No |
| **ConversableAgent** | ✅ | ✅ Enhanced | ❌ No |
| **AssistantAgent** | ✅ | ✅ | ❌ No |
| **UserProxyAgent** | ✅ | ✅ | ❌ No |
| **GroupChat** | ✅ | ✅ | ❌ No |
| **GroupChatManager** | ✅ | ✅ | ❌ No |
| **Function Registration** | ✅ | ✅ | ❌ No |
| **Code Execution** | ✅ | ✅ | ❌ No |
| **Config Files** | ✅ | ✅ | ❌ No |
| **LLM Providers** | ✅ | ✅ | ❌ No |
| **AutoGen Studio** | ✅ | ✅ Enhanced | ❌ No |
| **Context Management** | Basic | Enhanced | ⚠️ Optional upgrade |
| **Telemetry** | ❌ | ✅ New | ⚠️ New feature |
| **Cost Management** | ❌ | ✅ New | ⚠️ New feature |

### Python Version Support

Both AutoGen and AG2 support:

- **Python 3.10** ✅
- **Python 3.11** ✅
- **Python 3.12** ✅
- **Python 3.13** ✅ (AG2 adds this)

---

## Migration Steps

### Step 1: Install AG2

#### Option A: Fresh Installation

```bash
# Install AG2
pip install ag2

# Or install AG2 with specific extras
pip install 'ag2[openai]'
pip install 'ag2[anthropic]'
pip install 'ag2[all]'
```

#### Option B: Replace AutoGen

```bash
# Uninstall AutoGen
pip uninstall autogen-agentchat

# Install AG2
pip install ag2
```

#### Option C: Virtual Environment (Recommended)

```bash
# Create new virtual environment
python -m venv ag2_env

# Activate it
source ag2_env/bin/activate  # macOS/Linux
# ag2_env\Scripts\activate  # Windows

# Install AG2
pip install ag2
```

### Step 2: Update Imports (Optional)

You have **three options** for imports:

#### Option 1: No Change (Recommended)

```python
# Keep existing imports - works perfectly with AG2!
import autogen

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

agent = autogen.ConversableAgent(
    name="my_agent",
    llm_config=llm_config
)
```

**Advantage**: No code changes needed. Easy gradual migration.

#### Option 2: Alias AG2 as AutoGen

```python
# Import AG2 with AutoGen alias
import ag2 as autogen

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

agent = autogen.ConversableAgent(
    name="my_agent",
    llm_config=llm_config
)
```

**Advantage**: Makes AG2 usage explicit while maintaining code compatibility.

#### Option 3: Full AG2 Imports

```python
# Use AG2 imports directly
from ag2 import ConversableAgent, AssistantAgent, UserProxyAgent

config_list = ConversableAgent.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

agent = ConversableAgent(
    name="my_agent",
    llm_config=llm_config
)
```

**Advantage**: Fully committed to AG2. Clearest intention.

### Step 3: Test Your Code

Run your existing test suite:

```bash
# Run tests
pytest tests/

# Or your test command
python -m unittest discover tests/
```

**Expected result**: All tests should pass without changes.

### Step 4: Update Configuration (Optional)

Your existing configuration files work as-is, but you can optionally rename them:

#### Before (AutoGen):

```
OAI_CONFIG_LIST.json
```

#### After (AG2 - optional):

```
AG2_CONFIG_LIST.json
```

**Configuration file format stays the same:**

```json
[
    {
        "model": "gpt-4",
        "api_key": "sk-..."
    }
]
```

### Step 5: Leverage New AG2 Features (Optional)

Now you can optionally use AG2 2025 enhancements:

```python
import autogen

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")

# NEW: Enhanced context management
llm_config = {
    "config_list": config_list,
    "context_truncation_strategy": "smart",  # AG2 feature
    "context_window_size": 128000  # AG2 feature
}

agent = autogen.ConversableAgent(
    name="enhanced_agent",
    llm_config=llm_config
)

# NEW: Cost tracking
cost_stats = agent.get_cost_stats()
print(f"Cost: ${cost_stats['total_cost']:.4f}")
```

### Step 6: Update Dependencies

Update your `requirements.txt` or `pyproject.toml`:

#### requirements.txt

```txt
# Before
autogen-agentchat>=0.10.0

# After
ag2>=0.10.0
```

#### pyproject.toml

```toml
# Before
[project]
dependencies = [
    "autogen-agentchat>=0.10.0",
]

# After
[project]
dependencies = [
    "ag2>=0.10.0",
]
```

### Step 7: Update Documentation

Update your project documentation to reference AG2:

- Update README.md
- Update installation instructions
- Update code examples
- Update links to documentation

---

## Code Examples

### Example 1: Simple Agent Migration

**AutoGen Code (Before):**

```python
import autogen

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    code_execution_config={"work_dir": "coding"},
    human_input_mode="NEVER"
)

user_proxy.initiate_chat(
    assistant,
    message="Calculate fibonacci(10)"
)
```

**AG2 Code (After - No Changes Needed!):**

```python
# Same code works perfectly!
import autogen

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    code_execution_config={"work_dir": "coding"},
    human_input_mode="NEVER"
)

user_proxy.initiate_chat(
    assistant,
    message="Calculate fibonacci(10)"
)
```

**AG2 Code (After - With Enhancements):**

```python
import autogen

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")

# NEW: Add AG2 enhancements
llm_config = {
    "config_list": config_list,
    "context_truncation_strategy": "smart",  # AG2 feature
}

assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    code_execution_config={"work_dir": "coding"},
    human_input_mode="NEVER"
)

user_proxy.initiate_chat(
    assistant,
    message="Calculate fibonacci(10)"
)

# NEW: Get cost statistics
cost_stats = assistant.get_cost_stats()
print(f"Total cost: ${cost_stats['total_cost']:.4f}")
```

### Example 2: Group Chat Migration

**AutoGen Code (Before):**

```python
import autogen

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

# Create agents
coder = autogen.AssistantAgent(
    name="coder",
    llm_config=llm_config
)

reviewer = autogen.AssistantAgent(
    name="reviewer",
    llm_config=llm_config
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    code_execution_config={"work_dir": "coding"},
    human_input_mode="NEVER"
)

# Create group chat
groupchat = autogen.GroupChat(
    agents=[coder, reviewer, user_proxy],
    messages=[],
    max_round=10
)

manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config
)

user_proxy.initiate_chat(
    manager,
    message="Create a Python REST API"
)
```

**AG2 Code (After - Identical!):**

```python
# Exact same code works with AG2
import autogen

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

coder = autogen.AssistantAgent(
    name="coder",
    llm_config=llm_config
)

reviewer = autogen.AssistantAgent(
    name="reviewer",
    llm_config=llm_config
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    code_execution_config={"work_dir": "coding"},
    human_input_mode="NEVER"
)

groupchat = autogen.GroupChat(
    agents=[coder, reviewer, user_proxy],
    messages=[],
    max_round=10
)

manager = autogen.GroupChatManager(
    groupchat=groupchat,
    llm_config=llm_config
)

user_proxy.initiate_chat(
    manager,
    message="Create a Python REST API"
)
```

### Example 3: Function Calling Migration

**AutoGen Code (Before):**

```python
import autogen
from typing import Annotated

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

def get_weather(city: Annotated[str, "City name"]) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny, 72°F"

assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    code_execution_config=False,
    human_input_mode="NEVER"
)

# Register function
assistant.register_for_llm(name="get_weather", description="Get weather")(get_weather)
user_proxy.register_for_execution(name="get_weather")(get_weather)

user_proxy.initiate_chat(
    assistant,
    message="What's the weather in London?"
)
```

**AG2 Code (After - Same!):**

```python
# Identical code works with AG2
import autogen
from typing import Annotated

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")
llm_config = {"config_list": config_list}

def get_weather(city: Annotated[str, "City name"]) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny, 72°F"

assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config=llm_config
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    code_execution_config=False,
    human_input_mode="NEVER"
)

assistant.register_for_llm(name="get_weather", description="Get weather")(get_weather)
user_proxy.register_for_execution(name="get_weather")(get_weather)

user_proxy.initiate_chat(
    assistant,
    message="What's the weather in London?"
)
```

---

## New AG2 Features

### 1. Enhanced Context Management

```python
import autogen

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")

# Smart context truncation
llm_config = {
    "config_list": config_list,
    "context_truncation_strategy": "smart",  # NEW
    "context_window_size": 128000  # NEW
}

agent = autogen.ConversableAgent(
    name="smart_agent",
    llm_config=llm_config
)
```

**Truncation Strategies:**

- `"recent"`: Keep most recent messages
- `"semantic"`: Preserve semantically important messages
- `"summarize"`: Summarize old messages

### 2. Cost Management

```python
import autogen

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")

# Configure cost limits
llm_config = {
    "config_list": config_list,
    "cost_management": {
        "max_cost_per_conversation": 1.00,
        "max_cost_per_day": 10.00,
        "track_token_usage": True
    }
}

agent = autogen.ConversableAgent(
    name="cost_managed_agent",
    llm_config=llm_config
)

# Get cost statistics
cost_stats = agent.get_cost_stats()
print(f"Cost: ${cost_stats['total_cost']:.4f}")
print(f"Tokens: {cost_stats['total_tokens']}")
```

### 3. Telemetry and Monitoring

```python
import autogen
from autogen.telemetry import TelemetryConfig

config_list = autogen.config_list_from_json("OAI_CONFIG_LIST.json")

# Enable telemetry
telemetry = TelemetryConfig(
    enable_metrics=True,
    enable_tracing=True,
    metrics_endpoint="http://localhost:9090/metrics"
)

llm_config = {
    "config_list": config_list,
    "telemetry": telemetry
}

agent = autogen.ConversableAgent(
    name="monitored_agent",
    llm_config=llm_config
)

# Metrics tracked:
# - Response times
# - Token usage
# - Error rates
# - Success rates
```

### 4. AutoGen Studio 2025

```bash
# Install AutoGen Studio
pip install autogenstudio

# Launch Studio
autogenstudio ui --port 8081

# Export workflow as code
autogenstudio export --workflow my_workflow --output workflow.py

# Deploy to cloud
autogenstudio deploy --target azure --workflow my_workflow
```

**New Studio Features:**

- Visual flow builder
- Real-time collaboration
- Step-through debugging
- Template library
- Cloud deployment
- Analytics dashboard

---

## Troubleshooting

### Issue 1: Import Errors

**Problem:**
```python
ImportError: No module named 'autogen'
```

**Solution:**
```bash
# Ensure AG2 is installed
pip install ag2

# Or reinstall
pip uninstall ag2
pip install ag2
```

### Issue 2: Configuration File Not Found

**Problem:**
```
FileNotFoundError: OAI_CONFIG_LIST.json not found
```

**Solution:**
```python
# Use absolute path
import os

config_path = os.path.join(os.getcwd(), "OAI_CONFIG_LIST.json")
config_list = autogen.config_list_from_json(config_path)

# Or specify full path
config_list = autogen.config_list_from_json("/path/to/OAI_CONFIG_LIST.json")
```

### Issue 3: Version Conflicts

**Problem:**
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed.
```

**Solution:**
```bash
# Create fresh virtual environment
python -m venv ag2_env
source ag2_env/bin/activate
pip install ag2

# Or use pip with --force-reinstall
pip install --force-reinstall ag2
```

### Issue 4: New Features Not Working

**Problem:**
```python
AttributeError: 'ConversableAgent' object has no attribute 'get_cost_stats'
```

**Solution:**
```bash
# Ensure you have latest AG2 version
pip install --upgrade ag2

# Check version
pip show ag2
```

### Issue 5: AutoGen Studio Not Starting

**Problem:**
```
Command 'autogenstudio' not found
```

**Solution:**
```bash
# Install AutoGen Studio separately
pip install autogenstudio

# Ensure it's in PATH
python -m autogenstudio ui --port 8081
```

---

## FAQ

### Q1: Do I have to migrate from AutoGen to AG2?

**A:** No, migration is completely optional. Your AutoGen code will continue to work indefinitely because AG2 is fully backward compatible.

### Q2: Will my existing AutoGen code break with AG2?

**A:** No, AG2 is 100% backward compatible. All existing code works without changes.

### Q3: Can I use both AutoGen and AG2 in the same project?

**A:** No, you should use one or the other. Since AG2 is backward compatible, just use AG2.

### Q4: What's the difference between `autogen` and `ag2` imports?

**A:** They're the same! AG2 supports both for backward compatibility. Use whichever you prefer.

### Q5: Do I need to change my configuration files?

**A:** No, all existing configuration files (OAI_CONFIG_LIST.json) work as-is.

### Q6: Will AutoGen continue to be maintained?

**A:** The AutoGen codebase has evolved into AG2. Future development happens under AG2.

### Q7: Is AG2 production-ready?

**A:** Yes! AG2 is the same battle-tested code as AutoGen, with additional enhancements.

### Q8: How do I get support for AG2?

**A:** Join the AG2 community on Discord, GitHub, or visit the documentation at https://ag2.ai/docs

### Q9: What Python versions does AG2 support?

**A:** Python 3.10, 3.11, 3.12, and 3.13 (newly added in 2025).

### Q10: Can I contribute to AG2?

**A:** Absolutely! AG2 is open source with open governance. Visit https://github.com/ag2ai/ag2

---

## Resources

### Official Documentation

- **AG2 Homepage**: https://ag2.ai
- **Documentation**: https://ag2.ai/docs
- **API Reference**: https://ag2.ai/docs/api
- **Migration Guide**: https://ag2.ai/docs/migration

### Community

- **GitHub**: https://github.com/ag2ai/ag2
- **Discord**: https://discord.gg/ag2ai
- **Twitter/X**: https://twitter.com/ag2ai
- **Blog**: https://ag2.ai/blog

### Learning Resources

- **Tutorials**: https://ag2.ai/docs/tutorials
- **Examples**: https://github.com/ag2ai/ag2/tree/main/examples
- **Videos**: https://youtube.com/@ag2ai
- **Community Guides**: https://ag2.ai/community

### Support

- **GitHub Issues**: https://github.com/ag2ai/ag2/issues
- **Discord Support**: https://discord.gg/ag2ai
- **Stack Overflow**: Tag `ag2` or `autogen`

---

## Migration Checklist

Use this checklist to track your migration:

### Pre-Migration

- [ ] Review this migration guide
- [ ] Understand AG2 benefits
- [ ] Check Python version compatibility (3.10-3.13)
- [ ] Backup existing project
- [ ] Review current AutoGen version

### Migration Process

- [ ] Install AG2: `pip install ag2`
- [ ] Test existing code (no changes)
- [ ] Review test results
- [ ] Update imports (optional)
- [ ] Update configuration files (optional)
- [ ] Explore new AG2 features
- [ ] Update dependencies (requirements.txt)
- [ ] Update documentation

### Post-Migration

- [ ] Run full test suite
- [ ] Test in staging environment
- [ ] Update CI/CD pipelines
- [ ] Train team on AG2
- [ ] Update project documentation
- [ ] Join AG2 community
- [ ] Provide feedback to AG2 team

### Optional Enhancements

- [ ] Enable cost management
- [ ] Configure telemetry
- [ ] Try AutoGen Studio 2025
- [ ] Implement smart context truncation
- [ ] Explore new agent patterns

---

## Conclusion

Migrating from AutoGen to AG2 is straightforward because:

✅ **100% backward compatible** - your code works as-is
✅ **No breaking changes** - zero risk migration
✅ **Optional upgrade** - migrate at your own pace
✅ **New features** - leverage 2025 enhancements when ready

**Key Takeaway**: AG2 is the future of agentic AI development. Whether you migrate today or tomorrow, you're positioned for success with the industry-standard framework for multi-agent AI systems.

**Welcome to the AG2 community!**

---

## Additional Examples

### Complete Project Migration

**Project Structure Before:**

```
my_autogen_project/
├── requirements.txt          # autogen-agentchat>=0.10.0
├── OAI_CONFIG_LIST.json
├── main.py                   # import autogen
├── agents/
│   ├── __init__.py
│   └── custom_agents.py      # import autogen
└── tests/
    └── test_agents.py        # import autogen
```

**Project Structure After (No Changes!):**

```
my_ag2_project/
├── requirements.txt          # ag2>=0.10.0  (ONLY CHANGE)
├── OAI_CONFIG_LIST.json      # Same file
├── main.py                   # import autogen (works!)
├── agents/
│   ├── __init__.py
│   └── custom_agents.py      # import autogen (works!)
└── tests/
    └── test_agents.py        # import autogen (works!)
```

**Migration Command:**

```bash
# Just update the package
pip uninstall autogen-agentchat
pip install ag2

# Everything else stays the same!
```

---

**Happy migrating! 🚀**
