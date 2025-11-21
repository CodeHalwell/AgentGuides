Latest: 0.1.6
# Anthropic Claude Agent SDK - Comprehensive Technical Guide

> **Exhaustive Reference for Building, Deploying, and Scaling Production AI Agents with Claude**

---

## Table of Contents

1. [Introduction](#introduction)
2. [Installation & Authentication](#installation--authentication)
3. [Core Architecture & Concepts](#core-architecture--concepts)
4. [Simple Agents](#simple-agents)
5. [Tools & Integration](#tools--integration)
6. [Multi-Agent Systems](#multi-agent-systems)
7. [Computer Use API](#computer-use-api)
8. [Model Context Protocol (MCP)](#model-context-protocol-mcp)
9. [Structured Output](#structured-output)
10. [Session Management](#session-management)
11. [Context Engineering](#context-engineering)
12. [Advanced Permissions](#advanced-permissions)
13. [Agentic Patterns](#agentic-patterns)
14. [API Reference](#api-reference)

---

## Introduction

### What is the Claude Agent SDK?

The **Claude Agent SDK** is the evolution of the Claude Code SDK, renamed to reflect its expanded capabilities beyond coding tasks. It provides a comprehensive framework for building production-ready AI agents that can:

- Execute complex, multi-step tasks autonomously
- Control computers through mouse, keyboard, and screen interactions ("giving agents a computer")
- Integrate custom tools and services via the Model Context Protocol (MCP)
- Maintain context and state across long-running sessions
- Operate with fine-grained permission controls and safety guarantees
- Scale from simple single-agent applications to complex multi-agent orchestration systems

### Key Differentiators

| Feature | Claude Agent SDK | Traditional LLM Libraries |
|---------|------------------|--------------------------|
| Computer Control | ✅ Native support | ❌ Not built-in |
| Tool Integration | ✅ MCP-based standardisation | ⚠️ Ad-hoc implementations |
| Context Management | ✅ Automatic compaction | ❌ Manual management |
| Permissions System | ✅ Fine-grained control | ❌ Limited |
| Session Persistence | ✅ Built-in | ⚠️ Requires external storage |
| Multi-Agent Coordination | ✅ Native patterns | ⚠️ Custom implementation |

### Who Should Use This Guide

- **Backend engineers** building agent-based services
- **DevOps specialists** deploying agents to production
- **AI practitioners** implementing complex agentic workflows
- **Researchers** exploring autonomous agent capabilities
- **Full-stack developers** adding AI automation to applications

---

## Installation & Authentication

### TypeScript/JavaScript Installation

#### Prerequisites
- Node.js 16.0.0 or later (18+ recommended for best performance)
- npm 7+ or yarn 1.22+
- TypeScript 4.5+ (for type safety)

#### Installation Steps

```bash
# Create a new project
mkdir my-agent
cd my-agent

# Initialise Node project
npm init -y

# Install Claude Agent SDK
npm install @anthropic-ai/claude-agent-sdk

# Install TypeScript and types (optional but recommended)
npm install --save-dev typescript @types/node ts-node

# Install recommended dependencies
npm install dotenv  # For environment variable management
npm install zod     # For schema validation
```

#### Project Structure

```
my-agent/
├── src/
│   ├── agents/
│   │   ├── basic-agent.ts
│   │   └── advanced-agent.ts
│   ├── tools/
│   │   ├── custom-tools.ts
│   │   └── mcp-servers.ts
│   ├── index.ts
│   └── types.ts
├── .env
├── .env.example
├── package.json
├── tsconfig.json
└── README.md
```

### Python Installation

#### Prerequisites
- Python 3.10 or later (3.11+ recommended)
- pip package manager
- Virtual environment manager (venv or poetry recommended)

#### Installation Steps

```bash
# Create project directory
mkdir my-agent
cd my-agent

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install Claude Agent SDK
pip install claude-agent-sdk

# Install recommended dependencies
pip install python-dotenv pydantic aiohttp
```

#### Project Structure

```
my-agent/
├── src/
│   ├── agents/
│   │   ├── basic_agent.py
│   │   └── advanced_agent.py
│   ├── tools/
│   │   ├── custom_tools.py
│   │   └── mcp_servers.py
│   ├── main.py
│   └── types.py
├── .env
├── .env.example
├── requirements.txt
├── pyproject.toml
└── README.md
```

### Authentication & Configuration

#### Setting Up API Keys

**Step 1: Get Your API Key**
1. Go to https://console.anthropic.com
2. Sign up or log in
3. Navigate to "API Keys"
4. Click "Generate API Key"
5. Copy the generated key

**Step 2: Configure Environment**

**TypeScript:**
```typescript
// .env
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
```

**Python:**
```python
# .env
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022
```

#### Loading Configuration

**TypeScript:**
```typescript
import dotenv from 'dotenv';
import { query } from '@anthropic-ai/claude-agent-sdk';

dotenv.config();

const apiKey = process.env.ANTHROPIC_API_KEY;
const model = process.env.ANTHROPIC_MODEL || 'claude-3-5-sonnet-20241022';

if (!apiKey) {
  throw new Error('ANTHROPIC_API_KEY environment variable is required');
}
```

**Python:**
```python
import os
from dotenv import load_dotenv
from claude_agent_sdk import query

load_dotenv()

api_key = os.getenv('ANTHROPIC_API_KEY')
model = os.getenv('ANTHROPIC_MODEL', 'claude-3-5-sonnet-20241022')

if not api_key:
    raise ValueError('ANTHROPIC_API_KEY environment variable is required')
```

### Alternative Authentication Methods

#### AWS Bedrock

```typescript
// TypeScript
import { ClaudeSDKClient, ClaudeAgentOptions } from '@anthropic-ai/claude-agent-sdk';

const options = ClaudeAgentOptions({
  apiProvider: 'bedrock',
  bedrockRegion: 'us-east-1',
  // AWS credentials will be loaded from environment or IAM role
});
```

```python
# Python
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

options = ClaudeAgentOptions(
    api_provider='bedrock',
    bedrock_region='us-east-1',
    # AWS credentials will be loaded from environment or IAM role
)
```

#### Google Vertex AI

```typescript
// TypeScript
import { ClaudeSDKClient, ClaudeAgentOptions } from '@anthropic-ai/claude-agent-sdk';

const options = ClaudeAgentOptions({
  apiProvider: 'vertexai',
  vertexProjectId: 'your-project-id',
  vertexRegion: 'us-central1',
});
```

```python
# Python
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions

options = ClaudeAgentOptions(
    api_provider='vertexai',
    vertex_project_id='your-project-id',
    vertex_region='us-central1',
)
```

---

## Core Architecture & Concepts

### Architecture Overview

The Claude Agent SDK is built on the same agent harness that powers Claude Code, providing:

```
┌─────────────────────────────────────────────────────────────┐
│                    Claude Agent SDK                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         User Application Layer                       │  │
│  │  (Your agents, business logic, orchestration)        │  │
│  └────────────────────┬─────────────────────────────────┘  │
│                       │                                     │
│  ┌────────────────────▼─────────────────────────────────┐  │
│  │         Agent Execution Engine                       │  │
│  │  (Session management, context compaction,            │  │
│  │   tool execution, streaming)                         │  │
│  └────────────────────┬─────────────────────────────────┘  │
│                       │                                     │
│  ┌────────────────────▼──────┬──────────┬────────────────┐ │
│  │   Tool Ecosystem          │   MCP    │   Claude API  │ │
│  │  (File ops, Bash,         │ Server   │   Integration │ │
│  │   Web search)             │ support  │   & Caching   │ │
│  └───────────────────────────┴──────────┴────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Core Concepts Explained

#### 1. Agents
An **agent** is an autonomous entity that:
- Accepts a task or prompt as input
- Plans and reasons about how to accomplish the task
- Executes tools and commands to achieve goals
- Iteratively refines its approach based on feedback
- Returns results to the user

```typescript
// TypeScript: A basic agent
const agent = query({
  prompt: "Analyze this codebase and suggest improvements",
  options: {
    model: "claude-3-5-sonnet-20241022"
  }
});
```

#### 2. Tools
**Tools** are capabilities that agents can invoke to accomplish tasks:

- **File Operations**: Read, write, edit files
- **Command Execution**: Run bash/shell commands
- **Web Search**: Query the internet for information
- **Custom Tools**: Domain-specific functions via MCP
- **Computer Use**: Mouse, keyboard, screen control

#### 3. Sessions
A **session** maintains state across multiple interactions:

```typescript
// TypeScript: Create a session
let sessionId: string | undefined;

const response = query({
  prompt: "First, analyse the data",
  options: { model: "claude-3-5-sonnet-20241022" }
});

for await (const message of response) {
  if (message.type === 'system' && message.subtype === 'init') {
    sessionId = message.session_id;
  }
}

// Resume the same session
const continued = query({
  prompt: "Now generate a report based on that analysis",
  options: {
    resume: sessionId,
    model: "claude-3-5-sonnet-20241022"
  }
});
```

#### 4. Context
**Context** is the information available to the agent, including:
- System prompts and instructions
- Conversation history
- Tool outputs and results
- Environment variables and file system state
- User-provided background information

#### 5. Permissions
**Permissions** control what agents can and cannot do:
- Tool access restrictions
- File system boundaries
- Command execution policies
- User approval workflows
- Cost and usage limits

### Supported Claude Models

#### Claude 3.5 Sonnet (Recommended for Agents)
- **Best For**: Most agentic tasks, complex reasoning, balanced cost/performance
- **Context Window**: 200K tokens
- **Strengths**: Excellent at tool use, reasoning, code understanding
- **Cost**: Mid-range (balance of capability and cost)

```typescript
// TypeScript
const response = query({
  prompt: "Your task here",
  options: { model: "claude-3-5-sonnet-20241022" }
});
```

#### Claude 3.5 Opus (Complex Reasoning)
- **Best For**: Complex multi-step reasoning, large context windows
- **Context Window**: 200K tokens
- **Strengths**: Strongest reasoning, best at complex coordination
- **Cost**: Premium (highest capability tier)

```typescript
const response = query({
  prompt: "Complex orchestration task",
  options: { model: "claude-3-5-opus-20241022" }
});
```

#### Claude 3.5 Haiku (Cost-Efficient)
- **Best For**: Simple tasks, lightweight operations, cost-sensitive applications
- **Context Window**: 200K tokens
- **Strengths**: Fast, low-cost, good for simple logic
- **Cost**: Economy (lowest cost tier)

```typescript
const response = query({
  prompt: "Simple task",
  options: { model: "claude-3-5-haiku-20241022" }
});
```

---

## Simple Agents

### Creating Your First Agent

#### TypeScript Example

```typescript
import { query } from '@anthropic-ai/claude-agent-sdk';
import dotenv from 'dotenv';

dotenv.config();

async function main() {
  const response = query({
    prompt: "What is the capital of France?",
    options: {
      model: "claude-3-5-sonnet-20241022"
    }
  });

  for await (const message of response) {
    if (message.type === 'assistant') {
      console.log('Claude:', message.content);
    } else if (message.type === 'system' && message.subtype === 'init') {
      console.log(`Session ID: ${message.session_id}`);
    }
  }
}

main().catch(console.error);
```

#### Python Example

```python
import asyncio
from claude_agent_sdk import query
from dotenv import load_dotenv

load_dotenv()

async def main():
    async for message in query(prompt="What is the capital of France?"):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(f"Claude: {block.text}")
        elif message.type == 'system' and message.subtype == 'init':
            print(f"Session ID: {message.session_id}")

asyncio.run(main())
```

### System Prompts & Instructions

System prompts define an agent's behaviour, role, and constraints:

#### TypeScript

```typescript
import { query, ClaudeAgentOptions } from '@anthropic-ai/claude-agent-sdk';

const response = query({
  prompt: "Help me optimise this function for performance",
  options: {
    systemPrompt: `You are an expert performance engineer.
Your task is to:
1. Identify performance bottlenecks
2. Suggest optimisations using best practices
3. Provide concrete code examples
4. Estimate performance improvements

Always prioritise readability alongside performance improvements.
Avoid premature optimisation for simple cases.`,
    model: "claude-3-5-sonnet-20241022"
  }
});
```

#### Python

```python
from claude_agent_sdk import query, ClaudeAgentOptions

options = ClaudeAgentOptions(
    system_prompt="""You are an expert performance engineer.
Your task is to:
1. Identify performance bottlenecks
2. Suggest optimisations using best practices
3. Provide concrete code examples
4. Estimate performance improvements

Always prioritise readability alongside performance improvements.
Avoid premature optimisation for simple cases.""",
    model="claude-3-5-sonnet-20241022"
)

async for message in query(
    prompt="Help me optimise this function for performance",
    options=options
):
    print(message)
```

### Streaming Responses

Streaming allows real-time feedback as the agent processes tasks:

#### TypeScript Streaming

```typescript
import { query } from '@anthropic-ai/claude-agent-sdk';

async function streamingExample() {
  const response = query({
    prompt: "Write a detailed essay on climate change",
    options: {
      model: "claude-3-5-sonnet-20241022"
    }
  });

  let totalTokens = 0;
  let assistantMessages = 0;

  for await (const message of response) {
    switch (message.type) {
      case 'assistant':
        assistantMessages++;
        // Display content as it arrives
        if (typeof message.content === 'string') {
          process.stdout.write(message.content);
        } else {
          message.content.forEach(block => {
            if (block.type === 'text') {
              process.stdout.write(block.text);
            }
          });
        }
        break;

      case 'system':
        if (message.subtype === 'completion') {
          console.log('\n\n[Task Completed]');
          if (message.usage) {
            totalTokens = message.usage.output_tokens + message.usage.input_tokens;
            console.log(`Total tokens used: ${totalTokens}`);
          }
        }
        break;
    }
  }
}

streamingExample().catch(console.error);
```

#### Python Streaming

```python
import asyncio
from claude_agent_sdk import query, AssistantMessage, TextBlock

async def streaming_example():
    total_tokens = 0
    assistant_messages = 0

    async for message in query(
        prompt="Write a detailed essay on climate change"
    ):
        if isinstance(message, AssistantMessage):
            assistant_messages += 1
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text, end='', flush=True)

        elif message.type == 'system':
            if message.subtype == 'completion':
                print('\n\n[Task Completed]')
                if hasattr(message, 'usage'):
                    total_tokens = (
                        message.usage.output_tokens +
                        message.usage.input_tokens
                    )
                    print(f'Total tokens used: {total_tokens}')

asyncio.run(streaming_example())
```

### Single-Task Agent Execution

For simple, single-purpose tasks:

#### TypeScript

```typescript
import { query } from '@anthropic-ai/claude-agent-sdk';

async function analyseJSON(jsonString: string) {
  const response = query({
    prompt: `Analyse this JSON and explain its structure:

\`\`\`json
${jsonString}
\`\`\``,
    options: {
      model: "claude-3-5-sonnet-20241022",
      systemPrompt: "You are a JSON analyst. Explain structure clearly."
    }
  });

  let result = '';
  for await (const message of response) {
    if (message.type === 'assistant') {
      if (typeof message.content === 'string') {
        result += message.content;
      }
    }
  }
  return result;
}

// Usage
const jsonData = JSON.stringify({
  users: [
    { id: 1, name: "Alice", role: "admin" },
    { id: 2, name: "Bob", role: "user" }
  ]
}, null, 2);

analyseJSON(jsonData).then(analysis => {
  console.log("Analysis:", analysis);
});
```

#### Python

```python
import asyncio
from claude_agent_sdk import query, AssistantMessage, TextBlock

async def analyse_json(json_string: str) -> str:
    result = ''
    
    async for message in query(
        prompt=f"""Analyse this JSON and explain its structure:

```json
{json_string}
```""",
        options=ClaudeAgentOptions(
            model="claude-3-5-sonnet-20241022",
            system_prompt="You are a JSON analyst. Explain structure clearly."
        )
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    result += block.text
    
    return result

# Usage
import json
json_data = json.dumps({
    "users": [
        {"id": 1, "name": "Alice", "role": "admin"},
        {"id": 2, "name": "Bob", "role": "user"}
    ]
}, indent=2)

analysis = asyncio.run(analyse_json(json_data))
print("Analysis:", analysis)
```

### Error Handling

Proper error handling ensures robustness:

#### TypeScript Error Handling

```typescript
import { query } from '@anthropic-ai/claude-agent-sdk';

async function robustQuery(prompt: string, maxRetries = 3) {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const response = query({
        prompt,
        options: {
          model: "claude-3-5-sonnet-20241022"
        }
      });

      let result = '';
      for await (const message of response) {
        if (message.type === 'assistant') {
          if (typeof message.content === 'string') {
            result += message.content;
          }
        } else if (message.type === 'error') {
          throw new Error(`Agent error: ${message.error?.message}`);
        }
      }

      return result;

    } catch (error) {
      const err = error as any;
      console.error(`Attempt ${attempt} failed:`, err.message);

      // Handle specific error types
      if (err.code === 'RATE_LIMIT_EXCEEDED') {
        const waitTime = Math.min(1000 * Math.pow(2, attempt - 1), 30000);
        console.log(`Rate limited. Waiting ${waitTime}ms...`);
        await new Promise(resolve => setTimeout(resolve, waitTime));
        continue;
      }

      if (err.code === 'CONTEXT_LENGTH_EXCEEDED') {
        console.error('Context too large. Consider breaking into smaller tasks.');
        throw error;
      }

      if (attempt === maxRetries) {
        throw error;
      }

      // Exponential backoff
      const delay = Math.min(1000 * Math.pow(2, attempt - 1), 10000);
      console.log(`Retrying in ${delay}ms...`);
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }

  throw new Error('Failed after max retries');
}

// Usage
robustQuery("Perform a complex analysis").then(result => {
  console.log(result);
}).catch(error => {
  console.error('Final error:', error);
});
```

#### Python Error Handling

```python
import asyncio
from claude_agent_sdk import (
    query,
    ClaudeSDKError,
    CLINotFoundError,
    ProcessError,
    CLIJSONDecodeError
)

async def robust_query(prompt: str, max_retries: int = 3) -> str:
    for attempt in range(1, max_retries + 1):
        try:
            result = ''
            async for message in query(prompt):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            result += block.text
                elif message.type == 'error':
                    raise Exception(f"Agent error: {message.error}")
            
            return result

        except ProcessError as e:
            print(f"Attempt {attempt} failed: Process error {e.exit_code}")
            
            if "rate limit" in str(e).lower():
                wait_time = min(1000 * (2 ** (attempt - 1)), 30000)
                print(f"Rate limited. Waiting {wait_time}ms...")
                await asyncio.sleep(wait_time / 1000)
                continue
            
            if attempt == max_retries:
                raise
            
            delay = min(1000 * (2 ** (attempt - 1)), 10000)
            print(f"Retrying in {delay}ms...")
            await asyncio.sleep(delay / 1000)

        except CLINotFoundError:
            print("Error: Claude Code not installed")
            raise

        except ClaudeSDKError as e:
            print(f"SDK error: {e}")
            if attempt == max_retries:
                raise

    raise Exception("Failed after max retries")

# Usage
try:
    result = asyncio.run(robust_query("Perform a complex analysis"))
    print(result)
except Exception as e:
    print(f"Final error: {e}")
```

### Token Usage Tracking

Monitor token consumption for cost control:

#### TypeScript Token Tracking

```typescript
import { query } from '@anthropic-ai/claude-agent-sdk';

interface TokenUsage {
  inputTokens: number;
  outputTokens: number;
  totalTokens: number;
  estimatedCostUSD: number;
}

async function trackTokenUsage(
  prompt: string
): Promise<{ result: string; usage: TokenUsage }> {
  let inputTokens = 0;
  let outputTokens = 0;
  let result = '';

  const response = query({
    prompt,
    options: { model: "claude-3-5-sonnet-20241022" }
  });

  for await (const message of response) {
    if (message.type === 'assistant') {
      if (typeof message.content === 'string') {
        result += message.content;
      } else {
        message.content.forEach(block => {
          if (block.type === 'text') {
            result += block.text;
          }
        });
      }
    } else if (message.type === 'system' && message.subtype === 'completion') {
      if (message.usage) {
        inputTokens = message.usage.input_tokens;
        outputTokens = message.usage.output_tokens;
      }
    }
  }

  const totalTokens = inputTokens + outputTokens;
  
  // Pricing as of 2024 (Claude 3.5 Sonnet)
  const inputCostPer1kTokens = 0.003;
  const outputCostPer1kTokens = 0.015;
  
  const estimatedCost =
    (inputTokens / 1000) * inputCostPer1kTokens +
    (outputTokens / 1000) * outputCostPer1kTokens;

  return {
    result,
    usage: {
      inputTokens,
      outputTokens,
      totalTokens,
      estimatedCostUSD: estimatedCost
    }
  };
}

// Usage
trackTokenUsage("Generate a detailed technical specification").then(data => {
  console.log("Result:", data.result);
  console.log("Tokens - Input:", data.usage.inputTokens);
  console.log("Tokens - Output:", data.usage.outputTokens);
  console.log("Tokens - Total:", data.usage.totalTokens);
  console.log("Estimated Cost: $" + data.usage.estimatedCostUSD.toFixed(4));
});
```

#### Python Token Tracking

```python
import asyncio
from claude_agent_sdk import query, AssistantMessage, ResultMessage, TextBlock

async def track_token_usage(prompt: str) -> dict:
    input_tokens = 0
    output_tokens = 0
    result = ''

    async for message in query(prompt):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    result += block.text
        
        elif isinstance(message, ResultMessage):
            input_tokens = message.usage.input_tokens
            output_tokens = message.usage.output_tokens

    total_tokens = input_tokens + output_tokens
    
    # Pricing as of 2024 (Claude 3.5 Sonnet)
    input_cost_per_1k = 0.003
    output_cost_per_1k = 0.015
    
    estimated_cost = (
        (input_tokens / 1000) * input_cost_per_1k +
        (output_tokens / 1000) * output_cost_per_1k
    )

    return {
        "result": result,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": estimated_cost
        }
    }

# Usage
data = asyncio.run(track_token_usage("Generate a detailed technical specification"))
print(f"Result: {data['result']}")
print(f"Tokens - Input: {data['usage']['input_tokens']}")
print(f"Tokens - Output: {data['usage']['output_tokens']}")
print(f"Tokens - Total: {data['usage']['total_tokens']}")
print(f"Estimated Cost: ${data['usage']['estimated_cost_usd']:.4f}")
```

---

## Tools & Integration

### Built-in Tool Ecosystem

The Claude Agent SDK includes a rich set of built-in tools:

#### File Operations

**Read Files:**

```typescript
// TypeScript - Read file tool
const response = query({
  prompt: `Read and summarise the README.md file`,
  options: {
    allowedTools: ['Read'],
    workingDirectory: '/path/to/project',
    model: "claude-3-5-sonnet-20241022"
  }
});
```

**Write Files:**

```typescript
// TypeScript - Write file tool
const response = query({
  prompt: `Create a file named "output.txt" with a detailed report`,
  options: {
    allowedTools: ['Write'],
    permissionMode: 'acceptEdits',
    workingDirectory: '/path/to/project',
    model: "claude-3-5-sonnet-20241022"
  }
});
```

**Edit Files:**

```typescript
// TypeScript - Edit file tool
const response = query({
  prompt: `Fix the bug in auth.ts by updating the validation logic`,
  options: {
    allowedTools: ['Edit', 'Read'],
    permissionMode: 'acceptEdits',
    workingDirectory: '/path/to/project',
    model: "claude-3-5-sonnet-20241022"
  }
});
```

#### Bash Command Execution

```typescript
// TypeScript - Execute bash commands
const response = query({
  prompt: `Run npm test and report any failures`,
  options: {
    allowedTools: ['Bash', 'Read'],
    workingDirectory: '/path/to/project',
    model: "claude-3-5-sonnet-20241022"
  }
});
```

#### File Search Operations

```typescript
// TypeScript - Search operations
const response = query({
  prompt: `Find all TypeScript files that import '@anthropic-ai/claude-agent-sdk'`,
  options: {
    allowedTools: ['Glob', 'Grep', 'Read'],
    workingDirectory: '/path/to/project',
    model: "claude-3-5-sonnet-20241022"
  }
});
```

### Permission Modes

Control how agents use tools:

#### Accept Edits Mode

```typescript
// TypeScript - Automatically accept file modifications
const response = query({
  prompt: "Refactor this code for better performance",
  options: {
    allowedTools: ['Read', 'Write', 'Edit'],
    permissionMode: 'acceptEdits',  // Auto-approves edits
    model: "claude-3-5-sonnet-20241022"
  }
});
```

#### Default Mode with Prompting

```typescript
// TypeScript - Ask for confirmation
const response = query({
  prompt: "Update the configuration file",
  options: {
    allowedTools: ['Read', 'Write'],
    permissionMode: 'default',  // Prompts for each action
    model: "claude-3-5-sonnet-20241022"
  }
});
```

#### Custom Permission Callback

```typescript
// TypeScript - Fine-grained control
const response = query({
  prompt: "Perform system tasks",
  options: {
    allowedTools: ['Read', 'Write', 'Bash'],
    canUseTool: async (toolName, input) => {
      // Block destructive operations
      if (toolName === 'Bash' && input.command.includes('rm -rf')) {
        return {
          behavior: 'deny',
          message: 'Destructive operations not allowed'
        };
      }

      // Require confirmation for deploys
      if (toolName === 'Bash' && input.command.includes('deploy')) {
        return {
          behavior: 'ask',
          message: 'Confirm deployment to production?'
        };
      }

      return { behavior: 'allow' };
    },
    model: "claude-3-5-sonnet-20241022"
  }
});
```

#### Python Permission Control

```python
from claude_agent_sdk import (
    ClaudeAgentOptions,
    query,
    PermissionResultDeny,
    PermissionResultAllow
)

async def can_use_tool(tool_name: str, tool_input: dict, context):
    """Custom tool permission logic."""
    
    # Block writes to sensitive files
    if tool_name == "Write":
        file_path = tool_input.get("file_path", "")
        if "config" in file_path.lower() or "secret" in file_path.lower():
            return PermissionResultDeny(
                behavior="deny",
                message="Cannot write to sensitive files"
            )
    
    # Modify bash commands for safety
    if tool_name == "Bash":
        command = tool_input.get("command", "")
        if command.startswith("rm"):
            # Add confirmation flag
            modified_input = {**tool_input, "command": f"{command} -i"}
            return PermissionResultAllow(
                behavior="allow",
                updated_input=modified_input
            )
    
    return PermissionResultAllow(behavior="allow")

options = ClaudeAgentOptions(
    allowed_tools=["Read", "Write", "Bash"],
    can_use_tool=can_use_tool
)

async for message in query(
    prompt="Perform system tasks",
    options=options
):
    print(message)
```

### Custom Tool Creation with MCP

Custom tools are exposed via the Model Context Protocol:

#### Creating a Custom Tool

```typescript
// TypeScript - Custom tool example
import { query, createSdkMcpServer, tool } from '@anthropic-ai/claude-agent-sdk';
import { z } from 'zod';

// Define a weather tool
const weatherTool = tool(
  'get_weather',
  'Get current weather for a location',
  {
    location: z.string().describe('City name or coordinates'),
    units: z.enum(['celsius', 'fahrenheit']).default('celsius')
  },
  async (args) => {
    try {
      const response = await fetch(
        `https://api.weather.com/v1/current?location=${args.location}&units=${args.units}`
      );
      const data = await response.json();

      return {
        content: [{
          type: 'text',
          text: `Temperature: ${data.temp}° ${args.units === 'celsius' ? 'C' : 'F'}\nConditions: ${data.conditions}\nHumidity: ${data.humidity}%`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error fetching weather: ${(error as Error).message}`
        }],
        isError: true
      };
    }
  }
);

// Create MCP server with tools
const weatherServer = createSdkMcpServer({
  name: 'weather-service',
  version: '1.0.0',
  tools: [weatherTool]
});

// Use in agent
const response = query({
  prompt: "What's the weather in San Francisco?",
  options: {
    mcpServers: {
      'weather': weatherServer
    },
    allowedTools: ['mcp__weather__get_weather'],
    model: "claude-3-5-sonnet-20241022"
  }
});

for await (const message of response) {
  if (message.type === 'assistant') {
    console.log(message.content);
  }
}
```

#### Python Custom Tools

```python
from claude_agent_sdk import (
    tool,
    create_sdk_mcp_server,
    query,
    ClaudeAgentOptions
)

@tool(
    "get_weather",
    "Get current weather for a location",
    {"location": str, "units": str}
)
async def get_weather(args: dict) -> dict:
    import aiohttp
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"https://api.weather.com/v1/current?"
                f"location={args['location']}&"
                f"units={args.get('units', 'celsius')}"
            ) as resp:
                data = await resp.json()
        
        units_symbol = '°C' if args.get('units') == 'celsius' else '°F'
        return {
            "content": [{
                "type": "text",
                "text": (
                    f"Temperature: {data['temp']}{units_symbol}\n"
                    f"Conditions: {data['conditions']}\n"
                    f"Humidity: {data['humidity']}%"
                )
            }]
        }
    
    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"Error fetching weather: {str(e)}"
            }],
            "is_error": True
        }

# Create MCP server
weather_server = create_sdk_mcp_server(
    name="weather-service",
    version="1.0.0",
    tools=[get_weather]
)

# Use in agent
options = ClaudeAgentOptions(
    mcp_servers={"weather": weather_server},
    allowed_tools=["mcp__weather__get_weather"]
)

async for message in query(
    prompt="What's the weather in San Francisco?",
    options=options
):
    print(message)
```

---

## Multi-Agent Systems

### Agent Coordination Patterns

#### Sequential Coordination

```typescript
// TypeScript - Sequential agent execution
async function sequentialAnalysis() {
  // Agent 1: Data analysis
  const analysisResponse = query({
    prompt: "Analyse this dataset for trends",
    options: {
      systemPrompt: "You are a data analyst. Focus on statistical significance.",
      model: "claude-3-5-sonnet-20241022"
    }
  });

  let analysisResult = '';
  for await (const message of analysisResponse) {
    if (message.type === 'assistant') {
      if (typeof message.content === 'string') {
        analysisResult += message.content;
      }
    }
  }

  // Agent 2: Generate report based on analysis
  const reportResponse = query({
    prompt: `Based on this analysis:\n${analysisResult}\n\nGenerate an executive summary`,
    options: {
      systemPrompt: "You are a business analyst. Create concise, actionable summaries.",
      model: "claude-3-5-sonnet-20241022"
    }
  });

  let reportResult = '';
  for await (const message of reportResponse) {
    if (message.type === 'assistant') {
      if (typeof message.content === 'string') {
        reportResult += message.content;
      }
    }
  }

  return {
    analysis: analysisResult,
    report: reportResult
  };
}
```

#### Parallel Coordination

```typescript
// TypeScript - Parallel agent execution
async function parallelReview(codeSnippet: string) {
  const [securityReview, performanceReview, qualityReview] = await Promise.all([
    // Security review agent
    (async () => {
      let result = '';
      const response = query({
        prompt: `Review this code for security vulnerabilities:\n${codeSnippet}`,
        options: {
          systemPrompt: "You are a security expert. Identify and explain all security issues.",
          model: "claude-3-5-sonnet-20241022"
        }
      });

      for await (const message of response) {
        if (message.type === 'assistant') {
          if (typeof message.content === 'string') {
            result += message.content;
          }
        }
      }
      return result;
    })(),

    // Performance review agent
    (async () => {
      let result = '';
      const response = query({
        prompt: `Review this code for performance improvements:\n${codeSnippet}`,
        options: {
          systemPrompt: "You are a performance engineer. Identify bottlenecks and optimisations.",
          model: "claude-3-5-sonnet-20241022"
        }
      });

      for await (const message of response) {
        if (message.type === 'assistant') {
          if (typeof message.content === 'string') {
            result += message.content;
          }
        }
      }
      return result;
    })(),

    // Quality review agent
    (async () => {
      let result = '';
      const response = query({
        prompt: `Review this code for quality and best practices:\n${codeSnippet}`,
        options: {
          systemPrompt: "You are a code quality expert. Focus on maintainability and clarity.",
          model: "claude-3-5-sonnet-20241022"
        }
      });

      for await (const message of response) {
        if (message.type === 'assistant') {
          if (typeof message.content === 'string') {
            result += message.content;
          }
        }
      }
      return result;
    })()
  ]);

  return {
    securityReview,
    performanceReview,
    qualityReview
  };
}
```

#### Hierarchical Agent Structure

```typescript
// TypeScript - Hierarchical agents (coordinator + specialists)
interface AgentSpecialist {
  name: string;
  role: string;
  tools: string[];
  systemPrompt: string;
}

const specialists: Record<string, AgentSpecialist> = {
  researcher: {
    name: "Researcher",
    role: "Gathers and analysed information",
    tools: ["Read", "Glob", "Grep"],
    systemPrompt: "You are a research specialist. Find and synthesise information."
  },
  architect: {
    name: "Architect",
    role: "Designs solutions and systems",
    tools: ["Read"],
    systemPrompt: "You are a system architect. Design scalable solutions."
  },
  implementer: {
    name: "Implementer",
    role: "Implements solutions",
    tools: ["Read", "Write", "Edit"],
    systemPrompt: "You are a developer. Implement solutions with clean code."
  },
  reviewer: {
    name: "Reviewer",
    role: "Reviews work for quality",
    tools: ["Read", "Grep"],
    systemPrompt: "You are a quality reviewer. Check for issues and improvements."
  }
};

async function hierarchicalWorkflow(task: string) {
  const results: Record<string, string> = {};

  // Coordinator delegates to specialists
  for (const [specKey, spec] of Object.entries(specialists)) {
    console.log(`\n[${spec.name}] Processing...`);

    const response = query({
      prompt: `${task}\n\nYour role: ${spec.role}`,
      options: {
        systemPrompt: spec.systemPrompt,
        allowedTools: spec.tools,
        model: "claude-3-5-sonnet-20241022"
      }
    });

    let result = '';
    for await (const message of response) {
      if (message.type === 'assistant') {
        if (typeof message.content === 'string') {
          result += message.content;
        }
      }
    }

    results[specKey] = result;
  }

  return results;
}
```

### Shared Context Management

```typescript
// TypeScript - Shared context across agents
interface SharedContext {
  projectName: string;
  requirements: string[];
  decisions: Record<string, string>;
  results: Record<string, string>;
}

async function multiAgentWithSharedContext() {
  const context: SharedContext = {
    projectName: "ML Pipeline",
    requirements: [
      "Handle large datasets efficiently",
      "Provide real-time predictions",
      "Support model versioning"
    ],
    decisions: {},
    results: {}
  };

  // Agent 1: Requirements analysis
  context.decisions['requirements'] = `Analysed ${context.requirements.length} requirements`;

  // Agent 2: Architecture design (has access to shared context)
  const archResponse = query({
    prompt: `Design architecture for project: ${context.projectName}
Requirements:
${context.requirements.map((r, i) => `${i + 1}. ${r}`).join('\n')}

Previous decisions: ${JSON.stringify(context.decisions)}`,
    options: {
      model: "claude-3-5-sonnet-20241022"
    }
  });

  let architecture = '';
  for await (const message of archResponse) {
    if (message.type === 'assistant') {
      if (typeof message.content === 'string') {
        architecture += message.content;
      }
    }
  }

  context.results['architecture'] = architecture;

  // Agent 3: Implementation planning (has access to architecture)
  const implResponse = query({
    prompt: `Create implementation plan based on this architecture:
${architecture}`,
    options: {
      model: "claude-3-5-sonnet-20241022"
    }
  });

  let implementation = '';
  for await (const message of implResponse) {
    if (message.type === 'assistant') {
      if (typeof message.content === 'string') {
        implementation += message.content;
      }
    }
  }

  context.results['implementation'] = implementation;

  return context;
}
```

---

## Computer Use API

### Mouse and Keyboard Control

The Computer Use API allows agents to interact with graphical interfaces:

#### Screen Interaction

```typescript
// TypeScript - Computer use for UI automation
const response = query({
  prompt: `
Please help me automate the following task:
1. Open the web browser
2. Navigate to example.com
3. Search for "Claude Agent SDK"
4. Click on the first result
5. Take a screenshot of the page
6. Close the browser
`,
  options: {
    allowedTools: ['ComputerUse'],
    model: "claude-3-5-sonnet-20241022"
  }
});

for await (const message of response) {
  if (message.type === 'tool_call' && message.tool_name === 'ComputerUse') {
    console.log(`Computer action: ${JSON.stringify(message.input)}`);
  } else if (message.type === 'assistant') {
    console.log(`Agent: ${message.content}`);
  }
}
```

#### Mouse Control

Computer Use API supports:
- **Mouse movement**: `mouse_move(x, y)`
- **Clicks**: `left_click()`, `right_click()`, `double_click()`
- **Scrolling**: `scroll(direction, amount)`
- **Dragging**: `mouse_drag(start_x, start_y, end_x, end_y)`

#### Keyboard Control

Keyboard operations include:
- **Typing text**: `type(text)`
- **Key presses**: `key(key_name)` - Enter, Tab, Escape, Backspace, etc.
- **Keyboard shortcuts**: `key(Ctrl+C)`, `key(Cmd+V)`, etc.
- **Text selection**: `key(Ctrl+A)` combined with actions

### Computer Control Use Cases

#### Research and Data Collection

```typescript
// TypeScript - Automated research workflow
const response = query({
  prompt: `
Perform market research on the Claude Agent SDK:
1. Navigate to GitHub and search for "claude-agent-sdk"
2. Find the official repository
3. Extract key metrics (stars, forks, recent commits)
4. Navigate to the npm package page
5. Get current download statistics
6. Compile into a structured report
`,
  options: {
    allowedTools: ['ComputerUse', 'Write'],
    permissionMode: 'acceptEdits',
    model: "claude-3-5-sonnet-20241022"
  }
});

for await (const message of response) {
  if (message.type === 'assistant') {
    console.log(message.content);
  }
}
```

#### Video and Content Creation

```typescript
// TypeScript - Automated video editing
const response = query({
  prompt: `
Create a tutorial video for the Claude Agent SDK:
1. Open video editing software
2. Import the pre-recorded footage
3. Add title and intro text
4. Insert chapter markers at 5-minute intervals
5. Add captions from the transcript file
6. Apply professional transitions
7. Add background music from the assets folder
8. Export as MP4 with optimised settings
`,
  options: {
    allowedTools: ['ComputerUse', 'Read', 'Write'],
    workingDirectory: '/home/user/video-project',
    model: "claude-3-5-sonnet-20241022"
  }
});
```

#### Note-Taking and Document Management

```typescript
// Python - Automated note-taking
async def automated_notetaking():
    async for message in query(
        prompt="""
        Take structured notes from this meeting:
        1. Open note-taking application
        2. Create new document titled "Project Kickoff - 2024"
        3. Add sections: Objectives, Decisions, Action Items, Timeline
        4. Fill in details from the meeting materials
        5. Create a summary section
        6. Save and sync to cloud storage
        """
    ):
        if message.type == 'tool_call' and message.tool_name == 'ComputerUse':
            print(f"Action: {message.input}")
        elif isinstance(message, AssistantMessage):
            print(f"Status: {message.content}")
```

### Safety and Permissions with Computer Use

```typescript
// TypeScript - Safe computer use with permission control
const response = query({
  prompt: "Help me organize my files by creating folders and moving documents",
  options: {
    allowedTools: ['ComputerUse'],
    canUseTool: async (toolName, input) => {
      if (toolName === 'ComputerUse') {
        // Allow viewing and navigation
        if (input.action === 'screenshot' || input.action === 'mouse_move') {
          return { behavior: 'allow' };
        }

        // Require confirmation for system changes
        if (input.action === 'type' && input.text.includes('format') ||
            input.action === 'type' && input.text.includes('delete')) {
          return {
            behavior: 'ask',
            message: `Confirm: Execute this potentially destructive action?`
          };
        }

        // Restrict access to sensitive folders
        if (input.action === 'key' && input.key.includes('C:\\Windows') ||
            input.action === 'key' && input.key.includes('/etc')) {
          return {
            behavior: 'deny',
            message: 'Cannot access system folders'
          };
        }
      }

      return { behavior: 'allow' };
    },
    model: "claude-3-5-sonnet-20241022"
  }
});
```

---

## Model Context Protocol (MCP)

### MCP Fundamentals

MCP is a standardised protocol for tool integration:

```
┌─────────────────────────────────────────────────────┐
│              Claude Agent Application                │
└──────────────────┬──────────────────────────────────┘
                   │
                   │ (MCP Protocol)
                   │
        ┌──────────┼──────────┐
        │          │          │
        ▼          ▼          ▼
    ┌────────┐ ┌────────┐ ┌────────┐
    │ MCP    │ │ MCP    │ │ MCP    │
    │Server 1│ │Server 2│ │Server N│
    └────────┘ └────────┘ └────────┘
    (Custom)  (External) (Third-party)
```

### Creating MCP Servers

#### TypeScript MCP Server

```typescript
import { createSdkMcpServer, tool } from '@anthropic-ai/claude-agent-sdk';
import { z } from 'zod';

// Define multiple tools
const calculateTool = tool(
  'calculate',
  'Perform mathematical calculations',
  {
    expression: z.string().describe('Math expression to evaluate'),
    precision: z.number().default(2)
  },
  async (args) => {
    try {
      // Use a safe math evaluator
      const result = evaluateMath(args.expression);
      return {
        content: [{
          type: 'text',
          text: `Result: ${result.toFixed(args.precision)}`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error: ${(error as Error).message}`
        }],
        isError: true
      };
    }
  }
);

const convertTool = tool(
  'convert_units',
  'Convert between different units',
  {
    value: z.number(),
    fromUnit: z.enum(['m', 'km', 'mi', 'ft']),
    toUnit: z.enum(['m', 'km', 'mi', 'ft'])
  },
  async (args) => {
    const conversions: Record<string, number> = {
      'm': 1,
      'km': 1000,
      'mi': 1609.34,
      'ft': 0.3048
    };

    const meters = args.value * conversions[args.fromUnit];
    const result = meters / conversions[args.toUnit];

    return {
      content: [{
        type: 'text',
        text: `${args.value} ${args.fromUnit} = ${result.toFixed(2)} ${args.toUnit}`
      }]
    };
  }
);

// Create server
const mathServer = createSdkMcpServer({
  name: 'math-tools',
  version: '1.0.0',
  tools: [calculateTool, convertTool]
});

// Use in agent
export { mathServer };

// Usage example
import { query } from '@anthropic-ai/claude-agent-sdk';

const response = query({
  prompt: "Calculate the result of (15 + 25) * 3, then convert 100 miles to kilometres",
  options: {
    mcpServers: { 'math': mathServer },
    allowedTools: [
      'mcp__math__calculate',
      'mcp__math__convert_units'
    ],
    model: "claude-3-5-sonnet-20241022"
  }
});
```

#### Python MCP Server

```python
from claude_agent_sdk import (
    tool,
    create_sdk_mcp_server,
    query,
    ClaudeAgentOptions
)

@tool(
    "calculate",
    "Perform mathematical calculations",
    {"expression": str, "precision": int}
)
async def calculate(args: dict) -> dict:
    try:
        # Use a safe math evaluator
        result = eval(args["expression"])  # In production, use a safer evaluator
        precision = args.get("precision", 2)
        return {
            "content": [{
                "type": "text",
                "text": f"Result: {result:.{precision}f}"
            }]
        }
    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"Error: {str(e)}"
            }],
            "is_error": True
        }

@tool(
    "convert_units",
    "Convert between different units",
    {"value": float, "from_unit": str, "to_unit": str}
)
async def convert_units(args: dict) -> dict:
    conversions = {
        'm': 1,
        'km': 1000,
        'mi': 1609.34,
        'ft': 0.3048
    }

    from_unit = args["from_unit"]
    to_unit = args["to_unit"]
    value = args["value"]

    meters = value * conversions[from_unit]
    result = meters / conversions[to_unit]

    return {
        "content": [{
            "type": "text",
            "text": f"{value} {from_unit} = {result:.2f} {to_unit}"
        }]
    }

# Create server
math_server = create_sdk_mcp_server(
    name="math-tools",
    version="1.0.0",
    tools=[calculate, convert_units]
)

# Usage example
async def main():
    options = ClaudeAgentOptions(
        mcp_servers={"math": math_server},
        allowed_tools=[
            "mcp__math__calculate",
            "mcp__math__convert_units"
        ]
    )

    async for message in query(
        prompt="Calculate (15 + 25) * 3, then convert 100 miles to kilometres",
        options=options
    ):
        print(message)
```

### Integrating External MCP Servers

```typescript
// TypeScript - Using external MCP servers
import { query } from '@anthropic-ai/claude-agent-sdk';

const response = query({
  prompt: "List files in the current directory and analyze the git history",
  options: {
    mcpServers: {
      // In-process server (from SDK)
      'calculations': mathServer,

      // External stdio server
      'filesystem': {
        type: 'stdio',
        command: 'npx',
        args: ['@modelcontextprotocol/server-filesystem'],
        env: {
          ALLOWED_PATHS: '/home/user/projects:/tmp'
        }
      },

      // External git server
      'git': {
        type: 'stdio',
        command: 'node',
        args: ['/usr/local/bin/git-mcp-server.js'],
        env: {
          GIT_REPO_PATH: '/home/user/projects/myrepo'
        }
      },

      // HTTP/SSE server
      'remote-api': {
        url: 'https://api.example.com/mcp',
        headers: {
          'Authorization': `Bearer ${process.env.API_TOKEN}`
        }
      }
    },
    allowedTools: [
      'mcp__filesystem__list_files',
      'mcp__git__log',
      'mcp__remote-api__query'
    ],
    model: "claude-3-5-sonnet-20241022"
  }
});

for await (const message of response) {
  if (message.type === 'assistant') {
    console.log(message.content);
  }
}
```

---

## Structured Output

### Response Schemas

```typescript
// TypeScript - Define response schemas with Zod
import { z } from 'zod';
import { query } from '@anthropic-ai/claude-agent-sdk';

const ProductSchema = z.object({
  id: z.string(),
  name: z.string(),
  price: z.number(),
  inStock: z.boolean(),
  tags: z.array(z.string())
});

const APIResponseSchema = z.object({
  status: z.enum(['success', 'error']),
  data: z.array(ProductSchema),
  timestamp: z.string()
});

async function getStructuredResponse() {
  const response = query({
    prompt: `Generate sample product data in JSON format.
Return it in this exact structure:
{
  "status": "success",
  "data": [
    {
      "id": "string",
      "name": "string",
      "price": number,
      "inStock": boolean,
      "tags": ["string"]
    }
  ],
  "timestamp": "ISO8601 string"
}`,
    options: {
      model: "claude-3-5-sonnet-20241022"
    }
  });

  let jsonString = '';
  for await (const message of response) {
    if (message.type === 'assistant') {
      if (typeof message.content === 'string') {
        jsonString += message.content;
      }
    }
  }

  // Extract JSON and validate
  const jsonMatch = jsonString.match(/\{[\s\S]*\}/);
  if (!jsonMatch) {
    throw new Error('No JSON found in response');
  }

  const parsed = JSON.parse(jsonMatch[0]);
  const validated = APIResponseSchema.parse(parsed);
  
  return validated;
}
```

### JSON Mode

```python
# Python - Structured output with JSON mode
from claude_agent_sdk import query, ClaudeAgentOptions

async def get_structured_output():
    options = ClaudeAgentOptions(
        system_prompt="""You MUST respond with valid JSON only.
No markdown, no explanation, just the JSON object."""
    )

    result = {}
    async for message in query(
        prompt="""Generate a technical specification document in JSON format:
{
  "title": "...",
  "version": "...",
  "sections": [
    {
      "name": "...",
      "content": "..."
    }
  ]
}""",
        options=options
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    result['content'] = block.text

    # Parse and validate
    import json
    spec = json.loads(result['content'])
    return spec
```

---

## Session Management

### Session Lifecycle

```typescript
// TypeScript - Complete session lifecycle
import { query } from '@anthropic-ai/claude-agent-sdk';

interface SessionMetadata {
  sessionId?: string;
  startTime: Date;
  endTime?: Date;
  interactions: number;
  totalTokens: number;
}

async function completeSessionLifecycle() {
  const metadata: SessionMetadata = {
    startTime: new Date(),
    interactions: 0,
    totalTokens: 0
  };

  // Step 1: Initialize session
  console.log('[Session] Initializing...');
  const initResponse = query({
    prompt: "Start a new project planning session",
    options: {
      model: "claude-3-5-sonnet-20241022"
    }
  });

  for await (const message of initResponse) {
    if (message.type === 'system' && message.subtype === 'init') {
      metadata.sessionId = message.session_id;
      console.log(`[Session] Started with ID: ${metadata.sessionId}`);
    } else if (message.type === 'assistant') {
      metadata.interactions++;
    }
  }

  // Step 2: Continue in same session
  console.log('[Session] Continuing conversation...');
  if (metadata.sessionId) {
    const continueResponse = query({
      prompt: "Now let's define the project requirements",
      options: {
        resume: metadata.sessionId,
        model: "claude-3-5-sonnet-20241022"
      }
    });

    for await (const message of continueResponse) {
      if (message.type === 'assistant') {
        metadata.interactions++;
      } else if (message.type === 'system' && message.subtype === 'completion') {
        if (message.usage) {
          metadata.totalTokens += message.usage.input_tokens + message.usage.output_tokens;
        }
      }
    }
  }

  // Step 3: Fork session for alternative approach
  console.log('[Session] Forking for alternative approach...');
  if (metadata.sessionId) {
    const forkResponse = query({
      prompt: "Let's explore an alternative architecture approach",
      options: {
        resume: metadata.sessionId,
        forkSession: true,
        model: "claude-3-5-sonnet-20241022"
      }
    });

    let forkedSessionId: string | undefined;
    for await (const message of forkResponse) {
      if (message.type === 'system' && message.subtype === 'init') {
        forkedSessionId = message.session_id;
        console.log(`[Session] Forked to new session: ${forkedSessionId}`);
      }
    }
  }

  metadata.endTime = new Date();
  const duration = (metadata.endTime.getTime() - metadata.startTime.getTime()) / 1000;

  console.log(`\n[Session] Summary:
  Duration: ${duration}s
  Interactions: ${metadata.interactions}
  Total Tokens: ${metadata.totalTokens}
  Session ID: ${metadata.sessionId}`);

  return metadata;
}
```

### State Persistence

```python
# Python - Persistent session storage
import asyncio
import json
from pathlib import Path
from claude_agent_sdk import query, ClaudeAgentOptions

class SessionManager:
    def __init__(self, storage_dir: str = "./sessions"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)

    def save_session(self, session_id: str, metadata: dict):
        """Save session metadata to disk."""
        file_path = self.storage_dir / f"{session_id}.json"
        with open(file_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

    def load_session(self, session_id: str) -> dict | None:
        """Load session metadata from disk."""
        file_path = self.storage_dir / f"{session_id}.json"
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return None

    async def resume_session(self, session_id: str, new_prompt: str):
        """Resume a persisted session."""
        metadata = self.load_session(session_id)
        if not metadata:
            raise ValueError(f"Session {session_id} not found")

        async for message in query(
            prompt=new_prompt,
            options=ClaudeAgentOptions(resume=session_id)
        ):
            yield message

# Usage
async def main():
    manager = SessionManager()

    # Create new session
    session_id = None
    async for message in query(prompt="Start a research session"):
        if hasattr(message, 'session_id'):
            session_id = message.session_id
            manager.save_session(session_id, {
                "created_at": str(asyncio.get_event_loop().time()),
                "interactions": 1
            })

    # Later, resume the session
    if session_id:
        async for message in manager.resume_session(
            session_id,
            "Continue the research from before"
        ):
            print(message)
```

---

## Context Engineering

### System Prompt Design

```typescript
// TypeScript - Well-designed system prompts
import { query, ClaudeAgentOptions } from '@anthropic-ai/claude-agent-sdk';

// Example 1: Security-focused system prompt
const securityPrompt = `You are a senior security engineer specialising in application security.

**Your Responsibilities:**
1. Identify security vulnerabilities in code and systems
2. Assess risk levels and provide remediation strategies
3. Recommend security best practices and standards
4. Explain security implications in clear terms
5. Suggest preventative measures for future development

**Guidelines:**
- Always consider OWASP Top 10 vulnerabilities
- Evaluate both risks and likelihood of exploitation
- Provide practical, implementable solutions
- Consider performance vs security trade-offs
- Document findings with severity levels

**Format Requirements:**
- Always structure findings as: Issue → Risk Level → Impact → Remediation
- Include code examples for all recommendations
- Reference official security standards when applicable`;

// Example 2: Data analysis system prompt
const dataAnalysisPrompt = `You are a data scientist and statistical expert.

**Your Approach:**
1. Validate data quality and completeness first
2. Identify patterns, trends, and anomalies
3. Perform statistical significance testing
4. Generate actionable insights
5. Recommend further analysis or data collection

**Statistical Rigor:**
- Always report confidence intervals
- Test assumptions before applying methods
- Explain limitations of analysis
- Suggest alternatives for invalid assumptions
- Use appropriate significance levels (p < 0.05 default)

**Communication:**
- Explain technical findings in business terms
- Use visualisations to support findings
- Provide recommendations with confidence levels
- Suggest next steps for deeper analysis`;

// Example 3: Creative writing system prompt
const creativePrompt = `You are an award-winning creative writer and storyteller.

**Your Style:**
- Vivid, descriptive language that engages the senses
- Well-developed characters with distinct voices
- Compelling narrative arcs with tension and resolution
- Appropriate pacing for the story length
- Thematic depth beneath entertaining surface

**Your Process:**
1. Understand the core story concept
2. Develop compelling characters
3. Create tension and conflict
4. Build towards satisfying resolution
5. Layer in thematic elements

**Quality Standards:**
- Show, don't tell (use examples and dialogue)
- Vary sentence structure for rhythm
- Avoid clichés and overused phrases
- Ensure emotional authenticity
- End with impact`;

// Using system prompts
async function exampleUsage() {
  const securityResponse = query({
    prompt: "Review this authentication code for vulnerabilities",
    options: {
      systemPrompt: securityPrompt,
      model: "claude-3-5-sonnet-20241022"
    }
  });

  for await (const message of securityResponse) {
    if (message.type === 'assistant') {
      console.log(message.content);
    }
  }
}
```

### Few-Shot Prompting

```typescript
// TypeScript - Few-shot examples for better outputs
const fewShotPrompt = `You are a code reviewer focusing on code quality.

**Review Format Example:**
Issue: Function lacks error handling
Severity: Medium
Suggestion:
\`\`\`typescript
// Before
async function fetchData(url: string) {
  const response = await fetch(url);
  return response.json();
}

// After
async function fetchData(url: string) {
  try {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(\`HTTP \${response.status}: \${response.statusText}\`);
    }
    return response.json();
  } catch (error) {
    console.error('Failed to fetch data:', error);
    throw new Error('Data fetch failed');
  }
}
\`\`\`

Now review this code:
\`\`\`typescript
function processUserData(users: User[]) {
  return users.map(user => {
    return {
      ...user,
      email: user.email.toLowerCase(),
      fullName: user.first + ' ' + user.last
    };
  });
}
\`\`\`

Please identify issues following the format above.`;

const response = query({
  prompt: fewShotPrompt,
  options: {
    model: "claude-3-5-sonnet-20241022"
  }
});
```

---

## Advanced Permissions

### Permission Architecture

```typescript
// TypeScript - Comprehensive permission system
import { query, ClaudeAgentOptions } from '@anthropic-ai/claude-agent-sdk';

interface ToolPermission {
  tool: string;
  allowed: boolean;
  restrictions?: {
    pathPatterns?: string[];
    commandPatterns?: string[];
    requireApproval?: boolean;
  };
}

const permissionPolicy: ToolPermission[] = [
  {
    tool: 'Read',
    allowed: true,
    restrictions: {
      pathPatterns: ['/home/user/**', '/var/log/**']
    }
  },
  {
    tool: 'Write',
    allowed: true,
    restrictions: {
      pathPatterns: ['/home/user/documents/**', '/tmp/**'],
      requireApproval: true
    }
  },
  {
    tool: 'Bash',
    allowed: true,
    restrictions: {
      commandPatterns: ['npm *', 'git *', 'ls *'],
      requireApproval: true
    }
  },
  {
    tool: 'ComputerUse',
    allowed: false  // Completely disabled for this agent
  }
];

async function queryWithPermissions(prompt: string) {
  const response = query({
    prompt,
    options: {
      canUseTool: async (toolName, input) => {
        const permission = permissionPolicy.find(p => p.tool === toolName);

        if (!permission?.allowed) {
          return {
            behavior: 'deny',
            message: `Tool '${toolName}' is not permitted for this agent`
          };
        }

        // Check restrictions
        if (permission.restrictions) {
          // Check path patterns
          if (permission.restrictions.pathPatterns && input.path) {
            const isAllowed = permission.restrictions.pathPatterns.some(pattern =>
              matchPattern(input.path, pattern)
            );

            if (!isAllowed) {
              return {
                behavior: 'deny',
                message: `Path '${input.path}' is not permitted`
              };
            }
          }

          // Check command patterns
          if (permission.restrictions.commandPatterns && input.command) {
            const isAllowed = permission.restrictions.commandPatterns.some(pattern =>
              matchPattern(input.command, pattern)
            );

            if (!isAllowed) {
              return {
                behavior: 'deny',
                message: `Command pattern not permitted`
              };
            }
          }

          // Check if approval required
          if (permission.restrictions.requireApproval) {
            return {
              behavior: 'ask',
              message: `Confirm permission to execute: ${toolName}`
            };
          }
        }

        return { behavior: 'allow' };
      },
      model: "claude-3-5-sonnet-20241022"
    }
  });

  for await (const message of response) {
    console.log(message);
  }
}

function matchPattern(input: string, pattern: string): boolean {
  // Simple glob pattern matching
  const regex = pattern
    .replace(/\./g, '\\.')
    .replace(/\*/g, '.*')
    .replace(/\?/g, '.');
  return new RegExp(`^${regex}$`).test(input);
}
```

---

## Agentic Patterns

### Self-Correction Loop

```typescript
// TypeScript - Self-correcting agent
async function selfCorrectingAgent(task: string, maxAttempts = 3) {
  let attempts = 0;
  let previousAttempt = '';
  let result = null;

  while (attempts < maxAttempts && !result) {
    attempts++;
    console.log(`\n[Attempt ${attempts}/${maxAttempts}]`);

    const prompt = previousAttempt
      ? `${task}

Previous attempt (which had issues):
${previousAttempt}

Please try again, addressing any issues from the previous attempt.`
      : task;

    const response = query({
      prompt,
      options: {
        systemPrompt: `You are a problem-solving agent. 
When given a task, attempt to solve it.
If the previous attempt had issues, learn from them and try a different approach.
Always explain your reasoning.`,
        model: "claude-3-5-sonnet-20241022"
      }
    });

    let attemptResult = '';
    let hasErrors = false;

    for await (const message of response) {
      if (message.type === 'assistant') {
        if (typeof message.content === 'string') {
          attemptResult += message.content;
        }
      } else if (message.type === 'error') {
        hasErrors = true;
        console.log(`Error in attempt ${attempts}: ${message.error?.message}`);
      }
    }

    previousAttempt = attemptResult;

    // Validate result
    if (attemptResult.includes('[SUCCESS]') || attemptResult.includes('completed')) {
      result = attemptResult;
    } else if (attempts < maxAttempts && hasErrors) {
      console.log('Attempt failed, retrying...');
    } else if (attempts === maxAttempts) {
      console.log(`Max attempts (${maxAttempts}) reached`);
      result = attemptResult;
    }
  }

  return result;
}

// Usage
selfCorrectingAgent("Create a Python script that reads JSON and generates a CSV").then(result => {
  console.log("\nFinal Result:\n", result);
});
```

### Planning and Execution Separation

```typescript
// TypeScript - Separate planning from execution
async function planThenExecute(task: string) {
  console.log('[Phase 1] Planning...\n');

  // Phase 1: Create plan
  const planResponse = query({
    prompt: `Create a detailed step-by-step plan to accomplish this task:
${task}

Format your plan as a numbered list with clear, actionable steps.`,
    options: {
      systemPrompt: `You are a strategic planner.
Create detailed, practical plans that break complex tasks into manageable steps.
Consider dependencies between steps.
Identify potential challenges.`,
      model: "claude-3-5-sonnet-20241022"
    }
  });

  let plan = '';
  for await (const message of planResponse) {
    if (message.type === 'assistant') {
      if (typeof message.content === 'string') {
        plan += message.content;
      }
    }
  }

  console.log('Plan:\n', plan);

  console.log('\n[Phase 2] Execution...\n');

  // Phase 2: Execute plan
  const execResponse = query({
    prompt: `Execute this plan:
${plan}

Provide updates as you complete each step.
Note any challenges encountered and how you addressed them.`,
    options: {
      systemPrompt: `You are a task executor.
Follow the provided plan step by step.
Provide clear status updates for each step.
Adapt if you encounter issues not anticipated in the plan.`,
      model: "claude-3-5-sonnet-20241022"
    }
  });

  let execution = '';
  for await (const message of execResponse) {
    if (message.type === 'assistant') {
      if (typeof message.content === 'string') {
        execution += message.content;
      }
    }
  }

  console.log('Execution Result:\n', execution);

  return { plan, execution };
}
```

---

## API Reference

### Query Function

```typescript
// TypeScript - query() function signature
interface QueryOptions {
  // Model selection
  model?: string;

  // Agent customisation
  systemPrompt?: string;
  allowedTools?: string[];

  // Execution control
  workingDirectory?: string;
  maxTurns?: number;
  maxBudgetUsd?: number;

  // Permissions
  permissionMode?: 'acceptEdits' | 'default' | 'plan' | 'bypassPermissions';
  canUseTool?: (toolName: string, input: any) => Promise<PermissionResult>;

  // MCP integration
  mcpServers?: Record<string, MCPServerConfig>;

  // Session management
  resume?: string;
  forkSession?: boolean;

  // Streaming
  include_partial_messages?: boolean;

  // Other options
  settingSources?: string[];
  env?: Record<string, string>;
}

async function query(
  options: {
    prompt: string | AsyncIterable<string>;
    options?: QueryOptions;
  }
): AsyncIterableIterator<Message>
```

### Message Types

```typescript
interface AssistantMessage {
  type: 'assistant';
  content: string | ContentBlock[];
  model: string;
}

interface ToolCallMessage {
  type: 'tool_call';
  tool_name: string;
  input: any;
}

interface ToolResultMessage {
  type: 'tool_result';
  tool_name: string;
  result: string;
}

interface SystemMessage {
  type: 'system';
  subtype?: 'init' | 'completion' | 'progress';
  session_id?: string;
  usage?: { input_tokens: number; output_tokens: number };
}

interface ErrorMessage {
  type: 'error';
  error?: {
    type: string;
    message: string;
    code?: string;
  };
}
```

---

**This comprehensive guide covers the Claude Agent SDK from installation through advanced patterns. For additional examples and recipes, refer to the Recipes guide.**
