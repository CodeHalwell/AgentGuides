# Migration Guide: Claude Code SDK → Claude Agent SDK (TypeScript)

**Version:** 2025.1
**Target Audience:** TypeScript/JavaScript developers migrating from Claude Code SDK
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

The **Claude Code SDK** has evolved into the **Claude Agent SDK**, reflecting a major expansion from coding-focused workflows to comprehensive general-purpose agent capabilities.

### Why the Rebrand?

**Claude Code SDK** was designed for:
- Code generation and analysis
- Software development automation
- Engineering workflow optimization

**Claude Agent SDK** now enables:
- General-purpose task automation
- CSV processing and data manipulation
- Web research and content gathering
- Visualization and document generation
- Complex multi-agent orchestration
- Digital work automation
- **Everything the Code SDK did, plus much more**

### Support Timeline

- **Pre-2025**: Claude Code SDK (coding focus)
- **2025+**: Claude Agent SDK (general-purpose)
- **Deprecation**: Claude Code SDK support ends Q4 2025

---

## Critical Changes

### 1. Package Name Change

```typescript
// OLD - Claude Code SDK
import { code } from '@anthropic-ai/claude-code';

// NEW - Claude Agent SDK
import { query } from '@anthropic-ai/claude-agent-sdk';
```

**Key Difference:** The primary function changed from `code()` to `query()` to reflect broader capabilities.

### 2. Node.js Version Requirement

- **Minimum**: Node.js 18.0.0+ (previously 16.0.0+)
- **Recommended**: Node.js 20+ for optimal performance
- **Reason**: Leverage native fetch API and improved async handling

```bash
# Check your Node version
node --version

# Should output: v18.0.0 or higher
```

### 3. TypeScript Version Requirement

- **Minimum**: TypeScript 5.0+
- **Recommended**: TypeScript 5.3+
- **Reason**: Enhanced type inference and modern features

```bash
# Check TypeScript version
npx tsc --version

# Upgrade if needed
npm install -D typescript@latest
```

### 4. Model Names Updated

```typescript
// OLD
const options = {
  model: 'claude-3-5-sonnet-20241022'
};

// NEW - Claude Sonnet 4.5 (recommended for 2025)
const options = {
  model: 'claude-sonnet-4-5'
};

// Previous models still supported for compatibility
const compatOptions = {
  model: 'claude-3-5-sonnet-20241022'
};
```

### 5. Enhanced Built-in Tools

**New Built-in Tool Ecosystem:**

- `Read` - File reading operations
- `Write` - File creation and modification
- `Edit` - Precise file editing
- `Bash` - Command-line execution
- `Grep` - Content search
- `Glob` - File pattern matching
- `WebFetch` - Web content retrieval
- `WebSearch` - Internet search

```typescript
// Agents automatically select appropriate tools
const response = query({
  prompt: 'Search the web for TypeScript 5.3 features and create a summary document',
  options: {
    allowedTools: ['WebSearch', 'WebFetch', 'Write']
  }
});
```

---

## Step-by-Step Migration

### Step 1: Update Dependencies

**Remove Old Package:**

```bash
npm uninstall @anthropic-ai/claude-code
# or
yarn remove @anthropic-ai/claude-code
```

**Install New Package:**

```bash
npm install @anthropic-ai/claude-agent-sdk
# or
yarn add @anthropic-ai/claude-agent-sdk
```

**Update package.json:**

```json
{
  "dependencies": {
    "@anthropic-ai/claude-agent-sdk": "^1.0.0",
    "zod": "^3.22.0",
    "dotenv": "^16.3.0"
  },
  "devDependencies": {
    "typescript": "^5.3.0",
    "@types/node": "^20.10.0"
  },
  "engines": {
    "node": ">=18.0.0"
  }
}
```

### Step 2: Update Import Statements

**Automated Migration Script:**

```typescript
// migrate-imports.ts
import { readFileSync, writeFileSync } from 'fs';
import { glob } from 'glob';

async function migrateImports() {
  const files = await glob('src/**/*.{ts,tsx,js,jsx}');

  for (const file of files) {
    let content = readFileSync(file, 'utf-8');

    // Replace package imports
    content = content.replace(
      /@anthropic-ai\/claude-code/g,
      '@anthropic-ai/claude-agent-sdk'
    );

    // Replace code() with query()
    content = content.replace(
      /import\s+{\s*code\s*}/g,
      'import { query }'
    );

    // Replace function calls
    content = content.replace(
      /\bcode\(/g,
      'query('
    );

    // Update MCP server creation
    content = content.replace(
      /createMcpServer/g,
      'createSdkMcpServer'
    );

    writeFileSync(file, content);
    console.log(`✓ Migrated: ${file}`);
  }
}

// Run migration
migrateImports().catch(console.error);
```

**Run the migration:**

```bash
npx ts-node migrate-imports.ts
```

### Step 3: Update Configuration

**Environment Variables (NO CHANGES REQUIRED):**

```bash
# .env file
ANTHROPIC_API_KEY=sk-ant-your-key-here
ANTHROPIC_MODEL=claude-sonnet-4-5
```

**TypeScript Configuration:**

```typescript
// OLD
import { code, ClaudeOptions } from '@anthropic-ai/claude-code';

const options: ClaudeOptions = {
  model: 'claude-3-5-sonnet-20241022'
};

// NEW - Enhanced with 2025 features
import { query, QueryOptions } from '@anthropic-ai/claude-agent-sdk';

const options: QueryOptions = {
  model: 'claude-sonnet-4-5',       // Latest model
  maxBudgetUsd: 5.0,                 // Cost controls
  allowedTools: [                    // Fine-grained permissions
    'Read', 'Write', 'WebSearch', 'Bash'
  ],
  hooks: {                           // NEW: Hooks system
    preToolExecution: validateToolInput,
    postToolExecution: logToolResult
  },
  agents: {                          // NEW: Subagent configuration
    researcher: {
      description: 'Web research specialist',
      tools: ['WebSearch', 'WebFetch']
    }
  }
};
```

### Step 4: Update Code Patterns

**Basic Query Pattern:**

```typescript
// OLD
import { code } from '@anthropic-ai/claude-code';

async function main() {
  for await (const message of code({ prompt: 'Analyze this code' })) {
    console.log(message);
  }
}

// NEW
import { query } from '@anthropic-ai/claude-agent-sdk';

async function main() {
  for await (const message of query({
    prompt: 'Analyze this code',
    options: { model: 'claude-sonnet-4-5' }
  })) {
    console.log(message);
  }
}
```

**Advanced Patterns with 2025 Features:**

```typescript
import { query, QueryOptions } from '@anthropic-ai/claude-agent-sdk';

async function advancedAgent() {
  const options: QueryOptions = {
    model: 'claude-sonnet-4-5',
    systemPrompt: 'You are a research and analysis specialist.',
    allowedTools: ['WebSearch', 'WebFetch', 'Read', 'Write'],
    agents: {
      researcher: {
        description: 'Gathers information from the web',
        prompt: 'Research the topic thoroughly',
        tools: ['WebSearch', 'WebFetch']
      },
      analyst: {
        description: 'Analyzes gathered data',
        prompt: 'Analyze the research findings',
        tools: ['Read', 'Write']
      }
    }
  };

  for await (const message of query({
    prompt: 'Research AI safety trends and create a comprehensive report',
    options
  })) {
    if (message.type === 'system' && message.subtype === 'subagent_start') {
      console.log(`Starting subagent: ${message.agentName}`);
    } else if (message.type === 'assistant') {
      console.log(message.content);
    }
  }
}
```

---

## Breaking Changes

### 1. Function Name Change

**BREAKING:** Primary function renamed

```typescript
// OLD
import { code } from '@anthropic-ai/claude-code';
const response = code({ prompt: 'task' });

// NEW
import { query } from '@anthropic-ai/claude-agent-sdk';
const response = query({ prompt: 'task' });
```

### 2. MCP Server Creation

**BREAKING:** Function renamed and signature changed

```typescript
// OLD
import { createMcpServer } from '@anthropic-ai/claude-code';

const server = createMcpServer({
  name: 'my-tools',
  tools: [myTool]
});

// NEW
import { createSdkMcpServer } from '@anthropic-ai/claude-agent-sdk';

const server = createSdkMcpServer({
  name: 'my-tools',
  version: '1.0.0',  // NEW: Version required
  tools: [myTool]
});
```

### 3. Message Type Enhancements

**ENHANCED:** New message subtypes

```typescript
// OLD
if (message.type === 'system') {
  console.log('System message');
}

// NEW - More granular control
if (message.type === 'system') {
  switch (message.subtype) {
    case 'subagent_start':
      console.log(`Subagent started: ${message.agentName}`);
      break;
    case 'subagent_end':
      console.log(`Subagent completed: ${message.agentName}`);
      break;
    case 'hook_execution':
      console.log(`Hook executed: ${message.hookName}`);
      break;
  }
}
```

### 4. Permission Callback Signature

**ENHANCED:** Additional context parameter

```typescript
// OLD
type PermissionCallback = (
  toolName: string,
  input: Record<string, unknown>
) => Promise<PermissionResult>;

// NEW - Enhanced with context
type PermissionCallback = (
  toolName: string,
  input: Record<string, unknown>,
  context: {
    sessionId: string;
    previousTools: string[];
    costSoFar: number;
    timestamp: number;
  }
) => Promise<PermissionResult>;

// Usage
async function canUseTool(
  toolName: string,
  input: Record<string, unknown>,
  context: PermissionContext
): Promise<PermissionResult> {
  // Access context for advanced decisions
  if (context.costSoFar > 10.0) {
    return { behavior: 'deny', message: 'Budget exceeded' };
  }
  return { behavior: 'allow' };
}
```

### 5. Type System Updates

**ENHANCED:** Stronger type inference

```typescript
import { query, Message, AssistantMessage, ToolCallMessage } from '@anthropic-ai/claude-agent-sdk';

// Type guards now more precise
for await (const message of query({ prompt: 'task' })) {
  if (message.type === 'assistant') {
    // TypeScript knows this is AssistantMessage
    const content: string | MessageContent[] = message.content;
  } else if (message.type === 'tool_call') {
    // TypeScript knows this is ToolCallMessage
    const toolName: string = message.toolName;
    const input: Record<string, unknown> = message.input;
  }
}
```

---

## New Features in 2025

### 1. Claude Sonnet 4.5 Integration

**Latest Frontier Model:**

```typescript
import { query } from '@anthropic-ai/claude-agent-sdk';

const response = query({
  prompt: 'Design a complete microservices architecture for an e-commerce platform',
  options: {
    model: 'claude-sonnet-4-5',
    temperature: 1.0,
    maxTokens: 8192
  }
});

for await (const message of response) {
  console.log(message);
}
```

**Benefits of Claude Sonnet 4.5:**
- Superior multi-step reasoning
- Enhanced tool use accuracy
- Better context retention
- Improved planning capabilities
- 200K token context window

### 2. Subagents for Specialized Tasks

**NEW:** Parallel subagent execution

```typescript
import { query, QueryOptions } from '@anthropic-ai/claude-agent-sdk';

interface SubagentConfig {
  description: string;
  prompt?: string;
  tools?: string[];
  model?: string;
  temperature?: number;
}

const options: QueryOptions = {
  model: 'claude-sonnet-4-5',
  agents: {
    researcher: {
      description: 'Web research and information gathering',
      tools: ['WebSearch', 'WebFetch', 'Read'],
      model: 'claude-sonnet-4-5'
    },
    analyst: {
      description: 'Data analysis and insights generation',
      tools: ['Read', 'Write'],
      model: 'claude-3-5-sonnet'
    },
    visualizer: {
      description: 'Create charts and visualizations',
      tools: ['Write', 'Bash'],
      model: 'claude-3-5-haiku'
    }
  }
};

// Automatic subagent orchestration
for await (const message of query({
  prompt: 'Research AI trends, analyze the data, and create visualizations',
  options
})) {
  switch (message.subtype) {
    case 'subagent_start':
      console.log(`⚙️  Starting ${message.agentName}`);
      break;
    case 'subagent_end':
      console.log(`✓ Completed ${message.agentName}`);
      break;
  }
}
```

### 3. Hooks System

**NEW:** Inject logic at execution points

```typescript
import { query, QueryOptions, HookContext } from '@anthropic-ai/claude-agent-sdk';
import { z } from 'zod';

// Type-safe hook definitions
type PreToolHook = (
  toolName: string,
  toolInput: Record<string, unknown>,
  context: HookContext
) => Promise<{ allow: boolean; message?: string }>;

type PostToolHook = (
  toolName: string,
  toolResult: string,
  context: HookContext
) => Promise<void>;

// Pre-execution validation hook
const validateFileAccess: PreToolHook = async (toolName, toolInput, context) => {
  if (['Read', 'Write', 'Edit'].includes(toolName)) {
    const filePath = toolInput.filePath as string;

    // Block access to sensitive directories
    const forbidden = ['/etc', '/sys', '/root', 'C:\\Windows'];
    if (forbidden.some(dir => filePath.startsWith(dir))) {
      return {
        allow: false,
        message: `Access denied to ${filePath}`
      };
    }
  }

  return { allow: true };
};

// Post-execution audit hook
const auditToolUsage: PostToolHook = async (toolName, toolResult, context) => {
  await fetch('/api/audit', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      sessionId: context.sessionId,
      toolName,
      timestamp: context.timestamp,
      resultSize: toolResult.length
    })
  });
};

// Session lifecycle hooks
const onSessionStart = async (context: HookContext) => {
  console.log(`Session ${context.sessionId} started at ${new Date(context.timestamp)}`);
};

const onSessionEnd = async (context: HookContext) => {
  const duration = (Date.now() - context.timestamp) / 1000;
  console.log(`Session completed in ${duration.toFixed(2)}s`);
};

// Use hooks in configuration
const options: QueryOptions = {
  model: 'claude-sonnet-4-5',
  hooks: {
    preToolExecution: validateFileAccess,
    postToolExecution: auditToolUsage,
    sessionStart: onSessionStart,
    sessionEnd: onSessionEnd
  }
};

for await (const message of query({
  prompt: 'Read configuration files and generate documentation',
  options
})) {
  console.log(message);
}
```

### 4. Model Context Protocol (MCP) Enhancements

**ENHANCED:** Define TypeScript functions as tools

```typescript
import { tool, createSdkMcpServer, query } from '@anthropic-ai/claude-agent-sdk';
import { z } from 'zod';

// Type-safe tool with Zod schemas
const processCsvTool = tool(
  'process_csv',
  'Process CSV files with statistical analysis',
  {
    filePath: z.string().describe('Path to CSV file'),
    operations: z.array(z.enum(['summary', 'missing', 'outliers'])),
    outputFormat: z.enum(['json', 'markdown']).default('json')
  },
  async (args) => {
    // Mock CSV processing (replace with real logic)
    const stats = {
      rows: 1000,
      columns: ['id', 'name', 'value'],
      summary: {
        mean: 42.5,
        median: 40.0,
        std: 12.3
      }
    };

    const result = args.outputFormat === 'json'
      ? JSON.stringify(stats, null, 2)
      : `# CSV Statistics\nRows: ${stats.rows}\nColumns: ${stats.columns.join(', ')}`;

    return {
      content: [{
        type: 'text',
        text: result
      }]
    };
  }
);

// Database query tool
const queryDatabaseTool = tool(
  'query_database',
  'Execute SQL queries with type safety',
  {
    query: z.string().describe('SQL query to execute'),
    parameters: z.record(z.any()).optional(),
    timeout: z.number().default(30000)
  },
  async (args) => {
    // Mock database query (replace with real logic)
    const result = {
      rows: [
        { id: 1, name: 'Item 1' },
        { id: 2, name: 'Item 2' }
      ],
      rowCount: 2
    };

    return {
      content: [{
        type: 'text',
        text: JSON.stringify(result, null, 2)
      }]
    };
  }
);

// Create MCP server
const dataToolsServer = createSdkMcpServer({
  name: 'data-tools',
  version: '1.0.0',
  tools: [processCsvTool, queryDatabaseTool]
});

// Use in agent
const response = query({
  prompt: 'Process the sales.csv file and query the database for additional insights',
  options: {
    mcpServers: { 'data': dataToolsServer },
    allowedTools: [
      'mcp__data__process_csv',
      'mcp__data__query_database'
    ]
  }
});
```

### 5. General-Purpose Agent Development

**NEW:** Beyond coding tasks

```typescript
import { query, QueryOptions } from '@anthropic-ai/claude-agent-sdk';

// Example 1: Research Agent
async function researchAgent(topic: string) {
  const options: QueryOptions = {
    model: 'claude-sonnet-4-5',
    systemPrompt: 'You are a research specialist with web access.',
    allowedTools: ['WebSearch', 'WebFetch', 'Write']
  };

  for await (const message of query({
    prompt: `Research: ${topic}

    Tasks:
    1. Search for recent articles and papers
    2. Gather key findings
    3. Identify trends
    4. Create a comprehensive report in Markdown
    5. Save as research_report.md`,
    options
  })) {
    if (message.type === 'assistant') {
      console.log(message.content);
    }
  }
}

// Example 2: Data Processing Agent
async function dataProcessor(csvFile: string) {
  const options: QueryOptions = {
    model: 'claude-sonnet-4-5',
    systemPrompt: 'You are a data analysis specialist.',
    allowedTools: ['Read', 'Write', 'Bash']
  };

  for await (const message of query({
    prompt: `Process ${csvFile}

    Tasks:
    1. Read and analyze the data
    2. Calculate statistics
    3. Identify outliers
    4. Create visualizations
    5. Generate summary report`,
    options
  })) {
    console.log(message);
  }
}

// Example 3: Content Creation Agent
async function contentCreator(topic: string, format: string) {
  const options: QueryOptions = {
    model: 'claude-3-5-sonnet',
    systemPrompt: 'You are a content creation specialist.',
    allowedTools: ['WebSearch', 'Write']
  };

  for await (const message of query({
    prompt: `Create ${format} content about: ${topic}

    Requirements:
    1. Research the topic
    2. Outline the structure
    3. Write engaging content
    4. Format appropriately
    5. Save the final document`,
    options
  })) {
    console.log(message);
  }
}

// Example 4: Automation Agent
async function automationAgent(workflow: string) {
  const options: QueryOptions = {
    model: 'claude-sonnet-4-5',
    allowedTools: ['Read', 'Write', 'Bash', 'WebFetch']
  };

  for await (const message of query({
    prompt: `Automate: ${workflow}

    Analyze the workflow, break it down, and execute each step.`,
    options
  })) {
    console.log(message);
  }
}
```

### 6. Enhanced Built-in Tools

**Complete Tool Ecosystem:**

```typescript
import { query, QueryOptions } from '@anthropic-ai/claude-agent-sdk';

const options: QueryOptions = {
  model: 'claude-sonnet-4-5',
  allowedTools: [
    'Read',       // Read file contents
    'Write',      // Create/overwrite files
    'Edit',       // Modify file sections
    'Bash',       // Execute shell commands
    'Grep',       // Search file contents
    'Glob',       // File pattern matching
    'WebSearch',  // Internet search
    'WebFetch'    // Fetch web content
  ],
  permissionMode: 'default'  // Require approval for modifications
};

for await (const message of query({
  prompt: `
  1. Search the web for TypeScript best practices
  2. Create a summary document
  3. Find all TypeScript files in this project
  4. Run the test suite
  `,
  options
})) {
  if (message.type === 'tool_call') {
    console.log(`Using tool: ${message.toolName}`);
  }
}
```

---

## Code Migration Examples

### Example 1: Simple Agent

**Before (Claude Code SDK):**

```typescript
import { code } from '@anthropic-ai/claude-code';

async function main() {
  for await (const message of code({
    prompt: 'Explain async/await in TypeScript'
  })) {
    console.log(message);
  }
}

main();
```

**After (Claude Agent SDK):**

```typescript
import { query, QueryOptions } from '@anthropic-ai/claude-agent-sdk';

async function main() {
  const options: QueryOptions = {
    model: 'claude-sonnet-4-5',
    maxBudgetUsd: 1.0,
    temperature: 0.8
  };

  for await (const message of query({
    prompt: 'Explain async/await in TypeScript with modern examples',
    options
  })) {
    console.log(message);
  }
}

main();
```

### Example 2: Custom Tool

**Before (Claude Code SDK):**

```typescript
import { tool, createMcpServer } from '@anthropic-ai/claude-code';

const myTool = tool('my_tool', 'Description', async (args) => {
  return { result: 'data' };
});

const server = createMcpServer({
  name: 'my-server',
  tools: [myTool]
});
```

**After (Claude Agent SDK):**

```typescript
import { tool, createSdkMcpServer } from '@anthropic-ai/claude-agent-sdk';
import { z } from 'zod';

const myTool = tool(
  'my_tool',
  'Enhanced tool with Zod validation',
  {
    inputData: z.string(),
    options: z.record(z.any()).optional()
  },
  async (args) => {
    return {
      content: [{
        type: 'text',
        text: `Processed: ${args.inputData}`
      }]
    };
  }
);

const server = createSdkMcpServer({
  name: 'my-server',
  version: '2.0.0',  // Required
  tools: [myTool]
});
```

### Example 3: Multi-Agent System

**Before (Claude Code SDK):**

```typescript
import { code } from '@anthropic-ai/claude-code';

async function pipeline() {
  let result1 = '';

  // Agent 1
  for await (const msg of code({ prompt: 'Task 1' })) {
    if (msg.type === 'assistant') result1 += msg.content;
  }

  let result2 = '';

  // Agent 2 (uses result1)
  for await (const msg of code({ prompt: `Task 2: ${result1}` })) {
    if (msg.type === 'assistant') result2 += msg.content;
  }

  return result2;
}
```

**After (Claude Agent SDK with Subagents):**

```typescript
import { query, QueryOptions } from '@anthropic-ai/claude-agent-sdk';

async function pipeline() {
  const options: QueryOptions = {
    model: 'claude-sonnet-4-5',
    agents: {
      agent1: {
        description: 'First task specialist',
        tools: ['Read', 'WebSearch']
      },
      agent2: {
        description: 'Second task specialist',
        tools: ['Write', 'Bash']
      }
    }
  };

  // Automatic orchestration
  for await (const message of query({
    prompt: 'Complete Task 1, then use results for Task 2',
    options
  })) {
    if (message.subtype === 'subagent_start') {
      console.log(`Starting: ${message.agentName}`);
    } else if (message.subtype === 'subagent_end') {
      console.log(`Completed: ${message.agentName}`);
    }
  }
}
```

---

## Troubleshooting

### Issue 1: Import Errors

**Problem:**

```
Cannot find module '@anthropic-ai/claude-code'
```

**Solution:**

```bash
# Uninstall old package
npm uninstall @anthropic-ai/claude-code

# Install new package
npm install @anthropic-ai/claude-agent-sdk

# Update imports
# From: import { code } from '@anthropic-ai/claude-code'
# To:   import { query } from '@anthropic-ai/claude-agent-sdk'
```

### Issue 2: Node Version Compatibility

**Problem:**

```
Error: Node.js version 16.x is not supported
```

**Solution:**

```bash
# Check Node version
node --version

# Upgrade using nvm
nvm install 20
nvm use 20

# Or using system package manager
# macOS:
brew install node@20

# Ubuntu:
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs
```

### Issue 3: Type Errors After Migration

**Problem:**

```typescript
Property 'code' does not exist on type...
```

**Solution:**

```typescript
// OLD (causes error)
import { code } from '@anthropic-ai/claude-agent-sdk';

// CORRECT
import { query } from '@anthropic-ai/claude-agent-sdk';

// Update all function calls
const response = query({ prompt: 'task' });
```

### Issue 4: MCP Server Version Required

**Problem:**

```
Argument of type '{ name: string; tools: Tool[]; }' is not assignable...
Property 'version' is missing
```

**Solution:**

```typescript
// Add version property
const server = createSdkMcpServer({
  name: 'my-server',
  version: '1.0.0',  // Add this
  tools: [myTool]
});
```

### Issue 5: Hooks Type Errors

**Problem:**

```
Type 'Function' is not assignable to type 'PreToolHook'
```

**Solution:**

```typescript
// Ensure proper type signature
import { HookContext } from '@anthropic-ai/claude-agent-sdk';

const myHook = async (
  toolName: string,
  toolInput: Record<string, unknown>,
  context: HookContext
): Promise<{ allow: boolean; message?: string }> => {
  return { allow: true };
};

// Use in options
const options: QueryOptions = {
  hooks: {
    preToolExecution: myHook  // Correctly typed
  }
};
```

---

## Migration Checklist

### Pre-Migration

- [ ] Audit current Claude Code SDK usage
- [ ] Document all custom tools
- [ ] List all MCP server configurations
- [ ] Check Node.js version (must be 18+)
- [ ] Check TypeScript version (must be 5.0+)
- [ ] Backup current codebase

### Migration Steps

- [ ] Install Claude Agent SDK
- [ ] Run automated import migration script
- [ ] Update `code()` calls to `query()`
- [ ] Rename `createMcpServer` to `createSdkMcpServer`
- [ ] Add version numbers to MCP servers
- [ ] Update model names to Claude Sonnet 4.5
- [ ] Update permission callbacks with context parameter
- [ ] Add type annotations for better type safety
- [ ] Test basic functionality
- [ ] Test custom tools
- [ ] Implement hooks if needed
- [ ] Configure subagents for workflows

### Post-Migration

- [ ] Run full test suite
- [ ] Verify all tools work correctly
- [ ] Monitor costs with new models
- [ ] Update documentation
- [ ] Train team on new features
- [ ] Remove old package
- [ ] Deploy to staging environment
- [ ] Monitor production deployment

---

## Additional Resources

### Official Documentation

- **Claude Agent SDK Docs**: https://docs.anthropic.com/agent-sdk
- **TypeScript API Reference**: https://docs.anthropic.com/agent-sdk/typescript
- **Migration Support**: https://docs.anthropic.com/agent-sdk/migration

### Community Resources

- **GitHub Repository**: https://github.com/anthropics/claude-agent-sdk
- **Example Projects**: https://github.com/anthropics/claude-agent-sdk-examples
- **Discord Community**: https://discord.gg/anthropic
- **Stack Overflow**: Tag `claude-agent-sdk`

### Getting Help

1. Check official documentation
2. Search GitHub issues
3. Ask in Discord community
4. Contact Anthropic support (enterprise)

---

## Summary

**Key Takeaways:**

1. **Package Renamed**: `@anthropic-ai/claude-code` → `@anthropic-ai/claude-agent-sdk`
2. **Function Renamed**: `code()` → `query()`
3. **Node 18+ Required**: Upgrade from Node 16
4. **TypeScript 5+ Required**: Update TypeScript
5. **New Model**: Claude Sonnet 4.5 recommended
6. **Enhanced Features**: Subagents, hooks, expanded tools
7. **General-Purpose**: Beyond coding tasks

**Migration Effort:**

- **Simple Projects**: 1-2 hours (automated migration)
- **Medium Projects**: 4-8 hours (custom tools, testing)
- **Complex Projects**: 1-2 days (full integration, subagents, hooks)

**Recommended Approach:**

1. Use automated migration script for imports
2. Update one module at a time
3. Test thoroughly before production
4. Leverage new features incrementally

---

**Ready to migrate?** Start with the automated script and update incrementally!

**Questions?** Check the comprehensive guide or community resources.
