# SmolAgents Architecture & Flow Diagrams

## Visual Guide to SmolAgents Concepts

This document provides comprehensive ASCII diagrams and conceptual visualisations of SmolAgents' architecture, workflows, and design patterns.

---

## 1. SmolAgents Framework Architecture

### High-Level Component Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SmolAgents Framework                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              User Application Code                           │  │
│  │  • Task definition                                           │  │
│  │  • Agent configuration                                       │  │
│  │  • Result handling                                           │  │
│  └─────────────────────────────┬──────────────────────────────┘  │
│                                │                                 │
│  ┌──────────────────────────────┴──────────────────────────────┐  │
│  │        Agent Classes (Request Processing)                   │  │
│  ├──────────────────────┬─────────────────────────────────────┤  │
│  │   CodeAgent          │   ToolCallingAgent                  │  │
│  │  • Code generation   │  • JSON tool calling               │  │
│  │  • Python execution  │  • Traditional workflows            │  │
│  │  • Loop support      │  • Structured ordering              │  │
│  │  • Composability     │  • Legacy compatibility             │  │
│  └──────────────────────┴─────────────────────────────────────┘  │
│                                │                                 │
│  ┌──────────────────────────────┴──────────────────────────────┐  │
│  │         LLM Model Layer (Abstract Interface)                │  │
│  ├──────────────┬──────────────┬──────────────────────────────┤  │
│  │ InferenceC.  │  LiteLLMModel │ TransformersModel           │  │
│  │ (HF Infer.)  │ (100+ providers) │ (Local models)            │  │
│  └──────────────┴──────────────┴──────────────────────────────┘  │
│                                │                                 │
│  ┌──────────────────────────────┴──────────────────────────────┐  │
│  │           Tool System (Core Abstraction)                    │  │
│  ├──────────────┬──────────────┬──────────────────────────────┤  │
│  │ @tool        │ Tool subclass │ MCP tools   │ Hub Spaces     │  │
│  │ decorator    │ (stateful)    │ (protocol)  │ (gradio API)   │  │
│  └──────────────┴──────────────┴──────────────────────────────┘  │
│                                │                                 │
│  ┌──────────────────────────────┴──────────────────────────────┐  │
│  │         Execution Engines (Code Runtime)                    │  │
│  ├──────────┬────────┬─────────┬────────┬──────────────────┤  │
│  │  Local   │ Docker │ E2B     │ Modal  │ WebAssembly      │  │
│  │ Python   │        │ Cloud   │ Lambda │ (browser)        │  │
│  └──────────┴────────┴─────────┴────────┴──────────────────┘  │
│                                │                                 │
│  ┌──────────────────────────────┴──────────────────────────────┐  │
│  │        Hub Integration & Persistence                        │  │
│  │  • Agent sharing (push_to_hub)                             │  │
│  │  • Agent loading (from_hub)                                │  │
│  │  • Version management                                       │  │
│  │  • Community tools                                          │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow: Single Agent Execution

```
User Input (Natural Language Task)
        │
        ▼
┌─────────────────────────────────┐
│  Agent.run(task_description)    │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│ Build System Prompt                         │
│ • Tool descriptions                         │
│ • Available capabilities                    │
│ • Execution guidelines                      │
└────────────┬────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│ Send to LLM                                 │
│ Prompt: [system] + [task description]       │
└────────────┬────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│ LLM Response                                │
│ CodeAgent: Python code with tool calls      │
│ ToolCallingAgent: JSON tool definitions     │
└────────────┬────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│ Execute Generated Code/Calls                │
│ • Sandbox execution                         │
│ • Capture stdout/stderr                     │
│ • Handle errors gracefully                  │
└────────────┬────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│ Collect Observations                        │
│ • Tool execution results                    │
│ • Errors (if any)                          │
│ • Return values                             │
└────────────┬────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────┐
│ Decision: Continue or Stop?                 │
│ • max_steps reached? → Stop                 │
│ • Agent called final_answer? → Stop         │
│ • Error occurred? → Stop or retry           │
│ • Else → Loop to LLM                        │
└────────────┬────────────────────────────────┘
             │
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
  STOP            CONTINUE
    │                 │
    │                 ▼
    │        ┌───────────────────────┐
    │        │ Send Results to LLM   │
    │        │ Task + previous steps │
    │        │ + new observations    │
    │        └──────────┬────────────┘
    │                   │
    │                   └─→ Execute Generated Code (loops back)
    │
    ▼
┌─────────────────────────────────────────────┐
│ Return Final Result                         │
│ • output: final answer string               │
│ • steps: list of all iterations             │
│ • success: boolean status                   │
│ • token_usage: input/output counts          │
└─────────────────────────────────────────────┘
        │
        ▼
    Return to User
```

---

## 2. CodeAgent vs ToolCallingAgent

### Execution Paradigm Comparison

```
CODEAGENT: "Agents That Think in Code"
═══════════════════════════════════════════════════════════

    LLM                          Agent                      World
    ───                          ─────                      ─────
    
    │                             │                          │
    │  "Find Bitcoin price,       │                          │
    │   calculate 2%"             │                          │
    │  (Natural language)         │                          │
    │                             │                          │
    ├────────────────────────────►│                          │
    │                             │                          │
    │ ◄────────────────────────────┤                          │
    │ Returns Python code:        │                          │
    │                             │                          │
    │ btc_price = web_search( │                          │
    │     "bitcoin price"      │                          │
    │ )                        │                          │
    │ percentage = btc_price * 0.02
    │ answer = f"2% = {percentage}"
    │                             │                          │
    │                             ├─────────────────────────►│
    │                             │ Execute code            │
    │                             │ (single step)           │
    │                             │                          │
    │                             │◄─────────────────────────┤
    │                             │ Results returned        │
    │                             │                          │
    ├◄────────────────────────────┤                          │
    │ Results: btc_price=67000    │                          │
    │ Answer: "2% = 1340"         │                          │
    │                             │                          │
    ▼ No more iterations needed   ▼                          ▼


TOOLCALLINGAGENT: "Traditional JSON Tool Calling"
═══════════════════════════════════════════════════════════

    LLM                          Agent                      World
    ───                          ─────                      ─────
    
    │                             │                          │
    │  "Find Bitcoin price,       │                          │
    │   calculate 2%"             │                          │
    │                             │                          │
    ├────────────────────────────►│                          │
    │                             │                          │
    │ ◄────────────────────────────┤                          │
    │ Returns:                    │                          │
    │ {                           │                          │
    │   "tool": "web_search",     │                          │
    │   "args": {                 │                          │
    │     "query": "bitcoin price"│                          │
    │   }                         │                          │
    │ }                           │                          │
    │                             ├─────────────────────────►│
    │                             │ Call web_search         │
    │                             │                          │
    │                             │◄─────────────────────────┤
    │                             │ Result: 67000           │
    │ "Call 1 complete"           │                          │
    ├◄────────────────────────────┤                          │
    │                             │                          │
    │  "Now I need to calculate   │                          │
    │   2% of 67000"              │                          │
    │                             │                          │
    │  Process result, think      │                          │
    │  about what to do next      │                          │
    │                             │                          │
    ├────────────────────────────►│                          │
    │                             │                          │
    │ ◄────────────────────────────┤                          │
    │ Returns:                    │                          │
    │ {                           │                          │
    │   "tool": "calculator",     │                          │
    │   "args": {                 │                          │
    │     "operation": "multiply",│                          │
    │     "a": 67000,             │                          │
    │     "b": 0.02               │                          │
    │   }                         │                          │
    │ }                           │                          │
    │                             ├─────────────────────────►│
    │                             │ Call calculator         │
    │                             │                          │
    │                             │◄─────────────────────────┤
    │                             │ Result: 1340            │
    │ "Call 2 complete"           │                          │
    ├◄────────────────────────────┤                          │
    │                             │                          │
    │  "Report result"            │                          │
    │                             │                          │
    ├────────────────────────────►│                          │
    │ Returns:                    │                          │
    │ {                           │                          │
    │   "tool": "report",         │                          │
    │   "answer": "2% = 1340"     │                          │
    │ }                           │                          │
    │ "Call 3 complete"           │                          │
    │                             │                          │
    ▼ Multiple iterations needed  ▼                          ▼

KEY DIFFERENCE: CodeAgent completes in 1 LLM call + 1 execution.
                ToolCallingAgent requires 3+ LLM calls for same task.
```

---

## 3. Tool Architecture & Integration

### Tool Creation Methods

```
┌─────────────────────────────────────────────────────────────────┐
│                 Two Paths to Tool Creation                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Path A: @tool Decorator              Path B: Tool Subclass    │
│  ────────────────────────────          ──────────────────────   │
│                                                                 │
│  ┌─────────────────────────────┐    ┌─────────────────────────┐
│  │ @tool                       │    │ class MyTool(Tool):     │
│  │ def my_function():          │    │   name = "my_tool"      │
│  │   """Docstring"""           │    │   description = "..."   │
│  │   return result             │    │   inputs = {...}        │
│  │                             │    │   output_type = "str"   │
│  └──────────┬──────────────────┘    │                         │
│             │                       │   def forward(self):    │
│             │                       │     return result       │
│             │                       │                         │
│             │                       └──────────┬──────────────┘
│             │                                  │
│             ▼                                  ▼
│  • Simple functions              • Complex stateful logic
│  • Minimal code                  • Resource management
│  • Fast to write                 • Pre/post-processing
│  • Pure functions                • Database connections
│  • Best for: simple tools        • Best for: complex tools
│                                  • Connection pooling
│                                  • Cached resources
│
└─────────────────────────────────────────────────────────────────┘
```

### Tool Request & Execution Flow

```
┌──────────────────────────────────────────────────┐
│  Registered Tools (available to agent)           │
├──────────────────────────────────────────────────┤
│                                                  │
│  Tool 1: WebSearchTool                          │
│  ├─ name: "web_search"                         │
│  ├─ description: "Search the web"              │
│  ├─ inputs: {query: string}                    │
│  └─ forward(query) → results                   │
│                                                  │
│  Tool 2: Calculator                             │
│  ├─ name: "calculate"                          │
│  ├─ description: "Perform calculations"        │
│  ├─ inputs: {expr: string}                     │
│  └─ forward(expr) → result                     │
│                                                  │
│  Tool 3: DatabaseQuery                         │
│  ├─ name: "query_db"                           │
│  ├─ description: "Query database"              │
│  ├─ inputs: {sql: string}                      │
│  └─ forward(sql) → rows                        │
│                                                  │
└────────┬──────────────────────────────────────┘
         │
         │ Agent receives task: "Find top customers"
         │
         ▼
┌──────────────────────────────────────────────────┐
│  Generate System Prompt with Tools               │
├──────────────────────────────────────────────────┤
│                                                  │
│  Available tools:                                │
│                                                  │
│  def web_search(query: str) -> str:             │
│      """Search the web"""                       │
│                                                  │
│  def calculate(expr: str) -> float:             │
│      """Perform calculations"""                 │
│                                                  │
│  def query_db(sql: str) -> list:                │
│      """Query database"""                       │
│                                                  │
│  Generate Python code to solve the task.        │
│                                                  │
└────────┬──────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────┐
│  LLM Generates Code                              │
├──────────────────────────────────────────────────┤
│                                                  │
│  top_customers = query_db(                      │
│      "SELECT * FROM customers ORDER BY         │
│       lifetime_value DESC LIMIT 10"             │
│  )                                               │
│                                                  │
│  analysis = f"Found {len(top_customers)} top"  │
│      + f" customers with total value: "         │
│      + f"{sum(c['value'] for c in top_...)}"   │
│                                                  │
└────────┬──────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────┐
│  Agent Executes Code                             │
├──────────────────────────────────────────────────┤
│                                                  │
│  Sandbox Environment:                            │
│  ├─ query_db() → calls DatabaseQuery.forward()│
│  │                  ├─ Connection pooling      │
│  │                  ├─ SQL validation          │
│  │                  ├─ Execute query           │
│  │                  └─ Return results          │
│  │                                              │
│  ├─ Processing results                          │
│  │                                              │
│  └─ Capture: analysis variable                 │
│                                                  │
└────────┬──────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────┐
│  Return Results                                  │
├──────────────────────────────────────────────────┤
│                                                  │
│  analysis = "Found 10 top customers with        │
│             total value: $2,450,000"            │
│                                                  │
└──────────────────────────────────────────────────┘
```

---

## 4. Multi-Agent Orchestration

### Hierarchical Multi-Agent System

```
┌────────────────────────────────────────────────────────────────┐
│           Project Manager (Coordinator Agent)                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Task: "Create comprehensive market analysis"           │  │
│  │  ├─ Sub-task 1: Research market trends                 │  │
│  │  ├─ Sub-task 2: Analyse competitors                    │  │
│  │  └─ Sub-task 3: Write executive summary                │  │
│  └──────────────┬───────────────────────────────────────┬──┘  │
│                 │                                       │       │
│       ┌─────────▼──────────┐               ┌──────────▼────────┐
│       │  Research Agent    │               │ Analyst Agent     │
│       │                    │               │                   │
│       │  Tools:            │               │  Tools:           │
│       │  • WebSearchTool   │               │  • PythonTool     │
│       │  • Wikipedia API   │               │  • DatabaseTool   │
│       │                    │               │  • StatsTool      │
│       │  Task:             │               │                   │
│       │  "Find market      │               │  Task:            │
│       │   trends in AI"    │               │  "Analyse data"   │
│       │                    │               │                   │
│       └────────┬───────────┘               └─────────┬─────────┘
│               │                                     │
│               │  Result: "Market growing            │  Result: "Market
│               │  at 25% CAGR, led by               │  will reach $2.5T
│               │  cloud AI applications"            │  by 2030"
│               │                                    │
│       ┌───────┴──────────────────────────────────┴─────────┐
│       │  Writing Agent (Composition)                       │
│       │                                                    │
│       │  Task: Write summary based on:                    │
│       │  • Research findings                              │
│       │  • Analysis results                               │
│       │                                                    │
│       │  Output:                                           │
│       │  ┌──────────────────────────────────────────────┐ │
│       │  │  EXECUTIVE SUMMARY                          │ │
│       │  │                                              │ │
│       │  │  Market Opportunity:                        │ │
│       │  │  The AI market is experiencing rapid growth │ │
│       │  │  (25% CAGR) with projected value reaching  │ │
│       │  │  $2.5T by 2030. Cloud-based AI solutions   │ │
│       │  │  lead the sector.                           │ │
│       │  └──────────────────────────────────────────────┘ │
│       │                                                    │
│       └────────────┬──────────────────────────────────────┘
│                    │
└────────────────────┼────────────────────────────────────────┘
                     │
                     ▼
              Final Report to User
```

### Parallel Agent Processing

```
┌─────────────────────────────────────────────────────────────┐
│  Task: Process 1,000 customer inquiries concurrently        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Main Coordinator                                          │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  Split inquiries into batches (100 per agent)      │  │
│  │  Launch 10 agent instances in parallel             │  │
│  │  Collect results as they complete                  │  │
│  └────────────┬────────────────────────────────────┬──┘  │
│               │                                    │      │
│  Agent 1      │  Agent 2      │ ... │  Agent 10  │      │
│  ┌────────┐   │  ┌────────┐   │     │  ┌────────┐│      │
│  │Process │   │  │Process │   │     │  │Process ││      │
│  │  100   │   │  │  100   │   │     │  │  100   ││      │
│  │inquiries   │  │inquiries   │     │  │inquiries│      │
│  └─────┬──┘   │  └─────┬──┘   │     │  └─────┬──┘      │
│        │      │        │      │     │        │         │
│  ┌─────▼──────┴────┐   │      │ ... │        │         │
│  │  Immediate     │   │      │     │        │         │
│  │  Result: 85    │   │      │     │        │         │
│  │  Escalated: 15 │   │      │     │        │         │
│  └────────────────┘   │      │     │        │         │
│                       │      │     │        │         │
│                  ┌─────▼──────┴────┐ ... │        │     │
│                  │  Result: 92     │     │        │     │
│                  │  Result: 8      │     │        │     │
│                  └──────────────────┘     │        │     │
│                                          │        │     │
│                                      ┌───▼────────▼──┐  │
│                                      │  Result: 88   │  │
│                                      │  Result: 12   │  │
│                                      └───────────────┘  │
│                                                        │
│  Aggregation:                                          │
│  ┌────────────────────────────────────────────────┐   │
│  │ Total Immediate: 850                          │   │
│  │ Total Escalated: 150                          │   │
│  │ Resolution Rate: 85%                          │   │
│  │ Processing Time: 3.2 seconds (vs 32 seconds  │   │
│  │                  sequentially)                │   │
│  └────────────────────────────────────────────────┘   │
│                                                        │
└────────────────────────────────────────────────────────┘
```

---

## 5. Code Execution Sandbox Options

### Executor Type Hierarchy & Isolation

```
┌────────────────────────────────────────────────────────────┐
│         Execution Isolation Level                          │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  WASM (Browser)          ◄ Maximum Isolation             │
│  • Browser sandbox                                        │
│  • Offline capable                                        │
│  • Limited resources                                      │
│  • Perfect for: Client-side agents                       │
│  └─ Use: executor_type="wasm"                           │
│                                                           │
│  E2B (Cloud)             ◄ Strong Isolation             │
│  • Cloud-managed sandbox                                │
│  • Auto-scaling                                          │
│  • Secure environment                                    │
│  • Perfect for: Production deployments                  │
│  └─ Use: executor_type="e2b"                           │
│                                                           │
│  Modal (Serverless)      ◄ Managed Isolation           │
│  • Serverless containers                                │
│  • Auto-scaling                                          │
│  • Limited time (15 min)                                │
│  • Perfect for: Short-lived tasks                       │
│  └─ Use: executor_type="modal"                         │
│                                                           │
│  Docker (Local)          ◄ Container Isolation         │
│  • Local Docker container                              │
│  • Full control                                         │
│  • Some resource limits                                │
│  • Perfect for: Development & testing                  │
│  └─ Use: executor_type="docker"                       │
│                                                           │
│  Local Python            ◄ Minimal Isolation           │
│  • Same Python process                                 │
│  • No isolation                                        │
│  • Fastest execution                                   │
│  • Perfect for: Trusted code only                      │
│  └─ Use: executor_type="local" (default)              │
│                                                           │
└────────────────────────────────────────────────────────────┘
```

### Code Execution Flow with Sandboxing

```
Agent Generated Code:
────────────────────

    btc_price = web_search("bitcoin price")
    usd_to_eur = 0.92
    eur_price = float(btc_price) * usd_to_eur
    answer = f"Bitcoin in EUR: {eur_price}"


Execution Routing:
──────────────────

    ┌─────────────────────────────────────────────────────┐
    │  Choose Executor Based on Config                    │
    └────────┬────────────────────────────────────────────┘
             │
    ┌────────┴─────────────────────────────────────────┐
    │                                                  │
    ▼ executor_type="local"                           ▼ executor_type="docker"
    
    ┌──────────────────────┐               ┌──────────────────────┐
    │  Current Process     │               │  Docker Container    │
    │                      │               │                      │
    │  Python Interpreter  │               │  Isolated Python     │
    │  ├─ globals()        │               │  ├─ Separate vars    │
    │  ├─ locals()         │               │  ├─ No access to     │
    │  ├─ Execute code     │               │  │  host files       │
    │  └─ Access host      │               │  ├─ Network isolated │
    │                      │               │  └─ Resource limits  │
    │  Security: ✗ Low     │               │  Security: ✓ Medium  │
    │  Speed: ✓ Fastest    │               │  Speed: ○ Slower     │
    └──────────────────────┘               └──────────────────────┘


    ▼ executor_type="e2b"                     ▼ executor_type="wasm"
    
    ┌──────────────────────┐               ┌──────────────────────┐
    │  E2B Cloud Sandbox   │               │  Browser WASM        │
    │                      │               │                      │
    │  Managed Container   │               │  WebAssembly Runtime │
    │  ├─ Auto-scaling     │               │  ├─ Client-side      │
    │  ├─ Ephemeral        │               │  ├─ Offline capable  │
    │  ├─ Timeout: 1hr     │               │  ├─ Timeout: limited │
    │  └─ Full API access  │               │  └─ Limited libs     │
    │                      │               │                      │
    │  Security: ✓ High    │               │  Security: ✓ Very Hi │
    │  Speed: ○ Medium     │               │  Speed: ✓ Very Fast  │
    └──────────────────────┘               └──────────────────────┘


Output Handling:
────────────────

    ┌─────────────────────────────────────┐
    │  Execution Result                   │
    ├─────────────────────────────────────┤
    │                                     │
    │  ✓ Success:                         │
    │    Variables captured               │
    │    • btc_price = "67850"           │
    │    • eur_price = 62,442.0          │
    │    • answer = "Bitcoin in EUR:..."│
    │                                    │
    │  ✗ Error:                          │
    │    Exception caught                │
    │    • Type: ValueError              │
    │    • Message: "invalid literal"   │
    │    • Line: 2                       │
    │    • Recovery: Retry or escalate   │
    │                                     │
    └─────────────────────────────────────┘
          │
          ▼
    Return to Agent
```

---

## 6. Memory & Conversation State

### Conversation History Management

```
Agent Lifecycle with Memory
═════════════════════════════════════════════════════════════

Session Start:
┌────────────────────────┐
│ agent = CodeAgent(...) │
│ memory_size=3          │  ◄ Remember last 3 interactions
└──────────┬─────────────┘
           │
           ▼
┌───────────────────────────────────────────────────────┐
│ Memory Buffer (empty initially)                      │
│ ┌─────────────────────────────────────────────────┐ │
│ │  [Slot 0] - Empty                              │ │
│ │  [Slot 1] - Empty                              │ │
│ │  [Slot 2] - Empty                              │ │
│ └─────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────┘
           │
           ▼
First Interaction:
┌────────────────────────────────────────┐
│ agent.run("What is the capital of     │
│            France?")                   │
└──────────┬─────────────────────────────┘
           │
           ▼
┌───────────────────────────────────────────────────────┐
│ Memory Buffer                                        │
│ ┌─────────────────────────────────────────────────┐ │
│ │  [Slot 0] ← "What is capital of France?"       │ │
│ │           → "Paris"                             │ │
│ │  [Slot 1] - Empty                              │ │
│ │  [Slot 2] - Empty                              │ │
│ └─────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────┘
           │
           ▼
Second Interaction (agent remembers):
┌────────────────────────────────────────┐
│ agent.run("And France borders which    │
│            countries?")                 │
│                                        │
│ Agent's context:                       │
│ ← "What is capital of France?"        │
│ → "Paris"                             │
│ ← "And France borders which...?"      │
└──────────┬─────────────────────────────┘
           │
           ▼
┌───────────────────────────────────────────────────────┐
│ Memory Buffer                                        │
│ ┌─────────────────────────────────────────────────┐ │
│ │  [Slot 0] - "What is capital of France?" →    │ │
│ │             "Paris"                            │ │
│ │  [Slot 1] ← "And France borders which..."     │ │
│ │           → "Spain, Germany, Italy, ..."       │ │
│ │  [Slot 2] - Empty                              │ │
│ └─────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────┘
           │
           ▼
Third Interaction (buffer filling):
┌────────────────────────────────────────┐
│ agent.run("Calculate the area of       │
│            France.")                    │
└──────────┬─────────────────────────────┘
           │
           ▼
┌───────────────────────────────────────────────────────┐
│ Memory Buffer (now full)                             │
│ ┌─────────────────────────────────────────────────┐ │
│ │  [Slot 0] - "What is capital of France?" →    │ │
│ │             "Paris"                            │ │
│ │  [Slot 1] - "And France borders which..." →   │ │
│ │             "Spain, Germany, Italy, ..."       │ │
│ │  [Slot 2] ← "Calculate area of France."       │ │
│ │           → "643,801 km²"                      │ │
│ └─────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────┘
           │
           ▼
Fourth Interaction (buffer rotates):
┌────────────────────────────────────────┐
│ agent.run("What's France's GDP?")      │
└──────────┬─────────────────────────────┘
           │
           ▼
┌───────────────────────────────────────────────────────┐
│ Memory Buffer (oldest entry removed)                 │
│ ┌─────────────────────────────────────────────────┐ │
│ │  [Slot 0] - "France borders..." → "Spain,..."│ │
│ │  [Slot 1] - "Calculate area of France" →     │ │
│ │             "643,801 km²"                    │ │
│ │  [Slot 2] ← "What's France's GDP?" →        │ │
│ │           → "$2.78 trillion"                 │ │
│ └─────────────────────────────────────────────────┘ │
│                                                      │
│ Note: Earliest interaction forgotten                │
│       (first question about capital)                │ │
└───────────────────────────────────────────────────────┘
```

### State Preservation Across Tasks

```
┌─────────────────────────────────────────────────────────┐
│  Persistent Agent State                                │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Agent Configuration (unchanging):                     │
│  ┌─────────────────────────────────────────────────┐   │
│  │  model: InferenceClientModel(...)              │   │
│  │  tools: [WebSearchTool(), PythonTool()]        │   │
│  │  max_steps: 10                                 │   │
│  │  executor_type: "docker"                       │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  Runtime State (accumulating):                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Run 1:                                         │   │
│  │  ├─ Input: "Find Bitcoin price"               │   │
│  │  ├─ Steps: 2                                   │   │
│  │  ├─ Output: "67850 USD"                       │   │
│  │  └─ Tokens: 345 in, 89 out                    │   │
│  │                                                │   │
│  │  Run 2:                                         │   │
│  │  ├─ Input: "Convert to EUR"                   │   │
│  │  ├─ Context: Remembers "Bitcoin 67850"       │   │
│  │  ├─ Steps: 1                                   │   │
│  │  ├─ Output: "62,442 EUR"                      │   │
│  │  └─ Tokens: 287 in, 45 out                    │   │
│  │                                                │   │
│  │  Run 3:                                         │   │
│  │  ├─ Input: "Calculate 10% gain"               │   │
│  │  ├─ Context: Remembers BTC & EUR prices      │   │
│  │  ├─ Steps: 1                                   │   │
│  │  ├─ Output: "6,244.20 EUR gain"               │   │
│  │  └─ Tokens: 298 in, 52 out                    │   │
│  │                                                │   │
│  │  Cumulative:                                   │   │
│  │  ├─ Total runs: 3                             │   │
│  │  ├─ Total steps: 4                            │   │
│  │  ├─ Total tokens: 930 in, 186 out            │   │
│  │  └─ Session time: 8.3 seconds                │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 7. Agent Decision Tree (When to Use What)

```
                        ┌─────────────────┐
                        │  Build an Agent │
                        └────────┬────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                        │
              Complexity?                Expressivity?
                    │                        │
        ┌───────────┼───────────┐   ┌────────┼─────────┐
        │           │           │   │        │         │
       LOW        MED         HIGH  │      NEED       SIMPLE
        │           │           │   │     CODE?      JSON?
        │           │           │   │      │         │
        ▼           ▼           ▼   │      │         │
    ┌──────┐   ┌──────┐   ┌──────┐ │      ▼         ▼
    │Simple│   │Multi-│   │Complex│ │   ┌───────┐ ┌──────────┐
    │JSON  │   │step  │   │logic  │ │   │Code   │ │Traditional
    │calls │   │reason│   │needed │ │   │Agent  │ │ToolCalling
    └──────┘   └──────┘   └──────┘ │   └───────┘ └──────────┘
        │           │           │   │      ▲          ▲
        └───────────┴───────────┘   │      │          │
                                    │      └──────────┘
                                    │
                    ┌───────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
     YES, need      NO, simple
     complex         JSON
     logic           calls
        │                │
        ▼                ▼
    ┌──────────┐    ┌──────────┐
    │CodeAgent │    │ToolCalling
    │          │    │Agent
    │Wins when:│    │          │
    │• Loops   │    │Wins when:
    │• Conds.  │    │• Side effects
    │• Vars    │    │• Strict order
    │• Compose │    │• Legacy sys
    └──────────┘    └──────────┘
```

---

## 8. Performance Characteristics

### Latency Comparison: CodeAgent vs ToolCallingAgent

```
Task: Multi-step data analysis
(Search → Process → Analyse → Report)

Time ────────────────────────────────────────────────

ToolCallingAgent (3-4 LLM calls):
├─ Call 1: LLM generates search → 400ms
├─ Call 1: Execute search → 800ms
├─ Call 2: LLM thinks → 350ms
├─ Call 2: Process data → 200ms
├─ Call 3: LLM thinks → 350ms
├─ Call 3: Analyse → 300ms
├─ Call 4: LLM thinks → 350ms
└─ Call 4: Format report → 50ms
   Total: ~3,200ms (3.2 seconds)

CodeAgent (1 LLM call + execution):
├─ LLM generates code → 400ms
├─ Execute all steps → 1,200ms
│  ├─ Search: 800ms
│  ├─ Process: 200ms
│  ├─ Analyse: 300ms
│  └─ Format: 50ms (no network delay, all in one block)
└─ Total: ~1,600ms (1.6 seconds)

Efficiency Gain: 50% faster (1,600 vs 3,200 ms)
─────────────────────────────────────────────────────

Token Usage (typical):

ToolCallingAgent:
├─ Call 1: 200 in, 80 out
├─ Call 2: 185 in, 75 out
├─ Call 3: 190 in, 70 out
├─ Call 4: 180 in, 65 out
└─ Total: 755 in, 290 out

CodeAgent:
├─ Single call: 200 in, 150 out
└─ Total: 200 in, 150 out

Token efficiency: ~40% fewer tokens consumed
```

---

## 9. Deployment Architecture

### Production Deployment Flow

```
┌────────────────────────────────────────────────────────┐
│        SmolAgents in Production                        │
├────────────────────────────────────────────────────────┤
│                                                        │
│  ┌──────────────────────────────────────────────────┐ │
│  │  API Gateway / Load Balancer                    │ │
│  │  (FastAPI / Flask / Serverless)                 │ │
│  └────────────┬─────────────────────────────────┬──┘ │
│               │                                  │     │
│       ┌───────▼───────┐              ┌──────────▼──┐ │
│       │  Worker Pool  │              │   Cache     │ │
│       │               │              │   Layer     │ │
│       │ ┌───┐ ┌───┐  │              │             │ │
│       │ │ A │ │ B │  │              │  Redis /    │ │
│       │ │ g │ │ g │  │              │  Memcached  │ │
│       │ │ e │ │ e │  │              │             │ │
│       │ │ n │ │ n │  │              │ • Results   │ │
│       │ │ t │ │ t │  │              │ • Sessions  │ │
│       │ │ 1 │ │ 2 │  │              │             │ │
│       │ └─┬─┘ └─┬─┘  │              └─────────────┘ │
│       │   │     │    │                      ▲       │
│       │   └──┬──┘    │                      │       │
│       │      │       │                      │       │
│       └──────┼───────┘                      │       │
│              │                              │       │
│       ┌──────▼──────────────────────────────┴─────┐ │
│       │    LLM Model Service                      │ │
│       │                                           │ │
│       │  ┌─────────────────────────────────────┐ │ │
│       │  │ Model Selection & Load Balancing    │ │ │
│       │  │                                     │ │ │
│       │  │ InferenceClient → HF API           │ │ │
│       │  │ LiteLLM → 100+ providers           │ │ │
│       │  │ TransformersModel → Local GPU      │ │ │
│       │  └─────────────────────────────────────┘ │ │
│       └────────────────────────────────────────────┘ │
│              │                                      │
│       ┌──────▼──────────────────────────────────┐  │
│       │    Execution Layer (Sandboxing)         │  │
│       │                                         │  │
│       │  ┌──────────┬────────┬────────────┐    │  │
│       │  │  Docker  │  E2B   │   Modal    │    │  │
│       │  │Containers│ Cloud  │ Serverless │    │  │
│       │  └──────────┴────────┴────────────┘    │  │
│       └────────────────────────────────────────┘  │
│              │                                    │
│       ┌──────▼──────────────────────────────┐   │
│       │    Monitoring & Observability        │   │
│       │                                     │   │
│       │  • Agent execution metrics          │   │
│       │  • Token usage tracking             │   │
│       │  • Latency monitoring               │   │
│       │  • Error logging                    │   │
│       │  • Cost tracking                    │   │
│       └─────────────────────────────────────┘   │
│                                                │
└────────────────────────────────────────────────────┘
```

---

This comprehensive diagram guide visualises all major concepts, workflows, and architectural patterns in SmolAgents. Refer back to these diagrams when implementing agents or troubleshooting issues.


