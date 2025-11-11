# Pydantic AI: Architecture & Flow Diagrams

Visual representations of Pydantic AI patterns, architectures, and workflows.

---

## 1. Agent Lifecycle

```
┌─────────────────────────────────────────────────────────┐
│                    AGENT CREATION                        │
│  Agent('openai:gpt-4o', instructions='...', deps_type)  │
└─────────────┬───────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│              SYSTEM PROMPT EVALUATION                    │
│  • Static prompts loaded                                │
│  • Dynamic prompts executed                             │
│  • System behavior configured                           │
└─────────────┬───────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│          USER PROMPT + DEPENDENCIES                      │
│  • User message provided                                │
│  • RunContext created with deps                         │
│  • Message history prepared                             │
└─────────────┬───────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│            TOOL AVAILABILITY CHECK                       │
│  • Analyse available tools                              │
│  • Generate JSON schemas                                │
│  • Prepare for model                                    │
└─────────────┬───────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────┐
│         LLM API CALL (With Full Context)                │
│  • Model: gpt-4o, claude-3-5-sonnet, etc.              │
│  • System prompt                                        │
│  • Message history                                      │
│  • Available tools                                      │
│  • Model settings (temperature, etc.)                   │
└─────────────┬───────────────────────────────────────────┘
              │
              ▼
         ┌────────────────┐
         │ Model Response │
         └────────────────┘
              │
       ┌──────┴──────┬──────────────┬──────────────┐
       │             │              │              │
   TEXT_ONLY    TOOL_CALL    STRUCTURED    TEXT+TOOL
       │             │              │              │
       ▼             ▼              ▼              ▼
    Output    Tool Execution   Validation    Multiple Steps
    │             │              │              │
    └──────────────┴──────────────┴──────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ Output Validation    │
        │ • Pydantic models    │
        │ • Custom validators  │
        │ • ModelRetry logic   │
        └──────────┬───────────┘
                   │
           ┌───────┴────────┐
           │                │
       Valid          Retry Loop
       │                │
       ▼                ▼
   Return        Re-invoke Model
   Result        (with retry prompt)
```

---

## 2. Type Safety Flow

```
┌─────────────────────────────────────────────────┐
│        USER-PROVIDED DATA / LLM OUTPUT          │
│  "Extract user info from: John Doe, 25, NYC"   │
└─────────┬───────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────┐
│      PYDANTIC MODEL DEFINITION (Type-Safe)      │
│  class User(BaseModel):                         │
│      name: str                                  │
│      age: int (0 <= age <= 150)               │
│      city: str                                  │
│      email: Optional[str]                       │
└─────────┬───────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────┐
│        JSON SCHEMA GENERATION                   │
│  • Automatic from type annotations              │
│  • Sent to LLM for structured generation        │
│  • Enforces constraints (min, max, regex, etc)  │
└─────────┬───────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────┐
│        LLM GENERATES STRUCTURED JSON            │
│  {                                              │
│    "name": "John Doe",                          │
│    "age": 25,                                   │
│    "city": "NYC"                                │
│  }                                              │
└─────────┬───────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────┐
│       PYDANTIC VALIDATION (Runtime)             │
│  • Type checking (age: int)                     │
│  • Constraints (0 <= age <= 150)                │
│  • Required fields                              │
│  • Custom validators                            │
└─────────┬───────────────────────────────────────┘
          │
      ┌───┴────┐
      │         │
   Valid     Invalid
      │         │
      ▼         ▼
   User     ModelRetry
   Object   (Ask model to fix)
      │         │
      │         └──→ Back to LLM
      │              (with error details)
      │
      ▼
  ✅ Type-safe, validated data
  Ready for application logic
```

---

## 3. Dependency Injection Pattern

```
┌──────────────────────────────────┐
│   APPLICATION INITIALIZATION     │
│  • Setup dependencies            │
│  • Configure services            │
│  • Prepare agent                 │
└──────────────┬───────────────────┘
               │
               ▼
        ┌─────────────────┐
        │  Dependencies   │
        │  @dataclass     │
        │  - db_conn      │
        │  - cache        │
        │  - logger       │
        └────────┬────────┘
                 │
                 ▼
        ┌──────────────────────────────────┐
        │  Create Agent with deps_type     │
        │  Agent(..., deps_type=Deps)      │
        └─────────────┬────────────────────┘
                      │
                      ▼
           ┌────────────────────┐
           │ User calls agent:  │
           │ agent.run(         │
           │   'Query...',      │
           │   deps=my_deps     │
           │ )                  │
           └──────────┬─────────┘
                      │
                      ▼
        ┌──────────────────────────────────┐
        │   RunContext created              │
        │   - deps injected                │
        │   - messages tracked             │
        │   - model available              │
        └─────────────┬────────────────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
        ▼                           ▼
    @agent.tool             @agent.system_prompt
    async def tool:         async def prompt:
      ctx.deps.db_conn        await ctx.deps.cache.get()
      await ctx.deps.cache    
        │                       │
        └───────────┬───────────┘
                    │
                    ▼
            Tool executes with
            full context access
```

---

## 4. Multi-Agent Coordination (A2A Protocol)

```
                    ┌──────────────────────────┐
                    │  COORDINATOR AGENT       │
                    │  (Orchestrator)          │
                    └────────┬─────────────────┘
                             │
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
     ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
     │RESEARCH_AGT  │  │ WRITER_AGT   │  │ EDITOR_AGT   │
     │              │  │              │  │              │
     │• Gathers     │  │• Structures  │  │• Reviews     │
     │  information │  │  content     │  │  quality     │
     │• Validates   │  │• Generates   │  │• Suggests    │
     │  sources     │  │  drafts      │  │  improvements│
     └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
            │                 │                 │
            └────────────────→ Tools ←──────────┘
                      (Call each other)

WORKFLOW:
1. Coordinator: "Research topic X"
2. ResearchAgent: Gathers information
3. Coordinator: "Write article using research"
4. WriterAgent: Creates draft
5. Coordinator: "Edit the article"
6. EditorAgent: Reviews and suggests improvements
7. Coordinator: Returns final result to user
```

---

## 5. Tool Calling Flow

```
┌─────────────────────────────────────┐
│   Agent receives user query         │
│   "What's the weather in London?"   │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  Analyse available tools:           │
│  ✓ get_location_coords()            │
│  ✓ get_weather()                    │
│  ✓ format_response()                │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  Generate JSON schema for each tool │
│  {                                  │
│    "name": "get_location_coords",   │
│    "description": "...",            │
│    "parameters": {                  │
│      "location": "string"           │
│    }                                │
│  }                                  │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  Send to model with tool info       │
│  Model decides to call tools        │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  Model response includes:           │
│  Tool: get_location_coords          │
│  Args: {"location": "London"}       │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  Execute tool in Python:            │
│  get_location_coords(               │
│    ctx=RunContext,                  │
│    location="London"                │
│  )                                  │
│  Returns: lat=51.5074, lng=-0.1278  │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  Send tool result back to model:    │
│  "Tool result: lat=51.5074, ..."    │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  Model may call more tools or       │
│  generate final response            │
│                                     │
│  "The weather in London is 15°C..." │
└─────────────────────────────────────┘
```

---

## 6. Streaming Architecture

```
CLIENT (Browser/App)
    │
    │ POST /api/chat?message=...
    │
    ▼
┌──────────────────────────┐
│   FastAPI Endpoint       │
│   /api/chat/stream       │
└─────────┬────────────────┘
          │
          ▼
┌──────────────────────────────────────┐
│  agent.run_stream(message)           │
│  Opens connection to LLM             │
└──────────┬───────────────────────────┘
           │
           ▼
┌────────────────────────────────────────────┐
│  LLM streams response token by token       │
│  Token 1: "The"     ────────┐             │
│  Token 2: " capital"  ──────┤─────┐       │
│  Token 3: " of"       ──────┤─────┤──┐    │
│  Token 4: " France"   ──────┤─────┤──┤──┐ │
│  Token 5: " is"       ──────┤─────┤──┤──┤─┴─────┐
│  Token 6: " Paris"    ──────┤─────┤──┤──┤───────┐
└────────────────────────────┼───┼──┼──┼──┤───────┤
                             │   │  │  │  │       │
                             ▼   ▼  ▼  ▼  ▼       ▼
                    ┌─────────────────────────────┐
                    │  Stream to Client (SSE)     │
                    │  data: "The"                │
                    │  data: " capital"           │
                    │  data: " of"                │
                    │  data: " France"            │
                    │  data: " is"                │
                    │  data: " Paris"             │
                    └─────────────────────────────┘
                             │
                             ▼
                    CLIENT RECEIVES EVENTS
                    JavaScript updates DOM:
                    element.textContent += token
                    
                    Real-time display:
                    "The capital of France is Paris"
                    (appears as each token arrives)
```

---

## 7. Production Deployment Stack

```
┌─────────────────────────────────────────────────────────┐
│                  FRONTEND (User-facing)                 │
│  • Web browser (React, Vue, etc.)                       │
│  • Mobile app                                           │
│  • Desktop application                                  │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP/WebSocket
                       ▼
┌─────────────────────────────────────────────────────────┐
│           API GATEWAY / LOAD BALANCER                   │
│  • nginx / HAProxy                                      │
│  • Rate limiting                                        │
│  • Request routing                                      │
│  • SSL/TLS termination                                  │
└──────────────────────┬──────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ▼              ▼              ▼
   ┌─────────┐  ┌─────────┐  ┌─────────┐
   │ Agent   │  │ Agent   │  │ Agent   │
   │ Pod 1   │  │ Pod 2   │  │ Pod 3   │
   │ (K8s)   │  │ (K8s)   │  │ (K8s)   │
   └────┬────┘  └────┬────┘  └────┬────┘
        │            │            │
        └────────────┼────────────┘
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
   ┌─────────────────┐  ┌──────────────────┐
   │  PostgreSQL DB  │  │  Redis Cache     │
   │  • Messages     │  │  • Query cache   │
   │  • User data    │  │  • Sessions      │
   │  • Analytics    │  │  • Queues        │
   └─────────────────┘  └──────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│            EXTERNAL SERVICES                            │
│  • OpenAI / Anthropic / Google APIs                     │
│  • Third-party integrations                             │
│  • Payment processors                                   │
└─────────────────────────────────────────────────────────┘
```

---

## 8. Error Recovery Flow

```
                    START: agent.run()
                           │
                           ▼
                    ┌─────────────────┐
    Attempt 1 ─────→│  Call LLM Model │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
                Success            Error
                    │                 │
                    │                 ▼
                    │          ┌──────────────┐
                    │          │ Categorize   │
                    │          │ Error        │
                    │          └──────┬───────┘
                    │                 │
                    │         ┌───────┴──────────┐
                    │         │                  │
                    │      Retryable      Non-retryable
                    │         │                  │
                    │         ▼                  ▼
                    │    ┌──────────┐      ┌────────────┐
                    │    │ Backoff  │      │ Throw      │
                    │    │ Wait     │      │ Error      │
                    │    └────┬─────┘      │ to User    │
                    │         │            └────────────┘
                    │    Attempt 2
                    │         │
                    │    (repeat...)
                    │
                    ▼
            ┌─────────────────┐
            │ Check Output    │
            │ Validation      │
            └────────┬────────┘
                     │
            ┌────────┴────────┐
            │                 │
          Valid           Invalid
            │                 │
            │                 ▼
            │          ┌──────────────────┐
            │          │ ModelRetry?      │
            │          └────┬──────┬──────┘
            │               │      │
            │            Yes │      │ No
            │               │      │
            │               │      ▼
            │               │   ┌────────────┐
            │               │   │ Throw      │
            │               │   │ Validation │
            │               │   │ Error      │
            │               │   └────────────┘
            │               │
            │               ▼
            │          Attempt N+1
            │
            ▼
        RETURN RESULT
```

---

## 9. Context Engineering Flow

```
        ┌──────────────────────────────┐
        │  System Prompt (Static)      │
        │  "You are a Python expert"   │
        └──────────────┬───────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Dynamic System Prompt Fn        │
        │  @agent.system_prompt            │
        │  async def get_prompt(ctx):      │
        │    return f"Time: {time.now()}"  │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Instructions                    │
        │  "Be concise. Provide examples"  │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Few-Shot Examples (Optional)    │
        │  Q: "What is X?"                 │
        │  A: "X is..."                    │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  Message History                 │
        │  [Previous Q&A exchanges]        │
        │  Provides context for response   │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  User Query                      │
        │  "How do I implement type       │
        │   safety in Python?"             │
        └──────────────┬───────────────────┘
                       │
                       ▼
        ┌──────────────────────────────────┐
        │  FINAL CONTEXT SENT TO MODEL     │
        │  [All above combined]            │
        │  [With available tools]          │
        │  [With output schema]            │
        └──────────────┬───────────────────┘
                       │
                       ▼
                    LLM Response
```

---

## 10. Memory Persistence Pattern

```
┌─────────────────────────────┐
│  Application Starts         │
│  Initialises Agent          │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────┐
│  Conversation 1: User Query                     │
│  "What is my favourite language?"               │
│                                                 │
│  Agent: Checks memory store...                  │
│  ❌ Not found. Asks: "I don't know yet"         │
│                                                 │
│  User: "My favourite is Python"                │
│  Agent: Stores in memory cache                  │
│  ✓ stored: favourite_language = "Python"       │
└────────────┬────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────┐
│  Time Passes (minutes/hours/days)               │
│  Session ends and restarts                      │
│  Application reloads                            │
└────────────┬────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────┐
│  Conversation 2: User Query (New Session)       │
│  "Remind me of my favourite language"          │
│                                                 │
│  Agent: Queries persistent memory...            │
│  ✓ Found: favourite_language = "Python"        │
│                                                 │
│  Agent: "Your favourite language is Python"    │
│                                                 │
│  Storage: PostgreSQL with TTL/expiry           │
│  Key: user_123_favourite_language               │
│  Value: "Python"                                │
│  Expires: 2025-03-25 (or never)                │
└─────────────────────────────────────────────────┘
```

---

## 11. Testing Architecture

```
┌──────────────────────────────────┐
│  Agent Code                      │
│  @agent.tool                     │
│  async def get_data():           │
└──────────────┬───────────────────┘
               │
   ┌───────────┴────────────┐
   │                        │
   ▼                        ▼
UNIT TEST            INTEGRATION TEST
│                        │
├─ Mock Model           ├─ Real Model (or Sandbox)
│  TestModel()          │  gpt-4o-mini
│                       │
├─ Mock Dependencies    ├─ Real Services
│  db_connection        │  PostgreSQL, Redis
│  api_client           │
│                       │
├─ Test tool           ├─ Test full workflow
│  independently        │  end-to-end
│                       │
└─ Fast (~100ms)       └─ Slow (~5s+)
  ✅ Unit tests         ✅ Integration tests
```

---

(Diagrams continue with more architectural patterns, message flows, and system designs)


