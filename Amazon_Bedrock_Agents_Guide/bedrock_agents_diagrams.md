# Amazon Bedrock Agents: Architecture Diagrams and Visual References

Comprehensive visual representations of Bedrock Agents architecture, data flows, and deployment patterns.

---

## 1. Core Agent Architecture

### Simple Agent Request Flow

```
User Input
    ↓
┌─────────────────────────────┐
│  Bedrock Agent Runtime      │
│  - Parse Input              │
│  - Load Context             │
│  - Initialize State         │
└────────────┬────────────────┘
             ↓
┌─────────────────────────────┐
│  Foundation Model           │
│  - Claude 3 Sonnet         │
│  - Reasoning Phase          │
│  - Action Selection         │
└────────────┬────────────────┘
             ↓
       ┌─────┴─────┐
       │           │
       ↓           ↓
  ┌────────┐  ┌────────┐
  │ Action │  │Knowledge
  │ Groups │  │ Bases  │
  └────┬───┘  └────┬───┘
       │           │
       └─────┬─────┘
             ↓
┌─────────────────────────────┐
│  Response Generation        │
│  - Format Output            │
│  - Apply Guardrails         │
│  - Generate Citations       │
└────────────┬────────────────┘
             ↓
        User Response
```

### Multi-Layer Bedrock Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Presentation Layer                         │
│  Web UI | Mobile App | Chat Interface | API Gateway           │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│            Agent Orchestration & Routing Layer               │
│  - Request Reception        - Session Management             │
│  - Agent Selection          - State Tracking                 │
│  - Multi-Agent Coordination - Error Handling                 │
└──────────────────────┬───────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
        ↓              ↓              ↓
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│   Supervisor │ │ Specialist   │ │ Specialist   │
│    Agent     │ │   Agent 1    │ │   Agent 2    │
│              │ │ (Support)    │ │ (Billing)    │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
       ┌────────────────┼────────────────┐
       │                │                │
       ↓                ↓                ↓
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Foundation  │ │  Action      │ │  Knowledge   │
│  Models      │ │  Groups      │ │  Bases       │
│              │ │              │ │              │
│ Claude 3     │ │ Lambda Funcs │ │ OpenSearch   │
│ Llama 2      │ │ APIs         │ │ Pinecone     │
│ Titan        │ │ Step Funcs   │ │ Kendra       │
└──────┬───────┘ └──────┬───────┘ └──────┬───────┘
       │                │                │
       └────────────────┼────────────────┘
                        │
       ┌────────────────┼────────────────┐
       │                │                │
       ↓                ↓                ↓
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Guardrails  │ │  Prompt      │ │  Memory &    │
│  - Content   │ │  Flows       │ │  State       │
│  - PII       │ │              │ │              │
│  - Topics    │ │              │ │  DynamoDB    │
└──────────────┘ └──────────────┘ └──────────────┘
```

---

## 2. Multi-Agent Collaboration (MAS) Architecture

### Supervisor-Specialist Pattern

```
                    ┌────────────────────┐
                    │  User Query        │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Supervisor Agent  │
                    │  - Intent Analysis │
                    │  - Task Planning   │
                    │  - Routing Logic   │
                    └─────────┬──────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ↓                     ↓                     ↓
    ┌────────────┐    ┌─────────────┐    ┌──────────────┐
    │  Support   │    │   Billing   │    │   Technical  │
    │  Agent     │    │   Agent     │    │   Agent      │
    │            │    │             │    │              │
    │ • Tickets  │    │ • Invoices  │    │ • Bugs       │
    │ • Issues   │    │ • Payments  │    │ • Features   │
    │ • FAQs     │    │ • Refunds   │    │ • API Status │
    └────────────┘    └─────────────┘    └──────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Response          │
                    │  Consolidation     │
                    │  - Merge Responses │
                    │  - Format Output   │
                    │  - Citations       │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  User Response     │
                    └────────────────────┘
```

### Task Decomposition Flow

```
User Task: "I want to know my account balance and update my contact information"

          ┌──────────────────────────────────────┐
          │  Supervisor Agent                    │
          │  Analyzes: 2 distinct tasks          │
          └──────────────────────┬───────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
          Task 1: Get Account Balance    Task 2: Update Contact Info
                    │                         │
                    ↓                         ↓
        ┌──────────────────────┐   ┌──────────────────────┐
        │  Billing Agent       │   │  Support Agent       │
        │  Action: GetBalance  │   │  Action: UpdateInfo  │
        └──────────┬───────────┘   └──────────┬───────────┘
                   │                          │
                   ↓                          ↓
        ┌──────────────────────┐   ┌──────────────────────┐
        │ DynamoDB Query       │   │ Lambda Function      │
        │ Result: $5,000.00    │   │ Result: Updated ✓    │
        └──────────┬───────────┘   └──────────┬───────────┘
                   │                          │
                   └────────────┬─────────────┘
                                │
                    ┌───────────▼────────────┐
                    │  Consolidate Results   │
                    │  1. Account Balance    │
                    │     $5,000.00          │
                    │  2. Contact Info      │
                    │     Updated ✓          │
                    └───────────┬────────────┘
                                │
                         Final Response
```

---

## 3. Action Group Execution Patterns

### Lambda Action Group Flow

```
┌──────────────────────────────────────────────────────────────┐
│                    Agent Processing                          │
│  Model Decides: "I need to get customer information"         │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│            Action Group Selection                            │
│  Action Group: "GetCustomerData"                             │
│  Function: "GetCustomer"                                     │
│  Parameters: {customerId: "CUST-12345"}                      │
└──────────────────────┬───────────────────────────────────────┘
                       │
┌──────────────────────▼───────────────────────────────────────┐
│       Lambda Function Invocation                             │
│  Function: bedrock-get-customer                              │
│  Runtime: Python 3.11                                        │
└──────────────────────┬───────────────────────────────────────┘
                       │
       ┌───────────────┼───────────────┐
       │               │               │
       ↓               ↓               ↓
   ┌────────┐   ┌──────────┐   ┌──────────┐
   │Parse   │   │Query     │   │Format    │
   │Params  │   │Database  │   │Response  │
   └────┬───┘   └────┬─────┘   └────┬─────┘
        │            │              │
        └────────────┼──────────────┘
                     │
         ┌───────────▼────────────┐
         │ Return to Agent        │
         │ {                      │
         │   customerId: "C12345" │
         │   name: "John Doe"     │
         │   email: "john@ex.com" │
         │ }                      │
         └───────────┬────────────┘
                     │
        Agent Continues Reasoning
```

### API Action Group Flow

```
Agent Decision: "I need to check external system status"

       ┌──────────────────────────────────────┐
       │  Select API Action Group             │
       │  API Endpoint: https://api.ext.com   │
       └──────────────────┬───────────────────┘
                          │
       ┌──────────────────▼───────────────────┐
       │  Construct HTTP Request              │
       │  Method: GET                         │
       │  Path: /status/system                │
       │  Headers: {Auth: Bearer TOKEN}       │
       └──────────────────┬───────────────────┘
                          │
       ┌──────────────────▼───────────────────┐
       │  Execute HTTP Call                   │
       │  With Retry Logic                    │
       │  Timeout: 30 seconds                 │
       └──────────────────┬───────────────────┘
                          │
         ┌────────────────┴────────────────┐
         │                                 │
    Success (200)                     Error/Retry
         │                                 │
         ↓                                 ↓
   ┌──────────────┐            ┌──────────────────┐
   │Parse Response│            │Retry with Backoff│
   │  {           │            │Max 3 attempts    │
   │  status:     │            └──────────┬───────┘
   │  "HEALTHY"   │                       │
   │  }           │            ┌──────────▼───────┐
   └──────┬───────┘            │Success/Failure   │
          │                    └──────────┬───────┘
          └────────────┬─────────────────┘
                       │
              Return to Agent
```

---

## 4. Knowledge Base Integration Pattern

### RAG (Retrieval-Augmented Generation) Flow

```
User: "What is our refund policy for damaged items?"

       ┌────────────────────────────┐
       │  Agent Input Processing    │
       │  - Analyze Question        │
       │  - Extract Intent          │
       │  - Generate Query Vector   │
       └────────────────┬───────────┘
                        │
       ┌────────────────▼───────────┐
       │  Query Knowledge Base      │
       │  - Embedding Generation    │
       │  - Vector Search           │
       │  - Semantic Matching       │
       └────────────────┬───────────┘
                        │
       ┌────────────────▼──────────────────┐
       │  Retrieved Documents               │
       │  - Return Policy (Section 3.2)    │
       │  - Damage Claims Process (5.1)    │
       │  - Refund Timeline (4.3)          │
       │  Relevance Score: 0.92, 0.87, 0.85│
       └────────────────┬──────────────────┘
                        │
       ┌────────────────▼──────────────────┐
       │  Augmented Context for Model      │
       │  User Query + Retrieved Context   │
       └────────────────┬──────────────────┘
                        │
       ┌────────────────▼──────────────────┐
       │  Model Generation                 │
       │  - Synthesise Answer             │
       │  - Reference Sources             │
       │  - Generate Citations            │
       └────────────────┬──────────────────┘
                        │
       Final Response with:
       - Answer
       - Source References
       - Confidence Score
```

### Knowledge Base Vector Storage

```
┌─────────────────────────────────────────────────────────┐
│              Data Ingestion Pipeline                    │
└──────────────────────┬──────────────────────────────────┘
                       │
       ┌───────────────┴───────────────┐
       │                               │
       ↓                               ↓
┌──────────────┐            ┌─────────────────┐
│  Document    │            │  File Type      │
│  Sources     │            │  - PDF          │
│              │            │  - DOCX         │
│ - S3 Bucket  │            │  - TXT          │
│ - RDS        │            │  - JSON         │
│ - Web URLs   │            │  - Websites     │
└──────┬───────┘            └────────┬────────┘
       │                            │
       └────────────┬───────────────┘
                    │
       ┌────────────▼──────────────┐
       │  Document Processing     │
       │  - Text Extraction       │
       │  - Preprocessing         │
       │  - Cleaning              │
       └────────────┬──────────────┘
                    │
       ┌────────────▼──────────────────────┐
       │  Chunking Strategy                │
       │  - Fixed Size (1000 tokens)      │
       │  - Semantic (Sentence Boundary)  │
       │  - Hierarchical (Section-aware)  │
       └────────────┬───────────────────────┘
                    │
       ┌────────────▼──────────────┐
       │  Embedding Generation    │
       │  Model: Titan Embed      │
       │  Dimension: 1536         │
       └────────────┬──────────────┘
                    │
       ┌────────────▼──────────────┐
       │  Vector Storage          │
       │  OpenSearch Serverless   │
       │  Vector Index            │
       │  Metadata Attached       │
       └────────────┬──────────────┘
                    │
          Ready for Retrieval
```

---

## 5. Guardrails Architecture

### Multi-Layer Safety Architecture

```
┌──────────────────────────────────────────────────────────┐
│                 User Input                               │
└────────────────────┬─────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────┐
│  Layer 1: Input Guardrails                              │
│  ✓ Content Filtering (Hate, Violence, Explicit)        │
│  ✓ PII Detection (Names, Emails, Phone Numbers)        │
│  ✓ Denied Topics Detection                             │
│  ✓ Prompt Injection Prevention                         │
└────────────────────┬─────────────────────────────────────┘
                     │
         ┌───────────┴──────────┐
         │                      │
    Block (Deny)         Continue
         │                      │
         ↓                      ↓
    Return Error         Foundation Model
                         Processing
                              │
                    ┌─────────▼─────────┐
                    │ Generate Response │
                    └─────────┬─────────┘
                              │
        ┌─────────────────────▼─────────────────────┐
        │  Layer 2: Output Guardrails              │
        │  ✓ Content Validation                   │
        │  ✓ PII Redaction                        │
        │  ✓ Denied Topic Filtering               │
        │  ✓ Hallucination Detection              │
        └─────────────────────┬─────────────────────┘
                              │
              ┌───────────────┴──────────────┐
              │                              │
         Blocked/Modified              Approved
              │                              │
              ↓                              ↓
         Alert Log              Return to User
```

### Content Policy Categories

```
Bedrock Guardrails Coverage Matrix

┌─────────────────┬──────────┬──────────┬──────────┬──────────┐
│ Category        │ Input    │ Output   │ Severity │ Action   │
├─────────────────┼──────────┼──────────┼──────────┼──────────┤
│ Hate Speech     │ HIGH     │ HIGH     │ Critical │ BLOCK    │
│ Violence        │ HIGH     │ HIGH     │ Critical │ BLOCK    │
│ Sexual Content  │ HIGH     │ HIGH     │ Critical │ BLOCK    │
│ Self-Harm       │ HIGH     │ HIGH     │ Critical │ BLOCK    │
│ PII (Email)     │ MEDIUM   │ MEDIUM   │ High     │ ANON     │
│ PII (Phone)     │ MEDIUM   │ MEDIUM   │ High     │ ANON     │
│ PII (Name)      │ MEDIUM   │ MEDIUM   │ High     │ ANON     │
│ Denied Topics   │ MEDIUM   │ MEDIUM   │ Medium   │ BLOCK    │
│ Custom Keywords │ LOW      │ LOW      │ Custom   │ CUSTOM   │
└─────────────────┴──────────┴──────────┴──────────┴──────────┘
```

---

## 6. Deployment Architecture Patterns

### Single-Region Agent Deployment

```
┌────────────────────────────────────────────────────────┐
│                    us-east-1 Region                    │
├────────────────────────────────────────────────────────┤
│                                                        │
│  ┌──────────────────────────────────────┐             │
│  │  Amazon Bedrock Agent                │             │
│  │  - Agent ID: ag-12345                │             │
│  │  - Foundation Model: Claude 3        │             │
│  │  - Status: ACTIVE                    │             │
│  └──────────────────────────────────────┘             │
│                                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │ Action   │  │Knowledge │  │Guardrails│            │
│  │ Groups   │  │ Bases    │  │          │            │
│  │          │  │          │  │          │            │
│  │ Lambda   │  │OpenSearch│  │Content   │            │
│  │ API      │  │Serverless│  │Filtering │            │
│  └──────────┘  └──────────┘  └──────────┘            │
│                                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │CloudWatch│  │CloudTrail│  │X-Ray     │            │
│  │Logs      │  │Events    │  │Tracing   │            │
│  └──────────┘  └──────────┘  └──────────┘            │
│                                                        │
│  ┌────────────────────────────────────┐              │
│  │ IAM Roles & Policies                │              │
│  │ - Agent Execution Role              │              │
│  │ - Service Roles                     │              │
│  └────────────────────────────────────┘              │
│                                                        │
└────────────────────────────────────────────────────────┘
```

### Multi-Region Agent Deployment

```
┌─────────────────────────────────────────────────────────────┐
│                 Global Agent Architecture                   │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
    ┌──────────▼──────────┐      ┌───────────▼────────────┐
    │   us-east-1 Region  │      │  eu-west-1 Region     │
    │                     │      │                        │
    │ ┌────────────────┐  │      │ ┌────────────────┐    │
    │ │Primary Agent   │  │      │ │Secondary Agent │    │
    │ │(Active)        │  │      │ │(Standby)       │    │
    │ │ID: ag-primary  │  │      │ │ID: ag-secondary    │
    │ └────────────────┘  │      │ └────────────────┘    │
    │                     │      │                        │
    │ ┌────────────────┐  │      │ ┌────────────────┐    │
    │ │Knowledge Base  │  │      │ │Knowledge Base  │    │
    │ │(Replicated)    │  │      │ │(Replicated)    │    │
    │ └────────────────┘  │      │ └────────────────┘    │
    └──────────┬──────────┘      └──────────┬────────────┘
               │                            │
               └────────────┬───────────────┘
                            │
                ┌───────────▼──────────┐
                │  Route 53 Failover   │
                │  - Health Checks     │
                │  - Automatic Routing │
                │  - Latency Based     │
                └──────────────────────┘
```

---

## 7. Prompt Flow Architecture

### Complex Conditional Flow

```
Start User Input
      │
      ↓
┌──────────────────────────────┐
│ Classify Intent              │
│ - Support Request            │
│ - Sales Inquiry              │
│ - Billing Question           │
│ - Technical Issue            │
└────────┬─────────────────────┘
         │
    ┌────┴──────────────────────────────────────┐
    │                                           │
    ↓                                           ↓
┌──────────────────┐                ┌───────────────────┐
│ Is Urgent?       │                │ Authentication    │
│ Check Priority   │                │ Required?         │
└─────┬────────────┘                └────────┬──────────┘
      │                                      │
  ┌───┴────┐                          ┌──────┴──────┐
  │         │                          │             │
  ↓         ↓                          ↓             ↓
HIGH      NORMAL                    YES           NO
  │         │                          │             │
  ↓         ↓                          ↓             ↓
Escalate  Route         ┌────────────────────┐  Direct
Priority  Standard      │ Verify Identity    │  Routing
Queue     Queue         │ - Check User DB    │
                        │ - Verify Token     │
                        └────────┬───────────┘
                                 │
                        ┌────────▼──────────┐
                        │ Assign Agent      │
                        │ Based on Capacity │
                        └────────┬──────────┘
                                 │
                        ┌────────▼──────────┐
                        │ Generate Response │
                        │ - Process Request │
                        │ - Format Output   │
                        └────────┬──────────┘
                                 │
                              Response
```

---

## 8. State Management and Memory Architecture

### Session State Management

```
┌─────────────────────────────────────────────────────┐
│              Session Lifecycle                      │
└──────────────────┬──────────────────────────────────┘
                   │
         ┌─────────▼─────────┐
         │ Session Created   │
         │ SessionID: s-xxxx │
         └─────────┬─────────┘
                   │
    ┌──────────────┴──────────────┐
    │ Store in DynamoDB           │
    │ TTL: 24 hours               │
    │ Attributes: {               │
    │   sessionId                 │
    │   userId                    │
    │   conversationHistory       │
    │   contextVariables          │
    │   userPreferences           │
    │   tempData                  │
    │ }                           │
    └──────────────┬──────────────┘
                   │
         ┌─────────▼──────────────┐
         │ Agent Processing       │
         │ - Read Context         │
         │ - Process Request      │
         │ - Update State         │
         └─────────┬──────────────┘
                   │
    ┌──────────────┴──────────────┐
    │ Persist Changes             │
    │ - Append to History         │
    │ - Update Variables          │
    │ - Refresh TTL               │
    └──────────────┬──────────────┘
                   │
         ┌─────────▼──────────────┐
         │ Session Expiration     │
         │ - Auto-cleanup         │
         │ - Archive History      │
         └───────────────────────┘
```

### Memory Architecture Layers

```
┌────────────────────────────────────────────────────────┐
│            Multi-Layer Memory System                   │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Layer 1: Short-Term Context                         │
│  ┌──────────────────────────────────────────┐        │
│  │ Current Request Context                  │        │
│  │ - User Input                             │        │
│  │ - Immediate Conversation History (5-10   │        │
│  │   exchanges)                             │        │
│  │ - Current State Variables                │        │
│  │ Storage: Session Memory (In-Memory)      │        │
│  └──────────────────────────────────────────┘        │
│                                                        │
│  Layer 2: Session Memory                             │
│  ┌──────────────────────────────────────────┐        │
│  │ Full Conversation in Current Session     │        │
│  │ - All messages from session start        │        │
│  │ - Action results                         │        │
│  │ - Intermediate states                    │        │
│  │ Storage: DynamoDB (24hr TTL)             │        │
│  └──────────────────────────────────────────┘        │
│                                                        │
│  Layer 3: Long-Term User Memory                      │
│  ┌──────────────────────────────────────────┐        │
│  │ User Profile & Preferences               │        │
│  │ - Past conversations (archived)          │        │
│  │ - User preferences                       │        │
│  │ - Account information                    │        │
│  │ Storage: RDS/DynamoDB (Permanent)        │        │
│  └──────────────────────────────────────────┘        │
│                                                        │
│  Layer 4: Knowledge Base Memory                      │
│  ┌──────────────────────────────────────────┐        │
│  │ Enterprise Knowledge & Context           │        │
│  │ - Product documentation                  │        │
│  │ - Historical patterns                    │        │
│  │ - Best practices                         │        │
│  │ Storage: OpenSearch, Vector DBs          │        │
│  └──────────────────────────────────────────┘        │
│                                                        │
└────────────────────────────────────────────────────────┘
```

---

## 9. Integration Points Architecture

### Comprehensive Integration Diagram

```
┌────────────────────────────────────────────────────────────────┐
│                      Bedrock Agent Hub                         │
│                     (Central Orchestrator)                      │
└──────────┬────────────────────────────────────────┬─────────────┘
           │                                        │
    ┌──────▼────────┐              ┌───────────────▼────────┐
    │ AWS Services  │              │  External Integrations │
    │               │              │                        │
    ├─────────────┐ │         ┌────┴──────────────────┐   │
    │ Lambda      │─┼─────────│ API Gateway           │   │
    │             │ │         │ (Custom APIs)         │   │
    ├─────────────┤ │         └───────────────────────┘   │
    │ S3          │ │                                      │
    │             │ │         ┌───────────────────────┐   │
    ├─────────────┤ │────────│ Salesforce CRM        │   │
    │ DynamoDB    │ │         │ (Customer Data)       │   │
    │             │ │         └───────────────────────┘   │
    ├─────────────┤ │                                      │
    │ RDS         │ │         ┌───────────────────────┐   │
    │             │ │────────│ Slack Integration     │   │
    ├─────────────┤ │         │ (Notifications)       │   │
    │ Step Funcs  │ │         └───────────────────────┘   │
    │             │ │                                      │
    ├─────────────┤ │         ┌───────────────────────┐   │
    │ SNS/SQS     │ │────────│ ServiceNow ITSM       │   │
    │             │ │         │ (Ticketing)           │   │
    ├─────────────┤ │         └───────────────────────┘   │
    │ EventBridge │ │                                      │
    │             │ │         ┌───────────────────────┐   │
    ├─────────────┤ │────────│ Datadog Monitoring    │   │
    │ Secrets Mgr │ │         │ (Analytics)           │   │
    │             │ │         └───────────────────────┘   │
    ├─────────────┤ │                                      │
    │ CloudWatch  │ │                                      │
    │             │ │                                      │
    └─────────────┘ │                                      │
                    │                                      │
    ┌───────────────▼─────────────────────────────────────┐
    │  Knowledge Base Sources                            │
    ├────────────────────────────────────────────────────┤
    │ • S3 Documents        • Salesforce KB             │
    │ • RDS Databases       • Confluence Wiki            │
    │ • Web URLs            • SharePoint                 │
    │ • OpenSearch          • Kendra                     │
    └────────────────────────────────────────────────────┘
```

---

## 10. Monitoring and Observability Architecture

### Comprehensive Monitoring Stack

```
┌──────────────────────────────────────────────────────────┐
│              Observability Stack                         │
└─────────────────┬────────────────────────────────────────┘
                  │
        ┌─────────┼─────────┐
        │         │         │
        ↓         ↓         ↓
   ┌─────────┐ ┌────────┐ ┌────────┐
   │ Metrics │ │  Logs  │ │ Traces │
   └────┬────┘ └────┬───┘ └────┬───┘
        │           │          │
        ↓           ↓          ↓
   ┌──────────────────────────────────────┐
   │    CloudWatch Central Repository     │
   │  - Request Latency                  │
   │  - Error Rates                      │
   │  - Token Usage                      │
   │  - Agent Invocations                │
   │  - Action Group Performance         │
   │  - Knowledge Base Queries           │
   └─────────┬──────────────────────────┘
             │
    ┌────────┴────────┬────────────┬────────────┐
    │                 │            │            │
    ↓                 ↓            ↓            ↓
┌──────────┐   ┌──────────┐  ┌──────────┐  ┌──────────┐
│Dashboards│   │ Alarms   │  │  Logs    │  │Analytics │
│          │   │          │  │  Insights│  │          │
│- Real    │   │- Critical│  │- Errors  │  │- Trends  │
│  Time    │   │- Warning │  │- Warnings   │- Patterns│
│- KPIs    │   │- Info    │  │- Info    │  │- Anomaly │
└──────────┘   └──────────┘  └──────────┘  └──────────┘
```


