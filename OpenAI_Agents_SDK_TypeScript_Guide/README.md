# OpenAI Agents SDK TypeScript: Complete Developer Guide (2025 Edition)

**🎯 PRODUCTION-READY | TypeScript-First | Official Swarm Replacement**

**Version:** 0.3.2
**Status:** Production-Ready with 2025 Features
**Last Updated:** November 2025
**Language:** TypeScript 5.0+
**Framework:** OpenAI Agents SDK

---

## 🚀 Critical 2025 Updates

### ⭐ **NEW: Production-Ready & Swarm Replacement**
The OpenAI Agents SDK for TypeScript is the **official production-ready replacement** for experimental Swarm. Complete migration path available:
- **[Swarm Migration Guide →](openai_agents_sdk_typescript_swarm_migration_guide.md)**

### ⭐ **NEW: 2025 Features Guide**
Critical new features for TypeScript developers:
- **Human-in-the-Loop Approvals** (NEW 2025!)
- Built-in tracing and visualization
- Provider-agnostic support (100+ LLMs)
- Type-safe guardrails with Zod
- MCP integration
- **[Complete 2025 Features Guide →](openai_agents_sdk_typescript_2025_features.md)**

---

## 🆕 What's New in 2025?

| Feature | Description | Guide |
|---------|-------------|-------|
| **Human-in-Loop** | Approval workflows for critical actions | [2025 Features](openai_agents_sdk_typescript_2025_features.md) |
| **Built-in Tracing** | Comprehensive observability & visualization | [2025 Features](openai_agents_sdk_typescript_2025_features.md) |
| **100+ LLM Providers** | Claude, Gemini, Llama, Mistral & more | [2025 Features](openai_agents_sdk_typescript_2025_features.md) |
| **Type-Safe Guardrails** | Input/output validation with Zod | [2025 Features](openai_agents_sdk_typescript_2025_features.md) |
| **MCP Integration** | Filesystem, git, HTTP support | [Comprehensive Guide](openai_agents_sdk_typescript_comprehensive_guide.md) |
| **Swarm Migration** | Complete migration from Swarm | [Migration Guide](openai_agents_sdk_typescript_swarm_migration_guide.md) |

---

## 📚 Documentation Structure

### 🆕 **0. Swarm Migration Guide** ⭐
**openai_agents_sdk_typescript_swarm_migration_guide.md**
**CRITICAL FOR SWARM USERS**: Complete TypeScript migration from experimental Swarm to production Agents SDK:
- Why migrate? (TypeScript-first design, production stability, active maintenance)
- Side-by-side TypeScript code comparisons
- Type safety improvements over Swarm
- Breaking changes and solutions
- Testing strategies with Jest
- Production deployment patterns

### 🆕 **0.5. 2025 Features Guide** ⭐
**openai_agents_sdk_typescript_2025_features.md**
Latest TypeScript-specific features and improvements:
- **Human-in-the-Loop Approvals** (NEW 2025!) - Approval workflows for critical operations
- Built-in tracing and observability
- Type-safe guardrails with Zod schemas
- Handoffs with full type inference
- MCP integration (filesystem, git, HTTP)
- Provider-agnostic support (100+ LLMs)
- Production features (error handling, cost tracking)

### 1. **Comprehensive Guide**
**openai_agents_sdk_typescript_comprehensive_guide.md**
**Beginner → Expert | ~80+ pages | Complete Reference**

The complete technical reference covering all aspects of the OpenAI Agents SDK with TypeScript:

#### Core Sections:
- **Core Fundamentals** (Installation, TypeScript setup, design philosophy, core primitives)
- **Simple Agents** (Creating agents, configuration, execution patterns)
- **Multi-Agent Systems** (Handoffs, delegation, coordination, workflows)
- **Tools Integration** (Function tools, custom tools, OAI hosted tools, error handling)
- **Structured Output** (Schema definition, validation, JSON mode, type enforcement)
- **Model Context Protocol (MCP)** (Building with MCP, tool discovery, integration)
- **Agentic Patterns** (Deterministic workflows, routing, multi-step reasoning)
- **Guardrails** (Input/output validation, safety checks, content filtering)
- **Memory Systems** (Session management, storage strategies, state persistence)
- **Context Engineering** (Prompt templates, dynamic context, file handling)
- **Responses API Integration** (Unified patterns, type-safe handling)
- **Tracing & Observability** (Built-in tracing, debugging, performance profiling)
- **Real-Time Experiences** (Web applications, voice agents, streaming)
- **Model Providers** (OpenAI, Anthropic, Google, Mistral integration)
- **Testing** (Unit testing, integration testing, mocking)
- **Deployment Patterns** (Docker, Kubernetes, Express.js, Next.js)
- **TypeScript Patterns** (Generic types, interfaces, type guards)
- **Advanced Topics** (Custom implementations, enterprise patterns, security)

**Key Features:**
✓ 50+ complete, production-ready TypeScript code examples
✓ Full type annotations and interfaces
✓ Real-world use cases and scenarios
✓ Progressive complexity from simple to advanced
✓ Best practices throughout

---

### 2. **Production Guide**
**openai_agents_sdk_typescript_production_guide.md**
**Enterprise Focus | Reliability & Scale**

Enterprise-grade patterns and best practices for production deployments:

#### Core Sections:
- **Deployment Architecture**
  - Docker containerisation with multi-stage builds
  - Kubernetes manifests with auto-scaling
  - Express.js API integration
  - Health checks and readiness probes

- **Error Handling & Resilience**
  - Comprehensive error classification
  - Retry strategies with exponential backoff
  - Circuit breaker patterns
  - Timeout management

- **Performance Optimisation**
  - Caching strategies (LRU, LFU, FIFO)
  - Connection pooling
  - Request batching
  - Token usage optimisation

- **Security Best Practices**
  - API key management and secret rotation
  - Input validation and sanitisation
  - XSS prevention
  - Rate limiting and DDoS protection

- **Monitoring & Observability**
  - Distributed tracing with Jaeger
  - Metrics collection with Prometheus
  - Logging strategies
  - Performance profiling

- **Scaling Strategies**
  - Load balancing algorithms
  - Horizontal scaling with message queues
  - Database connection pooling
  - State management at scale

- **Multi-Tenancy**
  - Tenant isolation patterns
  - Quota management
  - Resource allocation
  - Security boundaries

- **Testing Strategies**
  - Unit testing with Jest
  - Integration testing
  - Mocking LLM responses
  - Test coverage strategies

- **CI/CD Integration**
  - Automated testing pipelines
  - Continuous deployment
  - Version management
  - Rollback strategies

**Key Focus:**
✓ Production-ready code patterns
✓ Enterprise scalability
✓ Operational excellence
✓ Security hardening
✓ Cost optimisation

---

### 3. **Practical Recipes**
**openai_agents_sdk_typescript_recipes.md**
**Copy-Paste Ready | 18+ Real-World Examples**

Battle-tested implementations for common scenarios:

#### Recipe Categories:

**Basic Agent Recipes (3 recipes)**
- Simple Q&A Agent
- Translation Agent with Multiple Languages
- Content Classification Agent

**Multi-Agent Workflows (3 recipes)**
- Research & Summary Workflow
- Customer Support Routing
- Parallel Processing Pipeline

**Data Processing (2 recipes)**
- Data Validation Agent
- CSV to Structured Format

**Customer Service (2 recipes)**
- FAQ System with Agent
- Appointment Scheduling Assistant

**Content Generation (2 recipes)**
- Blog Post Generator
- Social Media Content Creator

**Research & Analysis (2 recipes)**
- Market Analysis Agent
- Code Review Agent

**Integration Patterns (2 recipes)**
- Webhook Handler with Agent
- Scheduled Agent Tasks

**Advanced Orchestration (2 recipes)**
- Complex Workflow with Conditions
- Feedback Loop with Refinement

**Features:**
✓ 18+ complete, runnable examples
✓ Immediately applicable patterns
✓ Minimal setup required
✓ Real-world use cases
✓ Well-commented code

---

### 4. **Architecture Diagrams**
**openai_agents_sdk_typescript_diagrams.md**
**Visual Reference | ASCII Diagrams & Flowcharts**

Visual representations of architecture and patterns:

#### Diagrams Included:
- **Core Architecture**: Component overview and relationships
- **Agent Execution Flow**: Step-by-step execution pipeline
- **Multi-Agent Handoff Pattern**: Routing and delegation
- **Session & Memory Management**: State persistence lifecycle
- **Tool Integration Pattern**: Tool invocation flow
- **Error Handling & Resilience**: Recovery mechanisms
- **Structured Output Processing**: Validation pipeline
- **Deployment Architecture**: Production infrastructure
- **Type Safety Flow**: TypeScript type checking layers
- **Component Interaction**: System integration diagram

**Benefits:**
✓ Quick visual understanding
✓ Architecture reference
✓ Process documentation
✓ Educational value
✓ Communication tool

---

## 🚀 Quick Start (2025 Edition)

### Installation

```bash
npm install @openai/agents zod dotenv
npm install --save-dev typescript @types/node
```

### Create Your First Agent with 2025 Features

```typescript
import { Agent, run, trace, tool } from '@openai/agents';
import { z } from 'zod';
import dotenv from 'dotenv';

dotenv.config();

// Type-safe tool with Zod validation
const weatherTool = tool({
  name: 'get_weather',
  description: 'Get weather information',
  parameters: z.object({
    location: z.string(),
    units: z.enum(['celsius', 'fahrenheit']).default('celsius'),
  }),
  execute: async ({ location, units }) => ({
    location,
    temperature: 72,
    units,
    condition: 'Sunny',
  }),
});

// Agent with built-in tracing
const agent = new Agent({
  name: 'Weather Assistant',
  instructions: 'Provide weather information using available tools',
  tools: [weatherTool],
});

async function main() {
  // Execute with tracing
  await trace('Weather Query', async () => {
    const result = await run(agent, 'What is the weather in London?');
    console.log(result.finalOutput);
  });
}

main().catch(console.error);
```

### Environment Setup

```bash
# .env
OPENAI_API_KEY=your_api_key_here
```

---

## 💡 2025 Key Features Quickstart

### 1. Human-in-the-Loop Approvals (NEW!)

```typescript
import { Agent, run, tool } from '@openai/agents';
import { z } from 'zod';

const deleteTool = tool({
  name: 'delete_user',
  description: 'Delete user account',
  parameters: z.object({ userId: z.string().uuid() }),
  requiresApproval: true, // NEW: Require human approval
  execute: async ({ userId }) => ({ status: 'deleted', userId }),
});

const agent = new Agent({
  name: 'Admin Agent',
  tools: [deleteTool],
});

const result = await run(agent, 'Delete user account', {
  approvalHandler: async (request) => ({
    approved: await getUserConfirmation(request),
    reason: 'Approved by admin',
  }),
});
```

### 2. Built-in Tracing

```typescript
import { trace } from '@openai/agents';

await trace('Customer Support Workflow', async () => {
  const step1 = await run(triageAgent, query1);
  const step2 = await run(specialistAgent, query2);
  return { step1, step2 };
}, {
  metadata: {
    environment: 'production',
    customer_id: 'cust_123',
  },
});

// View detailed traces at: https://platform.openai.com/traces
```

### 3. Provider-Agnostic (100+ LLMs)

```typescript
// Use Claude
const claudeAgent = new Agent({
  name: 'Claude',
  model: 'litellm/anthropic/claude-3-5-sonnet-20240620',
});

// Use Gemini
const geminiAgent = new Agent({
  name: 'Gemini',
  model: 'litellm/gemini/gemini-2.0-flash',
});

// Use Llama
const llamaAgent = new Agent({
  name: 'Llama',
  model: 'litellm/replicate/meta-llama/llama-2-70b-chat',
});
```

### 4. Type-Safe Guardrails

```typescript
import { inputGuardrail } from '@openai/agents';
import { z } from 'zod';

const emailGuardrail = inputGuardrail({
  name: 'validate_email',
  schema: z.object({ email: z.string().email() }),
  onViolation: (input) => {
    throw new Error(`Invalid email: ${input.email}`);
  },
});

const agent = new Agent({
  name: 'Secure Agent',
  inputGuardrails: [emailGuardrail],
});
```

### 5. MCP Integration

```typescript
import { createMCPServer } from '@openai/agents/mcp';

const filesystemServer = await createMCPServer({
  name: 'filesystem',
  transport: 'stdio',
  command: 'npx',
  args: ['-y', '@modelcontextprotocol/server-filesystem', './project'],
});

const agent = new Agent({
  name: 'File Assistant',
  mcpServers: [filesystemServer],
});
```

---

## 📋 Learning Path (2025 Edition)

### Beginner Level (Day 1-3)
1. **NEW**: Read Swarm Migration Guide (if coming from Swarm)
2. **NEW**: Review 2025 Features Guide
3. Read: Core Fundamentals section of Comprehensive Guide
4. Run: Basic Agent Recipes
5. Implement: Simple Q&A Agent with tracing

### Intermediate Level (Week 1)
1. **NEW**: Implement human-in-the-loop approvals
2. **NEW**: Add guardrails to agents
3. Study: Multi-Agent Systems
4. Run: Multi-Agent Workflow recipes
5. Build: Customer support system with handoffs

### Advanced Level (Week 2-3)
1. **NEW**: Integrate MCP servers
2. **NEW**: Use multiple LLM providers
3. Study: Production Guide
4. Run: Complex recipes and integration patterns
5. Design: Production architecture with observability

### Enterprise Level (Week 4+)
1. **NEW**: Implement comprehensive approval workflows
2. **NEW**: Set up distributed tracing
3. Study: Full production guide
4. Implement: Multi-tenant system
5. Deploy: Production application with monitoring

---

## 📖 Key Concepts (2025 Edition)

### Why TypeScript Agents SDK Over Swarm?

| Feature | Swarm (❌ Deprecated) | TypeScript Agents SDK (✅ Production) |
|---------|----------------------|--------------------------------------|
| **Status** | Experimental | Production-Ready |
| **Type Safety** | Limited | Full TypeScript 5.0+ |
| **IDE Support** | Basic | Excellent IntelliSense |
| **Human-in-Loop** | None | ✅ Built-in (NEW 2025) |
| **Guardrails** | None | Type-safe with Zod |
| **Tracing** | Basic logging | Comprehensive + Visualization |
| **Sessions** | Manual | Automatic |
| **Providers** | OpenAI only | 100+ via LiteLLM |
| **MCP Support** | None | First-class |
| **Maintenance** | Deprecated | Active Development |

### Core Primitives (The Six Building Blocks)

1. **Agent**: Type-safe LLM entity with instructions, tools, and guardrails
2. **Handoff**: Typed delegation mechanism between agents
3. **Tool**: Type-safe functions with Zod schema validation
4. **Guardrail**: Type-safe input/output validation
5. **Session**: Typed conversation history with generic support
6. **Runner**: Typed orchestrator with async/await patterns

### TypeScript Advantages

- **Full Type Safety**: Catch errors at compile time, not runtime
- **IDE Support**: IntelliSense autocomplete for all APIs
- **Zod Integration**: Runtime validation with automatic type inference
- **Generic Patterns**: Reusable typed components
- **Interface-Driven**: Clear contracts and documentation
- **Production-Ready**: Active maintenance and regular updates

---

## 🏗️ Use Case Examples

### Business Intelligence
- Market analysis agents
- Competitor monitoring
- Trend analysis
- Report generation

### Customer Service
- Support ticket routing with approvals
- FAQ systems
- Appointment scheduling
- Issue escalation

### Content Operations
- Blog post generation
- Social media content
- Email campaigns
- Translation services

### Data Processing
- CSV parsing and validation
- Data enrichment
- Format conversion
- Quality checks with guardrails

### Software Development
- Code review automation
- Documentation generation
- Bug analysis
- Performance profiling

---

## 🔧 Developer Experience (2025 Features)

### Features
✓ **Type-Safe**: Full TypeScript 5.0+ support
✓ **Human-in-Loop**: Approval workflows for critical operations (NEW 2025)
✓ **Observable**: Built-in tracing and visualization (NEW 2025)
✓ **Flexible**: Works with 100+ LLM providers (NEW 2025)
✓ **Secure**: Type-safe guardrails with Zod (ENHANCED 2025)
✓ **Scalable**: Production-ready patterns
✓ **MCP-Ready**: First-class Model Context Protocol support (NEW 2025)

### Best Practices Included
- Error handling strategies with retries
- Security hardening with guardrails
- Performance optimisation
- Testing approaches
- Deployment patterns with Docker/Kubernetes

---

## 📚 Code Examples Statistics

- **Total Code Examples**: 150+
- **2025 Feature Examples**: 50+ NEW
- **Production Patterns**: 50+
- **Recipes**: 18 complete implementations
- **Lines of Code**: 6000+
- **Type Annotations**: 100%
- **Real-world Use Cases**: 30+

---

## 🔐 Security Considerations (Enhanced 2025)

The guide covers:
- **NEW**: Human-in-the-loop approvals for critical operations
- **NEW**: Type-safe guardrails for input/output validation
- API key management and rotation
- Input validation and sanitisation
- XSS prevention
- CSRF protection
- Rate limiting
- DDoS mitigation
- Data privacy
- Audit logging with tracing

---

## 📊 Performance & Scalability

Topics covered:
- **NEW**: Built-in tracing for performance monitoring
- **NEW**: Provider-specific optimizations
- Caching strategies
- Connection pooling
- Load balancing
- Horizontal scaling
- Token optimisation
- Cost management
- Latency reduction

---

## 🧪 Testing Coverage

Includes patterns for:
- Unit testing with Jest
- Integration testing
- **NEW**: Testing approval workflows
- **NEW**: Testing guardrails
- Mocking LLM responses
- Test fixtures
- Coverage strategies
- CI/CD integration

---

## 📝 Code Quality

All code follows:
- TypeScript 5.0+ strict mode
- ESLint standards
- Prettier formatting
- Industry best practices
- SOLID principles
- Clean code practices

---

## 🌐 Language & Conventions

**Spelling Convention**: British English
- Optimisation (not optimization)
- Favour (not favor)
- Analyse (not analyze)
- Centre (not center)

---

## 📖 Document Breakdown

| Document | Pages | Focus | Audience |
|----------|-------|-------|----------|
| **Swarm Migration** | 30+ | Migration from Swarm | Swarm users |
| **2025 Features** | 40+ | Latest features | All developers |
| **Comprehensive** | 80+ | Complete reference | All levels |
| **Production** | 40+ | Enterprise patterns | DevOps/Architects |
| **Recipes** | 50+ | Practical examples | Developers |
| **Diagrams** | 15+ | Visual reference | All levels |
| **README** | This | Navigation | Getting started |

---

## 🎓 Learning Resources

Within the guides you'll find:
- **Code Examples**: Copy-paste ready implementations
- **Architecture Diagrams**: Visual representations
- **Real-World Scenarios**: Practical use cases
- **Best Practices**: Industry standards
- **Type Patterns**: TypeScript idioms
- **Error Handling**: Resilience strategies
- **Performance Tips**: Optimisation techniques
- **Security Measures**: Hardening guidelines
- **NEW**: Approval workflow patterns
- **NEW**: Guardrail implementations
- **NEW**: Tracing strategies

---

## 🔗 Related Documentation

These guides complement:
- [Official OpenAI Agents SDK Documentation](https://github.com/openai/openai-agents-js)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [TypeScript Handbook](https://www.typescriptlang.org/docs/)
- [Zod Documentation](https://zod.dev)
- **NEW**: [OpenAI Tracing Documentation](https://platform.openai.com/docs/tracing)

---

## 📌 Important Notes (2025 Edition)

1. **API Keys**: Never commit API keys; use environment variables
2. **Rate Limiting**: Implement backoff strategies for production
3. **Costs**: Monitor token usage; implement caching
4. **Testing**: Always test with mocked models before production
5. **Security**: Follow security best practices from production guide
6. **Monitoring**: Implement observability from day one
7. **NEW**: Use human-in-the-loop for critical operations
8. **NEW**: Implement guardrails for all user inputs
9. **NEW**: Enable tracing for production debugging
10. **NEW**: Consider multiple LLM providers for redundancy

---

## 🎯 Next Steps

1. **Choose Your Path**:
   - Coming from Swarm? → Start with **Swarm Migration Guide**
   - Want latest features? → Read **2025 Features Guide**
   - Beginner? → Start with **Basic Agent Recipes**
   - Building production? → Read **Production Guide**
   - Need examples? → Check **Practical Recipes**
   - Need architecture? → Review **Diagrams**

2. **Start Small**:
   - Create your first simple agent with tracing
   - Run a recipe that matches your use case
   - Understand the core concepts

3. **Add 2025 Features**:
   - Implement guardrails for input validation
   - Add human-in-the-loop for critical operations
   - Enable tracing for observability
   - Try multiple LLM providers

4. **Scale Up**:
   - Build multi-agent systems with handoffs
   - Implement production patterns
   - Deploy and monitor with comprehensive tracing

5. **Keep Learning**:
   - Reference comprehensive guide as needed
   - Follow best practices
   - Stay updated with SDK changes

---

## 📞 Support & Questions

When implementing:
- Check the **2025 Features Guide** for latest capabilities
- Check the **Swarm Migration Guide** if migrating
- Check the **Comprehensive Guide** for detailed explanations
- Review **Recipes** for similar implementations
- Study **Diagrams** for architectural understanding
- Follow **Production Guide** for enterprise patterns

---

## 🎉 You're Ready!

This complete guide provides everything needed to:
✓ **NEW**: Migrate from Swarm to production-ready SDK
✓ **NEW**: Implement human-in-the-loop approvals
✓ **NEW**: Use 100+ LLM providers
✓ **NEW**: Add comprehensive guardrails
✓ **NEW**: Enable built-in tracing
✓ Build simple to complex agent systems
✓ Implement production-grade applications
✓ Scale to enterprise levels
✓ Follow security and performance best practices
✓ Maintain observable, testable code

Happy building! 🚀

---

## 📋 Quick Reference

### Files in This Guide
- **🆕** [openai_agents_sdk_typescript_swarm_migration_guide.md](openai_agents_sdk_typescript_swarm_migration_guide.md) - Swarm to Agents SDK migration
- **🆕** [openai_agents_sdk_typescript_2025_features.md](openai_agents_sdk_typescript_2025_features.md) - 2025 features guide
- [openai_agents_sdk_typescript_comprehensive_guide.md](openai_agents_sdk_typescript_comprehensive_guide.md) - Complete reference
- [openai_agents_sdk_typescript_production_guide.md](openai_agents_sdk_typescript_production_guide.md) - Enterprise patterns
- [openai_agents_sdk_typescript_recipes.md](openai_agents_sdk_typescript_recipes.md) - Practical examples
- [openai_agents_sdk_typescript_diagrams.md](openai_agents_sdk_typescript_diagrams.md) - Visual architecture
- `README.md` - This file

### Key Topics Index
- **2025 Features**: 2025 Features Guide → All sections
- **Swarm Migration**: Swarm Migration Guide → Complete migration path
- Installation & Setup: Comprehensive Guide → Core Fundamentals
- Creating Agents: Comprehensive Guide → Simple Agents
- Multi-Agent Systems: Comprehensive Guide → Multi-Agent Systems
- Production Deployment: Production Guide → Deployment Architecture
- Code Examples: Recipes → All sections
- Architecture: Diagrams → All diagrams

---

**Version 0.3.2 - 2025 Edition**
**Updated: November 2025**
**Focus: TypeScript 5.0+ | Production Ready | Official Swarm Replacement**
**Status: ✅ Production-Ready | 🆕 2025 Features | 🎯 TypeScript-First**

---

## Streaming Examples
- [openai_agents_streaming_server_express.md](openai_agents_streaming_server_express.md)
