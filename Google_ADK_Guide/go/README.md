# Google Agent Development Kit (ADK) for Go

**Version:** 1.0.0
**Release Date:** November 7, 2025
**License:** Apache 2.0
**Official Repository:** https://github.com/google/adk-go
**Documentation:** https://google.github.io/adk-docs/

---

## Overview

The Google Agent Development Kit (ADK) for Go is an open-source, code-first toolkit that brings powerful AI agent development capabilities to the Go ecosystem. Announced in November 2025, ADK for Go enables developers to build, evaluate, and deploy sophisticated AI agents leveraging Go's strengths: type safety, performance, concurrency, and simplicity.

ADK for Go is optimized for Google's Gemini models and Google Cloud Platform, but maintains a model-agnostic and deployment-agnostic architecture that allows integration with other LLMs and platforms.

### Why ADK for Go?

**Idiomatic Go Design:** ADK for Go embraces Go's design philosophy with clear interfaces, strong typing, explicit error handling, and composable patterns that feel natural to Go developers.

**Performance & Concurrency:** Leverage Go's goroutines and channels to build highly concurrent, scalable agent systems that can handle multiple conversations, parallel tool execution, and distributed agent orchestration.

**Production-Ready:** Built for backend teams deploying production AI systems, with native support for containerization, microservices architecture, and cloud-native deployment patterns.

**Type Safety:** Strong typing throughout the API ensures compile-time safety for agent configurations, tool definitions, and data schemas, reducing runtime errors and improving maintainability.

**Google Cloud Integration:** First-class support for Vertex AI, Cloud Run, BigQuery, Firestore, Cloud Storage, and other GCP services with idiomatic Go clients.

---

## Key Features

### 1. Code-First Agent Development

Define agents entirely in Go code with full control over logic, orchestration, and behavior:

```go
agent := llmagent.New(
    llmagent.Name("TimeZoneExpert"),
    llmagent.Model("gemini-2.0-flash"),
    llmagent.Instruction("You are an expert at finding timezone information for cities worldwide."),
    llmagent.Tools(geminitool.GoogleSearch()),
)
```

### 2. Multi-Agent Systems

Build complex systems with multiple specialized agents working together:

- **Sequential Agents:** Execute agents in a defined order
- **Parallel Agents:** Run multiple agents concurrently
- **Loop Agents:** Iterate over data with agent processing
- **Hierarchical Agents:** Create supervisor-worker patterns with agent delegation

### 3. Agent2Agent (A2A) Protocol

Native support for the A2A protocol enables agents to communicate across different frameworks, platforms, and organizations:

```go
// Expose your agent via A2A
a2aServer := a2a.NewServer(agent)

// Consume remote A2A agents
remoteAgent := a2a.NewClient("https://remote-agent.example.com")
```

### 4. Model Context Protocol (MCP)

Integrate with the MCP ecosystem to connect agents with external data sources and tools:

```go
// Use MCP server
mcpClient := mcp.NewClient("mcp-server-url")
agent := llmagent.New(
    llmagent.Tools(mcpClient.GetTools()...),
)
```

### 5. Built-in Development UI

Test, debug, and showcase your agents with the included web interface:

```bash
go run agent.go web api webui
# Access at http://localhost:8080
```

### 6. Comprehensive Tool Ecosystem

**Built-in Tools:**
- Google Search (Gemini 2.0+)
- Code Execution (Python, sandboxed)
- Vertex AI RAG Engine
- Vertex AI Search
- BigQuery (7 tools)
- Cloud Spanner (7 tools)
- Bigtable (5 tools)

**Custom Tools:**
- Define tools as Go functions
- OpenAPI/Swagger integration
- MCP tool integration
- Tool composition and chaining

### 7. Structured Output

Generate type-safe structured responses with JSON schema validation:

```go
type WeatherReport struct {
    Temperature float64 `json:"temperature"`
    Conditions  string  `json:"conditions"`
    Humidity    int     `json:"humidity"`
}

agent := llmagent.New(
    llmagent.OutputSchema(WeatherReport{}),
)
```

### 8. Production Deployment

**Cloud Run:** Containerized deployment with automatic scaling
**Vertex AI Agent Engine:** Managed agent hosting with monitoring
**Kubernetes/GKE:** Full control over orchestration and scaling
**Docker:** Standard containerization for any platform

---

## Installation

### Prerequisites

- **Go:** Version 1.24.4 or later
- **Google Cloud Account:** (Optional, for GCP features)
- **Gemini API Key:** Get from [aistudio.google.com](https://aistudio.google.com/app/apikey)

### Quick Start

1. **Initialize a new Go module:**

```bash
mkdir my-agent
cd my-agent
go mod init my-agent/main
```

2. **Install ADK for Go:**

```bash
go get google.golang.org/adk
```

3. **Create your first agent:**

```go
// agent.go
package main

import (
    "context"
    "fmt"
    "log"
    "os"

    "google.golang.org/adk/agent/llmagent"
    "google.golang.org/adk/model/gemini"
    "google.golang.org/adk/runner"
    "google.golang.org/adk/tool/geminitool"
)

func main() {
    // Configure Gemini model
    model := gemini.New(
        gemini.WithAPIKey(os.Getenv("GOOGLE_API_KEY")),
        gemini.WithModelName("gemini-2.0-flash"),
    )

    // Create agent
    agent := llmagent.New(
        llmagent.Name("TimeZoneHelper"),
        llmagent.Model(model),
        llmagent.Instruction("You help users find timezone information for cities."),
        llmagent.Tools(geminitool.GoogleSearch()),
    )

    // Run agent
    r := runner.New(agent)

    ctx := context.Background()
    response, err := r.Run(ctx, "What timezone is Tokyo in?")
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println(response)
}
```

4. **Set your API key:**

```bash
export GOOGLE_API_KEY="your-api-key-here"
```

5. **Run your agent:**

```bash
go run agent.go
```

---

## Architecture Overview

### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                      ADK for Go                              │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Agent     │  │   Runner    │  │   Session   │         │
│  │             │  │             │  │             │         │
│  │ - LLMAgent  │  │ - Execute   │  │ - State     │         │
│  │ - Custom    │  │ - Parallel  │  │ - History   │         │
│  │ - Workflow  │  │ - Stream    │  │ - Memory    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │    Tool     │  │    Model    │  │   Memory    │         │
│  │             │  │             │  │             │         │
│  │ - Built-in  │  │ - Gemini    │  │ - In-Memory │         │
│  │ - Custom    │  │ - Vertex    │  │ - Firestore │         │
│  │ - MCP       │  │ - Others    │  │ - Custom    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
│                                                               │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Deployment & Operations                 │    │
│  │                                                       │    │
│  │  - Server (HTTP/gRPC)    - Telemetry                │    │
│  │  - A2A Protocol          - Evaluation                │    │
│  │  - MCP Integration       - Artifacts                 │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### Package Structure

```
google.golang.org/adk/
├── agent/              # Agent implementations
│   ├── llmagent/      # LLM-powered agents
│   ├── sequential/    # Sequential agent orchestration
│   ├── parallel/      # Parallel agent execution
│   └── loop/          # Loop-based agents
├── tool/              # Tool system
│   ├── geminitool/   # Google-specific tools
│   ├── openapi/      # OpenAPI tool integration
│   └── custom/       # Custom tool helpers
├── model/             # Model integrations
│   ├── gemini/       # Gemini API
│   └── vertex/       # Vertex AI
├── runner/            # Agent execution
├── session/           # Session management
├── memory/            # Memory systems
├── server/            # Server infrastructure
├── a2a/              # Agent2Agent protocol
├── mcp/              # Model Context Protocol
├── artifact/         # Artifact handling
├── telemetry/        # Observability
└── types/            # Core interfaces and types
```

---

## Core Concepts

### Agents

An **agent** is the fundamental building block in ADK. Agents encapsulate:
- **Identity:** Name and description
- **Intelligence:** LLM model configuration
- **Capabilities:** Available tools and functions
- **Behavior:** System instructions and constraints
- **I/O:** Input/output schemas

**Agent Types:**

1. **LLMAgent:** Uses language models for reasoning and decision-making
2. **SequentialAgent:** Executes multiple agents in order
3. **ParallelAgent:** Runs agents concurrently
4. **LoopAgent:** Iterates over data with agent processing
5. **CustomAgent:** Implement custom logic with the Agent interface

### Tools

**Tools** extend agent capabilities by providing access to external functions, APIs, and data:

```go
// Function-based tool
func getCurrentWeather(location string) (string, error) {
    // Implementation
}

tool := tool.FromFunc(getCurrentWeather)

// Agent with tool
agent := llmagent.New(
    llmagent.Tools(tool),
)
```

### Runner

The **Runner** executes agents and manages the execution lifecycle:

```go
r := runner.New(agent)

// Synchronous execution
response, err := r.Run(ctx, "user message")

// Streaming execution
stream, err := r.Stream(ctx, "user message")
```

### Session

**Sessions** manage conversation state, history, and context:

```go
// Create session
sess := session.New(session.ID("user-123"))

// Run with session
response, err := r.RunWithSession(ctx, sess, "message")

// Session persists state across multiple turns
```

### Memory

**Memory** systems provide long-term storage and retrieval for agent context:

- **In-Memory:** Fast, temporary storage
- **Firestore:** Persistent cloud storage
- **Custom:** Implement the Memory interface

---

## Use Cases

### Customer Support Agent

Build intelligent support agents that can search documentation, access customer data, and escalate to human agents:

```go
agent := llmagent.New(
    llmagent.Name("SupportAgent"),
    llmagent.Tools(
        searchKnowledgeBase,
        lookupCustomerAccount,
        createTicket,
    ),
)
```

### Data Analysis Agent

Create agents that query databases, run analyses, and generate reports:

```go
agent := llmagent.New(
    llmagent.Name("DataAnalyst"),
    llmagent.Tools(
        bigqueryTool,
        visualizationTool,
        reportGenerator,
    ),
)
```

### Multi-Agent Research System

Build systems where specialized agents collaborate on complex tasks:

```go
supervisor := llmagent.New(
    llmagent.Name("Supervisor"),
    llmagent.Tools(
        agent.Tool(researchAgent),
        agent.Tool(analysisAgent),
        agent.Tool(writerAgent),
    ),
)
```

### DevOps Automation

Automate infrastructure tasks with agents that interact with cloud services:

```go
agent := llmagent.New(
    llmagent.Name("DevOpsAgent"),
    llmagent.Tools(
        gcpComputeTool,
        kubernetesTool,
        monitoringTool,
    ),
)
```

---

## Documentation Structure

This repository contains comprehensive documentation for ADK for Go:

### **GUIDE_INDEX.md**
Navigation guide with links to all documentation resources

### **google_adk_go_comprehensive_guide.md** (15,000+ lines)
Complete technical reference covering:
- Installation and setup
- All packages, types, interfaces, and functions
- Core concepts and patterns
- Simple and multi-agent systems
- Tools, MCP, and A2A integration
- Memory and session management
- Google Cloud integration
- Advanced topics (concurrency, testing, performance)
- Hundreds of code examples

### **google_adk_go_production_guide.md** (8,000+ lines)
Production deployment guide including:
- Architecture patterns
- Cloud Run deployment
- Vertex AI integration
- Security and authentication
- Monitoring and observability
- Performance optimization
- CI/CD pipelines
- Production best practices

### **google_adk_go_diagrams.md**
Visual architecture diagrams and system designs

### **google_adk_go_recipes.md**
Copy-paste ready code examples for common patterns

---

## Community and Support

### Official Resources

- **Documentation:** https://google.github.io/adk-docs/
- **GitHub Repository:** https://github.com/google/adk-go
- **Issue Tracker:** https://github.com/google/adk-go/issues
- **Release Notes:** https://github.com/google/adk-go/releases

### Community

- **Reddit:** r/agentdevelopmentkit
- **Discord:** (Check official docs for invite)
- **Stack Overflow:** Tag `google-adk-go`

### Related Projects

- **A2A Protocol:** https://a2a-protocol.org/
- **A2A Go SDK:** https://github.com/go-a2a/adk-go
- **MCP Toolbox:** https://github.com/google/mcp-toolbox
- **ADK Python:** https://github.com/google/adk-python
- **ADK Java:** https://github.com/google/adk-java

---

## Quick Links

- [Installation Guide](./google_adk_go_comprehensive_guide.md#installation-and-setup)
- [First Agent Tutorial](./google_adk_go_comprehensive_guide.md#creating-your-first-agent)
- [Multi-Agent Systems](./google_adk_go_comprehensive_guide.md#multi-agent-systems)
- [Tools Guide](./google_adk_go_comprehensive_guide.md#tools-integration)
- [Production Deployment](./google_adk_go_production_guide.md)
- [Code Recipes](./google_adk_go_recipes.md)
- [Architecture Diagrams](./google_adk_go_diagrams.md)

---

## Contributing

ADK for Go is open source and welcomes contributions:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

See [CONTRIBUTING.md](https://github.com/google/adk-go/blob/main/CONTRIBUTING.md) for detailed guidelines.

---

## License

Apache 2.0 License. See [LICENSE](https://github.com/google/adk-go/blob/main/LICENSE) for details.

---

## Version History

### v1.0.0 (November 7, 2025)
- Initial release of ADK for Go
- LLMAgent implementation
- Multi-agent orchestration (Sequential, Parallel, Loop)
- A2A protocol support
- MCP integration
- Built-in tools (Google Search, Code Execution)
- Cloud Run deployment support
- Development UI
- Comprehensive documentation

---

**Ready to build AI agents in Go?** Start with the [Comprehensive Guide](./google_adk_go_comprehensive_guide.md) or jump straight to [Code Recipes](./google_adk_go_recipes.md).
