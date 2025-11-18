# Google ADK for Go - Comprehensive Technical Guide

**Complete API Reference and Developer Guide**

Version: 0.1.0
Last Updated: November 2025
License: Apache 2.0

---

## Table of Contents

1. [Installation and Setup](#installation-and-setup)
2. [Core Architecture](#core-architecture)
3. [Package Overview](#package-overview)
4. [Types and Interfaces](#types-and-interfaces)
5. [LLMAgent](#llmagent)
6. [Agent Orchestration](#agent-orchestration)
7. [Tools System](#tools-system)
8. [Model Integration](#model-integration)
9. [Runner and Execution](#runner-and-execution)
10. [Session Management](#session-management)
11. [Memory System](#memory-system)
12. [Model Context Protocol (MCP)](#model-context-protocol-mcp)
13. [Agent2Agent (A2A) Protocol](#agent2agent-a2a-protocol)
14. [Server and API](#server-and-api)
15. [Artifacts](#artifacts)
16. [Telemetry](#telemetry)
17. [Go Idioms and Patterns](#go-idioms-and-patterns)
18. [Context Handling](#context-handling)
19. [Error Handling](#error-handling)
20. [Goroutine Patterns](#goroutine-patterns)
21. [Testing](#testing)
22. [Best Practices](#best-practices)

---

## Installation and Setup

### Prerequisites

**Go Version**: 1.24.4 or later

Verify your Go installation:

```bash
go version
# go version go1.24.4 linux/amd64
```

### Install ADK Package

```bash
# Initialize a new Go module
mkdir my-agent && cd my-agent
go mod init github.com/yourorg/my-agent

# Install ADK for Go
go get google.golang.org/adk@latest
```

### Environment Setup

Create a `.env` file for configuration:

```bash
# .env
GOOGLE_API_KEY=your_api_key_here
GOOGLE_CLOUD_PROJECT=your-gcp-project
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
```

Load environment variables:

```go
package main

import (
    "github.com/joho/godotenv"
    "log"
)

func init() {
    if err := godotenv.Load(); err != nil {
        log.Printf("Warning: .env file not found")
    }
}
```

### Verify Installation

```go
package main

import (
    "context"
    "fmt"
    "log"
    "os"

    "google.golang.org/adk/agent/llmagent"
    "google.golang.org/adk/model/gemini"
)

func main() {
    ctx := context.Background()

    // Initialize model
    model, err := gemini.New(ctx, &gemini.Config{
        APIKey: os.Getenv("GOOGLE_API_KEY"),
        Model:  "gemini-2.0-flash",
    })
    if err != nil {
        log.Fatal(err)
    }

    // Create agent
    agent := llmagent.New(&llmagent.Config{
        Name:  "test_agent",
        Model: model,
    })

    fmt.Printf("Agent created successfully: %s\n", agent.Name())
}
```

---

## Core Architecture

### Design Philosophy

ADK for Go follows these principles:

1. **Idiomatic Go**: Embraces Go conventions and patterns
2. **Explicit over Implicit**: Clear configuration and behavior
3. **Composable**: Build complex agents from simple primitives
4. **Type-Safe**: Leverage Go's type system for safety
5. **Context-Aware**: Proper context propagation throughout
6. **Error Transparent**: Clear error handling and reporting

### Architecture Layers

```
┌─────────────────────────────────────────┐
│         Application Layer                │
│    (Your Agent Implementation)           │
├─────────────────────────────────────────┤
│         Agent Orchestration              │
│   (Sequential, Parallel, Loop)           │
├─────────────────────────────────────────┤
│            Core Agent Layer              │
│         (LLMAgent, Custom)               │
├─────────────────────────────────────────┤
│      Tools & Capabilities Layer          │
│  (Function, Agent, MCP, Gemini)          │
├─────────────────────────────────────────┤
│          Model Integration               │
│       (Gemini, Vertex AI)                │
├─────────────────────────────────────────┤
│       Session & Memory Layer             │
│   (State, History, Persistence)          │
├─────────────────────────────────────────┤
│       Infrastructure Layer               │
│ (Server, A2A, Telemetry, Artifacts)      │
└─────────────────────────────────────────┘
```

### Request Flow

```
User Input
    ↓
Context Creation
    ↓
Session Load (if exists)
    ↓
Agent.Run()
    ↓
Model Request (with tools)
    ↓
Tool Execution (if needed)
    ↓
Response Generation
    ↓
Session Save
    ↓
Response to User
```

---

## Package Overview

### `google.golang.org/adk/agent`

Core agent implementations and interfaces.

**Subpackages:**
- `llmagent` - LLM-powered agents
- `loopagent` - Iterative agent execution
- `parallelagent` - Concurrent agent orchestration
- `sequentialagent` - Sequential agent workflows
- `remoteagent` - Remote agent communication via A2A

### `google.golang.org/adk/model`

LLM model integrations.

**Subpackages:**
- `gemini` - Google Gemini API integration

### `google.golang.org/adk/tool`

Tool system for extending agent capabilities.

**Subpackages:**
- `functiontool` - Go function-based tools
- `agenttool` - Use agents as tools
- `geminitool` - Native Gemini tools (Search, Code Execution)
- `mcptoolset` - Model Context Protocol integration
- `exitlooptool` - Loop termination control
- `loadartifactstool` - Artifact loading

### `google.golang.org/adk/session`

Session and state management.

**Subpackages:**
- `database` - Persistent session storage

### `google.golang.org/adk/memory`

Long-term memory and knowledge storage.

### `google.golang.org/adk/runner`

Agent execution and lifecycle management.

### `google.golang.org/adk/server`

Server implementations for agent deployment.

**Subpackages:**
- `restapi` - REST API server
- `adka2a` - Agent2Agent protocol server
- `webui` - Web user interface

### `google.golang.org/adk/artifact`

Artifact storage and retrieval.

**Subpackages:**
- `gcs` - Google Cloud Storage integration

### `google.golang.org/adk/telemetry`

Observability, metrics, and tracing.

---

## Types and Interfaces

### Core Agent Interface

```go
package agent

// Agent is the fundamental interface all agents must implement
type Agent interface {
    // Name returns the agent's unique identifier
    Name() string

    // Description returns a human-readable description
    Description() string

    // Run executes the agent with the given input
    Run(ctx context.Context, input string) (*Response, error)

    // RunWithSession executes with session state
    RunWithSession(ctx context.Context, sess *session.Session, input string) (*Response, error)

    // Stream executes and streams responses
    Stream(ctx context.Context, input string) (<-chan *StreamChunk, error)
}

// Response represents an agent's output
type Response struct {
    Text      string                 // Primary text response
    Metadata  map[string]interface{} // Additional metadata
    ToolCalls []ToolCall            // Tools invoked
    Usage     *Usage                // Token usage
}

// StreamChunk represents a piece of streaming response
type StreamChunk struct {
    Text      string
    Delta     string
    Done      bool
    Error     error
}

// Usage tracks token consumption
type Usage struct {
    InputTokens  int
    OutputTokens int
    TotalTokens  int
}

// ToolCall represents a tool invocation
type ToolCall struct {
    Name   string
    Args   map[string]interface{}
    Result interface{}
    Error  error
}
```

### Tool Interface

```go
package tool

// Tool defines the interface for agent tools
type Tool interface {
    // Name returns the tool's identifier
    Name() string

    // Description returns what the tool does
    Description() string

    // Schema returns the input schema (JSON Schema)
    Schema() *Schema

    // Execute runs the tool with given arguments
    Execute(ctx context.Context, args map[string]interface{}) (interface{}, error)
}

// Schema defines tool input parameters
type Schema struct {
    Type       string                 `json:"type"`
    Properties map[string]*Property   `json:"properties"`
    Required   []string               `json:"required,omitempty"`
}

// Property defines a single parameter
type Property struct {
    Type        string   `json:"type"`
    Description string   `json:"description"`
    Enum        []string `json:"enum,omitempty"`
}
```

### Model Interface

```go
package model

// LLM defines the interface for language models
type LLM interface {
    // GenerateContent creates a response from a prompt
    GenerateContent(ctx context.Context, req *GenerateRequest) (*GenerateResponse, error)

    // StreamContent streams response chunks
    StreamContent(ctx context.Context, req *GenerateRequest) (<-chan *StreamChunk, error)

    // CountTokens returns token count for given input
    CountTokens(ctx context.Context, input string) (int, error)
}

// GenerateRequest contains generation parameters
type GenerateRequest struct {
    Messages    []*Message
    Tools       []Tool
    Temperature *float64
    MaxTokens   *int
    TopP        *float64
    TopK        *int
}

// Message represents a conversation message
type Message struct {
    Role    string // "user", "model", "system"
    Content string
    Parts   []Part
}

// Part represents multimodal content
type Part interface {
    PartType() string
}
```

### Session Interface

```go
package session

// Session manages conversation state
type Session struct {
    ID       string
    UserID   string
    Metadata map[string]interface{}
    State    map[string]interface{}
    History  []*Message
}

// Store defines session persistence
type Store interface {
    // Save persists session data
    Save(ctx context.Context, sess *Session) error

    // Load retrieves session data
    Load(ctx context.Context, id string) (*Session, error)

    // Delete removes session data
    Delete(ctx context.Context, id string) error

    // List returns sessions for a user
    List(ctx context.Context, userID string) ([]*Session, error)
}
```

---

## LLMAgent

LLMAgent is the core agent type powered by language models.

### Basic Configuration

```go
package main

import (
    "context"
    "log"
    "os"

    "google.golang.org/adk/agent/llmagent"
    "google.golang.org/adk/model/gemini"
)

func main() {
    ctx := context.Background()

    // Initialize model
    model, err := gemini.New(ctx, &gemini.Config{
        APIKey: os.Getenv("GOOGLE_API_KEY"),
        Model:  "gemini-2.0-flash",
    })
    if err != nil {
        log.Fatal(err)
    }

    // Create agent with minimal config
    agent := llmagent.New(&llmagent.Config{
        Name:        "assistant",
        Description: "A helpful AI assistant",
        Instruction: "You are a helpful AI assistant. Be concise and accurate.",
        Model:       model,
    })

    // Run agent
    response, err := agent.Run(ctx, "Hello, how can you help me?")
    if err != nil {
        log.Fatal(err)
    }

    log.Printf("Response: %s\n", response.Text)
}
```

### Full Configuration Options

```go
type Config struct {
    // Identity
    Name        string // Required: unique agent identifier
    Description string // Optional: human-readable description
    Instruction string // Optional: system instruction/prompt

    // Model
    Model model.LLM // Required: language model instance

    // Tools
    Tools []tool.Tool // Optional: available tools

    // Generation Settings
    Temperature *float64 // Optional: 0.0-2.0, controls randomness
    MaxTokens   *int     // Optional: max output tokens
    TopP        *float64 // Optional: nucleus sampling
    TopK        *int     // Optional: top-k sampling

    // Input/Output
    InputSchema  *Schema // Optional: validate input
    OutputSchema *Schema // Optional: structured output

    // Behavior
    StopSequences []string // Optional: generation stop tokens
    SafetySettings []*SafetySetting // Optional: content safety

    // Advanced
    ToolChoice    string // "auto", "none", "any", or specific tool
    ParallelTools bool   // Allow parallel tool execution
}
```

### Advanced LLMAgent Example

```go
func createAdvancedAgent() *llmagent.Agent {
    ctx := context.Background()

    model, _ := gemini.New(ctx, &gemini.Config{
        APIKey: os.Getenv("GOOGLE_API_KEY"),
        Model:  "gemini-2.0-flash",
    })

    temp := 0.7
    maxTokens := 2048
    topP := 0.95
    topK := 40

    agent := llmagent.New(&llmagent.Config{
        Name: "advanced_assistant",
        Description: "An advanced AI assistant with custom settings",
        Instruction: `You are an expert AI assistant specializing in:
        - Technical explanations
        - Code generation
        - Problem solving

        Always:
        - Provide clear, concise answers
        - Use examples when helpful
        - Cite sources when available
        - Admit when uncertain`,

        Model:       model,
        Temperature: &temp,
        MaxTokens:   &maxTokens,
        TopP:        &topP,
        TopK:        &topK,

        StopSequences: []string{"END", "STOP"},

        SafetySettings: []*SafetySetting{
            {
                Category:  "HARM_CATEGORY_HATE_SPEECH",
                Threshold: "BLOCK_MEDIUM_AND_ABOVE",
            },
        },

        ToolChoice:    "auto",
        ParallelTools: true,
    })

    return agent
}
```

### Structured Output

Generate type-safe responses with JSON schema:

```go
package main

import (
    "context"
    "encoding/json"
    "log"

    "google.golang.org/adk/agent/llmagent"
)

// Define output structure
type WeatherReport struct {
    Location    string  `json:"location"`
    Temperature float64 `json:"temperature"`
    Conditions  string  `json:"conditions"`
    Humidity    int     `json:"humidity"`
    WindSpeed   float64 `json:"wind_speed"`
    Forecast    string  `json:"forecast"`
}

func structuredOutputExample() {
    ctx := context.Background()
    model := createModel()

    // Define schema from struct
    schema := &llmagent.Schema{
        Type: "object",
        Properties: map[string]*llmagent.Property{
            "location":    {Type: "string", Description: "City name"},
            "temperature": {Type: "number", Description: "Temperature in Celsius"},
            "conditions":  {Type: "string", Description: "Weather conditions"},
            "humidity":    {Type: "integer", Description: "Humidity percentage"},
            "wind_speed":  {Type: "number", Description: "Wind speed in km/h"},
            "forecast":    {Type: "string", Description: "Weather forecast"},
        },
        Required: []string{"location", "temperature", "conditions"},
    }

    agent := llmagent.New(&llmagent.Config{
        Name:         "weather_reporter",
        Model:        model,
        OutputSchema: schema,
    })

    response, err := agent.Run(ctx, "What's the weather in Tokyo?")
    if err != nil {
        log.Fatal(err)
    }

    // Parse structured output
    var weather WeatherReport
    if err := json.Unmarshal([]byte(response.Text), &weather); err != nil {
        log.Fatal(err)
    }

    log.Printf("Weather in %s: %.1f°C, %s\n",
        weather.Location, weather.Temperature, weather.Conditions)
}
```

---

## Agent Orchestration

Build complex multi-agent systems with orchestration primitives.

### SequentialAgent

Execute agents in a defined order, passing output from one to the next.

```go
package main

import (
    "context"
    "log"

    "google.golang.org/adk/agent/sequentialagent"
    "google.golang.org/adk/agent/llmagent"
)

func sequentialExample() {
    ctx := context.Background()
    model := createModel()

    // Research agent
    researcher := llmagent.New(&llmagent.Config{
        Name:        "researcher",
        Instruction: "Research the given topic and provide key facts.",
        Model:       model,
    })

    // Analysis agent
    analyzer := llmagent.New(&llmagent.Config{
        Name:        "analyzer",
        Instruction: "Analyze the research and identify main themes.",
        Model:       model,
    })

    // Writer agent
    writer := llmagent.New(&llmagent.Config{
        Name:        "writer",
        Instruction: "Write a concise summary based on the analysis.",
        Model:       model,
    })

    // Create sequential workflow
    workflow := sequentialagent.New(&sequentialagent.Config{
        Name:        "research_workflow",
        Description: "Research, analyze, and write",
        Agents:      []agent.Agent{researcher, analyzer, writer},
    })

    // Execute workflow
    response, err := workflow.Run(ctx, "Explain quantum computing")
    if err != nil {
        log.Fatal(err)
    }

    log.Printf("Final output: %s\n", response.Text)
}
```

### ParallelAgent

Run multiple agents concurrently and aggregate results.

```go
package main

import (
    "context"
    "log"

    "google.golang.org/adk/agent/parallelagent"
    "google.golang.org/adk/agent/llmagent"
)

func parallelExample() {
    ctx := context.Background()
    model := createModel()

    // Create specialized agents
    techAgent := llmagent.New(&llmagent.Config{
        Name:        "tech_expert",
        Instruction: "Provide technical analysis",
        Model:       model,
    })

    businessAgent := llmagent.New(&llmagent.Config{
        Name:        "business_expert",
        Instruction: "Provide business analysis",
        Model:       model,
    })

    marketAgent := llmagent.New(&llmagent.Config{
        Name:        "market_expert",
        Instruction: "Provide market analysis",
        Model:       model,
    })

    // Create parallel execution
    parallel := parallelagent.New(&parallelagent.Config{
        Name:        "multi_perspective",
        Description: "Get multiple perspectives simultaneously",
        Agents:      []agent.Agent{techAgent, businessAgent, marketAgent},
        Aggregator:  aggregateResponses, // Custom aggregation function
    })

    response, err := parallel.Run(ctx, "Analyze the AI industry")
    if err != nil {
        log.Fatal(err)
    }

    log.Printf("Aggregated analysis: %s\n", response.Text)
}

// Custom aggregation function
func aggregateResponses(responses []*agent.Response) (*agent.Response, error) {
    combined := "# Multi-Perspective Analysis\n\n"

    for i, resp := range responses {
        combined += fmt.Sprintf("## Perspective %d\n%s\n\n", i+1, resp.Text)
    }

    return &agent.Response{
        Text: combined,
    }, nil
}
```

### LoopAgent

Iterate over data or repeat execution until a condition is met.

```go
package main

import (
    "context"
    "log"

    "google.golang.org/adk/agent/loopagent"
    "google.golang.org/adk/agent/llmagent"
    "google.golang.org/adk/tool/exitlooptool"
)

func loopExample() {
    ctx := context.Background()
    model := createModel()

    // Agent that processes items
    processor := llmagent.New(&llmagent.Config{
        Name:        "processor",
        Instruction: "Process the given item. Call exit_loop when done with all items.",
        Model:       model,
        Tools:       []tool.Tool{exitlooptool.New()},
    })

    // Create loop agent
    loop := loopagent.New(&loopagent.Config{
        Name:        "item_processor",
        Description: "Process items until completion",
        Agent:       processor,
        MaxIterations: 10, // Safety limit
    })

    // Execute loop
    response, err := loop.Run(ctx, "Process items: A, B, C")
    if err != nil {
        log.Fatal(err)
    }

    log.Printf("Processing complete: %s\n", response.Text)
}
```

### Hierarchical Multi-Agent System

Build supervisor-worker patterns:

```go
package main

import (
    "context"
    "log"

    "google.golang.org/adk/agent/llmagent"
    "google.golang.org/adk/tool/agenttool"
)

func hierarchicalExample() {
    ctx := context.Background()
    model := createModel()

    // Worker agents
    coder := llmagent.New(&llmagent.Config{
        Name:        "coder",
        Instruction: "Write clean, efficient code",
        Model:       model,
    })

    reviewer := llmagent.New(&llmagent.Config{
        Name:        "reviewer",
        Instruction: "Review code for bugs and improvements",
        Model:       model,
    })

    documenter := llmagent.New(&llmagent.Config{
        Name:        "documenter",
        Instruction: "Write clear documentation",
        Model:       model,
    })

    // Convert agents to tools
    coderTool := agenttool.New(coder)
    reviewerTool := agenttool.New(reviewer)
    documenterTool := agenttool.New(documenter)

    // Supervisor agent
    supervisor := llmagent.New(&llmagent.Config{
        Name: "supervisor",
        Instruction: `You are a software development supervisor.
        Delegate tasks to your team:
        - Use 'coder' for writing code
        - Use 'reviewer' for code review
        - Use 'documenter' for documentation
        Coordinate their work to complete the task.`,
        Model: model,
        Tools: []tool.Tool{coderTool, reviewerTool, documenterTool},
    })

    // Execute hierarchical workflow
    response, err := supervisor.Run(ctx,
        "Create a function to calculate fibonacci numbers")
    if err != nil {
        log.Fatal(err)
    }

    log.Printf("Project complete: %s\n", response.Text)
}
```

---

## Tools System

Tools extend agent capabilities with external functions, APIs, and services.

### Function Tool

Convert Go functions into agent tools:

```go
package main

import (
    "context"
    "fmt"
    "time"

    "google.golang.org/adk/tool/functiontool"
)

// Simple function tool
func getCurrentTime(ctx context.Context) (string, error) {
    return time.Now().Format(time.RFC3339), nil
}

// Function with parameters
func getWeather(ctx context.Context, location string, units string) (map[string]interface{}, error) {
    // In real implementation, call weather API
    return map[string]interface{}{
        "location":    location,
        "temperature": 22.5,
        "units":       units,
        "conditions":  "Partly cloudy",
    }, nil
}

// Complex function with struct return
type CalculationResult struct {
    Input  float64 `json:"input"`
    Result float64 `json:"result"`
    Method string  `json:"method"`
}

func calculate(ctx context.Context, operation string, value float64) (*CalculationResult, error) {
    var result float64

    switch operation {
    case "square":
        result = value * value
    case "sqrt":
        result = math.Sqrt(value)
    case "double":
        result = value * 2
    default:
        return nil, fmt.Errorf("unknown operation: %s", operation)
    }

    return &CalculationResult{
        Input:  value,
        Result: result,
        Method: operation,
    }, nil
}

func functionToolExample() {
    ctx := context.Background()
    model := createModel()

    // Create tools from functions
    timeTool := functiontool.New(getCurrentTime, &functiontool.Config{
        Name:        "get_current_time",
        Description: "Get the current time in RFC3339 format",
    })

    weatherTool := functiontool.New(getWeather, &functiontool.Config{
        Name:        "get_weather",
        Description: "Get weather for a location",
        Parameters: map[string]*functiontool.Parameter{
            "location": {
                Type:        "string",
                Description: "City name",
                Required:    true,
            },
            "units": {
                Type:        "string",
                Description: "Temperature units: celsius or fahrenheit",
                Required:    false,
                Default:     "celsius",
            },
        },
    })

    calcTool := functiontool.New(calculate, &functiontool.Config{
        Name:        "calculate",
        Description: "Perform mathematical calculations",
        Parameters: map[string]*functiontool.Parameter{
            "operation": {
                Type:        "string",
                Description: "Operation to perform",
                Required:    true,
                Enum:        []string{"square", "sqrt", "double"},
            },
            "value": {
                Type:        "number",
                Description: "Input value",
                Required:    true,
            },
        },
    })

    // Create agent with tools
    agent := llmagent.New(&llmagent.Config{
        Name:        "assistant",
        Instruction: "Use available tools to help the user",
        Model:       model,
        Tools:       []tool.Tool{timeTool, weatherTool, calcTool},
    })

    response, _ := agent.Run(ctx, "What's the time and weather in Tokyo?")
    log.Printf("Response: %s\n", response.Text)
}
```

### Custom Tool Implementation

Implement the Tool interface for full control:

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "net/http"

    "google.golang.org/adk/tool"
)

// Custom database query tool
type DatabaseTool struct {
    name        string
    description string
    connString  string
}

func NewDatabaseTool(connString string) *DatabaseTool {
    return &DatabaseTool{
        name:        "query_database",
        description: "Execute SQL queries on the database",
        connString:  connString,
    }
}

func (t *DatabaseTool) Name() string {
    return t.name
}

func (t *DatabaseTool) Description() string {
    return t.description
}

func (t *DatabaseTool) Schema() *tool.Schema {
    return &tool.Schema{
        Type: "object",
        Properties: map[string]*tool.Property{
            "query": {
                Type:        "string",
                Description: "SQL query to execute",
            },
            "limit": {
                Type:        "integer",
                Description: "Maximum number of rows to return",
            },
        },
        Required: []string{"query"},
    }
}

func (t *DatabaseTool) Execute(ctx context.Context, args map[string]interface{}) (interface{}, error) {
    query, ok := args["query"].(string)
    if !ok {
        return nil, fmt.Errorf("query parameter required")
    }

    limit := 100
    if l, ok := args["limit"].(float64); ok {
        limit = int(l)
    }

    // Execute query (simplified)
    results, err := t.executeQuery(ctx, query, limit)
    if err != nil {
        return nil, fmt.Errorf("query failed: %w", err)
    }

    return results, nil
}

func (t *DatabaseTool) executeQuery(ctx context.Context, query string, limit int) ([]map[string]interface{}, error) {
    // Implement actual database query logic
    return []map[string]interface{}{
        {"id": 1, "name": "Example"},
    }, nil
}

// Custom API tool
type APICaller struct {
    baseURL string
    apiKey  string
}

func NewAPICaller(baseURL, apiKey string) *APICaller {
    return &APICaller{
        baseURL: baseURL,
        apiKey:  apiKey,
    }
}

func (a *APICaller) Name() string {
    return "call_api"
}

func (a *APICaller) Description() string {
    return "Make HTTP API calls"
}

func (a *APICaller) Schema() *tool.Schema {
    return &tool.Schema{
        Type: "object",
        Properties: map[string]*tool.Property{
            "endpoint": {
                Type:        "string",
                Description: "API endpoint path",
            },
            "method": {
                Type:        "string",
                Description: "HTTP method",
                Enum:        []string{"GET", "POST", "PUT", "DELETE"},
            },
            "body": {
                Type:        "object",
                Description: "Request body (for POST/PUT)",
            },
        },
        Required: []string{"endpoint", "method"},
    }
}

func (a *APICaller) Execute(ctx context.Context, args map[string]interface{}) (interface{}, error) {
    endpoint := args["endpoint"].(string)
    method := args["method"].(string)

    url := a.baseURL + endpoint
    var req *http.Request
    var err error

    if method == "POST" || method == "PUT" {
        body, _ := json.Marshal(args["body"])
        req, err = http.NewRequestWithContext(ctx, method, url, bytes.NewBuffer(body))
    } else {
        req, err = http.NewRequestWithContext(ctx, method, url, nil)
    }

    if err != nil {
        return nil, err
    }

    req.Header.Set("Authorization", "Bearer "+a.apiKey)
    req.Header.Set("Content-Type", "application/json")

    client := &http.Client{Timeout: 30 * time.Second}
    resp, err := client.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    var result map[string]interface{}
    json.NewDecoder(resp.Body).Decode(&result)

    return result, nil
}
```

### Gemini Native Tools

Use Google Search and Code Execution built into Gemini:

```go
package main

import (
    "context"
    "log"

    "google.golang.org/adk/agent/llmagent"
    "google.golang.org/adk/tool/geminitool"
)

func geminiToolsExample() {
    ctx := context.Background()
    model := createModel()

    // Create agent with Google Search
    searchAgent := llmagent.New(&llmagent.Config{
        Name:        "search_assistant",
        Instruction: "Help users find information using Google Search",
        Model:       model,
        Tools:       []tool.Tool{geminitool.GoogleSearch()},
    })

    response, err := searchAgent.Run(ctx, "What are the latest AI developments?")
    if err != nil {
        log.Fatal(err)
    }
    log.Printf("Search results: %s\n", response.Text)

    // Create agent with code execution
    coderAgent := llmagent.New(&llmagent.Config{
        Name:        "code_runner",
        Instruction: "Execute Python code to solve problems",
        Model:       model,
        Tools:       []tool.Tool{geminitool.CodeExecution()},
    })

    response, err = coderAgent.Run(ctx, "Calculate the first 10 Fibonacci numbers")
    if err != nil {
        log.Fatal(err)
    }
    log.Printf("Code result: %s\n", response.Text)
}
```

---

## Model Context Protocol (MCP)

MCP enables standardized integration with external tools and services.

### Using MCP Tools

```go
package main

import (
    "context"
    "log"

    "google.golang.org/adk/agent/llmagent"
    "google.golang.org/adk/tool/mcptoolset"
)

func mcpExample() {
    ctx := context.Background()
    model := createModel()

    // Connect to MCP server
    mcpTools, err := mcptoolset.New(ctx, &mcptoolset.Config{
        ServerURL: "mcp://localhost:8080/tools",
        Transport: "stdio", // or "http"
    })
    if err != nil {
        log.Fatal(err)
    }

    // Create agent with MCP tools
    agent := llmagent.New(&llmagent.Config{
        Name:        "mcp_agent",
        Instruction: "Use MCP tools to help the user",
        Model:       model,
        Tools:       mcpTools.GetTools(),
    })

    response, err := agent.Run(ctx, "Query the database for user information")
    if err != nil {
        log.Fatal(err)
    }

    log.Printf("Response: %s\n", response.Text)
}
```

### Exposing ADK Tools via MCP

```go
package main

import (
    "context"
    "log"

    "google.golang.org/adk/server/mcp"
    "google.golang.org/adk/tool/functiontool"
)

func exposeMCPTools() {
    ctx := context.Background()

    // Create tools
    tools := []tool.Tool{
        functiontool.New(getWeather, nil),
        functiontool.New(getCurrentTime, nil),
    }

    // Create MCP server
    server := mcp.NewServer(&mcp.Config{
        Name:        "my-tools",
        Description: "My custom tools exposed via MCP",
        Tools:       tools,
        Transport:   "stdio",
    })

    // Start server
    if err := server.Serve(ctx); err != nil {
        log.Fatal(err)
    }
}
```

---

## Agent2Agent (A2A) Protocol

A2A enables agents to communicate across services and frameworks.

### Exposing Agent via A2A

```go
package main

import (
    "context"
    "log"
    "net/http"

    "google.golang.org/adk/agent/llmagent"
    "google.golang.org/adk/server/adka2a"
)

func exposeA2AAgent() {
    ctx := context.Background()
    model := createModel()

    // Create agent to expose
    agent := llmagent.New(&llmagent.Config{
        Name:        "specialist_agent",
        Description: "Specialized agent for data analysis",
        Instruction: "Perform detailed data analysis",
        Model:       model,
    })

    // Create A2A server
    server := adka2a.NewServer(&adka2a.Config{
        Agent:       agent,
        Port:        8080,
        AuthEnabled: true,
        APIKey:      "your-secure-api-key",
    })

    // Start A2A server
    log.Println("A2A server listening on :8080")
    if err := server.ListenAndServe(ctx); err != nil && err != http.ErrServerClosed {
        log.Fatal(err)
    }
}
```

### Consuming Remote A2A Agent

```go
package main

import (
    "context"
    "log"

    "google.golang.org/adk/agent/remoteagent"
    "google.golang.org/adk/agent/llmagent"
    "google.golang.org/adk/tool/agenttool"
)

func consumeA2AAgent() {
    ctx := context.Background()
    model := createModel()

    // Connect to remote A2A agent
    remoteAgent, err := remoteagent.New(&remoteagent.Config{
        URL:    "https://specialist.example.com/a2a",
        APIKey: "remote-api-key",
    })
    if err != nil {
        log.Fatal(err)
    }

    // Use remote agent as a tool
    remoteTool := agenttool.New(remoteAgent)

    // Create orchestrator agent
    orchestrator := llmagent.New(&llmagent.Config{
        Name: "orchestrator",
        Instruction: "Delegate complex analysis to the specialist agent",
        Model:       model,
        Tools:       []tool.Tool{remoteTool},
    })

    response, err := orchestrator.Run(ctx, "Analyze this dataset")
    if err != nil {
        log.Fatal(err)
    }

    log.Printf("Analysis: %s\n", response.Text)
}
```

### Multi-Agent A2A System

```go
package main

import (
    "context"
    "log"

    "google.golang.org/adk/agent/llmagent"
    "google.golang.org/adk/agent/remoteagent"
    "google.golang.org/adk/tool/agenttool"
)

func multiAgentA2A() {
    ctx := context.Background()
    model := createModel()

    // Connect to multiple remote agents
    dataAgent, _ := remoteagent.New(&remoteagent.Config{
        URL: "https://data-agent.example.com/a2a",
    })

    analysisAgent, _ := remoteagent.New(&remoteagent.Config{
        URL: "https://analysis-agent.example.com/a2a",
    })

    reportAgent, _ := remoteagent.New(&remoteagent.Config{
        URL: "https://report-agent.example.com/a2a",
    })

    // Create coordinator with remote agents
    coordinator := llmagent.New(&llmagent.Config{
        Name: "coordinator",
        Instruction: `Coordinate multiple specialist agents:
        - data_agent: for data retrieval
        - analysis_agent: for analysis
        - report_agent: for report generation`,
        Model: model,
        Tools: []tool.Tool{
            agenttool.New(dataAgent),
            agenttool.New(analysisAgent),
            agenttool.New(reportAgent),
        },
    })

    response, err := coordinator.Run(ctx, "Generate quarterly report")
    if err != nil {
        log.Fatal(err)
    }

    log.Printf("Report: %s\n", response.Text)
}
```

---

## Go Idioms and Patterns

### Idiomatic Error Handling

```go
package main

import (
    "context"
    "errors"
    "fmt"
    "log"

    "google.golang.org/adk/agent"
)

// Custom error types
var (
    ErrAgentNotFound    = errors.New("agent not found")
    ErrInvalidInput     = errors.New("invalid input")
    ErrToolExecutionFailed = errors.New("tool execution failed")
)

// Wrapped errors
type AgentError struct {
    Agent string
    Op    string
    Err   error
}

func (e *AgentError) Error() string {
    return fmt.Sprintf("agent %s: %s: %v", e.Agent, e.Op, e.Err)
}

func (e *AgentError) Unwrap() error {
    return e.Err
}

// Error handling in agent execution
func runAgentSafely(ctx context.Context, ag agent.Agent, input string) (response *agent.Response, err error) {
    // Defer for panic recovery
    defer func() {
        if r := recover(); r != nil {
            err = fmt.Errorf("panic in agent %s: %v", ag.Name(), r)
        }
    }()

    // Validate input
    if input == "" {
        return nil, &AgentError{
            Agent: ag.Name(),
            Op:    "validate",
            Err:   ErrInvalidInput,
        }
    }

    // Execute with context
    response, err = ag.Run(ctx, input)
    if err != nil {
        return nil, &AgentError{
            Agent: ag.Name(),
            Op:    "run",
            Err:   err,
        }
    }

    return response, nil
}

// Using errors.Is and errors.As
func handleAgentError(err error) {
    if err == nil {
        return
    }

    // Check for specific error
    if errors.Is(err, ErrInvalidInput) {
        log.Println("Input validation failed")
        return
    }

    // Unwrap custom error
    var agentErr *AgentError
    if errors.As(err, &agentErr) {
        log.Printf("Agent %s failed during %s: %v",
            agentErr.Agent, agentErr.Op, agentErr.Err)
        return
    }

    log.Printf("Unknown error: %v", err)
}
```

### Context Handling

```go
package main

import (
    "context"
    "log"
    "time"

    "google.golang.org/adk/agent"
)

// Context with timeout
func runWithTimeout(ag agent.Agent, input string, timeout time.Duration) (*agent.Response, error) {
    ctx, cancel := context.WithTimeout(context.Background(), timeout)
    defer cancel()

    response, err := ag.Run(ctx, input)
    if err != nil {
        if ctx.Err() == context.DeadlineExceeded {
            return nil, fmt.Errorf("agent timed out after %v", timeout)
        }
        return nil, err
    }

    return response, nil
}

// Context with cancellation
func runWithCancel(ag agent.Agent, input string) (*agent.Response, context.CancelFunc, error) {
    ctx, cancel := context.WithCancel(context.Background())

    response, err := ag.Run(ctx, input)
    return response, cancel, err
}

// Context with values
type contextKey string

const (
    userIDKey    contextKey = "user_id"
    requestIDKey contextKey = "request_id"
    sessionIDKey contextKey = "session_id"
)

func runWithMetadata(ag agent.Agent, input string, userID, requestID string) (*agent.Response, error) {
    ctx := context.Background()
    ctx = context.WithValue(ctx, userIDKey, userID)
    ctx = context.WithValue(ctx, requestIDKey, requestID)

    return ag.Run(ctx, input)
}

// Extract context values in tools
func toolWithContext(ctx context.Context) (interface{}, error) {
    userID, ok := ctx.Value(userIDKey).(string)
    if !ok {
        return nil, errors.New("user ID not found in context")
    }

    log.Printf("Tool called by user: %s", userID)

    // Use userID for authorization, logging, etc.
    return processForUser(userID)
}
```

### Goroutine Patterns

```go
package main

import (
    "context"
    "fmt"
    "sync"

    "google.golang.org/adk/agent"
)

// Concurrent agent execution with WaitGroup
func runAgentsConcurrently(ctx context.Context, agents []agent.Agent, input string) ([]*agent.Response, error) {
    var wg sync.WaitGroup
    responses := make([]*agent.Response, len(agents))
    errors := make([]error, len(agents))

    for i, ag := range agents {
        wg.Add(1)
        go func(index int, agent agent.Agent) {
            defer wg.Done()
            resp, err := agent.Run(ctx, input)
            responses[index] = resp
            errors[index] = err
        }(i, ag)
    }

    wg.Wait()

    // Check for errors
    for _, err := range errors {
        if err != nil {
            return nil, fmt.Errorf("agent execution failed: %w", err)
        }
    }

    return responses, nil
}

// Worker pool pattern
func workerPool(ctx context.Context, ag agent.Agent, inputs <-chan string, results chan<- *agent.Response, numWorkers int) {
    var wg sync.WaitGroup

    // Start workers
    for i := 0; i < numWorkers; i++ {
        wg.Add(1)
        go func(workerID int) {
            defer wg.Done()
            for input := range inputs {
                select {
                case <-ctx.Done():
                    return
                default:
                    resp, err := ag.Run(ctx, input)
                    if err != nil {
                        log.Printf("Worker %d error: %v", workerID, err)
                        continue
                    }
                    results <- resp
                }
            }
        }(i)
    }

    wg.Wait()
    close(results)
}

// Fan-out, fan-in pattern
func fanOutFanIn(ctx context.Context, agents []agent.Agent, input string) (*agent.Response, error) {
    // Fan-out: distribute to multiple agents
    responses := make(chan *agent.Response, len(agents))
    errors := make(chan error, len(agents))

    for _, ag := range agents {
        go func(agent agent.Agent) {
            resp, err := agent.Run(ctx, input)
            if err != nil {
                errors <- err
                return
            }
            responses <- resp
        }(ag)
    }

    // Fan-in: collect results
    var collectedResponses []*agent.Response
    for i := 0; i < len(agents); i++ {
        select {
        case resp := <-responses:
            collectedResponses = append(collectedResponses, resp)
        case err := <-errors:
            return nil, err
        case <-ctx.Done():
            return nil, ctx.Err()
        }
    }

    // Aggregate responses
    return aggregateResponses(collectedResponses)
}

// Pipeline pattern
func pipeline(ctx context.Context, stages ...func(context.Context, string) (string, error)) func(string) (string, error) {
    return func(input string) (string, error) {
        var err error
        result := input

        for _, stage := range stages {
            select {
            case <-ctx.Done():
                return "", ctx.Err()
            default:
                result, err = stage(ctx, result)
                if err != nil {
                    return "", err
                }
            }
        }

        return result, nil
    }
}
```

### Interface-Based Design

```go
package main

import (
    "context"
)

// Define custom interfaces
type Analyzer interface {
    Analyze(ctx context.Context, data string) (*Analysis, error)
}

type Validator interface {
    Validate(ctx context.Context, input string) error
}

type Enricher interface {
    Enrich(ctx context.Context, data string) (string, error)
}

// Compose interfaces
type SmartAgent interface {
    agent.Agent
    Analyzer
    Validator
}

// Implementation
type customAgent struct {
    *llmagent.Agent
}

func (a *customAgent) Analyze(ctx context.Context, data string) (*Analysis, error) {
    response, err := a.Run(ctx, "Analyze: "+data)
    if err != nil {
        return nil, err
    }

    return parseAnalysis(response.Text)
}

func (a *customAgent) Validate(ctx context.Context, input string) error {
    if len(input) == 0 {
        return ErrInvalidInput
    }

    response, err := a.Run(ctx, "Validate: "+input)
    if err != nil {
        return err
    }

    return checkValidation(response.Text)
}

// Use interface composition
func processWithInterfaces(ctx context.Context, ag SmartAgent, input string) error {
    // Validate
    if err := ag.Validate(ctx, input); err != nil {
        return fmt.Errorf("validation failed: %w", err)
    }

    // Analyze
    analysis, err := ag.Analyze(ctx, input)
    if err != nil {
        return fmt.Errorf("analysis failed: %w", err)
    }

    log.Printf("Analysis: %+v", analysis)
    return nil
}
```

---

## Session Management

### Basic Session Usage

```go
package main

import (
    "context"
    "log"

    "google.golang.org/adk/agent"
    "google.golang.org/adk/session"
)

func sessionExample() {
    ctx := context.Background()
    ag := createAgent()

    // Create session
    sess := session.New(&session.Config{
        ID:     "user-123-conv-456",
        UserID: "user-123",
        Metadata: map[string]interface{}{
            "source": "web",
            "lang":   "en",
        },
    })

    // Multi-turn conversation
    turn1, err := ag.RunWithSession(ctx, sess, "Hello, who are you?")
    if err != nil {
        log.Fatal(err)
    }
    log.Printf("Turn 1: %s\n", turn1.Text)

    turn2, err := ag.RunWithSession(ctx, sess, "What did I just ask you?")
    if err != nil {
        log.Fatal(err)
    }
    log.Printf("Turn 2: %s\n", turn2.Text)

    // Session maintains history
    log.Printf("History length: %d\n", len(sess.History))
}
```

### Persistent Sessions

```go
package main

import (
    "context"
    "log"

    "google.golang.org/adk/session"
    "google.golang.org/adk/session/database"
)

func persistentSessionExample() {
    ctx := context.Background()
    ag := createAgent()

    // Create database session store
    store, err := database.New(ctx, &database.Config{
        Type:       "firestore",
        ProjectID:  "my-project",
        Collection: "agent_sessions",
    })
    if err != nil {
        log.Fatal(err)
    }

    sessionID := "user-123-conv-456"

    // Try to load existing session
    sess, err := store.Load(ctx, sessionID)
    if err != nil {
        // Create new session if not found
        sess = session.New(&session.Config{
            ID:     sessionID,
            UserID: "user-123",
        })
    }

    // Run agent with session
    response, err := ag.RunWithSession(ctx, sess, "Continue our conversation")
    if err != nil {
        log.Fatal(err)
    }

    // Save updated session
    if err := store.Save(ctx, sess); err != nil {
        log.Fatal(err)
    }

    log.Printf("Response: %s\n", response.Text)
}
```

### Session State Management

```go
package main

import (
    "context"
    "log"

    "google.golang.org/adk/session"
)

func sessionStateExample() {
    ctx := context.Background()
    ag := createAgent()

    sess := session.New(&session.Config{
        ID: "game-session-123",
    })

    // Initialize state
    sess.State["player_name"] = "Alice"
    sess.State["level"] = 1
    sess.State["score"] = 0
    sess.State["inventory"] = []string{"sword", "shield"}

    // Use state in conversation
    response1, _ := ag.RunWithSession(ctx, sess, "What's my current level?")
    log.Printf("Response: %s\n", response1.Text)

    // Update state
    sess.State["score"] = 100
    sess.State["level"] = 2

    response2, _ := ag.RunWithSession(ctx, sess, "Did I level up?")
    log.Printf("Response: %s\n", response2.Text)
}
```

---

## Testing

### Unit Testing Agents

```go
package main

import (
    "context"
    "testing"

    "google.golang.org/adk/agent/llmagent"
    "google.golang.org/adk/model/gemini"
)

func TestAgent(t *testing.T) {
    ctx := context.Background()

    model, err := gemini.New(ctx, &gemini.Config{
        APIKey: "test-key",
        Model:  "gemini-2.0-flash",
    })
    if err != nil {
        t.Fatal(err)
    }

    agent := llmagent.New(&llmagent.Config{
        Name:        "test_agent",
        Instruction: "You are a test assistant",
        Model:       model,
    })

    tests := []struct {
        name    string
        input   string
        wantErr bool
    }{
        {
            name:    "simple query",
            input:   "Hello",
            wantErr: false,
        },
        {
            name:    "empty input",
            input:   "",
            wantErr: true,
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            resp, err := agent.Run(ctx, tt.input)
            if (err != nil) != tt.wantErr {
                t.Errorf("Run() error = %v, wantErr %v", err, tt.wantErr)
                return
            }
            if !tt.wantErr && resp == nil {
                t.Error("Expected response, got nil")
            }
        })
    }
}

func TestToolExecution(t *testing.T) {
    ctx := context.Background()

    called := false
    testTool := functiontool.New(func(ctx context.Context) string {
        called = true
        return "success"
    }, &functiontool.Config{
        Name: "test_tool",
    })

    agent := llmagent.New(&llmagent.Config{
        Name:  "tool_tester",
        Model: createMockModel(),
        Tools: []tool.Tool{testTool},
    })

    _, err := agent.Run(ctx, "Use the test tool")
    if err != nil {
        t.Fatal(err)
    }

    if !called {
        t.Error("Tool was not called")
    }
}
```

### Mock Model for Testing

```go
package main

import (
    "context"

    "google.golang.org/adk/model"
)

type mockModel struct {
    responses []string
    index     int
}

func createMockModel(responses ...string) model.LLM {
    return &mockModel{
        responses: responses,
        index:     0,
    }
}

func (m *mockModel) GenerateContent(ctx context.Context, req *model.GenerateRequest) (*model.GenerateResponse, error) {
    if m.index >= len(m.responses) {
        return &model.GenerateResponse{
            Text: "Default response",
        }, nil
    }

    resp := &model.GenerateResponse{
        Text: m.responses[m.index],
    }
    m.index++

    return resp, nil
}

func (m *mockModel) StreamContent(ctx context.Context, req *model.GenerateRequest) (<-chan *model.StreamChunk, error) {
    ch := make(chan *model.StreamChunk)
    go func() {
        defer close(ch)
        ch <- &model.StreamChunk{
            Text: m.responses[0],
            Done: true,
        }
    }()
    return ch, nil
}

func (m *mockModel) CountTokens(ctx context.Context, input string) (int, error) {
    return len(input) / 4, nil // rough estimate
}
```

---

## Best Practices

### 1. Context Propagation

Always pass context through the call chain:

```go
func goodExample(ctx context.Context) error {
    agent := createAgent()
    response, err := agent.Run(ctx, "input")
    // Process response
    return err
}

// Don't create new contexts unnecessarily
func badExample() error {
    ctx := context.Background() // Creates context in function
    // ...
}
```

### 2. Error Handling

Use explicit error handling:

```go
response, err := agent.Run(ctx, input)
if err != nil {
    // Handle or wrap error
    return fmt.Errorf("agent execution failed: %w", err)
}
```

### 3. Resource Cleanup

Use defer for cleanup:

```go
func processWithCleanup(ctx context.Context) error {
    sess, err := store.Load(ctx, sessionID)
    if err != nil {
        return err
    }
    defer store.Save(ctx, sess) // Always save

    // Process with session
    return nil
}
```

### 4. Configuration

Use struct configs, not variadic options when possible:

```go
// Good: explicit configuration
agent := llmagent.New(&llmagent.Config{
    Name:        "agent",
    Model:       model,
    Temperature: &temp,
})

// Avoid: long parameter lists
```

### 5. Concurrent Execution

Use goroutines with proper synchronization:

```go
var wg sync.WaitGroup
responses := make(chan *agent.Response, len(agents))

for _, ag := range agents {
    wg.Add(1)
    go func(a agent.Agent) {
        defer wg.Done()
        resp, _ := a.Run(ctx, input)
        responses <- resp
    }(ag)
}

wg.Wait()
close(responses)
```

---

**This comprehensive guide covers the complete ADK for Go API. For production deployment, see [google_adk_go_production_guide.md](./google_adk_go_production_guide.md). For copy-paste examples, see [google_adk_go_recipes.md](./google_adk_go_recipes.md).**
