# Google ADK for Go - Code Recipes

**Copy-paste ready examples for common patterns**

Version: 0.1.0
Last Updated: November 2025

---

## Table of Contents

1. [Hello World](#hello-world)
2. [Basic Agents](#basic-agents)
3. [Tools](#tools)
4. [Multi-Agent Systems](#multi-agent-systems)
5. [Sessions and State](#sessions-and-state)
6. [Streaming](#streaming)
7. [MCP Integration](#mcp-integration)
8. [A2A Protocol](#a2a-protocol)
9. [Production Patterns](#production-patterns)
10. [Cloud Integration](#cloud-integration)

---

## Hello World

### Simplest Agent

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

    // Create model
    model, err := gemini.New(ctx, &gemini.Config{
        APIKey: os.Getenv("GOOGLE_API_KEY"),
        Model:  "gemini-2.0-flash",
    })
    if err != nil {
        log.Fatal(err)
    }

    // Create agent
    agent := llmagent.New(&llmagent.Config{
        Name:  "hello_agent",
        Model: model,
    })

    // Run agent
    response, err := agent.Run(ctx, "Hello! Who are you?")
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println(response.Text)
}
```

---

## Basic Agents

### Q&A Bot with Tools

```go
package main

import (
    "context"
    "log"
    "os"
    "time"

    "google.golang.org/adk/agent/llmagent"
    "google.golang.org/adk/model/gemini"
    "google.golang.org/adk/tool/functiontool"
    "google.golang.org/adk/tool/geminitool"
)

func getCurrentTime(ctx context.Context, timezone string) (string, error) {
    loc, err := time.LoadLocation(timezone)
    if err != nil {
        return "", err
    }
    return time.Now().In(loc).Format(time.RFC3339), nil
}

func main() {
    ctx := context.Background()

    model, _ := gemini.New(ctx, &gemini.Config{
        APIKey: os.Getenv("GOOGLE_API_KEY"),
        Model:  "gemini-2.0-flash",
    })

    // Create tools
    timeTool := functiontool.New(getCurrentTime, &functiontool.Config{
        Name:        "get_time",
        Description: "Get current time for a timezone",
    })

    // Create agent with tools
    agent := llmagent.New(&llmagent.Config{
        Name:        "qa_bot",
        Description: "Helpful Q&A assistant",
        Instruction: `You are a helpful assistant. You can:
        - Answer general questions
        - Search the web for information
        - Tell the current time in any timezone`,
        Model: model,
        Tools: []tool.Tool{
            timeTool,
            geminitool.GoogleSearch(),
        },
    })

    // Interactive loop
    queries := []string{
        "What time is it in Tokyo?",
        "What's the weather like there?",
        "What are the top tech news today?",
    }

    for _, query := range queries {
        log.Printf("Q: %s\n", query)
        response, err := agent.Run(ctx, query)
        if err != nil {
            log.Printf("Error: %v\n", err)
            continue
        }
        log.Printf("A: %s\n\n", response.Text)
    }
}
```

### Structured Output Agent

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "log"

    "google.golang.org/adk/agent/llmagent"
)

type ProductAnalysis struct {
    Name        string   `json:"name"`
    Category    string   `json:"category"`
    Pros        []string `json:"pros"`
    Cons        []string `json:"cons"`
    Rating      float64  `json:"rating"`
    Recommended bool     `json:"recommended"`
}

func main() {
    ctx := context.Background()
    model := createModel()

    schema := &llmagent.Schema{
        Type: "object",
        Properties: map[string]*llmagent.Property{
            "name":        {Type: "string", Description: "Product name"},
            "category":    {Type: "string", Description: "Product category"},
            "pros":        {Type: "array", Items: &llmagent.Property{Type: "string"}},
            "cons":        {Type: "array", Items: &llmagent.Property{Type: "string"}},
            "rating":      {Type: "number", Description: "Rating 1-10"},
            "recommended": {Type: "boolean", Description: "Is recommended"},
        },
        Required: []string{"name", "category", "rating"},
    }

    agent := llmagent.New(&llmagent.Config{
        Name:         "product_analyzer",
        Instruction:  "Analyze products and provide structured ratings",
        Model:        model,
        OutputSchema: schema,
    })

    response, err := agent.Run(ctx, "Analyze the iPhone 15 Pro")
    if err != nil {
        log.Fatal(err)
    }

    var analysis ProductAnalysis
    if err := json.Unmarshal([]byte(response.Text), &analysis); err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Product: %s\n", analysis.Name)
    fmt.Printf("Category: %s\n", analysis.Category)
    fmt.Printf("Rating: %.1f/10\n", analysis.Rating)
    fmt.Printf("Recommended: %v\n", analysis.Recommended)
    fmt.Printf("Pros: %v\n", analysis.Pros)
    fmt.Printf("Cons: %v\n", analysis.Cons)
}

func createModel() model.LLM {
    ctx := context.Background()
    m, _ := gemini.New(ctx, &gemini.Config{
        APIKey: os.Getenv("GOOGLE_API_KEY"),
        Model:  "gemini-2.0-flash",
    })
    return m
}
```

---

## Tools

### Function Tool - Weather API

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "net/http"

    "google.golang.org/adk/tool/functiontool"
)

type WeatherData struct {
    Temperature float64 `json:"temperature"`
    Conditions  string  `json:"conditions"`
    Humidity    int     `json:"humidity"`
}

func getWeather(ctx context.Context, city string, units string) (*WeatherData, error) {
    // Call weather API (example)
    url := fmt.Sprintf("https://api.weather.com/v1/current?city=%s&units=%s", city, units)

    req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
    if err != nil {
        return nil, err
    }

    client := &http.Client{}
    resp, err := client.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()

    var weather WeatherData
    if err := json.NewDecoder(resp.Body).Decode(&weather); err != nil {
        return nil, err
    }

    return &weather, nil
}

func main() {
    ctx := context.Background()
    model := createModel()

    weatherTool := functiontool.New(getWeather, &functiontool.Config{
        Name:        "get_weather",
        Description: "Get current weather for a city",
        Parameters: map[string]*functiontool.Parameter{
            "city": {
                Type:        "string",
                Description: "City name",
                Required:    true,
            },
            "units": {
                Type:        "string",
                Description: "Temperature units: celsius or fahrenheit",
                Required:    false,
                Default:     "celsius",
                Enum:        []string{"celsius", "fahrenheit"},
            },
        },
    })

    agent := llmagent.New(&llmagent.Config{
        Name:        "weather_agent",
        Instruction: "Help users check weather conditions",
        Model:       model,
        Tools:       []tool.Tool{weatherTool},
    })

    response, _ := agent.Run(ctx, "What's the weather in San Francisco?")
    fmt.Println(response.Text)
}
```

### Database Tool

```go
package main

import (
    "context"
    "database/sql"
    "fmt"

    _ "github.com/lib/pq"
    "google.golang.org/adk/tool"
)

type DatabaseTool struct {
    db *sql.DB
}

func NewDatabaseTool(connString string) (*DatabaseTool, error) {
    db, err := sql.Open("postgres", connString)
    if err != nil {
        return nil, err
    }

    return &DatabaseTool{db: db}, nil
}

func (dt *DatabaseTool) Name() string {
    return "query_database"
}

func (dt *DatabaseTool) Description() string {
    return "Execute SQL queries on the database"
}

func (dt *DatabaseTool) Schema() *tool.Schema {
    return &tool.Schema{
        Type: "object",
        Properties: map[string]*tool.Property{
            "query": {
                Type:        "string",
                Description: "SQL SELECT query to execute",
            },
            "limit": {
                Type:        "integer",
                Description: "Maximum number of rows",
            },
        },
        Required: []string{"query"},
    }
}

func (dt *DatabaseTool) Execute(ctx context.Context, args map[string]interface{}) (interface{}, error) {
    query := args["query"].(string)
    limit := 100
    if l, ok := args["limit"].(float64); ok {
        limit = int(l)
    }

    rows, err := dt.db.QueryContext(ctx, query+" LIMIT $1", limit)
    if err != nil {
        return nil, fmt.Errorf("query failed: %w", err)
    }
    defer rows.Close()

    columns, err := rows.Columns()
    if err != nil {
        return nil, err
    }

    var results []map[string]interface{}
    for rows.Next() {
        values := make([]interface{}, len(columns))
        valuePtrs := make([]interface{}, len(columns))
        for i := range values {
            valuePtrs[i] = &values[i]
        }

        if err := rows.Scan(valuePtrs...); err != nil {
            return nil, err
        }

        row := make(map[string]interface{})
        for i, col := range columns {
            row[col] = values[i]
        }
        results = append(results, row)
    }

    return results, nil
}

func main() {
    ctx := context.Background()
    model := createModel()

    dbTool, err := NewDatabaseTool("postgres://user:pass@localhost/db")
    if err != nil {
        log.Fatal(err)
    }

    agent := llmagent.New(&llmagent.Config{
        Name:        "data_analyst",
        Instruction: "Analyze data by querying the database",
        Model:       model,
        Tools:       []tool.Tool{dbTool},
    })

    response, _ := agent.Run(ctx, "Show me the top 10 customers by revenue")
    fmt.Println(response.Text)
}
```

---

## Multi-Agent Systems

### Sequential Workflow

```go
package main

import (
    "context"
    "log"

    "google.golang.org/adk/agent/llmagent"
    "google.golang.org/adk/agent/sequentialagent"
)

func main() {
    ctx := context.Background()
    model := createModel()

    // Stage 1: Research
    researcher := llmagent.New(&llmagent.Config{
        Name:        "researcher",
        Instruction: "Research the topic and gather key facts. Output: bullet points of facts.",
        Model:       model,
        Tools:       []tool.Tool{geminitool.GoogleSearch()},
    })

    // Stage 2: Analyze
    analyzer := llmagent.New(&llmagent.Config{
        Name:        "analyzer",
        Instruction: "Analyze the research and identify patterns. Output: analysis summary.",
        Model:       model,
    })

    // Stage 3: Summarize
    summarizer := llmagent.New(&llmagent.Config{
        Name:        "summarizer",
        Instruction: "Create a concise executive summary. Output: 3-5 sentence summary.",
        Model:       model,
    })

    // Create sequential workflow
    workflow := sequentialagent.New(&sequentialagent.Config{
        Name:        "research_pipeline",
        Description: "Research, analyze, and summarize topics",
        Agents:      []agent.Agent{researcher, analyzer, summarizer},
    })

    // Execute
    response, err := workflow.Run(ctx, "Latest developments in quantum computing")
    if err != nil {
        log.Fatal(err)
    }

    log.Printf("Summary: %s\n", response.Text)
}
```

### Parallel Analysis

```go
package main

import (
    "context"
    "fmt"
    "strings"

    "google.golang.org/adk/agent/llmagent"
    "google.golang.org/adk/agent/parallelagent"
)

func main() {
    ctx := context.Background()
    model := createModel()

    // Create specialized analysts
    techAnalyst := llmagent.New(&llmagent.Config{
        Name:        "tech_analyst",
        Instruction: "Provide technical analysis focusing on technology and implementation",
        Model:       model,
    })

    businessAnalyst := llmagent.New(&llmagent.Config{
        Name:        "business_analyst",
        Instruction: "Provide business analysis focusing on market and revenue",
        Model:       model,
    })

    riskAnalyst := llmagent.New(&llmagent.Config{
        Name:        "risk_analyst",
        Instruction: "Provide risk analysis focusing on challenges and threats",
        Model:       model,
    })

    // Parallel execution
    parallel := parallelagent.New(&parallelagent.Config{
        Name:   "multi_perspective_analysis",
        Agents: []agent.Agent{techAnalyst, businessAnalyst, riskAnalyst},
        Aggregator: func(responses []*agent.Response) (*agent.Response, error) {
            var combined strings.Builder
            combined.WriteString("# Comprehensive Analysis\n\n")

            perspectives := []string{"Technical", "Business", "Risk"}
            for i, resp := range responses {
                combined.WriteString(fmt.Sprintf("## %s Perspective\n%s\n\n",
                    perspectives[i], resp.Text))
            }

            return &agent.Response{Text: combined.String()}, nil
        },
    })

    response, _ := parallel.Run(ctx, "Analyze launching a new AI product")
    fmt.Println(response.Text)
}
```

### Supervisor-Worker Pattern

```go
package main

import (
    "context"
    "log"

    "google.golang.org/adk/agent/llmagent"
    "google.golang.org/adk/tool/agenttool"
)

func main() {
    ctx := context.Background()
    model := createModel()

    // Worker agents
    coder := llmagent.New(&llmagent.Config{
        Name:        "coder",
        Instruction: "Write clean, efficient code. Only write code, no explanations.",
        Model:       model,
        Tools:       []tool.Tool{geminitool.CodeExecution()},
    })

    tester := llmagent.New(&llmagent.Config{
        Name:        "tester",
        Instruction: "Test code and report bugs. Provide test cases and results.",
        Model:       model,
        Tools:       []tool.Tool{geminitool.CodeExecution()},
    })

    documenter := llmagent.New(&llmagent.Config{
        Name:        "documenter",
        Instruction: "Write clear documentation and usage examples.",
        Model:       model,
    })

    // Supervisor
    supervisor := llmagent.New(&llmagent.Config{
        Name: "tech_lead",
        Instruction: `You are a tech lead managing a development team.
        Delegate tasks to your team members:
        - 'coder': for implementing features
        - 'tester': for testing code
        - 'documenter': for writing docs

        Coordinate their work to deliver complete solutions.`,
        Model: model,
        Tools: []tool.Tool{
            agenttool.New(coder),
            agenttool.New(tester),
            agenttool.New(documenter),
        },
    })

    response, err := supervisor.Run(ctx,
        "Create a function to validate email addresses, test it, and document it")
    if err != nil {
        log.Fatal(err)
    }

    log.Println(response.Text)
}
```

---

## Sessions and State

### Multi-Turn Conversation

```go
package main

import (
    "bufio"
    "context"
    "fmt"
    "os"
    "strings"

    "google.golang.org/adk/session"
)

func main() {
    ctx := context.Background()
    model := createModel()

    agent := llmagent.New(&llmagent.Config{
        Name:        "assistant",
        Instruction: "You are a helpful assistant. Remember context from previous messages.",
        Model:       model,
    })

    // Create session
    sess := session.New(&session.Config{
        ID:     "user-chat-001",
        UserID: "user-123",
    })

    // Interactive chat loop
    scanner := bufio.NewScanner(os.Stdin)
    fmt.Println("Chat with the assistant (type 'quit' to exit)")

    for {
        fmt.Print("\nYou: ")
        if !scanner.Scan() {
            break
        }

        input := strings.TrimSpace(scanner.Text())
        if input == "quit" {
            break
        }

        if input == "" {
            continue
        }

        response, err := agent.RunWithSession(ctx, sess, input)
        if err != nil {
            fmt.Printf("Error: %v\n", err)
            continue
        }

        fmt.Printf("Assistant: %s\n", response.Text)
    }
}
```

### Persistent Session with Firestore

```go
package main

import (
    "context"
    "log"

    "google.golang.org/adk/session"
    "google.golang.org/adk/session/database"
)

func main() {
    ctx := context.Background()
    model := createModel()

    agent := llmagent.New(&llmagent.Config{
        Name:  "persistent_agent",
        Model: model,
    })

    // Create Firestore session store
    store, err := database.New(ctx, &database.Config{
        Type:       "firestore",
        ProjectID:  os.Getenv("GOOGLE_CLOUD_PROJECT"),
        Collection: "agent_sessions",
    })
    if err != nil {
        log.Fatal(err)
    }

    userID := "user-123"
    sessionID := fmt.Sprintf("session-%s-%d", userID, time.Now().Unix())

    // Try to load existing session
    sess, err := store.Load(ctx, sessionID)
    if err != nil {
        // Create new session
        sess = session.New(&session.Config{
            ID:     sessionID,
            UserID: userID,
            Metadata: map[string]interface{}{
                "created_at": time.Now(),
                "platform":   "web",
            },
        })
    }

    // Run conversation
    messages := []string{
        "My name is Alice",
        "What's my name?",
        "What did we talk about?",
    }

    for _, msg := range messages {
        response, err := agent.RunWithSession(ctx, sess, msg)
        if err != nil {
            log.Fatal(err)
        }

        log.Printf("Q: %s\n", msg)
        log.Printf("A: %s\n\n", response.Text)

        // Save session after each turn
        if err := store.Save(ctx, sess); err != nil {
            log.Fatal(err)
        }
    }
}
```

---

## Streaming

### Stream Response Chunks

```go
package main

import (
    "context"
    "fmt"
    "log"
)

func main() {
    ctx := context.Background()
    model := createModel()

    agent := llmagent.New(&llmagent.Config{
        Name:  "streaming_agent",
        Model: model,
    })

    // Start streaming
    stream, err := agent.Stream(ctx, "Write a short story about a robot")
    if err != nil {
        log.Fatal(err)
    }

    // Process chunks
    for chunk := range stream {
        if chunk.Error != nil {
            log.Printf("Error: %v\n", chunk.Error)
            continue
        }

        if chunk.Delta != "" {
            fmt.Print(chunk.Delta)
        }

        if chunk.Done {
            fmt.Println("\n[Complete]")
            break
        }
    }
}
```

### HTTP Streaming Endpoint

```go
package main

import (
    "context"
    "encoding/json"
    "fmt"
    "net/http"
)

func streamHandler(agent agent.Agent) http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        var req struct {
            Message string `json:"message"`
        }

        if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
            http.Error(w, err.Error(), http.StatusBadRequest)
            return
        }

        // Set headers for SSE
        w.Header().Set("Content-Type", "text/event-stream")
        w.Header().Set("Cache-Control", "no-cache")
        w.Header().Set("Connection", "keep-alive")

        ctx := r.Context()
        stream, err := agent.Stream(ctx, req.Message)
        if err != nil {
            http.Error(w, err.Error(), http.StatusInternalServerError)
            return
        }

        flusher, ok := w.(http.Flusher)
        if !ok {
            http.Error(w, "Streaming not supported", http.StatusInternalServerError)
            return
        }

        for chunk := range stream {
            if chunk.Error != nil {
                fmt.Fprintf(w, "event: error\ndata: %s\n\n", chunk.Error)
                flusher.Flush()
                continue
            }

            data, _ := json.Marshal(map[string]interface{}{
                "delta": chunk.Delta,
                "done":  chunk.Done,
            })

            fmt.Fprintf(w, "data: %s\n\n", data)
            flusher.Flush()

            if chunk.Done {
                break
            }
        }
    }
}

func main() {
    agent := createAgent()
    http.HandleFunc("/stream", streamHandler(agent))
    log.Fatal(http.ListenAndServe(":8080", nil))
}
```

---

## MCP Integration

### Using MCP Tools

```go
package main

import (
    "context"
    "log"

    "google.golang.org/adk/agent/llmagent"
    "google.golang.org/adk/tool/mcptoolset"
)

func main() {
    ctx := context.Background()
    model := createModel()

    // Connect to MCP server
    mcpTools, err := mcptoolset.New(ctx, &mcptoolset.Config{
        ServerURL: "mcp://localhost:8080",
        Transport: "stdio",
    })
    if err != nil {
        log.Fatal(err)
    }

    // Create agent with MCP tools
    agent := llmagent.New(&llmagent.Config{
        Name:        "mcp_agent",
        Instruction: "Use available MCP tools to help users",
        Model:       model,
        Tools:       mcpTools.GetTools(),
    })

    response, err := agent.Run(ctx, "Query the customer database for recent orders")
    if err != nil {
        log.Fatal(err)
    }

    log.Println(response.Text)
}
```

---

## A2A Protocol

### Expose Agent via A2A

```go
package main

import (
    "context"
    "log"

    "google.golang.org/adk/agent/llmagent"
    "google.golang.org/adk/server/adka2a"
)

func main() {
    ctx := context.Background()
    model := createModel()

    // Create specialized agent
    agent := llmagent.New(&llmagent.Config{
        Name:        "data_analyst",
        Description: "Specialized agent for data analysis",
        Instruction: "Perform detailed data analysis and provide insights",
        Model:       model,
    })

    // Create A2A server
    server := adka2a.NewServer(&adka2a.Config{
        Agent:  agent,
        Port:   8080,
        APIKey: os.Getenv("A2A_API_KEY"),
    })

    log.Println("A2A agent server listening on :8080")
    if err := server.ListenAndServe(ctx); err != nil {
        log.Fatal(err)
    }
}
```

### Consume Remote A2A Agent

```go
package main

import (
    "context"
    "log"

    "google.golang.org/adk/agent/llmagent"
    "google.golang.org/adk/agent/remoteagent"
    "google.golang.org/adk/tool/agenttool"
)

func main() {
    ctx := context.Background()
    model := createModel()

    // Connect to remote agents
    dataAgent, err := remoteagent.New(&remoteagent.Config{
        URL:    "https://data-agent.example.com/a2a",
        APIKey: os.Getenv("DATA_AGENT_KEY"),
    })
    if err != nil {
        log.Fatal(err)
    }

    mlAgent, err := remoteagent.New(&remoteagent.Config{
        URL:    "https://ml-agent.example.com/a2a",
        APIKey: os.Getenv("ML_AGENT_KEY"),
    })
    if err != nil {
        log.Fatal(err)
    }

    // Create orchestrator
    orchestrator := llmagent.New(&llmagent.Config{
        Name: "orchestrator",
        Instruction: `Coordinate specialized agents:
        - data_analyst: for data retrieval and analysis
        - ml_specialist: for machine learning tasks`,
        Model: model,
        Tools: []tool.Tool{
            agenttool.New(dataAgent),
            agenttool.New(mlAgent),
        },
    })

    response, err := orchestrator.Run(ctx, "Analyze sales data and predict next quarter")
    if err != nil {
        log.Fatal(err)
    }

    log.Println(response.Text)
}
```

---

## Production Patterns

### Web Server with Agent

```go
package main

import (
    "context"
    "encoding/json"
    "log"
    "net/http"
    "time"

    "google.golang.org/adk/agent"
    "google.golang.org/adk/session"
)

type Server struct {
    agent agent.Agent
    store session.Store
}

type ChatRequest struct {
    SessionID string `json:"session_id"`
    Message   string `json:"message"`
}

type ChatResponse struct {
    Response  string `json:"response"`
    SessionID string `json:"session_id"`
}

func (s *Server) handleChat(w http.ResponseWriter, r *http.Request) {
    var req ChatRequest
    if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
        http.Error(w, err.Error(), http.StatusBadRequest)
        return
    }

    ctx := r.Context()

    // Load or create session
    sess, err := s.store.Load(ctx, req.SessionID)
    if err != nil {
        sess = session.New(&session.Config{
            ID: req.SessionID,
        })
    }

    // Run agent
    response, err := s.agent.RunWithSession(ctx, sess, req.Message)
    if err != nil {
        http.Error(w, err.Error(), http.StatusInternalServerError)
        return
    }

    // Save session
    if err := s.store.Save(ctx, sess); err != nil {
        log.Printf("Failed to save session: %v", err)
    }

    // Return response
    json.NewEncoder(w).Encode(ChatResponse{
        Response:  response.Text,
        SessionID: sess.ID,
    })
}

func main() {
    ctx := context.Background()
    agent := createAgent()

    store, _ := database.New(ctx, &database.Config{
        Type:      "firestore",
        ProjectID: os.Getenv("GOOGLE_CLOUD_PROJECT"),
    })

    server := &Server{
        agent: agent,
        store: store,
    }

    http.HandleFunc("/chat", server.handleChat)
    http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
        w.WriteHeader(http.StatusOK)
        w.Write([]byte("OK"))
    })

    log.Println("Server starting on :8080")
    log.Fatal(http.ListenAndServe(":8080", nil))
}
```

### Background Worker

```go
package main

import (
    "context"
    "log"
    "time"

    "cloud.google.com/go/pubsub"
)

type Worker struct {
    agent        agent.Agent
    subscription *pubsub.Subscription
}

func (w *Worker) Start(ctx context.Context) error {
    return w.subscription.Receive(ctx, func(ctx context.Context, msg *pubsub.Message) {
        log.Printf("Received message: %s", msg.ID)

        // Process with agent
        response, err := w.agent.Run(ctx, string(msg.Data))
        if err != nil {
            log.Printf("Agent error: %v", err)
            msg.Nack()
            return
        }

        // Store result
        if err := storeResult(ctx, msg.ID, response); err != nil {
            log.Printf("Storage error: %v", err)
            msg.Nack()
            return
        }

        msg.Ack()
        log.Printf("Processed message: %s", msg.ID)
    })
}

func main() {
    ctx := context.Background()
    agent := createAgent()

    client, err := pubsub.NewClient(ctx, os.Getenv("GOOGLE_CLOUD_PROJECT"))
    if err != nil {
        log.Fatal(err)
    }
    defer client.Close()

    worker := &Worker{
        agent:        agent,
        subscription: client.Subscription("agent-tasks"),
    }

    log.Println("Worker starting...")
    if err := worker.Start(ctx); err != nil {
        log.Fatal(err)
    }
}
```

---

## Cloud Integration

### BigQuery Integration

```go
package main

import (
    "context"
    "fmt"

    "cloud.google.com/go/bigquery"
    "google.golang.org/adk/tool"
)

type BigQueryTool struct {
    client *bigquery.Client
}

func NewBigQueryTool(ctx context.Context, projectID string) (*BigQueryTool, error) {
    client, err := bigquery.NewClient(ctx, projectID)
    if err != nil {
        return nil, err
    }

    return &BigQueryTool{client: client}, nil
}

func (bq *BigQueryTool) Name() string {
    return "query_bigquery"
}

func (bq *BigQueryTool) Description() string {
    return "Execute BigQuery SQL queries"
}

func (bq *BigQueryTool) Schema() *tool.Schema {
    return &tool.Schema{
        Type: "object",
        Properties: map[string]*tool.Property{
            "query": {Type: "string", Description: "SQL query"},
        },
        Required: []string{"query"},
    }
}

func (bq *BigQueryTool) Execute(ctx context.Context, args map[string]interface{}) (interface{}, error) {
    query := args["query"].(string)

    q := bq.client.Query(query)
    it, err := q.Read(ctx)
    if err != nil {
        return nil, err
    }

    var rows []map[string]bigquery.Value
    for {
        var row map[string]bigquery.Value
        err := it.Next(&row)
        if err == iterator.Done {
            break
        }
        if err != nil {
            return nil, err
        }
        rows = append(rows, row)
    }

    return rows, nil
}

func main() {
    ctx := context.Background()
    model := createModel()

    bqTool, err := NewBigQueryTool(ctx, os.Getenv("GOOGLE_CLOUD_PROJECT"))
    if err != nil {
        log.Fatal(err)
    }

    agent := llmagent.New(&llmagent.Config{
        Name:        "data_agent",
        Instruction: "Analyze data in BigQuery",
        Model:       model,
        Tools:       []tool.Tool{bqTool},
    })

    response, _ := agent.Run(ctx, "What were total sales last month?")
    fmt.Println(response.Text)
}
```

### Cloud Storage Agent

```go
package main

import (
    "context"
    "fmt"
    "io"

    "cloud.google.com/go/storage"
    "google.golang.org/adk/tool/functiontool"
)

func listFiles(ctx context.Context, bucket string, prefix string) ([]string, error) {
    client, err := storage.NewClient(ctx)
    if err != nil {
        return nil, err
    }
    defer client.Close()

    var files []string
    it := client.Bucket(bucket).Objects(ctx, &storage.Query{Prefix: prefix})

    for {
        attrs, err := it.Next()
        if err == iterator.Done {
            break
        }
        if err != nil {
            return nil, err
        }
        files = append(files, attrs.Name)
    }

    return files, nil
}

func readFile(ctx context.Context, bucket string, object string) (string, error) {
    client, err := storage.NewClient(ctx)
    if err != nil {
        return "", err
    }
    defer client.Close()

    rc, err := client.Bucket(bucket).Object(object).NewReader(ctx)
    if err != nil {
        return "", err
    }
    defer rc.Close()

    data, err := io.ReadAll(rc)
    if err != nil {
        return "", err
    }

    return string(data), nil
}

func main() {
    ctx := context.Background()
    model := createModel()

    listTool := functiontool.New(listFiles, &functiontool.Config{
        Name:        "list_files",
        Description: "List files in Cloud Storage bucket",
    })

    readTool := functiontool.New(readFile, &functiontool.Config{
        Name:        "read_file",
        Description: "Read file contents from Cloud Storage",
    })

    agent := llmagent.New(&llmagent.Config{
        Name:        "storage_agent",
        Instruction: "Help users manage Cloud Storage files",
        Model:       model,
        Tools:       []tool.Tool{listTool, readTool},
    })

    response, _ := agent.Run(ctx, "List all CSV files in my-bucket")
    fmt.Println(response.Text)
}
```

---

**These recipes provide ready-to-use code for common ADK patterns. Modify them for your specific needs. For complete API reference, see [google_adk_go_comprehensive_guide.md](./google_adk_go_comprehensive_guide.md).**
