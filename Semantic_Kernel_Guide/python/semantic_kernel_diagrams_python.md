# Semantic Kernel Diagrams (Python)

**Visual Architecture and Flow Diagrams for Python Implementation**

Last Updated: November 2025

---

## Overview

This document provides Python-specific architecture diagrams and visual representations. For general SK architecture diagrams, see [../semantic_kernel_diagrams.md](../semantic_kernel_diagrams.md).

---

## Python Request Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                       FastAPI Application                        │
│  ┌────────────┐  ┌──────────────┐  ┌─────────────────────────┐  │
│  │  Endpoint  │→ │  Middleware  │→ │ Resilient Kernel Invoke │  │
│  └────────────┘  └──────────────┘  └─────────────────────────┘  │
└──────────────────────────│──────────────────────────────────────┘
                           ↓
            ┌──────────────────────────────────┐
            │      Semantic Kernel (Python)    │
            │  ┌────────┐    ┌──────────────┐  │
            │  │ Kernel │ →  │ AI Service   │  │
            │  └────────┘    │ (OpenAI/Az)  │  │
            │      ↓          └──────────────┘  │
            │  ┌────────┐    ┌──────────────┐  │
            │  │Plugins │    │   Memory     │  │
            │  └────────┘    └──────────────┘  │
            └─────────────────┬────────────────┘
                              ↓
                    ┌──────────────────┐
                    │ OpenTelemetry    │
                    │ (Azure Monitor)  │
                    └──────────────────┘
```

---

## Async Execution Pattern

```
User Request
     │
     ├─→ async def handler()
     │        │
     │        ├─→ await kernel.invoke(func1)  ──┐
     │        ├─→ await kernel.invoke(func2)  ──┼─→ asyncio.gather()
     │        └─→ await kernel.invoke(func3)  ──┘        │
     │                                                     ↓
     │                                            Parallel Execution
     │                                                     │
     └─────────────────────────────────────────←──────────┘
                                               Combined Results
```

---

## Docker/Kubernetes Deployment

```
┌───────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                      │
│                                                            │
│  ┌──────────────────────────────────────────────────┐     │
│  │              Load Balancer Service                │     │
│  └──────────────────┬───────────────────────────────┘     │
│                     │                                      │
│         ┌───────────┼───────────┐                          │
│         ↓           ↓           ↓                          │
│    ┌────────┐  ┌────────┐  ┌────────┐                     │
│    │ Pod 1  │  │ Pod 2  │  │ Pod 3  │  ← HPA Autoscaling  │
│    │        │  │        │  │        │                     │
│    │ Python │  │ Python │  │ Python │                     │
│    │  App   │  │  App   │  │  App   │                     │
│    └───┬────┘  └───┬────┘  └───┬────┘                     │
│        │           │           │                          │
│        └───────────┼───────────┘                          │
│                    │                                      │
└────────────────────┼──────────────────────────────────────┘
                     │
         ┌───────────┼───────────────┐
         ↓           ↓               ↓
    ┌─────────┐ ┌──────────┐  ┌─────────────┐
    │ OpenAI  │ │ Key Vault│  │ App Insights│
    │  API    │ │ (Secrets)│  │ (Telemetry) │
    └─────────┘ └──────────┘  └─────────────┘
```

---

## MCP Architecture (2025)

```
┌──────────────────────────────────────────────┐
│        Semantic Kernel Python App            │
│                                              │
│  ┌────────────────────────────────────────┐  │
│  │          MCP Client                    │  │
│  │  - list_tools()                        │  │
│  │  - call_tool(name, args)               │  │
│  │  - list_resources()                    │  │
│  │  - read_resource(uri)                  │  │
│  └──────────────┬─────────────────────────┘  │
└─────────────────┼────────────────────────────┘
                  │ HTTP/JSON-RPC
                  ↓
┌─────────────────────────────────────────────┐
│           MCP Server (Python/Node)          │
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │  Tools:                             │   │
│  │   - database_query                  │   │
│  │   - file_search                     │   │
│  │   - api_call                        │   │
│  └─────────────────────────────────────┘   │
│                                             │
│  ┌─────────────────────────────────────┐   │
│  │  Resources:                         │   │
│  │   - file:///data/docs.json          │   │
│  │   - db://customers                  │   │
│  └─────────────────────────────────────┘   │
└─────────────────────────────────────────────┘
```

---

## Multi-Agent Communication (A2A Protocol)

```
┌──────────────────┐        A2A Message Bus         ┌──────────────────┐
│  SK Agent (Py)   │◄──────────────────────────────►│ LangGraph Agent  │
│                  │                                │    (Python)      │
│  agent.sk.       │    ┌───────────────────┐       │  agent.lg.       │
│  researcher      │◄───┤  Message Router   ├──────►│  analyzer        │
│                  │    │  - Discovery      │       │                  │
└──────────────────┘    │  - Routing        │       └──────────────────┘
                        │  - Protocol       │
┌──────────────────┐    └───────┬───────────┘       ┌──────────────────┐
│  SK Agent (Py)   │◄───────────┼──────────────────►│ AutoGen Agent    │
│                  │            │                   │   (Python)       │
│  agent.sk.       │            ↓                   │  agent.ag.       │
│  writer          │     Event Broadcasting         │  critic          │
└──────────────────┘                                └──────────────────┘

Message Format:
{
  "id": "msg_123",
  "type": "REQUEST",
  "sender": "agent.sk.researcher",
  "recipient": "agent.lg.analyzer",
  "payload": {"content": "Analyze this data"},
  "correlation_id": null
}
```

---

## Monitoring Stack

```
┌─────────────────────────────────────────────────────┐
│           Python SK Application                     │
│                                                     │
│  ┌────────────────────────────────────────────┐    │
│  │  OpenTelemetry SDK                         │    │
│  │  - Traces (spans)                          │    │
│  │  - Metrics (counters, histograms)          │    │
│  │  - Logs (structured)                       │    │
│  └──────────────┬─────────────────────────────┘    │
└─────────────────┼──────────────────────────────────┘
                  │
                  ↓
    ┌─────────────────────────────────┐
    │  Azure Monitor Exporter         │
    │  - Batch processing             │
    │  - Retry logic                  │
    └──────────────┬──────────────────┘
                   │
                   ↓
    ┌──────────────────────────────────────────┐
    │     Azure Application Insights          │
    │                                          │
    │  ┌────────────┐  ┌──────────────────┐   │
    │  │  Traces    │  │    Metrics       │   │
    │  │  Timeline  │  │    Dashboards    │   │
    │  └────────────┘  └──────────────────┘   │
    │                                          │
    │  ┌────────────┐  ┌──────────────────┐   │
    │  │   Logs     │  │     Alerts       │   │
    │  │   Search   │  │    Notifications │   │
    │  └────────────┘  └──────────────────┘   │
    └──────────────────────────────────────────┘
```

---

For more diagrams, see [../semantic_kernel_diagrams.md](../semantic_kernel_diagrams.md)

**[Back to Python README](README.md)** | **[Complete Index](GUIDE_INDEX.md)**
