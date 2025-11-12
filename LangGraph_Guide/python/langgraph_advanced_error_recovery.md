---
layout: default
title: LangGraph Advanced Error Handling and Recovery (Python)
description: Patterns for retries, timeouts, compensation, and safe state recovery in LangGraph Python.
---

# LangGraph Advanced Error Handling and Recovery (Python)

Last verified: 2025-11 • Source: https://github.com/langchain-ai/langgraph

## Goals
- Add per-node retries/backoff and timeouts
- Implement compensating actions and dead-letter nodes
- Ensure resumability with persisted state and checkpoints

## Per-Node Policies

```python
from langgraph.graph import StateGraph, END
from tenacity import retry, stop_after_attempt, wait_exponential

class State(dict):
    pass

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=8))
def fetch(state: State) -> State:
    # transient IO; may fail
    return {**state, "data": "ok"}

def transform(state: State) -> State:
    # pure; safe to retry
    return {**state, "result": state["data"].upper()}

def compensate(state: State) -> State:
    # undo side effects if any
    return {**state, "compensated": True}

def build_graph():
    g = StateGraph(State)
    g.add_node("fetch", fetch)
    g.add_node("transform", transform)
    g.add_node("compensate", compensate)
    g.set_entry_point("fetch")
    g.add_edge("fetch", "transform")
    g.add_edge("transform", END)
    return g
```

## Error Routes and Dead Letters

```python
from langgraph.errors import GraphRunError

def runner(inputs: State):
    g = build_graph()
    app = g.compile()
    try:
        return app.invoke(inputs)
    except Exception as e:
        # dead-letter path
        return {"error": str(e), "dead_letter": True}
```

## Checkpointing and Resumption

Persist state between nodes (e.g., to a DB). On restart, resume from last successful node.

## Telemetry

- Emit metrics per node: attempts, duration, failures
- Correlate with request ID through the graph

