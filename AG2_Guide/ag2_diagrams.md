# AG2 (AutoGen) Architecture Diagrams

**Version:** 0.3.2
**Last Updated:** November 2025
**Focus:** Visualizing AG2 Systems

## Overview

This document provides visual representations of AG2 architectures and workflows.

## Agent Interaction Flow

```mermaid
sequenceDiagram
    participant User
    participant UserProxy
    participant Assistant
    participant Tool

    User->>UserProxy: Task Request
    UserProxy->>Assistant: Forward Request
    Assistant->>Assistant: Think / Plan
    Assistant->>Tool: Execute Tool (if needed)
    Tool-->>Assistant: Tool Output
    Assistant->>UserProxy: Response
    UserProxy->>User: Final Answer
```

## Group Chat Architecture

```mermaid
graph TD
    User[User] --> Manager[GroupChatManager]
    Manager --> AgentA[Agent A]
    Manager --> AgentB[Agent B]
    Manager --> AgentC[Agent C]
    AgentA --> Manager
    AgentB --> Manager
    AgentC --> Manager
```

## RAG (Retrieval-Augmented Generation) Flow

```mermaid
graph LR
    Query[User Query] --> Retriever[Retriever Agent]
    Retriever --> DB[(Vector DB)]
    DB --> Context[Retrieved Context]
    Context --> Generator[Generator Agent]
    Generator --> Answer[Final Answer]
```
