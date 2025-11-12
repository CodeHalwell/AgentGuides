---
layout: default
title: Semantic Kernel Comprehensive Guide (Python)
description: End-to-end Python guide for Microsoft Semantic Kernel agents, plugins, memory, planners, and production.
---

# Semantic Kernel Comprehensive Guide (Python)

Last verified: 2025-11
Source of truth: https://github.com/microsoft/semantic-kernel

## Overview
- Python-first, agent-focused reference for SK
- Covers agents, functions, plugins, memory, planning, observability, deployment

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install "semantic-kernel[openai,azure]" python-dotenv tenacity opentelemetry-sdk
```

## Quick Start: Kernel + Chat Service

```python
import os
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

kernel = sk.Kernel()
kernel.add_chat_service(
    "openai",
    OpenAIChatCompletion(model_id="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"]),
)
```

## Agents
- Use ChatCompletionAgent and AgentGroupChat for multi-agent workflows.

```python
from semantic_kernel.agents import ChatCompletionAgent, AgentGroupChat

researcher = ChatCompletionAgent(kernel=kernel, name="researcher", instructions="Research and cite.")
writer = ChatCompletionAgent(kernel=kernel, name="writer", instructions="Draft and refine copy.")

chat = AgentGroupChat(researcher, writer)
await chat.add_user_message("Draft a one-paragraph summary of LangGraph with sources")
result = await chat.invoke()
print(result)
```

## Plugins & Functions
- Semantic functions (prompt-based) and native functions (Python code) compose capabilities.

```python
calc_plugin = kernel.create_plugin("calc")

@calc_plugin.define_native_function
async def add(a: int, b: int) -> int:
    return a + b

semantic = kernel.create_function_from_prompt("Summarize: {{$input}} in 3 bullet points")
```

## Structured Output
- Ask models for JSON matching Pydantic models; validate and retry on failure.

```python
from pydantic import BaseModel

class Summary(BaseModel):
    bullets: list[str]

prompt = "Return a JSON object {\"bullets\": string[]} summarizing: {{$input}}"
fn = kernel.create_function_from_prompt(prompt)
raw = await kernel.invoke_async(fn, input_text="LangGraph overview")
data = Summary.model_validate_json(str(raw))
```

## Memory
- Use external vector stores (e.g., Azure AI Search, Qdrant) for retrieval.

## Planning
- Stepwise/Sequential planners (if available in your SK version) orchestrate multi-step tasks.

## Observability
- Add OpenTelemetry spans for function/agent boundaries; record tokens and latency.

## Error Handling
- Timeouts, retries (tenacity), and circuit breakers around tool calls.

## Deployment
- Containerize with uvicorn/Gunicorn; store secrets in Key Vault; enable tracing.

## Examples & Recipes
- See semantic_kernel_recipes.md for end-to-end examples.

