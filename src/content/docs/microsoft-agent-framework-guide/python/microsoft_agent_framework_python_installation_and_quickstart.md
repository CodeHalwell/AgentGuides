---
title: "Azure AI Agents SDK (Python) — Installation & Quickstart"
description: "Get started with azure-ai-agents 1.1.0: installation, authentication, your first agent, and the complete thread-run-message lifecycle."
framework: microsoft-agent-framework
language: python
---

# Azure AI Agents SDK (Python) — Installation & Quickstart

This guide covers everything you need to get your first Azure AI agent running in Python using the `azure-ai-agents` library (version 1.1.0).

---

## Package Information

| Property | Value |
|---|---|
| Package name | `azure-ai-agents` |
| Current version | `1.1.0` |
| PyPI | [https://pypi.org/project/azure-ai-agents/](https://pypi.org/project/azure-ai-agents/) |
| Python requirement | 3.9 or later |

```bash
pip install azure-ai-agents
pip install azure-identity
```

The `azure-identity` package provides credential classes (`DefaultAzureCredential`, `AzureCliCredential`, `ManagedIdentityCredential`) used to authenticate against Azure.

> **Note:** While `azure-ai-agents` can be used independently, Microsoft also recommends the companion package `azure-ai-projects`, which provides simplified access to agents alongside other Azure AI Foundry capabilities such as model management, datasets, search indexes, and evaluation.

---

## Architecture Overview

The `azure-ai-agents` SDK is the Python client for the Azure AI Agents service. Every interaction follows the same core lifecycle:

1. **Create an `AgentsClient`** — your authenticated gateway to the service, constructed with your project endpoint and a credential object.
2. **Create an `Agent`** — a persisted resource in Azure that holds your model choice and system instructions.
3. **Create an `AgentThread`** — a conversation session. Threads are server-side, so history is managed automatically.
4. **Add `ThreadMessage`s** — user messages posted to the thread.
5. **Create a `ThreadRun`** — instructs the agent to process the messages on a given thread.
6. **Poll for completion** — either manually (`runs.create` + polling loop) or automatically (`runs.create_and_process`).
7. **Retrieve messages** — read the agent's responses from the thread.
8. **Clean up** — delete the agent and thread when you are done.

This design means that conversation history lives on Azure rather than in your application's memory, which makes it straightforward to resume sessions across processes.

---

## Environment Variables

Before running any code, set your project endpoint as an environment variable. You can find the endpoint string in the **overview** page of your Azure AI Foundry project, under **Libraries > Foundry**.

The format is:

```
https://<aiservices-id>.services.ai.azure.com/api/projects/<project-name>
```

Set it in your shell or `.env` file:

```bash
AZURE_AI_AGENTS_ENDPOINT=https://myresource.services.ai.azure.com/api/projects/myproject
MODEL_DEPLOYMENT_NAME=gpt-4o
```

> **Foundry project endpoint only:** As of May 2025, the Azure AI Agent Service uses a Foundry project endpoint. The older hub-based connection string format is not supported by SDK version 1.1.0.

---

## Authentication

The SDK requires a credential object that implements the Azure `TokenCredential` interface. The right choice depends on your environment:

| Scenario | Credential class |
|---|---|
| Local development (Azure CLI) | `AzureCliCredential` |
| Local development (any available method) | `DefaultAzureCredential` |
| Production (Azure-hosted workload) | `ManagedIdentityCredential` |
| Production (general) | `DefaultAzureCredential` |

```python
import os
from azure.identity import DefaultAzureCredential
from azure.ai.agents import AgentsClient

endpoint = os.environ["AZURE_AI_AGENTS_ENDPOINT"]
credential = DefaultAzureCredential()

client = AgentsClient(endpoint=endpoint, credential=credential)
```

`DefaultAzureCredential` tries several authentication methods in order — Azure CLI, environment variables, managed identity, and so on — which makes it a good default for both local and deployed scenarios.

### Prerequisites for local development

1. Install the [Azure CLI](https://learn.microsoft.com/cli/azure/install-azure-cli).
2. Run `az login` to authenticate.
3. If you have multiple Azure subscriptions, ensure the one containing your AI Foundry project is the default: `az account set --subscription "Your Subscription ID or Name"`.
4. Assign yourself an appropriate role on the Azure AI Project resource (see [Role-based access control in Azure AI Foundry](https://learn.microsoft.com/azure/ai-foundry/concepts/rbac-ai-foundry)).

---

## Complete Quickstart — Simple Agent (No Tools)

The following example shows the complete lifecycle from creating an agent through to reading its response and cleaning up.

```python
import os
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import ListSortOrder
from azure.identity import DefaultAzureCredential

# 1. Initialise the client
client = AgentsClient(
    endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

# 2. Create an agent (persisted in Azure)
agent = client.create_agent(
    model=os.environ["MODEL_DEPLOYMENT_NAME"],  # e.g. "gpt-4o"
    name="my-assistant",
    instructions="You are a helpful assistant. Answer concisely and clearly.",
)
print(f"Created agent, ID: {agent.id}")

# 3. Create a conversation thread
thread = client.threads.create()
print(f"Created thread, ID: {thread.id}")

# 4. Post a user message to the thread
message = client.messages.create(
    thread_id=thread.id,
    role="user",
    content="What is the capital of France?",
)
print(f"Created message, ID: {message.id}")

# 5. Run the agent on the thread (create_and_process polls until completion)
run = client.runs.create_and_process(
    thread_id=thread.id,
    agent_id=agent.id,
)
print(f"Run finished with status: {run.status}")

if run.status == "failed":
    print(f"Run failed: {run.last_error}")

# 6. Retrieve and print all messages in chronological order
messages = client.messages.list(
    thread_id=thread.id,
    order=ListSortOrder.ASCENDING,
)
for msg in messages:
    if msg.text_messages:
        last_text = msg.text_messages[-1]
        print(f"{msg.role}: {last_text.text.value}")

# 7. Clean up
client.delete_agent(agent.id)
print("Deleted agent")

client.threads.delete(thread.id)
print("Deleted thread")
```

### What each step does

- **`client.create_agent(...)`** — registers a new agent definition in Azure. The agent persists until you explicitly delete it, so you can reuse the same agent across multiple threads and sessions.
- **`client.threads.create()`** — creates a new conversation session. Each thread maintains its own message history server-side.
- **`client.messages.create(...)`** — adds a message to the thread. The `role` parameter must be `"user"` for human messages.
- **`client.runs.create_and_process(...)`** — submits the thread to the agent for processing and blocks until the run reaches a terminal state (`completed`, `failed`, `cancelled`, or `expired`). The SDK handles all polling internally.
- **`client.messages.list(..., order=ListSortOrder.ASCENDING)`** — retrieves all messages on the thread ordered oldest-first, so you can read the conversation naturally from top to bottom.
- **`msg.text_messages[-1].text.value`** — accesses the final text content of a message. Messages can contain multiple content blocks (text, images, file citations); `text_messages` is a helper property that filters to text-only blocks. You may also access content directly via `msg.content[0].text.value`, though this will raise an `IndexError` if the message has no content items, so the `text_messages` guard is generally safer.

---

## One-Shot Convenience Method

If you want to create a thread and run the agent in a single call, use `create_thread_and_process_run`. This is useful for simple single-turn interactions where you do not need to reuse the thread.

```python
import os
from azure.ai.agents import AgentsClient
from azure.identity import DefaultAzureCredential

client = AgentsClient(
    endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
    credential=DefaultAzureCredential(),
)

agent = client.create_agent(
    model=os.environ["MODEL_DEPLOYMENT_NAME"],
    name="quick-agent",
    instructions="You are a helpful assistant.",
)

# Creates a thread, posts the message, runs the agent, and returns the run — all in one call
run = client.create_thread_and_process_run(
    agent_id=agent.id,
    thread={
        "messages": [
            {"role": "user", "content": "Give me a one-sentence summary of the Python programming language."}
        ]
    },
)
print(f"Run status: {run.status}")

client.delete_agent(agent.id)
```

---

## Using the Context Manager (Synchronous)

The `AgentsClient` supports Python's context manager protocol, which ensures the underlying HTTP session is closed cleanly even if an exception occurs.

```python
import os
from azure.ai.agents import AgentsClient
from azure.ai.agents.models import ListSortOrder
from azure.identity import DefaultAzureCredential

with AgentsClient(
    endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
    credential=DefaultAzureCredential(),
) as client:
    agent = client.create_agent(
        model=os.environ["MODEL_DEPLOYMENT_NAME"],
        name="context-manager-agent",
        instructions="You are a helpful assistant.",
    )

    thread = client.threads.create()

    client.messages.create(
        thread_id=thread.id,
        role="user",
        content="Hello! What can you help me with?",
    )

    run = client.runs.create_and_process(
        thread_id=thread.id,
        agent_id=agent.id,
    )

    messages = client.messages.list(
        thread_id=thread.id,
        order=ListSortOrder.ASCENDING,
    )
    for msg in messages:
        if msg.text_messages:
            print(f"{msg.role}: {msg.text_messages[-1].text.value}")

    client.delete_agent(agent.id)
# HTTP session is automatically closed here
```

---

## Async Pattern

For asynchronous applications (FastAPI, async scripts, and so on), import `AgentsClient` from `azure.ai.agents.aio` instead. You will also need the `aiohttp` package:

```bash
pip install aiohttp
```

```python
import os
import asyncio
from azure.ai.agents.aio import AgentsClient
from azure.ai.agents.models import ListSortOrder
from azure.identity.aio import DefaultAzureCredential


async def main() -> None:
    async with AgentsClient(
        endpoint=os.environ["AZURE_AI_AGENTS_ENDPOINT"],
        credential=DefaultAzureCredential(),
    ) as client:
        agent = await client.create_agent(
            model=os.environ["MODEL_DEPLOYMENT_NAME"],
            name="async-agent",
            instructions="You are a helpful assistant.",
        )

        thread = await client.threads.create()

        await client.messages.create(
            thread_id=thread.id,
            role="user",
            content="What are the three laws of thermodynamics?",
        )

        run = await client.runs.create_and_process(
            thread_id=thread.id,
            agent_id=agent.id,
        )
        print(f"Run status: {run.status}")

        messages = client.messages.list(
            thread_id=thread.id,
            order=ListSortOrder.ASCENDING,
        )
        async for msg in messages:
            if msg.text_messages:
                print(f"{msg.role}: {msg.text_messages[-1].text.value}")

        await client.delete_agent(agent.id)


asyncio.run(main())
```

Key differences from the synchronous pattern:

- Import from `azure.ai.agents.aio`, not `azure.ai.agents`.
- Use `azure.identity.aio.DefaultAzureCredential` (the async variant).
- All client method calls are `await`-ed.
- Iteration over message lists uses `async for`.
- The context manager uses `async with`.

---

## Manual Polling (Alternative to `create_and_process`)

If you need fine-grained control over the polling loop — for example, to log progress or implement a timeout — you can use `runs.create` and poll manually:

```python
import time
from azure.ai.agents.models import RunStatus

run = client.runs.create(thread_id=thread.id, agent_id=agent.id)

while run.status in [RunStatus.QUEUED, RunStatus.IN_PROGRESS, RunStatus.REQUIRES_ACTION]:
    time.sleep(1)
    run = client.runs.get(thread_id=thread.id, run_id=run.id)
    print(f"Run status: {run.status}")

print(f"Run completed with status: {run.status}")
```

> Note: when using function tools without `enable_auto_function_calls`, the `REQUIRES_ACTION` status means your code must call the function and submit the result back to the SDK. See the [function tools documentation](https://learn.microsoft.com/python/api/overview/azure/ai-agents-readme?view=azure-python#create-agent-with-function-call) for details.

---

## Common Errors

### `ValueError: endpoint is required`

You have not set the `PROJECT_ENDPOINT` (or equivalent) environment variable, or it is empty. Double-check that the variable is exported in your shell session and matches the format `https://<aiservices-id>.services.ai.azure.com/api/projects/<project-name>`.

### `azure.core.exceptions.ClientAuthenticationError`

Your credential does not have permission to access the Azure AI Project resource. Common causes:

- You are not logged in: run `az login`.
- The wrong Azure subscription is active: run `az account set --subscription "<ID>"`.
- Your account or managed identity has not been assigned the required role on the project resource. Check **Access Control (IAM)** in the Azure portal.

### `azure.core.exceptions.ResourceNotFoundError` on an agent ID

The agent ID you are passing does not exist in the service — it may have been deleted, or you may be pointing at the wrong project endpoint. Verify the endpoint and re-create the agent if necessary.

### Run status `"failed"`

Inspect `run.last_error` for details:

```python
if run.status == "failed":
    print(f"Error code: {run.last_error.code}")
    print(f"Error message: {run.last_error.message}")
```

Common causes include model capacity limits, content policy violations, or invalid tool configurations.

---

## Next Steps

- [Function tools](https://learn.microsoft.com/python/api/overview/azure/ai-agents-readme?view=azure-python#create-agent-with-function-call) — give your agent access to Python functions.
- [File search](https://learn.microsoft.com/python/api/overview/azure/ai-agents-readme?view=azure-python#create-agent-with-file-search) — upload documents and let the agent search them.
- [Code interpreter](https://learn.microsoft.com/python/api/overview/azure/ai-agents-readme?view=azure-python#create-agent-with-code-interpreter) — let the agent write and execute Python to answer questions.
- [Streaming](https://learn.microsoft.com/python/api/overview/azure/ai-agents-readme?view=azure-python#execute-run-run_and_process-or-stream) — receive token-by-token output as the agent generates its response.
- [Tracing with Azure Monitor](https://learn.microsoft.com/python/api/overview/azure/ai-agents-readme?view=azure-python#tracing) — observe the full execution path through Application Insights.
- [API reference](https://aka.ms/azsdk/azure-ai-agents/python/reference)
- [SDK samples](https://aka.ms/azsdk/azure-ai-projects/python/samples/)
