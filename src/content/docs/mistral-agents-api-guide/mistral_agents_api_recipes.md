---
title: "Mistral Agents API: Ready-to-Use Recipes"
description: "Copy-and-adapt Python recipes for Mistral's Agents API: web search, persistent chat, multi-agent routing, function tools, RAG, streaming, branching, error handling and a small CLI app."
framework: mistral-agents-api
---

# Mistral Agents API: Ready-to-Use Recipes

Verified against `mistralai` %%version=mistralai%% (checked %%versions-checked%%). Install with `pip install "mistralai[agents]==%%version=mistralai%%"` and set `MISTRAL_API_KEY`.

Every recipe uses this shared module:

```python
# common.py
import os
from mistralai.client import Mistral
from mistralai.client.models import ConversationResponse, MessageOutputEntry, TextChunk

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

def text_of(response: ConversationResponse) -> str:
    """Text of the last message in a conversation response."""
    for entry in reversed(response.outputs):
        if isinstance(entry, MessageOutputEntry):
            if isinstance(entry.content, str):
                return entry.content
            return "".join(c.text for c in entry.content if isinstance(c, TextChunk))
    return ""
```

---

## 1. Web search agent

```python
from common import client, text_of

agent = client.beta.agents.create(
    model="mistral-medium-latest",
    name="web-researcher",
    description="Answers questions with current information from the web.",
    instructions="Search before answering anything time-sensitive and cite your sources.",
    tools=[{"type": "web_search"}],
    completion_args={"temperature": 0.3, "max_tokens": 1024},
)

conversation = client.beta.conversations.start(agent_id=agent.id, inputs="What are this week's main AI policy developments in the EU?")
print(text_of(conversation))

follow_up = client.beta.conversations.append(
    conversation_id=conversation.conversation_id,
    inputs="Tell me more about the second point.",
)
print(text_of(follow_up))

history = client.beta.conversations.get_history(conversation_id=conversation.conversation_id)
print(f"{len(history.entries)} entries:", [e.type for e in history.entries])
```

---

## 2. Persistent chatbot

Conversations live on Mistral's side, so a chatbot only has to remember one ID per user.

```python
import json
from pathlib import Path
from common import client, text_of

STORE = Path("conversations.json")

def load() -> dict[str, str]:
    return json.loads(STORE.read_text()) if STORE.exists() else {}

def chat(agent_id: str, user_id: str, message: str) -> str:
    sessions = load()
    if user_id in sessions:
        response = client.beta.conversations.append(conversation_id=sessions[user_id], inputs=message)
    else:
        response = client.beta.conversations.start(agent_id=agent_id, inputs=message, metadata={"user_id": user_id})
        sessions[user_id] = response.conversation_id
        STORE.write_text(json.dumps(sessions))
    return text_of(response)

def forget(user_id: str) -> None:
    sessions = load()
    if conversation_id := sessions.pop(user_id, None):
        client.beta.conversations.delete(conversation_id=conversation_id)
        STORE.write_text(json.dumps(sessions))

bot = client.beta.agents.create(model="mistral-small-latest", name="concierge", instructions="You are a friendly hotel concierge.")
print(chat(bot.id, "u-42", "Hi, I'm arriving on Friday with two children."))
print(chat(bot.id, "u-42", "What did I say about my arrival?"))
```

---

## 3. Multi-agent routing with handoffs

```python
from common import client, text_of

maths = client.beta.agents.create(
    model="mistral-medium-latest", name="maths",
    description="Solves quantitative problems with Python.",
    instructions="Use the code interpreter for every calculation.",
    tools=[{"type": "code_interpreter"}],
)
news = client.beta.agents.create(
    model="mistral-medium-latest", name="news",
    description="Answers questions about current events.",
    instructions="Search the web and cite sources.",
    tools=[{"type": "web_search"}],
)
router = client.beta.agents.create(
    model="mistral-small-latest", name="router",
    description="Routes questions to the right specialist.",
    instructions="Hand off to maths for calculations and to news for current events. Do not answer yourself.",
    handoffs=[maths.id, news.id],
)

response = client.beta.conversations.start(
    agent_id=router.id,
    inputs="If the ECB's current deposit rate held for 5 years, what would 10,000 EUR grow to?",
    handoff_execution="server",
)
print([e.next_agent_name for e in response.outputs if e.type == "agent.handoff"])
print(text_of(response))
```

More patterns, including code-driven pipelines, are in the [orchestration guide](/mistral-agents-api-guide/mistral_agents_api_orchestration_guide/).

---

## 4. Custom function tool

`RunContext` turns a typed, documented Python function into a tool and runs the loop for you.

```python
import asyncio
import os
from mistralai.client import Mistral
from mistralai.extra.run.context import RunContext

STOCK = {"SKU-1": 14, "SKU-2": 0}

def check_stock(sku: str) -> dict:
    """Return the number of units in stock for a SKU.

    Args:
        sku: Product code, e.g. SKU-1.
    """
    return {"sku": sku, "units": STOCK.get(sku, 0)}

async def main() -> None:
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    async with RunContext(model="mistral-small-latest") as run_ctx:
        run_ctx.register_func(check_stock)
        result = await client.beta.conversations.run_async(
            run_ctx=run_ctx,
            inputs="Can I order 3 of SKU-1 and 1 of SKU-2 today?",
            instructions="Check stock before promising anything.",
        )
        print(result.output_as_text)

asyncio.run(main())
```

The manual equivalent (declare a `function` tool, answer `function.call` entries with `FunctionResultEntry`) is in the [comprehensive guide](/mistral-agents-api-guide/mistral_agents_api_comprehensive_guide/#8-function-calling).

---

## 5. RAG over your documents

```python
import time
from pathlib import Path
from mistralai.client.models import File
from common import client, text_of

library = client.beta.libraries.create(name="hr-policies", description="Employee handbook and HR policies")
docs = [
    client.beta.libraries.documents.upload(
        library_id=library.id, file=File(file_name=p.name, content=p.read_bytes())
    )
    for p in Path("policies").glob("*.pdf")
]

# Wait until processing has finished
pending = {d.id for d in docs}
while pending:
    for doc_id in list(pending):
        status = client.beta.libraries.documents.status(library_id=library.id, document_id=doc_id)
        if status.process_status in ("done", "error", "missing_content"):
            pending.discard(doc_id)
    time.sleep(2)

hr = client.beta.agents.create(
    model="mistral-medium-latest",
    name="hr-assistant",
    description="Answers HR policy questions from the handbook.",
    instructions="Answer only from the handbook and name the policy you used. If it is not covered, say so.",
    tools=[{"type": "document_library", "library_ids": [library.id]}],
)
print(text_of(client.beta.conversations.start(agent_id=hr.id, inputs="How many days of parental leave do we offer?")))
```

`process_status` moves through `todo` and `in_progress` to `done` (or `error`). The older `processing_status` string field is deprecated.

---

## 6. Streaming responses

```python
from common import client

def stream_answer(agent_id: str, question: str) -> str:
    parts: list[str] = []
    with client.beta.conversations.start_stream(agent_id=agent_id, inputs=question) as stream:
        for event in stream:
            data = event.data
            if data.type == "message.output.delta" and isinstance(data.content, str):
                parts.append(data.content)
                print(data.content, end="", flush=True)
            elif data.type == "tool.execution.started":
                print(f"\n[{data.name}…]", flush=True)
            elif data.type == "conversation.response.error":
                raise RuntimeError(data.message)
    print()
    return "".join(parts)
```

---

## 7. Branching a conversation

```python
from common import client, text_of

def branch_from_first_message(conversation_id: str, new_input: str) -> str:
    history = client.beta.conversations.get_history(conversation_id=conversation_id)
    first_input = next(e for e in history.entries if e.type == "message.input")
    branch = client.beta.conversations.restart(
        conversation_id=conversation_id,
        from_entry_id=first_input.id,
        inputs=new_input,
    )
    print("new branch:", branch.conversation_id)
    return text_of(branch)
```

Both conversations remain; use this for "try a different approach" buttons or A/B comparisons.

---

## 8. Error handling

```python
import time
from mistralai.client import errors
from common import client, text_of

def start_with_retry(agent_id: str, inputs: str, attempts: int = 4):
    for attempt in range(1, attempts + 1):
        try:
            return client.beta.conversations.start(agent_id=agent_id, inputs=inputs)
        except errors.HTTPValidationError as e:
            raise ValueError(f"invalid request: {e.data}") from e            # fix the request; do not retry
        except errors.MistralError as e:
            if e.status_code in (401, 403, 404):
                raise                                                         # configuration problem
            if attempt == attempts or e.status_code not in (429, 500, 502, 503, 504):
                raise
            retry_after = float(e.headers.get("retry-after", 2 ** attempt))
            time.sleep(min(retry_after, 30))

def safe_append(conversation_id: str, inputs: str) -> str | None:
    try:
        return text_of(client.beta.conversations.append(conversation_id=conversation_id, inputs=inputs))
    except errors.MistralError as e:
        if e.status_code == 404:
            return None          # conversation deleted or never stored (store=False)
        raise
```

The client also accepts a `RetryConfig` for automatic retries; see the [production guide](/mistral-agents-api-guide/mistral_agents_api_production_guide/#2-client-concurrency-and-retries).

---

## 9. A small CLI application

```python
from common import client, text_of

def main() -> None:
    agent = client.beta.agents.create(
        model="mistral-medium-latest",
        name="cli-assistant",
        instructions="You are a concise assistant.",
        tools=[{"type": "web_search"}],
    )
    conversation_id: str | None = None
    print("Commands: /new, /list, /exit")
    while True:
        line = input("> ").strip()
        if line == "/exit":
            break
        if line == "/new":
            conversation_id = None
            print("new conversation")
            continue
        if line == "/list":
            for conv in client.beta.conversations.list(page_size=10):
                print(" ", conv.id, conv.created_at)
            continue
        if conversation_id is None:
            response = client.beta.conversations.start(agent_id=agent.id, inputs=line)
            conversation_id = response.conversation_id
        else:
            response = client.beta.conversations.append(conversation_id=conversation_id, inputs=line)
        print(text_of(response))

if __name__ == "__main__":
    main()
```
