---
title: "Haystack Recipes: Production-Ready Use Cases and Examples"
description: "Haystack 3 recipes — knowledge-base Q&A, multi-tenant support, document summarisation, live-data agents, multi-agent collaboration, research agents and a conversational code assistant."
framework: "haystack"
language: "python"
---

> Verified against `haystack-ai` %%version=haystack-ai%% (checked %%versions-checked%%). Each recipe runs as written with `OPENAI_API_KEY` set; to try one offline, replace `OpenAIChatGenerator(...)` with `MockChatGenerator()` from `haystack.components.generators.chat`.

## Table of Contents

1. [Knowledge Base Question Answering System](#knowledge-base-question-answering-system)
2. [Multi-Tenant Customer Support Agent](#multi-tenant-customer-support-agent)
3. [Document Analysis and Summarisation Pipeline](#document-analysis-and-summarisation-pipeline)
4. [Real-Time Data Retrieval Agent](#real-time-data-retrieval-agent)
5. [Multi-Agent Collaboration System](#multi-agent-collaboration-system)
6. [Autonomous Research Agent](#autonomous-research-agent)
7. [Conversational Code Assistant](#conversational-code-assistant)

---

## Knowledge Base Question Answering System

An indexing pipeline that cleans and splits documents (word-based splitting needs no extra dependencies; `split_by="sentence"` needs `pip install "nltk>=3.9.1"`), and a query pipeline that retrieves, prompts and returns a `GeneratedAnswer` with its source documents. The in-memory store keeps the recipe self-contained; the Elasticsearch variant is shown at the end.

```python
from haystack import Document, Pipeline
from haystack.components.builders import AnswerBuilder, ChatPromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.preprocessors import DocumentCleaner, DocumentSplitter
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.components.writers import DocumentWriter
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.document_stores.types import DuplicatePolicy

store = InMemoryDocumentStore()

# --- Indexing -------------------------------------------------------------
indexing = Pipeline()
indexing.add_component("cleaner", DocumentCleaner())
indexing.add_component("splitter", DocumentSplitter(split_by="word", split_length=150, split_overlap=20))
indexing.add_component("writer", DocumentWriter(document_store=store, policy=DuplicatePolicy.OVERWRITE))
indexing.connect("cleaner.documents", "splitter.documents")
indexing.connect("splitter.documents", "writer.documents")

indexing.run({"cleaner": {"documents": [
    Document(content="Our return policy allows returns within 30 days for a full refund.", meta={"source": "policies.md"}),
    Document(content="Standard shipping takes 5-7 business days.", meta={"source": "shipping.md"}),
    Document(content="All products carry a one-year warranty.", meta={"source": "warranty.md"}),
]}})

# --- Querying -------------------------------------------------------------
template = [
    ChatMessage.from_system(
        "Answer only from the documents. If they do not contain the answer, say so. "
        "Cite sources as [source]."
    ),
    ChatMessage.from_user(
        "{% for doc in documents %}[{{ doc.meta.source }}] {{ doc.content }}\n{% endfor %}\n"
        "Question: {{ question }}"
    ),
]

qa = Pipeline()
qa.add_component("retriever", InMemoryBM25Retriever(document_store=store, top_k=5))
qa.add_component("prompt", ChatPromptBuilder(template=template, required_variables=["question", "documents"]))
qa.add_component("llm", OpenAIChatGenerator(model="%%model=openai-small%%"))
qa.add_component("answer", AnswerBuilder())

qa.connect("retriever.documents", "prompt.documents")
qa.connect("prompt.prompt", "llm.messages")
qa.connect("llm.replies", "answer.replies")
qa.connect("retriever.documents", "answer.documents")


def ask(question: str) -> dict:
    result = qa.run({
        "retriever": {"query": question},
        "prompt": {"question": question},
        "answer": {"query": question},
    })
    answer = result["answer"]["answers"][0]
    return {"answer": answer.data, "sources": [d.meta.get("source") for d in answer.documents]}


print(ask("What's your return policy?"))
```

**Swapping in Elasticsearch** (`pip install elasticsearch-haystack`) changes only the store and retriever:

```python
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchBM25Retriever

store = ElasticsearchDocumentStore(hosts="http://localhost:9200", index="knowledge_base")
retriever = ElasticsearchBM25Retriever(document_store=store, top_k=5)
```

For semantic search, add `OpenAIDocumentEmbedder` before the writer, `OpenAITextEmbedder` before an embedding retriever, and join BM25 and embedding results with `DocumentJoiner(join_mode="reciprocal_rank_fusion")`.

---

## Multi-Tenant Customer Support Agent

One agent definition serves every tenant. The tenant ID is passed to `run()` as a state value, and each data-access tool reads it from a parameter typed as `State`. The Agent always injects that parameter itself, so the model can neither see nor replace the tenant ID.

```python
from typing import Annotated

from haystack import Document
from haystack.components.agents import Agent, State
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from haystack.dataclasses import ChatMessage
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.tools import tool

kb = InMemoryDocumentStore()
kb.write_documents([
    Document(content="Acme: refunds are processed within 3 days.", meta={"tenant_id": "acme"}),
    Document(content="Globex: refunds require a receipt.", meta={"tenant_id": "globex"}),
])
kb_retriever = InMemoryBM25Retriever(document_store=kb, top_k=3)

CUSTOMERS = {("acme", "C-1"): {"name": "Ada", "plan": "pro"}, ("globex", "C-9"): {"name": "Hank", "plan": "free"}}


@tool
def search_kb(query: Annotated[str, "What to look up"], state: State) -> str:
    """Search this company's support knowledge base."""
    filters = {"field": "meta.tenant_id", "operator": "==", "value": state.get("tenant_id")}
    docs = kb_retriever.run(query=query, filters=filters)["documents"]
    return "\n".join(d.content for d in docs) or "Nothing found."


@tool
def lookup_customer(customer_id: Annotated[str, "Customer ID"], state: State) -> dict:
    """Look up a customer record."""
    return CUSTOMERS.get((state.get("tenant_id"), customer_id), {"error": "Unknown customer"})


@tool
def create_ticket(summary: Annotated[str, "Issue summary"], state: State) -> str:
    """Open a support ticket."""
    return f"Ticket {state.get('tenant_id').upper()}-1042 created: {summary}"


support_agent = Agent(
    chat_generator=OpenAIChatGenerator(model="%%model=openai-small%%"),
    tools=[search_kb, lookup_customer, create_ticket],
    system_prompt="You are a support agent for {{ company_name }}. Be concise and polite.",
    state_schema={"tenant_id": {"type": str}},
    max_agent_steps=8,
)

TENANTS = {"acme": "Acme Ltd", "globex": "Globex Corp"}


def handle(tenant_id: str, text: str) -> str:
    if tenant_id not in TENANTS:
        raise ValueError("Unknown tenant")
    result = support_agent.run(
        messages=[ChatMessage.from_user(text)],
        tenant_id=tenant_id,                  # state value, invisible to the model
        company_name=TENANTS[tenant_id],      # system_prompt variable
    )
    return result["last_message"].text


print(handle("acme", "How long do refunds take? I'm customer C-1."))
```

A `State`-typed parameter is left out of the tool's JSON schema and is always filled by the Agent, even if the model sends an argument with the same name. If a prompt-injected request makes the model add `tenant_id` to a call, the tool fails with an unexpected-argument error (reported back to the model) instead of switching tenant. Apply the same filter in every data-access tool.

Do not use `inputs_from_state` for this. It is a convenience for filling arguments from state, not a security boundary: in Haystack 3, arguments the model supplies take precedence over the state lookup, so a tool declared with `inputs_from_state={"tenant_id": "tenant_id"}` will happily run against whatever tenant the model names.

---

## Document Analysis and Summarisation Pipeline

Convert files, split them into chunks, summarise each chunk with an `LLM` component, then combine the summaries (map-reduce).

```python
import tempfile
from pathlib import Path

from haystack import Pipeline
from haystack.components.converters import TextFileToDocument
from haystack.components.generators.chat import LLM, OpenAIChatGenerator
from haystack.components.preprocessors import DocumentSplitter


preprocess = Pipeline()
preprocess.add_component("convert", TextFileToDocument())
preprocess.add_component("split", DocumentSplitter(split_by="word", split_length=300, split_overlap=30))
preprocess.connect("convert.documents", "split.documents")

chunk_summariser = LLM(
    chat_generator=OpenAIChatGenerator(model="%%model=openai-small%%"),
    system_prompt="You summarise business documents accurately and briefly.",
    user_prompt="Summarise this excerpt in 2-3 bullet points:\n\n{{ text }}",
)
final_summariser = LLM(
    chat_generator=OpenAIChatGenerator(model="%%model=openai-small%%"),
    system_prompt="You write executive summaries.",
    user_prompt="Combine these partial summaries into one {{ length }} summary:\n\n{{ partials }}",
)


def summarise(path: str, length: str = "150-word") -> str:
    chunks = preprocess.run({"convert": {"sources": [path]}})["split"]["documents"]
    partials = [chunk_summariser.run(text=c.content)["last_message"].text for c in chunks]
    return final_summariser.run(partials="\n".join(partials), length=length)["last_message"].text


with tempfile.TemporaryDirectory() as tmp:
    report = Path(tmp) / "report.txt"
    report.write_text("Revenue grew 12% in Q3. Costs fell 3%. " * 50)
    print(summarise(str(report)))
```

`PyPDFToDocument`, `DOCXToDocument`, `HTMLToDocument`, `MarkdownToDocument` and `MultiFileConverter` (routes by file type) are drop-in replacements for the converter. For many chunks, call `await chunk_summariser.run_async(...)` concurrently with `asyncio.gather`.

---

## Real-Time Data Retrieval Agent

Live data comes from tools. `HTMLToDocument` needs `pip install trafilatura`. Here a small fetch-and-convert pipeline is exposed as a single tool with `PipelineTool`, alongside a function tool for prices.

```python
from typing import Annotated

from haystack import Pipeline
from haystack.components.agents import Agent
from haystack.components.converters import HTMLToDocument
from haystack.components.fetchers import LinkContentFetcher
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import PipelineTool, tool

fetch_pipe = Pipeline()
fetch_pipe.add_component("fetch", LinkContentFetcher(timeout=10))
fetch_pipe.add_component("to_doc", HTMLToDocument())
fetch_pipe.connect("fetch.streams", "to_doc.sources")

fetch_page = PipelineTool(
    pipeline=fetch_pipe,
    name="fetch_page",
    description="Fetch web pages and return their text.",
    input_mapping={"urls": ["fetch.urls"]},
    output_mapping={"to_doc.documents": "documents"},
    outputs_to_string={"source": "documents", "handler": lambda docs: "\n\n".join(d.content[:4000] for d in docs)},
)


@tool
def get_stock_price(symbol: Annotated[str, "Ticker symbol, e.g. MSFT"]) -> dict:
    """Get the latest stock price."""
    # Replace with a real market-data API call
    return {"symbol": symbol.upper(), "price": 412.30, "currency": "USD"}


live_agent = Agent(
    chat_generator=OpenAIChatGenerator(model="%%model=openai-small%%"),
    tools=[fetch_page, get_stock_price],
    system_prompt="Answer with up-to-date information from your tools and say where it came from.",
    max_agent_steps=6,
)

result = live_agent.run(messages=[ChatMessage.from_user("What is MSFT trading at?")])
print(result["last_message"].text)
```

For general web search, add `ComponentTool(component=SerperDevWebSearch(top_k=5), ...)` from `serperdev-haystack`.

---

## Multi-Agent Collaboration System

A complete report-writing app: a coordinator delegates to researcher, analyst, writer and critic agents exposed through `AgentTool`, and loops writer and critic until the draft is approved. The patterns it combines (coordinator, specialists, shared state, refinement) are explained in the [multi-agent guide](/haystack-guide/haystack_multi_agent_guide/).

```python
from typing import Annotated

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.tools import AgentTool, tool


@tool
def web_search(query: Annotated[str, "Search query"]) -> str:
    """Search the web."""
    return f"(stub) results for {query!r}"


def specialist(prompt: str, tools=None) -> Agent:
    return Agent(
        chat_generator=OpenAIChatGenerator(model="%%model=openai-small%%"),
        tools=tools,
        system_prompt=prompt,
        max_agent_steps=6,
    )


team = [
    AgentTool(agent=specialist("Research the task and list facts with sources.", [web_search]),
              name="researcher", description="Finds facts and sources on a topic."),
    AgentTool(agent=specialist("Analyse the facts you are given; identify trends and risks."),
              name="analyst", description="Analyses research findings."),
    AgentTool(agent=specialist("Write clear, structured prose from the material you are given."),
              name="writer", description="Drafts the report."),
    AgentTool(agent=specialist("Critique the draft; list concrete improvements, or reply APPROVED."),
              name="critic", description="Reviews a draft."),
]

coordinator = Agent(
    chat_generator=OpenAIChatGenerator(model="%%model=openai-small%%"),
    tools=team,
    system_prompt=(
        "You lead a small team. Use the researcher, then the analyst, then the writer. "
        "Send the draft to the critic and revise with the writer until the critic replies APPROVED "
        "(at most two revisions). Return the final report."
    ),
    max_agent_steps=15,
    tool_concurrency_limit=2,
)

result = coordinator.run(messages=[ChatMessage.from_user("Report on the European heat-pump market.")])
print(result["last_message"].text)
print(result["tool_call_counts"])
```

---

## Autonomous Research Agent

The agent accumulates findings in `State` and finishes by calling a `submit_report` tool, which is also its exit condition. An `on_exit` hook stops it from finishing with too few findings.

```python
from typing import Annotated

from haystack.components.agents import Agent, State
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.hooks import hook
from haystack.tools import tool


@tool
def search_papers(query: Annotated[str, "Search query"]) -> str:
    """Search academic databases."""
    return f"(stub) three abstracts about {query}"


@tool(outputs_to_state={"findings": {"source": "findings"}})
def record_finding(finding: Annotated[str, "One finding with its source"]) -> dict:
    """Record a key finding."""
    return {"findings": [finding]}


@tool(inputs_from_state={"findings": "findings"})
def submit_report(summary: Annotated[str, "Final synthesis"], findings: list[str]) -> dict:
    """Submit the final research report."""
    return {"summary": summary, "findings": findings}


@hook
def need_three_findings(state: State) -> None:
    if len(state.get("findings") or []) < 3 and state.get("step_count", 0) < 10:
        state.set("messages", [ChatMessage.from_user("Record at least three findings before submitting.")])
        state.set("continue_run", True)


researcher = Agent(
    chat_generator=OpenAIChatGenerator(model="%%model=openai-small%%"),
    tools=[search_papers, record_finding, submit_report],
    system_prompt="Research the question, record each finding, then call submit_report.",
    state_schema={"findings": {"type": list[str]}},
    exit_conditions=["submit_report"],
    hooks={"on_exit": [need_three_findings]},
    max_agent_steps=15,
)

result = researcher.run(messages=[ChatMessage.from_user("What limits solid-state battery adoption?")])
print(result.get("findings", []))
print(result["exit_reason"])   # "submit_report" when it finished normally
```

---

## Conversational Code Assistant

A code-review assistant with a syntax-check tool and per-session history. History is stored as JSON via `ChatMessage.to_dict()`, so the same code works with Redis or any key-value store.

```python
import ast
import json
from typing import Annotated, Protocol

from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage, ChatRole
from haystack.tools import tool


@tool
def check_syntax(code: Annotated[str, "Python source code"]) -> str:
    """Check Python code for syntax errors."""
    try:
        ast.parse(code)
        return "No syntax errors."
    except SyntaxError as exc:
        return f"SyntaxError on line {exc.lineno}: {exc.msg}"


class KV(Protocol):
    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str) -> None: ...


class DictKV(dict):
    """In-process stand-in; use redis.Redis(decode_responses=True) in production."""
    def set(self, key, value):
        self[key] = value


class CodeAssistant:
    def __init__(self, store: KV, max_messages: int = 30):
        self.store = store
        self.max_messages = max_messages
        self.agent = Agent(
            chat_generator=OpenAIChatGenerator(model="%%model=openai-small%%"),
            tools=[check_syntax],
            system_prompt="You are a senior Python reviewer. Check syntax before commenting on code.",
            max_agent_steps=5,
        )

    def _load(self, session: str) -> list[ChatMessage]:
        raw = self.store.get(f"chat:{session}")
        return [ChatMessage.from_dict(m) for m in json.loads(raw)] if raw else []

    def _save(self, session: str, messages: list[ChatMessage]) -> None:
        kept = [m for m in messages if not m.is_from(ChatRole.SYSTEM)][-self.max_messages:]
        self.store.set(f"chat:{session}", json.dumps([m.to_dict() for m in kept]))

    def chat(self, session: str, text: str) -> str:
        history = self._load(session) + [ChatMessage.from_user(text)]
        result = self.agent.run(messages=history)
        self._save(session, result["messages"])
        return result["last_message"].text


assistant = CodeAssistant(DictKV())
print(assistant.chat("s1", "Review this: def f(x) return x*2"))
print(assistant.chat("s1", "And how would you add type hints?"))
```

With Redis: `import redis; assistant = CodeAssistant(redis.Redis(decode_responses=True))`. Add an expiry (`store.set(key, value, ex=86400)`) if sessions should lapse.
