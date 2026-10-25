---
title: "Microsoft Agent Framework (Rust) — Context and memory"
description: "Context providers in agent-framework-rs: the ContextProvider trait and its run lifecycle, per-session memory, TextSearchProvider retrieval, compaction strategies and CompactionProvider, skills, the experimental agent harness (LoopAgent, judges, todos, modes, file memory) and the experimental vector store abstraction with vector collection tools."
framework: "microsoft-agent-framework"
language: "rust"
sidebar:
  label: "Context and memory"
  order: 7
---

Everything that adds to a run without being part of the caller's input goes through one trait, `ContextProvider`. History is a context provider (see [sessions](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_sessions/)), and so are retrieval, compaction, skills, long-term memory services and the harness's todo list and file memory. This page covers the trait, then each provider the core crate ships, then the experimental vector store abstraction. The memory services with their own backends (Redis, Mem0, Cosmos DB, Azure AI Search and Foundry memory) are on the [integrations page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_integrations/).

All examples were compiled against `agent-framework` %%version=agent-framework-rs%%. The ones that call OpenAI were run against a local stand-in for its API; the others use a small client that prints what the model would receive, which is the clearest way to see what a provider does. The harness and vector store need experimental features:

```toml
[dependencies]
agent-framework = { version = "%%version=agent-framework-rs%%", features = ["experimental-harness", "experimental-vector-stores"] }
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
async-trait = "0.1"
futures = "0.3"
```

## Context providers

```rust title="The trait, abridged"
#[async_trait]
pub trait ContextProvider: Send + Sync {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()>;
    async fn after_run(&self, request: &[Message], response: &[Message], error: Option<&Error>) -> Result<()> { Ok(()) }
    fn is_history_provider(&self) -> bool { false }
    // with experimental-session-store:
    fn snapshot_history(&self) -> Option<Vec<Message>> { None }
}
```

`before_run` gets a `SessionContext` and changes it in place. Its fields:

| Field | Read or write | Meaning |
| --- | --- | --- |
| `input_messages` | Read | The caller's messages for this run |
| `session_id`, `service_session_id` | Read | The session's ids |
| `session` | Read; its `state` is writable | A clone of the session. The state bag is shared, so inserts are visible to the caller and later runs |
| `instructions` | Write, through `add_instructions` | Extra instructions, joined with newlines |
| `messages` | Write | Messages placed before the input. History providers put stored turns here |
| `tools` | Write | Tools offered for this run only |

`after_run` gets the run's input and response messages, and the error when the run failed (with an empty response). It does not get the session; a provider that must write per-session state after a run keeps the state handle from `before_run`, as `TextSearchProvider` does. The upstream `invoking`/`invoked` hooks were renamed to `before_run`/`after_run` and `thread_created` was removed, here as in Python and .NET.

### Where providers attach, and the order they run in

Providers attach in two places. `AgentBuilder::context_provider(Arc<dyn ContextProvider>)` (or `context_providers`, which replaces the list) adds one to the agent, so it runs for every session. `AgentSession::with_context_providers(vec![...])` adds providers to one session, replacing that session's list. For each run the agent builds one list, the session's providers then the agent's, and calls each `before_run` in that order; after the run it calls each `after_run` in the same order. When the service holds the conversation, history providers are dropped from both passes.

The request is then assembled like this:

1. A system message holding the agent's instructions, then any per-run instructions, then each provider's instructions in provider order.
2. `ctx.messages`: stored history first (a history provider always puts it at the front), then other providers' messages in the order they were added.
3. The caller's input.

Tools are the agent's own, then providers' tools, then per-run tools, then tools from tool sources such as MCP servers. A tool whose name is already taken is skipped, so an agent tool wins over a provider tool of the same name.

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use async_trait::async_trait;
use serde_json::json;

/// Prints what reaches the model, in place of a real one.
struct Inspect;

#[async_trait]
impl ChatClient for Inspect {
    async fn get_response(&self, messages: Vec<Message>, options: ChatOptions) -> Result<ChatResponse> {
        for m in &messages {
            println!("  {}: {}", m.role, m.text().replace('\n', " / "));
        }
        let tools: Vec<&str> = options.tools.iter().map(|t| t.name.as_str()).collect();
        println!("  tools: {tools:?}");
        Ok(ChatResponse::from_text("ok"))
    }
    async fn get_streaming_response(&self, _: Vec<Message>, _: ChatOptions) -> Result<ChatStream> {
        Err(Error::other("not used"))
    }
}

fn tool(name: &str) -> ToolDefinition {
    FunctionTool::new(name, "A demo tool.", json!({"type": "object", "properties": {}}), |_| async {
        Ok(json!("done"))
    })
    .into_definition()
}

/// Adds an instruction, a context message and a tool to every run.
struct OfficeInfo;

#[async_trait]
impl ContextProvider for OfficeInfo {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        ctx.add_instructions("The office is closed on Sundays.");
        ctx.messages.push(Message::user("Context: today is Sunday."));
        ctx.tools.push(tool("next_open_day"));
        ctx.tools.push(tool("book_room")); // the agent's tool of this name wins
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = Agent::builder(Inspect)
        .instructions("You are a receptionist.")
        .tool(tool("book_room"))
        .context_provider(Arc::new(OfficeInfo))
        .build();

    // Earlier turns, held by the session's history provider.
    let history = InMemoryHistoryProvider::with_messages(vec![
        Message::user("Hello"),
        Message::assistant("Good morning."),
    ]);
    let mut session =
        AgentSession::new().with_context_providers(vec![Arc::new(history) as Arc<dyn ContextProvider>]);

    let options = AgentRunOptions::new()
        .with_chat_options(ChatOptions::new().with_instructions("Answer in one line."))
        .with_tool(tool("call_taxi"));
    agent
        .run_with_options(vec![Message::user("Can I come in today?")], Some(&mut session), options)
        .await?;
    Ok(())
}
```

That prints the system message (receptionist, then "one line", then "closed on Sundays"), the two stored turns, the provider's context message, the question, and the tools `book_room`, `next_open_day`, `call_taxi`.

### Errors and failures

An error from any `before_run` fails the run before the model is called, and no `after_run` runs for it. When the model call fails, every provider's `after_run` is called with the error and its result is ignored, so the caller sees the model's error. When the run succeeds, an error from an `after_run` stops the remaining providers' `after_run` and becomes the run's result, even though the model answered. The session's providers run first, so a history provider on the session has recorded the turn before an agent-level provider can fail. Streaming follows the same rules when the stream ends; a stream dropped early calls no `after_run` at all.

### Per-session memory

A provider instance on the agent is shared by every session and every user, so anything it keeps in its own fields is global. Keep per-conversation data in the session's state bag instead. It travels with `AgentSession::to_dict` and the session store, so the memory persists exactly as far as the session does:

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use async_trait::async_trait;
use serde_json::json;

/// Remembers "Remember: ..." facts per session, in the session's state.
struct Facts;

const KEY: &str = "facts";

#[async_trait]
impl ContextProvider for Facts {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        let Some(session) = ctx.session.clone() else {
            return Ok(()); // a direct call with no session
        };
        let mut facts: Vec<String> = session
            .state
            .get(KEY)
            .and_then(|v| serde_json::from_value(v).ok())
            .unwrap_or_default();
        for message in &ctx.input_messages {
            if let Some(fact) = message.text().strip_prefix("Remember: ") {
                facts.push(fact.to_string());
            }
        }
        if !facts.is_empty() {
            ctx.add_instructions(format!("Known facts about the user:\n- {}", facts.join("\n- ")));
            session.state.insert(KEY, json!(facts));
        }
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .context_provider(Arc::new(Facts))
        .build();

    let mut ada = agent.create_session();
    let mut grace = agent.create_session();
    agent.run(vec![Message::user("Remember: I prefer tea.")], Some(&mut ada)).await?;
    agent.run(vec![Message::user("Remember: I take the train.")], Some(&mut grace)).await?;
    agent.run(vec![Message::user("What should I drink?")], Some(&mut ada)).await?;

    println!("ada:   {:?}", ada.state.get(KEY));
    println!("grace: {:?}", grace.state.get(KEY));
    // The facts go wherever the session goes.
    println!("{}", ada.to_dict()["state"]);
    Ok(())
}
```

This records facts in `before_run`, so a fact stated in a turn that then fails is kept. To record only on success, keep the session's state handle from `before_run` and write in `after_run`, matching the run by its input messages. A provider can also be attached to one session only, which suits per-user configuration such as a tenant's document index.

## Retrieval with TextSearchProvider

`agent_framework::text_search::TextSearchProvider` (stable) is a port of .NET's `TextSearchProvider` and Go's `textsearch`. It wraps any async function from a query string to `Vec<TextSearchResult>`, so the search can be a vector store, a web search, a SQL query or a keyword index. It has two modes:

- `TextSearchBehavior::BeforeInvoke` (the default) searches with the text of the run's input before every run and adds the formatted results as one `user` message ahead of it. Nothing is added when there are no results.
- `TextSearchBehavior::OnDemandFunctionCalling` adds a `Search` tool taking one string argument, `userQuestion`, and searches only when the model calls it.

A failed search is logged at `error` level and the run carries on without the results; in on-demand mode the tool call fails and the model is told so (with the error's text only if `FunctionInvocationConfig::include_detailed_errors` is set).

| Builder method | Default | Effect |
| --- | --- | --- |
| `with_behavior(b)` | `BeforeInvoke` | When to search |
| `with_tool(name, description)` | `Search`, a generic description | The on-demand tool's name and description |
| `with_context_prompt(s)` | `## Additional Context` and a line asking the model to consider the sources | The text before the results |
| `with_citations_prompt(s)` | A request to cite document name and link | The text after the results |
| `with_context_formatter(f)` | none | Format the results yourself; called for every search, including empty ones |
| `with_recent_message_memory(n)` | `0` (off) | Also search with up to `n` earlier messages of the conversation |
| `with_recent_message_roles(roles)` | `[Role::user()]` | Which roles the memory keeps |
| `with_state_key(k)` | `"TextSearchProvider"` | The session-state key for the memory. Returns `Result`; a blank key is refused |
| `with_sensitive_logging(true)` | off | Log queries and formatted results at `trace` |

`TextSearchResult::new(text)` builds a hit; `with_source(name, link)` adds the source shown to the model, and `raw_representation` keeps the backend's own value for a custom formatter. `search_tool()` returns the on-demand tool and `search_and_format(query)` runs one search, both useful in tests.

Recent-message memory is what makes follow-up questions work. "And for sale items?" finds nothing on its own; searched together with the question before it, it does. The memory lives in the session's state, so it needs a session and is per conversation:

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::text_search::{TextSearchProvider, TextSearchResult};
use async_trait::async_trait;

/// Prints the retrieval message the model receives.
struct Inspect;

#[async_trait]
impl ChatClient for Inspect {
    async fn get_response(&self, messages: Vec<Message>, _: ChatOptions) -> Result<ChatResponse> {
        let context = messages.iter().rev().nth(1).map(Message::text).unwrap_or_default();
        println!("context message:\n{context}");
        Ok(ChatResponse::from_text("ok"))
    }
    async fn get_streaming_response(&self, _: Vec<Message>, _: ChatOptions) -> Result<ChatStream> {
        Err(Error::other("not used"))
    }
}

const DOCS: &[(&str, &str, &str)] = &[
    ("Returns", "https://example.com/returns", "Items can be returned within 30 days."),
    ("Sale items", "https://example.com/sale", "Returns of sale items are accepted within 14 days."),
];

#[tokio::main]
async fn main() -> Result<()> {
    let search = TextSearchProvider::new(|query: String| async move {
        println!("searching for: {query:?}");
        // A crude keyword match: "sale" alone finds nothing.
        let q = query.to_lowercase();
        Ok(DOCS
            .iter()
            .filter(|(name, _, _)| match *name {
                "Sale items" => q.contains("return") && q.contains("sale"),
                _ => q.contains("return"),
            })
            .map(|(name, link, text)| TextSearchResult::new(*text).with_source(*name, *link))
            .collect())
    })
    .with_recent_message_memory(2);

    let agent = Agent::builder(Inspect)
        .instructions("Answer from the provided context and cite the source.")
        .context_provider(Arc::new(search))
        .build();

    let mut session = agent.create_session();
    agent.run(vec![Message::user("What is the returns policy?")], Some(&mut session)).await?;
    // The memory adds the previous question to this search.
    agent.run(vec![Message::user("And for sale items?")], Some(&mut session)).await?;
    println!("memory: {}", session.state.get("TextSearchProvider").unwrap());
    Ok(())
}
```

Without the memory the second search would be `"And for sale items?"` and find nothing. With it the query is `"What is the returns policy?\nAnd for sale items?"`, which finds both documents. The memory is updated after a successful run only, and only for the session the run belonged to: `after_run` has no session, so the provider matches the run by its input, and if two different sessions are running the same input at once it records nothing rather than risk crossing them.

On-demand mode suits retrieval that is only sometimes needed, at the cost of an extra model round trip when it is:

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::text_search::{TextSearchBehavior, TextSearchProvider, TextSearchResult};

#[tokio::main]
async fn main() -> Result<()> {
    let search = TextSearchProvider::new(|query: String| async move {
        Ok(vec![TextSearchResult::new(format!("Policy text matching {query:?}: 30-day returns."))
            .with_source("Returns policy", "https://example.com/returns")])
    })
    .with_behavior(TextSearchBehavior::OnDemandFunctionCalling)
    .with_tool("search_policies", "Search the company policy documents.")
    .with_context_formatter(|results| {
        Ok(results
            .iter()
            .map(|r| format!("[{}] {}", r.source_name.as_deref().unwrap_or("?"), r.text))
            .collect::<Vec<_>>()
            .join("\n"))
    });

    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .instructions("Use search_policies for questions about company policy.")
        .context_provider(Arc::new(search))
        .build();
    println!("{}", agent.run_once("Can I return a jacket?").await?.text());
    Ok(())
}
```

Upstream filters the search input so history and other providers' messages are not searched; a Rust provider only ever sees the run's own input and response, so there is nothing to filter. For a vector index, call your embedding client inside the search function and query a [vector collection](#vector-stores); for Azure AI Search use `AzureAISearchProvider` from the integrations page instead.

## Compaction

Compaction bounds what is sent to the model as a conversation grows. `agent_framework::compaction` (stable) has a `CompactionStrategy` trait, four strategies, a `Tokenizer` trait, the `compact` function and `CompactionProvider`, the context provider that applies a strategy on every run.

| Strategy | Keeps | Counts |
| --- | --- | --- |
| `Truncation::new(max_messages)` | Leading system messages plus the newest messages, `max_messages` in total | Messages, system included |
| `SlidingWindow::new(window)` | Leading system messages plus the newest `window` others | Non-system messages |
| `TokenBudget::new(max_tokens)` | Leading system messages plus the newest messages that fit; the newest is always kept, even over budget | Tokens, system included |
| `SelectiveToolResult::new(keep_last)` | Every message, with all but the newest `keep_last` tool results replaced by `"[tool result omitted by compaction]"` (`OMITTED_TOOL_RESULT`) | Tool-result messages |

The first three take `.preserve_first_user()`, which keeps the earliest user message whatever the limit, since in a long task that message usually states the task. It is kept even when that puts the result one message over the limit. Upstream offers this only on truncation.

`compact(messages, &strategy, &tokenizer)` runs a strategy and then repairs the result. It never leaves half a tool exchange: a call without its result, or a result without its call, is removed, because providers reject both with a 400. It never returns only system messages: if the limit would leave nothing else, the newest non-system message is put back. And it drops approval responses already settled by a result, so an old approval cannot run its call again. Calling `strategy.compact(...)` directly gives the raw output without those repairs; use `compact`. Compaction never fails on content: any input gives some retained subset, in the original order.

`ApproxTokenizer`, the default, counts four characters as a token. Text, reasoning, tool names, tool arguments and tool results all count; images and other media count as nothing, because their cost is provider-specific. `count_message_tokens(&tokenizer, &message)` gives one message's count.

```rust
use agent_framework::compaction::count_message_tokens;
use agent_framework::prelude::*;
use agent_framework::types::FunctionArguments;
use serde_json::json;

fn call(id: &str, name: &str) -> Message {
    let call = FunctionCallContent::new(id, name, Some(FunctionArguments::Raw("{}".into())));
    Message::with_contents(Role::assistant(), vec![Content::FunctionCall(call)])
}

fn result(id: &str, payload: &str) -> Message {
    let result = FunctionResultContent::new(id, Some(json!(payload)));
    Message::with_contents(Role::tool(), vec![Content::FunctionResult(result)])
}

fn show(label: &str, messages: &[Message]) {
    let tokens: usize = messages.iter().map(|m| count_message_tokens(&ApproxTokenizer, m)).sum();
    println!("{label}: {} messages, ~{tokens} tokens", messages.len());
    for m in messages {
        let text = m.text();
        let what = if !text.is_empty() {
            text
        } else if let Some(c) = m.function_calls().first() {
            format!("call {} ({})", c.call_id, c.name)
        } else if let Some(r) = m.function_results().first() {
            format!("result {}: {}", r.call_id, r.result.as_ref().unwrap())
        } else {
            String::new()
        };
        println!("    {}: {what}", m.role);
    }
}

fn main() {
    let conversation = vec![
        Message::system("You are a travel assistant."),
        Message::user("Plan a weekend in Lisbon for two."),
        call("c1", "search_flights"),
        result("c1", "TP1350 08:15, TP1352 11:40, FR8343 14:05 (full fare table follows) ..."),
        Message::assistant("TP1352 at 11:40 looks best."),
        Message::user("And a hotel near Alfama?"),
        call("c2", "search_hotels"),
        result("c2", "Memmo Alfama, Santiago de Alfama, Palacio Belmonte ..."),
        Message::assistant("Memmo Alfama has a room."),
        Message::user("Book both."),
    ];
    show("original", &conversation);

    let tokenizer = ApproxTokenizer;
    show("Truncation(5)", &compact(&conversation, &Truncation::new(5), &tokenizer));
    show(
        "SlidingWindow(3) + first user",
        &compact(&conversation, &SlidingWindow::new(3).preserve_first_user(), &tokenizer),
    );
    show("TokenBudget(40)", &compact(&conversation, &TokenBudget::new(40), &tokenizer));
    show("SelectiveToolResult(1)", &compact(&conversation, &SelectiveToolResult::new(1), &tokenizer));

    // The raw strategy output keeps result c2 without its call; compact() drops it.
    let raw = SlidingWindow::new(3).compact(&conversation, &tokenizer);
    show("raw SlidingWindow(3)", &raw);
    show("compact SlidingWindow(3)", &compact(&conversation, &SlidingWindow::new(3), &tokenizer));
}
```

### CompactionProvider

`AgentBuilder::with_compaction(strategy)` wraps the strategy in a `CompactionProvider` with `ApproxTokenizer` and registers it on the agent. Agent providers run after the session's, so it sees the history the session's history provider has just loaded, plus whatever earlier agent providers added, and replaces `ctx.messages` with the compacted list. Three things follow:

- **Only the request shrinks.** The stored history keeps every message. Compaction is a view, recomputed on every run; there is no record of what was dropped, and nothing is summarised.
- **The run's input is never compacted.** It is appended after the providers run, so a single huge input still goes out whole, and a `SlidingWindow(4)` request plateaus at four history messages plus the input.
- **Registration order matters.** Compaction applies to messages added by providers before it, and not to those added after. Register it last if retrieval results or memory messages should survive it, or before them if they should be trimmed too.

For a real tokenizer, implement `Tokenizer` (one method, `count_tokens(&self, text) -> usize`) and register `CompactionProvider::with_tokenizer(strategy, tokenizer)` with `context_provider`:

```rust
use std::sync::{Arc, Mutex};

use agent_framework::prelude::*;
use async_trait::async_trait;

/// Records how many messages each request carried.
#[derive(Clone, Default)]
struct Recorder(Arc<Mutex<Vec<usize>>>);

#[async_trait]
impl ChatClient for Recorder {
    async fn get_response(&self, messages: Vec<Message>, _: ChatOptions) -> Result<ChatResponse> {
        self.0.lock().unwrap().push(messages.len());
        Ok(ChatResponse::from_text("A reply of about ten words, give or take a few."))
    }
    async fn get_streaming_response(&self, _: Vec<Message>, _: ChatOptions) -> Result<ChatStream> {
        Err(Error::other("not used"))
    }
}

/// Counts whitespace-separated words, standing in for a real tokenizer.
struct Words;

impl Tokenizer for Words {
    fn count_tokens(&self, text: &str) -> usize {
        text.split_whitespace().count()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let recorder = Recorder::default();
    let agent = Agent::builder(recorder.clone())
        .instructions("You are a planning assistant.")
        .context_provider(Arc::new(CompactionProvider::with_tokenizer(
            TokenBudget::new(60).preserve_first_user(),
            Words,
        )))
        .build();

    let history = Arc::new(InMemoryHistoryProvider::new());
    let mut session =
        AgentSession::new().with_context_providers(vec![history.clone() as Arc<dyn ContextProvider>]);
    agent.run(vec![Message::user("Plan our team offsite for March.")], Some(&mut session)).await?;
    for i in 1..=6 {
        agent
            .run(vec![Message::user(format!("Follow-up question number {i}."))], Some(&mut session))
            .await?;
    }
    // Requests level off; the stored history keeps everything.
    println!("request sizes: {:?}", recorder.0.lock().unwrap());
    println!("stored messages: {}", history.list_messages().len());
    Ok(())
}
```

What is not ported: upstream's annotation model (messages grouped into spans and flagged as excluded rather than removed), the LLM-backed `SummarizationStrategy`, `TokenBudgetComposedStrategy`, `ContextWindowStrategy` sized from the model's limits, compaction inside the tool loop between model calls, and per-run overrides. There is no trigger threshold either: a strategy runs on every request, and is a no-op while the conversation is within its limit. A custom strategy is one method, `compact(&self, messages: &[Message], tokenizer: &dyn Tokenizer) -> Vec<Message>`; run it through `compact` or `CompactionProvider` and it gets the same repairs as the built-in ones.

## Skills

A `Skill` is a named package of instructions and resources that the model loads only when it needs it. `SkillsProvider` adds a catalogue (each skill's name and description) to the instructions on every run, plus two tools: `load_skill(skill_name)` returns a skill's full instructions, and `read_skill_resource(skill_name, resource_name)` returns one resource. Once a skill is loaded, its instructions are added on every later run too. An unknown skill or resource gives the model a not-found message rather than an error.

```rust
use std::sync::Arc;

use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let skills = SkillsProvider::new(vec![
        Skill::new("code-review", "Review code changes for defects and style")
            .with_instructions("Check error handling first, then naming, then tests.")
            .with_resource("checklist", "1. unwrap() in library code\n2. errors without context"),
        Skill::new("release-notes", "Draft release notes from a changelog")
            .with_instructions("Group by area and lead with breaking changes."),
    ]);

    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .instructions("You are an engineering assistant.")
        .context_provider(Arc::new(skills.clone()))
        .build();
    println!("{}", agent.run_once("Review this: fn f() { x.unwrap() }").await?.text());

    // What the provider adds to the next run, now that code-review is loaded.
    let mut ctx = SessionContext::new(vec![]);
    skills.before_run(&mut ctx).await?;
    println!("{}", ctx.instructions.unwrap_or_default());
    Ok(())
}
```

The set of loaded skills belongs to the provider instance, not to a session: a provider registered on the agent, as here, keeps a skill loaded for every user once any conversation has loaded it. Attach a `SkillsProvider` to each session (it is cheap to clone, but clones share the loaded set, so build one per session with `SkillsProvider::new`) when that matters. Two skills with the same name: the later one wins.

Only inline skills built in code are ported. Upstream's file-based skills (`SKILL.md` directories), MCP-backed skills and the `run_skill_script` tool are not.

## The agent harness

The `experimental-harness` feature adds `agent_framework::harness`, a port of six parts of upstream's harness package: a loop that re-runs an agent until it is done, a judge, a todo list, operating modes, standing tool approvals and file memory. None of it has a semver promise. Upstream's background agents, the file access tools (`FileAccessProvider`), `_memory.py` and the `create_harness_agent` assembly are not ported.

In Python the loop and tool approval are agent middleware. In this port agent middleware runs inside the context-provider pass and its `next` can be called only once, so both are agent wrappers instead, the shape .NET uses: `LoopAgent` and `ToolApprovalAgent` implement `SupportsAgentRun` around another agent, and each loop iteration is a full run with all its providers.

### LoopAgent and the todo list

`LoopAgent::builder(inner, condition)` re-runs `inner` while `condition` says so. A condition is any `LoopCallback<LoopDecision>`: a closure `|ctx: &LoopContext| -> bool` works, as does `(bool, Option<String>)` for a decision with feedback, `async_loop_callback` for an async closure, or a ready-made one. `LoopContext` carries `iteration` (passes completed), `last_result`, the pass's input (`messages`), `original_messages`, the `session`, the `progress` log and the condition's `feedback`.

| Builder method | Default | Effect |
| --- | --- | --- |
| `max_iterations(Some(n))` | `Some(10)` (`DEFAULT_MAX_ITERATIONS`); 5 for a judge loop | Cap on passes. `None` removes it; `Some(0)` fails `build` |
| `next_message(cb)` | One user message, `DEFAULT_NEXT_MESSAGE` ("Continue working on the task. If it is complete, say so.") | Input for each further pass |
| `record_feedback(cb)` | The response text | Entry appended to the progress log after each pass |
| `inject_progress(bool)` | `true` | Add "Progress so far: ..." to the next input; outside fresh-context mode only the latest entry |
| `fresh_context(bool)` | `false` | Start every pass from the original input plus the progress log, resetting the session in between |
| `return_final_only(bool)` | `false` | Return only the last pass from a non-streaming run, instead of every pass with the loop's messages between |
| `additional_instructions(s)` | none | A system message prepended to the original input |

A pass that ends with a pending tool-approval request stops the loop so the caller can answer it. Usage is summed over all passes. With `fresh_context`, the session's state and service id are snapshotted before the loop and restored before each further pass, and the session's history providers are swapped for a scratch copy for the duration; afterwards the original history gets the pre-loop transcript plus the final pass. A server-side thread that the service appends to (rather than an immutable Responses `previous_response_id`) cannot be forked, so with such a session the passes are not isolated.

`TodoProvider` gives the agent five tools (`todos_add`, `todos_complete`, `todos_remove`, `todos_get_remaining`, `todos_get_all`), instructions on planning with them, and a user message listing the current items. Items live in the session's state under `"todo"` by default (`TodoSessionStore`); implement `TodoStore` to keep them elsewhere. `todos_remaining(&todos)` is a loop condition that continues while items are open, and `todos_remaining_message(&todos)` a next-message callback that lists them. Together they make the agent work until its own plan is done. This example uses a scripted client in place of a model so the run is deterministic:

```rust
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use agent_framework::harness::{todos_remaining, todos_remaining_message, LoopAgent, TodoProvider};
use agent_framework::prelude::*;
use agent_framework::types::FunctionArguments;
use async_trait::async_trait;

/// Plays back a fixed sequence of model replies.
struct Script(Mutex<VecDeque<Message>>);

fn tool_call(id: &str, name: &str, args: &str) -> Message {
    let call = FunctionCallContent::new(id, name, Some(FunctionArguments::Raw(args.into())));
    Message::with_contents(Role::assistant(), vec![Content::FunctionCall(call)])
}

#[async_trait]
impl ChatClient for Script {
    async fn get_response(&self, _: Vec<Message>, _: ChatOptions) -> Result<ChatResponse> {
        let next = self.0.lock().unwrap().pop_front().ok_or_else(|| Error::other("script ended"))?;
        Ok(ChatResponse { messages: vec![next], ..Default::default() })
    }
    async fn get_streaming_response(&self, _: Vec<Message>, _: ChatOptions) -> Result<ChatStream> {
        Err(Error::other("not used"))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let script = Script(Mutex::new(VecDeque::from([
        // Pass 1: plan two items, stop.
        tool_call("c1", "todos_add", r#"{"todos":[{"title":"Draft the report"},{"title":"Check the figures"}]}"#),
        Message::assistant("I have planned the work."),
        // Pass 2, after the loop's reminder: finish both.
        tool_call("c2", "todos_complete", r#"{"items":[{"id":1,"reason":"drafted"},{"id":2,"reason":"checked"}]}"#),
        Message::assistant("Both items are done."),
    ])));

    let todos = TodoProvider::new();
    let agent = Agent::builder(script).context_provider(Arc::new(todos.clone())).build();
    let looping = LoopAgent::builder(Arc::new(agent), todos_remaining(&todos))
        .next_message(todos_remaining_message(&todos))
        .max_iterations(Some(5))
        .build()?;

    let mut session = looping.create_session();
    let response = looping
        .run(vec![Message::user("Write the quarterly report.")], Some(&mut session))
        .await?;
    for message in &response.messages {
        let text = message.text();
        let shown = match message.function_calls().first() {
            Some(call) => format!("calls {}", call.name),
            None if text.is_empty() => "(tool result)".to_string(),
            None => text.lines().next().unwrap_or("").to_string(),
        };
        println!("{}: {shown}", message.role);
    }
    for item in todos.items(&session).await? {
        println!("todo {} {:?} complete={}", item.id, item.title, item.is_complete);
    }
    Ok(())
}
```

`TodoProvider` fails a run that has no session, but `Agent` and `LoopAgent` always supply one. Tool calls on one provider are serialised by a single lock, so different sessions' todo calls also take turns; the critical section is short. Upstream's `TodoFileStore` is not ported.

### A judge

`LoopAgent::with_judge(inner, judge)` loops until a second model decides the original request has been answered. `Judge::new(client)` is called directly, with no tools, session or middleware, with its instructions, the original input and the agent's latest response, and asks for a `JudgeVerdict { answered, reasoning }` as JSON schema output. When the reply is not structured, a `VERDICT: MORE` or `VERDICT: DONE` marker decides, and more work wins (so does a reply with neither). The reasoning becomes the next input through `judge_feedback_message`. `criteria([...])` lists requirements for both the judge and, as an extra instruction, the agent; `instructions` replaces `DEFAULT_JUDGE_INSTRUCTIONS` (put `{{criteria}}` where the criteria go); `response_format(None)` and `verdict_parser(f)` handle a provider that needs its own format.

```rust
use std::sync::Arc;

use agent_framework::harness::{Judge, LoopAgent};
use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let writer = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .instructions("You write short product descriptions.")
        .build();
    // A cheaper model is usually enough to judge.
    let judge = Judge::new(Arc::new(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?))
        .criteria(["Under 50 words", "Mentions the battery life"]);

    let looping = LoopAgent::with_judge(Arc::new(writer), judge)
        .return_final_only(true)
        .build()?;
    let response = looping.run(vec![Message::user("Describe our new e-reader.")], None).await?;
    println!("{}", response.text());
    Ok(())
}
```

The judge sees the request and every response, and its reasoning is fed back to the agent, so use a model you trust as much as the agent's own.

### Operating modes

`AgentModeProvider` tracks a mode per session, by default `plan` (the starting mode) or `execute`, each with its own instructions, and gives the agent `mode_get` and `mode_set` tools. The mode is stored in the session's state under `"agent_mode"`. When your code changes the mode with `set_mode` (or the free function `set_agent_mode` for the default configuration), the next run also gets a user message announcing the change, because a model that called `mode_set` earlier tends to follow that call over a changed system prompt; the announcement is cleared only when a run carrying it succeeds. `AgentModeProvider::builder()` sets custom modes (`mode_instructions`), the `default_mode`, the `source_id`, and hides either tool (`expose_mode_set(false)` leaves mode changes to the application). `todos_remaining(&todos).looping_modes(&modes, ["execute"])?` keeps a loop going only in execute mode.

Calling a provider's `before_run` directly on a `SessionContext` shows what it contributes, without a model, which is also how to unit-test a provider:

```rust
use agent_framework::harness::AgentModeProvider;
use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let modes = AgentModeProvider::new();
    let session = AgentSession::new();
    println!("starts in: {}", modes.mode(&session)?);

    // The application approves the plan and switches mode.
    modes.set_mode(&session, "EXECUTE")?; // normalised to lower case
    let mut ctx = SessionContext::new(vec![Message::user("Go ahead.")]);
    ctx.session = Some(session.clone());
    modes.before_run(&mut ctx).await?;

    let instructions = ctx.instructions.unwrap_or_default();
    println!("instructions start: {}", instructions.lines().next().unwrap_or(""));
    println!("tools: {:?}", ctx.tools.iter().map(|t| &t.name).collect::<Vec<_>>());
    println!("announcement: {}", ctx.messages[0].text());
    println!("state: {:?}", session.state.get("agent_mode"));
    assert!(modes.set_mode(&session, "review").is_err()); // not a configured mode
    Ok(())
}
```

`ToolApprovalAgent` wraps an agent and answers tool-approval requests from standing rules: approve a tool always, or for exact arguments, from the user's earlier answers (`always_approve_tool_response`) or from your own `ToolAutoApprovalRule`. It is covered with approvals on the [tools page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_tools/).

### File memory

`FileMemoryProvider` gives the agent a small file system for notes, so it can keep findings, plans and large tool outputs that would otherwise be lost to compaction. It adds seven tools, `file_memory_write` (with an optional description), `file_memory_read`, `file_memory_delete`, `file_memory_ls`, `file_memory_grep` (case-insensitive regex), `file_memory_replace` and `file_memory_replace_lines`, plus usage instructions and, when there is one, a user message holding the memory index.

Memories live in one flat folder per scope. By default the scope is the session id, so each conversation has its own notes and the instructions say so; `with_scope("user-42")` shares a folder between every session with that scope (per user, say) and switches to instructions saying the files are shared. A description is kept in a sidecar file (`notes.md_description.md`) and a `memories.md` index of up to 50 files is rebuilt on every write and delete; both are hidden from the agent's listings and searches. A run with neither a scope nor a session fails rather than falling back to the store root, which would expose everyone's notes. The scope is an opaque key encoded to exactly one folder name, so `tenants/alice` is one folder, not two, and two scopes never share one; a scope whose encoded name passes 255 bytes is refused.

The files go in an `AgentFileStore`. `InMemoryAgentFileStore` keeps them in memory (paths compared case-insensitively) and `FileSystemAgentFileStore::new(root)` on disk, creating the root on first write and refusing symbolic links and paths that leave the root. Paths with a colon, a segment ending in `.` or a space, or a Windows device name are refused on every platform. Searches cap patterns at 256 characters and give up after 10 seconds.

```rust
use std::sync::Arc;

use agent_framework::harness::{AgentFileStore, FileMemoryProvider, InMemoryAgentFileStore};
use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let store = Arc::new(InMemoryAgentFileStore::new());
    let memory = FileMemoryProvider::new(store.clone()).with_scope("user-42");

    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .instructions("You are a research assistant. Save anything worth keeping.")
        .context_provider(Arc::new(memory.clone()))
        .build();

    // The model writes a note through file_memory_write.
    let mut session = agent.create_session();
    let reply = agent
        .run(vec![Message::user("Note that the client prefers PDF reports.")], Some(&mut session))
        .await?;
    println!("{}", reply.text());

    // With a scope the folder does not depend on the session. The store also
    // holds the description sidecar and the index, which the agent's tools hide.
    let folder = memory.working_folder(None)?;
    for entry in store.list_children(&folder).await? {
        println!("{folder}/{}", entry.name);
    }
    if let Some(index) = store.read(&format!("{folder}/memories.md")).await? {
        println!("index given to the next run:\n{index}");
    }
    Ok(())
}
```

Mutating tools hold a lock per provider for their whole read-modify-write, which is enough while one provider writes to a store; two providers over the same store and folder can interleave. The disk store writes UTF-8 text only. Description file names keep the whole file name (`notes.md_description.md`), as Go does, where Python and .NET drop the extension.

## Vector stores

The `experimental-vector-stores` feature adds `agent_framework::vectors`, the provider-neutral core of upstream's vector store layer. It describes a collection once, reads and writes records through two object-safe traits, and filters with portable expressions that each connector translates. The crate ships `InMemoryVectorStore`; the Cosmos DB and Azure AI Search connectors implement the same traits and are on the [integrations page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_integrations/). Upstream's PostgreSQL, Qdrant, Redis, SQL Server and other connectors are not ported.

| Item | Purpose |
| --- | --- |
| `VectorStoreField::key(name)`, `::data(name)`, `::vector(name, dims)` | One field. Refine with `with_storage_name`, `with_type`, `indexed`, `full_text_indexed`, and for vectors `with_index_kind` and `with_distance_function` |
| `VectorStoreCollectionDefinition::new(fields)` | Validated on construction: exactly one key, no duplicate logical or storage names, dimensions on every vector field. `to_storage` and `from_storage` rename between logical and storage names |
| `VectorStore` | `get_collection(name, definition)` (no I/O), `list_collection_names`, `collection_exists` |
| `VectorCollection` | `ensure_collection_exists`, `collection_exists`, `ensure_collection_deleted`, `upsert`, `get`, `delete`, `search` |
| `VectorSearchOptions::new(top)` | `with_skip`, `with_include_vectors`, `with_vector_field_name`, `with_filter`, `with_provider_filter`. Default `top` is 10; `0` is refused |
| `VectorSearchResult` | `record`, `score`, `score_kind`, and `higher_is_closer(distance)` |
| `IndexKind`, `DistanceFunction` | Open string wrappers with constants (`HNSW`, `FLAT`, ...; `COSINE_SIMILARITY`, `DOT_PROD`, `EUCLIDEAN_DISTANCE`, ...). `DistanceFunction::higher_is_closer()` gives the ranking direction |
| `Filter`, `FilterGroup` | Portable filters: `Filter::eq`, `ne`, `gt`, `gte`, `lt`, `lte`, `between`, `any_of`, `none_of`, `exists`, `is_null`, `is_not_null`, `contains`, `contains_any`, `contains_all`, `starts_with`, `ends_with`, `contains_text`; `FilterGroup::and`, `or`, `not` |

Records are `serde_json::Value` objects keyed by logical field name, where Python registers a model class. A typed struct converts at the boundary with `serde_json::to_value` and `from_value`, which keeps the traits object-safe, so code written against `Box<dyn VectorCollection>` moves between stores by changing the line that builds the store. `get` returns one slot per key, `None` for a missing one, so the two lists line up. Vectors are left out of results unless asked for. Upserting a record without its key is an error; deleting an absent key is not.

```rust
use agent_framework::prelude::*;
use agent_framework::vectors::{
    DistanceFunction, Filter, FilterGroup, InMemoryVectorStore, VectorSearchOptions, VectorStore,
    VectorStoreCollectionDefinition, VectorStoreField,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Debug, Serialize, Deserialize)]
struct Doc {
    id: String,
    title: String,
    year: i64,
    tags: Vec<String>,
    embedding: Vec<f32>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let definition = VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id"),
        VectorStoreField::data("title").full_text_indexed(),
        VectorStoreField::data("year").indexed(),
        VectorStoreField::data("tags"),
        VectorStoreField::vector("embedding", 3)
            .with_distance_function(DistanceFunction::new(DistanceFunction::COSINE_SIMILARITY)),
    ])?;

    let store = InMemoryVectorStore::new();
    let docs = store.get_collection("docs", definition)?;
    docs.ensure_collection_exists().await?;

    // Hand-made three-axis "embeddings": rust, databases, cooking.
    let records = [
        Doc { id: "d1".into(), title: "Ownership and borrowing".into(), year: 2019, tags: vec!["rust".into()], embedding: vec![1.0, 0.1, 0.0] },
        Doc { id: "d2".into(), title: "Async Rust with tokio".into(), year: 2023, tags: vec!["rust".into(), "async".into()], embedding: vec![0.9, 0.2, 0.0] },
        Doc { id: "d3".into(), title: "Indexing in Postgres".into(), year: 2024, tags: vec!["sql".into()], embedding: vec![0.1, 1.0, 0.0] },
        Doc { id: "d4".into(), title: "A decent bolognese".into(), year: 2022, tags: vec!["food".into()], embedding: vec![0.0, 0.0, 1.0] },
    ];
    let values = records.iter().map(serde_json::to_value).collect::<std::result::Result<Vec<_>, _>>()?;
    println!("upserted {:?}", docs.upsert(values).await?);

    let query = vec![1.0, 0.0, 0.0];
    for hit in docs.search(query.clone(), &VectorSearchOptions::new(3)).await? {
        println!("{:.3} {}", hit.score.unwrap_or_default(), hit.record["title"]);
    }

    // The filter is applied before `top`, so it narrows rather than empties the page.
    let recent_rust = FilterGroup::and(vec![
        Filter::gte("year", 2020)?.into(),
        Filter::contains("tags", "rust")?.into(),
    ])?;
    let hits = docs.search(query, &VectorSearchOptions::new(3).with_filter(recent_rust)).await?;
    println!("recent rust: {:?}", hits.iter().map(|h| &h.record["title"]).collect::<Vec<_>>());

    // include_vectors = true, or the embedding field is left out of the record.
    let fetched = docs.get(vec![json!("d1"), json!("missing")], true).await?;
    let d1: Doc = serde_json::from_value(fetched[0].clone().unwrap())?;
    println!("d1 = {} ({} dims, missing = {:?})", d1.title, d1.embedding.len(), fetched[1]);

    docs.delete(vec![json!("d4"), json!("never-existed")]).await?;
    docs.ensure_collection_deleted().await?;
    println!("collections: {:?}", store.list_collection_names().await?);
    Ok(())
}
```

Points to know before relying on results:

- **Score direction.** Similarity functions rank higher-is-closer and distances lower-is-closer; read it from `DistanceFunction::higher_is_closer`, and prefer `VectorSearchResult::higher_is_closer`, which also reads `score_kind`. Azure AI Search, for one, returns a relevance score (`score_kind = "relevance"`) or a fusion score (`"rrf"`) whatever metric the field declares, and thresholding on the declared metric would invert the ranking.
- **The in-memory store** scores every record (linear in collection size), defaults to cosine similarity when the field declares no function, refuses a function whose direction it does not know, refuses a query of the wrong width or with non-finite values, skips stored vectors of the wrong width, and refuses `provider_filter`, since it has no dialect to apply one in. Two handles on one name share records.
- **Filters.** A missing field fails every operator except `Exists`, including `Ne`, so `Ne` is not the negation of `Eq`: use `FilterGroup::not`. Numbers compare numerically (`1` matches `1.0`) and never equal booleans; ordered comparisons work on numbers and strings only, which is why ISO-8601 timestamps compare correctly and other date formats do not. Trees are limited to depth 8 and 64 nodes. A connector that cannot express an operator refuses the search rather than dropping the condition. Upstream's late-bound `Param` placeholders are not ported; build the filter where the values are known.

### Vector collection tools

`VectorCollectionContextProvider` hands a collection to the model as tools: `search` (by similarity), `get` (by key), `upsert` and `delete`. Reads need no approval and writes need it by default (`ApprovalMode::AlwaysRequire`, see approvals on the [tools page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_tools/)), since a delete the model gets wrong is not recoverable from the conversation. The model never writes a vector: the builder takes an `EmbeddingClient`, the search tool embeds the query, and the upsert tool exists only when `embed_from_field(field)` names the string data field to embed. The vector field is not in any tool's schema.

| Builder method | Default | Effect |
| --- | --- | --- |
| `tool_prefix(p)` | none | Names tools `p_search` and so on. Without it the plain names collide with any other `search` tool |
| `scope_filter(f)` | none | Applied to every generated tool's search and checked on the records `get`, `upsert` and `delete` touch |
| `embed_from_field(name)` | none (no upsert tool) | Enables `upsert`, embedding that field |
| `vector_field(name)` | the only vector field | Required when the collection has several; `build` refuses an upsert tool over a multi-vector collection |
| `include_search`, `include_get`, `include_delete` | `true` | Turn tools off |
| `approval_mode(VectorToolKind::X, mode)` | reads `NeverRequire`, writes `AlwaysRequire` | Per-tool approval |
| `max_tool_batch_size(n)` | 10 | Records or keys one call may carry |
| `search_top(n)` | 5 | Hits per search |
| `search_embedding_options`, `upsert_embedding_options` | none | Per-operation embedding options, for providers such as Gemini that embed queries and documents differently. A conflicting `dimensions` fails `build` |
| `instructions(vec)`, `source_id(s)`, `additional_tool(t)` | generated, `"vector_collection"`, none | Replace the instructions, rename, add your own tool (the scope filter is not applied to it) |

`build` checks the configuration up front: a zero batch size or `search_top`, an `embed_from_field` that is not a string data field, a filter naming undeclared fields, a missing or ambiguous vector field, and duplicate tool names are all `Error::Configuration` there rather than at the first tool call. The scope filter groups records; it is not a security boundary, because the delete check is read-then-delete and an additional tool carries whatever filter you gave it. Where scoping must hold against a hostile party, give each party its own collection.

```rust
use std::sync::Arc;

use agent_framework::prelude::*;
use agent_framework::vectors::{
    Filter, InMemoryVectorStore, VectorCollection, VectorCollectionContextProvider, VectorStore,
    VectorStoreCollectionDefinition, VectorStoreField,
};
use async_trait::async_trait;
use serde_json::json;

/// A toy embedder: counts a few keywords. Use OpenAIEmbeddingClient or similar in practice.
struct Keywords;

#[async_trait]
impl EmbeddingClient for Keywords {
    async fn get_embeddings(
        &self,
        values: Vec<EmbeddingInput>,
        _: Option<EmbeddingGenerationOptions>,
    ) -> Result<GeneratedEmbeddings> {
        let embed = |text: &str| {
            let t = text.to_lowercase();
            ["coffee", "expense", "launch"].iter().map(|k| t.matches(k).count() as f32 + 0.01).collect()
        };
        Ok(GeneratedEmbeddings::new(
            values.iter().map(|v| Embedding::new(embed(v.as_text().unwrap_or("")))).collect(),
        ))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let definition = VectorStoreCollectionDefinition::new(vec![
        VectorStoreField::key("id").with_type("str"),
        VectorStoreField::data("text").with_type("str"),
        VectorStoreField::data("tenant").with_type("str"),
        VectorStoreField::vector("embedding", 3),
    ])?;
    let collection: Arc<dyn VectorCollection> =
        Arc::from(InMemoryVectorStore::new().get_collection("notes", definition)?);
    collection.ensure_collection_exists().await?;

    let embedder: Arc<dyn EmbeddingClient> = Arc::new(Keywords);
    let seed = [
        ("n1", "The coffee machine is on the third floor.", "acme"),
        ("n2", "File expenses in the portal by the 5th.", "acme"),
        ("n3", "Our launch date is 14 March.", "other"),
    ];
    let vectors = embedder
        .get_embeddings(seed.iter().map(|(_, text, _)| EmbeddingInput::text(*text)).collect(), None)
        .await?;
    let records = seed
        .iter()
        .zip(vectors.embeddings)
        .map(|((id, text, tenant), e)| json!({"id": id, "text": text, "tenant": tenant, "embedding": e.vector}))
        .collect();
    collection.upsert(records).await?;

    let provider = VectorCollectionContextProvider::builder(collection, embedder)
        .tool_prefix("notes")
        .scope_filter(Filter::eq("tenant", "acme")?)
        .embed_from_field("text")
        .build()?;
    for tool in provider.tools() {
        println!("{} (approval: {:?})", tool.name, tool.approval_mode);
    }

    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .instructions("Answer from the notes collection. Say so if you find nothing.")
        .context_provider(Arc::new(provider))
        .build();
    println!("{}", agent.run_once("Where is the coffee machine?").await?.text());
    Ok(())
}
```

In Python the collection owns an embedding generator and embeds internally. Here `VectorCollection::search` takes a vector, which is why the provider needs the embedder.

## Settings for your own providers

A custom provider or store usually needs an endpoint and a key. `load_setting(key, override, default)` resolves a setting the way every `from_env` constructor in the crate does: an explicit value first, then a `.env` file in the working directory, then the environment variable, then the default, giving `None` when nothing produced a value. Surrounding quotes in `.env` values are stripped. `SecretString` wraps a credential so that `Debug` and `Display` print `***`; `expose_secret()` returns the value, and equality compares values. A `#[derive(Debug)]` config struct holding one is then safe to log.

```rust
use agent_framework::prelude::*;

#[derive(Debug)]
struct SearchConfig {
    endpoint: String,
    api_key: SecretString,
    top: usize,
}

impl SearchConfig {
    /// Explicit values win; otherwise ./.env, then the environment, then defaults.
    fn load(api_key: Option<String>) -> Result<Self> {
        let endpoint = load_setting("SEARCH_ENDPOINT", None, Some("http://localhost:9200".into()))
            .expect("a default was given");
        let api_key = load_setting("SEARCH_API_KEY", api_key, None)
            .map(SecretString::new)
            .ok_or_else(|| Error::Configuration("SEARCH_API_KEY is not set".into()))?;
        let top = load_setting("SEARCH_TOP", None, Some("5".into()))
            .and_then(|v| v.parse().ok())
            .unwrap_or(5);
        Ok(Self { endpoint, api_key, top })
    }
}

fn main() -> Result<()> {
    match SearchConfig::load(None) {
        Err(e) => println!("without a key: {e}"),
        Ok(config) => println!("loaded from the environment: {config:?}"),
    }
    let config = SearchConfig::load(Some("sk-live-0123456789".into()))?;
    println!("{config:?}"); // the key prints as ***
    println!("{} top={}", config.endpoint, config.top);
    println!("header value: Bearer {}", config.api_key.expose_secret());
    Ok(())
}
```

Read `expose_secret()` only where the value is used, typically where the `Authorization` header is built, and keep the `SecretString` everywhere else.
