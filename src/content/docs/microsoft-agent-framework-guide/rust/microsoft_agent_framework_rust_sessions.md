---
title: "Microsoft Agent Framework (Rust) — Sessions and history"
description: "Multi-turn conversations in agent-framework-rs: AgentSession and its state bag, history providers, service-held versus local history, streaming with a session, saving and restoring sessions, writing your own chat message store, FileHistoryProvider and the experimental AgentSessionStore."
framework: "microsoft-agent-framework"
language: "rust"
sidebar:
  label: "Sessions"
  order: 6
---

A session is one conversation. In the Rust port, as in current Python and .NET, the session itself is small: an id, an optional service-side conversation id and a bag of free-form state. The messages live elsewhere, in a *history provider*, which is an ordinary [context provider](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_context_and_memory/) that loads stored turns before a run and records the new ones after it. This page covers how the two fit together, when the service keeps the history instead, and every way the crate offers to persist a conversation.

All examples were compiled against `agent-framework` %%version=agent-framework-rs%%. The ones that call OpenAI were run against a local stand-in for the OpenAI API; the rest use a small scripted client and need no network. `FileHistoryProvider` and the session store are experimental and need their features:

```toml
[dependencies]
agent-framework = { version = "%%version=agent-framework-rs%%", features = ["experimental-file-history", "experimental-session-store"] }
tokio = { version = "1", features = ["full"] }
serde_json = "1"
async-trait = "0.1"
futures = "0.3"
```

## The pieces

| Type | Module | Holds |
| --- | --- | --- |
| `AgentSession` | `session` | `session_id` (a UUID unless you restore one), `service_session_id` (optional), `state: SessionState`, `context_providers` |
| `SessionState` | `session` | A JSON map shared by reference: every clone of a session sees the same bag |
| `InMemoryHistoryProvider` | `history` | The messages, in an `Arc<Mutex<Vec<Message>>>` shared by its clones |
| `FileHistoryProvider` | `history` (feature `experimental-file-history`) | The messages, mirrored to one JSON file |
| `HistoryProvider` | `history` | A marker trait for context providers that own a conversation's history |
| `AgentSessionStore` and its two stores | `session_store` (feature `experimental-session-store`) | Whole-session snapshots under a key |

The split matters in three places. `AgentSession::to_dict()` serialises the id, service id and state, never the messages. Clones of a session share their state and their history providers (both are behind `Arc`), but not the service id, which each clone holds by value. And because history is a provider, anything that can be a context provider can be a history store: Redis, Cosmos DB, a SQL table or a file.

## Multi-turn conversations

`agent.create_session()` returns a session with a fresh `InMemoryHistoryProvider` already attached as its first context provider. Pass the same session to each `run` and every turn sees the earlier ones:

```rust
use std::sync::Arc;

use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .instructions("You are a friendly assistant. Keep answers short.")
        .build();

    // create_session() attaches an InMemoryHistoryProvider to the session.
    let mut session = agent.create_session();
    println!("session id: {}", session.session_id());
    for text in ["My name is Ada.", "I work on compilers.", "What do you know about me?"] {
        let reply = agent.run(vec![Message::user(text)], Some(&mut session)).await?;
        println!("> {text}\n{}", reply.text());
    }

    // To read the transcript, attach the provider yourself and keep a handle.
    // with_messages seeds it, for example with turns imported from elsewhere.
    let history = Arc::new(InMemoryHistoryProvider::with_messages(vec![
        Message::user("I prefer metric units."),
        Message::assistant("Noted: metric units."),
    ]));
    let mut seeded = AgentSession::new()
        .with_context_providers(vec![history.clone() as Arc<dyn ContextProvider>]);
    agent
        .run(vec![Message::user("How tall is Ben Nevis?")], Some(&mut seeded))
        .await?;
    for message in history.list_messages() {
        println!("{}: {}", message.role, message.text());
    }

    // No session, no memory: run_once and run(.., None) use a throwaway session.
    agent.run_once("My name is Grace.").await?;
    let reply = agent.run_once("What is my name?").await?;
    println!("without a session: {}", reply.text());
    Ok(())
}
```

What happens on each run, in order:

1. If the session has no service id and no history provider, the agent attaches an `InMemoryHistoryProvider` as its first provider. A session you built with `AgentSession::new()` therefore works too; you just have no handle to its history.
2. Every context provider's `before_run` runs: the session's providers first, then the agent's own. The history provider puts the stored messages at the front of the request.
3. The model is called with the agent's instructions, the provider messages and the new input.
4. After a successful run, every provider's `after_run` sees the input and the response. The history provider appends both. After a failed run `after_run` is called with the error and the history provider records nothing, so a failed turn leaves the conversation as it was.

History grows without limit: every turn sends the whole conversation. Bound what is sent with [compaction](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_context_and_memory/#compaction), which trims the request and leaves the stored history alone, or cap a store that supports it (`RedisChatMessageStore::with_max_messages`).

Attaching your own history provider replaces the default one rather than adding a second: the agent checks `ContextProvider::is_history_provider()` on the session's providers before attaching anything. A history provider registered on the agent with `AgentBuilder::context_provider` is not seen by that check, so the session would get an in-memory one as well and every turn would be stored, and sent, twice. Attach history providers to the session.

### Callers that resend the transcript

Some front ends keep their own copy of the conversation and send all of it every turn (AG-UI does this). The in-memory and file providers detect that the input already starts with the stored turns and neither inject them again nor store them twice. Alignment is by message id where messages carry one, otherwise by role and contents. An input that only repeats the stored tail, or is a single id-less user message that happens to equal the last stored one, is treated as genuinely new, because dropping a real turn cannot be undone and storing a duplicate can. The helpers that do this are public, so a custom store can behave the same way (see [writing a chat message store](#writing-a-chat-message-store)).

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
        Ok(ChatResponse::from_text(format!("reply {}", self.0.lock().unwrap().len())))
    }
    async fn get_streaming_response(&self, _: Vec<Message>, _: ChatOptions) -> Result<ChatStream> {
        Err(Error::other("not used"))
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let recorder = Recorder::default();
    let agent = Agent::builder(recorder.clone()).build();
    let history = Arc::new(InMemoryHistoryProvider::new());
    let mut session =
        AgentSession::new().with_context_providers(vec![history.clone() as Arc<dyn ContextProvider>]);

    // Turn 1 sends one message; the caller keeps its own transcript.
    let mut transcript = vec![Message::user("q1")];
    let reply = agent.run(transcript.clone(), Some(&mut session)).await?;
    transcript.extend(reply.messages);

    // Turn 2: the caller replays everything plus the new question.
    transcript.push(Message::user("q2"));
    agent.run(transcript.clone(), Some(&mut session)).await?;

    // Three messages went to the model, not five, and four are stored, not six.
    println!("request sizes: {:?}", recorder.0.lock().unwrap());
    println!("stored: {}", history.list_messages().len());
    Ok(())
}
```

## Service-held and local history

Some APIs keep the conversation on the server. The OpenAI Responses API (`OpenAIChatClient`) stores every response by default and returns its id; the next request sends `previous_response_id` and only the new input. Azure AI Foundry agents and threads work the same way with their own ids. The session's `service_session_id` is where the crate keeps that pointer.

Whether a run uses local or service history is decided per run:

- **Local.** The session has no service id and the run's options carry no `conversation_id`. History providers load and record the conversation.
- **Service.** The session has a service id, or the options carry a `conversation_id` (an agent default set through `AgentBuilder::chat_options`, or a per-run override). The session's id wins over the options. History providers are left out of the run entirely, both `before_run` and `after_run`, because the service already has those turns.

When a reply carries a conversation id, the agent adopts it into the session (`try_adopt_service_session_id`). So a Responses-API conversation that starts local becomes service-held after its first reply, and from then on the in-memory history provider is idle.

```rust
use agent_framework::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // OpenAIChatClient is the Responses API client.
    let agent = Agent::builder(OpenAIChatClient::from_env("%%model=openai-small%%")?)
        .instructions("You are terse.")
        .build();

    let mut session = agent.create_session();
    assert!(session.service_session_id().is_none());
    agent.run(vec![Message::user("My name is Ada.")], Some(&mut session)).await?;
    // The response id was adopted. Turn 2 sends previous_response_id and only
    // the new message.
    println!("service id: {:?}", session.service_session_id());
    let reply = agent.run(vec![Message::user("What is my name?")], Some(&mut session)).await?;
    println!("{}", reply.text());

    // Continue a conversation whose id you stored earlier.
    let saved_id = session.service_session_id().unwrap().to_string();
    let mut resumed = agent.create_session_with_service_id(saved_id);
    agent.run(vec![Message::user("And my name again?")], Some(&mut resumed)).await?;

    // store = false: the service keeps nothing, returns no conversation id,
    // and the session's in-memory history provider carries the conversation.
    let mut options = ChatOptions::new();
    options.store = Some(false);
    let local = Agent::builder(OpenAIChatClient::from_env("%%model=openai-small%%")?)
        .chat_options(options)
        .build();
    let mut local_session = local.create_session();
    local.run(vec![Message::user("My name is Ada.")], Some(&mut local_session)).await?;
    println!("store=false service id: {:?}", local_session.service_session_id());

    // A service-managed session needs a service that returns conversation ids.
    // Chat Completions returns none, so the run fails rather than silently
    // forgetting the conversation.
    let completions = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?).build();
    let mut wrong = completions.create_session_with_service_id("conv-123");
    match completions.run(vec![Message::user("hi")], Some(&mut wrong)).await {
        Err(Error::AgentExecution(message)) => println!("refused: {message}"),
        other => println!("unexpected: {:?}", other.map(|r| r.text())),
    }
    Ok(())
}
```

Points that follow from those rules:

- **Before 0.14.0 earlier turns were sent twice.** The session adopted the Responses id while the auto-attached in-memory provider went on loading and recording, so the next request replayed earlier turns and also set `previous_response_id` (issue #57). From 0.14.0 history providers are left out whenever the service holds the conversation, the behaviour .NET calls `ClearOnChatHistoryProviderConflict`. If you upgraded and your stored local histories stop growing on Responses-API agents, this is why.
- **A local transcript does not accumulate under service history.** The first turn of a Responses conversation is not recorded locally either: the reply carried an id, so the run counts as service-held. If you need a local copy (for audit, or to move the conversation to another provider), set `store` to `false` and keep history locally, or record it in agent middleware.
- **A service-managed session must keep getting ids back.** If a session has a service id and the reply has none, the run fails with `Error::AgentExecution` once the reply arrives (the model has been called and billed). That is what happens if you give a Chat Completions agent a session created with `create_session_with_service_id`, as above.
- **An agent-level `conversation_id`** makes `create_session()` return a service session for that id instead of a local one.
- **Sub-agents get their own pointer.** `AgentSession::child()`, which `as_tool` uses with `propagate_session(true)`, shares the parent's id and state but starts with no service id, so a sub-agent does not post into the parent's server-side conversation (see [tools](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_tools/)).

### Options that affect history

`AgentRunOptions::with_chat_options` merges a `ChatOptions` over the agent's defaults for one run. Three fields change how history is handled:

| Field | Effect on history |
| --- | --- |
| `conversation_id` | Continues that service conversation for this run. On a session that already has a service id the session's id wins. On a local session it makes the run service-held, so history providers are skipped for it |
| `store` | `Some(false)` asks the Responses API not to store the reply. No conversation id comes back and the session stays local |
| `instructions` | Appended to the agent's instructions for this run only; instructions are never stored in history |

Everything else in `ChatOptions` (temperature, tools, response format) has no bearing on what is stored. Per-run tools added with `AgentRunOptions::with_tool` exist only for that run, but the calls and results they produce are recorded in history like any others, so a later turn can see calls to a tool it is no longer offered.

## Streaming with a session

`run_stream` takes the session by value, `Option<AgentSession>`, because the returned stream outlives the call and records history when it finishes. Pass a clone. Since the clone shares the history provider and state bag, the write-back is visible through your copy once the stream has been read to the end. A stream dropped before its end records nothing.

The service id is the exception: it is held by value, so a conversation id adopted inside the stream lands on the clone and is lost with it. With a service-held conversation, take the id from the updates and set it yourself:

```rust
use std::io::Write;

use agent_framework::prelude::*;
use futures::StreamExt;

async fn stream_turn(agent: &Agent, session: &mut AgentSession, text: &str) -> Result<()> {
    let mut stream = agent
        .run_stream(vec![Message::user(text)], Some(session.clone()), None)
        .await?;
    let mut conversation_id = None;
    while let Some(update) = stream.next().await {
        let update = update?;
        if update.conversation_id.is_some() {
            conversation_id = update.conversation_id.clone();
        }
        print!("{}", update.text());
        std::io::stdout().flush().ok();
    }
    println!();
    // The stream adopted the id on its own copy of the session.
    if let Some(id) = conversation_id {
        session.set_service_session_id(id);
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    // Chat Completions: local history, recorded when each stream ends.
    let local = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?).build();
    let mut session = local.create_session();
    stream_turn(&local, &mut session, "My name is Ada.").await?;
    stream_turn(&local, &mut session, "What is my name?").await?;

    // Responses: service history, so the id has to be carried over by hand.
    let service = Agent::builder(OpenAIChatClient::from_env("%%model=openai-small%%")?).build();
    let mut session = service.create_session();
    stream_turn(&service, &mut session, "My name is Ada.").await?;
    println!("service id after streaming: {:?}", session.service_session_id());
    stream_turn(&service, &mut session, "What is my name?").await?;
    Ok(())
}
```

Use `agent.create_session()` rather than `AgentSession::new()` when you will stream. `create_session` attaches the history provider eagerly, so the clone you pass shares it. A bare `AgentSession::new()` gets its provider attached inside the run, on the clone, and your copy never sees the history. With agent middleware configured, `run_stream` runs to completion and replays the result as updates; the session handling is the same.

## Saving and restoring a session

A conversation has two halves to save: the session (`to_dict`) and its history (`InMemoryHistoryProvider::to_dict`). Both are `serde_json::Value`s, so put them in one envelope and store it wherever you like. Restoring reverses it: `AgentSession::from_dict` (or `agent.session_from_dict`, the same thing) and `InMemoryHistoryProvider::from_dict`, then reattach the provider, since `from_dict` never restores context providers.

```rust
use std::path::Path;
use std::sync::Arc;

use agent_framework::prelude::*;
use serde_json::{json, Value};

/// Save the session and its history as one JSON document.
fn suspend(path: &Path, session: &AgentSession, history: &InMemoryHistoryProvider) -> Result<()> {
    let envelope = json!({
        "format": 1,
        "session": session.to_dict(),
        "history": history.to_dict(),
    });
    std::fs::write(path, serde_json::to_vec_pretty(&envelope)?).map_err(Error::other)
}

/// Rebuild the session, with its history provider reattached.
fn resume(path: &Path, agent: &Agent) -> Result<(AgentSession, Arc<InMemoryHistoryProvider>)> {
    let envelope: Value = serde_json::from_slice(&std::fs::read(path).map_err(Error::other)?)?;
    let history = Arc::new(InMemoryHistoryProvider::from_dict(&envelope["history"])?);
    let session = agent
        .session_from_dict(&envelope["session"])?
        .with_context_providers(vec![history.clone() as Arc<dyn ContextProvider>]);
    Ok((session, history))
}

#[tokio::main]
async fn main() -> Result<()> {
    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?).build();
    let path = std::env::temp_dir().join("conversation-42.json");

    let history = Arc::new(InMemoryHistoryProvider::new());
    let mut session =
        AgentSession::new().with_context_providers(vec![history.clone() as Arc<dyn ContextProvider>]);
    session.state.insert("tier", json!("pro"));
    agent
        .run(vec![Message::user("Remember: my project is called skylark.")], Some(&mut session))
        .await?;
    let id = session.session_id().to_string();
    suspend(&path, &session, &history)?;
    drop((session, history));

    // Later, in another process.
    let (mut session, history) = resume(&path, &agent)?;
    assert_eq!(session.session_id(), id);
    assert_eq!(session.state.get("tier"), Some(json!("pro")));
    let reply = agent
        .run(vec![Message::user("What is my project called?")], Some(&mut session))
        .await?;
    println!("{}", reply.text());
    println!("{} messages in history", history.list_messages().len());
    std::fs::remove_file(&path).ok();
    Ok(())
}
```

The shapes are plain JSON. A session is `{"session_id": "...", "service_session_id": null, "state": {...}}`; a history is `{"messages": [...]}`, each message in the crate's serde form. `from_dict` is lenient: a missing `session_id` gets a new UUID and a missing `state` or `messages` reads as empty, so check that you are restoring the document you meant to. Values in `state` must be JSON, which is why the bag is a map of `serde_json::Value`; store your own types with `serde_json::to_value` and read them back with `from_value`. There is no counterpart to Python's `register_state_type`, and none is needed.

For a service-held conversation the session document alone is enough, because the service id is the conversation. For a local one, whatever you used as the history store is what has to survive: an in-memory provider must be serialised as above, while a file or database store only needs to be reattached with the same path or key.

Treat both documents as user data. They contain the conversation in plain text; encrypt them at rest if the conversation is sensitive.

## History providers

A history provider is a `ContextProvider` whose `is_history_provider()` returns `true`. Implementing the `HistoryProvider` marker trait documents the intent but changes nothing at run time; the method is what the agent checks. The crate ships four:

| Provider | Feature | Storage | Notes |
| --- | --- | --- | --- |
| `InMemoryHistoryProvider` | default | Process memory | `new`, `with_messages`, `list_messages`, `to_dict`, `from_dict`. Clones share storage |
| `FileHistoryProvider` | `experimental-file-history` | One JSON file | Loads on construction, rewrites the file after each successful run |
| `RedisChatMessageStore` | `redis` | One Redis list per session | `with_max_messages` trims old entries |
| `CosmosChatMessageStore` | `cosmos` | Cosmos DB documents | Partitioned by thread, or by tenant, user and thread |

The Redis and Cosmos stores are configured on the [integrations page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_integrations/). They are used exactly like the in-memory one: construct the store for a conversation and attach it to the session.

Python's history providers take `load_messages`, `store_inputs`, `store_outputs` and similar flags, and keep messages in `session.state`. The Rust providers have no such flags and keep their messages in the provider, not the state bag. To record without loading, or the reverse, write a provider that does only one of the two.

### Writing a chat message store

Implement `before_run` to inject stored messages and `after_run` to append the new ones, and return `true` from `is_history_provider`. The `history` module exports the helpers the built-in providers use, so a custom store gets the same handling of replayed transcripts:

- `inject_stored_history(ctx, stored)` puts `stored` ahead of any messages other providers have added, unless the input already carries them.
- `new_run_messages(existing, request, response)` returns what to append: the part of the input that is not a replay, then every response message.
- `StoredHistory::Window` with the `_from` variants of both helpers is for a store that keeps only the most recent messages.

This one keeps conversations in a shared map standing in for a database table, keyed by conversation id:

```rust
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use agent_framework::history::{inject_stored_history, new_run_messages};
use agent_framework::prelude::*;
use async_trait::async_trait;

/// Stands in for a table of (conversation_id, messages_json).
type Table = Arc<Mutex<HashMap<String, String>>>;

/// A history provider for one conversation in the table.
struct TableHistory {
    table: Table,
    conversation_id: String,
}

impl TableHistory {
    fn load(&self) -> Result<Vec<Message>> {
        match self.table.lock().unwrap().get(&self.conversation_id) {
            Some(json) => Ok(serde_json::from_str(json)?),
            None => Ok(Vec::new()),
        }
    }
}

#[async_trait]
impl ContextProvider for TableHistory {
    async fn before_run(&self, ctx: &mut SessionContext) -> Result<()> {
        let stored = self.load()?;
        inject_stored_history(ctx, stored);
        Ok(())
    }

    async fn after_run(
        &self,
        request: &[Message],
        response: &[Message],
        error: Option<&Error>,
    ) -> Result<()> {
        if error.is_some() {
            return Ok(()); // keep failed turns out of the conversation
        }
        let mut messages = self.load()?;
        messages.extend(new_run_messages(&messages, request, response));
        let json = serde_json::to_string(&messages)?;
        self.table.lock().unwrap().insert(self.conversation_id.clone(), json);
        Ok(())
    }

    fn is_history_provider(&self) -> bool {
        true
    }
}

impl HistoryProvider for TableHistory {}

fn session_for(table: &Table, conversation_id: &str) -> AgentSession {
    let history = TableHistory { table: table.clone(), conversation_id: conversation_id.into() };
    AgentSession::new().with_context_providers(vec![Arc::new(history) as Arc<dyn ContextProvider>])
}

#[tokio::main]
async fn main() -> Result<()> {
    let table: Table = Arc::default();
    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?).build();

    let mut session = session_for(&table, "conv-1");
    agent.run(vec![Message::user("My name is Ada.")], Some(&mut session)).await?;

    // A new session over the same row continues the conversation.
    let mut session = session_for(&table, "conv-1");
    let reply = agent.run(vec![Message::user("What is my name?")], Some(&mut session)).await?;
    println!("{}", reply.text());
    let stored: Vec<Message> = serde_json::from_str(&table.lock().unwrap()["conv-1"])?;
    println!("{} messages stored", stored.len());
    Ok(())
}
```

An error from `after_run` fails the run after the model has answered, so the caller sees an error for a turn that did happen. Decide whether a storage failure should do that or be logged. The built-in file provider returns it; the in-memory provider cannot fail. Two runs on the same conversation at once will race in a read-modify-write store like this one; the file provider serialises its own writes, but a database store needs a transaction or an append operation.

A provider that persists its own messages should leave `snapshot_history` (from `experimental-session-store`) at its default of `None`, so a session store reattaches it rather than copying its messages.

## FileHistoryProvider

`FileHistoryProvider::new(path)` (feature `experimental-file-history`) keeps one conversation in one JSON file, `{"messages": [...]}`. Construction reads the file if it exists, so a malformed file fails there with `Error::Serialization` rather than discarding the history; a missing or empty file starts empty. Each successful run writes the whole history to a temporary file and renames it over the old one, so a reader or a crash sees the old file or the new one, never half of one. It does not `fsync`, so a power cut straight after a run can lose that run. The in-memory copy is updated only after the write succeeds, and concurrent runs on clones of one provider take turns.

Python's `FileHistoryProvider` is attached to the agent with a directory and keys files by session id. The Rust one is attached to a session and given the file path, so the naming scheme is yours. Derive the file name from an id you control, never from raw user input, because the provider writes wherever the path points:

```rust
use std::path::{Path, PathBuf};
use std::sync::Arc;

use agent_framework::prelude::*;

/// One file per conversation, named by an id we generated.
fn history_path(dir: &Path, conversation_id: &str) -> Result<PathBuf> {
    if conversation_id.is_empty()
        || !conversation_id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-')
    {
        return Err(Error::Configuration(format!("bad conversation id {conversation_id:?}")));
    }
    Ok(dir.join(format!("{conversation_id}.json")))
}

async fn turn(agent: &Agent, dir: &Path, conversation_id: &str, text: &str) -> Result<()> {
    // Reloads whatever an earlier process wrote.
    let history = Arc::new(FileHistoryProvider::new(history_path(dir, conversation_id)?)?);
    println!("loaded {} messages from {}", history.list_messages().len(), history.path().display());
    let mut session =
        AgentSession::new().with_context_providers(vec![history.clone() as Arc<dyn ContextProvider>]);
    let reply = agent.run(vec![Message::user(text)], Some(&mut session)).await?;
    println!("> {text}\n{}", reply.text());
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let dir = std::env::temp_dir().join("agent-histories");
    std::fs::create_dir_all(&dir).map_err(Error::other)?;
    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?).build();

    // A fresh id per conversation: a new session's UUID will do.
    let id = AgentSession::new().session_id().to_string();
    turn(&agent, &dir, &id, "My favourite colour is teal.").await?;
    turn(&agent, &dir, &id, "What is my favourite colour?").await?;

    assert!(history_path(&dir, "../../etc/passwd").is_err());
    std::fs::remove_file(history_path(&dir, &id)?).ok();
    Ok(())
}
```

The provider does not use the session id at all, so nothing ties the file to a session except your naming. The session's state bag is not in the file; save `session.to_dict()` beside it if you use the state. Being experimental, the type may change in any minor release.

## The session store

The `experimental-session-store` feature adds `agent_framework::session_store`, a port of .NET's `AgentSessionStore` and Python's `SessionStore`. It saves and restores whole sessions, history included, under a key, which is the shape a request handler wants: load the session, run the agent, save the session.

| Item | Purpose |
| --- | --- |
| `AgentSessionStoreKey::new(id)` | A key for a session id. An empty or blank id is `Error::Configuration` |
| `.with_partition(name, value)` | Scope the id, for example by tenant and user. Keys compare by value and partition order does not matter; blank names or values are refused |
| `AgentSessionStore` | The trait: `save_session`, `get_session` (returns `None` when nothing is stored), `get_or_create_session` (provided), `delete_session` |
| `InMemoryAgentSessionStore` | Snapshots in a shared map. Cheap to clone, never evicts; `len` and `is_empty` |
| `FileAgentSessionStore::new(root)` | One file per session at `<root>/<agent id>/p-<partitions>/<session id>.json` |
| `serialize_session`, `deserialize_session` | The snapshot format, for writing your own store |
| `ContextProvider::snapshot_history` | How a snapshot asks a provider for in-memory history |

Every method takes the agent as well as the key, and stores are scoped by `agent.id()`. `Agent` ids are random UUIDs unless you set one with `AgentBuilder::id`, so give the agent a fixed id or a restarted process, or a second instance, will find nothing:

```rust
use agent_framework::prelude::*;
use agent_framework::session_store::{
    AgentSessionStore, AgentSessionStoreKey, FileAgentSessionStore, InMemoryAgentSessionStore,
};

/// One request: load (or start) the conversation, run a turn, save it.
async fn handle(
    store: &dyn AgentSessionStore,
    agent: &Agent,
    tenant: &str,
    user: &str,
    conversation: &str,
    text: &str,
) -> Result<String> {
    let key = AgentSessionStoreKey::new(conversation)?
        .with_partition("tenant", tenant)?
        .with_partition("user", user)?;
    let mut session = store.get_or_create_session(agent, &key).await?;
    let reply = agent.run(vec![Message::user(text)], Some(&mut session)).await?;
    store.save_session(agent, &key, &session).await?;
    Ok(reply.text())
}

#[tokio::main]
async fn main() -> Result<()> {
    // A stable id: stored sessions are scoped by it.
    let agent = Agent::builder(OpenAIChatCompletionClient::from_env("%%model=openai-small%%")?)
        .id("support-agent")
        .build();

    let root = std::env::temp_dir().join("agent-sessions");
    let files = FileAgentSessionStore::new(&root);
    handle(&files, &agent, "acme", "ada", "c1", "My name is Ada.").await?;
    println!("{}", handle(&files, &agent, "acme", "ada", "c1", "What is my name?").await?);

    // Same conversation id, different user: a different, empty session.
    println!("{}", handle(&files, &agent, "acme", "grace", "c1", "What is my name?").await?);

    // The in-memory store has the same interface, for tests and single processes.
    let memory = InMemoryAgentSessionStore::new();
    handle(&memory, &agent, "acme", "ada", "c1", "Hello").await?;
    println!("in memory: {} session(s)", memory.len());

    // Deleting is idempotent.
    let key = AgentSessionStoreKey::new("c1")?
        .with_partition("tenant", "acme")?
        .with_partition("user", "ada")?;
    files.delete_session(&agent, &key).await?;
    files.delete_session(&agent, &key).await?;
    assert!(files.get_session(&agent, &key).await?.is_none());
    std::fs::remove_dir_all(&root).ok();
    Ok(())
}
```

What a snapshot contains: the session's `to_dict()` fields plus, under `"history"`, the messages of the first context provider whose `snapshot_history()` returns `Some`. Only `InMemoryHistoryProvider` does, which covers the provider `create_session` attaches. Restoring starts from `agent.create_session()` and puts the id, service id, state and recorded history into it, so whatever the agent attaches to a new session is there on a restored one too. Other providers that were on the saved session are not restored; a provider that persists itself, such as a file, Redis or Cosmos store, has to be reattached by your code after `get_session`, as with `from_dict`. A service-held session's snapshot has no history and restores without a local history provider, so it does not start a second copy of a conversation the service already has.

Each `get_session` returns an independent session: changing one, or running two branches from the same key, does not touch the stored snapshot or each other. Saving is last-writer-wins, in both stores, with no version check. Two requests on one conversation at once will each save their own turn and one will be lost, so serialise requests per conversation if that matters.

`FileAgentSessionStore` encodes every path component so any id is a safe file name and distinct keys never share a file: lowercase ASCII letters, digits, `-` and `_` are kept, every other byte becomes `~XX` (so uppercase letters are encoded and `Foo` and `foo` stay distinct on a case-insensitive disk), and Windows device names such as `con` are escaped. A component longer than 255 bytes once encoded is refused rather than shortened. Directories are created on the first save. Writes go through a temporary file and a rename. The snapshots are plaintext, so treat the root as trusted application storage.

Differences from upstream: .NET can scope by a host-supplied storage identity, and its store supports `GetService`; here the scope is always `agent.id()` and the store is a plain trait object. Python keeps local history in `session.state`, so its snapshots carry it automatically; this port keeps it in the provider and asks for it through `snapshot_history`. The feature is experimental and can change in any minor release.

## Workflows and hosted agents

`WorkflowAgent` (a workflow wrapped as an agent) attaches an in-memory history provider in the same way, so a session works with it as with `Agent` (see [workflows](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_workflows/)). The hosting routers keep their own per-thread state and are covered on the [hosting page](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_hosting/). Workflow checkpoints are a separate mechanism from sessions and are covered with [checkpointing](/microsoft-agent-framework-guide/rust/microsoft_agent_framework_rust_workflows/#checkpointing).
