---
title: "Microsoft Agent Framework (Go) — Sessions and memory"
description: "Conversation state in the Go SDK: agent.Session and its state bag, service-held and local history, saving and restoring sessions, history providers and custom stores, message sources and filters, context providers, compaction strategies, text-search retrieval, vector-backed chat memory, Foundry memory stores, and the agent harness (todos, modes, file memory, skills)."
framework: "microsoft-agent-framework"
language: "go"
sidebar:
  label: "Sessions and memory"
  order: 7
---

Three pieces hold an agent's state between runs:

- **`*agent.Session`** is one conversation. It holds a key-value state bag and, when the model service keeps the history, the service's ID for it. It is plain data and serialises to JSON.
- **A history provider** (`agent.HistoryProvider`) loads the transcript into each run and stores the new messages afterwards. Every agent has one; the default keeps the messages in the session.
- **Context providers** (`agent.ContextProvider`) add instructions, messages or tools before a run and learn from the result after it. Memory, retrieval, compaction and the harness components are all context providers.

The agent itself is stateless and can be shared; the session is what you keep per user or per conversation. This page covers each piece and the ready-made providers built on them.

Every program compiles with `go vet` and `go build` against v%%version=github.com/microsoft/agent-framework-go%%. All except the OpenAI Conversations, Azure AI Search and Foundry memory examples were run against a local stand-in for the OpenAI API (with a deterministic bag-of-words embeddings endpoint), which calls the first tool it is offered and otherwise replies "Mock reply to: ..." with the last user message.

## Sessions

Create a session with `a.CreateSession(ctx)` and pass it to each run with `agent.WithSession(session)`:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/openai/openai-go/v3"
)

func main() {
	ctx := context.Background()
	a := openaiprovider.NewChatCompletionsAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai-small%%",
		Instructions: "You are a friendly assistant.",
	})

	session, err := a.CreateSession(ctx)
	if err != nil {
		log.Fatal(err)
	}
	for _, q := range []string{"My name is Ada.", "I live in Leeds.", "What do you know about me?"} {
		resp, err := a.RunText(ctx, q, agent.WithSession(session)).Collect()
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(resp)
	}

	// The default history provider keeps the transcript in the session.
	history, err := session.InMemoryHistory("")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("messages stored:", len(history))
	for _, m := range history {
		fmt.Printf("  %-9s %s\n", m.Role, m)
	}

	// A run without a session is independent: nothing is remembered.
	resp, err := a.RunText(ctx, "What is my name?").Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)
}
```

Facts about sessions:

- **A run without a session gets a throwaway one.** The agent creates a session for that run only, so middleware and context providers always find one, but nothing is kept. History is neither loaded nor stored.
- **`CreateSession` or a zero value.** `var s agent.Session` (or `&agent.Session{}`) is ready for sequential use. `CreateSession` also lets the provider seed provider-specific state, and accepts `agent.WithServiceID(id)` to attach an existing server-side conversation. Prefer it.
- **One agent per session.** Provider state can be tied to the agent that created it, so do not move a session between agents with different providers or history settings.
- **Concurrency.** Session methods are safe for concurrent use once the session is initialised (by `CreateSession`, a run, `Set` or decoding), but one conversation should not run twice at the same time: two runs would both append to the same history.

### The state bag

History providers, context providers and your own code keep per-conversation data in the session's state under string keys:

| Method | Behaviour |
| --- | --- |
| `Set(key, v)` | stores `v`, replacing any value; panics on a blank key |
| `Get(key, &v)` | reads into `v`; returns `ok=false` (and a nil error) if the key is missing **or** the stored value is not of `v`'s type |
| `Delete(key)` | removes the key; reports whether it existed |
| `Count()` | number of keys |
| `ServiceID()`, `SetServiceID(id)` | the provider's conversation or response ID, if the service holds the history |
| `InMemoryHistory(stateKey)`, `SetInMemoryHistory(stateKey, msgs)` | the transcript kept by an in-memory history provider; `""` selects the default key, `in-memory` |

`Get` is typed. Before a session is serialised it returns the stored value only if you ask for exactly its type, so a value stored as `*Profile` is not found by `Get(key, &profile)` with `profile` a `Profile`. After a session is decoded from JSON, the first `Get` decodes the raw JSON into whatever type you ask for and caches that type; a later `Get` with a different type returns false. Pick one type per key and use it everywhere. Values are not copied: change a value you got back, then `Set` it again so the change is serialised.

## Saving and restoring sessions

`agent.Session` implements `json.Marshaler` and `json.Unmarshaler`. The JSON holds the state bag and the service ID; for the default history provider that includes the whole transcript:

```go
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/openai/openai-go/v3"
)

type Profile struct {
	Name     string `json:"name"`
	Language string `json:"language"`
}

func main() {
	ctx := context.Background()
	a := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:              "%%model=openai%%",
		DisableStoreOutput: true, // history kept in the session, so the file is self-contained
	})

	session, err := a.CreateSession(ctx)
	if err != nil {
		log.Fatal(err)
	}
	session.Set("profile", Profile{Name: "Ada", Language: "en-GB"})
	if _, err := a.RunText(ctx, "Remember that my favourite colour is green.", agent.WithSession(session)).Collect(); err != nil {
		log.Fatal(err)
	}

	path := filepath.Join(os.TempDir(), "session-ada.json")
	data, err := json.MarshalIndent(session, "", "  ")
	if err != nil {
		log.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		log.Fatal(err)
	}
	fmt.Printf("saved %d bytes\n", len(data))

	// Later, perhaps in another process.
	raw, err := os.ReadFile(path)
	if err != nil {
		log.Fatal(err)
	}
	var restored agent.Session
	if err := json.Unmarshal(raw, &restored); err != nil {
		log.Fatal(err)
	}
	var p Profile
	if ok, err := restored.Get("profile", &p); err != nil || !ok {
		log.Fatal("profile missing")
	}
	fmt.Println("profile:", p.Name, p.Language)

	resp, err := a.RunText(ctx, "What is my favourite colour?", agent.WithSession(&restored)).Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)
	h, _ := restored.InMemoryHistory("")
	fmt.Println("messages after restore:", len(h))
}
```

The serialised form contains the conversation, so treat it as user data: encrypt it at rest and control who can read it. The session file of a long conversation grows with every turn. Keep the transcript somewhere else with a [custom history provider](#writing-a-history-store), so the session holds only a reference, or keep it small with [compaction](#compaction).

## Service-held and local history

Whether the transcript lives in the session or on the model service depends on the provider:

| Provider | Where history lives | What the session holds |
| --- | --- | --- |
| OpenAI Responses, Foundry (default) | on the service: each response is stored, and the next request sends `previous_response_id` | the latest response ID as its service ID |
| OpenAI Responses with an OpenAI conversation | in the Conversations object | the `conv_...` ID |
| OpenAI Responses or Foundry with `DisableStoreOutput: true` | in the session | the transcript, under the history provider's key |
| Chat Completions, Anthropic, Gemini | in the session | the transcript |
| GitHub Copilot | in the Copilot session | the Copilot session ID |
| A2A | on the remote agent | its context and task IDs |

When the session has a service ID, the agent does not load or store local history: the service is the source of truth, and each run sends only the new messages. A service ID starting `conv_` is sent as `conversation`, any other as `previous_response_id`. To continue an existing OpenAI conversation, pass its ID when creating the session:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/conversations"
)

func main() {
	ctx := context.Background()
	client := openai.NewClient()

	conv, err := client.Conversations.New(ctx, conversations.ConversationNewParams{})
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _, _ = client.Conversations.Delete(ctx, conv.ID) }()

	a := openaiprovider.NewResponsesAgent(client, openaiprovider.AgentConfig{Model: "%%model=openai%%"})
	session, err := a.CreateSession(ctx, agent.WithServiceID(conv.ID))
	if err != nil {
		log.Fatal(err)
	}
	for _, q := range []string{"What is the capital of France?", "How tall is its most famous tower?"} {
		resp, err := a.RunText(ctx, q, agent.WithSession(session)).Collect()
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(resp)
	}

	// The conversation can be read back, or shared with other OpenAI clients.
	items := client.Conversations.Items.ListAutoPaging(ctx, conv.ID, conversations.ItemListParams{})
	for items.Next() {
		fmt.Println("item:", items.Current().Type)
	}
	if err := items.Err(); err != nil {
		log.Fatal(err)
	}
}
```

Service-held history saves tokens and keeps sessions tiny, but the conversation then exists only on the service, under its retention rules, and cannot be moved to another provider. Local history (`DisableStoreOutput: true`) is portable and inspectable, and works with compaction, which skips service-managed sessions.

### History conflicts

If you configure your own `HistoryProvider` and the service also returns a conversation ID, both would claim the history. By default the run then **fails** with "only Session.ServiceID or HistoryProvider may be used, but not both". Three `*bool` fields on `agent.Config` change this:

| Field | Default | Effect |
| --- | --- | --- |
| `ThrowOnHistoryProviderConflict` | true | return the error above |
| `WarnOnHistoryProviderConflict` | true | log a warning through `Config.Logger` |
| `ClearOnHistoryProviderConflict` | true | when not failing, stop using the configured history provider for this agent from then on |

The usual fix is not to change these, but to set `DisableStoreOutput: true` on a Responses or Foundry agent that has a custom history provider.

### Per-call persistence

Normally history is stored once, when a run finishes. `Config.RequirePerServiceCallHistoryPersistence` stores it after **every model call** inside the tool loop instead, so a run that fails on its third model call keeps the first two calls' messages and tool results, and messages injected mid-run (see [middleware](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_middleware/#injecting-messages-during-a-run)) are kept between calls. The [agent harness](#the-agent-harness) turns it on.

## History providers

When `Config.HistoryProvider` is nil the agent uses `agent.NewInMemoryHistoryProvider` with default settings, storing the transcript in the session under `in-memory`. Configure your own to change the key, seed a conversation, or filter what is stored:

| `InMemoryHistoryProviderConfig` field | Default | Purpose |
| --- | --- | --- |
| `SourceID` | `in-memory` | source stamped on loaded messages |
| `StateKey` | `SourceID` | session key; use different keys for agents that share a session |
| `StateInitializer func(*Session) []*message.Message` | none | initial messages for a new session, such as a few-shot example |
| `ProvideOutputMessageFilter` | all | filter applied to stored messages as they are loaded |
| `StoreInputRequestMessageFilter` | messages not from a history provider | which request messages are stored |
| `StoreInputResponseMessageFilter` | all | which response messages are stored |

### Approvals and local history

The default in-memory history stores approval content like any other: the assistant message holding the `ToolApprovalRequestContent`, and the user message holding your `ToolApprovalResponseContent`, followed by the call and its result. Against the local stand-in (not a real model), the next run on such a session failed before calling the model with `ToolApprovalRequestContent found with ToolCall.CallID(s) 'call_3' that have no matching ToolApprovalResponseContent`, naming the earlier request even though its response is in the stored history. It happened with Chat Completions and with Responses and `DisableStoreOutput`, also with unique call IDs; Responses with service-held history was not affected. Until this is clarified upstream, loading history without the approval messages avoided it in the same test, across several approve-then-continue turns (approval binding is kept in separate session state, so it still works):

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/tool"
	"github.com/microsoft/agent-framework-go/tool/functool"
	"github.com/openai/openai-go/v3"
)

type NoteIn struct {
	Text string `json:"text"`
}

var saveNote = tool.ApprovalRequiredFunc(functool.MustNew(
	functool.Config{Name: "save_note", Description: "Save a note."},
	func(ctx context.Context, in NoteIn) (string, error) { return "saved", nil }))

// withoutApprovals drops stored messages that carry approval requests or responses.
func withoutApprovals(_ context.Context, msgs []*message.Message) ([]*message.Message, error) {
	out := msgs[:0]
	for _, m := range msgs {
		keep := true
		for _, c := range m.Contents {
			switch c.(type) {
			case *message.ToolApprovalRequestContent, *message.ToolApprovalResponseContent:
				keep = false
			}
		}
		if keep {
			out = append(out, m)
		}
	}
	return out, nil
}

func main() {
	ctx := context.Background()
	a := openaiprovider.NewChatCompletionsAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model: "%%model=openai%%",
		Config: agent.Config{
			Tools: []tool.Tool{saveNote},
			HistoryProvider: agent.NewInMemoryHistoryProvider(agent.InMemoryHistoryProviderConfig{
				ProvideOutputMessageFilter: withoutApprovals,
			}),
		},
	})
	session, err := a.CreateSession(ctx)
	if err != nil {
		log.Fatal(err)
	}
	for _, q := range []string{"Save a note: buy milk.", "Save another: call Ada."} {
		resp, err := a.RunText(ctx, q, agent.WithSession(session)).Collect()
		if err != nil {
			log.Fatal(err)
		}
		var answers []message.Content
		for c := range resp.Contents() {
			if req, ok := c.(*message.ToolApprovalRequestContent); ok {
				answers = append(answers, req.CreateResponse(true, ""))
			}
		}
		if len(answers) > 0 {
			if resp, err = a.RunMessage(ctx, message.New(answers...), agent.WithSession(session)).Collect(); err != nil {
				log.Fatal(err)
			}
		}
		fmt.Println(resp)
	}
}
```

The filter only changes what is loaded; the stored transcript still contains the approval messages.

### Message sources and filters

Each message carries a `Source` (`message.Source{ID, Type}`) saying where it came from. Your input is external (the zero value), messages loaded from history are stamped `agent.SourceTypeHistoryProvider`, context added by a context provider `agent.SourceTypeContextProvider`, and messages added by middleware `agent.SourceTypeMiddleware`. Providers use this to avoid feeding their own output back in: by default the history provider does not store messages it loaded, and context providers search and store only external messages.

`message/messagefilter` has the filters, all of the form `func(ctx, []*message.Message) ([]*message.Message, error)`:

| Filter | Keeps |
| --- | --- |
| `messagefilter.ExternalOnly` | messages with no source type (your input) |
| `messagefilter.PassThrough` | everything |
| `messagefilter.None` | nothing |
| `messagefilter.NotSourceTypes(types...)` | messages whose source type is not listed |
| `messagefilter.Sources(srcs...)`, `NotSources(srcs...)` | messages from, or not from, specific sources |
| `messagefilter.And(f...)`, `Or(f...)` | composition |

A common use is to keep retrieved documents out of the stored transcript: with a [text-search provider](#retrieval-with-textsearch) the retrieved context is a context-provider message, and you may not want it stored with every turn.

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/message/messagefilter"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/openai/openai-go/v3"
)

func main() {
	ctx := context.Background()

	// Seed every new session with an example exchange, and never store
	// messages that context providers or middleware added.
	history := agent.NewInMemoryHistoryProvider(agent.InMemoryHistoryProviderConfig{
		StateKey: "support-history",
		StateInitializer: func(*agent.Session) []*message.Message {
			q := message.NewText("Can I return an opened tent?")
			a := message.NewText("Yes, within 30 days, if it is clean and dry.")
			a.Role = message.RoleAssistant
			return []*message.Message{q, a}
		},
		StoreInputRequestMessageFilter: messagefilter.NotSourceTypes(
			agent.SourceTypeHistoryProvider, agent.SourceTypeContextProvider, agent.SourceTypeMiddleware,
		),
	})

	a := openaiprovider.NewChatCompletionsAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:  "%%model=openai-small%%",
		Config: agent.Config{HistoryProvider: history},
	})
	session, err := a.CreateSession(ctx)
	if err != nil {
		log.Fatal(err)
	}
	resp, err := a.RunText(ctx, "And sleeping bags?", agent.WithSession(session)).Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)

	msgs, err := session.InMemoryHistory("support-history")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("stored:", len(msgs)) // the two seed messages, the question and the answer
}
```

### Writing a history store

To keep transcripts in a database, a cache or files, build a provider with `agent.NewHistoryProvider` and two hooks. `Provide` returns the earlier messages in chronological order; `Store` receives the run's request and response messages after a successful run (it is skipped when the run fails). Keep only a key in the session, so the session stays small:

```go
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"

	"github.com/google/uuid"
	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/openai/openai-go/v3"
)

const keyConversation = "filestore.conversation"

// fileStore keeps one JSON file per conversation; the session holds only the conversation ID.
type fileStore struct {
	dir string
	mu  sync.Mutex
}

func (f *fileStore) conversationID(s *agent.Session) string {
	var id string
	if ok, _ := s.Get(keyConversation, &id); !ok {
		id = uuid.NewString()
		s.Set(keyConversation, id)
	}
	return id
}

func (f *fileStore) load(id string) ([]*message.Message, error) {
	data, err := os.ReadFile(filepath.Join(f.dir, id+".json"))
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var msgs []*message.Message
	return msgs, json.Unmarshal(data, &msgs)
}

func (f *fileStore) provide(ctx context.Context, inv agent.InvokingContext) ([]*message.Message, error) {
	s, _ := agent.GetOption(inv.Options, agent.WithSession)
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.load(f.conversationID(s))
}

func (f *fileStore) store(ctx context.Context, inv agent.InvokedContext) error {
	s, _ := agent.GetOption(inv.Options, agent.WithSession)
	f.mu.Lock()
	defer f.mu.Unlock()
	id := f.conversationID(s)
	msgs, err := f.load(id)
	if err != nil {
		return err
	}
	msgs = append(msgs, inv.RequestMessages...)
	msgs = append(msgs, inv.ResponseMessages...)
	data, err := json.Marshal(msgs)
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(f.dir, id+".json"), data, 0o600)
}

func main() {
	dir, err := os.MkdirTemp("", "conversations")
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = os.RemoveAll(dir) }()
	store := &fileStore{dir: dir}

	a := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:              "%%model=openai%%",
		DisableStoreOutput: true, // a custom history provider and service-held history conflict
		Config: agent.Config{
			HistoryProvider: agent.NewHistoryProvider(agent.HistoryProviderConfig{
				SourceID: "filestore",
				Provide:  store.provide,
				Store:    store.store,
			}),
		},
	})

	ctx := context.Background()
	session, err := a.CreateSession(ctx)
	if err != nil {
		log.Fatal(err)
	}
	for _, q := range []string{"Tell me a joke about pirates.", "Now tell it as a limerick."} {
		resp, err := a.RunText(ctx, q, agent.WithSession(session)).Collect()
		if err != nil {
			log.Fatal(err)
		}
		fmt.Println(resp)
	}

	data, _ := json.Marshal(session)
	fmt.Println("session:", string(data)) // only the conversation ID
	files, _ := os.ReadDir(dir)
	fmt.Println("files:", len(files))
}
```

`message.Message` and its content types round-trip through `encoding/json` (content parts carry a `Type` discriminator), so any store that holds JSON can hold a transcript. A history provider cannot add instructions, tools or options; that is what context providers are for. Implement `agent.HistoryProvider` directly (`Invoking` returns the full message list, `Invoked` stores) when you need to merge or reorder messages yourself.

The upstream sample `examples/02-agents/agents/step07_3rdparty_session_storage` stores one file per message, and `examples/02-agents/memory/step05_bounded_chat_history` keeps a short window in the session and moves older messages into [vector memory](#chat-history-memory).

## Context providers

A context provider runs before each run to add context, and after it to learn from what happened. `agent.NewContextProvider` builds one from two optional hooks:

| `ContextProviderConfig` field | Default | Purpose |
| --- | --- | --- |
| `SourceID` | required (panics if empty) | identifies the provider and stamps its messages |
| `Provide(ctx, InvokingContext) ([]*message.Message, []agent.Option, error)` | none | return extra messages (appended to the request) and run options (instructions, tools, anything) |
| `Store(ctx, InvokedContext) error` | none | called after a successful run with the request and response messages |
| `ProvideInputMessageFilter` | `ExternalOnly` | which request messages `Provide` sees |
| `StoreInputRequestMessageFilter` | `ExternalOnly` | which request messages `Store` sees |
| `StoreInputResponseMessageFilter` | all | which response messages `Store` sees |

Providers run in the order listed in `Config.ContextProviders`, each seeing the previous one's additions, and run once per run even when the tool loop calls the model several times. A `Provide` error fails the run before the model is called; a `Store` error is returned from the run after the response has streamed. Keep state in the session, as here, where a provider remembers the user's name across runs and turns it into instructions:

```go
package main

import (
	"context"
	"fmt"
	"log"
	"regexp"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/openai/openai-go/v3"
)

const keyUserName = "profile.name"

var nameRE = regexp.MustCompile(`(?i)\bmy name is (\w+)`)

var profile = agent.NewContextProvider(agent.ContextProviderConfig{
	SourceID: "profile",
	Provide: func(ctx context.Context, inv agent.InvokingContext) ([]*message.Message, []agent.Option, error) {
		s, _ := agent.GetOption(inv.Options, agent.WithSession)
		var name string
		if ok, _ := s.Get(keyUserName, &name); ok {
			return nil, []agent.Option{agent.WithInstructions("The user's name is " + name + ". Use it.")}, nil
		}
		return nil, []agent.Option{agent.WithInstructions("Ask for the user's name if you do not know it.")}, nil
	},
	Store: func(ctx context.Context, inv agent.InvokedContext) error {
		s, _ := agent.GetOption(inv.Options, agent.WithSession)
		for _, m := range inv.RequestMessages {
			if match := nameRE.FindStringSubmatch(m.String()); match != nil {
				s.Set(keyUserName, match[1])
			}
		}
		return nil
	},
})

func main() {
	ctx := context.Background()
	a := openaiprovider.NewChatCompletionsAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:  "%%model=openai-small%%",
		Config: agent.Config{ContextProviders: []agent.ContextProvider{profile}},
	})

	session, err := a.CreateSession(ctx)
	if err != nil {
		log.Fatal(err)
	}
	for _, q := range []string{"Hello, my name is Grace.", "What's my name?"} {
		if _, err := a.RunText(ctx, q, agent.WithSession(session)).Collect(); err != nil {
			log.Fatal(err)
		}
	}
	var name string
	ok, _ := session.Get(keyUserName, &name)
	fmt.Println("remembered:", ok, name)
}
```

`Provide` can also return tools, which the tool loop runs like any other; this is how a provider gives the model on-demand access to something (the text-search and memory providers below do it). Implement `agent.ContextProvider` directly when you need more than additions: its `Invoking` receives the full message list and options and returns replacements, so it can remove, reorder or rewrite messages, as compaction does. Treat the incoming slices and messages as read-only and return new ones. `agent.ContextProviderMiddleware(p)` wraps a context provider as agent middleware, for the rare case where it must run at a particular point in a middleware chain rather than in the agent lifecycle.

Context providers are trusted completely: they can add system messages, which the model weighs most heavily. Anything a provider fetches from a database, a search index or a memory service is a route for indirect prompt injection, so sanitise it and control who can write to those stores.

## Compaction

`agent/compaction` keeps the history sent to the model within budget. The transcript is grouped into atomic units in a `MessageIndex` (system messages, a user turn, an assistant reply, an assistant tool call together with its results, a summary), and strategies mark old groups as excluded or replace them with summaries. Groups are never split, so a tool call is never sent without its result.

| Strategy | What it does |
| --- | --- |
| `SlidingWindowStrategy` | drops the oldest user turns, keeping `MinimumPreservedTurns` |
| `TruncationStrategy` | drops the oldest non-system groups, keeping `MinimumPreservedGroups` |
| `ToolResultStrategy` | collapses old tool-call groups into one short assistant summary each (`DefaultToolCallFormatter`, or your `ToolCallFormatter`) |
| `SummarizationStrategy` | sends older groups to a `Summarizer` (typically another, cheaper agent) and inserts the summary; on failure it changes nothing |
| `ContextWindowStrategy` | sized from `MaxContextWindowTokens` and `MaxOutputTokens`: tool-result eviction at 50% of the input budget, truncation at 80% |
| `PipelineStrategy` | runs strategies in order, gentlest first |

Each strategy has a `Trigger` (when to start) and an optional `Target` (when to stop); triggers are functions of the index: `MessagesExceed(n)`, `TokensExceed(n)`, `TokensBelow(n)`, `TurnsExceed(n)`, `GroupsExceed(n)`, `HasToolCalls()`, `Always()`, `Never()`, combined with `All(...)` and `Any(...)`. Token counts are estimated from UTF-8 byte counts unless you supply a `TokenCounter` (any type with `CountTokens(string) int`). System messages are always kept.

There are two places to apply it:

- **`compaction.NewContextProvider`** compacts what is sent on each run but leaves the stored transcript whole. It keeps its index in the session so it updates incrementally, and skips service-managed sessions.
- **`compaction.NewHistoryProvider`** replaces the default history provider and compacts the stored transcript itself, either before history is loaded (`HistoryProviderTriggerEventBeforeMessagesRetrieval`, the default) or after new messages are stored (`HistoryProviderTriggerEventAfterMessageAdded`). Excluded messages are gone for good.

```go
package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/agent/compaction"
	"github.com/microsoft/agent-framework-go/message"
	"github.com/microsoft/agent-framework-go/message/messagefilter"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/openai/openai-go/v3"
)

// words is a crude TokenCounter; use a real tokenizer for production budgets.
type words struct{}

func (words) CountTokens(s string) int { return len(strings.Fields(s)) }

func main() {
	ctx := context.Background()
	client := openai.NewClient()

	// A cheaper agent writes the summaries.
	summariser := openaiprovider.NewChatCompletionsAgent(client, openaiprovider.AgentConfig{
		Model:        "%%model=openai-small%%",
		Instructions: "Summarise the conversation in three sentences, keeping names and numbers.",
	})
	summarise := compaction.SummarizerFunc(func(ctx context.Context, msgs []*message.Message) (string, error) {
		resp, err := summariser.Run(ctx, msgs).Collect()
		if err != nil {
			return "", err
		}
		return resp.String(), nil
	})

	strategy := &compaction.PipelineStrategy{Strategies: []compaction.Strategy{
		&compaction.ToolResultStrategy{Trigger: compaction.HasToolCalls(), MinimumPreservedGroups: new(4)},
		&compaction.SummarizationStrategy{Trigger: compaction.TurnsExceed(4), Summarizer: summarise, MinimumPreservedGroups: new(2)},
		&compaction.SlidingWindowStrategy{Trigger: compaction.TurnsExceed(6), MinimumPreservedTurns: new(3)},
	}}

	var sent []int // how many messages reach the model on each run

	a := openaiprovider.NewChatCompletionsAgent(client, openaiprovider.AgentConfig{
		Model: "%%model=openai%%",
		Config: agent.Config{
			ContextProviders: []agent.ContextProvider{
				compaction.NewContextProvider(compaction.ContextProviderConfig{Strategy: strategy, TokenCounter: words{}}),
				// Runs after compaction: record how many messages the model will see.
				agent.NewContextProvider(agent.ContextProviderConfig{
					SourceID:                  "counter",
					ProvideInputMessageFilter: messagefilter.PassThrough,
					Provide: func(ctx context.Context, inv agent.InvokingContext) ([]*message.Message, []agent.Option, error) {
						sent = append(sent, len(inv.Messages))
						return nil, nil, nil
					},
				}),
			},
		},
	})

	session, err := a.CreateSession(ctx)
	if err != nil {
		log.Fatal(err)
	}
	for i := 1; i <= 8; i++ {
		if _, err := a.RunText(ctx, fmt.Sprintf("Message number %d about camping.", i), agent.WithSession(session)).Collect(); err != nil {
			log.Fatal(err)
		}
	}
	stored, _ := session.InMemoryHistory("")
	fmt.Println("messages sent per run:", sent)
	fmt.Println("messages stored:", len(stored)) // the full transcript is kept

	// Compaction can also be applied to any message list directly.
	window := &compaction.SlidingWindowStrategy{MinimumPreservedTurns: new(2)}
	kept, err := compaction.Compact(ctx, window, stored, words{})
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("after an ad hoc sliding window:", len(kept))
}

```

Summaries are stamped as history messages, so the default history store does not save them as if they were part of the conversation. `ContextWindowStrategy` is the simplest choice when you know the model's limits: `&compaction.ContextWindowStrategy{MaxContextWindowTokens: 200_000, MaxOutputTokens: 16_000}`.

## Retrieval with `textsearch`

`agent/textsearch` turns any search function into a context provider for retrieval-augmented generation. You supply `func(ctx, query string) ([]textsearch.Result, error)`; the provider calls it and adds the results, with source names and links, as a context message. Two modes:

| `SearchTime` | Behaviour |
| --- | --- |
| `textsearch.BeforeInvoke` (default) | search before every run with the user's message (plus recent messages, if memory is on), and add the results as a user message |
| `textsearch.OnDemandFunctionCalling` | add a `Search` tool (name and description configurable) taking `userQuestion`; the model decides when to search |

Other settings: `ContextPrompt` and `CitationsPrompt` (the text before and after the results; an explicit empty string removes them), `ContextFormatter` (format the results yourself), `RecentMessageMemoryLimit` and `RecentMessageRolesIncluded` (prepend the last few messages to automatic queries, so "and the other one?" still finds something), and the usual filters and logging. A failing search in automatic mode leaves the run unchanged and is logged; in on-demand mode the error goes back to the model as a tool error.

The search function can query anything: a vector store, a SQL full-text index, Azure AI Search, a web API. Here it is an in-memory vector collection:

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/google/uuid"
	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/agent/textsearch"
	"github.com/microsoft/agent-framework-go/embedding"
	"github.com/microsoft/agent-framework-go/message/messagefilter"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/vectordata"
	"github.com/microsoft/agent-framework-go/vectordata/inmemory"
	"github.com/openai/openai-go/v3"
)

type Doc struct {
	ID     uuid.UUID `vectordata:"key"`
	Text   string    `vectordata:"vector,dimensions=256"` // embedded from the text itself
	Title  string    `vectordata:"data"`
	Source string    `vectordata:"data"`
}

func main() {
	ctx := context.Background()
	client := openai.NewClient()
	embedder := openaiprovider.NewEmbeddingGenerator(client, openaiprovider.EmbeddingGeneratorConfig{
		Model: "text-embedding-3-small", Dimensions: new(256),
	})

	docs, err := inmemory.NewCollection[uuid.UUID, Doc]("docs", &vectordata.Definition{EmbeddingGenerator: embedding.Bind(embedder)})
	if err != nil {
		log.Fatal(err)
	}
	if err := docs.EnsureExists(ctx); err != nil {
		log.Fatal(err)
	}
	if err := docs.Upsert(ctx,
		Doc{ID: uuid.New(), Title: "Returns", Source: "https://example.com/returns", Text: "Items may be returned within 30 days of delivery, unused and in original packaging."},
		Doc{ID: uuid.New(), Title: "Shipping", Source: "https://example.com/shipping", Text: "Standard shipping takes 3 to 5 working days and is free over 50 pounds."},
		Doc{ID: uuid.New(), Title: "Tent care", Source: "https://example.com/tents", Text: "Clean tent fabric with lukewarm water and let it dry fully before storage."},
	); err != nil {
		log.Fatal(err)
	}

	search := func(ctx context.Context, query string) ([]textsearch.Result, error) {
		var out []textsearch.Result
		for hit, err := range docs.Search(ctx, query, 2, nil) {
			if err != nil {
				return nil, err
			}
			out = append(out, textsearch.Result{SourceName: hit.Record.Title, SourceLink: hit.Record.Source, Text: hit.Record.Text})
		}
		return out, nil
	}

	rag := textsearch.NewContextProvider(search, textsearch.Config{RecentMessageMemoryLimit: 4})

	a := openaiprovider.NewChatCompletionsAgent(client, openaiprovider.AgentConfig{
		Model:        "%%model=openai%%",
		Instructions: "Answer from the provided context and cite the source.",
		Config: agent.Config{
			ContextProviders: []agent.ContextProvider{rag},
			// Store the conversation, not the retrieved documents.
			HistoryProvider: agent.NewInMemoryHistoryProvider(agent.InMemoryHistoryProviderConfig{
				StoreInputRequestMessageFilter: messagefilter.NotSourceTypes(agent.SourceTypeContextProvider, agent.SourceTypeHistoryProvider),
			}),
		},
	})

	session, err := a.CreateSession(ctx)
	if err != nil {
		log.Fatal(err)
	}
	resp, err := a.RunText(ctx, "How long do I have to return unused items?", agent.WithSession(session)).Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)

	// The same search as a tool the model calls when it wants to.
	onDemand := textsearch.NewContextProvider(search, textsearch.Config{
		SearchTime:       textsearch.OnDemandFunctionCalling,
		FunctionToolName: new("search_help_centre"),
	})
	b := openaiprovider.NewChatCompletionsAgent(client, openaiprovider.AgentConfig{
		Model:  "%%model=openai%%",
		Config: agent.Config{ContextProviders: []agent.ContextProvider{onDemand}},
	})
	resp, err = b.RunText(ctx, "Is shipping free?").Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)
}
```

With the default prompts, the message added before the run looks like this (the model then sees it after the user's question):

```text
## Additional Context
Consider the following information from source documents when responding to the user:
SourceDocName: Returns
SourceDocLink: https://example.com/returns
Contents: Items may be returned within 30 days of delivery, unused and in original packaging.
----
SourceDocName: Shipping
...
Include citations to the source document with document name and link if document name and link is available.
```

The upstream `examples/02-agents/rag` samples add a custom vector store, a custom data source, Foundry's hosted file search, and a Neo4j graph.

### Vector stores and embeddings

The vector store abstraction in `vectordata` is the Go counterpart of .NET's `Microsoft.Extensions.VectorData`: typed collections described by `vectordata` struct tags (`key`, `data`, `vector,dimensions=N`), with `Upsert`, `Get`, `Delete`, filtered retrieval and vector search. A `vector` field of type `string` is embedded with the collection's `EmbeddingGenerator` (an `embedding.Bind(generator)`), so you store and search by text.

| Store | Package | Notes |
| --- | --- | --- |
| in memory | `vectordata/inmemory` | `inmemory.New(Config{EmbeddingGenerator})` or `inmemory.NewCollection[K, R](name, def)`; exact search; `Export` and `Import` JSON snapshots |
| Azure AI Search | `provider/azureprovider` | `azureprovider.NewAISearch(endpoint, AISearchConfig{Credential or APIKey, ...})` |
| Azure Cosmos DB (NoSQL) | `provider/azureprovider` | `azureprovider.NewCosmosDB(client, database, CosmosDBConfig{...})`, with your `*azcosmos.Client` |

A memory or retrieval provider runs unchanged on a hosted store. Chat-history memory on Azure AI Search, with embeddings from a Foundry resource:

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/Azure/azure-sdk-for-go/sdk/azidentity"
	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/agent/memory"
	"github.com/microsoft/agent-framework-go/embedding"
	"github.com/microsoft/agent-framework-go/provider/azureprovider"
	"github.com/microsoft/agent-framework-go/provider/foundryprovider"
)

func main() {
	ctx := context.Background()
	cred, err := azidentity.NewDefaultAzureCredential(nil)
	if err != nil {
		log.Fatal(err)
	}
	project := os.Getenv("FOUNDRY_PROJECT_ENDPOINT")

	embedder := foundryprovider.NewEmbeddingGenerator(project, cred,
		foundryprovider.EmbeddingGeneratorConfig{Model: os.Getenv("FOUNDRY_EMBEDDING_MODEL")})
	search, err := azureprovider.NewAISearch(os.Getenv("AZURE_SEARCH_ENDPOINT"), azureprovider.AISearchConfig{
		Credential:         cred, // or APIKey: azcore.NewKeyCredential(key)
		EmbeddingGenerator: embedding.Bind(embedder),
	})
	if err != nil {
		log.Fatal(err)
	}

	user := "user-42"
	memories, err := memory.NewHistoryProvider(search.Store, "chat-memory", 3072,
		func(*agent.Session) *memory.State {
			return &memory.State{StorageScope: memory.Scope{UserID: &user}}
		}, memory.Config{})
	if err != nil {
		log.Fatal(err)
	}

	a := foundryprovider.NewAgent(project, cred, foundryprovider.ModelDeployment(os.Getenv("FOUNDRY_MODEL")),
		foundryprovider.AgentConfig{
			DisableStoreOutput: true,
			Config:             agent.Config{ContextProviders: []agent.ContextProvider{memories}},
		})
	resp, err := a.RunText(ctx, "What did I say about my travel plans?").Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)
}
```

Here the search scope defaults to the storage scope, so the user's memories are searched across all their sessions. 3072 is the dimension count of `text-embedding-3-large`; match it to your embedding deployment.

Embedding generators come from the providers (see [providers: embeddings](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_providers/#embeddings)) or from your own type implementing `embedding.Generator[string, *embedding.Of[float32]]`.

## Chat history memory

`agent/memory` stores every conversation in a vector collection and retrieves related earlier messages, across sessions, as context. It answers "what did this user tell me last week?" without keeping last week's transcript in the prompt. Memories are partitioned by a `memory.Scope` (`ApplicationID`, `AgentID`, `UserID`, `SessionID`, each an optional `*string`): each session's state says which scope to store under and which to search.

`memory.NewHistoryProvider(store, collectionName, dimensions, stateInitializer, config)` returns a context provider (despite the name, it is attached through `ContextProviders`). The store must support map records, UUID keys and embedding strings, as the in-memory store and the Azure stores do. The collection is created on first use.

| `memory.Config` field | Default | Purpose |
| --- | --- | --- |
| `SearchTime` | `memory.BeforeInvoke` | search before each run, or `OnDemandFunctionCalling` to add a `Search` tool |
| `MaxResults *int` | 3 | memories per search |
| `ContextPrompt *string` | a default memory prompt | text before the retrieved messages |
| `StateKey *string` | `ChatHistoryMemoryProvider` | session key for the scopes |
| `SearchInputMessageFilter`, `StorageInputRequestMessageFilter`, `StorageInputResponseMessageFilter` | external only, external only, all | what is searched with and stored |
| `Logger`, `LogSensitiveData`, `Redactor` | off | diagnostics; user IDs and text are redacted unless `LogSensitiveData` |

```go
package main

import (
	"context"
	"fmt"
	"log"

	"github.com/google/uuid"
	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/agent/memory"
	"github.com/microsoft/agent-framework-go/embedding"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/microsoft/agent-framework-go/vectordata/inmemory"
	"github.com/openai/openai-go/v3"
)

func main() {
	ctx := context.Background()
	client := openai.NewClient()
	embedder := openaiprovider.NewEmbeddingGenerator(client, openaiprovider.EmbeddingGeneratorConfig{
		Model: "text-embedding-3-small", Dimensions: new(256),
	})
	store := inmemory.New(inmemory.Config{EmbeddingGenerator: embedding.Bind(embedder)})

	userID := "user-42"
	memories, err := memory.NewHistoryProvider(store.Store, "chat_memory", 256,
		func(*agent.Session) *memory.State {
			// Store under this user and a fresh session ID; search across all of the user's sessions.
			return &memory.State{
				StorageScope: memory.Scope{UserID: &userID, SessionID: new(uuid.NewString())},
				SearchScope:  &memory.Scope{UserID: &userID},
			}
		},
		memory.Config{MaxResults: new(2)})
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = memories.Close() }()

	a := openaiprovider.NewAgent(client, openaiprovider.AgentConfig{
		Model:              "%%model=openai%%",
		DisableStoreOutput: true,
		Config:             agent.Config{ContextProviders: []agent.ContextProvider{memories}},
	})

	// Two separate conversations; the second one can recall the first.
	for i, q := range []string{"I'm vegetarian and I love Thai food.", "Suggest a dinner I would like."} {
		session, err := a.CreateSession(ctx)
		if err != nil {
			log.Fatal(err)
		}
		resp, err := a.RunText(ctx, q, agent.WithSession(session)).Collect()
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("session %d: %s\n", i+1, resp)
	}
}
```

Storage and automatic-search failures are logged (when a logger is set) and otherwise ignored, so a memory outage does not stop the agent. An empty scope searches every stored conversation; always set at least `UserID` in a multi-user service. The stored messages are user data under your retention rules.

## Foundry memory

`foundryprovider.NewMemoryProvider` is a context provider over a Microsoft Foundry memory store. Foundry extracts memories from the conversation with a chat model and an embedding model on the service side; the provider searches them before each run and submits each run's messages afterwards.

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/Azure/azure-sdk-for-go/sdk/azidentity"
	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/provider/foundryprovider"
)

func main() {
	ctx := context.Background()
	endpoint := os.Getenv("FOUNDRY_PROJECT_ENDPOINT")
	cred, err := azidentity.NewDefaultAzureCredential(nil)
	if err != nil {
		log.Fatal(err)
	}

	// The scope partitions memories; here, one partition per user ID kept in the session.
	scope := func(s *agent.Session) string {
		var user string
		if ok, _ := s.Get("user", &user); ok {
			return user
		}
		return "anonymous"
	}
	mem := foundryprovider.NewMemoryProvider(endpoint, cred, "customer-memories", scope,
		foundryprovider.MemoryProviderConfig{MaxMemories: new(int32(5))})

	// Create the store if it does not exist yet (idempotent).
	desc := "Preferences and facts about customers"
	if err := mem.EnsureMemoryStoreCreated(ctx, os.Getenv("FOUNDRY_MODEL"), os.Getenv("FOUNDRY_EMBEDDING_MODEL"), &desc); err != nil {
		log.Fatal(err)
	}

	a := foundryprovider.NewAgent(endpoint, cred, foundryprovider.ModelDeployment(os.Getenv("FOUNDRY_MODEL")),
		foundryprovider.AgentConfig{
			Instructions: "You are a helpful travel assistant.",
			Config:       agent.Config{ContextProviders: []agent.ContextProvider{mem}},
		})

	session, err := a.CreateSession(ctx)
	if err != nil {
		log.Fatal(err)
	}
	session.Set("user", "customer-1001")
	resp, err := a.RunText(ctx, "I prefer aisle seats and I'm allergic to peanuts.", agent.WithSession(session)).Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)

	// Right to be forgotten: delete everything stored for this scope.
	if err := mem.EnsureStoredMemoriesDeleted(ctx, session); err != nil {
		log.Fatal(err)
	}
}
```

`MemoryProviderConfig` also has `ContextPrompt` (default "## Memories\nConsider the following memories when answering user questions:"), `UpdateDelay` (seconds Foundry waits before extracting, default 0), the three message filters and `ClientOptions`. A blank scope panics when used.

## The agent harness

The `agent/harness` packages are context providers and middleware for long-running, tool-heavy agents. Each can be used on its own, or `harness.AgentConfig` assembles them with sensible defaults.

### Todos, modes and file memory

| Package | What the model gets | State |
| --- | --- | --- |
| `harness/todo` (`todo.New(opts)`) | tools `todos_add`, `todos_complete`, `todos_remove`, `todos_get_remaining`, `todos_get_all`, instructions, and the current list before each run | in the session; read it with `p.AllTodos(s)`, `p.RemainingTodos(s)` |
| `harness/agentmode` (`agentmode.New(cfg)`) | instructions for the current mode, and `mode_get` and `mode_set` tools; modes default to `plan` and `execute` | in the session; `p.Mode(s)`, `p.SetMode(s, mode, quiet)` |
| `harness/filememory` (`filememory.New(root, cfg)`) | tools `file_memory_write`, `_read`, `_delete`, `_ls`, `_grep`, `_replace`, `_replace_lines`, and a `memories.md` index before each run | files under `root`, confined to it (directories `0700`, files `0600`); the session selects a working folder |

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/agent/harness/agentmode"
	"github.com/microsoft/agent-framework-go/agent/harness/filememory"
	"github.com/microsoft/agent-framework-go/agent/harness/todo"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/openai/openai-go/v3"
)

func main() {
	ctx := context.Background()
	root := filepath.Join(os.TempDir(), "agent-memory")

	todos := todo.New(nil)
	modes := agentmode.New(agentmode.Config{
		Modes: []agentmode.Mode{
			{Name: "plan", Instructions: "Plan the work as todos. Do not change anything yet."},
			{Name: "execute", Instructions: "Work through the remaining todos one by one."},
		},
	})
	files, err := filememory.New(root, filememory.Config{
		// One folder per user, chosen from the session; the default shares the root.
		StateInitializer: func(s *agent.Session) filememory.State {
			var user string
			_, _ = s.Get("user", &user)
			return filememory.State{WorkingFolder: "users/" + user}
		},
	})
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = files.Close() }()

	a := openaiprovider.NewChatCompletionsAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai%%",
		Instructions: "You help plan trips.",
		Config: agent.Config{
			ContextProviders: []agent.ContextProvider{todos, modes, files},
		},
	})

	session, err := a.CreateSession(ctx)
	if err != nil {
		log.Fatal(err)
	}
	session.Set("user", "ada")
	resp, err := a.RunText(ctx, "Plan a weekend in York.", agent.WithSession(session)).Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)

	fmt.Println("mode:", modes.Mode(session))
	for _, t := range todos.AllTodos(session) {
		fmt.Printf("todo %d: %s (done=%v)\n", t.ID, t.Title, t.IsComplete)
	}
	if err := modes.SetMode(session, "execute", false); err != nil {
		log.Fatal(err)
	}
	fmt.Println("mode now:", modes.Mode(session))
}
```

The stand-in model calls the first tool it is offered, which here is `todos_add`, so the run above ends with one todo in the session.

### Skills

`agent/skills` implements Agent Skills: packages of instructions, resources and scripts that the model loads only when a task needs them, which keeps the base prompt short. The provider lists each skill's name and description in the instructions and adds three tools, `load_skill`, `read_skill_resource` and `run_skill_script`. **All three require approval by default**; switch that off per tool with `DisableLoadSkillApproval`, `DisableReadSkillResourceApproval` and `DisableRunSkillScriptApproval`, or auto-approve with `skills.ReadOnlyToolsAutoApprovalRule` or `skills.AllToolsAutoApprovalRule` in a [`toolapproval`](/microsoft-agent-framework-guide/go/microsoft_agent_framework_go_tools/#standing-approvals-with-toolapproval) middleware.

Skills come from code (`&skills.Skill{Frontmatter, GetContent, Resources, Scripts}`), from directories of `SKILL.md` files (`fsskills.NewSource(os.DirFS("skills"))`), or from any `skills.Source`. File-based scripts need a `ScriptRunner`, set through `fsskills.NewSourceOptions`; none is installed by default.

```go
package main

import (
	"context"
	"fmt"
	"log"
	"strconv"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/agent/harness/toolapproval"
	"github.com/microsoft/agent-framework-go/agent/skills"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/openai/openai-go/v3"
)

var converter = &skills.Skill{
	Frontmatter: skills.Frontmatter{
		Name:        "unit-converter",
		Description: "Convert miles, kilometres, pounds and kilograms.",
	},
	GetContent: func(context.Context) (string, error) {
		return "Read the factors resource, then run the convert script with [value, factor].", nil
	},
	Resources: []skills.Resource{{
		Name: "factors", Description: "Conversion factors.",
		Read: func(context.Context) (any, error) { return "miles->km 1.60934; kg->lb 2.20462", nil },
	}},
	Scripts: []skills.Script{{
		Name: "convert", Description: "Multiply a value by a factor. Arguments: [value, factor].",
		Run: func(_ context.Context, _ *skills.Skill, args []string) (any, error) {
			if len(args) != 2 {
				return nil, fmt.Errorf("want 2 arguments, got %d", len(args))
			}
			v, err := strconv.ParseFloat(args[0], 64)
			if err != nil {
				return nil, err
			}
			f, err := strconv.ParseFloat(args[1], 64)
			if err != nil {
				return nil, err
			}
			return v * f, nil
		},
	}},
}

func main() {
	provider := skills.NewContextProvider(skills.ContextProviderOptions{Skills: []*skills.Skill{converter}})

	a := openaiprovider.NewAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:              "%%model=openai%%",
		DisableStoreOutput: true,
		Config: agent.Config{
			ContextProviders: []agent.ContextProvider{provider},
			// Loading skills and reading resources needs no person; scripts still do.
			Middlewares: []agent.Middleware{toolapproval.New(toolapproval.Config{
				AutoApprovalRules: []toolapproval.AutoApprovalRule{skills.ReadOnlyToolsAutoApprovalRule},
			})},
		},
	})

	ctx := context.Background()
	session, err := a.CreateSession(ctx)
	if err != nil {
		log.Fatal(err)
	}
	resp, err := a.RunText(ctx, "How many kilometres is a marathon?", agent.WithSession(session)).Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)
}
```

The provider caches the skill list and the generated tools; `RefreshInterval`, `DisableCaching` and `CacheIsolationKeySelector` control that. `SkillFilter` hides skills, and `IncludeDetailedErrors` returns script error text to the model (only for trusted scripts, since error text can carry injected instructions).

### `harness.AgentConfig`

`harness.AgentConfig(settings, agentConfig)` returns an `agent.Config` with todos, modes, file memory, skills, hosted web search, standing tool approvals, OpenTelemetry tracing, message injection and per-call history persistence added, and harness instructions (`harness.DefaultInstructions`) placed before yours. Pass the result to any provider constructor:

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/microsoft/agent-framework-go/agent"
	"github.com/microsoft/agent-framework-go/agent/harness"
	"github.com/microsoft/agent-framework-go/provider/openaiprovider"
	"github.com/openai/openai-go/v3"
)

func main() {
	workdir, err := os.MkdirTemp("", "harness")
	if err != nil {
		log.Fatal(err)
	}
	defer func() { _ = os.RemoveAll(workdir) }()

	cfg, err := harness.AgentConfig(harness.Config{
		WorkingDirectory:     workdir, // skills are discovered here; memory goes in agent-file-memory/
		DisableWebSearch:     true,    // Chat Completions has no hosted web search for this model
		DisableOpenTelemetry: true,
	}, agent.Config{Name: "Engineer"})
	if err != nil {
		log.Fatal(err)
	}

	a := openaiprovider.NewChatCompletionsAgent(openai.NewClient(), openaiprovider.AgentConfig{
		Model:        "%%model=openai%%",
		Instructions: "Help with software engineering tasks.",
		Config:       cfg,
	})

	ctx := context.Background()
	session, err := a.CreateSession(ctx)
	if err != nil {
		log.Fatal(err)
	}
	resp, err := a.RunText(ctx, "Plan how to add rate limiting to our API.", agent.WithSession(session)).Collect()
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(resp)
	fmt.Println("context providers:", len(cfg.ContextProviders), "middlewares:", len(cfg.Middlewares))
}
```

| `harness.Config` field | Default |
| --- | --- |
| `WorkingDirectory` | the current directory; skills are discovered there, and each session gets a UUID folder under `agent-file-memory/` |
| `Instructions *string` | `harness.DefaultInstructions`; an empty string drops them |
| `DisableTodo`, `DisableAgentMode`, `DisableFileMemory`, `DisableSkills`, `DisableWebSearch`, `DisableToolApproval`, `DisableOpenTelemetry` | all false (everything on) |
| `AgentMode`, `Skills`, `ToolApproval`, `OpenTelemetry` | configuration for those components |

Call it once per agent configuration. Your middleware stays outermost, the harness's context providers run before yours, and the tool loop stays the provider's: do not add a second `toolautocall` middleware. The upstream `docs/agent-harness.md` documents every default and how it compares with the .NET harness.

## Differences from Python

- **Sessions are data.** Python's `AgentSession` (and the older `AgentThread`) is created by the agent and serialised with `to_dict`/`from_dict`; Go's `agent.Session` implements `encoding/json`, and its state bag is typed per key.
- **One history provider per agent.** Python's `history_provider` and `context_providers` map to `Config.HistoryProvider` and `Config.ContextProviders`; Go has no separate "message store factory".
- **Filters instead of flags.** What a provider sees and stores is controlled by `messagefilter` functions over message sources.
- **Compaction, text search, chat-history memory, Foundry memory, skills and the harness** all exist in Go. Mem0, Redis and Azure AI Search *context providers* from the Python packages do not; build them on `agent.NewContextProvider`, or put Azure AI Search behind `textsearch` or `memory` with `azureprovider.NewAISearch`.

The [Python sessions page](/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_sessions/) covers its threads and stores.
