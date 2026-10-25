---
title: "LangGraph Streaming Server (Express)"
description: "Stream LangGraph TypeScript graph updates and LLM tokens to clients over server-sent events with Express, with per-thread state, cancellation and a container build."
framework: "langgraph"
language: "typescript"
sidebar:
  label: "Streaming server (Express)"
  order: 40
---

> Verified against `@langchain/langgraph` %%version=@langchain/langgraph%% (checked %%versions-checked%%).

`graph.stream()` returns a promise of an async iterable, so `await` it and use `for await`. Each chunk is written as a server-sent event. Stop the run when the client disconnects by passing an `AbortSignal`: listen for `close` on the response, which fires when the connection goes away (the request's own `close` event fires once its body has been read, which for a POST can be before streaming starts).

```ts
import express from "express";
import { StateGraph, Annotation, START, END } from "@langchain/langgraph";

const State = Annotation.Root({
  input: Annotation<string>,
  output: Annotation<string>,
});

const graph = new StateGraph(State)
  .addNode("echo", (s) => ({ output: s.input }))
  .addEdge(START, "echo")
  .addEdge("echo", END)
  .compile();

const app = express();

app.get("/stream", async (req, res) => {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  res.setHeader("Connection", "keep-alive");

  const controller = new AbortController();
  res.on("close", () => controller.abort());

  try {
    const q = String(req.query.q ?? "");
    const stream = await graph.stream(
      { input: q },
      { streamMode: "updates", signal: controller.signal }
    );
    for await (const ev of stream) {
      res.write(`data: ${JSON.stringify(ev)}\n\n`);
    }
    res.write("event: done\ndata: {}\n\n");
  } catch (err) {
    if (!controller.signal.aborted) {
      res.write(`event: error\ndata: ${JSON.stringify({ message: String(err) })}\n\n`);
    }
  } finally {
    res.end();
  }
});

const port = Number(process.env.PORT ?? 8080);
app.listen(port, () => console.log(`listening on ${port}`));
```

Try it with `curl -N "localhost:8080/stream?q=hi"`:

```text
data: {"echo":{"output":"hi"}}

event: done
data: {}
```

## Token streaming with per-thread state

For a chat UI, stream LLM tokens with `streamMode: "messages"`, which yields `[messageChunk, metadata]` tuples; `metadata.langgraph_node` says which node produced the token. Add a checkpointer and take the `thread_id` from the URL so each conversation keeps its history between requests. The model here is `FakeListChatModel`, which streams its canned reply one character at a time; swap in any LangChain chat model.

```ts
import express from "express";
import { StateGraph, StateSchema, MessagesValue, START, END, MemorySaver } from "@langchain/langgraph";
import { AIMessageChunk, HumanMessage } from "@langchain/core/messages";
import { FakeListChatModel } from "@langchain/core/utils/testing";

// Stand-in for a real chat model (ChatAnthropic, ChatOpenAI, ...); it streams one character at a time.
const model = new FakeListChatModel({ responses: ["Tides are caused by the Moon."] });

const State = new StateSchema({ messages: MessagesValue });

const graph = new StateGraph(State)
  .addNode("chat", async (s) => ({ messages: [await model.invoke(s.messages)] }))
  .addEdge(START, "chat")
  .addEdge("chat", END)
  .compile({ checkpointer: new MemorySaver() });

const app = express();
app.use(express.json({ limit: "64kb" }));

app.post("/threads/:threadId/chat", async (req, res) => {
  const message = typeof req.body?.message === "string" ? req.body.message : "";
  if (!message) {
    res.status(400).json({ error: "message is required" });
    return;
  }

  res.writeHead(200, {
    "Content-Type": "text/event-stream",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
    "X-Accel-Buffering": "no", // stop nginx buffering the stream
  });

  const controller = new AbortController();
  res.on("close", () => controller.abort());

  try {
    const stream = await graph.stream(
      { messages: [new HumanMessage(message)] },
      {
        configurable: { thread_id: req.params.threadId },
        streamMode: "messages",
        signal: controller.signal,
      },
    );
    for await (const [chunk, metadata] of stream) {
      if (AIMessageChunk.isInstance(chunk) && typeof chunk.content === "string" && chunk.content) {
        res.write(`event: token\ndata: ${JSON.stringify({ node: metadata.langgraph_node, text: chunk.content })}\n\n`);
      }
    }
    res.write("event: done\ndata: {}\n\n");
  } catch (err) {
    if (!controller.signal.aborted) {
      res.write(`event: error\ndata: ${JSON.stringify({ message: (err as Error).message })}\n\n`);
    }
  } finally {
    res.end();
  }
});

const port = Number(process.env.PORT ?? 8080);
app.listen(port, () => console.log(`listening on ${port}`));
```

```bash
curl -N -X POST localhost:8080/threads/t1/chat \
  -H 'content-type: application/json' -d '{"message":"why tides?"}'
```

```text
event: token
data: {"node":"chat","text":"T"}

event: token
data: {"node":"chat","text":"i"}
...
event: done
data: {}
```

Notes:

- The model is called with `invoke`, yet tokens still stream: in `"messages"` mode LangGraph streams any chat model call made inside a node.
- Use `streamMode: ["messages", "updates"]` to send tokens and node updates on one connection; each chunk is then a `[mode, payload]` tuple.
- `MemorySaver` keeps state in process memory, which is lost on restart and not shared between replicas. With more than one replica use `PostgresSaver` from `@langchain/langgraph-checkpoint-postgres` (see the [production guide](/langgraph-guide/typescript/langchain_langgraph_production_guide/)).
- Browsers' `EventSource` only supports GET. For POST bodies, read the stream with `fetch()` and a `ReadableStream` reader, or use the `useStream` hook from `@langchain/langgraph-sdk/react` against a LangGraph server.
- Proxies may buffer the response; `X-Accel-Buffering: no` turns that off for nginx. Also raise the proxy's read timeout for long runs, or send a periodic comment line (`: ping\n\n`) to keep the connection open.

## Deployment

### Dockerfile

The examples run TypeScript directly with `tsx` during development. For the image, compile with `tsc` in a build stage (this assumes `"build": "tsc"` in `package.json` and `outDir: "dist"` in `tsconfig.json`):

```dockerfile
FROM node:22-slim AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:22-slim
WORKDIR /app
ENV NODE_ENV=production
COPY package*.json ./
RUN npm ci --omit=dev
COPY --from=build /app/dist ./dist
EXPOSE 8080
USER node
CMD ["node", "dist/server.js"]
```

### Kubernetes (deployment.yaml)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: langgraph-ts-stream
spec:
  replicas: 2
  selector:
    matchLabels:
      app: langgraph-ts-stream
  template:
    metadata:
      labels:
        app: langgraph-ts-stream
    spec:
      containers:
        - name: app
          image: ghcr.io/yourorg/langgraph-ts-stream:1.0.0   # pin a tag, not :latest
          env:
            - name: ANTHROPIC_API_KEY
              valueFrom: { secretKeyRef: { name: llm-keys, key: anthropic } }
          ports: [{ containerPort: 8080 }]
---
apiVersion: v1
kind: Service
metadata:
  name: langgraph-ts-stream
spec:
  selector: { app: langgraph-ts-stream }
  ports: [{ port: 80, targetPort: 8080 }]
```

### GitHub Actions

Build and push the image on every push to `main`. Deploying it is cluster-specific (for example `kubectl set image` after authenticating to your cluster), so it is left out here.

```yaml
name: build
on: { push: { branches: [ main ] } }
permissions: { contents: read, packages: write }
jobs:
  image:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with: { registry: ghcr.io, username: ${{ github.actor }}, password: ${{ secrets.GITHUB_TOKEN }} }
      - uses: docker/build-push-action@v6
        with: { push: true, tags: "ghcr.io/${{ github.repository }}:${{ github.sha }}" }
```

## Security

- Keep API keys in a secret store and pass them as environment variables (as in the Kubernetes manifest); never commit them.
- Limit request size (`express.json({ limit })`, as above) and add rate limiting in front of the model endpoint: each request costs money.
- Authenticate users before a request reaches the graph, and derive the `thread_id` from the authenticated user (or check that the user owns it); otherwise anyone who guesses a thread ID can read and extend that conversation.
- Terminate TLS at a reverse proxy or load balancer.
