---
title: "LangGraph Streaming Server (Express)"
description: "Stream LangGraph events over SSE using Express."
framework: langgraph
language: typescript
---

# LangGraph Streaming Server (Express)

> Verified against `@langchain/langgraph` %%version=@langchain/langgraph%% (checked %%versions-checked%%).

`graph.stream()` returns a promise of an async iterable, so `await` it and use `for await`. Stop the run when the client disconnects by passing an `AbortSignal`.

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
  req.on("close", () => controller.abort());

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

app.listen(8080, () => console.log("listening on 8080"));
```

For token streaming from an LLM node use `streamMode: "messages"`, which yields `[messageChunk, metadata]` tuples.

## Deployment

### Dockerfile

```dockerfile
FROM node:22-slim
WORKDIR /app
COPY package*.json ./
RUN npm ci --omit=dev
COPY . .
EXPOSE 8080
CMD ["node", "server.js"]
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
          image: ghcr.io/yourorg/langgraph-ts-stream:latest
          env: []
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


```yaml
name: deploy
on: { push: { branches: [ main ] } }
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: 22 }
      - run: npm ci --omit=dev
      - uses: docker/login-action@v3
        with: { registry: ghcr.io, username: ${{ github.actor }}, password: ${{ secrets.GITHUB_TOKEN }} }
      - uses: docker/build-push-action@v5
        with: { push: true, tags: ghcr.io/${{ github.repository }}:latest }
  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - uses: azure/k8s-deploy@v5
        with:
          action: deploy
          manifests: |
            deployment.yaml
          images: ghcr.io/${{ github.repository }}:latest
          namespace: default
```


## Security Best Practices
- Use API keys from secret stores; never embed in code
- Enable rate limiting and request size limits
- Prefer HTTPS with a reverse proxy terminating TLS
