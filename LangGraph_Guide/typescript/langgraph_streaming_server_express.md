---
layout: default
title: "LangGraph Streaming Server (Express)"
description: "Stream LangGraph events over SSE using Express."
---

# LangGraph Streaming Server (Express)

Latest: 1.0.2
Last verified: 2025-11

```ts
import express from "express";
import { StateGraph, END } from "@langchain/langgraph";

const app = express();

function build(){
  const g = new StateGraph<{ input: string; output?: string }>();
  g.addNode("echo", (s) => ({ ...s, output: s.input }));
  g.setEntryPoint("echo");
  g.addEdge("echo", END);
  return g.compile();
}

const graph = build();

app.get("/stream", (req, res) => {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  const q = String(req.query.q || "");
  for (const ev of graph.stream({ input: q })) {
    res.write(`data: ${JSON.stringify(ev)}\n\n`);
  }
  res.end();
});

app.listen(8080, () => console.log("listening on 8080"));
```

## Deployment

### Dockerfile

```dockerfile
FROM node:20-slim
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
        with: { node-version: 20 }
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