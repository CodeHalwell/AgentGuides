---
layout: default
title: Claude Streaming Server (Express)
description: Stream Claude responses via SSE using @anthropic-ai/sdk.
---

# Claude Streaming Server (Express)

Latest: 0.68.0
Last verified: 2025-11

```ts
import express from "express";
import Anthropic from "@anthropic-ai/sdk";

const app = express();
const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY! });

app.get("/stream", async (req, res) => {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  const q = String(req.query.q || "");
  const stream = await client.messages.stream({
    model: "claude-3-7-sonnet-2025-07-15",
    max_tokens: 512,
    messages: [{ role: "user", content: q }],
  });
  for await (const chunk of stream) {
    res.write(`data: ${JSON.stringify(chunk)}\n\n`);
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
  name: claude-ts-stream
spec:
  replicas: 2
  selector:
    matchLabels:
      app: claude-ts-stream
  template:
    metadata:
      labels:
        app: claude-ts-stream
    spec:
      containers:
        - name: app
          image: ghcr.io/yourorg/claude-ts-stream:latest
          env: [{ name: ANTHROPIC_API_KEY, valueFrom: { secretKeyRef: { name: anthropic-secrets, key: apiKey } } }]
          ports: [{ containerPort: 8080 }]
---
apiVersion: v1
kind: Service
metadata:
  name: claude-ts-stream
spec:
  selector: { app: claude-ts-stream }
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
- Use secrets from Kubernetes Secret or cloud secret managers
- Authenticate SSE endpoints; enable rate limiting and CORS controls
- Avoid logging request bodies; rotate API keys regularly
