---
layout: default
title: "OpenAI Agents Streaming Server (Express)"
description: "Stream OpenAI model output in Node/Express using the OpenAI SDK."
---

# OpenAI Agents Streaming Server (Express)

Latest: 6.8.1
Last verified: 2025-11

```ts
import express from "express";
import OpenAI from "openai";

const app = express();
const client = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });

app.get("/stream", async (req, res) => {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  const q = String(req.query.q || "");
  const stream = await client.chat.completions.create({
    model: "gpt-4o-mini",
    messages: [{ role: "user", content: q }],
    stream: true,
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
  name: openai-agents-ts-stream
spec:
  replicas: 2
  selector:
    matchLabels:
      app: openai-agents-ts-stream
  template:
    metadata:
      labels:
        app: openai-agents-ts-stream
    spec:
      containers:
        - name: app
          image: ghcr.io/yourorg/openai-agents-ts-stream:latest
          env: [{ name: OPENAI_API_KEY, valueFrom: { secretKeyRef: { name: openai-secrets, key: apiKey } } }]
          ports: [{ containerPort: 8080 }]
---
apiVersion: v1
kind: Service
metadata:
  name: openai-agents-ts-stream
spec:
  selector: { app: openai-agents-ts-stream }
  ports: [{ port: 80, targetPort: 8080 }]
```

### Security Best Practices
- Require auth tokens on streaming route; set CORS and no-cache headers
- Implement rate limiting and timeouts at proxy and app levels
- Store OPENAI_API_KEY in secrets; never log request bodies/raw chunks