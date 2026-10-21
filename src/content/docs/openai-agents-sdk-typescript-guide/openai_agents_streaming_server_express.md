---
title: "OpenAI Agents Streaming Server (Express)"
description: "Stream Agents SDK runs (and raw model output) to clients over SSE from Node/Express."
framework: openai-agents-sdk-typescript
---

> Verified against `@openai/agents` %%version=@openai/agents%% (checked %%versions-checked%%). Requires `express`, `zod` and `@openai/agents@%%version=@openai/agents%%`; Node.js 22+.

## Streaming an agent run over SSE

`run(agent, input, { stream: true })` returns a stream you can iterate for events, or turn into plain text with `toTextStream()`. Tools, handoffs and guardrails still run inside the agent loop.

```ts
import express from "express";
import { Agent, run } from "@openai/agents";

const app = express();
const agent = new Agent({ name: "Assistant", instructions: "Answer concisely." });

app.get("/stream", async (req, res) => {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  const q = String(req.query.q || "");

  const controller = new AbortController();
  req.on("close", () => controller.abort()); // stop the run if the client disconnects

  const stream = await run(agent, q, { stream: true, signal: controller.signal });
  for await (const event of stream) {
    if (event.type === "raw_model_stream_event" && event.data.type === "output_text_delta") {
      res.write(`data: ${JSON.stringify({ delta: event.data.delta })}\n\n`);
    } else if (event.type === "agent_updated_stream_event") {
      res.write(`event: agent\ndata: ${JSON.stringify({ agent: event.agent.name })}\n\n`);
    }
  }
  await stream.completed;
  res.write("event: done\ndata: {}\n\n");
  res.end();
});

app.listen(8080, () => console.log("listening on 8080"));
```

## Raw model streaming without an agent loop

For a single model call, the base `openai` SDK (v7, a dependency of the Agents SDK) is enough:

```ts
import express from "express";
import OpenAI from "openai";

const app = express();
const client = new OpenAI(); // reads OPENAI_API_KEY

app.get("/raw", async (req, res) => {
  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache");
  const stream = await client.responses.create({
    model: "gpt-5.6-luna",
    input: String(req.query.q || ""),
    stream: true,
  });
  for await (const event of stream) {
    if (event.type === "response.output_text.delta") res.write(`data: ${JSON.stringify(event.delta)}\n\n`);
  }
  res.end();
});

app.listen(8081);
```

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
