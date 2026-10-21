---
title: "Claude Streaming Server (Express)"
description: "Stream Claude Agent SDK runs to browsers over Server-Sent Events with Express."
framework: "claude-agent-sdk"
language: "typescript"
---

Verified against `@anthropic-ai/claude-agent-sdk` %%version=@anthropic-ai/claude-agent-sdk%% (checked %%versions-checked%%).

This server runs an agent per request and forwards text deltas, tool activity and the final result to the browser as Server-Sent Events. If you only need raw token streaming from a single model call, the plain `@anthropic-ai/sdk` client's `messages.stream()` is lighter; the point of the Agent SDK here is that the agent can use tools while it streams.

```bash
npm install @anthropic-ai/claude-agent-sdk@%%version=@anthropic-ai/claude-agent-sdk%% express zod@^4
npm install --save-dev @types/express typescript tsx
```

```typescript
// server.ts
import express from 'express';
import { query } from '@anthropic-ai/claude-agent-sdk';

const app = express();

app.get('/stream', async (req, res) => {
  const prompt = String(req.query.q ?? '').slice(0, 4000);
  if (!prompt) {
    res.status(400).json({ error: 'q is required' });
    return;
  }

  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');
  res.flushHeaders();

  const send = (event: string, data: unknown) => res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);

  // Stop the agent if the browser goes away
  const controller = new AbortController();
  req.on('close', () => controller.abort());

  try {
    for await (const message of query({
      prompt,
      options: {
        model: 'claude-sonnet-5-5',
        cwd: '/srv/knowledge-base',
        allowedTools: ['Read', 'Glob', 'Grep'],
        disallowedTools: ['Bash', 'Write', 'Edit'],
        includePartialMessages: true,
        maxTurns: 15,
        maxBudgetUsd: 0.5,
        abortController: controller,
        verbatimPrompts: true,   // user text cannot trigger @file reads or slash commands
      },
    })) {
      switch (message.type) {
        case 'stream_event': {
          const e = message.event;
          if (e.type === 'content_block_delta' && e.delta.type === 'text_delta') send('delta', e.delta.text);
          break;
        }
        case 'assistant':
          for (const block of message.message.content) {
            if (block.type === 'tool_use') send('tool', { name: block.name });
          }
          break;
        case 'result':
          send('done', {
            ok: message.subtype === 'success',
            sessionId: message.session_id,
            costUsd: message.total_cost_usd,
          });
          break;
      }
    }
  } catch (err) {
    if (!controller.signal.aborted) send('error', { message: (err as Error).message });
  } finally {
    res.end();
  }
});

app.listen(8080, () => console.log('listening on 8080'));
```

A browser client:

```typescript
const source = new EventSource(`/stream?q=${encodeURIComponent('Where is rate limiting configured?')}`);
source.addEventListener('delta', (e) => {
  document.querySelector('#out')!.textContent += JSON.parse((e as MessageEvent).data);
});
source.addEventListener('tool', (e) => console.log('using', JSON.parse((e as MessageEvent).data).name));
source.addEventListener('done', () => source.close());
```

For follow-up questions, return `sessionId` to the client and pass it back as `options.resume` on the next request.

## Deployment

### Dockerfile

```dockerfile
FROM node:22-slim
RUN useradd --create-home agent
WORKDIR /app
COPY package*.json ./
RUN npm ci --omit=dev
COPY --chown=agent:agent dist ./dist
USER agent
ENV HOME=/home/agent
EXPOSE 8080
CMD ["node", "dist/server.js"]
```

Build the image on the target architecture: the SDK installs a platform-specific Claude Code binary as an optional dependency.

### Kubernetes (deployment.yaml)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: claude-agent-stream
spec:
  replicas: 2
  selector:
    matchLabels:
      app: claude-agent-stream
  template:
    metadata:
      labels:
        app: claude-agent-stream
    spec:
      containers:
        - name: app
          image: ghcr.io/yourorg/claude-agent-stream:latest
          env:
            - name: ANTHROPIC_API_KEY
              valueFrom: { secretKeyRef: { name: anthropic-secrets, key: apiKey } }
          ports: [{ containerPort: 8080 }]
          resources:
            requests: { cpu: "500m", memory: "1Gi" }
            limits: { memory: "2Gi" }
---
apiVersion: v1
kind: Service
metadata:
  name: claude-agent-stream
spec:
  selector: { app: claude-agent-stream }
  ports: [{ port: 80, targetPort: 8080 }]
```

Proxies and load balancers must not buffer `text/event-stream` responses; with NGINX ingress set `nginx.ingress.kubernetes.io/proxy-buffering: "off"` and raise the read timeout above your longest run.

## Security

- Authenticate the endpoint and rate-limit per user; each request starts a process and costs money.
- Keep the tool set read-only for public endpoints, and cap `maxTurns` and `maxBudgetUsd`.
- Use `verbatimPrompts` and treat everything the agent reads as untrusted.
- Do not log prompts or outputs that may contain personal data; log session IDs and costs instead.
