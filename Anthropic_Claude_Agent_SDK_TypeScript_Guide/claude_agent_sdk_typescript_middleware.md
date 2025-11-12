---
layout: default
title: "Claude Agent SDK Middleware (TypeScript)"
description: "Guardrails, middleware, and tracing patterns for Claude Agent SDK (TypeScript)."
---

# Claude Agent SDK Middleware (TypeScript)


Latest: 0.68.0
Upstream: https://github.com/anthropics/anthropic-sdk-typescript | https://www.npmjs.com/package/@anthropic-ai/sdk

## Middleware Wrapper

```ts
type Next = (input: string) => Promise<string>;
type Middleware = (input: string, next: Next) => Promise<string>;

const policyMw: Middleware = async (input, next) => {
  if (/ssn|credit card/i.test(input)) throw new Error("Policy violation");
  return next(input);
};

const redactMw: Middleware = async (input, next) => {
  const out = await next(input);
  return out.replace(/\b\d{3}-\d{2}-\d{4}\b/g, "***-**-****");
};

function chain(mws: Middleware[], handler: Next): Next {
  return mws.reduceRight((acc, mw) => (input) => mw(input, acc), handler);
}
```

## Integrate with Anthropic SDK

```ts
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY! });

async function messagesHandler(input: string) {
  const msg = await client.messages.create({
    model: "claude-3-7-sonnet-2025-07-15",
    max_tokens: 512,
    messages: [{ role: "user", content: input }],
  });
  return msg.content[0].type === "text" ? msg.content[0].text : JSON.stringify(msg.content);
}

export const run = chain([policyMw, redactMw], messagesHandler);

## Streaming

```ts
const stream = await client.messages.stream({
  model: "claude-3-7-sonnet-2025-07-15",
  max_tokens: 512,
  messages: [{ role: "user", content: "Stream this response." }],
});
for await (const chunk of stream) {
  // chunk.type might be message_start, content_block_start, content_block_delta, message_delta, message_stop
  console.log(chunk);
}
```

## Deployment (Kubernetes)

deployment.yaml

{% raw %}
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: claude-agent
spec:
  replicas: 2
  selector:
    matchLabels:
      app: claude-agent
  template:
    metadata:
      labels:
        app: claude-agent
    spec:
      containers:
        - name: app
          image: ghcr.io/yourorg/claude-agent:latest
          env:
            - name: ANTHROPIC_API_KEY
              valueFrom:
                secretKeyRef:
                  name: anthropic-secrets
                  key: apiKey
          ports:
            - containerPort: 8080
---
apiVersion: v1
kind: Service
metadata:
  name: claude-agent
spec:
  selector:
    app: claude-agent
  ports:
    - port: 80
      targetPort: 8080
```

## CI/CD (GitHub Actions)

```yaml
{% raw %}
name: deploy
on:
  push:
    branches: [ main ]
jobs:
  build-and-push:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
      - run: npm ci
      - run: npm run build
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          push: true
          tags: ghcr.io/${{ github.repository }}:latest
  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    steps:
      - uses: azure/k8s-deploy@v5
        with:
          action: deploy
          manifests: |
            deployment.yaml
          images: ghcr.io/${{ github.repository }}:latest
          namespace: default
{% endraw %}
```
{% endraw %}
```