---
layout: default
title: "Claude Agent SDK Advanced Multi‑Agent (TypeScript)"
description: "Advanced collaboration, guardrails, and HITL in Claude Agent SDK TypeScript."
---

# Claude Agent SDK Advanced Multi‑Agent (TypeScript)


Latest: 0.68.0
Upstream: https://github.com/anthropics/anthropic-sdk-typescript | https://anthropic.mintlify.app/en/api

## Patterns
- Specialize agents and route by capability; guardrails at boundaries
- HITL for sensitive flows; persist state for review

## Messages with Tool Use + Router

```ts
import Anthropic, { Tool } from "@anthropic-ai/sdk";

const client = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY! });

const tools: Tool[] = [
  {
    name: "get_weather",
    description: "Get weather by city",
    input_schema: {
      type: "object",
      properties: { city: { type: "string" } },
      required: ["city"],
    },
  },
];

function route(input: string) {
  return /weather|temperature|forecast/i.test(input) ? "weather" : "general";
}

export async function handle(input: string) {
  const target = route(input);
  const msg = await client.messages.create({
    model: "claude-3-7-sonnet-2025-07-15",
    max_tokens: 512,
    tools,
    messages: [{ role: "user", content: input }],
  });

  const content = msg.content[0];
  if (content.type === "tool_use") {
    const { name, input: args, id } = content;
    if (name === "get_weather") {
      const result = { temp_c: 18, condition: "cloudy", city: args.city };
      const followup = await client.messages.create({
        model: "claude-3-7-sonnet-2025-07-15",
        max_tokens: 512,
        messages: [
          { role: "user", content: input },
          { role: "assistant", content: [content] },
          { role: "tool", content: [{ type: "tool_result", tool_use_id: id, content: JSON.stringify(result) }] },
        ],
      });
      return followup.content[0].type === "text" ? followup.content[0].text : JSON.stringify(followup.content);
    }
  }
  return content.type === "text" ? content.text : JSON.stringify(msg.content);
}
```
