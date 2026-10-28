---
title: "PydanticAI: Batch Processing"
description: "BatchModelRequests, model_request_batch and PendingModelResponse — send model requests through a provider's batch API at a lower price, in exchange for latency."
framework: "pydanticai"
language: "python"
sidebar:
  label: "Batch processing"
  order: 23
---

Verified against **pydantic-ai %%version=pydantic-ai%%** (checked %%versions-checked%%) — source modules: `pydantic_ai.capabilities.batch_model_requests`, `pydantic_ai.direct`, `pydantic_ai.models.batch`.

Provider batch APIs typically cost about half as much as live requests, but a result can take up to a day. They suit offline work (evaluation runs, classification of a backlog, nightly enrichment), not anything a person is waiting on. Pydantic AI offers three levels of access:

| Need | Use |
|---|---|
| Run ordinary agents, unchanged, at batch prices | `BatchModelRequests` capability |
| Send a list of bare model requests and wait for all of them | `direct.model_request_batch` |
| Submit now, collect in another process | `PendingModelResponse` output, or the `batch_*` methods on `Model` |

A model has a batch API when `model.batch_max_requests` is not `None`. The OpenAI, Google and Anthropic models implement it, as do `TestModel` and `FunctionModel`, which run each batch in memory so you can test the flow without a provider. Check `batch_max_requests` on the model you intend to use.

## Agents with `BatchModelRequests`

Add one `BatchModelRequests` instance to the agent, then run many agents or runs concurrently. Requests made through the same instance in the same process are queued and submitted together.

```python
import asyncio
from datetime import timedelta

from pydantic_ai import Agent
from pydantic_ai.capabilities import BatchModelRequests
from pydantic_ai.models.test import TestModel


async def main():
    batching = BatchModelRequests(max_requests=5_000, max_wait=timedelta(seconds=1))  # minutes, in real use
    # In real use: Agent('anthropic:%%model=anthropic-small%%', capabilities=[batching])
    agent = Agent(TestModel(custom_output_text='Paris'), capabilities=[batching])

    async with batching:
        results = await asyncio.gather(*(agent.run(f'Question {i}') for i in range(3)))
    print([r.output for r in results])
    #> ['Paris', 'Paris', 'Paris']


asyncio.run(main())
```

A batch is submitted when it holds `max_requests` requests, when its oldest request has waited `max_wait` (30 seconds by default), or when you `await batching.flush()`, whichever comes first. Using the capability as `async with` cancels anything still queued or polling when the block exits.

Other settings:

| Field | Meaning |
|---|---|
| `max_resubmissions` | How often a request that expired, or failed with a rate limit, timeout or server error, goes into a new batch. `0` raises the error instead. |
| `poll_interval`, `max_poll_interval` | Wait before the first status check; the interval then grows by half after each check, up to the maximum. |

Other errors, such as an invalid request or a cancelled batch, reach the run as usual. An agent spec can configure the capability too: `BatchModelRequests: {max_requests: 5000, max_wait: PT5M}`, with durations in seconds or ISO 8601.

Streamed requests wait for their batch as well, then deliver each part of the result in one event.

## Bare requests: `model_request_batch`

Without an agent, send a list of `BatchRequest`s. Each carries a unique `custom_id`, its own messages, and optionally its own settings and parameters. The call waits for every batch to end and returns one entry per request, in order: a `ModelResponse`, or a `ModelAPIError` in place of a request that failed.

```python
import asyncio

from pydantic_ai import ModelRequest
from pydantic_ai.direct import model_request_batch
from pydantic_ai.models.batch import BatchRequest
from pydantic_ai.models.test import TestModel


async def main():
    questions = {'a': 'What is the capital of France?', 'b': 'What is the capital of Italy?'}
    responses = await model_request_batch(
        TestModel(custom_output_text='Rome'),  # or e.g. 'anthropic:%%model=anthropic-small%%'
        [
            BatchRequest(custom_id=key, messages=[ModelRequest.user_text_prompt(question)])
            for key, question in questions.items()
        ],
    )
    for key, response in zip(questions, responses):
        print(key, response if isinstance(response, Exception) else response.text)
        #> a Rome
        #> b Rome


asyncio.run(main())
```

Requests are split into as few batches as the provider accepts. Cancelling the call, for example with `asyncio.timeout`, cancels batches that have not ended. An error creating or checking a batch is raised from the call, and a `request_timeout` setting on a request raises a `UserError`, since one request in a batch cannot be timed out separately. These requests are not instrumented with OpenTelemetry the way `model_request` is.

## Submit now, collect later

Waiting for a day inside a process is rarely what you want. Add `PendingModelResponse` to the agent's `output_type`, and the run ends as soon as a request is left waiting on the provider. The message history ends in a response with `state='suspended'` that records the batch.

```python
import asyncio

from pydantic_ai import Agent, PendingModelResponse
from pydantic_ai.capabilities import BatchModelRequests
from pydantic_ai.models.test import TestModel


async def main():
    agent = Agent(
        TestModel(custom_output_text='ok'),
        capabilities=[BatchModelRequests()],
        output_type=[str, PendingModelResponse],
    )
    result = await agent.run('Summarise the backlog')
    print(type(result.output).__name__, result.output.response.state)
    #> PendingModelResponse suspended

    # Later, in this process or another: resume from the stored history, with no new prompt.
    while not isinstance(result.output, str):
        result = await agent.run(message_history=result.all_messages())
    print(result.output)
    #> ok


asyncio.run(main())
```

Persist `result.all_messages_json()` (see [Message history](/pydanticai-guide/pydantic_ai_message_history/)) between the two steps. Running again with that history checks on the same batch rather than submitting the request again: the run continues if the response has arrived, and otherwise ends with a `PendingModelResponse` again. Resuming a suspended response whose server-side job is no longer available raises `SuspendedResponseExpired`; start a new run from the earlier messages in that case.

For finer control, the `batch_encode`, `batch_create` and related methods on `Model` are the provider-level primitives that both the capability and `model_request_batch` are built on.

## Limits to keep in mind

- Batch pricing and turnaround are the provider's, not Pydantic AI's. Check the provider's current terms.
- Agents with tools make several model requests per run, and each one waits for a batch, so a tool-heavy run takes several batch turnarounds.
- Batching needs an `asyncio` event loop, and requests only share batches within one process and one `BatchModelRequests` instance.
- The examples use `TestModel`, whose in-memory batches finish immediately. They have not been run against a live provider batch API.
