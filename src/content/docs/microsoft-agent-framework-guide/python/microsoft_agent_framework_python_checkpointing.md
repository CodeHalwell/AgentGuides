---
title: "Microsoft Agent Framework (Python) — Checkpointing"
description: "Persist workflow state with FileCheckpointStorage, InMemoryCheckpointStorage, or a custom CheckpointStorage backend. Resume long-running agents across process restarts."
framework: microsoft-agent-framework
language: python
---

# Checkpointing — Python

Checkpointing captures the full execution state of a workflow at every superstep so you can pause, crash, redeploy, or hand off to a different process and pick up exactly where you left off. It's the backbone for long-running agent workflows, human-in-the-loop pauses that last days, and durable multi-agent orchestrations.

Verified against `agent-framework-core==1.1.0` (`agent_framework._workflows._checkpoint`).

## What lives in a checkpoint

`WorkflowCheckpoint` captures:

- `workflow_name` + `graph_signature_hash` — identifies the workflow definition; restoration validates the topology.
- `checkpoint_id` (auto-generated UUID) and `previous_checkpoint_id` — a chain per run.
- `timestamp` — ISO 8601, for ordering.
- `messages` — in-flight messages between executors.
- `state` — committed workflow + executor states.
- `pending_request_info_events` — outstanding HITL requests.
- `iteration_count` and `metadata`.

A checkpoint is keyed on the **workflow definition**, not a specific instance — any process running the same `WorkflowBuilder` output can restore it.

## Storage backends

All three implementations share the `CheckpointStorage` protocol (`save`, `load`, `list_checkpoints`, `delete`, `get_latest`, `list_checkpoint_ids`).

```python
from agent_framework import (
    InMemoryCheckpointStorage,
    FileCheckpointStorage,
    CheckpointStorage,
)

mem = InMemoryCheckpointStorage()
disk = FileCheckpointStorage("/var/lib/agent-framework/checkpoints")
```

For Azure Cosmos DB, install `agent-framework-azure-cosmos` and use `agent_framework_azure_cosmos.CosmosCheckpointStorage`. For Redis, install `agent-framework-redis` and pick the Redis-backed storage class.

### Custom backend

Implement the protocol — structural typing, no inheritance needed:

```python
from dataclasses import asdict
import json, aioboto3
from agent_framework import CheckpointStorage, WorkflowCheckpoint


class S3CheckpointStorage:
    def __init__(self, bucket: str, prefix: str = "checkpoints/") -> None:
        self._bucket = bucket
        self._prefix = prefix

    async def save(self, checkpoint: WorkflowCheckpoint) -> str:
        key = f"{self._prefix}{checkpoint.workflow_name}/{checkpoint.checkpoint_id}.json"
        async with aioboto3.Session().client("s3") as s3:
            await s3.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=json.dumps(asdict(checkpoint)).encode(),
            )
        return checkpoint.checkpoint_id

    async def load(self, checkpoint_id: str) -> WorkflowCheckpoint: ...
    async def list_checkpoints(self, *, workflow_name: str) -> list[WorkflowCheckpoint]: ...
    async def delete(self, checkpoint_id: str) -> bool: ...
    async def get_latest(self, *, workflow_name: str) -> WorkflowCheckpoint | None: ...
    async def list_checkpoint_ids(self, *, workflow_name: str) -> list[str]: ...
```

## Attaching storage to a workflow

### Build-time

Pass storage to the builder — every superstep saves automatically:

```python
from agent_framework import FileCheckpointStorage, WorkflowBuilder

storage = FileCheckpointStorage("/var/lib/agents/checkpoints")
workflow = (
    WorkflowBuilder(start_executor=researcher, checkpoint_storage=storage, name="research-pipeline")
    .add_edge(researcher, writer)
    .build()
)
```

Orchestration builders accept the same parameter:

```python
from agent_framework_orchestrations import SequentialBuilder, ConcurrentBuilder, MagenticBuilder

SequentialBuilder(participants=[...], checkpoint_storage=storage).build()
ConcurrentBuilder(participants=[...], checkpoint_storage=storage).build()
MagenticBuilder(participants=[...], manager_agent=..., checkpoint_storage=storage).build()
```

`HandoffBuilder` uses `.with_checkpointing(storage)` instead:

```python
from agent_framework_orchestrations import HandoffBuilder

HandoffBuilder(participants=[...]).with_checkpointing(storage).build()
```

### Run-time (no storage at build time)

Build a stateless workflow and attach storage only when resuming — useful for warm pools or cross-process rehydration:

```python
fresh = WorkflowBuilder(start_executor=researcher, name="research-pipeline").add_edge(researcher, writer).build()

result = await fresh.run(
    checkpoint_id=some_id,
    checkpoint_storage=storage,    # runtime-only attachment
)
```

## Resuming a workflow

### Resume from latest

```python
latest = await storage.get_latest(workflow_name="research-pipeline")
if latest:
    result = await workflow.run(checkpoint_id=latest.checkpoint_id)
```

### Resume and reply to pending HITL requests

Combine `responses={}` with `checkpoint_id=`:

```python
stream = workflow.run(
    checkpoint_id=latest.checkpoint_id,
    responses={"<request-id>": "approve"},
    stream=True,
)
async for event in stream:
    ...
```

### List and prune old checkpoints

```python
ids = await storage.list_checkpoint_ids(workflow_name="research-pipeline")
for old in ids[:-10]:                 # keep only the last 10
    await storage.delete(old)
```

## Custom executor state

Executors can persist arbitrary state across checkpoints by implementing two hooks. The framework calls them on every save/restore:

```python
from typing import Any
from agent_framework import Executor, WorkflowContext, handler


class CounterExecutor(Executor):
    def __init__(self) -> None:
        super().__init__(id="counter")
        self._count = 0

    @handler
    async def tick(self, _: str, ctx: WorkflowContext[str, str]) -> None:
        self._count += 1
        await ctx.send_message(f"count={self._count}")

    async def on_checkpoint_save(self) -> dict[str, Any]:
        return {"count": self._count}

    async def on_checkpoint_restore(self, state: dict[str, Any]) -> None:
        self._count = state.get("count", 0)
```

### Persisting a dataclass

Dataclasses work cleanly because the built-in encoder picks up anything pickle-safe — register the type in `allowed_checkpoint_types` and it round-trips:

```python
from dataclasses import dataclass, field, asdict
from typing import Any
from agent_framework import Executor, WorkflowContext, FileCheckpointStorage, handler


@dataclass
class OrderState:
    seen_ids: set[str] = field(default_factory=set)
    total_cents: int = 0


class OrderCollector(Executor):
    def __init__(self) -> None:
        super().__init__(id="order-collector")
        self.state = OrderState()

    @handler
    async def on_order(self, order: dict[str, Any], ctx: WorkflowContext) -> None:
        if order["id"] in self.state.seen_ids:
            return                                # already processed
        self.state.seen_ids.add(order["id"])
        self.state.total_cents += int(order["total_cents"])

    async def on_checkpoint_save(self) -> dict[str, Any]:
        # Convert the dataclass to something the JSON/pickle encoder can handle.
        return {"state": asdict(self.state) | {"seen_ids": list(self.state.seen_ids)}}

    async def on_checkpoint_restore(self, state: dict[str, Any]) -> None:
        payload = state.get("state", {})
        self.state = OrderState(
            seen_ids=set(payload.get("seen_ids", [])),
            total_cents=payload.get("total_cents", 0),
        )


storage = FileCheckpointStorage(
    "/var/lib/agent-framework/checkpoints",
    allowed_checkpoint_types=["__main__:OrderState"],
)
```

## Inspecting the checkpoint chain

`list_checkpoints` returns the full `WorkflowCheckpoint` objects — use it to walk `previous_checkpoint_id` and show progress or build an audit trail:

```python
from agent_framework import FileCheckpointStorage

storage = FileCheckpointStorage("/var/lib/agent-framework/checkpoints")
checkpoints = await storage.list_checkpoints(workflow_name="research-pipeline")

# Build an id -> checkpoint map and walk from latest backwards.
by_id = {c.checkpoint_id: c for c in checkpoints}
latest = max(checkpoints, key=lambda c: c.timestamp) if checkpoints else None
chain = []
cursor = latest
while cursor is not None:
    chain.append(cursor)
    cursor = by_id.get(cursor.previous_checkpoint_id) if cursor.previous_checkpoint_id else None

for cp in reversed(chain):
    pending = len(cp.pending_request_info_events)
    print(f"{cp.timestamp} iter={cp.iteration_count} pending_hitl={pending}")
```

`pending_request_info_events` is populated whenever the workflow paused on a HITL request — the count tells you if the workflow is waiting on a human or still running.

## Multiple workflows, one storage directory

Storage is scoped to the **directory** but filtered by `workflow_name`. Run several workflows against the same `FileCheckpointStorage` and `get_latest(workflow_name=...)` picks only the relevant chain:

```python
storage = FileCheckpointStorage("/var/lib/agents/checkpoints")

research_wf = WorkflowBuilder(
    start_executor=researcher, checkpoint_storage=storage, name="research-pipeline"
).add_edge(researcher, writer).build()

support_wf = WorkflowBuilder(
    start_executor=triage, checkpoint_storage=storage, name="support-routing"
).add_edge(triage, specialist).build()

# Distinct namespaces — no collision between the two workflows.
latest_research = await storage.get_latest(workflow_name="research-pipeline")
latest_support = await storage.get_latest(workflow_name="support-routing")
```

## Pickle safety

`FileCheckpointStorage` serialises state as JSON with base64-encoded pickle for complex objects. By default it restores only a safe built-in set plus `agent_framework.*` types and `openai.types`. To allow your own types, pass fully-qualified names:

```python
storage = FileCheckpointStorage(
    "/var/lib/agents/checkpoints",
    allowed_checkpoint_types=[
        "my_app.models:OrderState",
        "my_app.events:CustomEvent",
    ],
)
```

Only accept types you fully control. Treat the checkpoint directory as a trusted boundary.

## Atomic writes & concurrency

`FileCheckpointStorage` writes `checkpoint_id.json.tmp` then renames atomically, so a crash mid-write never leaves partial state on disk. `InMemoryCheckpointStorage` deep-copies on save and is safe inside a single process but not across processes.

For multi-process workflows (e.g. Azure Functions scaled out) use Cosmos or Redis storage — `FileCheckpointStorage` is not designed for concurrent writers from different machines.

## Patterns

**Long-running Magentic research.** Build with `MagenticBuilder(..., checkpoint_storage=storage, enable_plan_review=True)`. The plan-review HITL event is persisted; the user can come back hours later and approve, and the workflow resumes in a different pod.

**Crash-recovery for tool-heavy agents.** Every tool loop iteration is a superstep. If the pod dies mid-tool-call, start a new pod, call `storage.get_latest(workflow_name=...)`, pass the ID to `workflow.run(...)` — the tool call re-issues, no duplicate billing of previous tools.

**Handoff with audit trail.** `previous_checkpoint_id` chains checkpoints. Walk the chain to reconstruct the full decision history for compliance.

**Deploy-through upgrades.** Pin a `workflow_name` per definition version (`research-pipeline-v2`). Old checkpoints fail the `graph_signature_hash` check with a clear error — upgrades never silently run against incompatible topologies.

## Observability

Each checkpoint save/load emits structured logs on `agent_framework._workflows._checkpoint`. When OpenTelemetry is on (see [Observability](./microsoft_agent_framework_python_observability/)), checkpoint operations appear as events inside the `workflow.run` span with `workflow.id` and `workflow.name` attributes.
