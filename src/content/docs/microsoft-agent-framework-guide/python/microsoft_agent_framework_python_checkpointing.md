---
title: "Microsoft Agent Framework (Python) — Checkpointing"
description: "Persist workflow state with FileCheckpointStorage, InMemoryCheckpointStorage, or a custom CheckpointStorage backend. Resume long-running agents across process restarts."
framework: microsoft-agent-framework
language: python
---

# Checkpointing — Python

Checkpointing captures the full execution state of a workflow at every superstep so you can pause, crash, redeploy, or hand off to a different process and pick up exactly where you left off. It's the backbone for long-running agent workflows, human-in-the-loop pauses that last days, and durable multi-agent orchestrations.

Verified against `agent-framework-core==1.5.0` (`agent_framework._workflows._checkpoint`).

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

### `WorkflowCheckpoint` field reference

`WorkflowCheckpoint` is a `slots=True` dataclass. Every field is constructible by keyword and round-trips through `to_dict()` / `from_dict()`:

| Field | Type | Default | Notes |
|---|---|---|---|
| `workflow_name` | `str` | required | Logical group key. Workflows sharing this name are expected to share topology. |
| `graph_signature_hash` | `str` | required | Hash of the topology — checked on restore for compatibility. |
| `checkpoint_id` | `str` | UUID4 | Auto-generated unless you override. |
| `previous_checkpoint_id` | `str \| None` | `None` | Forms a linked list across iterations of the same run. |
| `timestamp` | `str` | `datetime.now(UTC).isoformat()` | ISO 8601, used for ordering by `get_latest`. |
| `messages` | `dict[str, list[WorkflowMessage]]` | `{}` | In-flight inter-executor messages. |
| `state` | `dict[str, Any]` | `{}` | Committed user/executor state. Reserved keys live under `_executor_state`. |
| `pending_request_info_events` | `dict[str, WorkflowEvent]` | `{}` | Outstanding HITL requests not yet resolved. |
| `iteration_count` | `int` | `0` | Superstep number when the checkpoint was taken. |
| `metadata` | `dict[str, Any]` | `{}` | Free-form metadata (graph signature, environment, deploy id…). |
| `version` | `str` | `"1.0"` | Checkpoint format version. Bumped if the schema changes. |

`from_dict` raises `WorkflowCheckpointException` if required fields are missing or unknown fields appear — a useful boundary when you're loading checkpoints from an external system that might be on a stale schema. The exception message names the offending field so the cause is obvious in logs.

```python
import logging
from agent_framework import WorkflowCheckpoint, WorkflowCheckpointException

log = logging.getLogger(__name__)

# Required fields are `workflow_name` and `graph_signature_hash`.
# Omitting either — or passing an unrecognised field — raises.
stale = {"workflow_name": "research"}  # missing `graph_signature_hash`
try:
    cp = WorkflowCheckpoint.from_dict(stale)
except WorkflowCheckpointException as exc:
    log.error("malformed checkpoint payload: %s", exc)
    # exc message: "Failed to create WorkflowCheckpoint from dict: WorkflowCheckpoint.__init__()
    #               missing 1 required positional argument: 'graph_signature_hash'"
```

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

### Hands-on: `InMemoryCheckpointStorage` round-trip

The in-memory backend is the same code path the workflow runtime exercises in tests. Use it in unit tests, in notebook scratchpads, or when you want zero-config persistence for the duration of a single process. It deep-copies on `save`, so the saved checkpoint is immune to subsequent mutation of the originating object.

```python
import asyncio
from agent_framework import InMemoryCheckpointStorage, WorkflowCheckpoint


async def main() -> None:
    storage = InMemoryCheckpointStorage()

    cp = WorkflowCheckpoint(
        workflow_name="research",
        graph_signature_hash="abc123",
        state={"step_1": "done", "topic": "agent frameworks"},
        iteration_count=3,
        metadata={"environment": "staging"},
    )
    cp_id = await storage.save(cp)
    # cp_id is the same as cp.checkpoint_id (a generated UUID4 unless you set it)

    loaded = await storage.load(cp_id)
    assert loaded.state == {"step_1": "done", "topic": "agent frameworks"}
    assert loaded.iteration_count == 3

    # Filter by workflow_name — supports multi-tenant single-process tests.
    saved = await storage.list_checkpoints(workflow_name="research")
    assert len(saved) == 1

    # `get_latest` orders by `timestamp`, falling back to insertion order.
    latest = await storage.get_latest(workflow_name="research")
    assert latest is not None and latest.checkpoint_id == cp_id

    # Delete returns True on hit, False on miss — never raises.
    assert await storage.delete(cp_id) is True
    assert await storage.delete(cp_id) is False
    assert await storage.list_checkpoint_ids(workflow_name="research") == []


asyncio.run(main())
```

Three things to remember about the in-memory backend:

- **Process-scoped.** State is lost on restart. Don't reach for it in production — even a quick autoscaler reshuffle will lose all in-flight work.
- **Deep-copies on save and load.** Mutating the dataclass after `save` doesn't change what you'll get back from `load`. Unit tests that assert "checkpoint isolation" can rely on that contract.
- **`load` of a missing id raises `WorkflowCheckpointException`** — exactly the same behaviour as `FileCheckpointStorage`. Catch that exception (or its parent `WorkflowRunnerException`) in tests rather than reaching for `None` checks.

### Allow-listing app-specific types

`FileCheckpointStorage` deserializes pickled state behind a strict allow-list. Out of the box it accepts Python primitives, `datetime` / `uuid`, every `agent_framework` type, and `openai.types`. **Anything else raises `WorkflowCheckpointException` on load.** This is deliberate — pickle lets attackers run arbitrary code if a malicious checkpoint sneaks into the storage path.

When your workflow stores domain objects (Pydantic models, dataclasses, enums), declare them via `allowed_checkpoint_types`:

```python
from dataclasses import dataclass
from agent_framework import FileCheckpointStorage, WorkflowBuilder


@dataclass
class ResearchState:
    topic: str
    confidence: float


# Each entry is "module:qualname" — same shape pickle uses internally.
# The check is an exact match (no subclass walk), so list every concrete
# class you actually persist; allowing `enum:Enum` does NOT permit subclasses.
storage = FileCheckpointStorage(
    "/var/lib/agents/checkpoints",
    allowed_checkpoint_types=[
        "my_app.models:ResearchState",
        "my_app.models:ResearchOutcome",
        "my_app.models:ResearchStatus",      # list each enum subclass you store
    ],
)

workflow = (
    WorkflowBuilder(
        start_executor=researcher,
        checkpoint_storage=storage,
        name="research-pipeline",
    )
    .add_edge(researcher, writer)
    .build()
)
```

Two operational notes:

- The list is **frozen at construction time** — passing it later won't take effect. Build storage once at startup with the full app type set.
- The error message names the class that failed the allow-list, so missing entries are easy to fix during a staging run before production.

For multi-tenant deployments where each tenant has its own type universe, build a tenant-keyed dict of storages instead of granting one global allow-list everything.

### Custom backend

`CheckpointStorage` is a `Protocol` — structural typing means anything with the six required `async` methods satisfies it. No `isinstance` or inheritance check happens at attach time, only the duck-typed call. The full surface area:

```python
class CheckpointStorage(Protocol):
    async def save(self, checkpoint: WorkflowCheckpoint) -> str: ...
    async def load(self, checkpoint_id: str) -> WorkflowCheckpoint: ...
    async def list_checkpoints(self, *, workflow_name: str) -> list[WorkflowCheckpoint]: ...
    async def delete(self, checkpoint_id: str) -> bool: ...
    async def get_latest(self, *, workflow_name: str) -> WorkflowCheckpoint | None: ...
    async def list_checkpoint_ids(self, *, workflow_name: str) -> list[str]: ...
```

A complete S3-backed implementation. Two design choices that keep the hot path cheap as the bucket grows:

- **Flat object layout** (`{prefix}{checkpoint_id}.json`) so `load` and `delete` are single O(1) `get_object` / `delete_object` calls.
- **Workflow-name routing via S3 user metadata** (`x-amz-meta-workflow-name`) plus a `{prefix}_index/{workflow_name}/{checkpoint_id}` zero-byte index marker. `list_*` and `get_latest` hit only the index, never the full bodies.

```python
import json
from dataclasses import asdict
import aioboto3
from agent_framework import WorkflowCheckpoint, WorkflowCheckpointException


class S3CheckpointStorage:
    def __init__(self, bucket: str, prefix: str = "checkpoints/") -> None:
        self._bucket = bucket
        self._prefix = prefix.rstrip("/") + "/"
        self._session = aioboto3.Session()

    # Flat keys keep load/delete O(1); the index handles workflow_name filtering.
    def _data_key(self, checkpoint_id: str) -> str:
        return f"{self._prefix}{checkpoint_id}.json"

    def _index_key(self, workflow_name: str, checkpoint_id: str) -> str:
        return f"{self._prefix}_index/{workflow_name}/{checkpoint_id}"

    async def save(self, checkpoint: WorkflowCheckpoint) -> str:
        body = json.dumps(asdict(checkpoint)).encode()
        async with self._session.client("s3") as s3:
            await s3.put_object(
                Bucket=self._bucket,
                Key=self._data_key(checkpoint.checkpoint_id),
                Body=body,
                ContentType="application/json",
                Metadata={
                    "workflow-name": checkpoint.workflow_name,
                    "timestamp": checkpoint.timestamp,
                },
            )
            # Zero-byte index marker — used by list_* and get_latest.
            await s3.put_object(
                Bucket=self._bucket,
                Key=self._index_key(checkpoint.workflow_name, checkpoint.checkpoint_id),
                Body=b"",
            )
        return checkpoint.checkpoint_id

    async def load(self, checkpoint_id: str) -> WorkflowCheckpoint:
        async with self._session.client("s3") as s3:
            try:
                obj = await s3.get_object(Bucket=self._bucket, Key=self._data_key(checkpoint_id))
            except s3.exceptions.NoSuchKey:
                raise WorkflowCheckpointException(f"No checkpoint found with ID {checkpoint_id}")
            data = json.loads(await obj["Body"].read())
        return WorkflowCheckpoint.from_dict(data)

    async def list_checkpoint_ids(self, *, workflow_name: str) -> list[str]:
        prefix = f"{self._prefix}_index/{workflow_name}/"
        ids: list[str] = []
        async with self._session.client("s3") as s3:
            paginator = s3.get_paginator("list_objects_v2")
            async for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    ids.append(obj["Key"].rsplit("/", 1)[-1])
        return ids

    async def list_checkpoints(self, *, workflow_name: str) -> list[WorkflowCheckpoint]:
        ids = await self.list_checkpoint_ids(workflow_name=workflow_name)
        # Caller asked for full bodies — fan out the gets in parallel.
        return [await self.load(cid) for cid in ids]

    async def get_latest(self, *, workflow_name: str) -> WorkflowCheckpoint | None:
        prefix = f"{self._prefix}_index/{workflow_name}/"
        latest_marker = None
        async with self._session.client("s3") as s3:
            paginator = s3.get_paginator("list_objects_v2")
            async for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
                for obj in page.get("Contents", []):
                    # LastModified comes back from list_objects_v2 — no extra request.
                    if latest_marker is None or obj["LastModified"] > latest_marker["LastModified"]:
                        latest_marker = obj
        if latest_marker is None:
            return None
        checkpoint_id = latest_marker["Key"].rsplit("/", 1)[-1]
        return await self.load(checkpoint_id)        # one targeted get_object

    async def delete(self, checkpoint_id: str) -> bool:
        async with self._session.client("s3") as s3:
            try:
                head = await s3.head_object(Bucket=self._bucket, Key=self._data_key(checkpoint_id))
            except s3.exceptions.ClientError:
                return False
            workflow_name = head.get("Metadata", {}).get("workflow-name")
            await s3.delete_object(Bucket=self._bucket, Key=self._data_key(checkpoint_id))
            if workflow_name:
                await s3.delete_object(
                    Bucket=self._bucket,
                    Key=self._index_key(workflow_name, checkpoint_id),
                )
        return True
```

Why each shortcut matters as the bucket grows:

- `load` and `delete` issue **one** S3 request each (plus a tiny `head_object` for delete to find the index pointer). No scanning.
- `get_latest` lists only the index keys — small, zero-byte objects — and uses the `LastModified` field returned by `list_objects_v2` to pick the winner before fetching a single body.
- `list_checkpoint_ids` walks index keys alone, never downloading bodies. Use it whenever you only need ids (audit reports, prune jobs).

Three things to mirror from `FileCheckpointStorage` when rolling your own backend:

- **Atomic writes.** The built-in writes `<id>.json.tmp` then `os.replace` for crash safety. S3 `put_object` is atomic; for filesystem-derived backends (NFS, a custom on-disk format), keep the write-then-rename pattern. The two `put_object` calls in `save` are not transactionally atomic — if the index write fails the data object is still queryable by `load`. Surface the failure or run a periodic reconciler that re-creates missing index markers.
- **Path / id validation.** `FileCheckpointStorage._validate_file_path` rejects ids that resolve outside the storage root (path traversal). For S3 the equivalent is asserting the key starts with your prefix; for any backend, never blindly concatenate user-influenced ids into a path.
- **Raise `WorkflowCheckpointException` on miss.** The framework treats `load` failures as a recoverable "no such checkpoint" and surfaces the message — don't let the underlying client error bubble up unwrapped.

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

### Walking a checkpoint's pending requests before responding

`WorkflowCheckpoint.pending_request_info_events` carries every HITL request the workflow is blocked on. Inspect it to build the `responses=` map automatically — useful when you have many concurrent approvals waiting:

```python
from agent_framework import FileCheckpointStorage

storage = FileCheckpointStorage("/var/lib/agents/checkpoints")
latest = await storage.get_latest(workflow_name="research-pipeline")

if latest and latest.pending_request_info_events:
    # Auto-approve everything the user has already signed off on in an upstream system.
    responses = {
        event.request_id: approval_store.lookup_decision(event.request_id)
        for event in latest.pending_request_info_events
    }

    result = await workflow.run(
        checkpoint_id=latest.checkpoint_id,
        responses=responses,
        checkpoint_storage=storage,    # re-attach storage so further steps keep saving
    )
```

The workflow resumes, dispatches each response to the executor that raised the request, and continues stepping until it idles or hits the next HITL gate. Any request IDs you omit from `responses` stay pending and the workflow saves a fresh checkpoint with them still outstanding — safe to call repeatedly.

### Delete a partial run

`WorkflowCheckpoint.previous_checkpoint_id` forms a chain; `delete(...)` removes one file. To delete an entire aborted run, walk the chain and delete each step:

```python
from agent_framework import CheckpointStorage


async def delete_run(storage: CheckpointStorage, tip_checkpoint_id: str) -> int:
    """Works with any CheckpointStorage backend — File, InMemory, Redis, Cosmos."""
    count = 0
    cursor = await storage.load(tip_checkpoint_id)
    while True:
        await storage.delete(cursor.checkpoint_id)
        count += 1
        if not cursor.previous_checkpoint_id:
            break
        cursor = await storage.load(cursor.previous_checkpoint_id)
    return count
```

Prefer `list_checkpoint_ids(workflow_name=...)` + `get_latest` for the common case ("keep last N"). Only walk the chain when you need to surgically remove one branch of checkpoints without affecting sibling runs.

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
