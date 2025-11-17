# Pydantic AI: Durable Execution Guide
## Preserve Progress Across Failures & Restarts

**Version:** 1.0.0 (2025)
**Framework:** Pydantic AI with Durable Execution
**Focus:** Fault-tolerant agent systems with checkpoint/resume capabilities

---

## Table of Contents

1. [Overview](#overview)
2. [Core Concepts](#core-concepts)
3. [Prefect Integration](#prefect-integration)
4. [DBOS Integration](#dbos-integration)
5. [Temporal Integration](#temporal-integration)
6. [Custom Durable Backends](#custom-durable-backends)
7. [State Persistence Patterns](#state-persistence-patterns)
8. [Fault Tolerance](#fault-tolerance)
9. [Production Examples](#production-examples)

---

## Overview

### What is Durable Execution?

Durable execution ensures your AI agents can:
- **Resume from interruptions** - API failures, application crashes, or deployments
- **Preserve progress** - Don't lose expensive LLM calls or partial results
- **Recover automatically** - Continue exactly where you left off
- **Guarantee completion** - Execute long-running workflows reliably

### Why Durable Execution?

```python
# ❌ WITHOUT Durable Execution
# API fails at step 3 of 10 → Start over, waste tokens, lose progress

# ✅ WITH Durable Execution
# API fails at step 3 of 10 → Resume from step 3, preserve all prior work
```

**Use Cases:**
- Long-running multi-agent workflows
- Expensive research or analysis tasks
- Critical business processes requiring guarantees
- Production systems with high reliability requirements

---

## Core Concepts

### Durable Execution Primitives

```python
from pydantic_ai import Agent
from pydantic_ai.durable import DurableAgent, Checkpoint

# 1. Checkpoints - Save state at key points
checkpoint = Checkpoint(
    agent_state={'step': 3, 'results': [...]},
    message_history=[...],
    tool_results={...}
)

# 2. Resume Tokens - Resume from any checkpoint
resume_token = checkpoint.create_resume_token()

# 3. Durable Agent - Automatic checkpointing
agent = DurableAgent(
    'openai:gpt-4o',
    checkpoint_every_n_steps=1,  # Checkpoint after each step
    backend='prefect'  # or 'dbos', 'temporal', 'custom'
)
```

### Execution Guarantees

| Feature | Standard Agent | Durable Agent |
|---------|---------------|---------------|
| **Resume after crash** | ❌ No | ✅ Yes |
| **Preserve token usage** | ❌ Restart = new tokens | ✅ Only new work uses tokens |
| **Long-running workflows** | ⚠️ Risk of failure | ✅ Guaranteed completion |
| **Exactly-once execution** | ❌ May retry entire flow | ✅ Each step executes once |
| **State persistence** | ❌ Memory only | ✅ Durable storage |

---

## Prefect Integration

### Setup

```bash
# Install Pydantic AI with Prefect
pip install "pydantic-ai[prefect]"

# Configure Prefect
prefect cloud login
# or for self-hosted
prefect server start
```

### Basic Durable Agent with Prefect

```python
from pydantic_ai import Agent
from pydantic_ai.durable.prefect import PrefectDurableAgent
from pydantic import BaseModel
from prefect import flow, task
import asyncio

class ResearchResult(BaseModel):
    topic: str
    findings: list[str]
    summary: str
    sources: list[str]

@flow(name="durable-research-agent", retries=3)
async def research_workflow(topic: str) -> ResearchResult:
    """Durable research workflow that survives failures."""

    # Create durable agent with automatic checkpointing
    agent = PrefectDurableAgent(
        'openai:gpt-4o',
        output_type=ResearchResult,
        instructions='Conduct thorough research on the topic',
        checkpoint_every_n_steps=1,  # Checkpoint after each tool call
        checkpoint_on_error=True,    # Checkpoint before raising errors
    )

    # Add research tools
    @agent.tool
    async def search_web(ctx, query: str) -> list[str]:
        """Search the web for information."""
        # Simulate web search
        await asyncio.sleep(2)
        return [f"Result for {query}"]

    @agent.tool
    async def analyze_sources(ctx, sources: list[str]) -> dict:
        """Analyze and summarize sources."""
        await asyncio.sleep(3)
        return {'analysis': 'Detailed analysis'}

    # Run with automatic durability
    result = await agent.run(
        f"Research the topic: {topic}",
        flow_run_name=f"research-{topic}"  # Track in Prefect UI
    )

    return result.output

# Usage
if __name__ == '__main__':
    result = asyncio.run(research_workflow("Quantum Computing"))
    print(f"Research completed: {result.topic}")
    print(f"Findings: {len(result.findings)}")
```

### Advanced Prefect Patterns

```python
from pydantic_ai.durable.prefect import PrefectDurableAgent, FlowContext
from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact
from datetime import timedelta

@flow(
    name="multi-stage-agent-pipeline",
    retries=3,
    retry_delay_seconds=60,
    timeout_seconds=3600
)
async def multi_stage_pipeline(input_data: dict) -> dict:
    """Multi-stage durable pipeline with checkpoints."""

    logger = get_run_logger()

    # Stage 1: Data Collection
    collector_agent = PrefectDurableAgent(
        'openai:gpt-4o',
        checkpoint_every_n_steps=1,
        name='DataCollector'
    )

    logger.info("Starting data collection stage")
    collection_result = await collector_agent.run(
        f"Collect data about: {input_data['topic']}",
        flow_run_name="stage-1-collection"
    )

    # Create artifact in Prefect UI
    await create_markdown_artifact(
        key="collection-results",
        markdown=f"# Collection Results\n{collection_result.output}"
    )

    # Stage 2: Analysis
    analyst_agent = PrefectDurableAgent(
        'anthropic:claude-3-5-sonnet-latest',
        checkpoint_every_n_steps=1,
        name='DataAnalyst'
    )

    logger.info("Starting analysis stage")
    analysis_result = await analyst_agent.run(
        f"Analyze this data: {collection_result.output}",
        flow_run_name="stage-2-analysis"
    )

    # Stage 3: Report Generation
    writer_agent = PrefectDurableAgent(
        'openai:gpt-4o',
        checkpoint_every_n_steps=2,
        name='ReportWriter'
    )

    logger.info("Starting report generation")
    report_result = await writer_agent.run(
        f"Generate report from analysis: {analysis_result.output}",
        flow_run_name="stage-3-report"
    )

    return {
        'collection': collection_result.output,
        'analysis': analysis_result.output,
        'report': report_result.output
    }

# Manual checkpoint and resume
@flow(name="manual-checkpoint-example")
async def manual_checkpoint_workflow():
    """Demonstrate manual checkpoint control."""

    agent = PrefectDurableAgent(
        'openai:gpt-4o',
        checkpoint_every_n_steps=0,  # Disable automatic checkpointing
    )

    # Manual checkpoint creation
    checkpoint_id = await agent.create_checkpoint({
        'step': 1,
        'data': 'Initial state'
    })

    try:
        result = await agent.run("Complex query")

        # Checkpoint successful result
        await agent.create_checkpoint({
            'step': 2,
            'result': result.output
        })

    except Exception as e:
        # Resume from last checkpoint
        checkpoint = await agent.get_checkpoint(checkpoint_id)
        print(f"Resuming from: {checkpoint.state}")

        # Retry from checkpoint
        result = await agent.run_from_checkpoint(checkpoint_id)

    return result
```

### Monitoring Durable Flows

```python
from prefect import flow, get_run_logger
from prefect.client import get_client
from pydantic_ai.durable.prefect import PrefectDurableAgent

@flow(name="monitored-durable-agent")
async def monitored_workflow(query: str):
    """Workflow with comprehensive monitoring."""

    logger = get_run_logger()

    agent = PrefectDurableAgent(
        'openai:gpt-4o',
        checkpoint_every_n_steps=1,
        on_checkpoint=lambda cp: logger.info(f"Checkpoint created: {cp.id}"),
        on_resume=lambda cp: logger.info(f"Resuming from: {cp.id}")
    )

    # Get flow run context
    async with get_client() as client:
        flow_run = await client.read_flow_run(get_run_logger().context['flow_run_id'])
        logger.info(f"Flow run: {flow_run.name}")

    result = await agent.run(query)

    # Log metrics
    logger.info(f"Tokens used: {result.usage().total_tokens}")
    logger.info(f"Checkpoints created: {agent.checkpoint_count}")

    return result.output
```

---

## DBOS Integration

### Setup

```bash
# Install Pydantic AI with DBOS
pip install "pydantic-ai[dbos]"

# Initialize DBOS project
dbos init my-agent-project
cd my-agent-project
```

### DBOS Durable Agent

```python
from pydantic_ai import Agent
from pydantic_ai.durable.dbos import DBOSDurableAgent, workflow, transaction
from dbos import DBOS
import asyncio

# Initialize DBOS
dbos = DBOS()

@workflow()
async def durable_agent_workflow(topic: str) -> dict:
    """
    DBOS workflow with automatic durability.

    DBOS guarantees:
    - Exactly-once execution of each step
    - Automatic recovery from any failure
    - Complete audit trail
    - Time-travel debugging
    """

    # Create DBOS durable agent
    agent = DBOSDurableAgent(
        'openai:gpt-4o',
        instructions='You are a research assistant',
        workflow_id=DBOS.workflow_id  # Link to DBOS workflow
    )

    # Step 1: Research (durable)
    research_result = await agent.run(
        f"Research {topic}",
        step_name="research"  # Named step for checkpointing
    )

    # Step 2: Save to database (transactional)
    await save_research(research_result.output)

    # Step 3: Analyze (durable)
    analysis_result = await agent.run(
        f"Analyze: {research_result.output}",
        step_name="analysis"
    )

    return {
        'research': research_result.output,
        'analysis': analysis_result.output
    }

@transaction()
async def save_research(data: str):
    """Transactional database operation."""
    # DBOS ensures this executes exactly once
    async with DBOS.pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO research (data, timestamp) VALUES ($1, NOW())",
            data
        )

# Advanced: Multi-agent DBOS workflow
@workflow()
async def multi_agent_dbos_workflow(task: str) -> dict:
    """Coordinate multiple durable agents."""

    # Agent 1: Planner
    planner = DBOSDurableAgent(
        'openai:gpt-4o',
        name='Planner',
        workflow_id=DBOS.workflow_id
    )

    plan = await planner.run(
        f"Create a plan for: {task}",
        step_name="planning"
    )

    # Agent 2: Executor
    executor = DBOSDurableAgent(
        'anthropic:claude-3-5-sonnet-latest',
        name='Executor',
        workflow_id=DBOS.workflow_id
    )

    execution = await executor.run(
        f"Execute this plan: {plan.output}",
        step_name="execution"
    )

    # Agent 3: Reviewer
    reviewer = DBOSDurableAgent(
        'openai:gpt-4o',
        name='Reviewer',
        workflow_id=DBOS.workflow_id
    )

    review = await reviewer.run(
        f"Review execution: {execution.output}",
        step_name="review"
    )

    return {
        'plan': plan.output,
        'execution': execution.output,
        'review': review.output
    }

# Time-travel debugging
async def debug_workflow(workflow_id: str):
    """Debug workflow by replaying from any checkpoint."""

    # Get workflow history
    history = await DBOS.get_workflow_history(workflow_id)

    print(f"Workflow steps: {len(history.steps)}")

    for step in history.steps:
        print(f"Step: {step.name}, Status: {step.status}")
        print(f"  Input: {step.input}")
        print(f"  Output: {step.output}")
        print(f"  Tokens: {step.metadata.get('tokens_used')}")

    # Replay from specific step
    if history.status == 'failed':
        print(f"Replaying from step: {history.failed_step}")
        await DBOS.replay_workflow(workflow_id, from_step=history.failed_step)
```

### DBOS State Management

```python
from pydantic_ai.durable.dbos import DBOSDurableAgent, workflow, state
from pydantic import BaseModel
from typing import Any

class WorkflowState(BaseModel):
    """Type-safe workflow state."""
    current_step: int
    results: dict[str, Any]
    tokens_used: int
    errors: list[str]

@workflow()
async def stateful_workflow(query: str) -> dict:
    """Workflow with managed state."""

    # Initialize state
    wf_state = WorkflowState(
        current_step=0,
        results={},
        tokens_used=0,
        errors=[]
    )

    agent = DBOSDurableAgent('openai:gpt-4o')

    try:
        # Step 1
        wf_state.current_step = 1
        result1 = await agent.run("Step 1", step_name="step_1")
        wf_state.results['step_1'] = result1.output
        wf_state.tokens_used += result1.usage().total_tokens

        # Persist state
        await DBOS.set_state('workflow_state', wf_state.model_dump())

        # Step 2
        wf_state.current_step = 2
        result2 = await agent.run("Step 2", step_name="step_2")
        wf_state.results['step_2'] = result2.output
        wf_state.tokens_used += result2.usage().total_tokens

        await DBOS.set_state('workflow_state', wf_state.model_dump())

    except Exception as e:
        wf_state.errors.append(str(e))
        await DBOS.set_state('workflow_state', wf_state.model_dump())

        # Workflow will automatically retry from last checkpoint
        raise

    return wf_state.results

# Resume workflow from failure
async def resume_failed_workflow(workflow_id: str):
    """Resume a failed workflow."""

    # DBOS automatically resumes from last successful checkpoint
    result = await DBOS.resume_workflow(workflow_id)

    # Get final state
    final_state = await DBOS.get_state(workflow_id, 'workflow_state')

    return WorkflowState(**final_state)
```

---

## Temporal Integration

### Setup

```bash
# Install Pydantic AI with Temporal
pip install "pydantic-ai[temporal]"

# Start Temporal server (Docker)
docker run -d -p 7233:7233 temporalio/auto-setup:latest
```

### Temporal Durable Agent

```python
from pydantic_ai import Agent
from pydantic_ai.durable.temporal import TemporalDurableAgent
from temporalio import workflow, activity
from temporalio.client import Client
from temporalio.worker import Worker
from datetime import timedelta
import asyncio

@workflow.defn
class ResearchWorkflow:
    """Temporal workflow with durable agents."""

    @workflow.run
    async def run(self, topic: str) -> dict:
        """Main workflow execution."""

        # Create durable agent
        agent = TemporalDurableAgent(
            'openai:gpt-4o',
            workflow_context=workflow.info(),  # Link to Temporal workflow
            instructions='Conduct thorough research'
        )

        # Step 1: Initial research (automatically durable)
        research = await agent.run(
            f"Research: {topic}",
            activity_id="research_step"
        )

        # Step 2: Wait for human approval (Temporal signal)
        await workflow.wait_condition(lambda: self.approved)

        # Step 3: Continue after approval
        analysis = await agent.run(
            f"Analyze: {research.output}",
            activity_id="analysis_step"
        )

        return {
            'research': research.output,
            'analysis': analysis.output
        }

    @workflow.signal
    def approve(self):
        """Signal to approve and continue."""
        self.approved = True

# Long-running workflow with timeouts
@workflow.defn
class LongRunningAgentWorkflow:
    """Handle long-running agent tasks."""

    @workflow.run
    async def run(self, complex_task: str) -> str:
        """Execute long-running task with checkpoints."""

        agent = TemporalDurableAgent(
            'openai:gpt-4o',
            checkpoint_interval=timedelta(minutes=5)  # Checkpoint every 5 min
        )

        # Execute with activity heartbeating
        result = await agent.run(
            complex_task,
            activity_id="long_task",
            heartbeat_interval=timedelta(seconds=30),  # Heartbeat every 30s
            timeout=timedelta(hours=2)  # Max 2 hour timeout
        )

        return result.output

# Run workflow
async def run_temporal_workflow():
    """Execute Temporal workflow."""

    # Connect to Temporal
    client = await Client.connect("localhost:7233")

    # Start workflow
    handle = await client.start_workflow(
        ResearchWorkflow.run,
        "Quantum Computing",
        id="research-workflow-1",
        task_queue="agent-tasks"
    )

    print(f"Started workflow: {handle.id}")

    # Signal approval after 10 seconds
    await asyncio.sleep(10)
    await handle.signal(ResearchWorkflow.approve)

    # Wait for result
    result = await handle.result()
    print(f"Workflow completed: {result}")

# Worker to process workflows
async def run_worker():
    """Start Temporal worker."""

    client = await Client.connect("localhost:7233")

    worker = Worker(
        client,
        task_queue="agent-tasks",
        workflows=[ResearchWorkflow, LongRunningAgentWorkflow]
    )

    await worker.run()
```

---

## Custom Durable Backends

### Implement Custom Backend

```python
from pydantic_ai.durable import DurableBackend, Checkpoint
from typing import Optional, Any
import json
import asyncio
from datetime import datetime

class PostgreSQLDurableBackend(DurableBackend):
    """Custom durable backend using PostgreSQL."""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool = None

    async def initialize(self):
        """Initialize database connection pool."""
        import asyncpg

        self.pool = await asyncpg.create_pool(self.connection_string)

        # Create checkpoints table
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_checkpoints (
                    id SERIAL PRIMARY KEY,
                    workflow_id VARCHAR(255) NOT NULL,
                    checkpoint_id VARCHAR(255) UNIQUE NOT NULL,
                    step_number INTEGER NOT NULL,
                    agent_state JSONB NOT NULL,
                    message_history JSONB NOT NULL,
                    tool_results JSONB,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT NOW(),
                    INDEX idx_workflow (workflow_id),
                    INDEX idx_checkpoint (checkpoint_id)
                )
            """)

    async def save_checkpoint(
        self,
        workflow_id: str,
        checkpoint: Checkpoint
    ) -> str:
        """Save checkpoint to database."""

        async with self.pool.acquire() as conn:
            checkpoint_id = f"{workflow_id}-{checkpoint.step_number}"

            await conn.execute("""
                INSERT INTO agent_checkpoints (
                    workflow_id, checkpoint_id, step_number,
                    agent_state, message_history, tool_results, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (checkpoint_id) DO UPDATE SET
                    agent_state = EXCLUDED.agent_state,
                    message_history = EXCLUDED.message_history,
                    tool_results = EXCLUDED.tool_results,
                    metadata = EXCLUDED.metadata
            """,
                workflow_id,
                checkpoint_id,
                checkpoint.step_number,
                json.dumps(checkpoint.agent_state),
                json.dumps(checkpoint.message_history),
                json.dumps(checkpoint.tool_results or {}),
                json.dumps(checkpoint.metadata or {})
            )

            return checkpoint_id

    async def load_checkpoint(
        self,
        checkpoint_id: str
    ) -> Optional[Checkpoint]:
        """Load checkpoint from database."""

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT step_number, agent_state, message_history,
                       tool_results, metadata
                FROM agent_checkpoints
                WHERE checkpoint_id = $1
            """, checkpoint_id)

            if not row:
                return None

            return Checkpoint(
                step_number=row['step_number'],
                agent_state=json.loads(row['agent_state']),
                message_history=json.loads(row['message_history']),
                tool_results=json.loads(row['tool_results']),
                metadata=json.loads(row['metadata'])
            )

    async def list_checkpoints(
        self,
        workflow_id: str
    ) -> list[str]:
        """List all checkpoints for a workflow."""

        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT checkpoint_id
                FROM agent_checkpoints
                WHERE workflow_id = $1
                ORDER BY step_number ASC
            """, workflow_id)

            return [row['checkpoint_id'] for row in rows]

    async def delete_checkpoints(
        self,
        workflow_id: str,
        before_step: Optional[int] = None
    ):
        """Delete old checkpoints."""

        async with self.pool.acquire() as conn:
            if before_step:
                await conn.execute("""
                    DELETE FROM agent_checkpoints
                    WHERE workflow_id = $1 AND step_number < $2
                """, workflow_id, before_step)
            else:
                await conn.execute("""
                    DELETE FROM agent_checkpoints
                    WHERE workflow_id = $1
                """, workflow_id)

# Use custom backend
async def use_custom_backend():
    """Use custom PostgreSQL backend."""

    from pydantic_ai import Agent
    from pydantic_ai.durable import DurableAgent

    # Initialize backend
    backend = PostgreSQLDurableBackend(
        "postgresql://user:pass@localhost/agents"
    )
    await backend.initialize()

    # Create durable agent with custom backend
    agent = DurableAgent(
        'openai:gpt-4o',
        backend=backend,
        workflow_id='custom-workflow-123',
        checkpoint_every_n_steps=1
    )

    # Run workflow
    result = await agent.run("Complex multi-step query")

    # List checkpoints
    checkpoints = await backend.list_checkpoints('custom-workflow-123')
    print(f"Created {len(checkpoints)} checkpoints")

    # Resume from specific checkpoint
    if checkpoints:
        checkpoint = await backend.load_checkpoint(checkpoints[-1])
        resumed_agent = DurableAgent.from_checkpoint(
            checkpoint,
            backend=backend
        )

        continued_result = await resumed_agent.run("Continue from checkpoint")
```

---

## State Persistence Patterns

### Stateful Multi-Step Workflows

```python
from pydantic_ai import Agent
from pydantic_ai.durable import DurableAgent, StatefulWorkflow
from pydantic import BaseModel
from typing import Any, Optional
from enum import Enum

class WorkflowPhase(str, Enum):
    PLANNING = 'planning'
    EXECUTION = 'execution'
    VALIDATION = 'validation'
    COMPLETION = 'completion'

class WorkflowContext(BaseModel):
    """Persistent workflow context."""
    phase: WorkflowPhase
    plan: Optional[str] = None
    execution_results: list[str] = []
    validation_errors: list[str] = []
    attempt_count: int = 0
    total_tokens_used: int = 0

class StatefulAgentWorkflow(StatefulWorkflow):
    """Multi-phase stateful workflow."""

    def __init__(self, backend):
        super().__init__(backend)
        self.context = WorkflowContext(phase=WorkflowPhase.PLANNING)

    async def run(self, task: str) -> dict:
        """Execute stateful workflow with automatic persistence."""

        # Phase 1: Planning
        self.context.phase = WorkflowPhase.PLANNING
        await self.save_state()

        planner = DurableAgent(
            'openai:gpt-4o',
            backend=self.backend,
            workflow_id=self.workflow_id
        )

        plan_result = await planner.run(f"Create a plan for: {task}")
        self.context.plan = plan_result.output
        self.context.total_tokens_used += plan_result.usage().total_tokens
        await self.save_state()

        # Phase 2: Execution
        self.context.phase = WorkflowPhase.EXECUTION
        await self.save_state()

        executor = DurableAgent(
            'anthropic:claude-3-5-sonnet-latest',
            backend=self.backend,
            workflow_id=self.workflow_id
        )

        exec_result = await executor.run(f"Execute: {self.context.plan}")
        self.context.execution_results.append(exec_result.output)
        self.context.total_tokens_used += exec_result.usage().total_tokens
        await self.save_state()

        # Phase 3: Validation
        self.context.phase = WorkflowPhase.VALIDATION
        await self.save_state()

        validator = DurableAgent(
            'openai:gpt-4o',
            backend=self.backend,
            workflow_id=self.workflow_id
        )

        validation_result = await validator.run(
            f"Validate: {self.context.execution_results[-1]}"
        )

        if "error" in validation_result.output.lower():
            self.context.validation_errors.append(validation_result.output)
            self.context.attempt_count += 1

            if self.context.attempt_count < 3:
                # Retry execution phase
                self.context.phase = WorkflowPhase.EXECUTION
                await self.save_state()
                return await self.run(task)  # Recursive retry

        # Phase 4: Completion
        self.context.phase = WorkflowPhase.COMPLETION
        await self.save_state()

        return {
            'plan': self.context.plan,
            'results': self.context.execution_results,
            'tokens_used': self.context.total_tokens_used,
            'attempts': self.context.attempt_count
        }

    async def save_state(self):
        """Persist current context."""
        await self.backend.save_checkpoint(
            self.workflow_id,
            Checkpoint(
                step_number=len(self.context.execution_results),
                agent_state=self.context.model_dump(),
                message_history=[],
                metadata={'phase': self.context.phase.value}
            )
        )

    @classmethod
    async def resume_from_checkpoint(
        cls,
        workflow_id: str,
        backend: DurableBackend
    ) -> 'StatefulAgentWorkflow':
        """Resume workflow from saved state."""

        checkpoints = await backend.list_checkpoints(workflow_id)

        if not checkpoints:
            raise ValueError(f"No checkpoints found for {workflow_id}")

        # Load latest checkpoint
        latest = await backend.load_checkpoint(checkpoints[-1])

        workflow = cls(backend)
        workflow.workflow_id = workflow_id
        workflow.context = WorkflowContext(**latest.agent_state)

        return workflow
```

---

## Fault Tolerance

### Automatic Recovery Patterns

```python
from pydantic_ai.durable import DurableAgent, RetryPolicy
from typing import Optional
import asyncio

class ResilientDurableAgent:
    """Durable agent with comprehensive fault tolerance."""

    def __init__(
        self,
        model: str,
        backend: DurableBackend,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.agent = DurableAgent(
            model,
            backend=backend,
            checkpoint_every_n_steps=1,
            checkpoint_on_error=True
        )

        self.retry_policy = RetryPolicy(
            max_retries=max_retries,
            retry_delay=retry_delay,
            exponential_backoff=True,
            retry_on_errors=['APIError', 'TimeoutError', 'RateLimitError']
        )

    async def run_with_recovery(
        self,
        query: str,
        workflow_id: str
    ) -> Optional[str]:
        """Run agent with automatic recovery."""

        attempt = 0
        last_error = None

        while attempt < self.retry_policy.max_retries:
            try:
                # Try to run agent
                result = await self.agent.run(query, workflow_id=workflow_id)
                return result.output

            except Exception as e:
                last_error = e
                error_type = type(e).__name__

                # Check if error is retryable
                if error_type not in self.retry_policy.retry_on_errors:
                    raise

                attempt += 1

                # Calculate backoff delay
                if self.retry_policy.exponential_backoff:
                    delay = self.retry_policy.retry_delay * (2 ** attempt)
                else:
                    delay = self.retry_policy.retry_delay

                print(f"Attempt {attempt} failed: {e}. Retrying in {delay}s...")
                await asyncio.sleep(delay)

                # Try to resume from last checkpoint
                checkpoints = await self.agent.backend.list_checkpoints(workflow_id)

                if checkpoints:
                    print(f"Resuming from checkpoint: {checkpoints[-1]}")
                    checkpoint = await self.agent.backend.load_checkpoint(
                        checkpoints[-1]
                    )

                    # Resume execution
                    self.agent = DurableAgent.from_checkpoint(
                        checkpoint,
                        backend=self.agent.backend
                    )

        # All retries exhausted
        raise last_error

# Circuit breaker pattern
from enum import Enum
from datetime import datetime, timedelta

class CircuitState(str, Enum):
    CLOSED = 'closed'  # Normal operation
    OPEN = 'open'      # Failing, reject requests
    HALF_OPEN = 'half_open'  # Testing recovery

class CircuitBreaker:
    """Circuit breaker for durable agents."""

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: timedelta = timedelta(minutes=1)
    ):
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.last_failure_time = None

    async def call(self, agent_func):
        """Execute agent function through circuit breaker."""

        if self.state == CircuitState.OPEN:
            # Check if timeout has elapsed
            if datetime.now() - self.last_failure_time > self.timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")

        try:
            result = await agent_func()

            # Success - reset if in half-open state
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failure_count = 0

            return result

        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN

            raise

# Usage
async def fault_tolerant_workflow():
    """Complete fault-tolerant workflow."""

    backend = PostgreSQLDurableBackend("postgresql://localhost/agents")
    await backend.initialize()

    # Create resilient agent
    resilient_agent = ResilientDurableAgent(
        'openai:gpt-4o',
        backend=backend,
        max_retries=3
    )

    # Wrap in circuit breaker
    circuit_breaker = CircuitBreaker(failure_threshold=5)

    try:
        result = await circuit_breaker.call(
            lambda: resilient_agent.run_with_recovery(
                "Complex query that might fail",
                workflow_id="fault-tolerant-workflow-1"
            )
        )

        print(f"Success: {result}")

    except Exception as e:
        print(f"All recovery attempts failed: {e}")
```

---

## Production Examples

### Complete Production Workflow

```python
from pydantic_ai.durable.prefect import PrefectDurableAgent
from prefect import flow, task, get_run_logger
from prefect.blocks.system import Secret
from pydantic import BaseModel
from typing import List, Optional
import asyncio

class DocumentAnalysis(BaseModel):
    """Analysis result schema."""
    summary: str
    key_points: List[str]
    sentiment: str
    action_items: List[str]
    confidence: float

@flow(
    name="production-document-analysis",
    retries=3,
    retry_delay_seconds=[60, 300, 900],  # Exponential backoff
    timeout_seconds=3600,
    persist_result=True
)
async def production_analysis_workflow(
    document_url: str,
    user_id: str
) -> DocumentAnalysis:
    """
    Production-grade document analysis with full durability.

    Features:
    - Automatic checkpointing
    - Retry with exponential backoff
    - State persistence
    - Comprehensive error handling
    - Observability integration
    """

    logger = get_run_logger()
    logger.info(f"Starting analysis for user {user_id}")

    # Fetch API key from Prefect secret
    api_key_block = await Secret.load("openai-api-key")
    api_key = api_key_block.get()

    # Create durable agent with production configuration
    agent = PrefectDurableAgent(
        'openai:gpt-4o',
        output_type=DocumentAnalysis,
        instructions="""
        Analyze documents thoroughly and provide:
        1. Concise summary
        2. 3-5 key points
        3. Overall sentiment
        4. Actionable items
        5. Confidence score
        """,
        checkpoint_every_n_steps=1,
        checkpoint_on_error=True,
        name=f"DocumentAnalyzer-{user_id}"
    )

    # Add document fetching tool
    @agent.tool
    async def fetch_document(ctx, url: str) -> str:
        """Fetch document content from URL."""
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30)
            response.raise_for_status()
            return response.text[:10000]  # Limit to 10k chars

    # Execute analysis with full durability
    try:
        result = await agent.run(
            f"Analyze the document at: {document_url}",
            flow_run_name=f"analysis-{user_id}-{document_url}",
            metadata={
                'user_id': user_id,
                'document_url': document_url
            }
        )

        logger.info(f"Analysis completed: {result.usage().total_tokens} tokens used")

        # Validate output
        analysis = result.output

        if analysis.confidence < 0.7:
            logger.warning(f"Low confidence analysis: {analysis.confidence}")

        return analysis

    except Exception as e:
        logger.error(f"Analysis failed: {e}")

        # Log failure for monitoring
        from prefect.events import emit_event
        emit_event(
            event="document-analysis.failed",
            resource={"prefect.flow-run.id": str(get_run_logger().context['flow_run_id'])},
            payload={'error': str(e), 'user_id': user_id}
        )

        raise

# Deploy to Prefect Cloud
if __name__ == '__main__':
    # Deploy workflow
    from prefect.deployments import Deployment
    from prefect.server.schemas.schedules import CronSchedule

    deployment = Deployment.build_from_flow(
        flow=production_analysis_workflow,
        name="production-document-analysis",
        work_queue_name="agent-queue",
        parameters={
            "document_url": "https://example.com/doc.pdf",
            "user_id": "default"
        },
        description="Durable document analysis workflow",
        tags=["production", "agents", "durable"]
    )

    deployment.apply()
```

---

## Best Practices

### 1. Checkpoint Frequency

```python
# ✅ GOOD: Checkpoint after expensive operations
agent = DurableAgent(
    'openai:gpt-4o',
    checkpoint_every_n_steps=1,  # After each LLM call
)

# ❌ BAD: Too infrequent (lose progress)
agent = DurableAgent(
    'openai:gpt-4o',
    checkpoint_every_n_steps=100,  # Might lose 99 steps
)
```

### 2. State Size Management

```python
# ✅ GOOD: Compact state
class CompactState(BaseModel):
    step: int
    result_ids: list[str]  # Store IDs, not full data

# ❌ BAD: Large state
class BloatedState(BaseModel):
    full_results: list[dict]  # Megabytes of data
```

### 3. Error Categorization

```python
# ✅ GOOD: Distinguish retryable vs permanent errors
if isinstance(error, RateLimitError):
    # Retryable - checkpoint and retry
    await checkpoint()
    await retry_with_backoff()
elif isinstance(error, ValidationError):
    # Permanent - fail fast
    raise
```

---

## Summary

**Durable Execution enables:**
- ✅ Fault-tolerant AI agent workflows
- ✅ Resume from any point after failures
- ✅ Cost savings (avoid re-running expensive LLM calls)
- ✅ Production-grade reliability
- ✅ Long-running workflows (hours/days)
- ✅ Exactly-once execution guarantees

**Choose your backend:**
- **Prefect**: Best for Python-native workflows, great UI
- **DBOS**: Best for database-centric apps, time-travel debugging
- **Temporal**: Best for polyglot systems, mature ecosystem
- **Custom**: Best for specific infrastructure requirements

---

**Next Steps:**
1. Choose a durable backend (Prefect, DBOS, or Temporal)
2. Add checkpointing to critical workflows
3. Implement retry and recovery logic
4. Monitor checkpoint creation and resume events
5. Test failure scenarios thoroughly

**See Also:**
- `pydantic_ai_graph_support.md` - Graph-based workflows
- `pydantic_ai_production_guide.md` - Production deployment
- `pydantic_ai_comprehensive_guide.md` - Core concepts
