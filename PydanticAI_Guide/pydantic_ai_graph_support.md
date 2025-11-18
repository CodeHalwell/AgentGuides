# Pydantic AI: Graph Support Guide
## Type-Hint Based Graph Definitions for Complex Applications

**Version:** 1.0.0 (2025)
**Framework:** Pydantic AI with Graph Support
**Focus:** Declarative graph-based agent workflows with full type safety

---

## Table of Contents

1. [Overview](#overview)
2. [Core Graph Concepts](#core-graph-concepts)
3. [Type-Hint Graph Definitions](#type-hint-graph-definitions)
4. [State Machines](#state-machines)
5. [Conditional Branching](#conditional-branching)
6. [Graph Execution](#graph-execution)
7. [Multi-Agent Graphs](#multi-agent-graphs)
8. [Production Patterns](#production-patterns)

---

## Overview

### What is Graph Support?

Graph support in Pydantic AI allows you to define complex agent workflows as directed acyclic graphs (DAGs) or state machines using Python type hints and decorators.

**Key Features:**
- **Type-Safe Graphs**: Define nodes and edges with full type safety
- **Declarative Syntax**: Express workflows naturally in Python
- **Conditional Routing**: Dynamic flow control based on outputs
- **State Management**: Built-in state passing between nodes
- **Visual Debugging**: Generate graph visualizations
- **Parallel Execution**: Automatic parallelization of independent nodes

### Why Use Graphs?

```python
# ❌ WITHOUT Graphs: Imperative, hard to visualize
result1 = await agent1.run(input)
if result1.output.needs_review:
    result2 = await agent2.run(result1.output)
else:
    result2 = await agent3.run(result1.output)
final = await agent4.run(result2.output)

# ✅ WITH Graphs: Declarative, self-documenting
@graph
class AnalysisWorkflow:
    analyze -> review [if needs_review]
    analyze -> summarize [else]
    review -> finalize
    summarize -> finalize
```

**Use Cases:**
- Multi-stage agent pipelines
- Conditional workflows
- Parallel agent execution
- State machines
- Complex decision trees

---

## Core Graph Concepts

### Graph Terminology

```python
from pydantic_ai.graph import Graph, Node, Edge, State
from pydantic import BaseModel

# Node: A single execution unit (agent or function)
@Node
async def analyze_text(state: State) -> str:
    """Node that processes text."""
    return "analyzed text"

# Edge: Connection between nodes with optional conditions
edge = Edge(
    from_node="analyze",
    to_node="summarize",
    condition=lambda state: state.needs_summary
)

# State: Data passed between nodes
class WorkflowState(BaseModel):
    input_text: str
    analyzed: str = ""
    summary: str = ""
    final_output: str = ""

# Graph: Container for nodes and edges
graph = Graph(
    name="TextProcessing",
    nodes=[analyze_text, summarize_text, finalize],
    edges=[edge1, edge2, edge3],
    initial_state=WorkflowState
)
```

### Basic Graph Example

```python
from pydantic_ai import Agent
from pydantic_ai.graph import Graph, graph, node
from pydantic import BaseModel
from typing import Annotated

# Define state
class DocumentState(BaseModel):
    content: str
    summary: str = ""
    key_points: list[str] = []
    sentiment: str = ""

# Define graph with decorator
@graph(state_type=DocumentState)
class DocumentProcessing:
    """Document processing graph."""

    @node(name="summarize")
    async def create_summary(self, state: DocumentState) -> DocumentState:
        """Summarize document content."""
        agent = Agent('openai:gpt-4o')
        result = await agent.run(f"Summarize: {state.content}")

        state.summary = result.output
        return state

    @node(name="extract_points")
    async def extract_key_points(self, state: DocumentState) -> DocumentState:
        """Extract key points (runs in parallel with summarize)."""
        agent = Agent('openai:gpt-4o')
        result = await agent.run(f"Extract key points: {state.content}")

        state.key_points = result.output.split('\n')
        return state

    @node(name="analyze_sentiment")
    async def analyze_sentiment(self, state: DocumentState) -> DocumentState:
        """Analyze sentiment (runs after summary and key points)."""
        agent = Agent('openai:gpt-4o')
        result = await agent.run(
            f"Analyze sentiment of: {state.summary} | {state.key_points}"
        )

        state.sentiment = result.output
        return state

# Usage
async def run_graph():
    # Create graph instance
    graph = DocumentProcessing()

    # Define execution flow
    graph.add_edge("summarize", "analyze_sentiment")
    graph.add_edge("extract_points", "analyze_sentiment")

    # Set entry points (nodes with no predecessors)
    graph.set_entry_nodes(["summarize", "extract_points"])

    # Execute
    initial_state = DocumentState(content="Long document text...")
    final_state = await graph.execute(initial_state)

    print(f"Summary: {final_state.summary}")
    print(f"Sentiment: {final_state.sentiment}")
    print(f"Key points: {final_state.key_points}")
```

---

## Type-Hint Graph Definitions

### Declarative Graph Syntax

```python
from pydantic_ai.graph import Graph, declarative_graph, NodeResult
from pydantic import BaseModel
from typing import Annotated, Literal

class ResearchState(BaseModel):
    """Type-safe state for research workflow."""
    topic: str
    research_done: bool = False
    analysis_complete: bool = False
    report_ready: bool = False
    final_report: str = ""

@declarative_graph
class ResearchPipeline:
    """
    Declarative graph using type hints.

    Flow:
    1. research -> analyze -> write_report -> review
    2. If review fails -> write_report (retry)
    3. If review passes -> publish
    """

    # Define nodes with type hints
    def research(
        self,
        state: ResearchState
    ) -> Annotated[ResearchState, "research_done"]:
        """Research node - marks research as complete."""
        agent = Agent('openai:gpt-4o')
        result = agent.run_sync(f"Research: {state.topic}")

        state.research_done = True
        return state

    def analyze(
        self,
        state: Annotated[ResearchState, "research_done"]  # Depends on research
    ) -> Annotated[ResearchState, "analysis_complete"]:
        """Analysis node - requires research to be done."""
        agent = Agent('anthropic:claude-3-5-sonnet-latest')
        result = agent.run_sync("Analyze research findings")

        state.analysis_complete = True
        return state

    def write_report(
        self,
        state: Annotated[ResearchState, "analysis_complete"]
    ) -> Annotated[ResearchState, "report_ready"]:
        """Write report - requires analysis to be complete."""
        agent = Agent('openai:gpt-4o')
        result = agent.run_sync("Write detailed report")

        state.final_report = result.output
        state.report_ready = True
        return state

    def review(
        self,
        state: Annotated[ResearchState, "report_ready"]
    ) -> Annotated[NodeResult[ResearchState], Literal["approve", "reject"]]:
        """Review report - conditional routing based on quality."""
        agent = Agent('openai:gpt-4o')
        result = agent.run_sync(f"Review report: {state.final_report}")

        if "approve" in result.output.lower():
            return NodeResult(state, route="approve")
        else:
            state.report_ready = False  # Reset for retry
            return NodeResult(state, route="reject")

    def publish(
        self,
        state: Annotated[ResearchState, "review.approve"]  # Only if approved
    ) -> ResearchState:
        """Publish report - final node."""
        print(f"Publishing report: {state.final_report[:100]}...")
        return state

# Automatic graph construction from type hints
graph = ResearchPipeline.build_graph()

# Visualize graph
graph.visualize(output_path="research_pipeline.png")

# Execute
initial = ResearchState(topic="AI Safety")
final = await graph.execute(initial)
```

### Advanced Type-Hint Patterns

```python
from pydantic_ai.graph import graph, node, depends_on, conditional_edge
from typing import Annotated, Literal, Union, get_args
from pydantic import BaseModel, Field

class AnalysisState(BaseModel):
    """State with rich type information."""
    input_data: str
    complexity: Literal["simple", "medium", "complex"] = "simple"
    requires_human_review: bool = False
    confidence_score: float = Field(0.0, ge=0.0, le=1.0)
    result: str = ""

@graph(state_type=AnalysisState)
class SmartAnalysisGraph:
    """Graph with advanced type-based routing."""

    @node
    async def assess_complexity(
        self,
        state: AnalysisState
    ) -> Annotated[AnalysisState, "complexity_assessed"]:
        """Assess input complexity."""
        # Simple heuristic
        if len(state.input_data) < 100:
            state.complexity = "simple"
        elif len(state.input_data) < 1000:
            state.complexity = "medium"
        else:
            state.complexity = "complex"

        return state

    @node
    @depends_on("assess_complexity")
    async def simple_analysis(
        self,
        state: Annotated[AnalysisState, "complexity == 'simple'"]
    ) -> AnalysisState:
        """Fast analysis for simple inputs."""
        agent = Agent('openai:gpt-4o-mini')  # Cheaper model
        result = await agent.run(state.input_data)

        state.result = result.output
        state.confidence_score = 0.9
        return state

    @node
    @depends_on("assess_complexity")
    async def complex_analysis(
        self,
        state: Annotated[AnalysisState, "complexity in ['medium', 'complex']"]
    ) -> AnalysisState:
        """Deep analysis for complex inputs."""
        agent = Agent('openai:gpt-4o')  # Better model
        result = await agent.run(state.input_data)

        state.result = result.output
        state.confidence_score = 0.95
        return state

    @node
    @depends_on("simple_analysis", "complex_analysis")
    async def quality_check(
        self,
        state: AnalysisState
    ) -> Annotated[AnalysisState, Union[Literal["pass"], Literal["review"]]]:
        """Check if human review needed."""
        if state.confidence_score < 0.8 or state.complexity == "complex":
            state.requires_human_review = True
            return state, "review"
        else:
            return state, "pass"

    @node
    @conditional_edge(from_node="quality_check", condition="review")
    async def human_review(
        self,
        state: AnalysisState
    ) -> AnalysisState:
        """Request human review."""
        print(f"Requesting human review for: {state.input_data[:50]}...")
        # In production: send to review queue
        return state

    @node
    @conditional_edge(from_node="quality_check", condition="pass")
    async def finalize(
        self,
        state: AnalysisState
    ) -> AnalysisState:
        """Finalize result."""
        print(f"Analysis complete: {state.result[:100]}")
        return state
```

---

## State Machines

### Finite State Machine Pattern

```python
from pydantic_ai.graph import StateMachine, state_transition
from pydantic import BaseModel
from enum import Enum
from typing import Literal

class OrderStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    APPROVED = "approved"
    REJECTED = "rejected"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class OrderState(BaseModel):
    order_id: str
    status: OrderStatus = OrderStatus.PENDING
    items: list[str] = []
    total: float = 0.0
    approval_notes: str = ""

@StateMachine(initial_state=OrderStatus.PENDING, state_type=OrderState)
class OrderProcessingStateMachine:
    """State machine for order processing."""

    @state_transition(
        from_state=OrderStatus.PENDING,
        to_state=OrderStatus.PROCESSING
    )
    async def start_processing(self, state: OrderState) -> OrderState:
        """Transition: PENDING -> PROCESSING."""
        print(f"Starting to process order {state.order_id}")

        # Agent validates order
        agent = Agent('openai:gpt-4o')
        result = await agent.run(f"Validate order items: {state.items}")

        state.status = OrderStatus.PROCESSING
        return state

    @state_transition(
        from_state=OrderStatus.PROCESSING,
        to_state=[OrderStatus.APPROVED, OrderStatus.REJECTED]  # Multiple destinations
    )
    async def approve_or_reject(
        self,
        state: OrderState
    ) -> tuple[OrderState, OrderStatus]:
        """Transition: PROCESSING -> APPROVED or REJECTED."""

        agent = Agent('openai:gpt-4o')
        result = await agent.run(
            f"Review order: {state.order_id}, Total: ${state.total}"
        )

        if "approve" in result.output.lower():
            state.status = OrderStatus.APPROVED
            state.approval_notes = result.output
            return state, OrderStatus.APPROVED
        else:
            state.status = OrderStatus.REJECTED
            state.approval_notes = result.output
            return state, OrderStatus.REJECTED

    @state_transition(
        from_state=OrderStatus.APPROVED,
        to_state=OrderStatus.COMPLETED
    )
    async def complete_order(self, state: OrderState) -> OrderState:
        """Transition: APPROVED -> COMPLETED."""
        print(f"Completing order {state.order_id}")

        # Process payment, shipping, etc.
        state.status = OrderStatus.COMPLETED
        return state

    @state_transition(
        from_state=[OrderStatus.PENDING, OrderStatus.PROCESSING],
        to_state=OrderStatus.CANCELLED
    )
    async def cancel_order(self, state: OrderState) -> OrderState:
        """Transition: (PENDING or PROCESSING) -> CANCELLED."""
        print(f"Cancelling order {state.order_id}")

        state.status = OrderStatus.CANCELLED
        return state

# Usage
async def process_order():
    """Process order through state machine."""

    # Create state machine
    sm = OrderProcessingStateMachine()

    # Initialize state
    order = OrderState(
        order_id="ORD-123",
        items=["laptop", "mouse"],
        total=1500.00
    )

    # Execute state transitions
    order = await sm.transition("start_processing", order)
    assert order.status == OrderStatus.PROCESSING

    order, next_state = await sm.transition("approve_or_reject", order)

    if next_state == OrderStatus.APPROVED:
        order = await sm.transition("complete_order", order)
        assert order.status == OrderStatus.COMPLETED
    else:
        print(f"Order rejected: {order.approval_notes}")

    # Get state machine diagram
    sm.visualize_state_machine(output_path="order_fsm.png")
```

### Complex State Machine with Guards

```python
from pydantic_ai.graph import StateMachine, state_transition, guard
from pydantic import BaseModel
from typing import Optional

class DocumentState(BaseModel):
    content: str
    status: str = "draft"
    version: int = 1
    approver: Optional[str] = None
    published_url: Optional[str] = None

@StateMachine(initial_state="draft", state_type=DocumentState)
class DocumentLifecycle:
    """Document lifecycle state machine with guards."""

    @guard
    def can_submit_for_review(self, state: DocumentState) -> bool:
        """Guard: Check if document can be submitted."""
        return len(state.content) > 100  # Minimum length

    @guard
    def can_approve(self, state: DocumentState) -> bool:
        """Guard: Check if document can be approved."""
        return state.approver is not None

    @state_transition(
        from_state="draft",
        to_state="in_review",
        guard=can_submit_for_review
    )
    async def submit_for_review(self, state: DocumentState) -> DocumentState:
        """Submit document for review (guarded)."""
        if not self.can_submit_for_review(state):
            raise ValueError("Document too short for review")

        state.status = "in_review"
        return state

    @state_transition(
        from_state="in_review",
        to_state="approved",
        guard=can_approve
    )
    async def approve_document(self, state: DocumentState) -> DocumentState:
        """Approve document (guarded)."""
        state.status = "approved"
        return state

    @state_transition(
        from_state="in_review",
        to_state="draft"
    )
    async def reject_document(self, state: DocumentState) -> DocumentState:
        """Reject and return to draft."""
        state.status = "draft"
        state.version += 1
        return state

    @state_transition(
        from_state="approved",
        to_state="published"
    )
    async def publish_document(self, state: DocumentState) -> DocumentState:
        """Publish approved document."""
        # Generate publication URL
        state.published_url = f"https://docs.example.com/{state.version}"
        state.status = "published"
        return state
```

---

## Conditional Branching

### Dynamic Routing

```python
from pydantic_ai.graph import Graph, node, conditional_router
from pydantic import BaseModel
from typing import Literal

class ContentState(BaseModel):
    text: str
    content_type: Literal["question", "command", "statement"] = "statement"
    requires_context: bool = False
    response: str = ""

@Graph(state_type=ContentState)
class ContentProcessingGraph:
    """Graph with dynamic routing based on content type."""

    @node(name="classify")
    async def classify_content(self, state: ContentState) -> ContentState:
        """Classify input content type."""
        agent = Agent('openai:gpt-4o')
        result = await agent.run(
            f"Classify as question, command, or statement: {state.text}"
        )

        classification = result.output.lower()

        if "question" in classification:
            state.content_type = "question"
        elif "command" in classification:
            state.content_type = "command"
        else:
            state.content_type = "statement"

        return state

    @conditional_router(after="classify")
    def route_by_type(
        self,
        state: ContentState
    ) -> Literal["handle_question", "handle_command", "handle_statement"]:
        """Route based on content type."""
        routing = {
            "question": "handle_question",
            "command": "handle_command",
            "statement": "handle_statement"
        }
        return routing[state.content_type]

    @node(name="handle_question")
    async def handle_question(self, state: ContentState) -> ContentState:
        """Handle question-type content."""
        # Check if context needed
        agent = Agent('openai:gpt-4o')

        if "how" in state.text.lower() or "why" in state.text.lower():
            state.requires_context = True
            # Fetch context...

        result = await agent.run(f"Answer question: {state.text}")
        state.response = result.output
        return state

    @node(name="handle_command")
    async def handle_command(self, state: ContentState) -> ContentState:
        """Handle command-type content."""
        agent = Agent('openai:gpt-4o')
        result = await agent.run(f"Execute command: {state.text}")
        state.response = result.output
        return state

    @node(name="handle_statement")
    async def handle_statement(self, state: ContentState) -> ContentState:
        """Handle statement-type content."""
        agent = Agent('openai:gpt-4o')
        result = await agent.run(f"Respond to statement: {state.text}")
        state.response = result.output
        return state

    @node(name="finalize")
    async def finalize_response(self, state: ContentState) -> ContentState:
        """Final processing (all routes converge here)."""
        print(f"Final response: {state.response}")
        return state

# Build graph with automatic routing
graph = ContentProcessingGraph.build()

# Add convergence - all handlers lead to finalize
graph.add_edge("handle_question", "finalize")
graph.add_edge("handle_command", "finalize")
graph.add_edge("handle_statement", "finalize")
```

### Multi-Condition Branching

```python
from pydantic_ai.graph import Graph, node, multi_conditional_router
from pydantic import BaseModel
from typing import Literal, List

class AnalysisState(BaseModel):
    data: str
    complexity: Literal["low", "medium", "high"] = "low"
    urgency: Literal["low", "high"] = "low"
    result: str = ""

@Graph(state_type=AnalysisState)
class PriorityAnalysisGraph:
    """Graph with multi-condition routing."""

    @node
    async def assess_task(self, state: AnalysisState) -> AnalysisState:
        """Assess complexity and urgency."""
        # Assess complexity
        if len(state.data) > 1000:
            state.complexity = "high"
        elif len(state.data) > 100:
            state.complexity = "medium"

        # Assess urgency (in production: from metadata)
        state.urgency = "high" if "urgent" in state.data.lower() else "low"

        return state

    @multi_conditional_router(after="assess_task")
    def route_by_priority(
        self,
        state: AnalysisState
    ) -> List[str]:
        """
        Route based on multiple conditions.
        Returns list of nodes to execute (can be multiple).
        """
        routes = []

        # Route based on complexity
        if state.complexity == "high":
            routes.append("deep_analysis")
        else:
            routes.append("quick_analysis")

        # Add urgency handling
        if state.urgency == "high":
            routes.append("priority_queue")

        return routes

    @node(name="quick_analysis")
    async def quick_analysis(self, state: AnalysisState) -> AnalysisState:
        """Quick analysis for simple tasks."""
        agent = Agent('openai:gpt-4o-mini')
        result = await agent.run(state.data)
        state.result = result.output
        return state

    @node(name="deep_analysis")
    async def deep_analysis(self, state: AnalysisState) -> AnalysisState:
        """Deep analysis for complex tasks."""
        agent = Agent('openai:gpt-4o')
        result = await agent.run(state.data)
        state.result = result.output
        return state

    @node(name="priority_queue")
    async def handle_urgent(self, state: AnalysisState) -> AnalysisState:
        """Handle urgent tasks (runs in parallel)."""
        print(f"URGENT: Prioritizing task")
        # Send notification, escalate, etc.
        return state
```

---

## Graph Execution

### Parallel Execution

```python
from pydantic_ai.graph import Graph, node, ParallelExecution
from pydantic import BaseModel
import asyncio

class ParallelState(BaseModel):
    input: str
    translation: str = ""
    summary: str = ""
    sentiment: str = ""
    keywords: list[str] = []

@Graph(state_type=ParallelState, execution_mode=ParallelExecution.MAX)
class ParallelProcessingGraph:
    """Graph with maximum parallelization."""

    @node(parallel_group="preprocessing")
    async def translate(self, state: ParallelState) -> ParallelState:
        """Translate text (parallel group 1)."""
        agent = Agent('openai:gpt-4o')
        result = await agent.run(f"Translate to English: {state.input}")
        state.translation = result.output
        return state

    @node(parallel_group="preprocessing")
    async def extract_keywords(self, state: ParallelState) -> ParallelState:
        """Extract keywords (parallel group 1)."""
        agent = Agent('openai:gpt-4o')
        result = await agent.run(f"Extract keywords: {state.input}")
        state.keywords = result.output.split(',')
        return state

    @node(parallel_group="analysis", depends_on=["translate", "extract_keywords"])
    async def summarize(self, state: ParallelState) -> ParallelState:
        """Summarize (parallel group 2, after preprocessing)."""
        agent = Agent('openai:gpt-4o')
        result = await agent.run(f"Summarize: {state.translation}")
        state.summary = result.output
        return state

    @node(parallel_group="analysis", depends_on=["translate"])
    async def analyze_sentiment(self, state: ParallelState) -> ParallelState:
        """Analyze sentiment (parallel group 2, after preprocessing)."""
        agent = Agent('openai:gpt-4o')
        result = await agent.run(f"Sentiment: {state.translation}")
        state.sentiment = result.output
        return state

# Execute with automatic parallelization
graph = ParallelProcessingGraph.build()

# Execution plan:
# Stage 1 (parallel): translate, extract_keywords
# Stage 2 (parallel): summarize, analyze_sentiment (wait for translate)
execution_plan = graph.get_execution_plan()
print(execution_plan)
# Output:
# [
#   ["translate", "extract_keywords"],  # Execute in parallel
#   ["summarize", "analyze_sentiment"]  # Execute in parallel after stage 1
# ]

# Run
state = ParallelState(input="Foreign language text...")
final_state = await graph.execute(state)
```

### Streaming Graph Execution

```python
from pydantic_ai.graph import Graph, node, StreamingGraph
from pydantic import BaseModel
from typing import AsyncIterator

@Graph(state_type=BaseModel, streaming=True)
class StreamingAnalysisGraph:
    """Graph that streams intermediate results."""

    @node
    async def stage_1(self, state: dict) -> AsyncIterator[dict]:
        """Streaming node - yields intermediate results."""
        agent = Agent('openai:gpt-4o')

        async with agent.run_stream(state['input']) as response:
            partial_result = ""

            async for chunk in response.stream_text():
                partial_result += chunk

                # Yield intermediate state
                yield {
                    **state,
                    'stage_1_partial': partial_result
                }

            # Final yield
            result = await response.result()
            yield {
                **state,
                'stage_1_complete': result.output
            }

    @node
    async def stage_2(self, state: dict) -> AsyncIterator[dict]:
        """Second streaming stage."""
        agent = Agent('openai:gpt-4o')

        async with agent.run_stream(state['stage_1_complete']) as response:
            async for chunk in response.stream_text():
                yield {
                    **state,
                    'stage_2_partial': chunk
                }

# Execute with streaming
graph = StreamingAnalysisGraph.build()

async for intermediate_state in graph.execute_stream({'input': 'Analyze this'}):
    # Receive state updates in real-time
    if 'stage_1_partial' in intermediate_state:
        print(f"Stage 1: {intermediate_state['stage_1_partial']}")
    elif 'stage_2_partial' in intermediate_state:
        print(f"Stage 2: {intermediate_state['stage_2_partial']}")
```

---

## Multi-Agent Graphs

### Specialized Agent Graph

```python
from pydantic_ai.graph import Graph, node
from pydantic_ai import Agent
from pydantic import BaseModel

class MultiAgentState(BaseModel):
    task: str
    research: str = ""
    code: str = ""
    tests: str = ""
    documentation: str = ""

@Graph(state_type=MultiAgentState)
class SoftwareDevelopmentGraph:
    """Graph with specialized agents for each task."""

    def __init__(self):
        # Initialize specialized agents
        self.researcher = Agent(
            'openai:gpt-4o',
            instructions='You are a research specialist. Provide thorough research.'
        )

        self.coder = Agent(
            'anthropic:claude-3-5-sonnet-latest',
            instructions='You are a coding expert. Write clean, tested code.'
        )

        self.tester = Agent(
            'openai:gpt-4o',
            instructions='You are a QA engineer. Write comprehensive tests.'
        )

        self.documenter = Agent(
            'openai:gpt-4o',
            instructions='You are a technical writer. Create clear documentation.'
        )

    @node(name="research")
    async def research_task(self, state: MultiAgentState) -> MultiAgentState:
        """Research phase."""
        result = await self.researcher.run(f"Research: {state.task}")
        state.research = result.output
        return state

    @node(name="implement", depends_on=["research"])
    async def implement_solution(self, state: MultiAgentState) -> MultiAgentState:
        """Implementation phase."""
        result = await self.coder.run(
            f"Implement based on research: {state.research}"
        )
        state.code = result.output
        return state

    @node(name="test", depends_on=["implement"])
    async def write_tests(self, state: MultiAgentState) -> MultiAgentState:
        """Testing phase."""
        result = await self.tester.run(f"Write tests for: {state.code}")
        state.tests = result.output
        return state

    @node(name="document", depends_on=["implement", "test"])
    async def create_documentation(self, state: MultiAgentState) -> MultiAgentState:
        """Documentation phase."""
        result = await self.documenter.run(
            f"Document code and tests: {state.code} | {state.tests}"
        )
        state.documentation = result.output
        return state

# Execute development workflow
graph = SoftwareDevelopmentGraph()
built_graph = graph.build()

initial_state = MultiAgentState(task="Build a REST API for user management")
final_state = await built_graph.execute(initial_state)

print(f"Code:\n{final_state.code}\n")
print(f"Tests:\n{final_state.tests}\n")
print(f"Docs:\n{final_state.documentation}")
```

### Hierarchical Multi-Agent Graph

```python
from pydantic_ai.graph import Graph, node, hierarchical_graph
from pydantic import BaseModel

class TaskState(BaseModel):
    task: str
    subtasks: list[str] = []
    results: dict[str, str] = {}
    final_summary: str = ""

@hierarchical_graph
class HierarchicalTaskGraph:
    """Hierarchical graph with coordinator and workers."""

    @node(name="coordinator", role="coordinator")
    async def plan_and_coordinate(self, state: TaskState) -> TaskState:
        """Coordinator agent breaks down task."""
        coordinator_agent = Agent(
            'openai:gpt-4o',
            instructions='You are a coordinator. Break tasks into subtasks.'
        )

        result = await coordinator_agent.run(
            f"Break down this task into 3-5 subtasks: {state.task}"
        )

        state.subtasks = result.output.split('\n')
        return state

    @node(name="worker_pool", role="worker", parallel=True)
    async def execute_subtasks(self, state: TaskState) -> TaskState:
        """Worker agents execute subtasks in parallel."""
        worker_agent = Agent('anthropic:claude-3-5-sonnet-latest')

        # Execute each subtask
        import asyncio

        async def execute_one(subtask: str) -> tuple[str, str]:
            result = await worker_agent.run(f"Execute: {subtask}")
            return subtask, result.output

        # Parallel execution
        results = await asyncio.gather(*[
            execute_one(subtask) for subtask in state.subtasks
        ])

        state.results = dict(results)
        return state

    @node(name="aggregator", role="aggregator", depends_on=["worker_pool"])
    async def aggregate_results(self, state: TaskState) -> TaskState:
        """Aggregator combines worker results."""
        aggregator_agent = Agent('openai:gpt-4o')

        all_results = '\n'.join([
            f"{task}: {result}"
            for task, result in state.results.items()
        ])

        result = await aggregator_agent.run(
            f"Summarize these results: {all_results}"
        )

        state.final_summary = result.output
        return state

# Execute hierarchical workflow
graph = HierarchicalTaskGraph.build()
state = TaskState(task="Analyze quarterly sales data")
final = await graph.execute(state)

print(f"Subtasks: {final.subtasks}")
print(f"Results: {final.results}")
print(f"Summary: {final.final_summary}")
```

---

## Production Patterns

### Error Handling in Graphs

```python
from pydantic_ai.graph import Graph, node, error_handler
from pydantic import BaseModel
from typing import Optional

class ResilientState(BaseModel):
    input: str
    result: Optional[str] = None
    errors: list[str] = []
    retry_count: int = 0

@Graph(state_type=ResilientState)
class ResilientGraph:
    """Graph with comprehensive error handling."""

    @node
    @error_handler(max_retries=3, backoff=2.0)
    async def risky_operation(self, state: ResilientState) -> ResilientState:
        """Node that might fail."""
        try:
            agent = Agent('openai:gpt-4o')
            result = await agent.run(state.input)
            state.result = result.output
            return state

        except Exception as e:
            state.errors.append(str(e))
            state.retry_count += 1

            if state.retry_count >= 3:
                # Max retries reached - route to fallback
                return state, "fallback"

            # Retry
            raise

    @node(name="fallback")
    async def fallback_handler(self, state: ResilientState) -> ResilientState:
        """Fallback node when primary node fails."""
        # Use simpler model or cached response
        agent = Agent('openai:gpt-4o-mini')
        result = await agent.run(f"Simplified query: {state.input}")

        state.result = result.output
        return state

# Graph automatically retries failed nodes
graph = ResilientGraph.build()
result = await graph.execute(ResilientState(input="Complex query"))
```

### Monitoring and Observability

```python
from pydantic_ai.graph import Graph, node, monitor
from pydantic import BaseModel
import logfire

class MonitoredState(BaseModel):
    data: str
    processed: str = ""

@Graph(state_type=MonitoredState)
class MonitoredGraph:
    """Graph with full observability."""

    def __init__(self):
        # Configure Logfire
        logfire.configure()
        logfire.instrument_pydantic_ai()

    @node
    @monitor(
        track_duration=True,
        track_tokens=True,
        emit_events=True
    )
    async def process_data(self, state: MonitoredState) -> MonitoredState:
        """Monitored node with automatic tracking."""

        with logfire.span("graph_node", node_name="process_data"):
            agent = Agent('openai:gpt-4o')

            logfire.info("Processing data", data_length=len(state.data))

            result = await agent.run(state.data)

            logfire.info(
                "Node completed",
                tokens=result.usage().total_tokens,
                output_length=len(result.output)
            )

            state.processed = result.output
            return state

# Graph execution is automatically traced
graph = MonitoredGraph.build()

# All node executions appear in Logfire dashboard
result = await graph.execute(MonitoredState(data="Input data"))
```

### Production Graph Deployment

```python
from pydantic_ai.graph import Graph, node
from pydantic_ai.durable.prefect import PrefectDurableAgent
from prefect import flow, task
from pydantic import BaseModel

class ProductionState(BaseModel):
    """Production-ready state."""
    workflow_id: str
    input_data: dict
    results: dict = {}

@Graph(state_type=ProductionState, durable=True, backend='prefect')
class ProductionGraph:
    """Production graph with durability."""

    @node
    async def validate_input(self, state: ProductionState) -> ProductionState:
        """Validate input data."""
        # Input validation logic
        return state

    @node
    async def process(self, state: ProductionState) -> ProductionState:
        """Process with durable agent."""
        agent = PrefectDurableAgent(
            'openai:gpt-4o',
            checkpoint_every_n_steps=1
        )

        result = await agent.run(
            str(state.input_data),
            flow_run_name=f"graph-{state.workflow_id}"
        )

        state.results['processed'] = result.output
        return state

    @node
    async def save_results(self, state: ProductionState) -> ProductionState:
        """Save results to database."""
        # Database operations
        return state

# Deploy as Prefect flow
@flow(name="production-graph")
async def run_production_graph(input_data: dict):
    """Execute graph as Prefect flow."""
    graph = ProductionGraph.build()

    state = ProductionState(
        workflow_id=f"wf-{hash(str(input_data))}",
        input_data=input_data
    )

    final_state = await graph.execute(state)
    return final_state.results
```

---

## Summary

**Graph Support enables:**
- ✅ Declarative workflow definitions with type safety
- ✅ Automatic parallelization of independent nodes
- ✅ Complex conditional routing and state machines
- ✅ Visual workflow representation
- ✅ Durable execution integration
- ✅ Multi-agent coordination
- ✅ Production-ready error handling

**Key Patterns:**
- **DAGs**: Linear pipelines with dependencies
- **State Machines**: Finite state transitions
- **Conditional Graphs**: Dynamic routing based on state
- **Parallel Graphs**: Concurrent node execution
- **Hierarchical Graphs**: Coordinator-worker patterns

**Best Practices:**
1. Use type hints for automatic dependency resolution
2. Keep state models focused and minimal
3. Design for idempotency in nodes
4. Add error handlers for resilience
5. Monitor graph execution with Logfire
6. Visualize graphs for documentation

---

**Next Steps:**
1. Define your workflow as a graph
2. Use type hints to declare dependencies
3. Add conditional routing where needed
4. Enable parallel execution for independent nodes
5. Integrate with durable execution for long-running workflows
6. Monitor and optimize graph performance

**See Also:**
- `pydantic_ai_durable_execution.md` - Add durability to graphs
- `pydantic_ai_comprehensive_guide.md` - Core agent concepts
- `pydantic_ai_production_guide.md` - Production deployment
