# Mistral Agents API: Agent Orchestration Guide

**Version**: 2.0 (May 2025 Launch Edition)
**Last Updated**: May 27, 2025

## Overview

The May 27, 2025 Agents API launch introduced **built-in multi-agent orchestration** capabilities, enabling sophisticated agent collaboration without external frameworks. This guide covers sequential pipelines, parallel execution, hierarchical structures, and complex problem-solving workflows.

## Table of Contents

1. [Orchestration Fundamentals](#orchestration-fundamentals)
2. [Sequential Agent Pipelines](#sequential-agent-pipelines)
3. [Parallel Agent Execution](#parallel-agent-execution)
4. [Hierarchical Agent Structures](#hierarchical-agent-structures)
5. [Agent Handoff Mechanisms](#agent-handoff-mechanisms)
6. [State Management Across Agents](#state-management-across-agents)
7. [Complex Workflow Patterns](#complex-workflow-patterns)
8. [Error Handling in Orchestration](#error-handling-in-orchestration)
9. [Production Patterns](#production-patterns)

---

## Orchestration Fundamentals

### What is Agent Orchestration?

Agent orchestration is the coordination of multiple specialized agents to solve complex problems that exceed the capabilities of a single agent. The Mistral Agents API provides native orchestration through:

- **Conversation-based coordination**: Agents communicate via shared conversation context
- **Agent handoff mechanisms**: Transfer control between agents with full context
- **Persistent state**: Server-side conversation memory maintains state across agent transitions
- **No external frameworks**: Everything runs within Mistral platform

### Key Concepts

#### 1. Specialized Agents

```python
# Each agent has a specific role and expertise

research_agent = client.beta.agents.create(
    model="mistral-medium-latest",
    name="Research Specialist",
    instructions="You are a research specialist. Search for information and cite sources.",
    tools=[{"type": "web_search"}]
)

analysis_agent = client.beta.agents.create(
    model="mistral-large-latest",
    name="Data Analyst",
    instructions="You are a data analyst. Analyze data and create visualizations.",
    tools=[{"type": "code_interpreter"}]
)

writer_agent = client.beta.agents.create(
    model="mistral-medium-latest",
    name="Report Writer",
    instructions="You are a report writer. Create clear, professional reports.",
    tools=[{"type": "image_generation"}]
)
```

#### 2. Conversation as Coordination Medium

```python
# All agents share the same conversation for context continuity

# Agent 1 starts
conversation = client.beta.conversations.start(
    agent_id=research_agent.id,
    inputs="Research renewable energy trends"
)

# Agent 2 continues with full history
conversation = client.beta.conversations.append(
    conversation_id=conversation.id,
    agent_id=analysis_agent.id,
    inputs="Analyze the research findings"
)

# Agent 3 completes with full context
conversation = client.beta.conversations.append(
    conversation_id=conversation.id,
    agent_id=writer_agent.id,
    inputs="Write executive summary"
)
```

#### 3. State Persistence

```python
# Conversation history maintains all state automatically
history = client.beta.conversations.get_history(
    conversation_id=conversation.id
)

# Shows all agent interactions, tool executions, and results
for entry in history.entries:
    print(f"{entry.role} ({entry.agent_id}): {entry.type}")
```

---

## Sequential Agent Pipelines

Sequential orchestration chains agents where each agent's output feeds the next agent's input.

### Pattern: Linear Pipeline

```python
import os
from mistralai import Mistral

def create_data_pipeline():
    """Create a sequential data processing pipeline"""

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    # Agent 1: Data Collector
    collector = client.beta.agents.create(
        model="mistral-medium-latest",
        name="Data Collector",
        description="Collects raw data from various sources",
        instructions="""
        You collect data from web sources.
        Search for requested information and compile raw data.
        Format output as structured data (JSON or tables).
        Pass complete raw dataset to next agent.
        """,
        tools=[{"type": "web_search"}]
    )

    # Agent 2: Data Cleaner
    cleaner = client.beta.agents.create(
        model="mistral-medium-latest",
        name="Data Cleaner",
        description="Cleans and normalizes data",
        instructions="""
        You clean and normalize data.
        Remove duplicates, handle missing values, standardize formats.
        Use Python for data cleaning operations.
        Output clean dataset for analysis.
        """,
        tools=[{"type": "code_interpreter"}]
    )

    # Agent 3: Data Analyst
    analyst = client.beta.agents.create(
        model="mistral-large-latest",
        name="Data Analyst",
        description="Analyzes cleaned data",
        instructions="""
        You analyze cleaned data.
        Calculate statistics, identify trends, find insights.
        Create visualizations to illustrate findings.
        Provide detailed analysis for report writer.
        """,
        tools=[{"type": "code_interpreter"}]
    )

    # Agent 4: Report Writer
    writer = client.beta.agents.create(
        model="mistral-medium-latest",
        name="Report Writer",
        description="Creates final reports",
        instructions="""
        You write professional reports.
        Synthesize analysis into clear narrative.
        Include visualizations and key findings.
        Create executive summary.
        """,
        tools=[{"type": "image_generation"}]
    )

    return collector, cleaner, analyst, writer, client


def run_pipeline(user_request):
    """Execute the sequential pipeline"""

    collector, cleaner, analyst, writer, client = create_data_pipeline()

    print("🔄 Starting sequential pipeline...")

    # Step 1: Collect Data
    print("\n📊 Step 1: Collecting data...")
    conversation = client.beta.conversations.start(
        agent_id=collector.id,
        inputs=user_request
    )
    conversation_id = conversation.id
    print(f"✅ Data collected")

    # Step 2: Clean Data
    print("\n🧹 Step 2: Cleaning data...")
    conversation = client.beta.conversations.append(
        conversation_id=conversation_id,
        agent_id=cleaner.id,
        inputs="Clean and normalize the collected data. Handle any inconsistencies."
    )
    print(f"✅ Data cleaned")

    # Step 3: Analyze Data
    print("\n🔍 Step 3: Analyzing data...")
    conversation = client.beta.conversations.append(
        conversation_id=conversation_id,
        agent_id=analyst.id,
        inputs="Analyze the cleaned data. Find trends, calculate statistics, create visualizations."
    )
    print(f"✅ Analysis complete")

    # Step 4: Write Report
    print("\n📝 Step 4: Writing report...")
    conversation = client.beta.conversations.append(
        conversation_id=conversation_id,
        agent_id=writer.id,
        inputs="Create a comprehensive report based on the analysis. Include executive summary."
    )
    print(f"✅ Report complete")

    print("\n" + "="*50)
    print("FINAL REPORT")
    print("="*50)
    print(conversation.outputs[-1].content)

    return conversation_id


# Usage
if __name__ == "__main__":
    run_pipeline("Analyze global electric vehicle adoption rates in 2025")
```

### Pattern: Conditional Pipeline

```python
def create_conditional_pipeline():
    """Pipeline with conditional branching based on intermediate results"""

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    # Classifier Agent
    classifier = client.beta.agents.create(
        model="mistral-medium-latest",
        name="Request Classifier",
        instructions="""
        Classify requests into categories:
        - TECHNICAL: Requires technical analysis
        - RESEARCH: Requires web research
        - CREATIVE: Requires creative content
        - DATA: Requires data analysis

        Output: Single word classification
        """
    )

    # Specialized agents for each category
    technical_agent = client.beta.agents.create(
        model="mistral-large-latest",
        name="Technical Specialist",
        instructions="Handle technical queries with code execution",
        tools=[{"type": "code_interpreter"}]
    )

    research_agent = client.beta.agents.create(
        model="mistral-medium-latest",
        name="Research Specialist",
        instructions="Handle research queries with web search",
        tools=[{"type": "web_search"}]
    )

    creative_agent = client.beta.agents.create(
        model="mistral-medium-latest",
        name="Creative Specialist",
        instructions="Handle creative requests with image generation",
        tools=[{"type": "image_generation"}]
    )

    data_agent = client.beta.agents.create(
        model="mistral-large-latest",
        name="Data Specialist",
        instructions="Handle data analysis with code execution",
        tools=[{"type": "code_interpreter"}]
    )

    return {
        "classifier": classifier,
        "TECHNICAL": technical_agent,
        "RESEARCH": research_agent,
        "CREATIVE": creative_agent,
        "DATA": data_agent,
        "client": client
    }


def run_conditional_pipeline(user_request, agents):
    """Route request based on classification"""

    client = agents["client"]
    classifier = agents["classifier"]

    # Step 1: Classify
    conversation = client.beta.conversations.start(
        agent_id=classifier.id,
        inputs=f"Classify this request: {user_request}"
    )

    classification = conversation.outputs[-1].content.strip().upper()
    print(f"📋 Classification: {classification}")

    # Step 2: Route to appropriate agent
    specialist_agent = agents.get(classification)

    if specialist_agent:
        conversation = client.beta.conversations.append(
            conversation_id=conversation.id,
            agent_id=specialist_agent.id,
            inputs=user_request
        )
        print(f"✅ Processed by {classification} specialist")
    else:
        print(f"❌ Unknown classification: {classification}")

    return conversation


# Usage
agents = create_conditional_pipeline()
run_conditional_pipeline("What are the latest AI trends?", agents)  # → RESEARCH
run_conditional_pipeline("Calculate Fibonacci sequence", agents)     # → TECHNICAL
run_conditional_pipeline("Create a sunset landscape", agents)        # → CREATIVE
```

---

## Parallel Agent Execution

Parallel orchestration runs multiple agents simultaneously, then synthesizes results.

### Pattern: Fan-Out/Fan-In

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

def create_parallel_research_team():
    """Create team of parallel research agents"""

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    # Multiple specialist researchers
    tech_researcher = client.beta.agents.create(
        model="mistral-medium-latest",
        name="Tech Researcher",
        instructions="Research technology trends. Focus on technical details.",
        tools=[{"type": "web_search"}]
    )

    market_researcher = client.beta.agents.create(
        model="mistral-medium-latest",
        name="Market Researcher",
        instructions="Research market trends. Focus on business and economics.",
        tools=[{"type": "web_search"}]
    )

    academic_researcher = client.beta.agents.create(
        model="mistral-medium-latest",
        name="Academic Researcher",
        instructions="Research academic publications. Focus on peer-reviewed sources.",
        tools=[{"type": "web_search"}]
    )

    # Synthesizer agent
    synthesizer = client.beta.agents.create(
        model="mistral-large-latest",
        name="Research Synthesizer",
        instructions="""
        You synthesize research from multiple sources.
        Identify common themes, conflicts, and gaps.
        Create comprehensive unified report.
        Cite all sources from parallel researchers.
        """
    )

    return {
        "researchers": [tech_researcher, market_researcher, academic_researcher],
        "synthesizer": synthesizer,
        "client": client
    }


def run_parallel_research(topic):
    """Execute parallel research and synthesize"""

    team = create_parallel_research_team()
    client = team["client"]

    print(f"🔄 Starting parallel research on: {topic}")

    # Fan-out: Run researchers in parallel
    conversations = []

    def research_task(agent):
        """Execute single research task"""
        conv = client.beta.conversations.start(
            agent_id=agent.id,
            inputs=f"Research {topic} from your specialized perspective"
        )
        return conv

    # Execute in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(research_task, agent)
                   for agent in team["researchers"]]
        conversations = [future.result() for future in futures]

    print(f"✅ Parallel research complete ({len(conversations)} researchers)")

    # Fan-in: Synthesize results
    print("\n🔄 Synthesizing results...")

    # Combine all research into single context
    combined_research = "\n\n".join([
        f"RESEARCHER {i+1} FINDINGS:\n{conv.outputs[-1].content}"
        for i, conv in enumerate(conversations)
    ])

    # Create new conversation for synthesis
    synthesis_conv = client.beta.conversations.start(
        agent_id=team["synthesizer"].id,
        inputs=f"""
        Synthesize these parallel research findings:

        {combined_research}

        Create a comprehensive unified report.
        """
    )

    print("✅ Synthesis complete")
    print("\n" + "="*50)
    print("SYNTHESIZED REPORT")
    print("="*50)
    print(synthesis_conv.outputs[-1].content)

    return synthesis_conv


# Usage
if __name__ == "__main__":
    run_parallel_research("Artificial Intelligence in Healthcare")
```

### Pattern: Competitive Evaluation

```python
def create_competitive_team():
    """Create team with different approaches to same problem"""

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    # Multiple agents with different strategies
    conservative_agent = client.beta.agents.create(
        model="mistral-medium-latest",
        name="Conservative Analyst",
        instructions="""
        You take a conservative, risk-averse approach.
        Emphasize safety, proven methods, and gradual change.
        Highlight potential risks and downsides.
        """,
        completion_args={"temperature": 0.2}
    )

    aggressive_agent = client.beta.agents.create(
        model="mistral-medium-latest",
        name="Aggressive Analyst",
        instructions="""
        You take an aggressive, innovative approach.
        Emphasize opportunities, disruption, and rapid change.
        Highlight potential benefits and upsides.
        """,
        completion_args={"temperature": 0.8}
    )

    balanced_agent = client.beta.agents.create(
        model="mistral-medium-latest",
        name="Balanced Analyst",
        instructions="""
        You take a balanced, pragmatic approach.
        Weigh pros and cons equally.
        Seek middle-ground solutions.
        """,
        completion_args={"temperature": 0.5}
    )

    evaluator = client.beta.agents.create(
        model="mistral-large-latest",
        name="Strategy Evaluator",
        instructions="""
        You evaluate competing strategies.
        Compare approaches on multiple dimensions.
        Recommend best approach or hybrid solution.
        Provide clear rationale for recommendation.
        """
    )

    return {
        "analysts": [conservative_agent, aggressive_agent, balanced_agent],
        "evaluator": evaluator,
        "client": client
    }


def run_competitive_analysis(decision):
    """Get multiple perspectives and evaluate"""

    team = create_competitive_team()
    client = team["client"]

    print(f"🔄 Competitive analysis for: {decision}")

    # Get perspectives in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(
            lambda a: client.beta.conversations.start(
                agent_id=a.id,
                inputs=f"Analyze this decision: {decision}"
            ), agent
        ) for agent in team["analysts"]]

        perspectives = [future.result() for future in futures]

    # Combine perspectives
    combined = "\n\n".join([
        f"PERSPECTIVE {i+1} ({['CONSERVATIVE', 'AGGRESSIVE', 'BALANCED'][i]}):\n{p.outputs[-1].content}"
        for i, p in enumerate(perspectives)
    ])

    # Evaluate
    evaluation = client.beta.conversations.start(
        agent_id=team["evaluator"].id,
        inputs=f"""
        Evaluate these competing perspectives:

        {combined}

        Provide recommendation with rationale.
        """
    )

    print("\n" + "="*50)
    print("EVALUATION & RECOMMENDATION")
    print("="*50)
    print(evaluation.outputs[-1].content)

    return evaluation


# Usage
run_competitive_analysis("Should we migrate to microservices architecture?")
```

---

## Hierarchical Agent Structures

Hierarchical orchestration uses manager agents that delegate to specialist agents.

### Pattern: Manager-Worker

```python
def create_manager_worker_system():
    """Create hierarchical system with manager and workers"""

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    # Worker agents
    code_worker = client.beta.agents.create(
        model="mistral-medium-latest",
        name="Code Worker",
        instructions="Execute code tasks assigned by manager. Report results clearly.",
        tools=[{"type": "code_interpreter"}]
    )

    search_worker = client.beta.agents.create(
        model="mistral-medium-latest",
        name="Search Worker",
        instructions="Execute search tasks assigned by manager. Provide concise findings.",
        tools=[{"type": "web_search"}]
    )

    doc_worker = client.beta.agents.create(
        model="mistral-medium-latest",
        name="Doc Worker",
        instructions="Retrieve document information assigned by manager. Quote sources.",
        tools=[{"type": "document_retrieval", "config": {"library_id": "lib_xxx"}}]
    )

    # Manager agent
    manager = client.beta.agents.create(
        model="mistral-large-latest",
        name="Task Manager",
        instructions="""
        You are a task manager coordinating worker agents.

        Available workers:
        - Code Worker: Executes code, performs calculations
        - Search Worker: Searches web for information
        - Doc Worker: Retrieves company documents

        For each user request:
        1. Break down into subtasks
        2. Assign each subtask to appropriate worker
        3. Collect results from workers
        4. Synthesize final response

        Delegate efficiently. Track progress. Provide comprehensive answers.
        """
    )

    return {
        "manager": manager,
        "workers": {
            "code": code_worker,
            "search": search_worker,
            "doc": doc_worker
        },
        "client": client
    }


def run_manager_worker(user_request):
    """Execute task through manager-worker hierarchy"""

    system = create_manager_worker_system()
    client = system["client"]
    manager = system["manager"]
    workers = system["workers"]

    print(f"🔄 Manager processing: {user_request}")

    # Manager creates plan
    manager_conv = client.beta.conversations.start(
        agent_id=manager.id,
        inputs=f"""
        User request: {user_request}

        Create execution plan:
        1. List subtasks
        2. Assign each to worker (code/search/doc)
        3. Output plan in structured format
        """
    )

    plan = manager_conv.outputs[-1].content
    print(f"\n📋 Manager Plan:\n{plan}")

    # Simulate worker execution
    # In production, parse plan and execute workers
    print("\n🔄 Executing workers...")

    # For demo, continue conversation with manager
    result = client.beta.conversations.append(
        conversation_id=manager_conv.id,
        inputs="""
        Workers have completed their tasks. Synthesize results into final response.
        (In this demo, simulate worker completions)
        """
    )

    print("\n" + "="*50)
    print("MANAGER FINAL RESPONSE")
    print("="*50)
    print(result.outputs[-1].content)

    return result


# Usage
run_manager_worker("""
Analyze our company's Q4 performance:
1. Retrieve Q4 financial report from docs
2. Search for industry benchmarks
3. Calculate performance metrics
4. Compare to industry
""")
```

### Pattern: Recursive Decomposition

```python
def create_recursive_system():
    """Create system that recursively decomposes complex tasks"""

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    # Decomposer agent
    decomposer = client.beta.agents.create(
        model="mistral-large-latest",
        name="Task Decomposer",
        instructions="""
        You decompose complex tasks into simpler subtasks.

        For each task:
        1. Assess complexity
        2. If simple: mark as EXECUTABLE
        3. If complex: break into subtasks and mark as DECOMPOSE

        Output format:
        TASK: [description]
        COMPLEXITY: [SIMPLE/COMPLEX]
        ACTION: [EXECUTE/DECOMPOSE]
        SUBTASKS: [if decompose, list subtasks]
        """
    )

    # Executor agent
    executor = client.beta.agents.create(
        model="mistral-medium-latest",
        name="Task Executor",
        instructions="Execute simple, well-defined tasks. Provide clear results.",
        tools=[
            {"type": "code_interpreter"},
            {"type": "web_search"}
        ]
    )

    return {"decomposer": decomposer, "executor": executor, "client": client}


def recursive_execute(task, system, depth=0, max_depth=3):
    """Recursively decompose and execute task"""

    if depth >= max_depth:
        print(f"{'  '*depth}⚠️  Max depth reached")
        return None

    client = system["client"]
    decomposer = system["decomposer"]
    executor = system["executor"]

    print(f"{'  '*depth}🔄 Processing: {task[:50]}...")

    # Decompose
    decomp_conv = client.beta.conversations.start(
        agent_id=decomposer.id,
        inputs=f"Analyze this task: {task}"
    )

    response = decomp_conv.outputs[-1].content

    # Check if executable
    if "ACTION: EXECUTE" in response or "COMPLEXITY: SIMPLE" in response:
        print(f"{'  '*depth}▶️  Executing...")
        exec_conv = client.beta.conversations.start(
            agent_id=executor.id,
            inputs=task
        )
        result = exec_conv.outputs[-1].content
        print(f"{'  '*depth}✅ Completed")
        return result

    # Otherwise, decompose further
    elif "ACTION: DECOMPOSE" in response or "COMPLEXITY: COMPLEX" in response:
        print(f"{'  '*depth}📋 Decomposing...")
        # In production, parse subtasks from response
        # For demo, simulate subtasks
        subtasks = [
            f"Subtask 1 of: {task[:30]}",
            f"Subtask 2 of: {task[:30]}"
        ]

        results = []
        for i, subtask in enumerate(subtasks):
            print(f"{'  '*depth}├─ Subtask {i+1}:")
            result = recursive_execute(subtask, system, depth+1, max_depth)
            results.append(result)

        print(f"{'  '*depth}🔄 Synthesizing subtask results...")
        return results

    return None


# Usage
system = create_recursive_system()
recursive_execute(
    "Build a comprehensive AI strategy including technology assessment, talent planning, and implementation roadmap",
    system
)
```

---

## Agent Handoff Mechanisms

Agent handoffs transfer control between agents with full context preservation.

### Pattern: Explicit Handoff

```python
def create_handoff_system():
    """Create system with explicit agent handoffs"""

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    # Customer service agent
    cs_agent = client.beta.agents.create(
        model="mistral-medium-latest",
        name="Customer Service",
        instructions="""
        You handle general customer inquiries.

        If customer needs technical support, say: HANDOFF: TECHNICAL
        If customer needs billing support, say: HANDOFF: BILLING
        If customer needs sales info, say: HANDOFF: SALES

        Otherwise, handle the inquiry directly.
        """,
        completion_args={"temperature": 0.3}
    )

    # Specialist agents
    technical_agent = client.beta.agents.create(
        model="mistral-large-latest",
        name="Technical Support",
        instructions="You provide technical support. Access documentation and solve technical issues.",
        tools=[
            {"type": "document_retrieval", "config": {"library_id": "lib_tech_docs"}},
            {"type": "code_interpreter"}
        ]
    )

    billing_agent = client.beta.agents.create(
        model="mistral-medium-latest",
        name="Billing Support",
        instructions="You handle billing inquiries. Access billing records and explain charges.",
        tools=[{"type": "document_retrieval", "config": {"library_id": "lib_billing"}}]
    )

    sales_agent = client.beta.agents.create(
        model="mistral-medium-latest",
        name="Sales",
        instructions="You provide sales information. Explain features, pricing, and help with purchases.",
        tools=[{"type": "web_search"}]
    )

    return {
        "cs": cs_agent,
        "technical": technical_agent,
        "billing": billing_agent,
        "sales": sales_agent,
        "client": client
    }


def run_with_handoffs(user_message):
    """Execute with automatic handoffs"""

    system = create_handoff_system()
    client = system["client"]

    print(f"👤 User: {user_message}")

    # Start with CS agent
    current_agent = system["cs"]
    conversation = client.beta.conversations.start(
        agent_id=current_agent.id,
        inputs=user_message
    )

    response = conversation.outputs[-1].content
    print(f"🤖 CS Agent: {response}")

    # Check for handoff
    if "HANDOFF: TECHNICAL" in response:
        print("\n🔄 Handing off to Technical Support...")
        current_agent = system["technical"]
        conversation = client.beta.conversations.append(
            conversation_id=conversation.id,
            agent_id=current_agent.id,
            inputs="Customer needs technical support. Please assist."
        )
        print(f"🔧 Technical: {conversation.outputs[-1].content}")

    elif "HANDOFF: BILLING" in response:
        print("\n🔄 Handing off to Billing...")
        current_agent = system["billing"]
        conversation = client.beta.conversations.append(
            conversation_id=conversation.id,
            agent_id=current_agent.id,
            inputs="Customer needs billing support. Please assist."
        )
        print(f"💰 Billing: {conversation.outputs[-1].content}")

    elif "HANDOFF: SALES" in response:
        print("\n🔄 Handing off to Sales...")
        current_agent = system["sales"]
        conversation = client.beta.conversations.append(
            conversation_id=conversation.id,
            agent_id=current_agent.id,
            inputs="Customer needs sales information. Please assist."
        )
        print(f"💼 Sales: {conversation.outputs[-1].content}")

    return conversation


# Usage
run_with_handoffs("My API returns 500 errors")        # → Technical
run_with_handoffs("Why was I charged twice?")         # → Billing
run_with_handoffs("What features are in Pro plan?")   # → Sales
run_with_handoffs("What's your return policy?")       # → CS handles directly
```

---

## State Management Across Agents

Managing state as agents collaborate is critical for coherent orchestration.

### Pattern: Shared Conversation Context

```python
# All agents access same conversation history automatically
conversation = client.beta.conversations.start(
    agent_id=agent1.id,
    inputs="Initial request"
)

# Agent 2 sees Agent 1's work
conversation = client.beta.conversations.append(
    conversation_id=conversation.id,
    agent_id=agent2.id,
    inputs="Continue from where Agent 1 left off"
)

# Retrieve full history anytime
history = client.beta.conversations.get_history(
    conversation_id=conversation.id
)
```

### Pattern: Explicit State Passing

```python
def create_stateful_pipeline():
    """Pipeline with explicit state tracking"""

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    agent1 = client.beta.agents.create(
        model="mistral-medium-latest",
        name="Agent 1",
        instructions="""
        Process request and output state:
        STATE: {key: value, ...}
        RESULT: [your result]
        """
    )

    agent2 = client.beta.agents.create(
        model="mistral-medium-latest",
        name="Agent 2",
        instructions="""
        You receive state from previous agent.
        Process using provided state.
        Update state and pass forward:
        STATE: {updated state}
        RESULT: [your result]
        """
    )

    return agent1, agent2, client


# Usage
agent1, agent2, client = create_stateful_pipeline()

# Agent 1 creates state
conv = client.beta.conversations.start(
    agent_id=agent1.id,
    inputs="Process this data: [1,2,3,4,5]"
)

response1 = conv.outputs[-1].content
# Parse STATE from response1

# Agent 2 receives state
conv = client.beta.conversations.append(
    conversation_id=conv.id,
    agent_id=agent2.id,
    inputs=f"Previous state: {response1}\n\nContinue processing"
)
```

### Pattern: External State Store

```python
import json
from typing import Dict, Any

class OrchestrationState:
    """External state management for complex orchestrations"""

    def __init__(self):
        self.state: Dict[str, Any] = {}

    def update(self, key: str, value: Any):
        """Update state"""
        self.state[key] = value

    def get(self, key: str, default=None):
        """Get state value"""
        return self.state.get(key, default)

    def serialize(self) -> str:
        """Serialize state for passing to agents"""
        return json.dumps(self.state, indent=2)

    def inject_into_prompt(self, base_prompt: str) -> str:
        """Inject state into agent prompt"""
        return f"""{base_prompt}

CURRENT STATE:
{self.serialize()}

Use this state information to inform your processing.
"""


# Usage
state = OrchestrationState()

# Agent 1 updates state
conv1 = client.beta.conversations.start(
    agent_id=agent1.id,
    inputs="Collect data"
)
state.update("data_collected", True)
state.update("data_count", 100)

# Agent 2 receives state
conv2 = client.beta.conversations.start(
    agent_id=agent2.id,
    inputs=state.inject_into_prompt("Analyze the collected data")
)

# Agent 3 receives updated state
state.update("analysis_complete", True)
conv3 = client.beta.conversations.start(
    agent_id=agent3.id,
    inputs=state.inject_into_prompt("Create report from analysis")
)
```

---

## Complex Workflow Patterns

### Pattern: Event-Driven Orchestration

```python
class EventDrivenOrchestrator:
    """Orchestrator that responds to events"""

    def __init__(self, client):
        self.client = client
        self.event_handlers = {}
        self.conversation_id = None

    def register_handler(self, event_type: str, agent):
        """Register agent as handler for event type"""
        self.event_handlers[event_type] = agent

    def emit_event(self, event_type: str, data: str):
        """Emit event and trigger appropriate handler"""
        print(f"📢 Event: {event_type}")

        handler = self.event_handlers.get(event_type)
        if not handler:
            print(f"⚠️  No handler for {event_type}")
            return None

        # Execute handler agent
        if self.conversation_id:
            conv = self.client.beta.conversations.append(
                conversation_id=self.conversation_id,
                agent_id=handler.id,
                inputs=f"Event: {event_type}\nData: {data}"
            )
        else:
            conv = self.client.beta.conversations.start(
                agent_id=handler.id,
                inputs=f"Event: {event_type}\nData: {data}"
            )
            self.conversation_id = conv.id

        print(f"✅ Handler executed: {handler.name}")
        return conv.outputs[-1].content


# Usage
client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

orchestrator = EventDrivenOrchestrator(client)

# Register handlers
data_handler = client.beta.agents.create(
    model="mistral-medium-latest",
    name="Data Handler",
    instructions="Process data received events"
)

error_handler = client.beta.agents.create(
    model="mistral-medium-latest",
    name="Error Handler",
    instructions="Handle and recover from errors"
)

orchestrator.register_handler("data.received", data_handler)
orchestrator.register_handler("error.occurred", error_handler)

# Emit events
orchestrator.emit_event("data.received", "New data batch: 1000 records")
orchestrator.emit_event("error.occurred", "Validation failed on record 523")
```

---

## Production Patterns

### Complete Production Example

```python
import os
import logging
from typing import List, Dict, Any
from mistralai import Mistral

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionOrchestrator:
    """Production-grade agent orchestration system"""

    def __init__(self):
        self.client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
        self.agents = {}
        self._initialize_agents()

    def _initialize_agents(self):
        """Initialize all agents"""
        logger.info("Initializing agent team...")

        # Research agent
        self.agents["researcher"] = self.client.beta.agents.create(
            model="mistral-medium-latest",
            name="Research Agent",
            instructions="Conduct thorough research. Cite sources. Be comprehensive.",
            tools=[{"type": "web_search", "config": {"enable_premium_sources": True}}]
        )

        # Analysis agent
        self.agents["analyst"] = self.client.beta.agents.create(
            model="mistral-large-latest",
            name="Analysis Agent",
            instructions="Analyze data with statistical rigor. Create visualizations.",
            tools=[{"type": "code_interpreter"}]
        )

        # Report agent
        self.agents["reporter"] = self.client.beta.agents.create(
            model="mistral-medium-latest",
            name="Report Agent",
            instructions="Create professional, clear reports. Include executive summaries.",
            tools=[{"type": "image_generation"}]
        )

        logger.info(f"Initialized {len(self.agents)} agents")

    def execute_workflow(self, workflow_type: str, inputs: str) -> Dict[str, Any]:
        """Execute predefined workflow"""
        logger.info(f"Executing workflow: {workflow_type}")

        try:
            if workflow_type == "research_report":
                return self._research_report_workflow(inputs)
            elif workflow_type == "data_analysis":
                return self._data_analysis_workflow(inputs)
            else:
                raise ValueError(f"Unknown workflow: {workflow_type}")

        except Exception as e:
            logger.error(f"Workflow failed: {e}")
            return {"success": False, "error": str(e)}

    def _research_report_workflow(self, topic: str) -> Dict[str, Any]:
        """Research → Analysis → Report pipeline"""

        results = {}

        # Step 1: Research
        logger.info("Step 1: Research")
        research_conv = self.client.beta.conversations.start(
            agent_id=self.agents["researcher"].id,
            inputs=f"Research: {topic}"
        )
        results["research"] = research_conv.outputs[-1].content
        conversation_id = research_conv.id

        # Step 2: Analysis
        logger.info("Step 2: Analysis")
        analysis_conv = self.client.beta.conversations.append(
            conversation_id=conversation_id,
            agent_id=self.agents["analyst"].id,
            inputs="Analyze the research findings. Identify key trends and insights."
        )
        results["analysis"] = analysis_conv.outputs[-1].content

        # Step 3: Report
        logger.info("Step 3: Report")
        report_conv = self.client.beta.conversations.append(
            conversation_id=conversation_id,
            agent_id=self.agents["reporter"].id,
            inputs="Create comprehensive report with executive summary."
        )
        results["report"] = report_conv.outputs[-1].content

        logger.info("Workflow complete")
        return {
            "success": True,
            "conversation_id": conversation_id,
            "results": results
        }

    def _data_analysis_workflow(self, data: str) -> Dict[str, Any]:
        """Analysis → Report pipeline"""

        results = {}

        # Step 1: Analysis
        logger.info("Step 1: Analysis")
        analysis_conv = self.client.beta.conversations.start(
            agent_id=self.agents["analyst"].id,
            inputs=f"Analyze this data: {data}"
        )
        results["analysis"] = analysis_conv.outputs[-1].content
        conversation_id = analysis_conv.id

        # Step 2: Report
        logger.info("Step 2: Report")
        report_conv = self.client.beta.conversations.append(
            conversation_id=conversation_id,
            agent_id=self.agents["reporter"].id,
            inputs="Create report from analysis."
        )
        results["report"] = report_conv.outputs[-1].content

        logger.info("Workflow complete")
        return {
            "success": True,
            "conversation_id": conversation_id,
            "results": results
        }


# Usage
if __name__ == "__main__":
    orchestrator = ProductionOrchestrator()

    # Execute workflow
    result = orchestrator.execute_workflow(
        workflow_type="research_report",
        inputs="Quantum Computing Applications in 2025"
    )

    if result["success"]:
        print("\n" + "="*50)
        print("FINAL REPORT")
        print("="*50)
        print(result["results"]["report"])
        print(f"\nConversation ID: {result['conversation_id']}")
    else:
        print(f"Error: {result['error']}")
```

---

## Conclusion

The May 2025 Agents API launch brings native multi-agent orchestration capabilities:

1. **Sequential Pipelines** - Chain agents for data processing workflows
2. **Parallel Execution** - Run agents concurrently for faster results
3. **Hierarchical Structures** - Manager-worker patterns for complex delegation
4. **Agent Handoffs** - Transfer control between specialized agents
5. **State Management** - Built-in conversation persistence for context continuity

No external frameworks needed—everything runs on Mistral's platform with enterprise reliability.

**Next Steps:**
- Review [Connectors Guide](mistral_agents_api_connectors_guide.md) for agent capabilities
- Explore [MCP Integration](mistral_agents_api_mcp_guide.md) for third-party tools
- Study [Production Guide](mistral_agents_api_production_guide.md) for deployment

---

**Documentation Version**: 2.0 (May 2025 Launch Edition)
**Last Updated**: May 27, 2025
**Mistral AI - Orchestrate Intelligence**
