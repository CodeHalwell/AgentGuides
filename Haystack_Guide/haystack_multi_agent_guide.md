# Haystack Multi-Agent Systems Guide (2025)

**Complete Guide to Building Collaborative Multi-Agent Applications**

---

## Table of Contents

1. [Introduction to Multi-Agent Systems](#introduction-to-multi-agent-systems)
2. [Multi-Agent Architecture Patterns](#multi-agent-architecture-patterns)
3. [Agent Specialization & Roles](#agent-specialization--roles)
4. [Agent Communication Protocols](#agent-communication-protocols)
5. [Coordination & Orchestration](#coordination--orchestration)
6. [Multi-Agent Pipelines](#multi-agent-pipelines)
7. [State Sharing & Memory](#state-sharing--memory)
8. [Error Handling & Resilience](#error-handling--resilience)
9. [Performance Optimization](#performance-optimization)
10. [Production Deployment](#production-deployment)

---

## Introduction to Multi-Agent Systems

Multi-agent systems in Haystack (2025) enable **multiple specialized agents to collaborate** on complex tasks that would be difficult for a single agent to handle effectively.

### Why Multi-Agent Systems?

```python
# Single Agent Limitation:
# One agent trying to do research, analysis, coding, and writing
# → Jack of all trades, master of none

# Multi-Agent Solution:
# - Research Agent: Expert at finding information
# - Analysis Agent: Expert at data processing
# - Coding Agent: Expert at implementation
# - Writing Agent: Expert at communication
# → Each agent excels in its domain
```

### Core Concepts

```python
from haystack import Pipeline
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator

# Multi-agent system components:
# 1. Specialized Agents - Each with domain expertise
# 2. Orchestrator - Coordinates agent collaboration
# 3. Communication Protocol - How agents share information
# 4. Shared State - Common knowledge and context
# 5. Workflow - How tasks flow between agents

def create_simple_multi_agent_system():
    """
    Basic multi-agent system with two collaborating agents
    """
    pipeline = Pipeline()

    # Agent 1: Research specialist
    research_agent = Agent(
        llm=OpenAIChatGenerator(model="gpt-4o"),
        tools=[WebSearchTool(), DatabaseTool()],
        system_prompt="You are a research specialist. Find comprehensive information."
    )

    # Agent 2: Writing specialist
    writing_agent = Agent(
        llm=OpenAIChatGenerator(model="gpt-4o"),
        tools=[],
        system_prompt="You are a writing specialist. Create clear, engaging content."
    )

    # Connect agents in sequence
    pipeline.add_component("researcher", research_agent)
    pipeline.add_component("writer", writing_agent)
    pipeline.connect("researcher.output", "writer.input")

    return pipeline

# Usage
multi_agent = create_simple_multi_agent_system()
result = multi_agent.run({
    "researcher": {"query": "Latest developments in quantum computing"}
})
print(result["writer"]["output"])
# Researcher finds information → Writer creates polished article
```

---

## Multi-Agent Architecture Patterns

### Pattern 1: Sequential Collaboration

Agents work in a chain, each building on the previous agent's output:

```python
from haystack import Pipeline
from haystack.components.agents import Agent

def sequential_pattern():
    """
    Agent1 → Agent2 → Agent3 → Agent4
    Each agent processes and passes to next
    """
    pipeline = Pipeline()

    # Define specialized agents
    agents = [
        ("data_collector", create_data_collector_agent()),
        ("preprocessor", create_preprocessing_agent()),
        ("analyzer", create_analysis_agent()),
        ("reporter", create_reporting_agent())
    ]

    # Add to pipeline
    for name, agent in agents:
        pipeline.add_component(name, agent)

    # Connect sequentially
    pipeline.connect("data_collector.output", "preprocessor.input")
    pipeline.connect("preprocessor.output", "analyzer.input")
    pipeline.connect("analyzer.output", "reporter.input")

    return pipeline

def create_data_collector_agent():
    return Agent(
        llm=OpenAIChatGenerator(model="gpt-4o"),
        tools=[WebSearchTool(), APITool()],
        system_prompt="Collect comprehensive data from multiple sources"
    )

def create_preprocessing_agent():
    return Agent(
        llm=OpenAIChatGenerator(model="gpt-4o"),
        tools=[DataCleaningTool(), ValidationTool()],
        system_prompt="Clean, validate, and structure collected data"
    )

def create_analysis_agent():
    return Agent(
        llm=OpenAIChatGenerator(model="gpt-4o"),
        tools=[StatisticsTool(), MLTool()],
        system_prompt="Perform deep analysis and extract insights"
    )

def create_reporting_agent():
    return Agent(
        llm=OpenAIChatGenerator(model="gpt-4o"),
        tools=[VisualizationTool(), FormattingTool()],
        system_prompt="Create comprehensive, visually appealing reports"
    )

# Execute sequential workflow
sequential = sequential_pattern()
result = sequential.run({
    "data_collector": {"query": "Q4 2024 sales performance"}
})
```

### Pattern 2: Parallel Processing

Multiple agents work simultaneously on different aspects:

```python
from haystack import Pipeline
from haystack.components.joiners import DocumentJoiner

def parallel_pattern():
    """
    Task splits into parallel streams:
           → Agent1 →
    Task → → Agent2 → Joiner → Result
           → Agent3 →
    """
    pipeline = Pipeline()

    # Parallel agents for different aspects
    pipeline.add_component("market_analyst", create_market_agent())
    pipeline.add_component("competitor_analyst", create_competitor_agent())
    pipeline.add_component("customer_analyst", create_customer_agent())

    # Joiner combines outputs
    pipeline.add_component("synthesizer", DocumentJoiner())

    # All agents receive same input
    # Each produces specialized output
    # Joiner synthesizes results

    pipeline.connect("market_analyst.output", "synthesizer.documents")
    pipeline.connect("competitor_analyst.output", "synthesizer.documents")
    pipeline.connect("customer_analyst.output", "synthesizer.documents")

    return pipeline

def create_market_agent():
    return Agent(
        llm=OpenAIChatGenerator(model="gpt-4o"),
        tools=[MarketDataTool(), TrendAnalysisTool()],
        system_prompt="Analyze market trends and conditions"
    )

def create_competitor_agent():
    return Agent(
        llm=OpenAIChatGenerator(model="gpt-4o"),
        tools=[CompetitorTool(), BenchmarkingTool()],
        system_prompt="Analyze competitor strategies and positioning"
    )

def create_customer_agent():
    return Agent(
        llm=OpenAIChatGenerator(model="gpt-4o"),
        tools=[CustomerDataTool(), SentimentTool()],
        system_prompt="Analyze customer behavior and sentiment"
    )

# Execute parallel analysis
parallel = parallel_pattern()
result = parallel.run({
    "market_analyst": {"query": "Tech industry 2025"},
    "competitor_analyst": {"query": "Tech industry 2025"},
    "customer_analyst": {"query": "Tech industry 2025"}
})
```

### Pattern 3: Hierarchical Delegation

Manager agent delegates to specialist agents:

```python
from haystack import Pipeline
from haystack.components.agents import Agent

def hierarchical_pattern():
    """
    Manager Agent
         ├─→ Specialist Agent 1
         ├─→ Specialist Agent 2
         └─→ Specialist Agent 3

    Manager decides which specialists to use
    """
    pipeline = Pipeline()

    # Specialist agents
    research_agent = Agent(
        llm=OpenAIChatGenerator(model="gpt-4o"),
        tools=[WebSearchTool()],
        system_prompt="Expert researcher"
    )

    coding_agent = Agent(
        llm=OpenAIChatGenerator(model="gpt-4o"),
        tools=[PythonTool(), GitTool()],
        system_prompt="Expert programmer"
    )

    data_agent = Agent(
        llm=OpenAIChatGenerator(model="gpt-4o"),
        tools=[SQLTool(), AnalyticsTool()],
        system_prompt="Expert data analyst"
    )

    # Manager agent with access to specialists
    manager = Agent(
        llm=OpenAIChatGenerator(model="gpt-4o"),
        managed_agents=[research_agent, coding_agent, data_agent],
        system_prompt="""You are a project manager.
        Delegate tasks to appropriate specialists:
        - researcher: for information gathering
        - coder: for implementation tasks
        - analyst: for data analysis"""
    )

    pipeline.add_component("manager", manager)

    return pipeline

# Manager intelligently delegates
hierarchical = hierarchical_pattern()
result = hierarchical.run({
    "manager": {"query": "Build a sales dashboard with latest data"}
})
# Manager will:
# 1. Ask data_agent to gather sales data
# 2. Ask coding_agent to build dashboard
# 3. Ask research_agent for best practices
# 4. Synthesize final solution
```

### Pattern 4: Iterative Refinement Loop

Agents collaborate in cycles to refine outputs:

```python
from haystack import Pipeline
from haystack.components.routers import ConditionalRouter

def iterative_refinement_pattern():
    """
    Creator Agent → Critic Agent → Quality Check
                 ↑___________________|
                 (loop if not good enough)
    """
    pipeline = Pipeline()

    # Creator agent
    creator = Agent(
        llm=OpenAIChatGenerator(model="gpt-4o"),
        tools=[WritingTool(), ResearchTool()],
        system_prompt="Create high-quality content"
    )

    # Critic agent
    critic = Agent(
        llm=OpenAIChatGenerator(model="gpt-4o"),
        tools=[AnalysisTool()],
        system_prompt="Provide constructive criticism and improvement suggestions"
    )

    # Quality checker
    quality_router = ConditionalRouter(
        routes=[
            {
                "condition": "{quality_score} >= 0.9",
                "output_name": "approved",
                "output": "{content}"
            },
            {
                "condition": "{quality_score} < 0.9",
                "output_name": "revise",
                "output": "{feedback}"
            }
        ]
    )

    pipeline.add_component("creator", creator)
    pipeline.add_component("critic", critic)
    pipeline.add_component("quality_check", quality_router)

    # Connect with feedback loop
    pipeline.connect("creator.output", "critic.input")
    pipeline.connect("critic.feedback", "quality_check.feedback")
    pipeline.connect("quality_check.revise", "creator.revision_feedback")

    return pipeline

# Iterative improvement
iterative = iterative_refinement_pattern()
result = iterative.run({
    "creator": {"query": "Write executive summary of Q4 performance"}
})
# Creator writes → Critic reviews → If not good enough, loops back
```

### Pattern 5: Consensus Building

Multiple agents vote or reach agreement:

```python
from haystack import Pipeline
from haystack.components.joiners import AnswerJoiner

def consensus_pattern():
    """
    Multiple agents analyze independently,
    then consensus mechanism selects best answer
    """
    pipeline = Pipeline()

    # Multiple independent agents with different approaches
    agent_optimistic = Agent(
        llm=OpenAIChatGenerator(model="gpt-4o", temperature=0.9),
        tools=[AnalysisTool()],
        system_prompt="Analyze with optimistic perspective"
    )

    agent_pessimistic = Agent(
        llm=OpenAIChatGenerator(model="gpt-4o", temperature=0.3),
        tools=[AnalysisTool()],
        system_prompt="Analyze with conservative perspective"
    )

    agent_neutral = Agent(
        llm=OpenAIChatGenerator(model="gpt-4o", temperature=0.5),
        tools=[AnalysisTool()],
        system_prompt="Analyze with balanced perspective"
    )

    # Consensus builder
    consensus = Agent(
        llm=OpenAIChatGenerator(model="gpt-4o"),
        tools=[],
        system_prompt="Synthesize multiple viewpoints into consensus"
    )

    pipeline.add_component("optimist", agent_optimistic)
    pipeline.add_component("pessimist", agent_pessimistic)
    pipeline.add_component("neutral", agent_neutral)
    pipeline.add_component("consensus_builder", consensus)

    # All feed into consensus
    joiner = AnswerJoiner()
    pipeline.add_component("joiner", joiner)

    pipeline.connect("optimist.output", "joiner.answers")
    pipeline.connect("pessimist.output", "joiner.answers")
    pipeline.connect("neutral.output", "joiner.answers")
    pipeline.connect("joiner.output", "consensus_builder.input")

    return pipeline

# Multiple perspectives → consensus
consensus_system = consensus_pattern()
result = consensus_system.run({
    "optimist": {"query": "Forecast 2025 growth"},
    "pessimist": {"query": "Forecast 2025 growth"},
    "neutral": {"query": "Forecast 2025 growth"}
})
```

---

## Agent Specialization & Roles

### Defining Agent Expertise

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator

class AgentRole:
    """Define specialized agent roles"""

    @staticmethod
    def create_research_agent():
        return Agent(
            llm=OpenAIChatGenerator(model="gpt-4o"),
            tools=[
                WebSearchTool(),
                ScholarTool(),
                DatabaseTool(),
                NewsTool()
            ],
            system_prompt="""You are an expert research analyst.
            Your role:
            - Find comprehensive, accurate information
            - Verify sources and fact-check
            - Synthesize information from multiple sources
            - Cite your sources clearly
            - Identify knowledge gaps"""
        )

    @staticmethod
    def create_data_scientist_agent():
        return Agent(
            llm=OpenAIChatGenerator(model="gpt-4o"),
            tools=[
                PythonTool(),
                SQLTool(),
                StatisticsTool(),
                VisualizationTool(),
                MLTool()
            ],
            system_prompt="""You are an expert data scientist.
            Your role:
            - Analyze data rigorously
            - Apply appropriate statistical methods
            - Build predictive models
            - Create insightful visualizations
            - Explain findings clearly"""
        )

    @staticmethod
    def create_software_engineer_agent():
        return Agent(
            llm=OpenAIChatGenerator(model="gpt-4o"),
            tools=[
                PythonTool(),
                GitTool(),
                TestingTool(),
                DocumentationTool()
            ],
            system_prompt="""You are an expert software engineer.
            Your role:
            - Write clean, efficient code
            - Follow best practices and patterns
            - Test thoroughly
            - Document clearly
            - Consider edge cases"""
        )

    @staticmethod
    def create_business_analyst_agent():
        return Agent(
            llm=OpenAIChatGenerator(model="gpt-4o"),
            tools=[
                FinancialDataTool(),
                MarketAnalysisTool(),
                CompetitorTool()
            ],
            system_prompt="""You are an expert business analyst.
            Your role:
            - Understand business context
            - Identify opportunities and risks
            - Provide strategic recommendations
            - Consider financial implications
            - Think about stakeholders"""
        )

    @staticmethod
    def create_technical_writer_agent():
        return Agent(
            llm=OpenAIChatGenerator(model="gpt-4o"),
            tools=[
                DocumentationTool(),
                DiagramTool()
            ],
            system_prompt="""You are an expert technical writer.
            Your role:
            - Explain complex concepts clearly
            - Structure information logically
            - Write for target audience
            - Create helpful examples
            - Use appropriate formatting"""
        )

# Build multi-agent team with specialized roles
def create_product_development_team():
    pipeline = Pipeline()

    pipeline.add_component("researcher", AgentRole.create_research_agent())
    pipeline.add_component("data_scientist", AgentRole.create_data_scientist_agent())
    pipeline.add_component("engineer", AgentRole.create_software_engineer_agent())
    pipeline.add_component("business_analyst", AgentRole.create_business_analyst_agent())
    pipeline.add_component("writer", AgentRole.create_technical_writer_agent())

    # Connect workflow
    # Research → Business Analysis → Data Science → Engineering → Documentation
    pipeline.connect("researcher.output", "business_analyst.input")
    pipeline.connect("business_analyst.output", "data_scientist.input")
    pipeline.connect("data_scientist.output", "engineer.input")
    pipeline.connect("engineer.output", "writer.input")

    return pipeline
```

---

## Agent Communication Protocols

### Message Passing Between Agents

```python
from haystack import Pipeline
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage

class AgentMessage:
    """Structured message format for agent communication"""

    def __init__(self, sender: str, recipient: str, content: str, metadata: dict = None):
        self.sender = sender
        self.recipient = recipient
        self.content = content
        self.metadata = metadata or {}
        self.timestamp = datetime.now()

    def to_chat_message(self) -> ChatMessage:
        """Convert to Haystack ChatMessage format"""
        return ChatMessage.from_assistant(
            f"From {self.sender}: {self.content}"
        )

def create_communicating_agents():
    """Agents that explicitly communicate with message passing"""

    pipeline = Pipeline()

    # Agent 1: Sends structured requests
    agent1 = Agent(
        llm=OpenAIChatGenerator(model="gpt-4o"),
        tools=[],
        system_prompt="""You coordinate tasks.
        Send clear requests to other agents in this format:
        TO: [agent_name]
        REQUEST: [what you need]
        CONTEXT: [background information]"""
    )

    # Agent 2: Processes requests and responds
    agent2 = Agent(
        llm=OpenAIChatGenerator(model="gpt-4o"),
        tools=[DataTool()],
        system_prompt="""You handle data requests.
        Read incoming messages and respond with:
        FROM: [your name]
        RESPONSE: [your response]
        STATUS: [success/failure]"""
    )

    pipeline.add_component("coordinator", agent1)
    pipeline.add_component("data_handler", agent2)
    pipeline.connect("coordinator.output", "data_handler.input")

    return pipeline
```

### Shared Context & State

```python
from haystack.components.memory import ConversationMemory

class SharedAgentState:
    """Shared state accessible to all agents"""

    def __init__(self):
        self.shared_context = {}
        self.agent_outputs = {}
        self.conversation_history = ConversationMemory()

    def update(self, agent_name: str, data: dict):
        """Agent updates shared state"""
        self.agent_outputs[agent_name] = data
        self.shared_context.update(data)

    def get_context_for_agent(self, agent_name: str) -> dict:
        """Get relevant context for specific agent"""
        return {
            "shared_data": self.shared_context,
            "previous_agents": self.agent_outputs,
            "conversation": self.conversation_history.get_messages()
        }

def create_state_sharing_pipeline():
    """Pipeline where agents share state"""

    shared_state = SharedAgentState()

    pipeline = Pipeline()

    # Agents update and read from shared state
    agent1 = Agent(
        llm=OpenAIChatGenerator(model="gpt-4o"),
        tools=[],
        system_prompt="You find data and update shared state"
    )

    agent2 = Agent(
        llm=OpenAIChatGenerator(model="gpt-4o"),
        tools=[],
        system_prompt="You analyze data from shared state"
    )

    # Custom component to inject shared state
    @component
    class StateInjector:
        def __init__(self, shared_state: SharedAgentState, agent_name: str):
            self.shared_state = shared_state
            self.agent_name = agent_name

        @component.output_types(enhanced_input=dict)
        def run(self, input: dict) -> dict:
            context = self.shared_state.get_context_for_agent(self.agent_name)
            return {"enhanced_input": {**input, "context": context}}

    pipeline.add_component("state_injector_1", StateInjector(shared_state, "agent1"))
    pipeline.add_component("agent1", agent1)
    pipeline.add_component("state_injector_2", StateInjector(shared_state, "agent2"))
    pipeline.add_component("agent2", agent2)

    pipeline.connect("state_injector_1.enhanced_input", "agent1.input")
    pipeline.connect("agent1.output", "state_injector_2")
    pipeline.connect("state_injector_2.enhanced_input", "agent2.input")

    return pipeline
```

---

## Coordination & Orchestration

### Dynamic Agent Selection

```python
from haystack.components.routers import ConditionalRouter

def create_dynamic_routing_system():
    """System that routes to appropriate specialist based on query type"""

    pipeline = Pipeline()

    # Classifier determines query type
    @component
    class QueryClassifier:
        @component.output_types(query_type=str, original_query=str)
        def run(self, query: str) -> dict:
            # Simple keyword-based classification
            # In production, use ML classifier
            if any(word in query.lower() for word in ['code', 'implement', 'program']):
                return {"query_type": "coding", "original_query": query}
            elif any(word in query.lower() for word in ['analyze', 'data', 'statistics']):
                return {"query_type": "analysis", "original_query": query}
            elif any(word in query.lower() for word in ['research', 'find', 'search']):
                return {"query_type": "research", "original_query": query}
            else:
                return {"query_type": "general", "original_query": query}

    # Specialist agents
    coding_agent = AgentRole.create_software_engineer_agent()
    analysis_agent = AgentRole.create_data_scientist_agent()
    research_agent = AgentRole.create_research_agent()
    general_agent = Agent(llm=OpenAIChatGenerator(model="gpt-4o"), tools=[])

    # Router sends to appropriate agent
    router = ConditionalRouter(
        routes=[
            {"condition": "{query_type} == 'coding'", "output_name": "to_coder"},
            {"condition": "{query_type} == 'analysis'", "output_name": "to_analyst"},
            {"condition": "{query_type} == 'research'", "output_name": "to_researcher"},
            {"condition": "{query_type} == 'general'", "output_name": "to_general"}
        ]
    )

    pipeline.add_component("classifier", QueryClassifier())
    pipeline.add_component("router", router)
    pipeline.add_component("coder", coding_agent)
    pipeline.add_component("analyst", analysis_agent)
    pipeline.add_component("researcher", research_agent)
    pipeline.add_component("generalist", general_agent)

    # Connect routing
    pipeline.connect("classifier.query_type", "router.query_type")
    pipeline.connect("router.to_coder", "coder.query")
    pipeline.connect("router.to_analyst", "analyst.query")
    pipeline.connect("router.to_researcher", "researcher.query")
    pipeline.connect("router.to_general", "generalist.query")

    return pipeline

# Usage
router_system = create_dynamic_routing_system()
result = router_system.run({"classifier": {"query": "Implement binary search in Python"}})
# Automatically routes to coding agent
```

---

## Complete Production Multi-Agent Example

```python
from haystack import Pipeline
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.components.routers import ConditionalRouter
from haystack.tools import Tool

def create_enterprise_multi_agent_system():
    """
    Complete production multi-agent system (2025):
    - Research team finds information
    - Analysis team processes data
    - Engineering team implements solutions
    - QA team validates outputs
    - Management coordinates everything
    """

    pipeline = Pipeline()

    # Create specialized agent teams
    research_team = create_research_team()
    analysis_team = create_analysis_team()
    engineering_team = create_engineering_team()
    qa_team = create_qa_team()

    # Manager coordinates teams
    manager = Agent(
        llm=OpenAIChatGenerator(model="gpt-4o"),
        managed_agents=[research_team, analysis_team, engineering_team],
        system_prompt="""You are a project manager coordinating specialist teams.
        Break down complex tasks and delegate to appropriate teams."""
    )

    # Quality gate before final output
    quality_router = ConditionalRouter(
        routes=[
            {"condition": "{qa_score} >= 0.95", "output_name": "approved"},
            {"condition": "{qa_score} < 0.95", "output_name": "rework"}
        ]
    )

    pipeline.add_component("manager", manager)
    pipeline.add_component("qa_team", qa_team)
    pipeline.add_component("quality_gate", quality_router)

    # Connect with feedback loop
    pipeline.connect("manager.output", "qa_team.input")
    pipeline.connect("qa_team.score", "quality_gate.qa_score")
    pipeline.connect("quality_gate.rework", "manager.feedback")

    # Serialize for deployment
    pipeline.to_yaml("enterprise_multi_agent_system.yaml")

    return pipeline

# Deploy production system
production_system = create_enterprise_multi_agent_system()
result = production_system.run({
    "manager": {"query": "Build customer churn prediction system"}
})
```

---

## Summary

Haystack's 2025 multi-agent capabilities enable:

1. **Specialized Agent Teams**: Each agent excels in specific domain
2. **Flexible Coordination**: Sequential, parallel, hierarchical patterns
3. **Shared Context**: Agents communicate through shared state
4. **Dynamic Routing**: Intelligent task delegation
5. **Quality Assurance**: Built-in validation and feedback loops
6. **Production Ready**: Serializable, deployable, scalable

Build sophisticated multi-agent applications that tackle complex real-world problems through intelligent collaboration.
