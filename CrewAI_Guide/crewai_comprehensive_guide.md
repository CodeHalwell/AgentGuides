# CrewAI Comprehensive Technical Guide
## From Beginner to Expert - Role-Based Agent Collaboration

---

## Table of Contents

1. [Introduction](#introduction)
2. [Core Fundamentals](#core-fundamentals)
3. [Simple Agents](#simple-agents)
4. [Multi-Agent Systems](#multi-agent-systems)
5. [Tools Integration](#tools-integration)
6. [Structured Output](#structured-output)
7. [Memory Systems](#memory-systems)
8. [Context Engineering](#context-engineering)
9. [Task Management](#task-management)
10. [Process Types](#process-types)
11. [Crew Configuration](#crew-configuration)
12. [Agentic Patterns](#agentic-patterns)
13. [Model Context Protocol (MCP)](#model-context-protocol-mcp)

---

## Introduction

### What is CrewAI?

CrewAI is an exceptionally powerful Python framework designed for orchestrating collaborative autonomous AI agents. It enables the creation of sophisticated multi-agent systems where each agent possesses a distinct role, specialisation, and set of responsibilities. The framework facilitates seamless collaboration between agents through well-defined communication protocols, task delegation mechanisms, and intelligent workflow orchestration.

### Core Philosophy

CrewAI is fundamentally built upon the concept of **role-based agent collaboration**. This approach mirrors real-world organisational structures where individuals with specialised expertise work together to accomplish complex objectives. Each agent in a CrewAI system is assigned:

- A **specific role** defining their specialisation
- Clear **objectives and goals** guiding their actions
- A **backstory** providing context and depth
- **Available tools** enabling task execution
- **Memory systems** supporting decision-making

### Key Principles

1. **Specialisation Through Roles**: Each agent has a clearly defined role that guides its decision-making and task execution patterns.
2. **Collaboration Over Isolation**: Agents work together, sharing information and delegating tasks to achieve common objectives.
3. **Autonomous Decision-Making**: Agents make independent decisions within their domain of expertise whilst maintaining coordination with other team members.
4. **Structured Communication**: Agents communicate through well-defined interfaces and protocols.
5. **Scalable Architecture**: The framework supports scaling from simple single-agent systems to complex multi-agent hierarchies.

---

## Core Fundamentals

### Installation and Setup

#### Python Requirements

CrewAI requires Python 3.10 or later. Verify your Python version:

```bash
python --version
```

#### Basic Installation

Install the core CrewAI package:

```bash
pip install crewai
```

#### Installation with Tools

For comprehensive tool support including web scraping, file operations, and API integrations:

```bash
pip install 'crewai[tools]'
```

#### Installation with Embeddings Support

If you need embeddings functionality:

```bash
pip install 'crewai[embeddings]'
```

#### Dependency Troubleshooting

**Issue: ModuleNotFoundError: No module named 'tiktoken'**

Solution:
```bash
pip install 'crewai[embeddings]'
# or if using tools:
pip install 'crewai[tools]'
```

**Issue: Failed building wheel for tiktoken (Windows)**

Solution:
1. Install Visual C++ Build Tools for Windows
2. Ensure Rust compiler is installed
3. Upgrade pip: `pip install --upgrade pip`
4. Use pre-built wheel: `pip install tiktoken --prefer-binary`

#### Project Scaffolding

Create a new CrewAI project with automated scaffolding:

```bash
crewai create crew my_project_name
```

This generates the following directory structure:

```
my_project_name/
├── .gitignore
├── .env
├── pyproject.toml
├── README.md
└── src/
    └── my_project_name/
        ├── __init__.py
        ├── main.py
        ├── crew.py
        ├── tools/
        │   ├── __init__.py
        │   └── custom_tool.py
        └── config/
            ├── agents.yaml
            └── tasks.yaml
```

### Core Classes Overview

CrewAI's architecture revolves around four fundamental classes that interact to create autonomous agent systems:

#### 1. Agent Class

Represents an autonomous AI entity with specialised capabilities.

**Essential Attributes:**
- `role`: The agent's professional role or specialisation
- `goal`: The primary objective the agent pursues
- `backstory`: Context providing depth and expertise framing
- `tools`: List of tools available to the agent
- `llm`: Language model instance (if not provided, uses default)
- `memory`: Whether to enable memory capabilities
- `verbose`: Debug output level

**Basic Example:**

```python
from crewai import Agent, LLM

# Define a language model
llm = LLM(model="openai/gpt-4-turbo", temperature=0.7)

# Create an agent
researcher = Agent(
    role="Senior AI Research Analyst",
    goal="Conduct thorough analysis of emerging AI technologies and market trends",
    backstory="""You are an exceptionally skilled AI researcher with 15 years of 
    experience analysing emerging technologies. You have published numerous papers 
    in top-tier conferences and maintain deep expertise in machine learning, neural 
    architectures, and large language models. Your analytical approach is rigorous, 
    data-driven, and always considers multiple perspectives.""",
    llm=llm,
    verbose=True,
    memory=True
)
```

#### 2. Task Class

Defines discrete units of work assigned to specific agents.

**Essential Attributes:**
- `description`: Detailed description of what needs to be accomplished
- `expected_output`: Specification of desired output format and content
- `agent`: The agent responsible for completing the task
- `tools`: Optional task-specific tools (overrides agent tools)
- `async_execution`: Whether to execute asynchronously
- `callback`: Function to execute upon task completion

**Basic Example:**

```python
from crewai import Task

analysis_task = Task(
    description="""Analyse the latest developments in generative AI models released 
    in the past three months. Include information about model architecture innovations, 
    performance benchmarks, training methodologies, and practical applications. Focus 
    particularly on any breakthrough achievements or paradigm shifts.""",
    expected_output="""A comprehensive technical report (2000-3000 words) covering:
    1. Overview of new models released
    2. Key architectural innovations
    3. Performance comparisons with existing systems
    4. Practical applications and use cases
    5. Industry impact assessment
    6. Future research directions""",
    agent=researcher,
    async_execution=False
)
```

#### 3. Crew Class

Orchestrates teams of agents and manages their collaborative execution.

**Essential Attributes:**
- `agents`: List of agent instances in the crew
- `tasks`: List of tasks to be executed
- `process`: Execution strategy (sequential, hierarchical, etc.)
- `manager_llm`: LLM for hierarchical process manager
- `verbose`: Output verbosity level
- `memory`: Enable crew-level memory
- `max_rpm`: Maximum requests per minute (rate limiting)

**Basic Example:**

```python
from crewai import Crew, Process

crew = Crew(
    agents=[researcher],
    tasks=[analysis_task],
    process=Process.sequential,
    verbose=True,
    memory=True,
    max_rpm=10
)
```

#### 4. Process Class

Defines the execution workflow and collaboration patterns within a crew.

**Available Processes:**

1. **Sequential Process**: Tasks execute one after another, with each task's output potentially informing the next
2. **Hierarchical Process**: A manager agent coordinates task distribution and delegation
3. **Custom Process**: User-defined execution logic

**Example:**

```python
from crewai.process import Process

# Sequential execution
crew_sequential = Crew(
    agents=[researcher],
    tasks=[analysis_task],
    process=Process.sequential
)

# Hierarchical execution with manager
crew_hierarchical = Crew(
    agents=[researcher],
    tasks=[analysis_task],
    process=Process.hierarchical,
    manager_llm=LLM(model="openai/gpt-4-turbo")
)
```

### LLM Configuration

#### Supported LLM Providers

CrewAI supports numerous LLM providers through the LLM class:

**1. OpenAI**

```python
from crewai import LLM

llm = LLM(
    model="openai/gpt-4-turbo",
    api_key="your-api-key",
    temperature=0.7,
    max_tokens=4000
)
```

**2. Anthropic Claude**

```python
llm = LLM(
    model="anthropic/claude-3-opus",
    api_key="your-anthropic-api-key",
    temperature=0.5,
    max_tokens=2000
)
```

**3. Local Models (via Ollama)**

```python
llm = LLM(
    model="ollama/llama2",
    base_url="http://localhost:11434"
)
```

**4. Azure OpenAI**

```python
llm = LLM(
    model="azure/deployment-name",
    api_key="your-azure-key",
    api_version="2024-02-15-preview"
)
```

#### Environment Configuration

Store API keys securely in `.env` files:

```bash
# .env file
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_MODEL_NAME=gpt-4-turbo
```

Load environment variables in your code:

```python
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')
```

### Initialisation and First Crew

Complete minimal example creating and executing your first crew:

```python
from crewai import Agent, Task, Crew, Process, LLM
import os

# Initialise LLM
llm = LLM(model="openai/gpt-4-turbo", api_key=os.getenv('OPENAI_API_KEY'))

# Create Agent
agent = Agent(
    role="Information Analyst",
    goal="Provide accurate, well-researched information on requested topics",
    backstory="You are an expert analyst with exceptional research skills",
    llm=llm,
    verbose=True
)

# Create Task
task = Task(
    description="Research the history and evolution of artificial intelligence",
    expected_output="A comprehensive historical overview of AI development",
    agent=agent
)

# Create Crew
crew = Crew(
    agents=[agent],
    tasks=[task],
    process=Process.sequential,
    verbose=True
)

# Execute
result = crew.kickoff()
print(result)
```

---

## Simple Agents

This section explores creating and configuring individual agents, understanding agent behaviour, and managing simple single-agent systems.

### Agent Creation with Roles and Goals

#### Comprehensive Role Definition

A well-defined role guides all agent behaviour. Consider these examples across different specialisations:

**1. Research Agent**

```python
researcher = Agent(
    role="Market Research Analyst",
    goal="Identify emerging market opportunities and competitive threats",
    backstory="""You possess 12 years of market research experience across B2B and B2C sectors. 
    Your expertise spans competitive analysis, consumer behaviour research, and trend forecasting. 
    You have successfully identified three major market opportunities that led to multi-million 
    pound revenue streams. Your approach is systematic, data-driven, and always considers both 
    quantitative metrics and qualitative insights."""
)
```

**2. Writing Agent**

```python
writer = Agent(
    role="Technical Content Strategist",
    goal="Create compelling technical content that educates and engages developers",
    backstory="""You are an award-winning technical writer with a PhD in Computer Science. 
    Your articles have been published in leading tech publications and cited in academic papers. 
    You excel at breaking down complex technical concepts into digestible explanations without 
    sacrificing accuracy. Your writing style is engaging, conversational, yet maintains scientific rigor."""
)
```

**3. Analysis Agent**

```python
analyst = Agent(
    role="Data Scientist",
    goal="Transform raw data into actionable business intelligence",
    backstory="""With a background in statistical modelling and machine learning, you bring 
    rigorous analytical capabilities to every project. You have designed and deployed 
    machine learning systems that optimised operations and increased revenue by 40%. 
    Your strength lies in identifying patterns, creating predictive models, and communicating 
    findings to non-technical stakeholders."""
)
```

**4. Integration Agent**

```python
engineer = Agent(
    role="Solutions Architect",
    goal="Design scalable, maintainable system architectures",
    backstory="""You are a seasoned architect with 15 years building enterprise systems. 
    You have designed architectures serving millions of users globally. Your expertise spans 
    cloud platforms, microservices, system design patterns, and DevOps practices. You excel 
    at balancing technical requirements with business constraints and team capabilities."""
)
```

#### Goal Alignment and Clarity

Clear, measurable goals drive agent behaviour:

```python
# Vague goal (problematic)
agent = Agent(
    role="Analyst",
    goal="Analyse things"  # Too vague!
)

# Clear, specific goal (better)
agent = Agent(
    role="Financial Analyst",
    goal="Analyse quarterly financial statements to identify cost reduction opportunities",
)

# Highly specific goal (excellent)
agent = Agent(
    role="Senior Financial Analyst",
    goal="""Analyse quarterly financial statements and operational data to identify 
    and quantify cost reduction opportunities of at least £50k annually, providing 
    prioritised recommendations for implementation."""
)
```

### Backstory and Personality

The backstory is crucial for establishing agent expertise and decision-making patterns.

#### Crafting Effective Backstories

An effective backstory includes:
1. **Experience level** and years in the field
2. **Specific achievements** demonstrating expertise
3. **Methodological preferences** influencing approach
4. **Communication style** affecting interaction patterns
5. **Known expertise and limitations** managing expectations

**Example - Expert Researcher Backstory:**

```python
researcher = Agent(
    role="Academic Research Coordinator",
    goal="Synthesise complex research findings into actionable insights",
    backstory="""You hold a PhD in Computational Biology with 10 years of post-doctoral 
    research experience at leading institutions. You have published 47 peer-reviewed articles 
    and have expertise spanning genomics, bioinformatics, and systems biology. 
    
    Your approach is methodical: you prioritise peer-reviewed sources, critically evaluate 
    methodology and conclusions, and always consider alternative interpretations. You are 
    experienced in literature synthesis, systematic reviews, and meta-analysis. 
    
    When encountering conflicting research findings, you investigate the underlying methodological 
    differences before drawing conclusions. You maintain awareness of your knowledge cutoff 
    and acknowledge limitations honestly. Your communication style is precise but accessible, 
    making complex concepts understandable to audiences with varying technical backgrounds.""",
    verbose=True
)
```

**Example - Creative Content Creator Backstory:**

```python
creator = Agent(
    role="Creative Content Creator",
    goal="Produce engaging, original content that resonates with target audiences",
    backstory="""You are a seasoned content creator with 8 years of experience across multiple 
    platforms: blogs, social media, video content, and podcasts. Your content has reached millions 
    globally and consistently achieved viral engagement metrics.
    
    Your creative philosophy balances authenticity with strategic thinking. You understand narrative 
    structure, audience psychology, and platform-specific best practices. You excel at identifying 
    emerging trends early and adapting formats for maximum impact across different channels.
    
    Your process involves audience research, competitive analysis, and iterative refinement. 
    You measure success through engagement metrics, sharing, and audience feedback. 
    You're skilled at adapting tone and style to different audiences whilst maintaining 
    your unique voice and perspective.""",
    verbose=True
)
```

### Agent Configuration Options

#### Memory Configuration

Agents can maintain memory across tasks:

```python
# Disable memory (useful for stateless operations)
agent = Agent(
    role="Data Processor",
    goal="Process incoming data",
    memory=False
)

# Enable memory with default settings
agent = Agent(
    role="Conversational Assistant",
    goal="Provide consistent assistance",
    memory=True
)
```

#### Verbosity and Debugging

Control debug output levels:

```python
# Minimal output (production)
agent = Agent(
    role="Worker",
    goal="Execute tasks silently",
    verbose=False
)

# Detailed output (development/debugging)
agent = Agent(
    role="Worker",
    goal="Execute tasks with detailed logging",
    verbose=True
)
```

#### Model Configuration

Specify different models for different agents:

```python
# Fast, efficient model for routine tasks
fast_llm = LLM(model="openai/gpt-3.5-turbo", temperature=0.3)

# Powerful model for complex analysis
powerful_llm = LLM(model="openai/gpt-4-turbo", temperature=0.7)

routine_agent = Agent(
    role="Data Clerk",
    goal="Process routine data entry tasks",
    llm=fast_llm
)

analysis_agent = Agent(
    role="Strategy Advisor",
    goal="Provide strategic analysis and recommendations",
    llm=powerful_llm
)
```

#### Temperature and Creativity

Control response randomness:

```python
# Deterministic responses (good for factual tasks)
agent = Agent(
    role="Data Fact-Checker",
    goal="Verify factual accuracy",
    llm=LLM(model="openai/gpt-4-turbo", temperature=0.1)
)

# Creative responses (good for brainstorming)
agent = Agent(
    role="Brainstorm Facilitator",
    goal="Generate creative ideas and novel concepts",
    llm=LLM(model="openai/gpt-4-turbo", temperature=0.9)
)

# Balanced approach
agent = Agent(
    role="Content Analyst",
    goal="Analyse and summarise content with balanced insight",
    llm=LLM(model="openai/gpt-4-turbo", temperature=0.5)
)
```

### Single-Agent Crews

While CrewAI excels with multiple agents, single-agent crews are valuable for structured task execution.

#### Use Cases for Single-Agent Crews

1. **Task Isolation**: Running tasks without requiring other agents
2. **Tool Integration**: Leveraging CrewAI's tool integration system
3. **Structured Execution**: Utilising crew-level memory and logging
4. **Consistency**: Maintaining unified agent configuration and behaviour
5. **Scalability Path**: Starting simple, adding agents gradually

#### Complete Single-Agent Example

```python
from crewai import Agent, Task, Crew, Process, LLM

# Initialise LLM
llm = LLM(model="openai/gpt-4-turbo")

# Create single agent
data_analyst = Agent(
    role="Business Intelligence Analyst",
    goal="Transform data into strategic insights for decision-making",
    backstory="""You are an exceptional business intelligence analyst with expertise 
    in data warehousing, analytics, and business intelligence tools. You have successfully 
    guided organisations through digital transformation initiatives.""",
    llm=llm,
    verbose=True,
    memory=True
)

# Create multiple tasks for the same agent
data_collection_task = Task(
    description="Identify and compile quarterly sales data from the past three years",
    expected_output="Cleaned and organised dataset ready for analysis",
    agent=data_analyst
)

analysis_task = Task(
    description="Analyse trends, seasonality, and growth patterns in the compiled data",
    expected_output="Detailed analysis report with visualisation recommendations",
    agent=data_analyst
)

insight_task = Task(
    description="Generate actionable business insights and strategic recommendations",
    expected_output="Executive summary with top 10 recommendations prioritised by impact",
    agent=data_analyst
)

# Create crew with single agent, multiple tasks
crew = Crew(
    agents=[data_analyst],
    tasks=[data_collection_task, analysis_task, insight_task],
    process=Process.sequential,
    verbose=True,
    memory=True
)

# Execute
result = crew.kickoff()
print("Analysis Complete:")
print(result)
```

### Task Definition and Execution

#### Task Structure

A comprehensive task definition includes:

```python
task = Task(
    description="""Conduct a comprehensive audit of our current content marketing strategy, 
    including analysis of blog performance, social media engagement, email effectiveness, 
    and content distribution channels. Evaluate ROI for each channel and identify gaps.""",
    expected_output="""A detailed audit report including:
    1. Current strategy overview and history
    2. Performance metrics for each channel
    3. ROI analysis and cost-effectiveness evaluation
    4. Identified gaps and missed opportunities
    5. Competitive benchmarking
    6. Top 10 recommendations for improvement""",
    agent=content_strategist,
    async_execution=False
)
```

#### Task Outputs and Input Chaining

Tasks can use outputs from previous tasks as inputs:

```python
# Task 1: Gather information
research_task = Task(
    description="Research the latest market trends in sustainable fashion",
    expected_output="Comprehensive market trend report",
    agent=researcher
)

# Task 2: Use research output to create content
content_task = Task(
    description="""Write a blog article about sustainable fashion trends. 
    Base your article on the market research report provided. Include specific 
    statistics, expert quotes, and actionable advice for consumers.""",
    expected_output="1500-2000 word blog article with sources cited",
    agent=writer
)

# Task 3: Use content for promotion planning
promotion_task = Task(
    description="""Create a social media promotion strategy for the blog article. 
    Customise the core message for different platforms (LinkedIn, Twitter, Instagram, 
    TikTok) based on platform-specific best practices.""",
    expected_output="Platform-specific promotion strategy with sample posts",
    agent=marketing_agent
)

# Crew executes sequentially, passing outputs
crew = Crew(
    agents=[researcher, writer, marketing_agent],
    tasks=[research_task, content_task, promotion_task],
    process=Process.sequential
)
```

### Tool Assignment to Agents

#### Built-in Tools Available

CrewAI provides numerous built-in tools through the crewai-tools package:

1. **FileReadTool**: Read and process file contents
2. **FileWriteTool**: Write data to files
3. **DirectoryReadTool**: List and explore directories
4. **ScrapeWebsiteTool**: Extract content from web pages
5. **SerperDevTool**: Search the internet using Serper API
6. **GithubSearchTool**: Search GitHub repositories
7. **GmailTool**: Interact with Gmail
8. **JiraSearchTool**: Search Jira for issues and projects

#### Tool Assignment Example

```python
from crewai import Agent
from crewai_tools import ScrapeWebsiteTool, SerperDevTool

# Initialise tools
scraper = ScrapeWebsiteTool()
searcher = SerperDevTool()

# Create agent with tools
research_agent = Agent(
    role="Web Researcher",
    goal="Gather current information from the internet",
    backstory="Expert at finding and synthesising information from multiple sources",
    tools=[scraper, searcher],
    verbose=True
)
```

#### Tool Usage in Tasks

```python
task = Task(
    description="Search for recent developments in quantum computing and summarise findings",
    expected_output="Summary of top 5 quantum computing breakthroughs from 2024",
    agent=research_agent
)
```

---

## Multi-Agent Systems

### Multi-Agent Crew Assembly

#### Architecture Design

Creating effective multi-agent systems requires careful architecture design:

```python
from crewai import Agent, Task, Crew, Process, LLM

# Initialise different LLMs for different specialisations
fast_llm = LLM(model="openai/gpt-3.5-turbo", temperature=0.3)
powerful_llm = LLM(model="openai/gpt-4-turbo", temperature=0.7)

# Define specialised agents
analyst = Agent(
    role="Data Analyst",
    goal="Extract meaningful patterns from data",
    backstory="Expert data scientist with 10 years experience",
    llm=powerful_llm,
    memory=True,
    verbose=True
)

writer = Agent(
    role="Technical Writer",
    goal="Create clear, compelling technical documentation",
    backstory="Award-winning technical writer with PhD in Computer Science",
    llm=fast_llm,
    memory=True,
    verbose=True
)

designer = Agent(
    role="Visualisation Designer",
    goal="Create clear, impactful data visualisations",
    backstory="Expert designer with background in information architecture",
    llm=fast_llm,
    memory=True,
    verbose=True
)

# Create crew with multiple agents
crew = Crew(
    agents=[analyst, writer, designer],
    tasks=[],  # Tasks added separately
    process=Process.sequential,
    verbose=True
)
```

### Role Distribution and Specialisation

Effective crews distribute roles to leverage specialisation:

```python
# Financial Services Analysis Crew
finance_crew = Crew(
    agents=[
        Agent(
            role="Financial Analyst",
            goal="Analyse financial statements and provide investment insights",
            backstory="CFA with 15 years investment banking experience"
        ),
        Agent(
            role="Risk Manager",
            goal="Identify and quantify financial risks",
            backstory="Expert in risk management and compliance frameworks"
        ),
        Agent(
            role="Economist",
            goal="Provide macroeconomic context and forecasting",
            backstory="PhD Economist with central bank experience"
        ),
        Agent(
            role="Report Writer",
            goal="Synthesise findings into executive reports",
            backstory="Senior analyst skilled at communicating complex topics"
        )
    ],
    tasks=[],
    process=Process.hierarchical
)
```

### Sequential vs. Hierarchical Processes

#### Sequential Process

Tasks execute one after another, each potentially using outputs from previous tasks:

```python
crew = Crew(
    agents=[analyst, writer, designer],
    tasks=[
        data_analysis_task,
        document_writing_task,
        visualization_task
    ],
    process=Process.sequential,
    verbose=True
)

# Execution order: analysis → writing → visualisation
# Writer can reference analyst output
# Designer can reference both analysis and written output
```

#### Hierarchical Process

A manager agent coordinates task distribution and agent collaboration:

```python
from crewai import LLM

manager_llm = LLM(model="openai/gpt-4-turbo")

crew = Crew(
    agents=[analyst, writer, designer],
    tasks=[
        data_analysis_task,
        document_writing_task,
        visualization_task
    ],
    process=Process.hierarchical,
    manager_llm=manager_llm,
    verbose=True
)

# Manager decides task order and agent assignments
# Manager coordinates between agents
# Manager ensures task dependencies are respected
```

### Agent Collaboration Patterns

#### Pattern 1: Sequential Analysis and Synthesis

```python
# Agent 1 researches
research_task = Task(
    description="Research topic X thoroughly",
    agent=researcher
)

# Agent 2 synthesises
synthesis_task = Task(
    description="Create a comprehensive report based on research",
    agent=writer
)

# Agent 3 creates presentation
presentation_task = Task(
    description="Design visualisation and presentation",
    agent=designer
)
```

#### Pattern 2: Parallel Specialisation

```python
# Multiple agents work in parallel (requires async_execution)
financial_analysis = Task(
    description="Conduct financial analysis",
    agent=financial_analyst,
    async_execution=True
)

market_analysis = Task(
    description="Analyse market trends",
    agent=market_analyst,
    async_execution=True
)

operational_analysis = Task(
    description="Review operational metrics",
    agent=ops_analyst,
    async_execution=True
)

synthesis = Task(
    description="Synthesise all analyses into comprehensive report",
    agent=report_writer
)
```

#### Pattern 3: Delegation and Escalation

```python
# Junior agent attempts task
junior_task = Task(
    description="Analyse the data and provide initial insights",
    agent=junior_analyst
)

# Senior agent refines and escalates
senior_task = Task(
    description="""Review the junior analyst's work and provide senior-level insights. 
    If analysis is incomplete, conduct additional investigation.""",
    agent=senior_analyst
)

# Manager makes final decision
manager_task = Task(
    description="Review all analyses and make final recommendation",
    agent=manager
)
```

### Manager-Worker Hierarchies

```python
from crewai import LLM
from crewai.process import Process

# Define workers
researcher = Agent(role="Researcher", goal="Gather information")
analyst = Agent(role="Analyst", goal="Analyse information")
writer = Agent(role="Writer", goal="Document findings")

# Define manager
manager = Agent(
    role="Project Manager",
    goal="Coordinate team to deliver high-quality results",
    llm=LLM(model="openai/gpt-4-turbo")
)

# Create hierarchical crew
crew = Crew(
    agents=[manager, researcher, analyst, writer],
    tasks=[
        Task(description="Research topic", agent=researcher),
        Task(description="Analyse findings", agent=analyst),
        Task(description="Document results", agent=writer)
    ],
    process=Process.hierarchical,
    manager_llm=LLM(model="openai/gpt-4-turbo"),
    verbose=True
)

# Manager automatically coordinates
result = crew.kickoff()
```

---

## Tools Integration

### Built-in Tools Library

#### File Operations

```python
from crewai_tools import FileReadTool, FileWriteTool, DirectoryReadTool

# Read files
reader = FileReadTool(file_path="/path/to/file.txt")

# Write files
writer = FileWriteTool()

# Read directories
dir_reader = DirectoryReadTool(directory="/path/to/directory")

# Assign to agent
agent = Agent(
    role="File Processor",
    goal="Process and analyse files",
    tools=[reader, writer, dir_reader]
)
```

#### Web Operations

```python
from crewai_tools import ScrapeWebsiteTool, SerperDevTool

# Scrape web content
scraper = ScrapeWebsiteTool()

# Search the internet
searcher = SerperDevTool(
    n_results=10  # Number of results
)

agent = Agent(
    role="Web Researcher",
    goal="Find and synthesise web information",
    tools=[scraper, searcher]
)
```

#### API Operations

```python
from crewai_tools import GithubSearchTool, JiraSearchTool

# Search GitHub
github = GithubSearchTool()

# Search Jira
jira = JiraSearchTool(
    jira_server="https://your-jira.atlassian.net",
    username="your-email@example.com",
    api_token="your-api-token"
)

agent = Agent(
    role="Integration Specialist",
    goal="Integrate with external systems",
    tools=[github, jira]
)
```

### Custom Tool Creation with BaseTool

```python
from crewai_tools import BaseTool

class CalculationTool(BaseTool):
    name: str = "calculation_tool"
    description: str = "Performs advanced mathematical calculations"
    
    def _run(self, expression: str) -> str:
        try:
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {str(e)}"

# Create instance
calc_tool = CalculationTool()

# Assign to agent
agent = Agent(
    role="Mathematician",
    goal="Solve mathematical problems",
    tools=[calc_tool]
)
```

### Async Tool Support

```python
import asyncio
from crewai_tools import BaseTool

class AsyncAPITool(BaseTool):
    name: str = "async_api_tool"
    description: str = "Calls external APIs asynchronously"
    
    async def _arun(self, endpoint: str) -> str:
        # Simulated async API call
        await asyncio.sleep(2)
        return f"Data from {endpoint}"

# Use in async task
task = Task(
    description="Fetch data from multiple APIs",
    agent=api_agent,
    async_execution=True
)
```

---

## Structured Output

### Output with Pydantic Models

```python
from pydantic import BaseModel, Field
from typing import List

class ResearchFinding(BaseModel):
    title: str = Field(..., description="Finding title")
    description: str = Field(..., description="Detailed description")
    confidence: float = Field(..., ge=0, le=1, description="Confidence 0-1")

class ResearchReport(BaseModel):
    topic: str
    findings: List[ResearchFinding]
    summary: str

# Task with structured output
research_task = Task(
    description="Research the topic and provide structured findings",
    expected_output="Structured research report",
    agent=researcher,
    output_pydantic=ResearchReport
)
```

### JSON Output

```python
import json

# Get JSON output
task = Task(
    description="Analyse and provide JSON output",
    expected_output="JSON formatted analysis",
    agent=analyst
)

# Parse result
result = crew.kickoff()
data = json.loads(result)
```

---

## Memory Systems

### Short-Term Memory

```python
# Agent maintains context within a session
agent = Agent(
    role="Assistant",
    goal="Provide consistent assistance",
    memory=True  # Enables memory
)

# Memory is maintained during crew execution
# Useful for multi-task workflows where context matters
```

### Long-Term Memory

```python
# Enable long-term memory for persistence
agent = Agent(
    role="Assistant",
    goal="Learn from interactions",
    memory=True
)

# Memory persists across multiple crew kickoffs
# Useful for ongoing relationships with the agent
```

### Entity Memory

CrewAI tracks entities (people, places, concepts) automatically when memory is enabled.

---

## Context Engineering

### Context Passing Between Tasks

```python
# Sequential tasks naturally pass context
task1 = Task(
    description="Gather market data",
    expected_output="Market overview with key statistics",
    agent=analyst
)

task2 = Task(
    description="""Analyse the market data and provide strategic recommendations. 
    Use the market overview from the previous task.""",
    expected_output="Strategic recommendations based on analysis",
    agent=strategist
)

crew = Crew(
    agents=[analyst, strategist],
    tasks=[task1, task2],
    process=Process.sequential
)
```

### Prompt Engineering for Roles

```python
# Strong role backstory improves results
agent = Agent(
    role="AI Safety Researcher",
    goal="Ensure AI systems operate safely and ethically",
    backstory="""You are a renowned AI safety researcher with publications in top-tier 
    venues. Your expertise spans alignment problems, interpretability, robustness, and 
    societal impact. You approach every problem with rigorous methodology and consider 
    multiple perspectives. Your work has influenced policy and practice in major AI 
    organisations.""",
    verbose=True
)
```

---

## Task Management

### Task Dependencies

```python
# Explicit dependency through output reference
data_task = Task(
    description="Collect and clean data",
    expected_output="Cleaned dataset",
    agent=data_engineer
)

analysis_task = Task(
    description="""Analyse the cleaned data and identify patterns. 
    Reference the output from data collection.""",
    expected_output="Pattern analysis report",
    agent=analyst
)

# Sequential process ensures proper ordering
crew = Crew(
    agents=[data_engineer, analyst],
    tasks=[data_task, analysis_task],
    process=Process.sequential
)
```

### Async Task Execution

```python
# Multiple tasks execute in parallel
task1 = Task(
    description="Task 1",
    agent=agent1,
    async_execution=True
)

task2 = Task(
    description="Task 2",
    agent=agent2,
    async_execution=True
)

task3 = Task(
    description="Synthesise results from tasks 1 and 2",
    agent=synthesiser,
    async_execution=False
)

crew = Crew(
    agents=[agent1, agent2, synthesiser],
    tasks=[task1, task2, task3],
    process=Process.sequential
)
```

---

## Process Types

### Sequential Process (Default)

```python
# Tasks execute one after another
crew = Crew(
    agents=[agent1, agent2, agent3],
    tasks=[task1, task2, task3],
    process=Process.sequential
)

# Execution: task1 → task2 → task3
```

### Hierarchical Process

```python
# Manager coordinates tasks
crew = Crew(
    agents=[agent1, agent2, agent3],
    tasks=[task1, task2, task3],
    process=Process.hierarchical,
    manager_llm=LLM(model="openai/gpt-4-turbo")
)

# Manager decides order and delegation
```

---

## Crew Configuration

### Crew-Level Settings

```python
crew = Crew(
    agents=[agent1, agent2],
    tasks=[task1, task2],
    process=Process.sequential,
    verbose=True,  # Enable logging
    memory=True,  # Enable crew memory
    max_rpm=10,  # Rate limiting
    share_crew_state=True  # Share state between agents
)
```

### Verbose and Debugging

```python
# Development with verbose output
crew = Crew(
    agents=[agent1],
    tasks=[task1],
    verbose=True  # Shows all agent thinking and decisions
)

# Production with minimal output
crew = Crew(
    agents=[agent1],
    tasks=[task1],
    verbose=False  # Only shows final results
)
```

### Memory Settings

```python
# Enable crew-level memory
crew = Crew(
    agents=[agent1, agent2],
    tasks=[task1, task2],
    memory=True
)

# All agents share crew memory
# Information from earlier tasks is available to later tasks
```

---

## Agentic Patterns

### Research and Writing Workflow

```python
# Comprehensive research, writing, and review workflow
research_agent = Agent(
    role="Research Specialist",
    goal="Thoroughly research topics"
)

writer_agent = Agent(
    role="Content Writer",
    goal="Create engaging written content"
)

editor_agent = Agent(
    role="Editorial Director",
    goal="Ensure content quality and accuracy"
)

# Task sequence
research_task = Task(
    description="Research AI ethics comprehensively",
    expected_output="Detailed research findings with sources",
    agent=research_agent
)

writing_task = Task(
    description="Create article based on research",
    expected_output="Well-structured article ready for publication",
    agent=writer_agent
)

editorial_task = Task(
    description="Review and refine article for publication",
    expected_output="Publication-ready article",
    agent=editor_agent
)

crew = Crew(
    agents=[research_agent, writer_agent, editor_agent],
    tasks=[research_task, writing_task, editorial_task],
    process=Process.sequential,
    verbose=True
)

result = crew.kickoff()
```

---

## Model Context Protocol (MCP)

CrewAI integrates with the Model Context Protocol for enhanced tool and resource management. MCP enables:

1. **Tool Exposure**: Expose CrewAI agents' capabilities as MCP tools
2. **Resource Management**: Share resources between agents and external systems
3. **Context Sharing**: Maintain rich context across interactions

### Basic MCP Integration

```python
# MCP integration example (requires mcp-python-sdk)
from crewai import Agent

agent = Agent(
    role="MCP-Enabled Agent",
    goal="Work with external MCP services"
)

# MCP tools automatically integrated through configuration
```

This guide provides comprehensive coverage of CrewAI fundamentals and patterns. Refer to official documentation for advanced features and updates.


