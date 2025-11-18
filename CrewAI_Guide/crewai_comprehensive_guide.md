Latest: 1.4.1 (2025 Edition)
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
14. [**NEW: CrewAI Flows (2025)**](#crewai-flows-2025)
15. [**NEW: CrewAI AMP Suite**](#crewai-amp-suite)
16. [**NEW: UV Dependency Management**](#uv-dependency-management)

---

## Introduction

### What is CrewAI?

CrewAI is an exceptionally powerful Python framework designed for orchestrating collaborative autonomous AI agents. It enables the creation of sophisticated multi-agent systems where each agent possesses a distinct role, specialisation, and set of responsibilities. The framework facilitates seamless collaboration between agents through well-defined communication protocols, task delegation mechanisms, and intelligent workflow orchestration.

**2025 Update**: CrewAI has evolved into the industry-leading framework for agentic AI automation, trusted by over **100,000+ certified developers** worldwide. With the introduction of **CrewAI Flows** (event-driven workflows), the **CrewAI AMP Suite** (enterprise-grade automation), and **UV dependency management** (streamlined setup), CrewAI now provides a complete ecosystem for building production-ready agentic applications at any scale.

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

#### **NEW (2025): UV Dependency Management**

CrewAI now supports UV, the ultra-fast Python package installer and resolver, providing a significantly improved setup experience:

```bash
# Install UV (if not already installed)
pip install uv

# Create and activate project with UV
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install CrewAI with UV (10-100x faster than pip)
uv pip install crewai
```

**Benefits of UV**:
- **Blazing Fast**: 10-100x faster than traditional pip
- **Better Dependency Resolution**: Resolves complex dependency trees efficiently
- **Smaller Environments**: More compact virtual environments
- **Drop-in Replacement**: Compatible with existing pip workflows

#### Installation with Tools

For comprehensive tool support including web scraping, file operations, and API integrations:

```bash
pip install 'crewai[tools]'
# Or with UV:
uv pip install 'crewai[tools]'
```

#### Installation with Embeddings Support

If you need embeddings functionality:

```bash
pip install 'crewai[embeddings]'
# Or with UV:
uv pip install 'crewai[embeddings]'
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

---

## CrewAI Flows (2025)

### Introduction to Flows

**CrewAI Flows** is the revolutionary 2025 feature that extends CrewAI's capabilities beyond traditional agent orchestration into event-driven workflow automation. Flows enable you to build sophisticated applications that combine:

- **Event-Driven Execution**: Trigger workflows based on events and conditions
- **Conditional Logic**: Implement branching logic and decision trees
- **Loops and Iterations**: Process data collections and repeating patterns
- **Real-Time State Management**: Maintain and update application state dynamically
- **Hybrid Orchestration**: Seamlessly combine Python code, LLM calls, and crew executions

### Why Use Flows?

Traditional crew execution is linear and task-based. Flows add:

1. **Control Flow**: Conditional branching, loops, and complex decision logic
2. **State Persistence**: Maintain state across multiple crew executions
3. **Event Handling**: Respond to external events and triggers
4. **Modularity**: Break complex workflows into reusable flow components
5. **Observability**: Built-in monitoring and state inspection

### Core Flow Concepts

#### 1. Flow Class

The base class for creating event-driven workflows:

```python
from crewai.flow.flow import Flow, listen, start

class ContentCreationFlow(Flow):
    """Flow for automated content creation pipeline"""

    @start()
    def gather_requirements(self):
        """Entry point - gather content requirements"""
        print("Gathering requirements...")
        return {
            "topic": "AI Agents",
            "target_audience": "Developers",
            "word_count": 2000
        }

    @listen(gather_requirements)
    def research_topic(self, requirements):
        """Research the topic based on requirements"""
        print(f"Researching: {requirements['topic']}")
        # Research logic here
        return {"research_data": "..."}

    @listen(research_topic)
    def create_content(self, research_data):
        """Generate content from research"""
        print("Creating content...")
        # Content creation logic
        return {"content": "..."}

# Execute the flow
flow = ContentCreationFlow()
result = flow.kickoff()
```

#### 2. Flow Decorators

**@start()**: Defines the entry point of a flow

```python
@start()
def initialize(self):
    return {"status": "initialized"}
```

**@listen()**: Creates event listeners that trigger when specific methods complete

```python
@listen(initialize)
def process_data(self, init_data):
    # Processes after initialize completes
    return processed_data
```

**@router()**: Implements conditional routing based on conditions

```python
from crewai.flow.flow import router

@router(process_data)
def route_based_on_type(self, data):
    if data['type'] == 'urgent':
        return 'handle_urgent'
    elif data['type'] == 'normal':
        return 'handle_normal'
    else:
        return 'handle_default'
```

### Flows with Crews

Integrate crews into flows for powerful agentic workflows:

```python
from crewai import Agent, Task, Crew, LLM
from crewai.flow.flow import Flow, listen, start

class ResearchAndWriteFlow(Flow):
    """Flow combining crews for research and writing"""

    def __init__(self):
        super().__init__()
        self.llm = LLM(model="openai/gpt-4-turbo")

    @start()
    def define_topic(self):
        """Define research topic"""
        return {"topic": "Quantum Computing Applications"}

    @listen(define_topic)
    def research_crew_execution(self, topic_data):
        """Execute research crew"""
        # Create research agent
        researcher = Agent(
            role="AI Research Analyst",
            goal=f"Research {topic_data['topic']} comprehensively",
            backstory="Expert researcher with deep technical knowledge",
            llm=self.llm
        )

        # Create research task
        research_task = Task(
            description=f"Research {topic_data['topic']} and compile findings",
            expected_output="Comprehensive research report with key findings",
            agent=researcher
        )

        # Execute crew
        crew = Crew(
            agents=[researcher],
            tasks=[research_task],
            verbose=True
        )

        result = crew.kickoff()
        return {"research": result}

    @listen(research_crew_execution)
    def writing_crew_execution(self, research_data):
        """Execute writing crew"""
        writer = Agent(
            role="Technical Writer",
            goal="Create engaging technical content",
            backstory="Expert at translating research into accessible content",
            llm=self.llm
        )

        writing_task = Task(
            description=f"Write article based on research: {research_data['research']}",
            expected_output="Well-structured technical article",
            agent=writer
        )

        crew = Crew(
            agents=[writer],
            tasks=[writing_task],
            verbose=True
        )

        result = crew.kickoff()
        return {"article": result}

# Execute the flow
flow = ResearchAndWriteFlow()
final_result = flow.kickoff()
print(final_result)
```

### Conditional Logic in Flows

Implement branching logic using routers:

```python
from crewai.flow.flow import Flow, listen, start, router

class DataProcessingFlow(Flow):
    """Flow with conditional routing"""

    @start()
    def receive_data(self):
        return {
            "data_type": "customer_feedback",
            "sentiment": "negative",
            "priority": "high"
        }

    @router(receive_data)
    def route_by_priority(self, data):
        """Route based on priority level"""
        if data['priority'] == 'high':
            return 'handle_urgent'
        elif data['priority'] == 'medium':
            return 'handle_normal'
        else:
            return 'handle_low_priority'

    @listen('handle_urgent')
    def handle_urgent(self, data):
        print(f"URGENT: Processing {data['data_type']}")
        # Escalate and process immediately
        return {"status": "escalated"}

    @listen('handle_normal')
    def handle_normal(self, data):
        print(f"NORMAL: Processing {data['data_type']}")
        # Standard processing
        return {"status": "processed"}

    @listen('handle_low_priority')
    def handle_low_priority(self, data):
        print(f"LOW: Queuing {data['data_type']}")
        # Queue for later
        return {"status": "queued"}

flow = DataProcessingFlow()
flow.kickoff()
```

### Loops and Iterations

Process collections and implement iterative workflows:

```python
from crewai.flow.flow import Flow, listen, start

class BatchProcessingFlow(Flow):
    """Process multiple items iteratively"""

    @start()
    def get_items(self):
        """Get items to process"""
        return {
            "items": [
                {"id": 1, "content": "Item 1"},
                {"id": 2, "content": "Item 2"},
                {"id": 3, "content": "Item 3"}
            ]
        }

    @listen(get_items)
    def process_items(self, data):
        """Process each item"""
        results = []

        for item in data['items']:
            # Process each item
            processed = self.process_single_item(item)
            results.append(processed)

        return {"results": results}

    def process_single_item(self, item):
        """Process a single item"""
        print(f"Processing item {item['id']}: {item['content']}")
        # Processing logic here
        return {
            "id": item['id'],
            "status": "completed",
            "output": f"Processed: {item['content']}"
        }

    @listen(process_items)
    def summarize_results(self, results):
        """Summarize processing results"""
        total = len(results['results'])
        print(f"Processed {total} items")
        return {"summary": f"Completed {total} items"}

flow = BatchProcessingFlow()
flow.kickoff()
```

### Real-Time State Management

Maintain and update state throughout flow execution:

```python
from crewai.flow.flow import Flow, listen, start

class StatefulFlow(Flow):
    """Flow with persistent state"""

    def __init__(self):
        super().__init__()
        # Initialize state
        self.state = {
            "processed_count": 0,
            "errors": [],
            "results": []
        }

    @start()
    def begin_processing(self):
        print("Starting processing...")
        return {"status": "started"}

    @listen(begin_processing)
    def process_batch_1(self, data):
        """Process first batch"""
        try:
            # Processing logic
            self.state['processed_count'] += 10
            self.state['results'].append("Batch 1 complete")
        except Exception as e:
            self.state['errors'].append(str(e))

        return {"batch": 1, "state": self.state}

    @listen(process_batch_1)
    def process_batch_2(self, data):
        """Process second batch"""
        try:
            self.state['processed_count'] += 15
            self.state['results'].append("Batch 2 complete")
        except Exception as e:
            self.state['errors'].append(str(e))

        return {"batch": 2, "state": self.state}

    @listen(process_batch_2)
    def finalize(self, data):
        """Finalize and report state"""
        print(f"\nFinal State:")
        print(f"Processed: {self.state['processed_count']} items")
        print(f"Results: {self.state['results']}")
        print(f"Errors: {self.state['errors']}")

        return {"final_state": self.state}

flow = StatefulFlow()
flow.kickoff()
```

### Advanced Flow Patterns

#### Pattern 1: Multi-Stage Pipeline

```python
from crewai.flow.flow import Flow, listen, start
from crewai import Agent, Task, Crew, LLM

class ContentPipelineFlow(Flow):
    """Multi-stage content production pipeline"""

    def __init__(self):
        super().__init__()
        self.llm = LLM(model="openai/gpt-4-turbo")

    @start()
    def ideation(self):
        """Stage 1: Generate ideas"""
        ideator = Agent(
            role="Content Strategist",
            goal="Generate compelling content ideas",
            llm=self.llm
        )

        task = Task(
            description="Generate 5 article ideas about AI agents",
            expected_output="List of 5 article ideas with descriptions",
            agent=ideator
        )

        crew = Crew(agents=[ideator], tasks=[task])
        result = crew.kickoff()
        return {"ideas": result}

    @listen(ideation)
    def research(self, ideas_data):
        """Stage 2: Research selected ideas"""
        researcher = Agent(
            role="Research Analyst",
            goal="Conduct thorough research",
            llm=self.llm
        )

        task = Task(
            description=f"Research these ideas: {ideas_data['ideas']}",
            expected_output="Detailed research findings",
            agent=researcher
        )

        crew = Crew(agents=[researcher], tasks=[task])
        result = crew.kickoff()
        return {"research": result}

    @listen(research)
    def write(self, research_data):
        """Stage 3: Write content"""
        writer = Agent(
            role="Technical Writer",
            goal="Create engaging content",
            llm=self.llm
        )

        task = Task(
            description=f"Write article based on: {research_data['research']}",
            expected_output="Complete article draft",
            agent=writer
        )

        crew = Crew(agents=[writer], tasks=[task])
        result = crew.kickoff()
        return {"draft": result}

    @listen(write)
    def edit(self, draft_data):
        """Stage 4: Edit and refine"""
        editor = Agent(
            role="Editor",
            goal="Polish content to publication quality",
            llm=self.llm
        )

        task = Task(
            description=f"Edit and refine: {draft_data['draft']}",
            expected_output="Publication-ready article",
            agent=editor
        )

        crew = Crew(agents=[editor], tasks=[task])
        result = crew.kickoff()
        return {"final_article": result}

flow = ContentPipelineFlow()
final_output = flow.kickoff()
```

#### Pattern 2: Error Handling and Retry Logic

```python
from crewai.flow.flow import Flow, listen, start, router

class ResilientFlow(Flow):
    """Flow with error handling and retries"""

    def __init__(self):
        super().__init__()
        self.max_retries = 3
        self.retry_count = 0

    @start()
    def process_data(self):
        return {"data": "sensitive_operation"}

    @listen(process_data)
    def risky_operation(self, data):
        """Operation that might fail"""
        try:
            # Simulated risky operation
            import random
            if random.random() < 0.5:
                raise Exception("Operation failed")

            return {"status": "success", "result": data}
        except Exception as e:
            return {"status": "error", "error": str(e), "retry": True}

    @router(risky_operation)
    def handle_result(self, result):
        """Route based on operation result"""
        if result['status'] == 'success':
            return 'finalize_success'
        elif result.get('retry') and self.retry_count < self.max_retries:
            self.retry_count += 1
            print(f"Retrying... (attempt {self.retry_count})")
            return 'risky_operation'
        else:
            return 'handle_failure'

    @listen('finalize_success')
    def finalize_success(self, result):
        print("Operation completed successfully")
        return result

    @listen('handle_failure')
    def handle_failure(self, result):
        print(f"Operation failed after {self.max_retries} retries")
        return {"status": "failed", "error": result.get('error')}

flow = ResilientFlow()
flow.kickoff()
```

### Flow Execution and Monitoring

```python
from crewai.flow.flow import Flow, listen, start

class MonitoredFlow(Flow):
    """Flow with execution monitoring"""

    @start()
    def step_1(self):
        print("Step 1: Started")
        # Logic here
        print("Step 1: Completed")
        return {"step": 1}

    @listen(step_1)
    def step_2(self, data):
        print("Step 2: Started")
        # Logic here
        print("Step 2: Completed")
        return {"step": 2}

    @listen(step_2)
    def step_3(self, data):
        print("Step 3: Started")
        # Logic here
        print("Step 3: Completed")
        return {"step": 3, "final": True}

# Execute and monitor
flow = MonitoredFlow()
result = flow.kickoff()

# Access flow state
print(f"\nFlow execution completed")
print(f"Final result: {result}")
```

### Best Practices for Flows

1. **Clear State Management**: Keep state explicit and well-documented
2. **Modular Design**: Break complex flows into smaller, reusable components
3. **Error Handling**: Implement proper error handling and recovery mechanisms
4. **Logging**: Add comprehensive logging for debugging and monitoring
5. **Testing**: Test flows with various scenarios and edge cases
6. **Documentation**: Document flow logic, state transitions, and decision points

**For comprehensive Flow examples and patterns, see the dedicated [CrewAI Flows Guide](./crewai_flows_guide.md).**

---

## CrewAI AMP Suite

### Introduction to CrewAI AMP

**CrewAI AMP (Agent Management Platform)** is the enterprise-grade suite for secure, scalable agentic automation. Released in 2025, the AMP Suite transforms CrewAI from a development framework into a complete enterprise platform.

### What is CrewAI AMP?

The AMP Suite provides:

1. **Enterprise Security**: Advanced authentication, authorization, and audit logging
2. **Scalable Infrastructure**: Production-ready deployment and scaling capabilities
3. **Monitoring and Observability**: Real-time monitoring, metrics, and dashboards
4. **Team Collaboration**: Multi-user support with role-based access control
5. **Deployment Tools**: One-click deployment to cloud platforms
6. **SLA Guarantees**: Enterprise support with SLAs and uptime guarantees

### Key Features

#### 1. Enterprise Authentication and Authorization

```python
from crewai.amp import AMPCrew, AuthConfig

# Configure authentication
auth_config = AuthConfig(
    provider="oauth2",
    client_id="your-client-id",
    allowed_domains=["yourcompany.com"],
    role_based_access=True
)

# Create AMP-enabled crew
amp_crew = AMPCrew(
    name="enterprise_crew",
    auth_config=auth_config,
    agents=[agent1, agent2],
    tasks=[task1, task2]
)
```

#### 2. Scalable Deployment

```python
from crewai.amp import DeploymentConfig

deployment = DeploymentConfig(
    platform="kubernetes",
    replicas=3,
    auto_scaling=True,
    min_instances=2,
    max_instances=10,
    cpu_threshold=70,
    memory_threshold=80
)

amp_crew.deploy(deployment)
```

#### 3. Real-Time Monitoring

```python
from crewai.amp import MonitoringConfig

monitoring = MonitoringConfig(
    enable_metrics=True,
    enable_logging=True,
    log_level="INFO",
    metrics_endpoint="/metrics",
    dashboard_enabled=True
)

amp_crew.configure_monitoring(monitoring)

# Access metrics
metrics = amp_crew.get_metrics()
print(f"Total executions: {metrics['total_executions']}")
print(f"Success rate: {metrics['success_rate']}%")
print(f"Average duration: {metrics['avg_duration']}s")
```

#### 4. Audit Logging

```python
from crewai.amp import AuditConfig

audit = AuditConfig(
    enable_audit_log=True,
    log_all_actions=True,
    retention_days=90,
    compliance_mode="SOC2"
)

amp_crew.configure_audit(audit)

# Query audit logs
logs = amp_crew.get_audit_logs(
    start_date="2025-01-01",
    end_date="2025-01-31",
    user="john@company.com"
)
```

#### 5. Team Collaboration

```python
from crewai.amp import TeamConfig, Role

# Define roles
admin_role = Role(
    name="admin",
    permissions=["create", "read", "update", "delete", "deploy"]
)

developer_role = Role(
    name="developer",
    permissions=["create", "read", "update"]
)

viewer_role = Role(
    name="viewer",
    permissions=["read"]
)

# Configure team
team = TeamConfig(
    roles=[admin_role, developer_role, viewer_role],
    members=[
        {"email": "admin@company.com", "role": "admin"},
        {"email": "dev@company.com", "role": "developer"},
        {"email": "viewer@company.com", "role": "viewer"}
    ]
)

amp_crew.configure_team(team)
```

### AMP Pricing and Plans

- **Developer**: Free for development and testing
- **Team**: $99/month - Up to 10 users, basic monitoring
- **Business**: $499/month - Unlimited users, advanced features, 99.9% SLA
- **Enterprise**: Custom pricing - Dedicated support, custom SLAs, on-premise options

For more information, visit: https://www.crewai.com/amp

---

## UV Dependency Management

### Why UV?

UV is the next-generation Python package installer developed by Astral (creators of Ruff). It offers:

- **Speed**: 10-100x faster than pip
- **Reliability**: Better dependency resolution
- **Compatibility**: Drop-in replacement for pip
- **Modern Design**: Built in Rust for performance

### Installation with UV

```bash
# Install UV
pip install uv

# Or using standalone installer
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Creating CrewAI Projects with UV

```bash
# Create new project directory
mkdir my_crewai_project
cd my_crewai_project

# Create virtual environment with UV
uv venv

# Activate environment
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows

# Install CrewAI with UV (blazing fast!)
uv pip install crewai 'crewai[tools]'

# Install specific versions
uv pip install crewai==1.4.1

# Install from requirements file
uv pip install -r requirements.txt
```

### UV with CrewAI Development

```bash
# Create project structure
crewai create crew my_project

# Navigate to project
cd my_project

# Create UV environment
uv venv

# Activate
source .venv/bin/activate

# Install dependencies with UV
uv pip install -r requirements.txt

# Add additional packages
uv pip install pytest black ruff
```

### UV Advantages for CrewAI Projects

1. **Faster Installation**: Large projects install in seconds vs minutes
2. **Better Caching**: Intelligent caching reduces re-downloads
3. **Lock Files**: `uv.lock` files ensure reproducible builds
4. **Parallel Downloads**: Multiple packages download simultaneously
5. **Smart Resolution**: Advanced dependency conflict resolution

### Migrating Existing Projects to UV

```bash
# Navigate to existing CrewAI project
cd existing_project

# Create UV virtual environment
uv venv

# Activate environment
source .venv/bin/activate

# Install from existing requirements.txt
uv pip install -r requirements.txt

# Or install from pyproject.toml
uv pip install -e .

# Freeze dependencies to UV format
uv pip freeze > requirements.txt
```

### UV Commands Reference

```bash
# Install package
uv pip install package_name

# Install from requirements
uv pip install -r requirements.txt

# Uninstall package
uv pip uninstall package_name

# List installed packages
uv pip list

# Show package details
uv pip show package_name

# Create requirements file
uv pip freeze > requirements.txt

# Upgrade package
uv pip install --upgrade package_name

# Install specific version
uv pip install package_name==1.2.3
```

---

This comprehensive guide now includes all 2025 features including CrewAI Flows, the AMP Suite, and UV dependency management. For detailed Flows examples and patterns, refer to the dedicated [CrewAI Flows Guide](./crewai_flows_guide.md).

