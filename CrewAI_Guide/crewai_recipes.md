# CrewAI Recipes
## Practical Implementation Examples and Real-World Solutions

---

## Table of Contents

1. [Research and Content Creation](#research-and-content-creation)
2. [Data Analysis and Reporting](#data-analysis-and-reporting)
3. [Customer Service and Support](#customer-service-and-support)
4. [Software Development Assistance](#software-development-assistance)
5. [Business Intelligence](#business-intelligence)
6. [Marketing Campaign Planning](#marketing-campaign-planning)
7. [Financial Analysis](#financial-analysis)
8. [Legal Document Review](#legal-document-review)
9. [Scientific Research](#scientific-research)
10. [Project Management](#project-management)

---

## Recipe 1: Research and Content Creation

### Complete Blog Post Generation System

```python
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from pydantic import BaseModel, Field
from typing import List

# Define output schema
class BlogArticle(BaseModel):
    title: str = Field(..., description="Article title")
    introduction: str = Field(..., description="Compelling introduction")
    sections: List[dict] = Field(..., description="Article sections with headers and content")
    conclusion: str = Field(..., description="Conclusion summarising key points")
    call_to_action: str = Field(..., description="Clear CTA for readers")
    seo_keywords: List[str] = Field(..., description="Relevant keywords")

# Initialise LLM
llm = LLM(model="openai/gpt-4-turbo", temperature=0.7)

# Create agents
researcher = Agent(
    role="Senior Research Specialist",
    goal="Conduct comprehensive research on topics using current sources",
    backstory="""You are an exceptional researcher with 12 years of experience 
    gathering and synthesising information from multiple authoritative sources. 
    You excel at identifying credible sources, extracting key insights, and 
    providing well-sourced, factually accurate information. Your research 
    methodology is rigorous and your findings are always properly cited.""",
    tools=[ScrapeWebsiteTool(), SerperDevTool()],
    llm=llm,
    verbose=True
)

writer = Agent(
    role="Award-Winning Content Writer",
    goal="Create engaging, SEO-optimised articles that resonate with readers",
    backstory="""You are a celebrated content writer with numerous published 
    articles in major publications. Your writing style is engaging, informative, 
    and optimised for search engines. You excel at structuring content for 
    readability, crafting compelling headlines, and including relevant calls 
    to action. Your articles consistently achieve high engagement rates.""",
    llm=llm,
    verbose=True
)

editor = Agent(
    role="Editorial Director",
    goal="Ensure content quality, accuracy, and brand alignment",
    backstory="""As an editorial director with 15 years of experience, you ensure 
    all published content meets highest quality standards. You verify factual 
    accuracy, improve clarity, enhance SEO optimisation, and guarantee brand 
    consistency. Your editorial eye has improved countless publications.""",
    llm=llm,
    verbose=True
)

# Define tasks
research_task = Task(
    description="""Research the topic: {topic}
    
    Provide:
    1. Current trends and developments
    2. Expert opinions and perspectives
    3. Real-world examples and case studies
    4. Statistical data and findings
    5. Industry best practices
    
    Focus on finding credible, recent sources.""",
    expected_output="Comprehensive research findings with citations and key insights",
    agent=researcher
)

writing_task = Task(
    description="""Create a blog article based on the research findings.
    
    Requirements:
    1. Engaging title with primary keyword
    2. Compelling introduction (100-150 words)
    3. 4-5 well-structured sections with subheadings
    4. Real-world examples and case studies
    5. Actionable advice for readers
    6. Strong conclusion
    7. Clear call to action
    
    Target length: 2000-2500 words
    Tone: Professional but conversational""",
    expected_output="Well-structured blog article ready for editing",
    agent=writer
)

editorial_task = Task(
    description="""Review and optimise the article for publication.
    
    Tasks:
    1. Verify all facts and citations
    2. Improve clarity and readability
    3. Enhance SEO optimisation
    4. Ensure consistent brand voice
    5. Check grammar and formatting
    6. Suggest headline variations
    
    Provide feedback and final recommendation.""",
    expected_output="Editorial notes and publication recommendation",
    agent=editor,
    output_pydantic=BlogArticle
)

# Create and execute crew
crew = Crew(
    agents=[researcher, writer, editor],
    tasks=[research_task, writing_task, editorial_task],
    process=Process.sequential,
    verbose=True,
    memory=True
)

def generate_blog_post(topic: str) -> dict:
    """Generate complete blog post."""
    result = crew.kickoff(inputs={"topic": topic})
    return result

# Usage
if __name__ == "__main__":
    blog = generate_blog_post("Artificial Intelligence in Healthcare")
    print(blog)
```

---

## Recipe 2: Data Analysis and Reporting

### Comprehensive Business Analytics System

```python
from crewai import Agent, Task, Crew, Process, LLM
from pydantic import BaseModel, Field
from typing import List, Dict
import json

# Output schemas
class DataInsight(BaseModel):
    area: str
    finding: str
    importance: str  # high, medium, low
    supporting_data: str

class AnalyticsReport(BaseModel):
    executive_summary: str
    key_insights: List[DataInsight]
    trends: List[str]
    recommendations: List[str]
    risk_areas: List[str]
    opportunity_areas: List[str]

# Create specialised agents
data_engineer = Agent(
    role="Data Engineer",
    goal="Clean, transform, and prepare data for analysis",
    backstory="Expert data engineer with 10 years experience in data pipeline development"
)

analyst = Agent(
    role="Business Analyst",
    goal="Analyse data and extract meaningful business insights",
    backstory="Senior analyst with expertise in business intelligence and KPI interpretation"
)

data_scientist = Agent(
    role="Data Scientist",
    goal="Conduct advanced statistical analysis and identify patterns",
    backstory="PhD Data Scientist with specialisation in machine learning and statistical modelling"
)

strategist = Agent(
    role="Strategy Consultant",
    goal="Translate insights into actionable business recommendations",
    backstory="Senior consultant with 15 years advising C-level executives"
)

# Define workflow
llm = LLM(model="openai/gpt-4-turbo")

data_prep_task = Task(
    description="Prepare data for analysis: clean, validate, and transform dataset",
    expected_output="Cleaned and validated dataset ready for analysis",
    agent=data_engineer
)

analysis_task = Task(
    description="Conduct thorough business analysis: trends, patterns, KPIs",
    expected_output="Detailed analytical findings with supporting metrics",
    agent=analyst
)

modeling_task = Task(
    description="Perform advanced statistical analysis and identify predictive patterns",
    expected_output="Statistical findings, correlations, and predictive insights",
    agent=data_scientist
)

strategy_task = Task(
    description="Synthesise all analyses into strategic recommendations",
    expected_output="Executive summary with actionable recommendations",
    agent=strategist,
    output_pydantic=AnalyticsReport
)

# Create crew
analytics_crew = Crew(
    agents=[data_engineer, analyst, data_scientist, strategist],
    tasks=[data_prep_task, analysis_task, modeling_task, strategy_task],
    process=Process.sequential,
    verbose=True,
    memory=True,
    max_rpm=20
)

def generate_analytics_report(data_path: str) -> AnalyticsReport:
    """Generate comprehensive analytics report."""
    result = analytics_crew.kickoff(inputs={"data_path": data_path})
    return result
```

---

## Recipe 3: Customer Service and Support

### Multi-Tier Support System

```python
from crewai import Agent, Task, Crew, Process, LLM

# Tier 1: Initial support
tier1_agent = Agent(
    role="Support Specialist",
    goal="Handle common customer queries and provide immediate solutions",
    backstory="Friendly support specialist with comprehensive product knowledge",
    tools=[],
    verbose=False
)

# Tier 2: Technical support
tier2_agent = Agent(
    role="Technical Support Engineer",
    goal="Resolve technical issues with detailed troubleshooting",
    backstory="Senior technical engineer with 8 years troubleshooting experience",
    verbose=False
)

# Tier 3: Escalation
tier3_agent = Agent(
    role="Support Manager",
    goal="Handle escalated complex cases and ensure resolution",
    backstory="Manager with authority to make decisions and implement solutions",
    verbose=False
)

# Support workflow
def handle_support_ticket(issue_description: str) -> dict:
    """Route and handle customer support ticket."""
    
    # Initial assessment task
    assessment_task = Task(
        description=f"Assess customer issue: {issue_description}",
        expected_output="Assessment and initial solution if applicable",
        agent=tier1_agent
    )
    
    # Technical resolution task
    resolution_task = Task(
        description="Perform detailed troubleshooting and provide comprehensive resolution",
        expected_output="Detailed resolution steps and preventive measures",
        agent=tier2_agent
    )
    
    # Escalation task
    escalation_task = Task(
        description="Review issue and make management decisions if needed",
        expected_output="Final resolution or escalation plan",
        agent=tier3_agent
    )
    
    crew = Crew(
        agents=[tier1_agent, tier2_agent, tier3_agent],
        tasks=[assessment_task, resolution_task, escalation_task],
        process=Process.hierarchical,
        verbose=False
    )
    
    return crew.kickoff(inputs={"issue": issue_description})

# Usage
ticket_result = handle_support_ticket(
    "My software keeps crashing when I try to upload large files"
)
```

---

## Recipe 4: Software Development Assistance

### Code Review and Documentation System

```python
from crewai import Agent, Task, Crew, Process, LLM

code_reviewer = Agent(
    role="Code Reviewer",
    goal="Conduct thorough code reviews focusing on quality and best practices",
    backstory="Senior developer with 12 years experience and expertise in clean code principles"
)

security_expert = Agent(
    role="Security Specialist",
    goal="Identify security vulnerabilities and recommend fixes",
    backstory="Cybersecurity expert with specialisation in application security"
)

documentation_writer = Agent(
    role="Technical Documentation Writer",
    goal="Create comprehensive API documentation",
    backstory="Expert technical writer who makes complex code understandable"
)

# Code review workflow
def review_code(code_snippet: str) -> dict:
    """Comprehensive code review."""
    
    review_task = Task(
        description=f"""Review this code for:
        1. Code quality and best practices
        2. Performance optimisations
        3. Readability and maintainability
        4. Potential improvements
        
        Code:
        {code_snippet}""",
        expected_output="Detailed code review with improvement suggestions",
        agent=code_reviewer
    )
    
    security_task = Task(
        description="Identify security vulnerabilities and risks",
        expected_output="Security assessment and recommendations",
        agent=security_expert
    )
    
    docs_task = Task(
        description="Create comprehensive documentation for the code",
        expected_output="Well-structured code documentation with examples",
        agent=documentation_writer
    )
    
    crew = Crew(
        agents=[code_reviewer, security_expert, documentation_writer],
        tasks=[review_task, security_task, docs_task],
        process=Process.sequential,
        verbose=True
    )
    
    return crew.kickoff(inputs={"code": code_snippet})
```

---

## Recipe 5: Business Intelligence

### Competitive Analysis System

```python
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import ScrapeWebsiteTool, SerperDevTool

market_researcher = Agent(
    role="Market Researcher",
    goal="Analyse competitor offerings and market positioning",
    backstory="Experienced market researcher with 10 years analysing competitive landscapes",
    tools=[ScrapeWebsiteTool(), SerperDevTool()]
)

financial_analyst = Agent(
    role="Financial Analyst",
    goal="Analyse competitor financial performance",
    backstory="CFA with expertise in financial statement analysis"
)

strategy_consultant = Agent(
    role="Strategy Consultant",
    goal="Develop competitive strategy recommendations",
    backstory="Senior consultant who has advised Fortune 500 companies"
)

def competitive_analysis(company_name: str, competitors: list) -> dict:
    """Analyse company position versus competitors."""
    
    market_task = Task(
        description=f"""Analyse {company_name} market positioning against competitors: {', '.join(competitors)}
        
        Research:
        1. Product/service offerings
        2. Target markets
        3. Marketing strategies
        4. Unique value propositions
        5. Market positioning""",
        expected_output="Comprehensive market analysis",
        agent=market_researcher
    )
    
    financial_task = Task(
        description="Analyse financial performance and metrics",
        expected_output="Financial comparison and analysis",
        agent=financial_analyst
    )
    
    strategy_task = Task(
        description="Develop competitive strategy recommendations",
        expected_output="Strategic recommendations for competitive advantage",
        agent=strategy_consultant
    )
    
    crew = Crew(
        agents=[market_researcher, financial_analyst, strategy_consultant],
        tasks=[market_task, financial_task, strategy_task],
        process=Process.sequential,
        verbose=True
    )
    
    return crew.kickoff(inputs={
        "company": company_name,
        "competitors": competitors
    })

# Usage
analysis = competitive_analysis(
    "TechCorp",
    ["CompetitorA", "CompetitorB", "CompetitorC"]
)
```

---

## Recipe 6: Marketing Campaign Planning

### Full Campaign Development System

```python
from crewai import Agent, Task, Crew, Process, LLM

marketing_strategist = Agent(
    role="Marketing Strategist",
    goal="Develop comprehensive marketing strategies",
    backstory="Senior marketing strategist with 15 years experience launching successful campaigns"
)

content_creator = Agent(
    role="Content Creator",
    goal="Create compelling marketing content",
    backstory="Award-winning content creator with expertise in multiple formats"
)

social_media_manager = Agent(
    role="Social Media Manager",
    goal="Plan and optimise social media strategy",
    backstory="Expert in social media marketing with proven track record of viral campaigns"
)

analytics_specialist = Agent(
    role="Marketing Analytics Specialist",
    goal="Define metrics and measurement strategy",
    backstory="Data analyst specialising in marketing ROI and performance measurement"
)

def plan_marketing_campaign(product: str, target_audience: str, budget: str) -> dict:
    """Plan complete marketing campaign."""
    
    strategy_task = Task(
        description=f"""Develop marketing strategy for {product} targeting {target_audience} with budget {budget}
        
        Include:
        1. Target audience analysis
        2. Key messaging
        3. Channel strategy
        4. Timeline
        5. Budget allocation""",
        expected_output="Comprehensive marketing strategy",
        agent=marketing_strategist
    )
    
    content_task = Task(
        description="Create marketing content for all channels",
        expected_output="Content calendar with copy for all channels",
        agent=content_creator
    )
    
    social_task = Task(
        description="Plan social media strategy and content",
        expected_output="Social media strategy with posting schedule",
        agent=social_media_manager
    )
    
    analytics_task = Task(
        description="Define success metrics and measurement strategy",
        expected_output="KPI framework and measurement plan",
        agent=analytics_specialist
    )
    
    crew = Crew(
        agents=[marketing_strategist, content_creator, social_media_manager, analytics_specialist],
        tasks=[strategy_task, content_task, social_task, analytics_task],
        process=Process.sequential,
        verbose=True
    )
    
    return crew.kickoff(inputs={
        "product": product,
        "audience": target_audience,
        "budget": budget
    })
```

---

## Recipe 7: Financial Analysis

### Investment Analysis System

```python
from crewai import Agent, Task, Crew, Process, LLM

fundamental_analyst = Agent(
    role="Fundamental Analyst",
    goal="Analyse company fundamentals and financial health",
    backstory="CFA with 12 years analysing financial statements"
)

technical_analyst = Agent(
    role="Technical Analyst",
    goal="Analyse price trends and technical indicators",
    backstory="Chartered technical analyst with expertise in market patterns"
)

risk_manager = Agent(
    role="Risk Manager",
    goal="Identify and quantify risks",
    backstory="Senior risk manager with expertise in portfolio risk management"
)

investment_advisor = Agent(
    role="Investment Advisor",
    goal="Provide investment recommendations",
    backstory="Senior advisor with fiduciary responsibility and proven track record"
)

def investment_analysis(company_ticker: str) -> dict:
    """Comprehensive investment analysis."""
    
    fundamental_task = Task(
        description=f"Analyse fundamentals of {company_ticker}",
        expected_output="Fundamental analysis report",
        agent=fundamental_analyst
    )
    
    technical_task = Task(
        description="Analyse technical trends and indicators",
        expected_output="Technical analysis report",
        agent=technical_analyst
    )
    
    risk_task = Task(
        description="Identify and quantify investment risks",
        expected_output="Risk assessment report",
        agent=risk_manager
    )
    
    recommendation_task = Task(
        description="Provide investment recommendation based on all analyses",
        expected_output="Investment recommendation with rationale",
        agent=investment_advisor
    )
    
    crew = Crew(
        agents=[fundamental_analyst, technical_analyst, risk_manager, investment_advisor],
        tasks=[fundamental_task, technical_task, risk_task, recommendation_task],
        process=Process.sequential,
        verbose=True
    )
    
    return crew.kickoff(inputs={"ticker": company_ticker})

# Usage
analysis = investment_analysis("AAPL")
```

---

## Recipe 8: Legal Document Review

### Contract Analysis System

```python
from crewai import Agent, Task, Crew, Process, LLM
from crewai_tools import FileReadTool

contract_specialist = Agent(
    role="Contract Specialist",
    goal="Review contracts for completeness and fairness",
    backstory="Senior legal counsel with 15 years contract law experience"
)

compliance_expert = Agent(
    role="Compliance Expert",
    goal="Ensure compliance with relevant regulations",
    backstory="Compliance specialist with expertise in regulatory requirements"
)

risk_attorney = Agent(
    role="Risk Attorney",
    goal="Identify legal risks and liabilities",
    backstory="Experienced attorney focusing on risk mitigation"
)

def review_contract(contract_path: str) -> dict:
    """Comprehensive contract review."""
    
    review_task = Task(
        description=f"""Review contract at {contract_path} for:
        1. Terms and conditions appropriateness
        2. Missing important clauses
        3. Favourable vs unfavourable terms
        4. Negotiation opportunities""",
        expected_output="Detailed contract review",
        agent=contract_specialist,
        tools=[FileReadTool(file_path=contract_path)]
    )
    
    compliance_task = Task(
        description="Verify compliance with applicable laws and regulations",
        expected_output="Compliance assessment",
        agent=compliance_expert
    )
    
    risk_task = Task(
        description="Identify legal risks and liability concerns",
        expected_output="Risk assessment and mitigation strategies",
        agent=risk_attorney
    )
    
    crew = Crew(
        agents=[contract_specialist, compliance_expert, risk_attorney],
        tasks=[review_task, compliance_task, risk_task],
        process=Process.sequential,
        verbose=True
    )
    
    return crew.kickoff(inputs={"contract_path": contract_path})
```

---

## Recipe 9: Scientific Research

### Literature Review System

```python
from crewai import Agent, Task, Crew, Process, LLM

literature_researcher = Agent(
    role="Literature Researcher",
    goal="Conduct comprehensive literature searches",
    backstory="PhD researcher with expertise in academic research methodology"
)

methodologist = Agent(
    role="Research Methodologist",
    goal="Analyse research methodologies and quality",
    backstory="Experienced methodologist evaluating research quality"
)

synthesiser = Agent(
    role="Research Synthesiser",
    goal="Synthesise findings across multiple studies",
    backstory="Expert at identifying patterns and synthesising complex information"
)

def literature_review(topic: str, scope: str) -> dict:
    """Comprehensive literature review."""
    
    search_task = Task(
        description=f"Search literature on {topic} within {scope} scope",
        expected_output="Comprehensive literature findings",
        agent=literature_researcher
    )
    
    methodology_task = Task(
        description="Analyse research methodologies and quality across studies",
        expected_output="Methodology assessment",
        agent=methodologist
    )
    
    synthesis_task = Task(
        description="Synthesise findings and identify research gaps",
        expected_output="Synthesised literature review with research gaps identified",
        agent=synthesiser
    )
    
    crew = Crew(
        agents=[literature_researcher, methodologist, synthesiser],
        tasks=[search_task, methodology_task, synthesis_task],
        process=Process.sequential,
        verbose=True
    )
    
    return crew.kickoff(inputs={"topic": topic, "scope": scope})
```

---

## Recipe 10: Project Management

### Project Planning and Oversight System

```python
from crewai import Agent, Task, Crew, Process, LLM

project_manager = Agent(
    role="Project Manager",
    goal="Plan and oversee project delivery",
    backstory="PMP certified project manager with 10 years experience"
)

resource_planner = Agent(
    role="Resource Planner",
    goal="Optimise resource allocation",
    backstory="Expert in resource planning and capacity management"
)

risk_manager = Agent(
    role="Risk Manager",
    goal="Identify and mitigate project risks",
    backstory="Senior risk management professional"
)

quality_manager = Agent(
    role="Quality Assurance Manager",
    goal="Ensure project quality standards",
    backstory="Quality management expert with certification"
)

def plan_project(project_name: str, objectives: str, timeline: str, budget: str) -> dict:
    """Comprehensive project planning."""
    
    planning_task = Task(
        description=f"""Plan project: {project_name}
        Objectives: {objectives}
        Timeline: {timeline}
        Budget: {budget}""",
        expected_output="Detailed project plan with milestones",
        agent=project_manager
    )
    
    resource_task = Task(
        description="Develop resource allocation and capacity plan",
        expected_output="Resource plan and staffing requirements",
        agent=resource_planner
    )
    
    risk_task = Task(
        description="Identify and assess project risks",
        expected_output="Risk register with mitigation strategies",
        agent=risk_manager
    )
    
    quality_task = Task(
        description="Define quality standards and assurance plan",
        expected_output="Quality management plan",
        agent=quality_manager
    )
    
    crew = Crew(
        agents=[project_manager, resource_planner, risk_manager, quality_manager],
        tasks=[planning_task, resource_task, risk_task, quality_task],
        process=Process.sequential,
        verbose=True
    )
    
    return crew.kickoff(inputs={
        "project": project_name,
        "objectives": objectives,
        "timeline": timeline,
        "budget": budget
    })

# Usage
plan = plan_project(
    "Mobile App Development",
    "Develop iOS and Android native apps",
    "6 months",
    "$500,000"
)
```

---

These recipes demonstrate CrewAI's versatility across diverse industries and use cases. Each recipe can be customised based on specific requirements and extended with additional agents, tools, and tasks as needed.


