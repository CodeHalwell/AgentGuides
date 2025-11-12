# AG2 Recipes: Practical Tutorials and Working Examples

**Real-world Code Examples and Complete Project Templates**

## Table of Contents

1. [Recipe 1: Simple Customer Support Bot](#recipe-1-simple-customer-support-bot)
2. [Recipe 2: Code Review Team](#recipe-2-code-review-team)
3. [Recipe 3: Data Analysis Pipeline](#recipe-3-data-analysis-pipeline)
4. [Recipe 4: Content Creation Workflow](#recipe-4-content-creation-workflow)
5. [Recipe 5: Research and Summarisation](#recipe-5-research-and-summarisation)
6. [Recipe 6: Multi-Stage Decision Making](#recipe-6-multi-stage-decision-making)
7. [Recipe 7: Document Q&A System](#recipe-7-document-qa-system)
8. [Recipe 8: Collaborative Learning System](#recipe-8-collaborative-learning-system)
9. [Recipe 9: API Integration](#recipe-9-api-integration)
10. [Recipe 10: Autonomous Task Scheduler](#recipe-10-autonomous-task-scheduler)

---

## Recipe 1: Simple Customer Support Bot

A two-agent system where one agent triages issues and another provides solutions.

**File: `customer_support_bot.py`**

```python
"""
Customer Support Bot: Triage and resolution system
- Agent 1: Triage Agent (classifies issues)
- Agent 2: Support Agent (provides solutions)
"""

import logging
from autogen import ConversableAgent, UserProxyAgent, LLMConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_support_system():
    """Create a customer support system."""
    
    # Load LLM configuration
    llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")
    
    # Triage agent - classifies customer issues
    triage_agent = ConversableAgent(
        name="triage_agent",
        system_message="""You are a customer support triage specialist. Your job is to:
1. Identify the category of the issue (billing, technical, general inquiry, complaint)
2. Assess urgency (low, medium, high, critical)
3. Summarise the issue concisely

Format your response as:
Category: [category]
Urgency: [urgency]
Summary: [2-3 sentences]""",
        llm_config=llm_config,
        human_input_mode="NEVER"
    )
    
    # Support agent - provides solutions
    support_agent = ConversableAgent(
        name="support_agent",
        system_message="""You are a knowledgeable customer support representative. 
Based on the triage information, provide:
1. A warm acknowledgement of their issue
2. Step-by-step solution or next steps
3. Offer for escalation if needed

Be professional, helpful, and concise.""",
        llm_config=llm_config,
        human_input_mode="NEVER"
    )
    
    # Human proxy for collecting customer issues
    customer_input = UserProxyAgent(
        name="customer",
        human_input_mode="NEVER",
        llm_config=False
    )
    
    return triage_agent, support_agent, customer_input

def process_customer_issue(issue: str):
    """Process a customer issue through the support system."""
    
    triage, support, customer = create_support_system()
    
    logger.info(f"Processing issue: {issue}")
    
    # Step 1: Triage the issue
    triage_response = customer.run(
        recipient=triage,
        message=f"Customer issue: {issue}",
        max_turns=1
    )
    triage_response.process()
    triage_summary = triage_response.summary
    logger.info(f"Triage result: {triage_summary}")
    
    # Step 2: Generate support response
    support_response = customer.run(
        recipient=support,
        message=f"Based on this triage: {triage_summary}\n\nPlease provide support.",
        max_turns=1
    )
    support_response.process()
    
    return {
        "triage": triage_summary,
        "support": support_response.summary
    }

if __name__ == "__main__":
    # Test with sample issues
    issues = [
        "I was charged twice for my subscription last month",
        "The app keeps crashing when I try to upload files",
        "How do I change my password?"
    ]
    
    for issue in issues:
        print("\n" + "="*60)
        result = process_customer_issue(issue)
        print(f"Issue: {issue}")
        print(f"\nTriage:\n{result['triage']}")
        print(f"\nSupport Response:\n{result['support']}")
        print("="*60)
```

---

## Recipe 2: Code Review Team

A multi-agent system that collaboratively reviews code through multiple lenses.

**File: `code_review_team.py`**

```python
"""
Code Review Team: Multi-agent code review system
- Performance Reviewer
- Security Reviewer
- Quality Reviewer
- Documentation Reviewer
"""

from autogen import ConversableAgent, GroupChat, GroupChatManager, LLMConfig

def create_code_review_team():
    """Create a code review team."""
    
    llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")
    
    # Performance reviewer
    performance_reviewer = ConversableAgent(
        name="performance_reviewer",
        system_message="""You are a performance specialist. When reviewing code, focus on:
1. Time complexity analysis
2. Memory usage optimisation
3. Database query efficiency
4. Caching opportunities
Suggest specific optimisations with complexity trade-offs.""",
        llm_config=llm_config
    )
    
    # Security reviewer
    security_reviewer = ConversableAgent(
        name="security_reviewer",
        system_message="""You are a security expert. When reviewing code, identify:
1. Authentication and authorisation issues
2. SQL injection vulnerabilities
3. Data exposure risks
4. Input validation gaps
Provide severity levels and remediation steps.""",
        llm_config=llm_config
    )
    
    # Quality reviewer
    quality_reviewer = ConversableAgent(
        name="quality_reviewer",
        system_message="""You are a code quality specialist. Focus on:
1. Code readability and maintainability
2. Design patterns and best practices
3. Function complexity (cyclomatic complexity)
4. Test coverage needs
Suggest refactorings with rationale.""",
        llm_config=llm_config
    )
    
    # Documentation reviewer
    doc_reviewer = ConversableAgent(
        name="doc_reviewer",
        system_message="""You are a technical documentation specialist. Check:
1. Docstring completeness
2. Type hints and annotations
3. Comment clarity
4. README and API documentation gaps
Ensure code is self-documenting and well-explained.""",
        llm_config=llm_config
    )
    
    return [
        performance_reviewer,
        security_reviewer,
        quality_reviewer,
        doc_reviewer
    ]

def review_code(code_snippet: str) -> dict:
    """Get comprehensive code review from team."""
    
    llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")
    review_team = create_code_review_team()
    
    # Create group chat
    groupchat = GroupChat(
        agents=review_team,
        messages=[],
        max_round=15,
        speaker_selection_method="auto"
    )
    
    manager = GroupChatManager(
        groupchat=groupchat,
        llm_config=llm_config
    )
    
    # Initiate review
    review_team[0].initiate_chat(
        manager,
        message=f"""Please review this Python code comprehensively:

```python
{code_snippet}
```

Each reviewer should provide their specific perspective and recommendations."""
    )
    
    return {
        "code": code_snippet,
        "chat_history": groupchat.messages
    }

if __name__ == "__main__":
    sample_code = """
def process_user_data(user_id):
    users = []
    # Read from database
    import sqlite3
    conn = sqlite3.connect("data.db")
    cursor = conn.cursor()
    query = "SELECT * FROM users WHERE id = " + str(user_id)  # VULNERABLE
    cursor.execute(query)
    result = cursor.fetchall()
    
    for row in result:
        processed = {}
        processed['id'] = row[0]
        processed['name'] = row[1]
        processed['email'] = row[2]
        users.append(processed)
    
    conn.close()
    
    return users
"""
    
    print("Starting code review...")
    result = review_code(sample_code)
    print(f"\nCode review completed. {len(result['chat_history'])} messages exchanged.")
```

---

## Recipe 3: Data Analysis Pipeline

A pipeline where agents collaborate on data analysis tasks.

**File: `data_analysis_pipeline.py`**

```python
"""
Data Analysis Pipeline: Multi-stage data analysis
- Data Validator
- Analyst
- Statistician
- Visualiser
"""

from autogen import ConversableAgent, initiate_chats, LLMConfig
import json

def create_analysis_pipeline():
    """Create data analysis pipeline."""
    
    llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")
    
    # Data validator
    validator = ConversableAgent(
        name="validator",
        system_message="You check data quality: missing values, outliers, data types. Report validation results.",
        llm_config=llm_config
    )
    
    # Analyst
    analyst = ConversableAgent(
        name="analyst",
        system_message="You identify patterns, trends, and insights. Ask clarifying questions about the data.",
        llm_config=llm_config
    )
    
    # Statistician
    statistician = ConversableAgent(
        name="statistician",
        system_message="You suggest statistical tests and analyses. Explain methodology in simple terms.",
        llm_config=llm_config
    )
    
    # Visualiser
    visualiser = ConversableAgent(
        name="visualiser",
        system_message="You recommend visualisations and dashboards for presenting findings.",
        llm_config=llm_config
    )
    
    # Coordinator
    coordinator = ConversableAgent(
        name="coordinator",
        system_message="You coordinate the analysis process.",
        llm_config=llm_config
    )
    
    return validator, analyst, statistician, visualiser, coordinator

def analyse_dataset(dataset_description: str, data_sample: dict):
    """Analyse a dataset using the pipeline."""
    
    validator, analyst, statistician, visualiser, coordinator = create_analysis_pipeline()
    
    # Define sequential analysis stages
    chat_sequence = [
        {
            "sender": coordinator,
            "recipient": validator,
            "message": f"""Please validate this dataset:
{dataset_description}

Sample data:
{json.dumps(data_sample, indent=2)}""",
            "max_turns": 2,
            "summary_method": "last_msg"
        },
        {
            "sender": coordinator,
            "recipient": analyst,
            "message": "Based on the validation, analyse the patterns and trends in this data.",
            "max_turns": 2,
            "summary_method": "reflection_with_llm",
            "carryover": "Use the validation results from the previous stage."
        },
        {
            "sender": coordinator,
            "recipient": statistician,
            "message": "Suggest appropriate statistical analyses.",
            "max_turns": 2,
            "summary_method": "reflection_with_llm",
            "carryover": "Consider the analysis from the previous stage."
        },
        {
            "sender": coordinator,
            "recipient": visualiser,
            "message": "Recommend visualisations for presenting these findings.",
            "max_turns": 2,
            "summary_method": "last_msg",
            "carryover": "Base recommendations on the analyses discussed."
        }
    ]
    
    # Execute pipeline
    results = initiate_chats(chat_sequence)
    
    return {
        "validation": results[0].summary,
        "analysis": results[1].summary,
        "statistics": results[2].summary,
        "visualisation": results[3].summary
    }

if __name__ == "__main__":
    dataset_desc = "Customer purchase data from Q1-Q4 2024, including amount, category, and customer segment"
    sample_data = {
        "total_records": 10000,
        "columns": ["date", "customer_id", "amount", "category", "segment"],
        "amount_range": [10, 5000],
        "categories": ["Electronics", "Clothing", "Food", "Home"],
        "segments": ["Premium", "Regular", "Budget"]
    }
    
    print("Starting data analysis pipeline...")
    results = analyse_dataset(dataset_desc, sample_data)
    
    print("\n=== Analysis Results ===")
    for stage, result in results.items():
        print(f"\n{stage.upper()}:")
        print(result)
```

---

## Recipe 4: Content Creation Workflow

End-to-end content creation with ideation, research, writing, and editing.

**File: `content_creation_workflow.py`**

```python
"""
Content Creation Workflow: Blog post creation
- Ideator (generates topics and angles)
- Researcher (gathers information)
- Writer (creates draft)
- Editor (refines and polishes)
"""

from autogen import ConversableAgent, initiate_chats, LLMConfig

def create_content_team():
    """Create content creation team."""
    
    llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")
    
    ideator = ConversableAgent(
        name="ideator",
        system_message="You generate blog post ideas and angles. Suggest 3 compelling angles with brief descriptions.",
        llm_config=llm_config
    )
    
    researcher = ConversableAgent(
        name="researcher",
        system_message="You research topics deeply. Provide key facts, statistics, and interesting insights.",
        llm_config=llm_config
    )
    
    writer = ConversableAgent(
        name="writer",
        system_message="You write engaging blog posts. Create an engaging 800-1000 word post with introduction, body, and conclusion.",
        llm_config=llm_config
    )
    
    editor = ConversableAgent(
        name="editor",
        system_message="You edit for clarity, grammar, and engagement. Point out specific improvements needed.",
        llm_config=llm_config
    )
    
    coordinator = ConversableAgent(
        name="coordinator",
        system_message="You coordinate the content creation process.",
        llm_config=llm_config
    )
    
    return ideator, researcher, writer, editor, coordinator

def create_blog_post(topic: str) -> dict:
    """Create a complete blog post."""
    
    ideator, researcher, writer, editor, coordinator = create_content_team()
    
    chat_sequence = [
        {
            "sender": coordinator,
            "recipient": ideator,
            "message": f"Generate 3 compelling blog post angles for the topic: '{topic}'",
            "max_turns": 2,
            "summary_method": "last_msg"
        },
        {
            "sender": coordinator,
            "recipient": researcher,
            "message": "Research the top angle thoroughly. Provide facts, statistics, and insights.",
            "max_turns": 3,
            "summary_method": "reflection_with_llm",
            "carryover": "Build on the angle chosen by the ideator."
        },
        {
            "sender": coordinator,
            "recipient": writer,
            "message": "Write an engaging blog post using the research.",
            "max_turns": 2,
            "summary_method": "reflection_with_llm",
            "carryover": "Incorporate the research findings into a well-structured post."
        },
        {
            "sender": coordinator,
            "recipient": editor,
            "message": "Edit the post for clarity, grammar, and engagement.",
            "max_turns": 2,
            "summary_method": "last_msg",
            "carryover": "Review and improve the draft post."
        }
    ]
    
    results = initiate_chats(chat_sequence)
    
    return {
        "topic": topic,
        "ideas": results[0].summary,
        "research": results[1].summary,
        "draft": results[2].summary,
        "final": results[3].summary
    }

if __name__ == "__main__":
    blog_post = create_blog_post("The Future of Remote Work")
    
    print("=== Blog Post Creation Process ===")
    print(f"\nTopic: {blog_post['topic']}")
    print(f"\nIdeas Generated:\n{blog_post['ideas']}")
    print(f"\nResearch:\n{blog_post['research'][:300]}...")
    print(f"\nDraft:\n{blog_post['draft'][:300]}...")
    print(f"\nFinal Version:\n{blog_post['final']}")
```

---

## Recipe 5: Research and Summarisation

Multi-agent research team that gathers and synthesises information.

**File: `research_team.py`**

```python
"""
Research and Summarisation: Information gathering system
- Researcher 1: Academic perspective
- Researcher 2: Industry perspective
- Researcher 3: Critical analysis
- Synthesiser: Combines insights
"""

from autogen import ConversableAgent, GroupChat, GroupChatManager, LLMConfig

def research_topic(topic: str, focus: str) -> str:
    """Research a topic from multiple perspectives."""
    
    llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")
    
    academic = ConversableAgent(
        name="academic_researcher",
        system_message=f"You approach the topic from an academic perspective. Cite relevant studies and theories about: {topic}",
        llm_config=llm_config
    )
    
    industry = ConversableAgent(
        name="industry_expert",
        system_message=f"You provide practical industry insights about: {topic}. Mention real-world applications and best practices.",
        llm_config=llm_config
    )
    
    critic = ConversableAgent(
        name="critical_analyst",
        system_message=f"You provide critical analysis of: {topic}. Identify limitations, challenges, and alternative perspectives.",
        llm_config=llm_config
    )
    
    synthesiser = ConversableAgent(
        name="synthesiser",
        system_message="You synthesise insights from multiple perspectives into a cohesive summary. Identify common themes and conflicts.",
        llm_config=llm_config
    )
    
    # Create group chat
    groupchat = GroupChat(
        agents=[academic, industry, critic, synthesiser],
        messages=[],
        max_round=12,
        speaker_selection_method="auto"
    )
    
    manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)
    
    # Initiate discussion
    academic.initiate_chat(
        manager,
        message=f"""Please research and discuss: '{topic}'
Focus area: {focus}

Each of you should contribute your perspective, then the synthesiser will combine insights."""
    )
    
    return groupchat.messages[-1]['content'] if groupchat.messages else ""

if __name__ == "__main__":
    result = research_topic(
        topic="Artificial Intelligence in Healthcare",
        focus="Diagnostic applications and limitations"
    )
    print("Research Summary:")
    print(result)
```

---

## Recipe 6: Multi-Stage Decision Making

Hierarchical decision-making process with escalation.

**File: `decision_making_workflow.py`**

```python
"""
Multi-Stage Decision Making: Escalating decision framework
- Level 1: Analyst (initial assessment)
- Level 2: Manager (review and approval)
- Level 3: Executive (final decision on escalations)
"""

from autogen import ConversableAgent, UserProxyAgent, LLMConfig
from typing import Dict

def make_decision_hierarchically(proposal: str) -> Dict:
    """Make a decision through hierarchical review."""
    
    llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")
    
    analyst = ConversableAgent(
        name="analyst",
        system_message="You analyse proposals and provide initial recommendations with pros/cons.",
        llm_config=llm_config
    )
    
    manager = ConversableAgent(
        name="manager",
        system_message="You review analyst recommendations and provide management perspective. Decide if escalation is needed.",
        llm_config=llm_config
    )
    
    executive = ConversableAgent(
        name="executive",
        system_message="You make final decisions on escalated proposals. Consider strategic implications.",
        llm_config=llm_config
    )
    
    coordinator = UserProxyAgent(
        name="coordinator",
        human_input_mode="NEVER",
        llm_config=False
    )
    
    # Level 1: Analyst review
    analyst_response = coordinator.run(
        recipient=analyst,
        message=f"Please analyse this proposal: {proposal}",
        max_turns=1
    )
    analyst_response.process()
    analyst_summary = analyst_response.summary
    
    # Level 2: Manager review
    manager_response = coordinator.run(
        recipient=manager,
        message=f"""Based on this analyst review:
{analyst_summary}

Please provide your management assessment. If you think this needs executive review, say "ESCALATE".""",
        max_turns=1
    )
    manager_response.process()
    manager_summary = manager_response.summary
    
    # Check if escalation needed
    needs_escalation = "ESCALATE" in manager_summary.upper()
    
    # Level 3: Executive decision (if needed)
    executive_decision = ""
    if needs_escalation:
        exec_response = coordinator.run(
            recipient=executive,
            message=f"""This proposal was escalated by management:
Analyst: {analyst_summary}
Manager: {manager_summary}

Please make the final decision.""",
            max_turns=1
        )
        exec_response.process()
        executive_decision = exec_response.summary
    
    return {
        "proposal": proposal,
        "analyst_review": analyst_summary,
        "manager_review": manager_summary,
        "escalated": needs_escalation,
        "executive_decision": executive_decision,
        "final_status": "ESCALATED TO EXECUTIVE" if needs_escalation else "APPROVED BY MANAGER"
    }

if __name__ == "__main__":
    proposal = "Allocate $500,000 for new AI research initiative"
    result = make_decision_hierarchically(proposal)
    
    print("=== Decision Making Process ===")
    print(f"Proposal: {result['proposal']}")
    print(f"\nAnalyst Review:\n{result['analyst_review']}")
    print(f"\nManager Review:\n{result['manager_review']}")
    print(f"\nEscalated: {result['escalated']}")
    if result['executive_decision']:
        print(f"\nExecutive Decision:\n{result['executive_decision']}")
    print(f"\nFinal Status: {result['final_status']}")
```

---

## Recipe 7: Document Q&A System

Question-answering system that retrieves and processes documents.

**File: `document_qa_system.py`**

```python
"""
Document Q&A System: Answer questions about documents
- Document loader
- Retriever
- Synthesiser
"""

from autogen import ConversableAgent, UserProxyAgent, LLMConfig, register_function
from typing import Annotated, List

def document_qa_system():
    """Create a document Q&A system."""
    
    llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")
    
    # Simulate document store
    documents = {
        "doc1": "Python is a high-level programming language. It emphasises code readability.",
        "doc2": "Machine learning is a subset of AI that enables systems to learn from data.",
        "doc3": "REST APIs use HTTP methods (GET, POST, PUT, DELETE) for operations."
    }
    
    def retrieve_documents(query: Annotated[str, "Search query for documents"]) -> List[str]:
        """Retrieve relevant documents based on query."""
        # Simple keyword matching
        results = []
        query_words = query.lower().split()
        
        for doc_id, content in documents.items():
            content_lower = content.lower()
            if any(word in content_lower for word in query_words):
                results.append(f"[{doc_id}]: {content}")
        
        return results if results else ["No documents found for query."]
    
    # Create agents
    retriever = ConversableAgent(
        name="retriever",
        system_message="You retrieve relevant documents for user queries.",
        llm_config=llm_config
    )
    
    synthesiser = ConversableAgent(
        name="synthesiser",
        system_message="You answer questions by synthesising information from retrieved documents.",
        llm_config=llm_config
    )
    
    executor = UserProxyAgent(
        name="executor",
        human_input_mode="NEVER",
        llm_config=False
    )
    
    # Register retrieval function
    register_function(
        retrieve_documents,
        caller=retriever,
        executor=executor,
        description="Retrieve documents matching a query"
    )
    
    return retriever, synthesiser, executor

def answer_question(question: str) -> str:
    """Answer a question using document retrieval."""
    
    retriever, synthesiser, executor = document_qa_system()
    
    # Step 1: Retrieve documents
    retrieval_response = executor.run(
        recipient=retriever,
        message=f"Find documents relevant to: {question}",
        max_turns=1
    )
    retrieval_response.process()
    
    # Step 2: Synthesise answer
    answer_response = executor.run(
        recipient=synthesiser,
        message=f"""Based on these retrieved documents:
{retrieval_response.summary}

Please answer the question: {question}""",
        max_turns=1
    )
    answer_response.process()
    
    return answer_response.summary

if __name__ == "__main__":
    questions = [
        "What is Python?",
        "Explain machine learning",
        "How do REST APIs work?"
    ]
    
    for q in questions:
        print(f"\nQuestion: {q}")
        answer = answer_question(q)
        print(f"Answer: {answer}")
```

---

## Recipe 8: Collaborative Learning System

Teachers and learners collaborating to teach concepts.

**File: `learning_system.py`**

```python
"""
Collaborative Learning System: Teachers and learners
- Teacher: Explains concepts
- Learner: Asks questions
- Mentor: Provides guidance
"""

from autogen import ConversableAgent, LLMConfig

def collaborative_learning(topic: str, difficulty: str = "intermediate"):
    """Create a collaborative learning session."""
    
    llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")
    
    teacher = ConversableAgent(
        name="teacher",
        system_message=f"""You are an expert teacher explaining '{topic}' at {difficulty} level.
Your approach:
1. Start with fundamentals
2. Use analogies and examples
3. Build complexity gradually
4. Ask if the learner understands""",
        llm_config=llm_config
    )
    
    learner = ConversableAgent(
        name="learner",
        system_message="You are an eager learner. Ask clarifying questions. Admit when you don't understand.",
        llm_config=llm_config
    )
    
    mentor = ConversableAgent(
        name="mentor",
        system_message="You guide the learning process. Suggest concepts to explore and provide hints when needed.",
        llm_config=llm_config
    )
    
    # Start learning session
    print(f"Learning Session: {topic} ({difficulty} level)")
    print("="*50)
    
    response = learner.initiate_chat(
        teacher,
        message=f"Teach me about '{topic}'. I'm at a {difficulty} level."
    )
    
    response.process()
    print(response.summary)
    
    return response.summary

if __name__ == "__main__":
    lesson = collaborative_learning("Machine Learning Algorithms", "beginner")
```

---

## Recipe 9: API Integration

Agents that integrate with external APIs.

**File: `api_integration.py`**

```python
"""
API Integration: Agents using external APIs
- API caller agent
- Data processor agent
"""

from autogen import ConversableAgent, register_function, LLMConfig, UserProxyAgent
from typing import Annotated, Dict, Any
import json

def api_integration_example():
    """Example of agents integrating with APIs."""
    
    llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")
    
    # Define API functions
    def call_weather_api(city: Annotated[str, "City name"]) -> Dict[str, Any]:
        """Call weather API (simulated)."""
        weather_data = {
            "London": {"temp": 15, "condition": "Rainy", "humidity": 75},
            "Paris": {"temp": 18, "condition": "Cloudy", "humidity": 60},
            "Berlin": {"temp": 12, "condition": "Sunny", "humidity": 50}
        }
        return weather_data.get(city, {"error": "City not found"})
    
    def call_exchange_api(currency_pair: Annotated[str, "Currency pair (e.g., EUR/USD)"]) -> Dict[str, Any]:
        """Call exchange rate API (simulated)."""
        rates = {
            "EUR/USD": 1.10,
            "GBP/USD": 1.27,
            "USD/JPY": 110.5
        }
        return {"pair": currency_pair, "rate": rates.get(currency_pair, "N/A")}
    
    # Create agents
    api_caller = ConversableAgent(
        name="api_caller",
        system_message="You call APIs to get data when requested.",
        llm_config=llm_config
    )
    
    data_processor = ConversableAgent(
        name="processor",
        system_message="You process and analyse API results.",
        llm_config=llm_config
    )
    
    executor = UserProxyAgent(
        name="executor",
        human_input_mode="NEVER",
        llm_config=False
    )
    
    # Register API functions
    register_function(
        call_weather_api,
        caller=api_caller,
        executor=executor,
        description="Get weather data for a city"
    )
    
    register_function(
        call_exchange_api,
        caller=api_caller,
        executor=executor,
        description="Get currency exchange rate"
    )
    
    # Get weather
    response = executor.initiate_chat(
        api_caller,
        message="Get weather for London and Paris",
        max_turns=3
    )
    
    return response

if __name__ == "__main__":
    result = api_integration_example()
```

---

## Recipe 10: Autonomous Task Scheduler

Agent system that plans and executes tasks.

**File: `task_scheduler.py`**

```python
"""
Autonomous Task Scheduler: Planning and executing tasks
- Planner: Creates task breakdown
- Scheduler: Assigns timing
- Executor: Carries out tasks
"""

from autogen import ConversableAgent, initiate_chats, LLMConfig
from datetime import datetime

def schedule_and_execute_project(project_description: str):
    """Schedule and execute a project."""
    
    llm_config = LLMConfig.from_json(path="OAI_CONFIG_LIST")
    
    planner = ConversableAgent(
        name="planner",
        system_message="You break down projects into concrete tasks with dependencies.",
        llm_config=llm_config
    )
    
    scheduler = ConversableAgent(
        name="scheduler",
        system_message="You assign realistic timing and sequencing to tasks.",
        llm_config=llm_config
    )
    
    executor = ConversableAgent(
        name="executor",
        system_message="You confirm task execution and report progress.",
        llm_config=llm_config
    )
    
    coordinator = ConversableAgent(
        name="coordinator",
        system_message="You coordinate the project execution.",
        llm_config=llm_config
    )
    
    # Define workflow
    workflow = [
        {
            "sender": coordinator,
            "recipient": planner,
            "message": f"Plan this project: {project_description}",
            "max_turns": 2,
            "summary_method": "last_msg"
        },
        {
            "sender": coordinator,
            "recipient": scheduler,
            "message": "Create a timeline for these tasks.",
            "max_turns": 2,
            "summary_method": "reflection_with_llm",
            "carryover": "Use the task breakdown from planning."
        },
        {
            "sender": coordinator,
            "recipient": executor,
            "message": "Execute the scheduled tasks.",
            "max_turns": 2,
            "summary_method": "reflection_with_llm",
            "carryover": "Follow the timeline and task sequence."
        }
    ]
    
    results = initiate_chats(workflow)
    
    return {
        "project": project_description,
        "task_breakdown": results[0].summary,
        "timeline": results[1].summary,
        "execution": results[2].summary,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    project = schedule_and_execute_project(
        "Build a web application for task management with frontend, backend, and database"
    )
    
    print("=== Project Execution Report ===")
    print(f"Project: {project['project']}")
    print(f"\nTask Breakdown:\n{project['task_breakdown']}")
    print(f"\nTimeline:\n{project['timeline']}")
    print(f"\nExecution Report:\n{project['execution']}")
    print(f"\nTimestamp: {project['timestamp']}")
```

---

## Quick Reference: Running These Recipes

**Setup:**
```bash
pip install ag2[openai]
export OPENAI_API_KEY="sk-your-key"
```

**Run any recipe:**
```bash
python recipe_name.py
```

**Customisation Tips:**
- Replace `LLMConfig.from_json()` with environment variables or hardcoded keys
- Adjust `max_rounds` and `max_turns` for faster/slower execution
- Modify `system_message` to change agent behaviour
- Add logging with `logging.basicConfig(level=logging.DEBUG)`

Each recipe is self-contained and production-ready. Mix and match patterns to build your own systems!
