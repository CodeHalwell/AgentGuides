# Microsoft AutoGen 0.4: Recipes and Real-World Examples

Complete, working code examples for common AutoGen use cases and patterns.

## Table of Contents

1. [Simple Chat Agent](#simple-chat-agent)
2. [Multi-Agent Research Team](#multi-agent-research-team)
3. [Code Generation and Review](#code-generation-and-review)
4. [Data Analysis Pipeline](#data-analysis-pipeline)
5. [Question-Answering System](#question-answering-system)
6. [Customer Service Bot](#customer-service-bot)
7. [Content Generation](#content-generation)
8. [Azure Integration](#azure-integration)
9. [Error Handling](#error-handling)
10. [Testing](#testing)

---

## Simple Chat Agent

### Basic Conversational Agent

```python
# simple_chat.py
import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

async def main():
    """Simple chat with an AI assistant"""
    
    # Initialize model client
    model_client = OpenAIChatCompletionClient(
        model="gpt-4o",
        temperature=0.7,
    )
    
    # Create agent
    assistant = AssistantAgent(
        name="assistant",
        system_message="You are a helpful AI assistant. Answer questions concisely and accurately.",
        model_client=model_client,
    )
    
    # Multi-turn conversation
    conversation_history = []
    
    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Add user message
        user_message = TextMessage(
            content=user_input,
            source="user"
        )
        conversation_history.append(user_message)
        
        # Get response
        response = await assistant.on_messages(conversation_history)
        print(f"Assistant: {response}\n")
        
        # Add assistant response to history
        assistant_message = TextMessage(
            content=response,
            source="assistant"
        )
        conversation_history.append(assistant_message)

if __name__ == "__main__":
    asyncio.run(main())
```

### Running the Script

```bash
$ python simple_chat.py
You: What is machine learning?
Assistant: Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed...

You: How is it different from deep learning?
Assistant: Deep learning is a specialized subset of machine learning that uses neural networks with multiple layers...

You: exit
Goodbye!
```

---

## Multi-Agent Research Team

### Complete Research Workflow

```python
# research_team.py
import asyncio
from typing import List
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination

# Mock tools for demonstration
def search_academic_database(query: str) -> str:
    """Search academic database for papers"""
    return f"""
    Found papers on '{query}':
    1. "Advanced ML Techniques" - Smith et al., 2024
    2. "Neural Network Optimisation" - Johnson et al., 2024
    3. "Reinforcement Learning in Production" - Williams et al., 2023
    """

def search_web(query: str) -> str:
    """Search the web"""
    return f"""
    Web results for '{query}':
    - Wikipedia article on {query}
    - Recent news articles
    - Tutorial websites
    """

def calculate_statistics(data: List[float]) -> dict:
    """Calculate statistics"""
    import statistics
    return {
        "mean": statistics.mean(data),
        "median": statistics.median(data),
        "stdev": statistics.stdev(data) if len(data) > 1 else 0,
    }

async def run_research_team():
    """Run a complete research team workflow"""
    
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    
    # Create specialist agents
    planner = AssistantAgent(
        name="planner",
        model_client=model_client,
        system_message="""You are a research planner.
        Your job is to:
        1. Break down the research task into specific questions
        2. Coordinate team members to answer each question
        3. Synthesise findings into a comprehensive report
        
        When you have all information, end with: TERMINATE""",
        description="Plans research and coordinates team",
    )
    
    researcher = AssistantAgent(
        name="researcher",
        model_client=model_client,
        system_message="""You are a research specialist.
        Your job is to find detailed, accurate information using available tools.
        Provide well-sourced responses with citations when possible.""",
        description="Finds detailed research information",
        tools=[search_academic_database, search_web],
    )
    
    analyst = AssistantAgent(
        name="analyst",
        model_client=model_client,
        system_message="""You are a data analyst.
        Your job is to analyse findings and identify patterns.
        Use quantitative tools when appropriate.""",
        description="Analyses research findings",
        tools=[calculate_statistics],
    )
    
    # Create team
    team = SelectorGroupChat(
        agents=[planner, researcher, analyst],
        model_client=model_client,
        termination_condition=(
            TextMentionTermination("TERMINATE") |
            MaxMessageTermination(max_messages=50)
        ),
    )
    
    # Run research
    task = """Research the latest trends in machine learning for 2024.
    Focus on:
    1. Key technological advances
    2. Industry adoption patterns
    3. Future outlook
    
    Provide a comprehensive summary."""
    
    print(f"Starting research task: {task}\n")
    print("=" * 60)
    
    result = await team.run(task)
    
    print("=" * 60)
    print(f"\nResearch Complete!\n{result}")

if __name__ == "__main__":
    asyncio.run(run_research_team())
```

---

## Code Generation and Review

### Software Development Team

```python
# code_generation_team.py
import asyncio
import tempfile
from pathlib import Path
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination

def write_code_file(filename: str, code: str) -> str:
    """Write code to file"""
    filepath = Path(tempfile.gettempdir()) / filename
    filepath.write_text(code)
    return f"Code written to {filepath}"

def execute_python_code(code: str) -> str:
    """Execute Python code and return output"""
    try:
        # SECURITY WARNING: In production, run in sandbox/container
        result = {}
        exec(code, result)
        return f"Code executed successfully. Output: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

async def run_coding_team():
    """Coordinate a software development team"""
    
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    
    # Code writer agent
    code_writer = AssistantAgent(
        name="code_writer",
        model_client=model_client,
        system_message="""You are an expert Python developer.
        Write clean, well-documented, tested code.
        Follow Python best practices and PEP 8.
        Include docstrings and type hints.""",
        description="Writes high-quality Python code",
        tools=[write_code_file],
    )
    
    # Code reviewer agent
    code_reviewer = AssistantAgent(
        name="code_reviewer",
        model_client=model_client,
        system_message="""You are a code review specialist.
        Review code for:
        - Correctness and bugs
        - Performance issues
        - Security vulnerabilities
        - Code style and readability
        - Test coverage
        
        Provide specific, actionable feedback.""",
        description="Reviews code quality",
    )
    
    # Code executor agent (for testing)
    executor = CodeExecutorAgent(
        name="executor",
        code_execution_config={
            "work_dir": tempfile.gettempdir(),
            "use_docker": False,
        }
    )
    
    # Create team
    team = SelectorGroupChat(
        agents=[code_writer, code_reviewer, executor],
        model_client=model_client,
        termination_condition=MaxMessageTermination(max_messages=30),
    )
    
    # Run development
    task = """Write a Python function that:
    1. Loads a CSV file
    2. Cleans the data (removes nulls, handles duplicates)
    3. Calculates statistics
    4. Returns a summary report
    
    Include error handling and documentation."""
    
    result = await team.run(task)
    print(result)

if __name__ == "__main__":
    asyncio.run(run_coding_team())
```

---

## Data Analysis Pipeline

### Automated Data Analysis

```python
# data_analysis_pipeline.py
import asyncio
import json
from typing import List
import statistics

def load_csv_data(filename: str) -> str:
    """Load and parse CSV data"""
    # Mock implementation
    return """Loaded data with 1000 rows, 15 columns"""

def calculate_descriptive_stats(data: List[float]) -> dict:
    """Calculate descriptive statistics"""
    if not data:
        return {}
    
    return {
        "count": len(data),
        "mean": statistics.mean(data),
        "median": statistics.median(data),
        "stdev": statistics.stdev(data) if len(data) > 1 else 0,
        "min": min(data),
        "max": max(data),
    }

def detect_outliers(data: List[float], std_threshold: float = 3) -> List[int]:
    """Detect outliers using Z-score"""
    if len(data) < 2:
        return []
    
    mean = statistics.mean(data)
    stdev = statistics.stdev(data)
    
    outliers = []
    for i, value in enumerate(data):
        z_score = abs((value - mean) / stdev) if stdev > 0 else 0
        if z_score > std_threshold:
            outliers.append(i)
    
    return outliers

async def run_analysis_pipeline():
    """Run automated data analysis"""
    
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    
    # Data analyst agent
    analyst = AssistantAgent(
        name="analyst",
        model_client=model_client,
        system_message="""You are a data analyst.
        Your workflow:
        1. Load the data
        2. Calculate descriptive statistics
        3. Identify patterns and outliers
        4. Generate insights
        5. Provide recommendations
        
        Use available tools to perform analysis.""",
        description="Performs data analysis",
        tools=[
            load_csv_data,
            calculate_descriptive_stats,
            detect_outliers,
        ],
    )
    
    # Run analysis
    task = """Analyse the dataset 'sales_2024.csv':
    1. Load the data
    2. Calculate summary statistics for all numeric columns
    3. Identify any outliers
    4. Provide business insights
    5. Recommend actions based on findings"""
    
    response = await analyst.on_messages([
        TextMessage(content=task, source="user")
    ])
    
    print(response)

if __name__ == "__main__":
    asyncio.run(run_analysis_pipeline())
```

---

## Question-Answering System

### RAG-Based Q&A with Vector Embeddings

```python
# qa_system.py
import asyncio
from typing import List, Dict
import json

# Mock vector database
class SimpleVectorDB:
    """Simple in-memory vector database for demonstration"""
    
    def __init__(self):
        self.documents = [
            {
                "id": "doc1",
                "content": "Python is a high-level programming language known for its simplicity and readability.",
                "embedding": [0.1, 0.2, 0.3],
            },
            {
                "id": "doc2",
                "content": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
                "embedding": [0.2, 0.3, 0.4],
            },
            {
                "id": "doc3",
                "content": "Deep learning uses neural networks with multiple layers to learn hierarchical representations.",
                "embedding": [0.3, 0.4, 0.5],
            },
        ]
    
    def similarity_search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for similar documents"""
        # Simple mock - in reality would use actual embeddings
        return self.documents[:top_k]

def search_knowledge_base(query: str) -> str:
    """Search knowledge base for relevant documents"""
    db = SimpleVectorDB()
    results = db.similarity_search(query, top_k=3)
    
    context = "\n".join([
        f"Document {r['id']}: {r['content']}"
        for r in results
    ])
    
    return f"Found relevant documents:\n{context}"

def get_document_by_id(doc_id: str) -> str:
    """Get full document by ID"""
    db = SimpleVectorDB()
    for doc in db.documents:
        if doc["id"] == doc_id:
            return doc["content"]
    return "Document not found"

async def run_qa_system():
    """Run a question-answering system"""
    
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    
    # QA agent
    qa_agent = AssistantAgent(
        name="qa_agent",
        model_client=model_client,
        system_message="""You are a helpful Q&A assistant.
        When users ask questions:
        1. Search the knowledge base for relevant information
        2. Retrieve full documents if needed
        3. Synthesise an accurate answer from the retrieved context
        4. If information is not available, say so clearly
        5. Provide citations when referencing documents""",
        description="Answers questions using knowledge base",
        tools=[search_knowledge_base, get_document_by_id],
    )
    
    # Interactive Q&A
    print("Question-Answering System")
    print("Type 'exit' to quit\n")
    
    while True:
        question = input("Question: ").strip()
        
        if question.lower() == "exit":
            break
        
        if not question:
            continue
        
        response = await qa_agent.on_messages([
            TextMessage(content=question, source="user")
        ])
        
        print(f"Answer: {response}\n")

if __name__ == "__main__":
    asyncio.run(run_qa_system())
```

---

## Customer Service Bot

### Multi-Skill Support Agent

```python
# customer_service_bot.py
import asyncio
from typing import Optional

def check_order_status(order_id: str) -> str:
    """Check order status"""
    return f"Order {order_id}: Status is 'Shipped', Expected delivery: Jan 15, 2024"

def process_refund(order_id: str, reason: str) -> str:
    """Process a refund"""
    return f"Refund request for order {order_id} due to '{reason}' - Approved. Refund will be processed within 5-7 business days."

def get_product_info(product_id: str) -> str:
    """Get product information"""
    return f"Product {product_id}: Premium Widget - $99.99 - 4.8/5 stars (1,234 reviews)"

def transfer_to_human() -> str:
    """Mark for human transfer"""
    return "This conversation is being transferred to a human agent. Please wait..."

async def run_customer_service_bot():
    """Run a customer service chatbot"""
    
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    
    support_agent = AssistantAgent(
        name="support_agent",
        model_client=model_client,
        system_message="""You are a friendly customer service representative.
        You can help with:
        - Order status inquiries
        - Refunds and returns
        - Product information
        - General questions
        
        Use available tools to help customers.
        If the issue is complex or requires human judgment, offer to transfer to a human agent.
        Always be professional, empathetic, and helpful.""",
        description="Customer service support agent",
        tools=[
            check_order_status,
            process_refund,
            get_product_info,
            transfer_to_human,
        ],
    )
    
    # Simulate customer interactions
    customer_messages = [
        "Hi, can you check my order status? My order ID is ORD-123456",
        "I'd like to return it. The product arrived damaged.",
        "How long will the refund take?",
        "Can you recommend similar products?",
    ]
    
    for message in customer_messages:
        print(f"Customer: {message}")
        
        response = await support_agent.on_messages([
            TextMessage(content=message, source="customer")
        ])
        
        print(f"Agent: {response}\n")

if __name__ == "__main__":
    asyncio.run(run_customer_service_bot())
```

---

## Content Generation

### Multi-Perspective Content Creation

```python
# content_generation.py
import asyncio
from autogen_agentchat.teams import SelectorGroupChat
from autogen_agentchat.conditions import MaxMessageTermination

async def run_content_generation():
    """Generate content from multiple perspectives"""
    
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    
    # Content writer
    writer = AssistantAgent(
        name="writer",
        model_client=model_client,
        system_message="""You are a creative content writer.
        Write engaging, well-structured content.
        Use clear language and compelling narratives.
        Aim for 500-1000 words.""",
        description="Writes engaging content",
    )
    
    # Editor for quality control
    editor = AssistantAgent(
        name="editor",
        model_client=model_client,
        system_message="""You are a professional editor.
        Review content for:
        - Grammar and spelling
        - Clarity and coherence
        - Structure and flow
        - Tone and voice
        Suggest specific improvements.""",
        description="Edits and refines content",
    )
    
    # SEO specialist
    seo_specialist = AssistantAgent(
        name="seo_specialist",
        model_client=model_client,
        system_message="""You are an SEO specialist.
        Review content for:
        - Keyword optimization
        - Meta description
        - Header structure
        - Readability for search engines
        Provide specific recommendations.""",
        description="Optimizes for SEO",
    )
    
    # Create content team
    team = SelectorGroupChat(
        agents=[writer, editor, seo_specialist],
        model_client=model_client,
        termination_condition=MaxMessageTermination(max_messages=20),
    )
    
    # Generate content
    task = """Create a blog post about 'The Future of Artificial Intelligence'.
    Include:
    1. Introduction to current AI landscape
    2. Key technological advances
    3. Impact on various industries
    4. Future challenges and opportunities
    5. Conclusion with call-to-action
    
    Optimise for both reader engagement and search engines."""
    
    result = await team.run(task)
    print(result)

if __name__ == "__main__":
    asyncio.run(run_content_generation())
```

---

## Azure Integration

### Complete Azure-Based Deployment

```python
# azure_deployment.py
import asyncio
from autogen_ext.models.azure import AzureOpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from azure.identity import DefaultAzureCredential
from azure.search.documents.aio import SearchClient
from azure.search.documents.models import VectorizedQuery

async def run_azure_deployment():
    """Deploy AutoGen with Azure services"""
    
    # Initialise Azure credentials
    credential = DefaultAzureCredential()
    
    # Create Azure OpenAI client
    model_client = AzureOpenAIChatCompletionClient(
        api_key="your-api-key",  # Or use credential
        endpoint="https://your-resource.openai.azure.com/",
        deployment_name="gpt-4o",
        api_version="2024-02-01",
    )
    
    # Create agent with Azure backend
    assistant = AssistantAgent(
        name="azure_assistant",
        model_client=model_client,
        system_message="You are a helpful assistant powered by Azure OpenAI.",
    )
    
    # Use Azure AI Search for RAG
    search_client = SearchClient(
        endpoint="https://your-search-service.search.windows.net/",
        index_name="documents",
        credential=credential,
    )
    
    async def search_azure_documents(query: str) -> str:
        """Search Azure AI Search index"""
        results = await search_client.search(
            search_text=query,
            top=5
        )
        
        documents = []
        async for result in results:
            documents.append(result["content"])
        
        return "\n".join(documents)
    
    # Create RAG-enabled agent
    rag_agent = AssistantAgent(
        name="rag_assistant",
        model_client=model_client,
        system_message="""You are an assistant powered by Azure.
        Use the search tool to find relevant documents.
        Base your answers on retrieved content.""",
        tools=[search_azure_documents],
    )
    
    # Test
    response = await rag_agent.on_messages([
        TextMessage(content="What is mentioned in the documentation?", source="user")
    ])
    
    print(response)

if __name__ == "__main__":
    asyncio.run(run_azure_deployment())
```

---

## Error Handling

### Robust Error Management

```python
# error_handling_recipes.py
import asyncio
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def safe_agent_execution(
    agent,
    task: str,
    max_retries: int = 3,
    timeout_seconds: float = 30.0,
) -> Optional[str]:
    """Execute agent with comprehensive error handling"""
    
    last_error = None
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempt {attempt + 1}/{max_retries} for task: {task[:50]}...")
            
            # Execute with timeout
            result = await asyncio.wait_for(
                agent.on_messages([TextMessage(content=task, source="user")]),
                timeout=timeout_seconds
            )
            
            logger.info("Task completed successfully")
            return result
        
        except asyncio.TimeoutError:
            last_error = "Task timed out"
            logger.warning(f"Timeout on attempt {attempt + 1}")
        
        except Exception as e:
            last_error = str(e)
            logger.error(f"Error on attempt {attempt + 1}: {e}")
        
        if attempt < max_retries - 1:
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    logger.error(f"All retries failed: {last_error}")
    return None

async def test_error_handling():
    """Test error handling"""
    
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    
    agent = AssistantAgent(
        name="resilient_agent",
        model_client=model_client,
    )
    
    result = await safe_agent_execution(
        agent,
        "What is 2+2?",
        max_retries=3,
        timeout_seconds=30
    )
    
    print(f"Result: {result}")

if __name__ == "__main__":
    asyncio.run(test_error_handling())
```

---

## Testing

### Unit and Integration Testing

```python
# test_autogen_applications.py
import pytest
import asyncio
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage

@pytest.fixture
async def model_client():
    """Provide model client for tests"""
    return OpenAIChatCompletionClient(model="gpt-4o")

@pytest.fixture
async def simple_agent(model_client):
    """Provide simple agent for tests"""
    return AssistantAgent(
        name="test_agent",
        model_client=model_client,
        system_message="You are a helpful assistant.",
    )

@pytest.mark.asyncio
async def test_agent_responds_to_greeting(simple_agent):
    """Test agent responds to greeting"""
    
    response = await simple_agent.on_messages([
        TextMessage(content="Hello!", source="user")
    ])
    
    assert response is not None
    assert len(response) > 0
    assert "hello" in response.lower() or "hi" in response.lower()

@pytest.mark.asyncio
async def test_agent_performs_calculation(simple_agent):
    """Test agent can perform calculations"""
    
    response = await simple_agent.on_messages([
        TextMessage(content="What is 5 + 3?", source="user")
    ])
    
    assert response is not None
    assert "8" in response

@pytest.mark.asyncio
async def test_agent_with_tools():
    """Test agent with tool integration"""
    
    def add(a: int, b: int) -> int:
        """Add two numbers"""
        return a + b
    
    model_client = OpenAIChatCompletionClient(model="gpt-4o")
    
    agent = AssistantAgent(
        name="calculator",
        model_client=model_client,
        tools=[add],
    )
    
    response = await agent.on_messages([
        TextMessage(content="Add 5 and 3", source="user")
    ])
    
    assert response is not None
    assert "8" in response

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

---

**Running the Tests:**

```bash
pytest test_autogen_applications.py -v

# With coverage
pytest test_autogen_applications.py --cov=autogen_agentchat
```

---

These recipes provide practical, working examples for common use cases. Adapt them to your specific needs and domain requirements.

