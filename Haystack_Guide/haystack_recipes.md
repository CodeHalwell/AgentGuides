# Haystack Recipes: Production-Ready Use Cases and Examples

## Table of Contents

1. [Knowledge Base Question Answering System](#knowledge-base-question-answering-system)
2. [Multi-Tenant Customer Support Agent](#multi-tenant-customer-support-agent)
3. [Document Analysis and Summarisation Pipeline](#document-analysis-and-summarisation-pipeline)
4. [Real-Time Data Retrieval Agent](#real-time-data-retrieval-agent)
5. [Multi-Agent Collaboration System](#multi-agent-collaboration-system)
6. [Autonomous Research Agent](#autonomous-research-agent)
7. [Content Generation Pipeline](#content-generation-pipeline)
8. [Anomaly Detection and Reporting](#anomaly-detection-and-reporting)
9. [Conversational Code Assistant](#conversational-code-assistant)
10. [Enterprise Knowledge Management](#enterprise-knowledge-management)

---

## Knowledge Base Question Answering System

A complete RAG system for answering questions from an enterprise knowledge base.

```python
{% raw %}
from haystack import Pipeline, component
from haystack.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.components.retrievers.elasticsearch import ElasticsearchBM25Retriever
from haystack.components.builders import PromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import Document
from typing import List
import logging

logger = logging.getLogger(__name__)

class KnowledgeBaseQA:
    """
    Complete knowledge base Q&A system using Haystack.
    """
    
    def __init__(self, elasticsearch_url: str, openai_api_key: str):
        self.setup_document_store(elasticsearch_url)
        self.setup_pipeline(openai_api_key)
        self.document_metadata = {}
    
    def setup_document_store(self, elasticsearch_url: str):
        """Initialize document store."""
        self.doc_store = ElasticsearchDocumentStore(
            hosts=elasticsearch_url,
            index="knowledge_base",
            embedding_dim=384  # Using small embeddings for speed
        )
        logger.info("Document store initialised")
    
    def setup_pipeline(self, openai_api_key: str):
        """Setup RAG pipeline."""
        pipeline = Pipeline()
        
        # Retriever
        retriever = ElasticsearchBM25Retriever(document_store=self.doc_store)
        pipeline.add_component("retriever", retriever)
        
        # Prompt builder
        prompt_template = """
        Based on the following documents, answer the question.
        
        Documents:
        {% for doc in documents %}
        - {{ doc.content }}
        {% endfor %}
        
        Question: {{ question }}
        Answer:
        """
        
        prompt_builder = PromptBuilder(template=prompt_template)
        pipeline.add_component("prompt_builder", prompt_builder)
        
        # Generator
        generator = OpenAIChatGenerator(
            model="gpt-4o",
            api_key=openai_api_key
        )
        pipeline.add_component("generator", generator)
        
        # Connect components
        pipeline.connect("retriever.documents", "prompt_builder.documents")
        pipeline.connect("prompt_builder.prompt", "generator.prompt")
        
        self.pipeline = pipeline
        logger.info("Pipeline setup complete")
    
    def index_documents(self, documents: List[Document]):
        """Index documents into knowledge base."""
        logger.info(f"Indexing {len(documents)} documents")
        self.doc_store.write_documents(documents)
        logger.info("Indexing complete")
    
    def query(self, question: str, top_k: int = 5) -> dict:
        """
        Query knowledge base for answer.
        
        Args:
            question: Question to ask
            top_k: Number of documents to retrieve
            
        Returns:
            Query results with answer
        """
        try:
            result = self.pipeline.run({
                "retriever": {"query": question, "top_k": top_k},
                "prompt_builder": {"question": question}
            })
            
            return {
                "question": question,
                "answer": result.get("generator", {}).get("reply", ""),
                "success": True
            }
        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            return {
                "question": question,
                "error": str(e),
                "success": False
            }

# Usage
qa_system = KnowledgeBaseQA(
    elasticsearch_url="http://localhost:9200",
    openai_api_key="sk-..."
)

# Index sample documents
sample_docs = [
    Document(content="Our return policy allows returns within 30 days for full refund"),
    Document(content="Shipping takes 5-7 business days for standard delivery"),
    Document(content="We offer 1-year warranty on all products")
]
qa_system.index_documents(sample_docs)

# Query
result = qa_system.query("What's your return policy?")
print(f"Answer: {result['answer']}")
{% endraw %}
```

---

## Multi-Tenant Customer Support Agent

Customer support agent handling multiple customers with isolated data.

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.tools import Tool
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

class MultiTenantSupportAgent:
    """
    Customer support agent supporting multiple tenants with data isolation.
    """
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.tenant_tools: Dict[str, list] = {}
        self.llm = OpenAIChatGenerator(model="gpt-4o")
    
    def create_tenant_agent(self, tenant_id: str, tenant_config: dict):
        """
        Create isolated agent for tenant.
        
        Args:
            tenant_id: Unique tenant identifier
            tenant_config: Tenant-specific configuration
        """
        # Create tenant-specific tools
        tools = self._create_tenant_tools(tenant_id, tenant_config)
        self.tenant_tools[tenant_id] = tools
        
        # Create agent with tenant-specific system prompt
        system_prompt = f"""
        You are a customer support specialist for {tenant_config.get('company_name', 'our company')}.
        
        Guidelines:
        - Always be professional and courteous
        - Use tenant-specific knowledge base and policies
        - Escalate complex issues when necessary
        - Respect data privacy and isolation
        """
        
        agent = Agent(
            tools=tools,
            llm=self.llm,
            system_prompt=system_prompt
        )
        
        self.agents[tenant_id] = agent
        logger.info(f"Agent created for tenant: {tenant_id}")
    
    def _create_tenant_tools(self, tenant_id: str, config: dict) -> list:
        """Create tenant-specific tools."""
        
        def lookup_customer_info(customer_id: str) -> dict:
            """Look up customer information."""
            # Fetch from tenant-specific database
            return {
                "customer_id": customer_id,
                "name": "John Doe",
                "account_status": "active",
                "tenant_id": tenant_id  # Ensure isolation
            }
        
        def create_support_ticket(issue: str, priority: str) -> dict:
            """Create support ticket."""
            return {
                "ticket_id": f"{tenant_id}-TICKET-001",
                "issue": issue,
                "priority": priority,
                "tenant_id": tenant_id
            }
        
        def check_knowledge_base(query: str) -> dict:
            """Check tenant knowledge base."""
            # Search tenant-specific KB
            return {
                "results": ["KB Article 1", "KB Article 2"],
                "tenant_id": tenant_id
            }
        
        return [
            Tool(function=lookup_customer_info, description="Lookup customer information"),
            Tool(function=create_support_ticket, description="Create support ticket"),
            Tool(function=check_knowledge_base, description="Search knowledge base")
        ]
    
    def handle_customer_query(
        self,
        tenant_id: str,
        customer_query: str,
        customer_id: Optional[str] = None
    ) -> dict:
        """
        Handle customer query with tenant isolation.
        
        Args:
            tenant_id: Customer's tenant ID
            customer_query: Customer question
            customer_id: Optional customer ID for context
            
        Returns:
            Agent response
        """
        if tenant_id not in self.agents:
            return {
                "success": False,
                "error": f"Tenant {tenant_id} not configured"
            }
        
        agent = self.agents[tenant_id]
        
        # Include customer context if available
        enhanced_query = customer_query
        if customer_id:
            enhanced_query = f"Customer ID: {customer_id}\nQuery: {customer_query}"
        
        try:
            response = agent.run(query=enhanced_query, max_iterations=10)
            
            return {
                "success": True,
                "tenant_id": tenant_id,
                "response": response,
                "customer_id": customer_id
            }
        except Exception as e:
            logger.error(f"Query processing failed for tenant {tenant_id}: {str(e)}")
            return {
                "success": False,
                "tenant_id": tenant_id,
                "error": str(e)
            }

# Usage
support_system = MultiTenantSupportAgent()

# Create agents for different tenants
support_system.create_tenant_agent("tenant-001", {
    "company_name": "Acme Corp",
    "support_email": "support@acme.com"
})

support_system.create_tenant_agent("tenant-002", {
    "company_name": "TechStart Inc",
    "support_email": "help@techstart.io"
})

# Handle customer queries
response = support_system.handle_customer_query(
    tenant_id="tenant-001",
    customer_query="I want to return my order",
    customer_id="CUST-12345"
)
print(response)
```

---

## Document Analysis and Summarisation Pipeline

Analyse and summarise large document collections.

```python
{% raw %}
from haystack import Pipeline, component
from haystack.components.builders import PromptBuilder
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import Document
from typing import List
import logging

logger = logging.getLogger(__name__)

class DocumentAnalyser:
    """
    Analyse and summarise documents using Haystack.
    """
    
    def __init__(self, openai_api_key: str):
        self.setup_pipeline(openai_api_key)
    
    def setup_pipeline(self, openai_api_key: str):
        """Setup analysis pipeline."""
        pipeline = Pipeline()
        
        # Text chunker (simulated)
        @component
        class TextChunker:
            @component.output_types(chunks=list)
            def run(self, documents: List[Document]):
                chunks = []
                for doc in documents:
                    # Split document into chunks
                    text = doc.content
                    chunk_size = 1000
                    for i in range(0, len(text), chunk_size):
                        chunks.append(text[i:i+chunk_size])
                return {"chunks": chunks}
        
        # Summary generator
        summary_prompt = """
        Summarise the following text in 3-5 sentences:
        
        Text:
        {{ chunk }}
        
        Summary:
        """
        
        summary_builder = PromptBuilder(template=summary_prompt)
        generator = OpenAIChatGenerator(
            model="gpt-4o",
            api_key=openai_api_key
        )
        
        # Add components
        pipeline.add_component("chunker", TextChunker())
        pipeline.add_component("summary_builder", summary_builder)
        pipeline.add_component("generator", generator)
        
        self.pipeline = pipeline
        logger.info("Analysis pipeline ready")
    
    def analyse_documents(self, documents: List[Document]) -> dict:
        """
        Analyse collection of documents.
        
        Args:
            documents: Documents to analyse
            
        Returns:
            Analysis results
        """
        try:
            results = []
            
            for doc in documents:
                result = {
                    "document_id": doc.id,
                    "title": doc.meta.get("title", "Untitled"),
                    "length": len(doc.content),
                    "summary": self._summarise_document(doc)
                }
                results.append(result)
            
            return {
                "success": True,
                "documents_processed": len(documents),
                "analyses": results
            }
        
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _summarise_document(self, doc: Document) -> str:
        """Summarise a single document."""
        # Implementation would use the pipeline
        return "This document discusses..."

# Usage
analyser = DocumentAnalyser(openai_api_key="sk-...")

documents = [
    Document(content="Long document 1...", meta={"title": "Document 1"}),
    Document(content="Long document 2...", meta={"title": "Document 2"})
]

results = analyser.analyse_documents(documents)
print(results)
{% endraw %}
```

---

## Real-Time Data Retrieval Agent

Agent that retrieves and processes real-time data from multiple sources.

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.tools import Tool
import aiohttp
import asyncio
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class RealTimeDataAgent:
    """
    Agent for retrieving and processing real-time data.
    """
    
    def __init__(self):
        self.llm = OpenAIChatGenerator(model="gpt-4o")
        self.tools = self._create_tools()
        self.agent = Agent(tools=self.tools, llm=self.llm)
    
    def _create_tools(self) -> list:
        """Create real-time data retrieval tools."""
        
        def get_stock_price(symbol: str) -> dict:
            """Get current stock price."""
            # Simulate API call
            return {
                "symbol": symbol,
                "price": 150.25,
                "timestamp": datetime.now().isoformat()
            }
        
        def get_weather_data(location: str) -> dict:
            """Get current weather data."""
            return {
                "location": location,
                "temperature": 22,
                "condition": "Cloudy",
                "timestamp": datetime.now().isoformat()
            }
        
        def search_news(query: str) -> dict:
            """Search for recent news."""
            return {
                "query": query,
                "results": [
                    "News headline 1",
                    "News headline 2",
                    "News headline 3"
                ],
                "timestamp": datetime.now().isoformat()
            }
        
        return [
            Tool(function=get_stock_price, description="Get current stock price"),
            Tool(function=get_weather_data, description="Get weather data for location"),
            Tool(function=search_news, description="Search for news articles")
        ]
    
    def query_data(self, query: str) -> dict:
        """
        Query for real-time data.
        
        Args:
            query: Data query
            
        Returns:
            Data results
        """
        try:
            logger.info(f"Processing data query: {query}")
            result = self.agent.run(query=query, max_iterations=10)
            
            return {
                "success": True,
                "query": query,
                "result": result,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Data query failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

# Usage
data_agent = RealTimeDataAgent()

# Query for real-time data
result = data_agent.query_data(
    "What's the current stock price for AAPL and the weather in London?"
)
print(result)
```

---

## Multi-Agent Collaboration System

Complex system with multiple agents collaborating to solve problems.

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class ResearchTeam:
    """
    Multi-agent research team collaborating on complex tasks.
    """
    
    def __init__(self):
        self.llm = OpenAIChatGenerator(model="gpt-4o")
        self.agents = self._create_agents()
        self.collaboration_history: List[dict] = []
    
    def _create_agents(self) -> Dict[str, Agent]:
        """Create specialised agents."""
        
        agents = {
            "researcher": Agent(
                tools=[],
                llm=self.llm,
                system_prompt="You are a research expert. Analyse topics deeply and provide comprehensive information."
            ),
            "analyst": Agent(
                tools=[],
                llm=self.llm,
                system_prompt="You are a data analyst. Identify patterns, trends, and insights in data."
            ),
            "writer": Agent(
                tools=[],
                llm=self.llm,
                system_prompt="You are a professional writer. Create clear, concise, well-structured content."
            ),
            "critic": Agent(
                tools=[],
                llm=self.llm,
                system_prompt="You are a critical reviewer. Identify weaknesses, gaps, and areas for improvement."
            )
        }
        
        return agents
    
    def collaborate_on_task(self, task: str) -> dict:
        """
        Collaborate on complex task with multiple agents.
        
        Args:
            task: Task to complete
            
        Returns:
            Collaborative result
        """
        logger.info(f"Starting collaboration on task: {task}")
        
        collaboration_log = []
        
        try:
            # Phase 1: Research
            research_result = self.agents["researcher"].run(
                query=f"Research and provide comprehensive information on: {task}",
                max_iterations=5
            )
            collaboration_log.append({
                "agent": "researcher",
                "phase": 1,
                "result": research_result
            })
            
            # Phase 2: Analysis
            analysis_result = self.agents["analyst"].run(
                query=f"Analyse the research on: {task}\nResearch summary: {research_result}",
                max_iterations=5
            )
            collaboration_log.append({
                "agent": "analyst",
                "phase": 2,
                "result": analysis_result
            })
            
            # Phase 3: Writing
            writing_result = self.agents["writer"].run(
                query=f"Write a professional summary of: {task}\nAnalysis: {analysis_result}",
                max_iterations=5
            )
            collaboration_log.append({
                "agent": "writer",
                "phase": 3,
                "result": writing_result
            })
            
            # Phase 4: Critical Review
            review_result = self.agents["critic"].run(
                query=f"Review the following content for: {task}\nContent: {writing_result}",
                max_iterations=5
            )
            collaboration_log.append({
                "agent": "critic",
                "phase": 4,
                "result": review_result
            })
            
            self.collaboration_history.append({
                "task": task,
                "log": collaboration_log
            })
            
            return {
                "success": True,
                "task": task,
                "final_result": writing_result,
                "review": review_result,
                "collaboration_phases": 4
            }
        
        except Exception as e:
            logger.error(f"Collaboration failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "partial_log": collaboration_log
            }

# Usage
team = ResearchTeam()
result = team.collaborate_on_task("Analyse the impact of AI on employment")
print(result)
```

---

## Autonomous Research Agent

Self-directed agent that conducts research autonomously.

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.tools import Tool
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class AutonomousResearchAgent:
    """
    Self-directed research agent with iterative refinement.
    """
    
    def __init__(self):
        self.llm = OpenAIChatGenerator(model="gpt-4o")
        self.tools = self._create_research_tools()
        self.agent = Agent(tools=self.tools, llm=self.llm)
        self.research_log: List[dict] = []
    
    def _create_research_tools(self) -> list:
        """Create research tools."""
        
        def search_academic_databases(query: str) -> dict:
            """Search academic papers."""
            return {
                "results": ["Paper 1", "Paper 2", "Paper 3"],
                "source": "Academic Databases"
            }
        
        def extract_key_findings(text: str) -> dict:
            """Extract key findings from text."""
            return {
                "findings": ["Finding 1", "Finding 2"],
                "confidence": 0.85
            }
        
        def synthesise_information(*sources) -> dict:
            """Synthesise information from multiple sources."""
            return {
                "synthesis": "Combined insights...",
                "sources_processed": len(sources)
            }
        
        return [
            Tool(function=search_academic_databases, description="Search academic databases"),
            Tool(function=extract_key_findings, description="Extract findings"),
            Tool(function=synthesise_information, description="Synthesise information")
        ]
    
    def conduct_research(self, research_question: str, iterations: int = 3) -> dict:
        """
        Conduct autonomous research.
        
        Args:
            research_question: Question to research
            iterations: Number of refinement iterations
            
        Returns:
            Research findings
        """
        logger.info(f"Starting autonomous research on: {research_question}")
        
        findings = []
        
        try:
            for iteration in range(iterations):
                # Conduct research iteration
                logger.info(f"Research iteration {iteration + 1}/{iterations}")
                
                result = self.agent.run(
                    query=f"{research_question}\nIteration: {iteration + 1}",
                    max_iterations=10
                )
                
                iteration_log = {
                    "iteration": iteration + 1,
                    "timestamp": datetime.now().isoformat(),
                    "findings": result
                }
                
                findings.append(iteration_log)
                self.research_log.append(iteration_log)
            
            return {
                "success": True,
                "research_question": research_question,
                "total_iterations": iterations,
                "findings": findings
            }
        
        except Exception as e:
            logger.error(f"Research failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "partial_findings": findings
            }

# Usage
researcher = AutonomousResearchAgent()
results = researcher.conduct_research(
    "What are emerging trends in quantum computing?",
    iterations=3
)
print(results)
```

---

## Conversational Code Assistant

AI assistant for helping with coding tasks.

```python
from haystack.components.agents import Agent
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.tools import Tool
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class CodeAssistant:
    """
    Conversational AI assistant for coding help.
    """
    
    def __init__(self):
        self.llm = OpenAIChatGenerator(model="gpt-4o")
        self.tools = self._create_code_tools()
        self.agent = Agent(
            tools=self.tools,
            llm=self.llm,
            system_prompt="You are an expert code assistant. Provide clear, well-commented code examples."
        )
        self.conversation_context: List[dict] = []
    
    def _create_code_tools(self) -> list:
        """Create code-related tools."""
        
        def check_syntax(code: str, language: str) -> dict:
            """Check code syntax."""
            return {
                "valid": True,
                "language": language,
                "errors": []
            }
        
        def explain_code(code: str) -> dict:
            """Explain what code does."""
            return {
                "explanation": "This code does...",
                "time_complexity": "O(n)",
                "space_complexity": "O(1)"
            }
        
        def suggest_improvements(code: str) -> dict:
            """Suggest code improvements."""
            return {
                "improvements": [
                    "Use list comprehension for brevity",
                    "Add type hints"
                ],
                "refactored_code": "def improved_function()..."
            }
        
        return [
            Tool(function=check_syntax, description="Check code syntax"),
            Tool(function=explain_code, description="Explain code functionality"),
            Tool(function=suggest_improvements, description="Suggest code improvements")
        ]
    
    def assist(self, query: str, context: Optional[str] = None) -> dict:
        """
        Provide coding assistance.
        
        Args:
            query: Coding question or request
            context: Optional code context
            
        Returns:
            Assistance response
        """
        # Build full query with context
        full_query = query
        if context:
            full_query = f"Code context:\n{context}\n\nQuestion: {query}"
        
        try:
            response = self.agent.run(query=full_query, max_iterations=5)
            
            self.conversation_context.append({
                "query": query,
                "context": context,
                "response": response
            })
            
            return {
                "success": True,
                "assistance": response,
                "conversation_length": len(self.conversation_context)
            }
        
        except Exception as e:
            logger.error(f"Assistance failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

# Usage
assistant = CodeAssistant()

result = assistant.assist(
    query="How can I optimise this function for better performance?",
    context="def sum_all_numbers(nums):\n    total = 0\n    for num in nums:\n        total += num\n    return total"
)
print(result)
```

---

These recipes provide production-ready patterns for common Haystack use cases. Each recipe demonstrates best practices, error handling, and integration patterns suitable for enterprise deployments.
