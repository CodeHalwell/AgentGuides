# LlamaIndex Recipes and Practical Examples

Real-world recipes for building LlamaIndex applications: from simple RAG chatbots to complex multi-agent systems.

---

## Table of Contents

1. [Basic RAG Chatbot](#basic-rag-chatbot)
2. [Research Paper Analyzer](#research-paper-analyzer)
3. [Code Documentation Assistant](#code-documentation-assistant)
4. [Multi-Document Comparison Engine](#multi-document-comparison-engine)
5. [Real-time News Analysis Agent](#real-time-news-analysis-agent)
6. [Data Extraction Pipeline](#data-extraction-pipeline)
7. [Conversational SQL Agent](#conversational-sql-agent)
8. [Knowledge Graph Builder](#knowledge-graph-builder)
9. [Multi-Step Reasoning Agent](#multi-step-reasoning-agent)
10. [Customer Support Triage Agent](#customer-support-triage-agent)

---

## Basic RAG Chatbot

### Complete Implementation

```python
"""
simple_rag_chatbot.py
A basic RAG chatbot that answers questions about documents
"""

import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Document,
)
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.memory import ChatMemoryBuffer

def setup_rag_chatbot(documents_dir: str = "./documents"):
    """Setup RAG chatbot with memory."""
    
    # Load documents
    documents = SimpleDirectoryReader(
        input_dir=documents_dir
    ).load_data()
    
    if not documents:
        raise ValueError(f"No documents found in {documents_dir}")
    
    print(f"Loaded {len(documents)} documents")
    
    # Setup embedding and LLM
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-large"
    )
    llm = OpenAI(model="gpt-4", temperature=0.7)
    
    # Create index
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
        show_progress=True,
    )
    
    # Setup memory
    memory = ChatMemoryBuffer.from_defaults(token_limit=4096)
    
    # Create query engine
    query_engine = index.as_query_engine(
        llm=llm,
        memory=memory,
        similarity_top_k=3,
    )
    
    return query_engine

def run_chatbot():
    """Run interactive chatbot."""
    query_engine = setup_rag_chatbot()
    
    print("=" * 50)
    print("RAG Chatbot Ready! Type 'exit' to quit.")
    print("=" * 50)
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        response = query_engine.query(user_input)
        
        print(f"\nAssistant: {response.response}")
        print("\nSources:")
        for node in response.source_nodes:
            print(f"  - {node.text[:100]}...")

if __name__ == "__main__":
    run_chatbot()
```

### Docker Compose Setup

```yaml
# docker-compose.yml for RAG chatbot
version: '3.8'

services:
  chatbot:
    build: .
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    volumes:
      - ./documents:/app/documents
      - ./data:/app/data
    stdin_open: true
    tty: true
```

---

## Research Paper Analyzer

### Analysing Academic Papers

```python
"""
research_paper_analyzer.py
Analyze research papers with citation tracking
"""

from llama_index.core import (
    VectorStoreIndex,
    Document,
)
from llama_index.readers.file.pdf import PDFReader
from llama_index.core.tools import QueryEngineTool, FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
import os
from pathlib import Path

class ResearchPaperAnalyzer:
    """Analyze research papers using LlamaIndex."""
    
    def __init__(self, papers_dir: str):
        self.papers_dir = papers_dir
        self.llm = OpenAI(model="gpt-4")
        self.indexes = {}
        self.query_engines = {}
    
    def load_papers(self):
        """Load all PDF papers from directory."""
        pdf_reader = PDFReader()
        
        for pdf_file in Path(self.papers_dir).glob("*.pdf"):
            paper_name = pdf_file.stem
            print(f"Loading {paper_name}...")
            
            # Load PDF
            try:
                documents = pdf_reader.load_data(str(pdf_file))
                
                # Create index
                index = VectorStoreIndex.from_documents(
                    documents,
                    show_progress=True,
                )
                
                # Store index and engine
                self.indexes[paper_name] = index
                self.query_engines[paper_name] = index.as_query_engine()
                
                print(f"✓ {paper_name} loaded successfully")
            
            except Exception as e:
                print(f"✗ Error loading {paper_name}: {e}")
    
    def create_agent(self):
        """Create agent for paper analysis."""
        
        # Create tools for each paper
        tools = []
        for paper_name, query_engine in self.query_engines.items():
            tool = QueryEngineTool.from_defaults(
                query_engine=query_engine,
                name=f"analyze_{paper_name}",
                description=f"Analyse {paper_name}",
            )
            tools.append(tool)
        
        # Analysis tool
        def compare_papers(aspect: str, paper_names: list) -> str:
            """Compare papers on specific aspect."""
            results = {}
            for paper_name in paper_names:
                if paper_name in self.query_engines:
                    response = self.query_engines[paper_name].query(
                        f"What does this paper say about {aspect}?"
                    )
                    results[paper_name] = str(response)
            return results
        
        tools.append(FunctionTool.from_defaults(
            fn=compare_papers,
            name="compare_papers",
            description="Compare multiple papers on a topic",
        ))
        
        # Create agent
        agent = ReActAgent.from_tools(
            tools=tools,
            llm=self.llm,
            verbose=True,
        )
        
        return agent
    
    def analyze(self, query: str):
        """Analyse query across papers."""
        agent = self.create_agent()
        response = agent.run(query)
        return response

# Usage
if __name__ == "__main__":
    analyzer = ResearchPaperAnalyzer("./papers")
    analyzer.load_papers()
    
    # Analyse
    result = analyzer.analyze(
        "Compare the methodology across papers"
    )
    print(f"\nAnalysis Result:\n{result}")
```

---

## Code Documentation Assistant

### Automatic API Documentation

```python
"""
code_doc_assistant.py
Generate documentation from source code
"""

from llama_index.core import (
    VectorStoreIndex,
    Document,
)
from pathlib import Path
import ast

class CodeDocumentationAssistant:
    """Generate documentation from code."""
    
    def __init__(self, repo_path: str):
        self.repo_path = repo_path
        self.documents = []
    
    def extract_code_structure(self, python_file: Path) -> str:
        """Extract functions and classes from Python file."""
        with open(python_file, 'r') as f:
            try:
                tree = ast.parse(f.read())
            except SyntaxError:
                return ""
        
        structure = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                doc = ast.get_docstring(node)
                args = [arg.arg for arg in node.args.args]
                structure.append(
                    f"Function: {node.name}({', '.join(args)})\n"
                    f"Docstring: {doc or 'No docstring'}"
                )
            
            elif isinstance(node, ast.ClassDef):
                doc = ast.get_docstring(node)
                structure.append(
                    f"Class: {node.name}\n"
                    f"Docstring: {doc or 'No docstring'}"
                )
        
        return "\n\n".join(structure)
    
    def index_repository(self):
        """Index all Python files in repository."""
        repo = Path(self.repo_path)
        
        for py_file in repo.rglob("*.py"):
            structure = self.extract_code_structure(py_file)
            
            if structure:
                doc = Document(
                    text=structure,
                    metadata={
                        "file": str(py_file.relative_to(repo)),
                        "source": "code",
                    }
                )
                self.documents.append(doc)
        
        # Create index
        if self.documents:
            self.index = VectorStoreIndex.from_documents(
                self.documents,
                show_progress=True,
            )
            self.query_engine = self.index.as_query_engine()
        else:
            raise ValueError("No Python files found")
    
    def generate_docs(self, query: str) -> str:
        """Generate documentation based on query."""
        response = self.query_engine.query(query)
        return str(response)

# Usage
if __name__ == "__main__":
    assistant = CodeDocumentationAssistant("./myproject")
    assistant.index_repository()
    
    # Generate documentation
    docs = assistant.generate_docs(
        "What are the main classes and their purposes?"
    )
    print(docs)
```

---

## Multi-Document Comparison Engine

### Comparing Multiple Documents

```python
"""
multi_document_comparison.py
Compare and contrast multiple documents
"""

from llama_index.core import (
    VectorStoreIndex,
    Document,
)
from llama_index.core.query_engine import (
    SubQuestionQueryEngine,
)
from llama_index.core.tools import QueryEngineTool
from llama_index.llms.openai import OpenAI

class MultiDocumentComparison:
    """Compare content across multiple documents."""
    
    def __init__(self, documents_dict: dict):
        """
        Initialize with documents.
        documents_dict: {"doc_name": [Document, Document, ...]}
        """
        self.documents_dict = documents_dict
        self.indexes = {}
        self.query_engines = {}
        self.llm = OpenAI(model="gpt-4")
    
    def create_indexes(self):
        """Create indexes for each document group."""
        for doc_name, docs in self.documents_dict.items():
            print(f"Indexing {doc_name}...")
            
            index = VectorStoreIndex.from_documents(
                docs,
                show_progress=True,
            )
            
            self.indexes[doc_name] = index
            self.query_engines[doc_name] = index.as_query_engine()
    
    def create_comparison_engine(self):
        """Create sub-question engine for comparison."""
        
        tools = [
            QueryEngineTool.from_defaults(
                query_engine=engine,
                name=f"{doc_name}_engine",
                description=f"Query {doc_name}",
            )
            for doc_name, engine in self.query_engines.items()
        ]
        
        self.comparison_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=tools,
            llm=self.llm,
        )
    
    def compare(self, aspect: str) -> str:
        """Compare documents on specific aspect."""
        query = f"Compare how each document addresses {aspect}"
        response = self.comparison_engine.query(query)
        return str(response)
    
    def contrast(self, aspect: str) -> str:
        """Contrast documents on specific aspect."""
        query = f"What are the key differences in how documents address {aspect}?"
        response = self.comparison_engine.query(query)
        return str(response)

# Usage
if __name__ == "__main__":
    # Create sample documents
    docs = {
        "Approach_A": [
            Document(text="Approach A uses machine learning..."),
            Document(text="Key advantages of A..."),
        ],
        "Approach_B": [
            Document(text="Approach B uses statistical methods..."),
            Document(text="Key advantages of B..."),
        ],
    }
    
    comparison = MultiDocumentComparison(docs)
    comparison.create_indexes()
    comparison.create_comparison_engine()
    
    # Compare
    print("Comparison:")
    print(comparison.compare("implementation strategy"))
    
    print("\nContrast:")
    print(comparison.contrast("implementation strategy"))
```

---

## Real-time News Analysis Agent

### Analyzing News with External Tools

```python
"""
news_analysis_agent.py
Analyze news in real-time with agent
"""

from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
import feedparser

class NewsAnalysisAgent:
    """Analyse news using agents."""
    
    def __init__(self):
        self.llm = OpenAI(model="gpt-4")
    
    def fetch_rss_feed(self, feed_url: str, limit: int = 5) -> str:
        """Fetch latest articles from RSS feed."""
        feed = feedparser.parse(feed_url)
        
        articles = []
        for entry in feed.entries[:limit]:
            articles.append({
                "title": entry.title,
                "summary": entry.summary,
                "link": entry.link,
            })
        
        return str(articles)
    
    def search_news_topic(self, topic: str) -> str:
        """Search for news on specific topic."""
        # In real implementation, use news API
        return f"Recent articles about {topic}..."
    
    def sentiment_analysis(self, text: str) -> str:
        """Analyse sentiment of text."""
        # Simplified sentiment analysis
        from nltk.sentiment import SentimentIntensityAnalyzer
        
        sia = SentimentIntensityAnalyzer()
        scores = sia.polarity_scores(text)
        
        return f"Sentiment scores: {scores}"
    
    def create_agent(self):
        """Create news analysis agent."""
        
        tools = [
            FunctionTool.from_defaults(
                fn=self.fetch_rss_feed,
                name="fetch_rss",
                description="Fetch articles from RSS feed",
            ),
            FunctionTool.from_defaults(
                fn=self.search_news_topic,
                name="search_news",
                description="Search for news on topic",
            ),
            FunctionTool.from_defaults(
                fn=self.sentiment_analysis,
                name="analyze_sentiment",
                description="Analyse sentiment of text",
            ),
        ]
        
        agent = ReActAgent.from_tools(
            tools=tools,
            llm=self.llm,
            verbose=True,
        )
        
        return agent
    
    def analyze_news(self, query: str) -> str:
        """Analyse news based on query."""
        agent = self.create_agent()
        response = agent.run(query)
        return response

# Usage
if __name__ == "__main__":
    analyzer = NewsAnalysisAgent()
    
    result = analyzer.analyze_news(
        "Fetch the latest tech news and analyse the sentiment"
    )
    print(f"\nAnalysis:\n{result}")
```

---

## Data Extraction Pipeline

### Structured Data Extraction

```python
"""
data_extraction_pipeline.py
Extract structured data from unstructured text
"""

from llama_index.core import Document
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel, Field
from typing import List
from pathlib import Path

class Person(BaseModel):
    name: str = Field(description="Person's name")
    role: str = Field(description="Professional role")
    skills: List[str] = Field(description="List of skills")
    experience_years: int = Field(description="Years of experience")

class ExtractionPipeline:
    """Extract structured data from documents."""
    
    def __init__(self):
        self.llm = OpenAI(model="gpt-4")
        self.parser = PydanticOutputParser(output_class=Person)
    
    def extract_from_text(self, text: str) -> Person:
        """Extract person information from text."""
        prompt = f"""Extract person information from this text:
{text}

{self.parser.format_instructions()}"""
        
        response = self.llm.complete(prompt)
        parsed = self.parser.parse(str(response))
        
        return parsed
    
    def batch_extract(self, documents: List[Document]) -> List[Person]:
        """Extract from multiple documents."""
        results = []
        
        for doc in documents:
            try:
                person = self.extract_from_text(doc.text)
                results.append(person)
            except Exception as e:
                print(f"Error extracting from document: {e}")
        
        return results
    
    def save_results(self, results: List[Person], output_file: str):
        """Save extracted data to CSV."""
        import csv
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=['name', 'role', 'skills', 'experience_years']
            )
            writer.writeheader()
            
            for person in results:
                writer.writerow({
                    'name': person.name,
                    'role': person.role,
                    'skills': ', '.join(person.skills),
                    'experience_years': person.experience_years,
                })

# Usage
if __name__ == "__main__":
    pipeline = ExtractionPipeline()
    
    # Sample text
    text = """
    Alice Smith is a Senior Software Engineer with 8 years of experience.
    She specializes in Python, JavaScript, and AWS. She's an expert in
    building scalable distributed systems.
    """
    
    result = pipeline.extract_from_text(text)
    print(f"Extracted: {result}")
```

---

## Conversational SQL Agent

### Querying Databases Conversationally

```python
"""
sql_agent.py
Query databases using natural language
"""

from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
import sqlite3
from typing import List

class SQLAgent:
    """Query databases conversationally."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.llm = OpenAI(model="gpt-4")
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()
    
    def get_schema(self) -> str:
        """Get database schema."""
        self.cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        )
        tables = self.cursor.fetchall()
        
        schema = []
        for table in tables:
            table_name = table[0]
            self.cursor.execute(f"PRAGMA table_info({table_name})")
            columns = self.cursor.fetchall()
            
            col_info = []
            for col in columns:
                col_info.append(f"{col[1]} ({col[2]})")
            
            schema.append(f"{table_name}: {', '.join(col_info)}")
        
        return "\n".join(schema)
    
    def execute_query(self, query: str) -> str:
        """Execute SQL query safely."""
        try:
            # Validate query (prevent DROP, DELETE, etc.)
            unsafe_keywords = ["DROP", "DELETE", "TRUNCATE", "GRANT"]
            if any(kw in query.upper() for kw in unsafe_keywords):
                return "Query contains unsafe operations"
            
            self.cursor.execute(query)
            results = self.cursor.fetchall()
            
            return str(results[:10])  # Limit results
        
        except Exception as e:
            return f"Query error: {str(e)}"
    
    def translate_to_sql(self, question: str) -> str:
        """Translate natural language to SQL."""
        schema = self.get_schema()
        
        prompt = f"""Given this database schema:
{schema}

Translate this question to SQL:
{question}

Return only the SQL query."""
        
        response = self.llm.complete(prompt)
        return str(response).strip()
    
    def create_agent(self):
        """Create SQL agent."""
        
        tools = [
            FunctionTool.from_defaults(
                fn=self.translate_to_sql,
                name="translate_sql",
                description="Translate natural language to SQL",
            ),
            FunctionTool.from_defaults(
                fn=self.execute_query,
                name="execute_sql",
                description="Execute SQL query",
            ),
            FunctionTool.from_defaults(
                fn=self.get_schema,
                name="get_schema",
                description="Get database schema",
            ),
        ]
        
        agent = ReActAgent.from_tools(
            tools=tools,
            llm=self.llm,
            system_prompt="You are a helpful database assistant. Help users query the database.",
            verbose=True,
        )
        
        return agent
    
    def query(self, question: str) -> str:
        """Query database conversationally."""
        agent = self.create_agent()
        response = agent.run(question)
        return response

# Usage
if __name__ == "__main__":
    agent = SQLAgent("./database.db")
    
    result = agent.query("How many users registered last month?")
    print(f"Result: {result}")
```

---

## Knowledge Graph Builder

### Building Knowledge Graphs from Documents

{% raw %}
```python

"""
knowledge_graph_builder.py
Extract and build knowledge graphs from documents
"""

from llama_index.core import Document
from llama_index.llms.openai import OpenAI
from pydantic import BaseModel, Field
from typing import List
import json

class Entity(BaseModel):
    name: str
    entity_type: str  # person, organization, concept, etc.

class Relationship(BaseModel):
    source: str
    target: str
    relationship_type: str

class KnowledgeGraph(BaseModel):
    entities: List[Entity]
    relationships: List[Relationship]

class KnowledgeGraphBuilder:
    """Build knowledge graphs from text."""
    
    def __init__(self):
        self.llm = OpenAI(model="gpt-4")
        self.graph = {
            "entities": {},
            "relationships": []
        }
    
    def extract_entities_and_relations(self, text: str) -> KnowledgeGraph:
        """Extract entities and relationships from text."""
        
        prompt = f"""Extract entities and relationships from this text:
{text}

Return as JSON with this structure:
{{
  "entities": [
    {{"name": "Entity Name", "entity_type": "type"}},
    ...
  ],
  "relationships": [
    {{"source": "Entity1", "target": "Entity2", "relationship_type": "type"}},
    ...
  ]
}}"""
        
        response = self.llm.complete(prompt)
        
        try:
            # Parse JSON from response
            import json
            json_str = str(response)
            # Extract JSON from response
            start = json_str.find('{')
            end = json_str.rfind('}') + 1
            data = json.loads(json_str[start:end])
            
            return KnowledgeGraph(**data)
        
        except Exception as e:
            print(f"Error parsing: {e}")
            return KnowledgeGraph(entities=[], relationships=[])
    
    def add_documents(self, documents: List[Document]):
        """Add documents to knowledge graph."""
        
        for doc in documents:
            kg = self.extract_entities_and_relations(doc.text)
            
            # Add entities
            for entity in kg.entities:
                self.graph["entities"][entity.name] = entity.entity_type
            
            # Add relationships
            for rel in kg.relationships:
                self.graph["relationships"].append({
                    "source": rel.source,
                    "target": rel.target,
                    "type": rel.relationship_type,
                })
    
    def visualize(self) -> str:
        """Generate DOT format for visualization."""
        
        lines = ["digraph KnowledgeGraph {"]
        
        # Add nodes
        for entity, etype in self.graph["entities"].items():
            lines.append(f'  "{entity}" [label="{entity}", type="{etype}"];')
        
        # Add edges
        for rel in self.graph["relationships"]:
            lines.append(
                f'  "{rel["source"]}" -> "{rel["target"]}" '
                f'[label="{rel["type"]}"];'
            )
        
        lines.append("}")
        return "\n".join(lines)
    
    def save_graph(self, output_file: str):
        """Save knowledge graph."""
        with open(output_file, 'w') as f:
            f.write(self.visualize())

# Usage
if __name__ == "__main__":
    builder = KnowledgeGraphBuilder()
    
    docs = [
        Document(text="Alice works at TechCorp in San Francisco."),
        Document(text="TechCorp was founded in 2010."),
    ]
    
    builder.add_documents(docs)
    
    # Visualize
    graph_dot = builder.visualize()
    print(graph_dot)
    
    # Save
    builder.save_graph("knowledge_graph.dot")

```
{% endraw %}

---

## Multi-Step Reasoning Agent

### Complex Multi-Step Problem Solving

```python
"""
multi_step_reasoning_agent.py
Agent that can reason through complex problems
"""

from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

class MultiStepReasoningAgent:
    """Agent for multi-step reasoning."""
    
    def __init__(self):
        self.llm = OpenAI(model="gpt-4")
        self.reasoning_steps = []
    
    # Step 1: Information gathering
    def research_topic(self, topic: str) -> str:
        """Research a topic."""
        return f"Research data about {topic}..."
    
    # Step 2: Analysis
    def analyze_data(self, data: str) -> str:
        """Analyse gathered data."""
        return f"Analysis of data: {data[:100]}..."
    
    # Step 3: Synthesis
    def synthesize_findings(self, findings: list) -> str:
        """Synthesize findings into conclusions."""
        return f"Synthesized conclusion from {len(findings)} findings"
    
    # Step 4: Decision making
    def make_recommendation(self, conclusions: str) -> str:
        """Make recommendation based on conclusions."""
        return f"Recommendation: {conclusions[:100]}..."
    
    def create_agent(self):
        """Create multi-step reasoning agent."""
        
        tools = [
            FunctionTool.from_defaults(
                fn=self.research_topic,
                name="research",
                description="Research and gather information",
            ),
            FunctionTool.from_defaults(
                fn=self.analyze_data,
                name="analyze",
                description="Analyse gathered data",
            ),
            FunctionTool.from_defaults(
                fn=self.synthesize_findings,
                name="synthesize",
                description="Synthesize findings",
            ),
            FunctionTool.from_defaults(
                fn=self.make_recommendation,
                name="recommend",
                description="Make recommendations",
            ),
        ]
        
        system_prompt = """You are a strategic advisor. When solving problems:
1. First research and gather information
2. Then analyse the data deeply
3. Synthesize findings into key conclusions
4. Finally, make clear recommendations

Use the tools in this order and explain your reasoning at each step."""
        
        agent = ReActAgent.from_tools(
            tools=tools,
            llm=self.llm,
            system_prompt=system_prompt,
            verbose=True,
        )
        
        return agent
    
    def solve(self, problem: str) -> str:
        """Solve complex problem with multi-step reasoning."""
        agent = self.create_agent()
        response = agent.run(problem)
        return response

# Usage
if __name__ == "__main__":
    agent = MultiStepReasoningAgent()
    
    problem = "How should a startup enter the AI market?"
    result = agent.solve(problem)
    
    print(f"Solution:\n{result}")
```

---

## Customer Support Triage Agent

### Intelligent Support Ticket Routing

```python
"""
support_triage_agent.py
Triage customer support tickets
"""

from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
from enum import Enum
from dataclasses import dataclass

class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

class Department(Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    SALES = "sales"
    GENERAL = "general"

@dataclass
class TriagedTicket:
    ticket_id: str
    priority: Priority
    department: Department
    suggested_response: str

class SupportTriageAgent:
    """Triage support tickets using AI."""
    
    def __init__(self):
        self.llm = OpenAI(model="gpt-4")
    
    def assess_urgency(self, ticket: str) -> str:
        """Assess urgency of ticket."""
        prompt = f"""Assess the urgency of this support ticket:
{ticket}

Return: LOW, MEDIUM, HIGH, or CRITICAL"""
        
        response = self.llm.complete(prompt)
        return str(response).strip().upper()
    
    def categorize_issue(self, ticket: str) -> str:
        """Categorize issue type."""
        prompt = f"""Categorize this support issue:
{ticket}

Return one of: BILLING, TECHNICAL, SALES, GENERAL"""
        
        response = self.llm.complete(prompt)
        return str(response).strip().upper()
    
    def draft_response(self, ticket: str) -> str:
        """Draft initial response to customer."""
        prompt = f"""Write a professional initial response to this support ticket:
{ticket}

Keep it brief (2-3 sentences) and empathetic."""
        
        response = self.llm.complete(prompt)
        return str(response).strip()
    
    def create_agent(self):
        """Create triage agent."""
        
        tools = [
            FunctionTool.from_defaults(
                fn=self.assess_urgency,
                name="assess_urgency",
                description="Assess ticket urgency",
            ),
            FunctionTool.from_defaults(
                fn=self.categorize_issue,
                name=" categorize_issue",
                description="Categorize issue type",
            ),
            FunctionTool.from_defaults(
                fn=self.draft_response,
                name="draft_response",
                description="Draft response to customer",
            ),
        ]
        
        system_prompt = """You are a support ticket triage agent. For each ticket:
1. Assess urgency
2. Categorize the issue
3. Draft an appropriate response

Be thorough but efficient."""
        
        agent = ReActAgent.from_tools(
            tools=tools,
            llm=self.llm,
            system_prompt=system_prompt,
            verbose=True,
        )
        
        return agent
    
    def triage(self, ticket_id: str, ticket_text: str) -> TriagedTicket:
        """Triage a support ticket."""
        agent = self.create_agent()
        response = agent.run(f"Triage this ticket:\n{ticket_text}")
        
        # Parse response (simplified)
        urgency = self.assess_urgency(ticket_text)
        category = self.categorize_issue(ticket_text)
        suggestion = self.draft_response(ticket_text)
        
        priority_map = {
            "CRITICAL": Priority.CRITICAL,
            "HIGH": Priority.HIGH,
            "MEDIUM": Priority.MEDIUM,
            "LOW": Priority.LOW,
        }
        
        dept_map = {
            "BILLING": Department.BILLING,
            "TECHNICAL": Department.TECHNICAL,
            "SALES": Department.SALES,
            "GENERAL": Department.GENERAL,
        }
        
        return TriagedTicket(
            ticket_id=ticket_id,
            priority=priority_map.get(urgency, Priority.MEDIUM),
            department=dept_map.get(category, Department.GENERAL),
            suggested_response=suggestion,
        )

# Usage
if __name__ == "__main__":
    agent = SupportTriageAgent()
    
    ticket = """
    I'm unable to access my account and it's been down for 2 hours.
    This is affecting my work. Please help ASAP!
    """
    
    result = agent.triage("TICKET-001", ticket)
    
    print(f"Ticket ID: {result.ticket_id}")
    print(f"Priority: {result.priority.name}")
    print(f"Department: {result.department.value}")
    print(f"Suggested Response: {result.suggested_response}")
```

---

These recipes provide ready-to-use implementations for common LlamaIndex use cases. Each can be customised and extended based on specific requirements.

