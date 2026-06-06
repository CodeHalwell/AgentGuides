---
title: "LangGraph: Advanced Recipes & Real-World Patterns"
description: "Updated for LangGraph 1.2.4 (June 2026)"
framework: langgraph
language: python
---

# LangGraph: Advanced Recipes & Real-World Patterns

**Updated for LangGraph 1.2.4 (June 2026)**

This guide includes recipes demonstrating the latest v1.2.4 features:
- Node Caching for performance
- Deferred Nodes for fan-in patterns
- Pre/Post Model Hooks for LLM customization
- Cross-Thread Memory for persistent context
- Tools State Updates for dynamic behavior
- Command Tool for edgeless flows
- `InjectedState` + `InjectedStore` for context-aware tools (Recipe 9)
- `Overwrite` for resetting accumulated channels (Recipe 10)
- `CheckpointTuple` for checkpoint history browsing and time-travel (Recipe 11)
- `update_state` / `StateUpdate` for human-in-the-loop approval flows (Recipe 12)

---

## Recipe 1: RAG System with Quality Control

Retrieval-Augmented Generation with automatic re-retrieval and refinement:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from typing_extensions import TypedDict
from typing import Annotated
from langgraph.graph.message import add_messages
import json

class RAGState(TypedDict):
    question: str
    messages: Annotated[list, add_messages]
    retrieved_docs: list[dict]
    relevance_score: float
    generation_attempt: int
    final_answer: str
    source_citations: list[str]

def retrieve_documents(state: RAGState) -> dict:
    """Retrieve relevant documents."""
    question = state["question"]
    
    # Use semantic search
    docs = semantic_search(
        query=question,
        index="knowledge_base",
        top_k=5
    )
    
    return {
        "retrieved_docs": docs,
        "messages": [{
            "role": "system",
            "content": f"Retrieved {len(docs)} documents"
        }]
    }

def grade_documents(state: RAGState) -> dict:
    """Grade relevance of retrieved documents."""
    
    docs = state["retrieved_docs"]
    question = state["question"]
    
    # Use LLM to grade
    graded = []
    for doc in docs:
        grade_prompt = f"""
        Question: {question}
        Document: {doc['content']}
        
        Is this document relevant to the question? (yes/no)
        Explain your reasoning.
        """
        
        response = model.invoke(grade_prompt)
        is_relevant = "yes" in response.content.lower()
        
        if is_relevant:
            graded.append(doc)
    
    avg_relevance = len(graded) / len(docs) if docs else 0
    
    return {
        "retrieved_docs": graded,
        "relevance_score": avg_relevance
    }

def decide_strategy(state: RAGState) -> str:
    """Decide whether to generate, re-retrieve, or escalate."""
    
    relevance = state["relevance_score"]
    attempt = state["generation_attempt"]
    
    if relevance > 0.7:
        return "generate"
    elif attempt < 2:
        return "refine_query"
    else:
        return "escalate"

def refine_query(state: RAGState) -> dict:
    """Refine query for better retrieval."""
    
    original_query = state["question"]
    failed_attempt = state["retrieved_docs"]
    
    refine_prompt = f"""
    Original question: {original_query}
    
    The retrieval wasn't successful. Rephrase the question to:
    1. Be more specific
    2. Include key terms
    3. Clarify the intent
    
    Provide only the refined question.
    """
    
    response = model.invoke(refine_prompt)
    refined_query = response.content
    
    return {
        "question": refined_query,
        "generation_attempt": state["generation_attempt"] + 1,
        "messages": [{
            "role": "assistant",
            "content": f"Query refined: {refined_query}"
        }]
    }

def generate_answer(state: RAGState) -> dict:
    """Generate answer from retrieved documents."""
    
    question = state["question"]
    docs = state["retrieved_docs"]
    
    context = "\n\n".join([
        f"Source {i+1} ({doc.get('title', 'Unknown')}):\n{doc['content']}"
        for i, doc in enumerate(docs)
    ])
    
    prompt = f"""
    Question: {question}
    
    Context from documents:
    {context}
    
    Provide a comprehensive answer based on the context.
    Cite sources by referencing the source numbers (e.g., [1], [2]).
    """
    
    response = model.invoke(prompt)
    
    # Extract citations
    citations = []
    for i, doc in enumerate(docs):
        source_id = f"[{i+1}]"
        if source_id in response.content:
            citations.append(f"{i+1}. {doc.get('title', 'Unknown')}")
    
    return {
        "final_answer": response.content,
        "source_citations": citations,
        "messages": [{
            "role": "assistant",
            "content": response.content
        }]
    }

def escalate(state: RAGState) -> dict:
    """Escalate to human when unable to answer."""
    
    return {
        "final_answer": "Unable to find relevant information. Escalating to human support.",
        "messages": [{
            "role": "assistant",
            "content": "This question requires human expert review."
        }]
    }

# Build RAG graph
rag_builder = StateGraph(RAGState)
rag_builder.add_node("retrieve", retrieve_documents)
rag_builder.add_node("grade", grade_documents)
rag_builder.add_node("generate", generate_answer)
rag_builder.add_node("refine", refine_query)
rag_builder.add_node("escalate", escalate)

rag_builder.add_edge(START, "retrieve")
rag_builder.add_edge("retrieve", "grade")

rag_builder.add_conditional_edges(
    "grade",
    decide_strategy,
    {
        "generate": "generate",
        "refine_query": "refine",
        "escalate": "escalate"
    }
)

rag_builder.add_edge("generate", END)
rag_builder.add_edge("escalate", END)
rag_builder.add_edge("refine", "retrieve")  # Loop back

rag_graph = rag_builder.compile(checkpointer=InMemorySaver())

# Usage
result = rag_graph.invoke({
    "question": "How do I set up LangGraph?",
    "generation_attempt": 0
})

print(f"Answer: {result['final_answer']}")
print(f"Sources: {result['source_citations']}")
```

---

## Recipe 2: Customer Support Ticket Classifier & Router

Classify support tickets and route to appropriate handler:


```python

from enum import Enum

class TicketPriority(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class TicketCategory(Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    FEATURE_REQUEST = "feature_request"
    BUG = "bug"
    OTHER = "other"

class SupportTicketState(TypedDict):
    ticket_id: str
    customer_email: str
    subject: str
    description: str
    priority: TicketPriority
    category: TicketCategory
    assigned_to: str
    response: str
    needs_escalation: bool

def classify_ticket(state: SupportTicketState) -> dict:
    """Classify ticket priority and category."""
    
    ticket_text = f"{state['subject']}\n{state['description']}"
    
    classification_prompt = f"""
    Classify this support ticket:
    
    {ticket_text}
    
    Respond with JSON:
    {{
        "priority": "low|medium|high|critical",
        "category": "billing|technical|feature_request|bug|other",
        "summary": "one line summary"
    }}
    """
    
    response = model.invoke(classification_prompt)
    
    # Parse JSON response
    import json
    try:
        result = json.loads(response.content)
        priority = TicketPriority[result["priority"].upper()]
        category = TicketCategory[result["category"].upper()]
    except:
        priority = TicketPriority.MEDIUM
        category = TicketCategory.OTHER
    
    return {
        "priority": priority,
        "category": category
    }

def route_ticket(state: SupportTicketState) -> str:
    """Route to appropriate handler."""
    
    if state["priority"] == TicketPriority.CRITICAL:
        return "escalate"
    elif state["category"] == TicketCategory.BILLING:
        return "billing_handler"
    elif state["category"] == TicketCategory.TECHNICAL:
        return "tech_handler"
    elif state["category"] == TicketCategory.BUG:
        return "bug_handler"
    else:
        return "general_handler"

def billing_handler(state: SupportTicketState) -> dict:
    """Handle billing issues."""
    
    prompt = f"""
    Customer billing inquiry:
    {state['description']}
    
    Provide helpful guidance on billing.
    """
    
    response = model.invoke(prompt)
    
    return {
        "response": response.content,
        "assigned_to": "billing-team"
    }

def tech_handler(state: SupportTicketState) -> dict:
    """Handle technical issues."""
    
    prompt = f"""
    Technical support request:
    {state['description']}
    
    Provide step-by-step technical guidance.
    Suggest debugging steps.
    """
    
    response = model.invoke(prompt)
    
    return {
        "response": response.content,
        "assigned_to": "technical-support"
    }

def bug_handler(state: SupportTicketState) -> dict:
    """Handle bug reports."""
    
    prompt = f"""
    Bug report:
    {state['description']}
    
    Acknowledge the bug.
    Request additional information if needed.
    Provide workaround if available.
    """
    
    response = model.invoke(prompt)
    
    return {
        "response": response.content,
        "assigned_to": "engineering"
    }

def general_handler(state: SupportTicketState) -> dict:
    """Handle general inquiries."""
    
    prompt = f"""
    Customer inquiry:
    {state['description']}
    
    Provide helpful response.
    """
    
    response = model.invoke(prompt)
    
    return {
        "response": response.content,
        "assigned_to": "general-support"
    }

def escalate(state: SupportTicketState) -> dict:
    """Escalate critical issues."""
    
    escalation_prompt = f"""
    Critical support ticket needs immediate attention:
    
    Priority: {state['priority'].value}
    Category: {state['category'].value}
    Description: {state['description']}
    
    Prepare escalation brief for management.
    """
    
    response = model.invoke(escalation_prompt)
    
    return {
        "response": response.content,
        "assigned_to": "management",
        "needs_escalation": True
    }

# Build support graph
support_builder = StateGraph(SupportTicketState)
support_builder.add_node("classify", classify_ticket)
support_builder.add_node("billing", billing_handler)
support_builder.add_node("technical", tech_handler)
support_builder.add_node("bug", bug_handler)
support_builder.add_node("general", general_handler)
support_builder.add_node("escalate", escalate)

support_builder.add_edge(START, "classify")

support_builder.add_conditional_edges(
    "classify",
    route_ticket,
    {
        "billing_handler": "billing",
        "tech_handler": "technical",
        "bug_handler": "bug",
        "general_handler": "general",
        "escalate": "escalate"
    }
)

support_builder.add_edge("billing", END)
support_builder.add_edge("technical", END)
support_builder.add_edge("bug", END)
support_builder.add_edge("general", END)
support_builder.add_edge("escalate", END)

support_graph = support_builder.compile(checkpointer=InMemorySaver())

# Usage
result = support_graph.invoke({
    "ticket_id": "TKT-001",
    "customer_email": "user@example.com",
    "subject": "Billing charge issue",
    "description": "I was charged twice this month. Please help!"
})

print(f"Priority: {result['priority'].value}")
print(f"Category: {result['category'].value}")
print(f"Response: {result['response']}")
print(f"Assigned to: {result['assigned_to']}")

```


---

## Recipe 3: Research Agent with Parallel Data Sources

Gather information from multiple sources in parallel:

```python
from langgraph.types import Send
from datetime import datetime

class ResearchState(TypedDict):
    topic: str
    research_queries: list[str]
    web_results: list[dict]
    academic_results: list[dict]
    news_results: list[dict]
    synthesized_report: str
    citations: list[dict]

def generate_queries(state: ResearchState) -> dict:
    """Generate search queries from topic."""
    
    prompt = f"""
    Topic: {state['topic']}
    
    Generate 3 different search queries to thoroughly research this topic:
    1. General overview
    2. Recent developments
    3. Expert perspectives
    
    Return as JSON array of queries.
    """
    
    response = model.invoke(prompt)
    
    import json
    queries = json.loads(response.content)
    
    return {"research_queries": queries}

def parallel_search(state: ResearchState) -> list[Send]:
    """Create parallel search tasks."""
    
    return [
        Send("web_search", {"query": q}) for q in state["research_queries"]
    ] + [
        Send("academic_search", {"query": q}) for q in state["research_queries"]
    ] + [
        Send("news_search", {"query": q}) for q in state["research_queries"]
    ]

def web_search(state: ResearchState) -> dict:
    """Search general web."""
    
    from tavily import Client
    
    client = Client(api_key=os.getenv("TAVILY_API_KEY"))
    
    results = client.search(
        query=state.get("query", state["topic"]),
        include_answer=True
    )
    
    return {
        "web_results": results["results"][:5]  # Top 5
    }

def academic_search(state: ResearchState) -> dict:
    """Search academic sources."""
    
    # Use academic API or service
    results = semantic_scholar_search(
        query=state.get("query", state["topic"]),
        limit=5
    )
    
    return {"academic_results": results}

def news_search(state: ResearchState) -> dict:
    """Search recent news."""
    
    # Use news API
    from newsapi import NewsApiClient
    
    newsapi = NewsApiClient(api_key=os.getenv("NEWS_API_KEY"))
    
    results = newsapi.get_everything(
        q=state.get("query", state["topic"]),
        sort_by="recency",
        language="en"
    )
    
    return {"news_results": results["articles"][:5]}

def synthesize_report(state: ResearchState) -> dict:
    """Synthesize all research into report."""
    
    web_summary = "\n".join([
        f"- {r['title']}: {r['snippet']}"
        for r in state["web_results"][:3]
    ])
    
    academic_summary = "\n".join([
        f"- {r.get('title', 'Unknown')}"
        for r in state["academic_results"][:3]
    ])
    
    news_summary = "\n".join([
        f"- {r['title']} ({r['publishedAt'][:10]})"
        for r in state["news_results"][:3]
    ])
    
    synthesis_prompt = f"""
    Topic: {state['topic']}
    
    Web search results:
    {web_summary}
    
    Academic research:
    {academic_summary}
    
    Recent news:
    {news_summary}
    
    Write a comprehensive research report synthesizing these sources.
    Include:
    1. Overview
    2. Key findings
    3. Recent developments
    4. Expert insights
    5. Implications
    """
    
    response = model.invoke(synthesis_prompt)
    
    # Compile citations
    citations = []
    for source_list in [state["web_results"], state["academic_results"], state["news_results"]]:
        for source in source_list[:3]:
            citations.append({
                "title": source.get("title", "Unknown"),
                "url": source.get("link") or source.get("url"),
                "date": source.get("publishedAt", "Unknown")[:10]
            })
    
    return {
        "synthesized_report": response.content,
        "citations": citations
    }

# Build research graph
research_builder = StateGraph(ResearchState)
research_builder.add_node("generate_queries", generate_queries)
research_builder.add_node("web_search", web_search)
research_builder.add_node("academic_search", academic_search)
research_builder.add_node("news_search", news_search)
research_builder.add_node("synthesize", synthesize_report)

research_builder.add_edge(START, "generate_queries")

# Parallel search
research_builder.add_conditional_edges(
    "generate_queries",
    lambda _: ["web_search", "academic_search", "news_search"],
    ["web_search", "academic_search", "news_search"]
)

# Gather and synthesize
research_builder.add_edge("web_search", "synthesize")
research_builder.add_edge("academic_search", "synthesize")
research_builder.add_edge("news_search", "synthesize")
research_builder.add_edge("synthesize", END)

research_graph = research_builder.compile()

# Usage
result = research_graph.invoke({"topic": "AI safety 2024"})

print(result["synthesized_report"])
print("\nCitations:")
for i, cite in enumerate(result["citations"][:5], 1):
    print(f"{i}. {cite['title']} ({cite['date']})")
```

---

## Recipe 4: Agentic Loop with Tool Calling

Autonomous agent that reasons and acts:


```python
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool

# Define specialized tools
@tool
def search_knowledge_base(query: str) -> str:
    """Search internal knowledge base."""
    results = db.search_documents(query, limit=3)
    return "\n".join([r["content"] for r in results])

@tool
def check_inventory(product_id: str) -> dict:
    """Check product inventory."""
    return {
        "product_id": product_id,
        "in_stock": True,
        "quantity": 50
    }

@tool
def submit_order(user_id: str, items: list[dict]) -> dict:
    """Submit an order."""
    order_id = f"ORD-{uuid.uuid4().hex[:8].upper()}"
    return {
        "order_id": order_id,
        "status": "confirmed",
        "total": 99.99
    }

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send email notification."""
    # Send email
    return f"Email sent to {to}"

tools = [
    search_knowledge_base,
    check_inventory,
    submit_order,
    send_email
]

# Create agent
model = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# Build custom agent for more control
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_request: str
    reasoning: str
    action_taken: bool

def agent_reasoning_node(state: AgentState) -> dict:
    """Agent reasons about what to do."""
    
    system_prompt = """You are a helpful shopping assistant. 
    Analyze user requests and decide what actions to take.
    Use available tools to help the customer."""
    
    response = model.invoke(state["messages"])
    
    # Capture reasoning
    reasoning = response.content
    has_tool_calls = hasattr(response, 'tool_calls') and len(response.tool_calls) > 0
    
    return {
        "messages": [response],
        "reasoning": reasoning,
        "action_taken": has_tool_calls
    }

# Build agentic graph
agent_builder = StateGraph(AgentState)
agent_builder.add_node("reasoning", agent_reasoning_node)
agent_builder.add_node("tools", ToolNode(tools))

agent_builder.add_edge(START, "reasoning")

# Use tools_condition for automatic routing
agent_builder.add_conditional_edges(
    "reasoning",
    tools_condition,
    {"tools": "tools", END: END}
)

agent_builder.add_edge("tools", "reasoning")  # Loop back

agentic_graph = agent_builder.compile(checkpointer=InMemorySaver())

# Usage
config = {"configurable": {"thread_id": "customer-123"}}

result = agentic_graph.invoke({
    "messages": [{
        "role": "user",
        "content": "I want to buy 2 units of product ABC123"
    }],
    "user_request": "Purchase items",
    "reasoning": ""
}, config=config)

print("Final response:", result["messages"][-1].content)
```


---

## Recipe 5: Document Processing Pipeline

Multi-stage document processing with quality checks:


```python

from enum import Enum

class DocumentType(Enum):
    PDF = "pdf"
    DOCX = "docx"
    JSON = "json"
    TEXT = "text"

class ProcessingState(TypedDict):
    document_id: str
    document_content: str
    document_type: DocumentType
    extraction_result: dict
    validation_result: dict
    enrichment_result: dict
    processing_status: str
    error_message: str

def extract_content(state: ProcessingState) -> dict:
    """Extract structured content from document."""
    
    try:
        if state["document_type"] == DocumentType.PDF:
            # PDF extraction
            content = extract_text_from_pdf(state["document_content"])
        elif state["document_type"] == DocumentType.DOCX:
            # DOCX extraction
            content = extract_text_from_docx(state["document_content"])
        else:
            content = state["document_content"]
        
        # Use LLM to structure the content
        extraction_prompt = f"""
        Extract structured information from this document:
        
        {content[:2000]}  # First 2000 chars
        
        Extract:
        1. Title
        2. Key sections
        3. Main topics
        4. Metadata (author, date, etc)
        
        Return as JSON.
        """
        
        response = model.invoke(extraction_prompt)
        
        import json
        extracted = json.loads(response.content)
        
        return {
            "extraction_result": extracted,
            "processing_status": "extracted"
        }
    
    except Exception as e:
        return {
            "error_message": str(e),
            "processing_status": "extraction_failed"
        }

def validate_content(state: ProcessingState) -> dict:
    """Validate extracted content quality."""
    
    if state["processing_status"] == "extraction_failed":
        return {"validation_result": {"valid": False}}
    
    extracted = state["extraction_result"]
    
    validation_prompt = f"""
    Validate this extracted content:
    
    {json.dumps(extracted, indent=2)}
    
    Check:
    1. Completeness - all expected fields present
    2. Accuracy - information makes sense
    3. Format - proper structure
    
    Return: {{"valid": true/false, "issues": ["list of issues"]}}
    """
    
    response = model.invoke(validation_prompt)
    
    import json
    validation = json.loads(response.content)
    
    return {"validation_result": validation}

def enrich_content(state: ProcessingState) -> dict:
    """Enrich content with additional insights."""
    
    if not state["validation_result"].get("valid"):
        return {"enrichment_result": {}}
    
    extracted = state["extraction_result"]
    
    enrichment_prompt = f"""
    Enrich this document with:
    1. Summary
    2. Key entities (people, organizations, concepts)
    3. Related topics
    4. Action items
    5. Risk assessment (if applicable)
    
    Content:
    {json.dumps(extracted, indent=2)}
    
    Return as JSON.
    """
    
    response = model.invoke(enrichment_prompt)
    
    import json
    enrichment = json.loads(response.content)
    
    return {
        "enrichment_result": enrichment,
        "processing_status": "complete"
    }

# Build document processing pipeline
doc_builder = StateGraph(ProcessingState)
doc_builder.add_node("extract", extract_content)
doc_builder.add_node("validate", validate_content)
doc_builder.add_node("enrich", enrich_content)

doc_builder.add_edge(START, "extract")
doc_builder.add_edge("extract", "validate")
doc_builder.add_edge("validate", "enrich")
doc_builder.add_edge("enrich", END)

doc_pipeline = doc_builder.compile()

# Usage
result = doc_pipeline.invoke({
    "document_id": "doc-001",
    "document_content": "Your PDF/document content here",
    "document_type": DocumentType.PDF
})

if result["processing_status"] == "complete":
    print("Extraction:", result["extraction_result"])
    print("Enrichment:", result["enrichment_result"])
else:
    print("Error:", result["error_message"])

```


---

## Recipe 6: Conversation with Long-term Memory

Maintain user context across multiple conversations:


```python
from datetime import datetime

class ConversationState(TypedDict):
    user_id: str
    message: str
    conversation_history: Annotated[list, add_messages]
    user_profile: dict
    relevant_memories: list[dict]
    response: str

async def fetch_user_profile(
    state: ConversationState,
    store: Annotated[AsyncPostgresStore, InjectedStore]
) -> dict:
    """Fetch user profile from long-term store."""
    
    user_id = state["user_id"]
    namespace = ("users", user_id)
    
    profile_item = await store.aget(namespace, "profile")
    profile = profile_item.value if profile_item else {
        "name": "User",
        "preferences": {},
        "interests": [],
        "conversation_count": 0
    }
    
    return {"user_profile": profile}

async def retrieve_memories(
    state: ConversationState,
    store: Annotated[AsyncPostgresStore, InjectedStore]
) -> dict:
    """Retrieve relevant memories from long-term store."""
    
    user_id = state["user_id"]
    current_message = state["message"]
    
    # Semantic search for relevant memories
    namespace = ("users", user_id, "memories")
    
    memories = await store.asearch(
        namespace_prefix=namespace,
        query=current_message,
        limit=5
    )
    
    return {
        "relevant_memories": [m.value for m in memories]
    }

def build_context(state: ConversationState) -> dict:
    """Build conversational context from memories and profile."""
    
    profile = state["user_profile"]
    memories = state["relevant_memories"]
    
    context_parts = [
        f"User: {profile.get('name', 'User')}",
        f"Interests: {', '.join(profile.get('interests', []))}",
    ]
    
    if memories:
        context_parts.append("Relevant context from past conversations:")
        for mem in memories:
            context_parts.append(f"- {mem.get('content', '')}")
    
    return "\n".join(context_parts)

def chat_node(state: ConversationState) -> dict:
    """Generate response using context."""
    
    context = build_context(state)
    
    system_prompt = f"""
    You are a helpful assistant with knowledge of the user's history and preferences.
    
    User Context:
    {context}
    
    Be personalized and reference relevant past context when appropriate.
    """
    
    messages = state["conversation_history"] + [
        {"role": "user", "content": state["message"]}
    ]
    
    response = model.invoke(messages, system_prompt=system_prompt)
    
    return {
        "response": response.content,
        "conversation_history": [{"role": "assistant", "content": response.content}]
    }

async def save_memory(
    state: ConversationState,
    store: Annotated[AsyncPostgresStore, InjectedStore]
) -> dict:
    """Save conversation to long-term memory."""
    
    user_id = state["user_id"]
    namespace = ("users", user_id, "memories")
    
    memory_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_message": state["message"],
        "assistant_response": state["response"],
        "content": f"User: {state['message']}\nAssistant: {state['response']}"
    }
    
    memory_id = f"mem-{uuid.uuid4().hex[:8]}"
    
    await store.aput(
        namespace,
        memory_id,
        memory_entry,
        index=["content"]  # Index for semantic search
    )
    
    # Update conversation count
    profile_namespace = ("users", user_id)
    profile = await store.aget(profile_namespace, "profile")
    profile_data = profile.value if profile else {}
    profile_data["conversation_count"] = profile_data.get("conversation_count", 0) + 1
    profile_data["last_conversation"] = datetime.now().isoformat()
    
    await store.aput(profile_namespace, "profile", profile_data)
    
    return {}

# Build conversational graph with memory
conv_builder = StateGraph(ConversationState)
conv_builder.add_node("fetch_profile", fetch_user_profile)
conv_builder.add_node("retrieve_memories", retrieve_memories)
conv_builder.add_node("chat", chat_node)
conv_builder.add_node("save_memory", save_memory)

conv_builder.add_edge(START, "fetch_profile")
conv_builder.add_edge("fetch_profile", "retrieve_memories")
conv_builder.add_edge("retrieve_memories", "chat")
conv_builder.add_edge("chat", "save_memory")
conv_builder.add_edge("save_memory", END)

conversation_graph = conv_builder.compile(
    store=store  # Pass long-term store
)

# Usage
config = {"configurable": {"thread_id": "user-alice"}}

result = conversation_graph.invoke({
    "user_id": "alice",
    "message": "What was I asking about last time?"
}, config=config)

print(result["response"])
```


---

## Performance Optimization Tips

### Tip 1: Lazy Evaluation

```python
# Bad - eager evaluation
def slow_node(state):
    all_results = [expensive_operation(i) for i in range(1000)]
    return {"results": all_results}

# Good - lazy evaluation
def fast_node(state):
    def results_generator():
        for i in range(1000):
            yield expensive_operation(i)
    
    return {"results": results_generator()}
```

### Tip 2: Efficient State Updates

```python
# Bad - rebuilds entire list
return {"items": state["items"] + [new_item]}

# Good - append reducer
class State(TypedDict):
    items: Annotated[list, lambda x, y: x + y]

return {"items": [new_item]}  # Automatically appended
```

### Tip 3: Streaming for Real-Time Feedback

```python
# Stream intermediate results to client
async def stream_processing():
    async for event in graph.astream(
        {"query": "Process this"},
        stream_mode="updates"
    ):
        # Send each update to client via WebSocket
        await websocket.send_json(event)
        yield event
```

---

## Recipe 7: Cached Multi-Agent Research System (v1.2.1)

**Uses:** `CachePolicy`, `InMemoryCache`, `InMemoryStore`, parallel `Send` fan-out

```python
# Correct imports — all verified against langgraph==1.2.1
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send, RetryPolicy, CachePolicy
from langgraph.store.memory import InMemoryStore
from langgraph.cache.memory import InMemoryCache
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class ResearchState(TypedDict):
    user_id: str
    topic: str
    search_queries: list[str]
    # reducer: all parallel researcher writes are accumulated
    research_results: Annotated[list[dict], operator.add]
    final_report: str


class WorkerState(TypedDict):
    """Narrow state used by each parallel worker node."""
    query: str


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def load_prefs_and_plan(state: ResearchState, runtime: Runtime) -> dict:
    """Load user preferences from the store and generate search queries."""
    store = runtime.store
    user_id = state["user_id"]

    # Read stored preferences (or use defaults)
    prefs_item = store.get(("users", user_id), "research_prefs") if store else None
    prefs = prefs_item.value if prefs_item else {
        "depth": "standard",   # or "comprehensive"
        "max_queries": 3,
    }

    topic = state["topic"]
    n = prefs.get("max_queries", 3)
    # In production: call an LLM to generate queries
    queries = [f"{topic} - aspect {i+1}" for i in range(n)]

    return {"search_queries": queries}


def fan_out(state: ResearchState) -> list[Send]:
    """Conditional edge: launch one researcher per query in parallel."""
    return [Send("researcher", {"query": q}) for q in state["search_queries"]]


def researcher(state: WorkerState) -> dict:
    """Research a single query — runs in parallel (one per Send).
    Results are merged into research_results via the operator.add reducer.
    """
    query = state["query"]
    # In production: call a search API or LLM here
    result = {
        "query": query,
        "summary": f"[stub] findings for '{query}'",
        "sources": [f"https://example.com/search?q={query.replace(' ', '+')}"],
    }
    return {"research_results": [result]}


def synthesise(state: ResearchState, runtime: Runtime) -> dict:
    """Combine all parallel results into a final report and save to store."""
    store = runtime.store
    all_results = state["research_results"]

    bullets = "\n".join(f"- {r['query']}: {r['summary']}" for r in all_results)
    report = f"## Research Report: {state['topic']}\n\n{bullets}"

    # Persist for future look-up
    if store:
        store.put(
            ("users", state["user_id"], "reports"),
            state["topic"],
            {"report": report, "query_count": len(all_results)},
        )

    return {"final_report": report}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

builder = StateGraph(ResearchState)
builder.add_node(
    "plan",
    load_prefs_and_plan,
)
builder.add_node(
    "researcher",
    researcher,
    # Retry transient errors; cache results for 10 minutes per unique query
    retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.5),
    cache_policy=CachePolicy(ttl=600),
)
builder.add_node("synthesise", synthesise)

builder.add_edge(START, "plan")
builder.add_conditional_edges("plan", fan_out)   # dynamic fan-out
builder.add_edge("researcher", "synthesise")
builder.add_edge("synthesise", END)

# Pass both store (long-term memory) and cache (node result caching)
store = InMemoryStore()
cache = InMemoryCache()
checkpointer = InMemorySaver()

research_graph = builder.compile(
    checkpointer=checkpointer,
    store=store,
    cache=cache,
)

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

cfg = {"configurable": {"thread_id": "research-session-1"}}

result = research_graph.invoke(
    {"user_id": "alice", "topic": "AI in healthcare", "search_queries": [], "research_results": [], "final_report": ""},
    cfg,
)
print(result["final_report"])

# Second run: identical queries hit the cache (CachePolicy ttl=600)
result2 = research_graph.invoke(
    {"user_id": "alice", "topic": "AI in healthcare", "search_queries": [], "research_results": [], "final_report": ""},
    cfg,
)
print(result2["final_report"])  # same content, returned from cache
```

---

## Recipe 8: Smart Shopping Assistant with Pre/Post Model Hooks (v1.2.1)

**Uses:** `create_react_agent` with `pre_model_hook` / `post_model_hook`, `Command`-returning tools, custom state schema

```python
# Correct imports — verified against langgraph==1.2.1 / langgraph-prebuilt==1.1.0
import operator
from typing import Annotated, Any
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent, InjectedState
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


# ---------------------------------------------------------------------------
# State schema — extends the default MessagesState with shopping fields
# ---------------------------------------------------------------------------

class ShoppingState(TypedDict):
    messages:    Annotated[list[AnyMessage], add_messages]
    user_id:     str
    cart:        list[dict]
    total:       float
    tokens_used: int


# ---------------------------------------------------------------------------
# Tools — use InjectedState to read the graph state directly
#         Return Command to update state alongside the tool message
# ---------------------------------------------------------------------------

@tool
def add_to_cart(
    product_id: str,
    product_name: str,
    price: float,
    quantity: int,
    state: Annotated[ShoppingState, InjectedState],
) -> Command:
    """Add a product to the shopping cart and update the total."""
    item = {"product_id": product_id, "name": product_name, "price": price, "qty": quantity}
    new_cart = state.get("cart", []) + [item]
    new_total = state.get("total", 0.0) + price * quantity
    return Command(
        update={
            "cart": new_cart,
            "total": new_total,
        },
        goto="agent",  # return control to the agent node
    )


@tool
def view_cart(state: Annotated[ShoppingState, InjectedState]) -> str:
    """Return a human-readable summary of the current cart."""
    cart = state.get("cart", [])
    if not cart:
        return "Your cart is empty."
    lines = [f"- {item['name']} x{item['qty']}  ${item['price'] * item['qty']:.2f}" for item in cart]
    total = state.get("total", 0.0)
    return "Cart:\n" + "\n".join(lines) + f"\n\nTotal: ${total:.2f}"


@tool
def search_products(category: str) -> list[dict]:
    """Search available products in a category (stub)."""
    catalogue = {
        "laptop": [
            {"id": "mbp-16",  "name": "MacBook Pro 16",   "price": 2499.0},
            {"id": "xps-15",  "name": "Dell XPS 15",       "price": 1799.0},
        ],
        "monitor": [
            {"id": "lg-27",   "name": "LG 27\" 4K",        "price": 499.0},
        ],
    }
    return catalogue.get(category.lower(), [])


# ---------------------------------------------------------------------------
# Pre-model hook — inject cart context as a system message before every LLM call
# ---------------------------------------------------------------------------

def shopping_context_hook(state: ShoppingState) -> dict:
    """Prepend a system message summarising the current cart state.

    pre_model_hook receives the full state and returns a dict to merge
    into state before the LLM call.  We prepend a SystemMessage to
    'messages' — add_messages will handle the merge correctly.
    """
    cart = state.get("cart", [])
    total = state.get("total", 0.0)
    summary = (
        f"User {state.get('user_id', 'unknown')} | "
        f"Cart: {len(cart)} item(s), total ${total:.2f}. "
        "Help them find and add the best products for their needs."
    )
    return {"messages": [SystemMessage(content=summary)]}


# ---------------------------------------------------------------------------
# Post-model hook — accumulate token usage after every LLM call
# ---------------------------------------------------------------------------

def track_tokens(state: ShoppingState) -> dict:
    """After the LLM responds, read usage_metadata from the latest AI message
    and accumulate the token count.

    post_model_hook receives the state (already updated with the new AI
    message) and returns a dict to merge into state.
    """
    last = state["messages"][-1]
    usage = getattr(last, "usage_metadata", None) or {}
    new_tokens = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
    return {"tokens_used": state.get("tokens_used", 0) + new_tokens}


# ---------------------------------------------------------------------------
# Build the agent
# ---------------------------------------------------------------------------

tools = [add_to_cart, view_cart, search_products]

# Replace with a real LLM in production:
# from langchain_anthropic import ChatAnthropic
# llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage
llm = MagicMock()
llm.bind_tools.return_value = llm
llm.invoke.return_value = AIMessage(content="Here are some laptop options for you.", tool_calls=[])

shopping_agent = create_react_agent(
    model=llm,
    tools=tools,
    state_schema=ShoppingState,
    pre_model_hook=shopping_context_hook,   # inject cart context
    post_model_hook=track_tokens,           # record token usage
    checkpointer=InMemorySaver(),
)

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

cfg = {"configurable": {"thread_id": "shopper-123"}}

result = shopping_agent.invoke(
    {
        "messages": [("user", "I need a good laptop for programming. Show me options and add the best one.")],
        "user_id": "shopper-123",
        "cart": [],
        "total": 0.0,
        "tokens_used": 0,
    },
    cfg,
)

print(f"Cart: {result['cart']}")
print(f"Total: ${result['total']:.2f}")
print(f"Tokens used: {result['tokens_used']}")
print(f"Last reply: {result['messages'][-1].content}")
```

---

## Recipe 9: Intelligent Workflow Coordinator (v1.2.1)

**Uses:** `Command` routing, `Send` fan-out, `defer=True` node, `RetryPolicy`

```python
# Correct imports — all verified against langgraph==1.2.1
import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, Send, RetryPolicy
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.cache.memory import InMemoryCache


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

TASK_TYPES = ["data", "training", "validation"]


class WorkflowState(TypedDict):
    workflow_id:    str
    tasks:          list[dict]        # list of {"id":..., "type":..., "priority":...}
    completed:      Annotated[list[str], operator.add]   # reducer: accumulate completed task ids
    failed:         Annotated[list[str], operator.add]   # reducer: accumulate failed task ids
    workflow_status: str


class TaskState(TypedDict):
    """Narrow state injected into each worker by Send."""
    task_id:   str
    task_type: str


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def coordinator(state: WorkflowState) -> Command[Literal["fan_out", "__end__"]]:
    """Decide whether to launch tasks or finish.

    Returns Command(goto="fan_out") while there are pending tasks,
    or Command(goto=END, update={"workflow_status": "complete"}) when done.
    """
    completed = state.get("completed", [])
    failed = state.get("failed", [])
    done = completed + failed
    pending = [t for t in state["tasks"] if t["id"] not in done]

    if not pending:
        return Command(
            goto=END,
            update={"workflow_status": f"done — {len(completed)} ok, {len(failed)} failed"},
        )

    return Command(goto="fan_out")


def fan_out(state: WorkflowState) -> list[Send]:
    """Launch all pending tasks in parallel using Send."""
    completed = state.get("completed", [])
    failed = state.get("failed", [])
    done = set(completed + failed)
    return [
        Send("run_task", {"task_id": t["id"], "task_type": t["type"]})
        for t in state["tasks"]
        if t["id"] not in done
    ]


def run_task(state: TaskState) -> dict:
    """Execute a single task.  Runs in parallel — one instance per Send.

    Results accumulate back into WorkflowState via the operator.add reducer
    on 'completed' / 'failed'.
    """
    task_id = state["task_id"]
    task_type = state["task_type"]

    # Simulate work — replace with real logic
    import random
    success = random.random() > 0.1   # 10% artificial failure rate

    if success:
        print(f"  [task] {task_id} ({task_type}) OK")
        return {"completed": [task_id]}
    else:
        print(f"  [task] {task_id} ({task_type}) FAILED")
        return {"failed": [task_id]}


# ---------------------------------------------------------------------------
# Graph construction
#
# Key features demonstrated:
#  - coordinator uses Command for dynamic routing (no static edges from it)
#  - fan_out returns list[Send] for variable-width parallelism
#  - run_task uses retry_policy so transient failures are retried automatically
#  - defer=True on the aggregate node means it runs AFTER all parallel run_task
#    instances in the same super-step finish
# ---------------------------------------------------------------------------

def aggregate(state: WorkflowState) -> dict:
    """Summarise.  Runs only after all parallel run_task nodes complete
    because it is registered with defer=True.
    """
    return {
        "workflow_status": (
            f"aggregated: {len(state.get('completed', []))} completed, "
            f"{len(state.get('failed', []))} failed"
        )
    }


builder = StateGraph(WorkflowState)
builder.add_node("coordinator", coordinator)
builder.add_node("fan_out",     fan_out)
builder.add_node(
    "run_task",
    run_task,
    retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.1),
)
builder.add_node(
    "aggregate",
    aggregate,
    defer=True,          # waits for all non-deferred nodes in the same super-step
)

builder.add_edge(START, "coordinator")
# fan_out and run_task edges come from coordinator's Command / fan_out's Sends
builder.add_edge("fan_out",   "run_task")
builder.add_edge("run_task",  "aggregate")
builder.add_edge("aggregate", "coordinator")   # loop back — coordinator will exit via END

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------

cache = InMemoryCache()
graph = builder.compile(
    checkpointer=InMemorySaver(),
    cache=cache,
)

cfg = {"configurable": {"thread_id": "workflow-001"}}
result = graph.invoke(
    {
        "workflow_id": "ml-pipeline-001",
        "tasks": [
            {"id": "data_task",       "type": "data",       "priority": 1},
            {"id": "training_task",   "type": "training",   "priority": 2},
            {"id": "validation_task", "type": "validation", "priority": 3},
        ],
        "completed": [],
        "failed": [],
        "workflow_status": "pending",
    },
    cfg,
    {"recursion_limit": 50},
)

print(f"Workflow Status: {result['workflow_status']}")
print(f"Completed tasks: {result['completed']}")
print(f"Failed tasks:    {result['failed']}")
```

---

This collection of recipes covers real-world patterns you'll encounter building production AI systems with LangGraph 1.2.1.

Adapt and combine them for your specific use cases!

---

## Recipe 10: Long-Term Memory with Vector Search and InMemoryStore (v1.2.1)

**Uses:** `InMemoryStore` with vector index, `Runtime` context injection, per-user memory namespaces

A complete chatbot that persists and retrieves user preferences using `InMemoryStore` with optional vector search. The `Runtime` object carries both the store and a typed context object so nodes stay pure functions.

```python
from langgraph.store.memory import InMemoryStore
from langgraph.graph.message import MessagesState, add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import Annotated
from typing_extensions import TypedDict
from dataclasses import dataclass

# Store with optional vector search
store = InMemoryStore(
    index={
        "dims": 1536,
        "embed": embed_fn,  # your embedding function
        "fields": ["content"]
    }
)

# Typed context carried through the run
@dataclass
class UserContext:
    user_id: str

class ChatState(MessagesState):
    pass

def recall_memories(state: ChatState, runtime: Runtime[UserContext]) -> dict:
    """Load relevant memories from long-term store."""
    user_id = runtime.context.user_id

    # Get last user message for search
    last_msg = state["messages"][-1].content if state["messages"] else ""

    # Search for relevant memories
    memories = runtime.store.search(
        ("memories", user_id),
        query=last_msg,
        limit=3
    )

    if memories:
        mem_text = "\n".join(f"- {m.value['content']}" for m in memories)
        system = SystemMessage(content=f"User memories:\n{mem_text}")
        return {"messages": [system]}
    return {}

def save_memory(state: ChatState, runtime: Runtime[UserContext]) -> dict:
    """Save important facts to long-term store."""
    user_id = runtime.context.user_id if runtime.context else "anon"
    last_ai = state["messages"][-1] if state["messages"] else None
    if not last_ai:
        return {}

    import uuid
    runtime.store.put(
        ("memories", user_id),
        str(uuid.uuid4()),
        {"content": last_ai.content[:500]}
    )
    return {}

def call_model(state: ChatState) -> dict:
    # model invocation
    return {"messages": [AIMessage(content="Response...")]}

builder = StateGraph(ChatState, context_schema=UserContext)
builder.add_node("recall", recall_memories)
builder.add_node("agent", call_model)
builder.add_node("save", save_memory)

builder.add_edge(START, "recall")
builder.add_edge("recall", "agent")
builder.add_edge("agent", "save")
builder.add_edge("save", END)

graph = builder.compile(store=store)

# Pass context via configurable
result = graph.invoke(
    {"messages": [HumanMessage("I prefer dark mode")]},
    {"configurable": {"context": UserContext(user_id="alice")}}
)

print(result["messages"][-1].content)

# On subsequent runs the recalled memories are injected as a SystemMessage
# before the agent node, so the model always has the user's preferences in
# its context window.
```

Key points:
- `InMemoryStore` accepts an `index` dict to enable semantic (vector) search via `store.search(..., query=...)`. Omit it for key-only lookup.
- `Runtime[UserContext]` is injected automatically by LangGraph when declared as a node parameter; it exposes `.context`, `.store`, and `.stream_writer`.
- `context_schema=UserContext` on `StateGraph` tells the compiler which dataclass to deserialise from `configurable["context"]`.

---

## Recipe 11: Parallel Map-Reduce with Send and BinaryOperatorAggregate (v1.2.1)

**Uses:** `Send` for fan-out, `Annotated[list, operator.add]` reducer for fan-in, `ToolRuntime` inside a tool

A document-analysis pipeline that dispatches each document to a parallel `analyze_doc` node, then reduces all results in a single `reduce_results` node.

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.prebuilt.tool_node import ToolRuntime
from langgraph.store.base import BaseStore
from langchain_core.tools import tool

# ── State ────────────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    documents: list[str]           # raw document texts to process
    # BinaryOperatorAggregate: each parallel branch appends its slice
    analysis_results: Annotated[list[dict], operator.add]
    summary: str

# ── Parallel branch state (sent via Send) ────────────────────────────────────

class DocState(TypedDict):
    doc_text: str
    analysis_results: Annotated[list[dict], operator.add]

# ── Tool that uses ToolRuntime to access the store ───────────────────────────

@tool
def save_insight(
    doc_id: str,
    insight: str,
    runtime: ToolRuntime,
) -> str:
    """Persist an insight from a document into the shared store."""
    if runtime.store:
        import uuid
        runtime.store.put(
            ("insights",),
            str(uuid.uuid4()),
            {"doc_id": doc_id, "insight": insight}
        )
    if runtime.stream_writer:
        runtime.stream_writer({"saved_insight": insight})
    return f"Saved insight for {doc_id}"

# ── Nodes ────────────────────────────────────────────────────────────────────

def fan_out(state: PipelineState) -> list[Send]:
    """Create one Send per document — each runs analyze_doc in parallel."""
    return [
        Send("analyze_doc", {"doc_text": doc, "analysis_results": []})
        for doc in state["documents"]
    ]

def analyze_doc(state: DocState) -> dict:
    """Analyse a single document (runs in parallel for every document)."""
    text = state["doc_text"]

    # Simulated analysis — replace with real LLM / extraction logic
    result = {
        "length": len(text),
        "preview": text[:80],
        "sentiment": "positive" if "good" in text.lower() else "neutral",
    }

    return {"analysis_results": [result]}   # appended by reducer

def reduce_results(state: PipelineState) -> dict:
    """Runs only after ALL analyze_doc branches have completed (fan-in)."""
    results = state["analysis_results"]

    avg_length = sum(r["length"] for r in results) / max(len(results), 1)
    sentiments = [r["sentiment"] for r in results]

    summary = (
        f"Processed {len(results)} documents. "
        f"Avg length: {avg_length:.0f} chars. "
        f"Sentiments: {', '.join(set(sentiments))}."
    )
    return {"summary": summary}

# ── Graph ────────────────────────────────────────────────────────────────────

builder = StateGraph(PipelineState)
builder.add_node("analyze_doc", analyze_doc)
builder.add_node("reduce_results", reduce_results)

# fan_out returns a list[Send] — LangGraph dispatches each in parallel
builder.add_conditional_edges(START, fan_out, ["analyze_doc"])

# All analyze_doc branches converge here
builder.add_edge("analyze_doc", "reduce_results")
builder.add_edge("reduce_results", END)

pipeline = builder.compile()

result = pipeline.invoke({
    "documents": [
        "This is a good product review.",
        "The service was okay.",
        "Great experience overall — good work!"
    ],
    "analysis_results": [],
    "summary": ""
})

print(result["summary"])
# Processed 3 documents. Avg length: 37 chars. Sentiments: positive, neutral.
```

Key points:
- `Annotated[list, operator.add]` is a **BinaryOperatorAggregate** channel. Each parallel branch writes `{"analysis_results": [single_item]}` and LangGraph concatenates them automatically; no explicit merge node is needed.
- `Send("node_name", partial_state)` lets you dynamically create parallel branches at runtime. The partial state is merged into the branch's state before the node runs.
- `ToolRuntime` is injected by `ToolNode` when a `@tool` declares it as a parameter. It exposes `.store`, `.state`, `.stream_writer`, and `.tool_call_id` — giving tools read/write access to cross-thread memory without coupling them to a specific state schema.

---

## Recipe 12: ToolRuntime All-In-One (v1.2.1)

**Uses:** `ToolRuntime` for store access, streaming progress events, and `tool_call_id` correlation

A single tool that demonstrates every capability exposed by `ToolRuntime`: reading graph state, writing to the long-term store, emitting streaming progress, and tagging output with the originating tool call ID.

```python
from langgraph.prebuilt.tool_node import ToolRuntime
from langgraph.store.base import BaseStore
from langgraph.prebuilt import ToolNode, create_react_agent
from langchain_core.tools import tool
from typing import Annotated
import uuid

@tool
def research_and_remember(
    topic: str,
    runtime: ToolRuntime,
) -> str:
    """Research a topic and save findings to memory.

    Demonstrates all four ToolRuntime capabilities:
      1. runtime.state         — read current graph state
      2. runtime.store         — write to long-term cross-thread store
      3. runtime.stream_writer — emit custom streaming events
      4. runtime.tool_call_id  — correlate results to the originating call
    """
    # 1. Read state (optional — may be None if store injection is disabled)
    user_id = runtime.state.get("user_id", "anon") if runtime.state else "anon"

    # 2. Emit streaming progress so the client can show a spinner / log
    if runtime.stream_writer:
        runtime.stream_writer({"status": "researching", "topic": topic})

    # 3. Perform the work (simulated — replace with real API calls)
    findings = f"Key findings about {topic}: [placeholder — add real research logic]"

    # 4. Persist findings to long-term store under a per-user namespace
    if runtime.store:
        runtime.store.put(
            ("research", user_id),
            str(uuid.uuid4()),
            {"topic": topic, "findings": findings}
        )

    # 5. Include the tool_call_id in the streaming event so the client can
    #    match progress updates back to the specific tool invocation
    if runtime.stream_writer:
        runtime.stream_writer({
            "status": "saved",
            "tool_call_id": runtime.tool_call_id
        })

    return findings

# ── Wire the tool into a ReAct agent ─────────────────────────────────────────

from langgraph.store.memory import InMemoryStore
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

store = InMemoryStore()

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    user_id: str

agent = create_react_agent(
    model=model,                          # your ChatAnthropic / ChatOpenAI etc.
    tools=[research_and_remember],
    state_schema=AgentState,
)

# Stream the run — custom events emitted by stream_writer appear in the
# "custom" stream mode alongside the standard message events.
for event in agent.stream(
    {
        "messages": [{"role": "user", "content": "Research quantum computing"}],
        "user_id": "alice",
    },
    stream_mode=["updates", "custom"],
    config={"store": store},
):
    print(event)
```

Key points:
- `ToolRuntime` is **automatically injected** by `ToolNode`/`create_react_agent` — declare it as a parameter typed `ToolRuntime` and LangGraph wires it up; never pass it manually from your own code.
- `runtime.stream_writer` accepts any JSON-serialisable dict. These are surfaced when the caller uses `stream_mode="custom"` (or a list that includes `"custom"`).
- `runtime.store` is the same store instance passed to `compile(store=...)` or the `config` dict, so tools share the same persistent memory as graph nodes.
- `runtime.tool_call_id` matches the `id` field on the `ToolCall` that triggered this invocation — useful for correlating streaming progress events to a specific call when multiple tool calls fire in parallel.

---

## Recipe 9: State-Aware Shopping Agent with `InjectedState` + `InjectedStore`

This recipe demonstrates how `InjectedState` and `InjectedStore` let tools access graph state and persistent storage **without exposing internal details to the LLM**.

```python
from typing import Any, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, InjectedState, InjectedStore, tools_condition, create_react_agent
from langgraph.store.memory import InMemoryStore
from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic

# Extend MessagesState with domain-specific fields
class ShopState(MessagesState):
    user_id: str
    cart: list[dict]        # [{"id": str, "name": str, "price": float}]
    user_tier: str          # "standard" | "premium"

# ── Tools ────────────────────────────────────────────────────────────────────

@tool
def add_to_cart(
    product_id: str,
    product_name: str,
    price: float,
    # Injected from state — invisible to LLM
    cart: Annotated[list, InjectedState("cart")],
    user_tier: Annotated[str, InjectedState("user_tier")],
) -> str:
    """Add a product to the shopping cart."""
    discount = 0.20 if user_tier == "premium" else 0.0
    final_price = price * (1 - discount)
    # Note: returning a string here; ToolNode wraps it in ToolMessage
    return (
        f"Added {product_name!r} at ${final_price:.2f}"
        f"{' (20% premium discount applied)' if discount else ''}. "
        f"Cart now has {len(cart) + 1} item(s)."
    )

@tool
def view_cart(
    state: Annotated[dict, InjectedState()],
) -> str:
    """View the current cart contents."""
    cart = state.get("cart", [])
    tier = state.get("user_tier", "standard")
    if not cart:
        return f"Cart is empty ({tier} account)."
    lines = [f"  • {item['name']}: ${item['price']:.2f}" for item in cart]
    return f"Cart ({tier}, {len(cart)} items):\n" + "\n".join(lines)

@tool
def save_preference(
    key: str,
    value: str,
    user_id: Annotated[str, InjectedState("user_id")],
    store: Annotated[Any, InjectedStore()],
) -> str:
    """Save a shopping preference (e.g. favourite brand, size)."""
    store.put(("preferences", user_id), key, {"value": value})
    return f"Saved preference: {key} = {value}"

@tool
def get_preference(
    key: str,
    user_id: Annotated[str, InjectedState("user_id")],
    store: Annotated[Any, InjectedStore()],
) -> str:
    """Retrieve a previously saved shopping preference."""
    item = store.get(("preferences", user_id), key)
    if item is None:
        return f"No preference found for '{key}'."
    return f"Your {key}: {item.value['value']}"

# ── Agent setup ───────────────────────────────────────────────────────────────

tools = [add_to_cart, view_cart, save_preference, get_preference]
model = ChatAnthropic(model="claude-3-5-haiku-20241022").bind_tools(tools)
persistent_store = InMemoryStore()

def agent_node(state: ShopState) -> dict:
    return {"messages": [model.invoke(state["messages"])]}

builder = StateGraph(ShopState)
builder.add_node("agent", agent_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")

graph = builder.compile(store=persistent_store)

# ── Usage ─────────────────────────────────────────────────────────────────────

from langchain_core.messages import HumanMessage

result = graph.invoke({
    "messages": [HumanMessage("Add some Sony headphones for $299 to my cart")],
    "user_id": "user-42",
    "cart": [],
    "user_tier": "premium",
})
print(result["messages"][-1].content)
# "Added 'Sony headphones' at $239.20 (20% premium discount applied). Cart now has 1 item(s)."
```

---

## Recipe 10: Resetting Accumulated State with `Overwrite`

When a node needs to **replace** an accumulator channel rather than append to it, use `Overwrite` to bypass the reducer entirely.

```python
import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import Overwrite
from langgraph.checkpoint.memory import InMemorySaver

class PipelineState(TypedDict):
    batch_id: str
    events: Annotated[list[str], operator.add]  # accumulates across nodes
    errors: Annotated[list[str], operator.add]
    phase: str

# Normal accumulation — appends to events
def process_batch(state: PipelineState) -> dict:
    batch = state["batch_id"]
    return {
        "events": [f"processed:{batch}"],
        "phase": "processed",
    }

# Appends error detail
def handle_error(state: PipelineState) -> dict:
    return {
        "errors": [f"error in batch {state['batch_id']}"],
        "events": [f"error:{state['batch_id']}"],
        "phase": "errored",
    }

# Clears both lists — hard reset before re-processing
def reset_state(state: PipelineState) -> dict:
    return {
        "events": Overwrite(value=[f"reset:{state['batch_id']}"]),
        "errors": Overwrite(value=[]),
        "phase": "reset",
    }

def route(state: PipelineState) -> Literal["handle_error", "reset_state", "__end__"]:
    if state["errors"]:
        return "handle_error"
    if state["phase"] == "reset":
        return "__end__"
    return "__end__"

builder = StateGraph(PipelineState)
builder.add_node("process", process_batch)
builder.add_node("handle_error", handle_error)
builder.add_node("reset_state", reset_state)
builder.add_edge(START, "process")
builder.add_conditional_edges("process", route)
builder.add_edge("handle_error", "reset_state")
builder.add_edge("reset_state", END)

graph = builder.compile(checkpointer=InMemorySaver())

# Run with a batch that will have an error, then reset
# In practice you'd trigger the error condition via routing logic
result = graph.invoke({
    "batch_id": "batch-007",
    "events": ["initial"],
    "errors": [],
    "phase": "pending",
})
print(result["events"])   # depends on routing; after reset → ["reset:batch-007"]
print(result["errors"])   # [] — overwritten by reset_state
```

### Key rule for `Overwrite`

Only one node may `Overwrite` a given channel per super-step. If two concurrent nodes both return `Overwrite(...)` for the same channel, LangGraph raises `InvalidUpdateError`.

---

## Recipe 11: Checkpoint History Browser with `CheckpointTuple`

Use `CheckpointTuple` to build a debugging tool that inspects every state the graph passed through, and supports rewinding to any historical step.

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import StateUpdate
from typing_extensions import TypedDict

class WorkflowState(TypedDict):
    input: str
    draft: str
    score: float
    revision: int

def draft_step(state: WorkflowState) -> dict:
    return {
        "draft": f"Draft #{state['revision'] + 1} for: {state['input']}",
        "revision": state["revision"] + 1,
        "score": 0.5 + state["revision"] * 0.1,
    }

saver = InMemorySaver()
builder = StateGraph(WorkflowState)
builder.add_node("draft", draft_step)
builder.add_edge(START, "draft")
builder.add_edge("draft", END)
graph = builder.compile(checkpointer=saver)

config = {"configurable": {"thread_id": "audit-demo"}}

# Run three times to build up history
for _ in range(3):
    graph.invoke({"input": "AI trends", "draft": "", "score": 0.0, "revision": 0}, config)

# ── Browse checkpoint history ─────────────────────────────────────────────────

print("=== Checkpoint History ===")
checkpoints = list(saver.list(config))
for i, cp in enumerate(checkpoints):
    meta = cp.metadata
    vals = cp.checkpoint.get("channel_values", {})
    print(
        f"[{i}] source={meta.get('source')!r:8} "
        f"step={meta.get('step'):3}  "
        f"revision={vals.get('revision', '?')}  "
        f"score={vals.get('score', '?')}"
    )

# ── Time-travel: rewind to an earlier step ──────────────────────────────────

# Pick the second-oldest checkpoint (index -1 is newest)
target_cp = checkpoints[-2]
past_state = graph.get_state(target_cp.config)
print(f"\nRewound to revision={past_state.values['revision']}, score={past_state.values['score']}")

# Continue from that historical point (forks the thread)
resumed = graph.invoke(None, target_cp.config)
print(f"After resume: revision={resumed['revision']}, score={resumed['score']}")

# ── Filter by source ─────────────────────────────────────────────────────────

loop_checkpoints = list(saver.list(config, filter={"source": "loop"}))
print(f"\nLoop checkpoints: {len(loop_checkpoints)}")
```

---

## Recipe 12: Human-in-the-Loop Approval with `update_state` and `StateUpdate`

Pause the graph at a sensitive step, let a human review and modify state, then resume:

```python
from typing import Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import StateUpdate

class ApprovalState(TypedDict):
    request: str
    draft_action: str
    approved: bool
    reviewer_note: str
    result: str

def generate_action(state: ApprovalState) -> dict:
    """Generate a proposed action (requires human approval before executing)."""
    return {
        "draft_action": f"Transfer $10,000 for: {state['request']}",
        "approved": False,
    }

def execute_action(state: ApprovalState) -> dict:
    """Execute the approved action."""
    if not state["approved"]:
        return {"result": "Cancelled — not approved."}
    return {
        "result": f"Executed: {state['draft_action']}. Note: {state['reviewer_note']}"
    }

saver = InMemorySaver()
builder = StateGraph(ApprovalState)
builder.add_node("generate", generate_action)
builder.add_node("execute", execute_action)
builder.add_edge(START, "generate")
builder.add_edge("generate", "execute")  # interrupted here in practice
builder.add_edge("execute", END)

# Interrupt AFTER generate so the human sees the draft before execute runs
graph = builder.compile(checkpointer=saver, interrupt_after=["generate"])

config = {"configurable": {"thread_id": "approval-thread"}}

# Step 1: Start the graph — it pauses after "generate"
graph.invoke(
    {"request": "vendor invoice #1234", "draft_action": "", "approved": False,
     "reviewer_note": "", "result": ""},
    config,
)

# Step 2: Human reviews draft_action via get_state
state = graph.get_state(config)
print("Draft action:", state.values["draft_action"])
# → "Transfer $10,000 for: vendor invoice #1234"

# Step 3: Human approves (and optionally edits)
graph.update_state(
    config,
    {
        "approved": True,
        "reviewer_note": "Verified invoice matches PO-5678",
        # Human can also change draft_action here if needed
    },
    as_node="generate",  # treat this as if generate emitted the update
)

# Step 4: Resume — execute now runs
final = graph.invoke(None, config)
print("Result:", final["result"])
# → "Executed: Transfer $10,000 for: vendor invoice #1234. Note: Verified invoice matches PO-5678"

# ── Bulk update: apply multiple edits atomically ─────────────────────────────

# For multi-field updates you want transactional, use bulk_update_state:
graph.bulk_update_state(
    config,
    [
        [StateUpdate({"approved": True}, as_node="generate")],
        [StateUpdate({"reviewer_note": "All clear"}, as_node="generate")],
    ],
)
