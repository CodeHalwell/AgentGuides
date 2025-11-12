
# LangGraph: Advanced Recipes & Real-World Patterns

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
from langchain.tools import tool

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

This collection of recipes covers real-world patterns you'll encounter building production AI systems.
Adapt and combine them for your specific use cases!




