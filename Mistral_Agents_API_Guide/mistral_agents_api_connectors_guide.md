# Mistral Agents API: Comprehensive Connectors Guide

**Version**: 2.0 (May 2025 Launch Edition)
**Last Updated**: May 27, 2025

## Overview

On May 27, 2025, Mistral AI launched the Agents API with **5 powerful built-in connectors** that dramatically extend agent capabilities. These connectors enable autonomous code execution, image generation, web searching, document retrieval, and persistent memory—all managed server-side with enterprise-grade security.

## Table of Contents

1. [Python Code Execution Connector](#1-python-code-execution-connector)
2. [Image Generation Connector](#2-image-generation-connector)
3. [Web Search Connector](#3-web-search-connector)
4. [Document Library/RAG Connector](#4-document-libraryrag-connector)
5. [Persistent Memory Connector](#5-persistent-memory-connector)
6. [Connector Best Practices](#connector-best-practices)
7. [Security Considerations](#security-considerations)
8. [Performance Optimization](#performance-optimization)

---

## 1. Python Code Execution Connector

### Overview

The Python Code Execution connector provides a **secure, sandboxed environment** for autonomous Python code generation and execution. This enables agents to perform complex calculations, data analysis, file processing, and algorithmic tasks.

### Key Features

- **Secure Sandbox**: Isolated execution environment
- **Standard Library Access**: Full Python standard library
- **Data Processing**: NumPy, Pandas support
- **Visualization**: Matplotlib integration
- **Error Handling**: Automatic error capture and reporting
- **Result Streaming**: Real-time execution feedback

### Configuration

```python
import os
from mistralai import Mistral

client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

# Create agent with code execution
agent = client.beta.agents.create(
    model="mistral-medium-latest",
    name="Code Execution Agent",
    description="Agent capable of writing and executing Python code",
    instructions=(
        "You are a Python programming assistant. "
        "When asked to perform calculations or data analysis, "
        "write and execute Python code to provide accurate results. "
        "Always explain your code and results."
    ),
    tools=[
        {
            "type": "code_interpreter",
            "config": {
                "timeout": 30,  # Maximum execution time in seconds
                "max_iterations": 5  # Maximum retry attempts
            }
        }
    ]
)
```

### Usage Examples

#### Example 1: Mathematical Calculations

```python
# Start conversation with code execution request
conversation = client.beta.conversations.start(
    agent_id=agent.id,
    inputs="Calculate the first 10 Fibonacci numbers and plot them"
)

print(conversation.outputs[-1].content)
# Agent will:
# 1. Write Python code to generate Fibonacci sequence
# 2. Execute the code in sandbox
# 3. Create a matplotlib visualization
# 4. Return results and explanation
```

#### Example 2: Data Analysis

```python
conversation = client.beta.conversations.start(
    agent_id=agent.id,
    inputs="""
    Analyze this sales data and provide insights:
    Sales = [120, 135, 148, 162, 155, 170, 188, 195, 210, 225]
    Calculate mean, median, standard deviation, and trend.
    """
)

print(conversation.outputs[-1].content)
# Agent will:
# 1. Parse the data
# 2. Write code using numpy/pandas
# 3. Calculate statistics
# 4. Identify trends
# 5. Return comprehensive analysis
```

#### Example 3: File Processing

```python
conversation = client.beta.conversations.start(
    agent_id=agent.id,
    inputs="""
    Write code to:
    1. Generate 1000 random numbers from normal distribution
    2. Calculate statistical properties
    3. Create histogram
    4. Save results to CSV format
    """
)

print(conversation.outputs[-1].content)
```

### Advanced Features

#### Custom Package Imports

```python
# Agent can use common data science packages
conversation = client.beta.conversations.start(
    agent_id=agent.id,
    inputs="""
    Use pandas to:
    1. Create a DataFrame with columns: name, age, salary
    2. Add 5 sample records
    3. Calculate salary statistics by age group
    4. Show pivot table
    """
)
```

#### Error Handling

The code execution connector automatically handles errors:

```python
conversation = client.beta.conversations.start(
    agent_id=agent.id,
    inputs="Calculate the square root of -1 using real numbers"
)

# Agent will:
# 1. Attempt calculation
# 2. Catch ValueError
# 3. Explain the error
# 4. Suggest alternative (complex numbers)
# 5. Provide corrected solution
```

### Best Practices

1. **Clear Instructions**: Provide specific requirements for code behavior
2. **Timeout Configuration**: Set appropriate timeouts for complex operations
3. **Result Validation**: Have agent validate results before returning
4. **Error Explanation**: Instruct agent to explain errors in user-friendly terms
5. **Code Documentation**: Request commented code for transparency

### Limitations

- Execution timeout: Maximum 30 seconds per execution
- Memory limit: 2GB per execution
- No network access from sandbox
- No persistent file storage between executions
- Limited to Python standard library + numpy, pandas, matplotlib

---

## 2. Image Generation Connector

### Overview

The Image Generation connector integrates **Black Forest Lab's FLUX1.1 [pro] Ultra** model, providing state-of-the-art image generation capabilities directly within your agents.

### Key Features

- **FLUX1.1 [pro] Ultra**: Industry-leading image quality
- **Text-to-Image**: Natural language descriptions to images
- **High Resolution**: Up to 2048x2048 pixels
- **Style Control**: Various artistic styles and techniques
- **Prompt Enhancement**: Automatic prompt optimization
- **Fast Generation**: ~10-15 seconds per image

### Configuration

```python
agent = client.beta.agents.create(
    model="mistral-medium-latest",
    name="Image Generator Agent",
    description="Agent that can generate high-quality images",
    instructions=(
        "You are a creative image generation assistant. "
        "Help users create detailed, high-quality images based on their descriptions. "
        "Ask clarifying questions about style, composition, and details when needed. "
        "Generate images that match user requirements precisely."
    ),
    tools=[
        {
            "type": "image_generation",
            "config": {
                "model": "flux-1.1-pro-ultra",
                "default_size": "1024x1024",
                "quality": "high"
            }
        }
    ]
)
```

### Usage Examples

#### Example 1: Basic Image Generation

```python
conversation = client.beta.conversations.start(
    agent_id=agent.id,
    inputs="Generate an image of a serene Japanese garden at sunset with cherry blossoms"
)

# Access generated image
for output in conversation.outputs:
    if output.type == "image":
        image_url = output.content
        print(f"Generated image: {image_url}")
```

#### Example 2: Iterative Refinement

```python
# Initial generation
conversation = client.beta.conversations.start(
    agent_id=agent.id,
    inputs="Create a professional headshot of a female CEO"
)

# Refine based on feedback
conversation = client.beta.conversations.append(
    conversation_id=conversation.id,
    inputs="Make the background more neutral and add professional lighting"
)

# Further refinement
conversation = client.beta.conversations.append(
    conversation_id=conversation.id,
    inputs="Add a subtle smile and corporate attire"
)
```

#### Example 3: Style-Specific Generation

```python
conversation = client.beta.conversations.start(
    agent_id=agent.id,
    inputs="""
    Generate an image with these specifications:
    - Subject: Futuristic city skyline
    - Style: Cyberpunk aesthetic
    - Mood: Neon-lit, rainy night
    - Color palette: Purples, blues, and neon pinks
    - Composition: Wide angle, street-level perspective
    """
)
```

### Advanced Features

#### Multi-Image Generation

```python
conversation = client.beta.conversations.start(
    agent_id=agent.id,
    inputs="""
    Generate 3 variations of a logo design:
    1. Modern minimalist with geometric shapes
    2. Organic flowing design
    3. Bold angular design
    All using blue and silver color scheme
    """
)
```

#### Prompt Enhancement

The agent automatically enhances prompts for better results:

```python
conversation = client.beta.conversations.start(
    agent_id=agent.id,
    inputs="A cat"
)

# Agent enhances to:
# "A photorealistic portrait of a domestic cat with detailed fur texture,
# sharp focus, natural lighting, professional photography quality"
```

### Best Practices

1. **Detailed Descriptions**: Provide specific details about subject, style, mood, and composition
2. **Iterative Refinement**: Use conversation history to refine images
3. **Style References**: Mention specific artistic styles or techniques
4. **Quality Prompts**: Include lighting, perspective, and technical details
5. **Batch Generation**: Request variations to choose best result

### Limitations

- Generation time: ~10-15 seconds per image
- Maximum resolution: 2048x2048 pixels
- No image-to-image editing (text-to-image only)
- Rate limits apply (check your plan)

---

## 3. Web Search Connector

### Overview

The Web Search connector provides **real-time internet access** with both standard web search and premium authoritative sources (AFP, Associated Press), dramatically improving factual accuracy.

### Key Features

- **Standard Search**: Broad web coverage
- **Premium Sources**: AFP (Agence France-Presse), Associated Press
- **Real-Time Data**: Current information retrieval
- **Source Citations**: Automatic source attribution
- **Multi-Query**: Parallel search execution
- **Result Ranking**: Relevance-based ordering

### Performance Impact

**SimpleQA Benchmark Results:**
- Mistral Large: 23% → **75%** (+52pp with web search)
- Mistral Medium: 22.08% → **82.32%** (+60.24pp with web search)

### Configuration

```python
agent = client.beta.agents.create(
    model="mistral-medium-latest",
    name="Research Assistant",
    description="Agent with web search capabilities",
    instructions=(
        "You are a thorough research assistant. "
        "Use web search to find current, accurate information. "
        "Always cite your sources with URLs. "
        "Prioritize authoritative sources (AFP, AP) when available. "
        "Verify information across multiple sources when possible."
    ),
    tools=[
        {
            "type": "web_search",
            "config": {
                "enable_premium_sources": True,  # Enable AFP, AP
                "max_results": 10,
                "recency_bias": True  # Prioritize recent results
            }
        }
    ]
)
```

### Usage Examples

#### Example 1: Current Events

```python
conversation = client.beta.conversations.start(
    agent_id=agent.id,
    inputs="What are the latest developments in renewable energy technology in 2025?"
)

# Agent will:
# 1. Search web for recent renewable energy news
# 2. Prioritize premium sources (AFP, AP)
# 3. Synthesize findings
# 4. Cite all sources
```

#### Example 2: Fact Verification

```python
conversation = client.beta.conversations.start(
    agent_id=agent.id,
    inputs="Verify this claim: 'Solar panels are now cheaper than fossil fuels in most markets'"
)

# Agent will:
# 1. Search for authoritative sources
# 2. Find supporting/contradicting evidence
# 3. Provide nuanced verification
# 4. Cite sources for transparency
```

#### Example 3: Comparative Research

```python
conversation = client.beta.conversations.start(
    agent_id=agent.id,
    inputs="""
    Compare the features, pricing, and user reviews of:
    - iPhone 16
    - Samsung Galaxy S25
    - Google Pixel 10
    Provide a comparison table
    """
)

# Agent will:
# 1. Search for specs and pricing
# 2. Find user reviews
# 3. Search for expert opinions
# 4. Create structured comparison
# 5. Cite all sources
```

### Advanced Features

#### Multi-Query Search

```python
conversation = client.beta.conversations.start(
    agent_id=agent.id,
    inputs="""
    Research the following topics:
    1. Current inflation rates globally
    2. Central bank policy responses
    3. Economic forecasts for Q3 2025
    Provide a comprehensive economic overview
    """
)

# Agent executes multiple searches in parallel
```

#### Premium Source Filtering

```python
# Configure agent to prefer premium sources
agent = client.beta.agents.create(
    model="mistral-medium-latest",
    name="News Analyst",
    instructions=(
        "You are a news analyst. "
        "Prioritize information from AFP and Associated Press. "
        "Only use other sources if premium sources lack coverage. "
        "Always indicate source tier (premium vs. standard)."
    ),
    tools=[
        {
            "type": "web_search",
            "config": {
                "enable_premium_sources": True,
                "prefer_premium": True
            }
        }
    ]
)
```

### Best Practices

1. **Source Attribution**: Always cite sources in responses
2. **Recency Checking**: Configure recency bias for time-sensitive queries
3. **Cross-Verification**: Search multiple sources for important facts
4. **Premium Sources**: Enable premium for authoritative information
5. **Query Refinement**: Use specific search terms for better results

### Limitations

- Search quota limits (check your plan)
- Premium sources may have limited coverage for niche topics
- Real-time data has ~5-minute delay
- Some regions may have restricted search capabilities

---

## 4. Document Library/RAG Connector

### Overview

The Document Library/RAG connector provides **native Mistral Cloud document access** for knowledge retrieval, enabling agents to answer questions based on your uploaded documents.

### Key Features

- **Mistral Cloud Integration**: Native document storage
- **Semantic Search**: Vector-based retrieval
- **Multi-Document**: Search across document collections
- **Chunking Strategy**: Intelligent document segmentation
- **Citation**: Automatic source document attribution
- **Update Sync**: Real-time document updates

### Configuration

```python
# First, upload documents to Mistral Cloud
# (This is done separately via Mistral Cloud console or API)

# Create agent with document library access
agent = client.beta.agents.create(
    model="mistral-large-latest",  # Better for complex RAG
    name="Document Q&A Agent",
    description="Agent that answers questions from document library",
    instructions=(
        "You are a knowledgeable assistant with access to a document library. "
        "When answering questions, search the library for relevant information. "
        "Always cite the specific documents you reference. "
        "If information isn't in the library, clearly state this. "
        "Provide page numbers or section references when possible."
    ),
    tools=[
        {
            "type": "document_retrieval",
            "config": {
                "library_id": "lib_your_library_id",
                "max_documents": 5,
                "similarity_threshold": 0.7
            }
        }
    ]
)
```

### Usage Examples

#### Example 1: Company Knowledge Base

```python
# Upload company documents to Mistral Cloud first
# Then create agent

conversation = client.beta.conversations.start(
    agent_id=agent.id,
    inputs="What is our company's vacation policy for remote employees?"
)

# Agent will:
# 1. Search document library for vacation policy
# 2. Find relevant sections
# 3. Synthesize answer
# 4. Cite specific documents and sections
```

#### Example 2: Technical Documentation

```python
conversation = client.beta.conversations.start(
    agent_id=agent.id,
    inputs="How do I configure SSL/TLS for the API Gateway according to our docs?"
)

# Agent will:
# 1. Search technical documentation
# 2. Find configuration steps
# 3. Provide step-by-step instructions
# 4. Include code examples from docs
# 5. Cite source documents
```

#### Example 3: Multi-Document Analysis

```python
conversation = client.beta.conversations.start(
    agent_id=agent.id,
    inputs="""
    Compare the security recommendations across:
    - Security Best Practices 2024.pdf
    - Cloud Infrastructure Guide.pdf
    - Compliance Handbook.pdf
    Identify any conflicts or gaps
    """
)

# Agent will:
# 1. Retrieve relevant sections from all three documents
# 2. Compare recommendations
# 3. Identify conflicts
# 4. Highlight gaps
# 5. Provide synthesis with citations
```

### Advanced Features

#### Dynamic Library Updates

```python
# Documents added to library are immediately searchable
# Agent automatically has access to latest versions

conversation = client.beta.conversations.start(
    agent_id=agent.id,
    inputs="What are the updates in the latest product roadmap?"
)

# If roadmap was just uploaded, agent finds it immediately
```

#### Hybrid Search (RAG + Web Search)

```python
agent = client.beta.agents.create(
    model="mistral-large-latest",
    name="Hybrid Research Agent",
    instructions=(
        "You answer questions using both our document library and web search. "
        "First check the document library. "
        "If insufficient information, supplement with web search. "
        "Clearly distinguish between internal and external sources."
    ),
    tools=[
        {"type": "document_retrieval", "config": {"library_id": "lib_xxx"}},
        {"type": "web_search", "config": {"enable_premium_sources": True}}
    ]
)

conversation = client.beta.conversations.start(
    agent_id=agent.id,
    inputs="How does our authentication system compare to industry standards?"
)

# Agent will:
# 1. Search internal docs for our auth system
# 2. Search web for industry standards
# 3. Provide comparison with dual citations
```

### Best Practices

1. **Document Organization**: Structure documents with clear sections and headings
2. **Metadata**: Add metadata to documents for better retrieval
3. **Chunking**: Use appropriate document segmentation
4. **Citation Format**: Standardize how agent cites sources
5. **Fallback Strategy**: Define behavior when documents lack information
6. **Library Maintenance**: Regularly update and prune document library

### Limitations

- Document storage limits (check your plan)
- Maximum document size: 50MB per document
- Supported formats: PDF, TXT, DOCX, MD
- Indexing time: ~1-2 minutes for new documents
- Search latency: ~100-200ms per query

---

## 5. Persistent Memory Connector

### Overview

The Persistent Memory connector provides **server-side conversation state management**, enabling agents to maintain context across sessions, days, or even months.

### Key Features

- **Server-Side Storage**: All conversation history stored by Mistral
- **Cross-Session**: Resume conversations anytime
- **Unlimited History**: No practical limit on conversation length
- **Fast Retrieval**: ~50ms to load full conversation context
- **Branching**: Create alternative conversation paths
- **Export/Import**: Full conversation data export

### Configuration

```python
# Persistent memory is built into the Conversations API
# No explicit tool configuration needed - it's always active

agent = client.beta.agents.create(
    model="mistral-medium-latest",
    name="Personal Assistant",
    description="Agent with perfect memory of all interactions",
    instructions=(
        "You are a personal assistant with perfect memory. "
        "Remember all previous conversations with this user. "
        "Reference past interactions when relevant. "
        "Track user preferences, projects, and ongoing tasks. "
        "Maintain continuity across all conversations."
    )
)
```

### Usage Examples

#### Example 1: Multi-Session Continuity

```python
# Session 1: Initial conversation
conversation = client.beta.conversations.start(
    agent_id=agent.id,
    inputs="I'm working on a machine learning project to predict customer churn"
)

conversation_id = conversation.id
print(f"Save this ID: {conversation_id}")

# --- Hours or days later ---

# Session 2: Resume conversation
conversation = client.beta.conversations.append(
    conversation_id=conversation_id,
    inputs="I've finished the data preprocessing. What should I do next?"
)

# Agent remembers:
# - It's a customer churn prediction project
# - Previous discussions
# - Current stage (data preprocessing)
```

#### Example 2: Long-Term Project Tracking

```python
# Week 1
conversation = client.beta.conversations.start(
    agent_id=agent.id,
    inputs="Starting a 6-month product development project. Phase 1: Requirements gathering"
)

# Week 2
client.beta.conversations.append(
    conversation_id=conversation.id,
    inputs="Completed requirements. Moving to design phase"
)

# Week 4
client.beta.conversations.append(
    conversation_id=conversation.id,
    inputs="Design approved. Starting implementation"
)

# Month 3
client.beta.conversations.append(
    conversation_id=conversation.id,
    inputs="What were our original requirements from week 1?"
)

# Agent retrieves full history and references week 1 requirements
```

#### Example 3: Preference Learning

```python
# Conversation 1
conversation = client.beta.conversations.start(
    agent_id=agent.id,
    inputs="I prefer Python over JavaScript for backend development"
)

# Conversation 2 (same conversation_id)
client.beta.conversations.append(
    conversation_id=conversation.id,
    inputs="I need to build a new API. What language should I use?"
)

# Agent remembers preference and recommends Python
```

### Advanced Features

#### Conversation Branching

```python
# Original conversation
conversation = client.beta.conversations.start(
    agent_id=agent.id,
    inputs="Should I use SQL or NoSQL for this e-commerce app?"
)

# Get conversation history
history = client.beta.conversations.get_history(
    conversation_id=conversation.id
)

# Create branch to explore SQL path
sql_branch = client.beta.conversations.restart(
    conversation_id=conversation.id,
    from_entry_id=history.entries[0].id,
    inputs="Let's explore the SQL option in detail"
)

# Create another branch for NoSQL path
nosql_branch = client.beta.conversations.restart(
    conversation_id=conversation.id,
    from_entry_id=history.entries[0].id,
    inputs="Let's explore the NoSQL option in detail"
)

# Compare both paths
```

#### History Analysis

```python
# Retrieve full conversation history
history = client.beta.conversations.get_history(
    conversation_id=conversation.id
)

print(f"Total entries: {len(history.entries)}")
print(f"Total messages: {len([e for e in history.entries if e.type == 'message'])}")
print(f"Tool executions: {len([e for e in history.entries if e.type == 'tool'])}")

# Analyze conversation
for entry in history.entries:
    print(f"{entry.created_at}: {entry.type} - {entry.role}")
```

#### Memory-Based Personalization

```python
agent = client.beta.agents.create(
    model="mistral-medium-latest",
    name="Personalized Tutor",
    instructions=(
        "You are a personalized tutor. "
        "Track the student's learning progress, strengths, and weaknesses. "
        "Adapt difficulty based on past performance. "
        "Remember which topics have been covered. "
        "Personalize explanations to their learning style."
    )
)

# Over multiple sessions, agent learns and adapts to student
```

### Best Practices

1. **Conversation IDs**: Store conversation IDs for resumption
2. **Context Management**: Agent instructions should emphasize using history
3. **Branching Strategy**: Use branches for exploring alternatives
4. **History Cleanup**: Archive old conversations periodically
5. **Privacy**: Consider data retention policies for sensitive conversations

### Limitations

- Conversation storage: Unlimited (practically)
- Retrieval time: ~50ms for small conversations, ~200ms for very long ones
- Maximum messages per retrieval: 1000 (pagination for larger)
- Branch depth: Unlimited

---

## Connector Best Practices

### 1. Combining Multiple Connectors

```python
# Powerful agent with all connectors
agent = client.beta.agents.create(
    model="mistral-large-latest",
    name="Full-Featured Agent",
    instructions=(
        "You are a comprehensive AI assistant with multiple capabilities: "
        "1. Execute Python code for calculations and analysis "
        "2. Generate images when visual content is needed "
        "3. Search the web for current information "
        "4. Retrieve information from our document library "
        "5. Remember all previous conversations "
        "Choose the appropriate tool(s) for each task. "
        "Combine tools when beneficial (e.g., web search + code for data analysis)."
    ),
    tools=[
        {"type": "code_interpreter"},
        {"type": "image_generation"},
        {"type": "web_search", "config": {"enable_premium_sources": True}},
        {"type": "document_retrieval", "config": {"library_id": "lib_xxx"}}
    ]
)
```

### 2. Tool Selection Strategy

```python
# Instruct agent on tool selection
instructions = """
You have access to multiple tools. Use them wisely:

- Code Interpreter: For calculations, data analysis, algorithms
- Image Generation: For visual content creation
- Web Search: For current events, recent information, general knowledge
- Document Library: For company-specific information, internal docs
- Memory: Always active, reference past conversations

Tool Selection Rules:
1. Internal docs (RAG) before web search for company info
2. Code execution for anything numerical or computational
3. Web search for current events (recency matters)
4. Image generation when visual explanation helps
5. Combine tools when one feeds into another

Always explain which tool you're using and why.
"""
```

### 3. Error Recovery

```python
# Build robust error handling into instructions
instructions = """
When using tools:

1. Code Execution Errors:
   - Explain what went wrong
   - Show the error message
   - Provide corrected code
   - Re-execute if possible

2. Web Search Issues:
   - If no results, try alternative search terms
   - If sources conflict, acknowledge uncertainty
   - Always cite sources

3. Document Retrieval:
   - If not found in docs, acknowledge limitation
   - Offer to search web as fallback
   - Don't hallucinate document content

4. Image Generation:
   - If result doesn't match, refine prompt
   - Offer to regenerate with adjustments
"""
```

### 4. Performance Optimization

```python
# Configure connectors for optimal performance

agent = client.beta.agents.create(
    model="mistral-medium-latest",
    name="Optimized Agent",
    tools=[
        {
            "type": "code_interpreter",
            "config": {
                "timeout": 15,  # Shorter timeout for faster failures
                "max_iterations": 3
            }
        },
        {
            "type": "web_search",
            "config": {
                "max_results": 5,  # Fewer results = faster
                "recency_bias": True
            }
        },
        {
            "type": "document_retrieval",
            "config": {
                "max_documents": 3,  # Fewer docs = faster
                "similarity_threshold": 0.75  # Higher threshold = more precise
            }
        }
    ]
)
```

---

## Security Considerations

### 1. Code Execution Security

```python
# Code execution runs in secure sandbox
# No network access, no file persistence

# However, instruct agent on safe practices
instructions = """
Code Execution Security:
- Never attempt to access network resources
- Don't try to execute system commands
- Avoid infinite loops (timeout will kill)
- Don't process sensitive user data without confirmation
- Validate all user-provided data before processing
"""
```

### 2. Web Search Privacy

```python
# Web search queries are logged
# Be mindful of sensitive information

instructions = """
Web Search Privacy:
- Avoid searching for user PII
- Generalize searches when possible
- Warn user if search might expose sensitive data
- Prefer document library for confidential info
"""
```

### 3. Document Library Access Control

```python
# Ensure proper library permissions

# Create agent with specific library access
agent = client.beta.agents.create(
    model="mistral-medium-latest",
    name="Restricted Doc Agent",
    tools=[
        {
            "type": "document_retrieval",
            "config": {
                "library_id": "lib_public_docs_only",  # Restricted library
            }
        }
    ]
)

# Don't mix public and confidential documents in same library
```

### 4. Memory and Data Retention

```python
# Consider data retention policies

# For sensitive conversations, implement cleanup
def cleanup_old_conversations(client, agent_id, days_old=90):
    """Archive conversations older than specified days"""
    conversations = client.beta.conversations.list(agent_id=agent_id)

    for conv in conversations:
        age_days = (datetime.now() - conv.created_at).days
        if age_days > days_old:
            # Archive or delete
            client.beta.conversations.archive(conversation_id=conv.id)
```

---

## Performance Optimization

### 1. Connector Latency

| Connector | Typical Latency | Optimization Strategy |
|-----------|----------------|----------------------|
| Code Execution | 2-5 seconds | Reduce iteration count, optimize code |
| Image Generation | 10-15 seconds | Clear prompts, avoid regeneration |
| Web Search | 1-3 seconds | Limit results, specific queries |
| Document Retrieval | 0.1-0.3 seconds | Higher similarity threshold, fewer docs |
| Persistent Memory | 0.05-0.2 seconds | No optimization needed (fast) |

### 2. Parallel Tool Execution

```python
# Agent automatically parallelizes when possible

conversation = client.beta.conversations.start(
    agent_id=agent.id,
    inputs="""
    Perform these tasks:
    1. Search web for current stock market trends
    2. Retrieve our Q4 financial report from docs
    3. Generate chart visualization

    Then combine all information into analysis
    """
)

# Agent will:
# 1. Execute web search and document retrieval in parallel
# 2. Wait for both to complete
# 3. Use code execution to generate chart
# 4. Synthesize results
```

### 3. Caching Strategies

```python
# Document library has built-in caching
# Web search results cached for ~5 minutes

# Leverage conversation memory as cache
conversation = client.beta.conversations.start(
    agent_id=agent.id,
    inputs="Search for latest AI trends"
)

# Later in same conversation
conversation = client.beta.conversations.append(
    conversation_id=conversation.id,
    inputs="Based on those trends you found, what should we focus on?"
)

# Agent reuses search results from memory instead of re-searching
```

### 4. Model Selection for Connectors

```python
# Different models have different connector strengths

# For code-heavy tasks: mistral-large-latest (better reasoning)
code_agent = client.beta.agents.create(
    model="mistral-large-latest",
    name="Code Expert",
    tools=[{"type": "code_interpreter"}]
)

# For search-heavy tasks: mistral-medium-latest (faster, good enough)
search_agent = client.beta.agents.create(
    model="mistral-medium-latest",
    name="Research Assistant",
    tools=[{"type": "web_search"}]
)

# For RAG: mistral-large-latest (better comprehension)
rag_agent = client.beta.agents.create(
    model="mistral-large-latest",
    name="Doc Expert",
    tools=[{"type": "document_retrieval"}]
)
```

---

## Complete Example: Multi-Connector Agent

```python
import os
from mistralai import Mistral

def create_comprehensive_agent():
    """Create agent with all connectors for maximum capability"""

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    agent = client.beta.agents.create(
        model="mistral-large-latest",
        name="Comprehensive Research & Analysis Agent",
        description="Full-featured agent with all connectors",
        instructions="""
        You are a comprehensive AI assistant with five powerful capabilities:

        1. **Python Code Execution**: Use for calculations, data analysis, visualizations
        2. **Image Generation**: Create high-quality images using FLUX1.1 [pro] Ultra
        3. **Web Search**: Access current information (prefer premium sources: AFP, AP)
        4. **Document Library**: Retrieve company-specific information from our docs
        5. **Persistent Memory**: Perfect recall of all conversations

        **Tool Selection Strategy:**
        - Check document library FIRST for company/internal info
        - Use web search for current events, recent data, general knowledge
        - Execute code for anything numerical, analytical, or computational
        - Generate images when visual representation enhances understanding
        - Always reference conversation history for context

        **Best Practices:**
        - Cite all sources (web and docs)
        - Explain your code before executing
        - Combine tools when beneficial
        - Acknowledge limitations when information unavailable
        - Maintain conversation continuity

        Be helpful, accurate, and comprehensive in all responses.
        """,
        tools=[
            {
                "type": "code_interpreter",
                "config": {
                    "timeout": 30,
                    "max_iterations": 5
                }
            },
            {
                "type": "image_generation",
                "config": {
                    "model": "flux-1.1-pro-ultra",
                    "default_size": "1024x1024"
                }
            },
            {
                "type": "web_search",
                "config": {
                    "enable_premium_sources": True,
                    "max_results": 10,
                    "recency_bias": True
                }
            },
            {
                "type": "document_retrieval",
                "config": {
                    "library_id": os.environ.get("MISTRAL_LIBRARY_ID"),
                    "max_documents": 5,
                    "similarity_threshold": 0.7
                }
            }
        ],
        completion_args={
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": 4096
        }
    )

    print(f"✅ Agent created: {agent.id}")
    return agent, client


def run_comprehensive_demo(agent, client):
    """Demonstrate all connectors in one conversation"""

    # Start conversation
    conversation = client.beta.conversations.start(
        agent_id=agent.id,
        inputs="""
        I need a comprehensive analysis of renewable energy trends:

        1. Search web for latest 2025 renewable energy statistics
        2. Check our document library for our company's renewable energy strategy
        3. Calculate year-over-year growth rates from the data
        4. Generate a visualization showing trends
        5. Create an infographic summarizing key findings

        Provide a complete analysis with all sources cited.
        """
    )

    print("\n=== Agent Response ===")
    print(conversation.outputs[-1].content)

    # Continue conversation (demonstrates persistent memory)
    follow_up = client.beta.conversations.append(
        conversation_id=conversation.id,
        inputs="Based on this analysis, what should our 2026 strategy focus on?"
    )

    print("\n=== Follow-up Response ===")
    print(follow_up.outputs[-1].content)

    # Return conversation ID for future sessions
    return conversation.id


if __name__ == "__main__":
    agent, client = create_comprehensive_agent()
    conversation_id = run_comprehensive_demo(agent, client)
    print(f"\n💾 Save this conversation ID: {conversation_id}")
    print("   Use it to resume this conversation anytime!")
```

---

## Conclusion

The May 2025 launch of Mistral Agents API brings five powerful connectors that transform agent capabilities:

1. **Python Code Execution** - Autonomous computational tasks
2. **Image Generation** - High-quality visual content creation
3. **Web Search** - Real-time information with premium sources
4. **Document Library/RAG** - Company knowledge retrieval
5. **Persistent Memory** - Unlimited conversation context

By combining these connectors strategically, you can build sophisticated agents that rival or exceed specialized frameworks—all with simpler code, better integration, and enterprise-grade reliability.

**Next Steps:**
- Explore [Agent Orchestration Guide](mistral_agents_api_orchestration_guide.md) for multi-agent patterns
- Learn [MCP Integration](mistral_agents_api_mcp_guide.md) for third-party tool connections
- Review [Production Guide](mistral_agents_api_production_guide.md) for deployment best practices

---

**Documentation Version**: 2.0 (May 2025 Launch Edition)
**Last Updated**: May 27, 2025
**Mistral AI - Build Autonomous Agents with Confidence**
