# LlamaIndex: Advanced Implementation Patterns

This guide explores advanced, production-grade patterns for building sophisticated multi-agent and RAG systems with LlamaIndex.

---
## Advanced Multi-Agent Systems with `llama-agents`

While the comprehensive guide introduces `llama-agents`, this section dives deeper into orchestration and communication.

### Asynchronous Agent Orchestration

For complex workflows, running agents sequentially is inefficient. `llama-agents` supports asynchronous execution, allowing you to run multiple agents in parallel.

```python
# advanced_multi_agent.py
import asyncio
from llama_agents import (
    AgentService,
    ControlPlaneServer,
    SimpleMessageQueue,
    AgentOrchestrator,
)
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

# Define tools for different agents
def research_tool(topic: str) -> str:
    """Researches a topic online."""
    return f"Research complete for {topic}. Findings: ..."

def analysis_tool(data: str) -> str:
    """Analyzes data."""
    return f"Analysis complete. Key insight: ..."

# Create agents
llm = OpenAI(model="gpt-4")
research_agent_service = AgentService(
    agent=ReActAgent.from_tools([FunctionTool.from_defaults(research_tool)], llm=llm),
    message_queue=SimpleMessageQueue(),
)
analysis_agent_service = AgentService(
    agent=ReActAgent.from_tools([FunctionTool.from_defaults(analysis_tool)], llm=llm),
    message_queue=SimpleMessageQueue(),
)

# Create orchestrator
orchestrator = AgentOrchestrator(
    llm=llm,
    message_queue=SimpleMessageQueue.from_other(research_agent_service.message_queue),
)

async def main():
    # Define the task
    task_str = "Research 'quantum computing' and analyze the findings."
    
    # Let the orchestrator create a plan and execute it
    result = await orchestrator.orchestrate(
        task_str,
        agent_services=[research_agent_service, analysis_agent_service],
    )
    
    print(f"Final Result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

---
## Complex Human-in-the-Loop (HITL) Patterns

LlamaIndex agents can be designed to pause and wait for human input, which is crucial for approval workflows or interactive data labeling.

### Human Review and Approval Workflow

This pattern involves an agent performing a task, presenting the result for human review, and only proceeding after receiving approval.

```python
# human_approval_workflow.py
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI

def draft_email(recipient: str, topic: str) -> str:
    """Drafts an email."""
    return f"Subject: {topic}\n\nDear {recipient},\n\nThis is a draft about {topic}."

def send_email(email_content: str) -> str:
    """Sends an email. THIS IS A CRITICAL ACTION."""
    print(f"---
SENDING EMAIL ---
{email_content}")
    return "Email sent successfully."

tools = [
    FunctionTool.from_defaults(draft_email),
    FunctionTool.from_defaults(send_email),
]

llm = OpenAI(model="gpt-4")
agent = ReActAgent.from_tools(tools, llm=llm, verbose=True)

# Manual HITL implementation
task = "Draft an email to 'test@example.com' about 'Project Update' and send it."
result = agent.chat(task)

print(f"Agent wants to do: {result}")

# Human-in-the-loop
approval = input("Approve this action? (yes/no): ")
if approval.lower() == "yes":
    final_result = agent.chat("Yes, please proceed.")
    print(f"Final Result: {final_result}")
else:
    print("Action rejected by user.")
```

---
## Middleware and Observability

You can inject middleware-like functionality for logging, metrics, and tracing by creating custom `QueryEngine` or `Agent` classes.

### Custom Agent with Observability

This example shows a custom agent that logs every step and its duration.

```python
# observable_agent.py
import time
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAI
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

class ObservableReActAgent(ReActAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, *args, **kwargs):
        start_time = time.time()
        logging.info(f"Agent run started with task: {args[0]}")
        
        result = super().run(*args, **kwargs)
        
        duration = time.time() - start_time
        logging.info(f"Agent run finished in {duration:.2f} seconds.")
        return result

# Usage
tools = [FunctionTool.from_defaults(lambda x: x, name="echo")]
llm = OpenAI(model="gpt-4")
agent = ObservableReActAgent.from_tools(tools, llm=llm)
agent.run("This is a test.")
```

---
## Advanced Error Handling and Recovery

Production systems need to be resilient. This pattern shows how to add retry logic and fallbacks to tool calls.

```python
# resilient_tools.py
from llama_index.core.tools import FunctionTool
from tenacity import retry, stop_after_attempt, wait_exponential

# A tool that might fail
def unreliable_api_call(query: str) -> str:
    """This API call is unreliable and might fail."""
    import random
    if random.random() < 0.5:
        raise ConnectionError("Failed to connect to the API")
    return f"Data for {query}"

# A fallback tool
def fallback_search(query: str) -> str:
    """A reliable fallback search."""
    return f"Fallback search results for {query}"

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def resilient_api_call(query: str) -> str:
    """A resilient version of the API call with retries."""
    try:
        return unreliable_api_call(query)
    except ConnectionError as e:
        print(f"Attempt failed: {e}. Retrying...")
        raise

def tool_with_fallback(query: str) -> str:
    """A tool that uses a fallback if the primary function fails."""
    try:
        return resilient_api_call(query)
    except Exception as e:
        print(f"All retries failed: {e}. Using fallback.")
        return fallback_search(query)

# Create a tool from the resilient function
resilient_tool = FunctionTool.from_defaults(tool_with_fallback, name="resilient_search")
```
