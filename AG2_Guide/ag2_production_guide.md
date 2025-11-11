# AG2 Production Guide

**Enterprise-Grade Deployment, Monitoring, and Optimisation**

## Table of Contents

1. [Logging and Debugging](#logging-and-debugging)
2. [Cost Tracking and Token Optimisation](#cost-tracking-and-token-optimisation)
3. [Error Handling and Resilience](#error-handling-and-resilience)
4. [Testing Strategies](#testing-strategies)
5. [Deployment Patterns](#deployment-patterns)
6. [Performance Optimisation](#performance-optimisation)
7. [Security Considerations](#security-considerations)
8. [Monitoring and Observability](#monitoring-and-observability)
9. [Integration with LangChain and LlamaIndex](#integration-with-langchain-and-llamaindex)
10. [Advanced Async Execution](#advanced-async-execution)

---

## Logging and Debugging

### Configuring Python Logging

```python
import logging
from autogen import ConversableAgent, LLMConfig

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_debug.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

llm_config = LLMConfig.from_json("OAI_CONFIG_LIST")

agent = ConversableAgent(
    name="assistant",
    llm_config=llm_config,
    human_input_mode="NEVER"
)

logger.info(f"Created agent: {agent.name}")
```

### Detailed Message Inspection

```python
from autogen import ConversableAgent, LLMConfig

llm_config = LLMConfig.from_json("OAI_CONFIG_LIST")

agent1 = ConversableAgent("agent1", llm_config=llm_config)
agent2 = ConversableAgent("agent2", llm_config=llm_config)

response = agent1.run(recipient=agent2, message="Test", max_turns=2)
response.process()

# Inspect each message
for msg in response.chat_history:
    print(f"From: {msg.get('name')}")
    print(f"Role: {msg.get('role')}")
    print(f"Content: {msg.get('content')[:200]}...")
    print(f"Timestamp: {msg.get('timestamp', 'N/A')}")
    print("---")
```

### Custom Message Logging

```python
import json
from datetime import datetime
from autogen import ConversableAgent, LLMConfig

class LoggingAgent(ConversableAgent):
    def __init__(self, *args, log_file="agent_messages.jsonl", **kwargs):
        super().__init__(*args, **kwargs)
        self.log_file = log_file
    
    def _log_message(self, message):
        """Log message to JSONL file."""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent": self.name,
            "message": message
        }
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def send(self, message, recipient, request_reply=True, silent=False):
        self._log_message({"type": "sent", "content": message})
        return super().send(message, recipient, request_reply, silent)

# Usage
logging_agent = LoggingAgent(
    name="logger",
    llm_config=LLMConfig.from_json("OAI_CONFIG_LIST"),
    log_file="production_agent.jsonl"
)
```

---

## Cost Tracking and Token Optimisation

### Token Usage Monitoring

```python
from autogen import ConversableAgent, LLMConfig

class CostTrackingAgent(ConversableAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.total_tokens = 0
        self.total_cost = 0
        self.price_per_1k = 0.03  # Adjust based on model
    
    def track_usage(self, response):
        """Track token usage from response."""
        if hasattr(response, 'usage'):
            tokens = response.usage.total_tokens
            cost = (tokens / 1000) * self.price_per_1k
            self.total_tokens += tokens
            self.total_cost += cost
            
            return {
                "tokens": tokens,
                "cost": cost,
                "total_tokens": self.total_tokens,
                "total_cost": self.total_cost
            }

# Usage
tracker = CostTrackingAgent(
    name="tracker",
    llm_config=LLMConfig.from_json("OAI_CONFIG_LIST")
)

print(f"Total cost: ${tracker.total_cost:.4f}")
```

### Message Pruning Strategy

```python
from autogen import ConversableAgent, LLMConfig

llm_config = LLMConfig.from_json("OAI_CONFIG_LIST")

agent = ConversableAgent(
    name="assistant",
    llm_config=llm_config,
    human_input_mode="NEVER"
)

# Keep only last N messages to reduce tokens
def prune_messages(chat_history, max_messages=10):
    """Keep only the most recent messages."""
    if len(chat_history) > max_messages:
        return chat_history[-max_messages:]
    return chat_history

# Apply pruning
if len(agent.chat_messages) > 10:
    agent.chat_messages = {
        key: prune_messages(msgs) 
        for key, msgs in agent.chat_messages.items()
    }
```

### Dynamic Context Window Management

```python
from autogen import ConversableAgent, LLMConfig

class OptimisedAgent(ConversableAgent):
    def __init__(self, *args, max_context_tokens=4000, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_context_tokens = max_context_tokens
    
    def estimate_tokens(self, text):
        """Rough token estimation (1 token ≈ 4 characters)."""
        return len(text) / 4
    
    def prepare_context(self):
        """Prepare context within token limits."""
        total_tokens = 0
        selected_msgs = []
        
        for msg in reversed(self.chat_messages[self.name]):
            msg_tokens = self.estimate_tokens(str(msg))
            if total_tokens + msg_tokens > self.max_context_tokens:
                break
            selected_msgs.insert(0, msg)
            total_tokens += msg_tokens
        
        return selected_msgs

# Usage
opt_agent = OptimisedAgent(
    name="optimised",
    llm_config=LLMConfig.from_json("OAI_CONFIG_LIST"),
    max_context_tokens=3000
)
```

---

## Error Handling and Resilience

### Retry Logic with Exponential Backoff

```python
import time
from typing import Callable, Any
from autogen import ConversableAgent, LLMConfig

def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_wait: float = 1.0,
    backoff_factor: float = 2.0
) -> Any:
    """Retry function with exponential backoff."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            
            wait_time = base_wait * (backoff_factor ** attempt)
            print(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)

# Usage
def chat_with_retry():
    llm_config = LLMConfig.from_json("OAI_CONFIG_LIST")
    agent = ConversableAgent("test", llm_config=llm_config)
    
    return agent.run(
        recipient=agent,
        message="Test",
        max_turns=1
    )

result = retry_with_backoff(chat_with_retry)
```

### Graceful Degradation

```python
from autogen import ConversableAgent, LLMConfig, AssistantAgent, UserProxyAgent

def create_resilient_system():
    """Create system that degrades gracefully on API failures."""
    try:
        llm_config = LLMConfig.from_json("OAI_CONFIG_LIST")
    except Exception:
        print("Warning: Could not load LLM config. Using fallback...")
        # Fallback: use basic agent without LLM
        return create_fallback_agents()
    
    return AssistantAgent("assistant", llm_config=llm_config), UserProxyAgent("user_proxy")

def create_fallback_agents():
    """Create agents for offline mode."""
    return (
        ConversableAgent("assistant", llm_config=False, human_input_mode="ALL"),
        ConversableAgent("user_proxy", llm_config=False)
    )

assistant, executor = create_resilient_system()
```

### Validation and Sanitisation

```python
from autogen import ConversableAgent, LLMConfig
import re

class ValidatedAgent(ConversableAgent):
    def send(self, message, recipient, request_reply=True, silent=False):
        # Sanitise message
        message = self._sanitise_message(message)
        
        # Validate length
        if len(message) > 10000:
            raise ValueError("Message exceeds maximum length")
        
        return super().send(message, recipient, request_reply, silent)
    
    @staticmethod
    def _sanitise_message(message: str) -> str:
        """Remove potentially harmful content."""
        # Remove excessive whitespace
        message = re.sub(r'\s+', ' ', message)
        
        # Remove control characters
        message = ''.join(c for c in message if c.isprintable())
        
        return message.strip()

# Usage
validated = ValidatedAgent(
    name="validated",
    llm_config=LLMConfig.from_json("OAI_CONFIG_LIST")
)
```

---

## Testing Strategies

### Unit Testing Agents

```python
import unittest
from autogen import ConversableAgent, LLMConfig

class TestAgents(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.llm_config = LLMConfig.from_json("OAI_CONFIG_LIST")
    
    def test_agent_creation(self):
        """Test agent creation."""
        agent = ConversableAgent(
            name="test",
            llm_config=self.llm_config
        )
        self.assertEqual(agent.name, "test")
        self.assertIsNotNone(agent.llm_config)
    
    def test_agent_properties(self):
        """Test agent properties."""
        agent = ConversableAgent(
            name="test",
            llm_config=self.llm_config,
            human_input_mode="NEVER"
        )
        self.assertEqual(agent.human_input_mode, "NEVER")
    
    def test_system_message(self):
        """Test system message configuration."""
        system_msg = "You are helpful"
        agent = ConversableAgent(
            name="test",
            system_message=system_msg,
            llm_config=self.llm_config
        )
        self.assertEqual(agent.system_message, system_msg)

if __name__ == '__main__':
    unittest.main()
```

### Integration Testing

```python
import unittest
from autogen import ConversableAgent, LLMConfig

class TestIntegration(unittest.TestCase):
    def test_two_agent_conversation(self):
        """Test conversation between two agents."""
        llm_config = LLMConfig.from_json("OAI_CONFIG_LIST")
        
        agent1 = ConversableAgent(
            name="agent1",
            llm_config=llm_config,
            max_consecutive_auto_reply=2
        )
        
        agent2 = ConversableAgent(
            name="agent2",
            llm_config=llm_config,
            max_consecutive_auto_reply=2
        )
        
        # Run conversation
        response = agent1.run(
            recipient=agent2,
            message="Hello, how are you?",
            max_turns=3
        )
        
        # Verify conversation occurred
        self.assertGreater(len(response.chat_history), 0)
        
        # Verify both agents participated
        agents_in_chat = {msg['name'] for msg in response.chat_history}
        self.assertIn('agent1', agents_in_chat)
        self.assertIn('agent2', agents_in_chat)
```

### Mock Testing

```python
import unittest
from unittest.mock import Mock, patch, MagicMock
from autogen import ConversableAgent, LLMConfig

class TestMocking(unittest.TestCase):
    @patch('autogen.ConversableAgent.run')
    def test_agent_with_mock(self, mock_run):
        """Test agent with mocked response."""
        # Setup mock
        mock_response = Mock()
        mock_response.summary = "Test summary"
        mock_response.chat_history = [
            {"name": "agent1", "content": "Hello"},
            {"name": "agent2", "content": "Hi there"}
        ]
        mock_run.return_value = mock_response
        
        # Create agent
        agent = ConversableAgent(
            name="test",
            llm_config=LLMConfig.from_json("OAI_CONFIG_LIST")
        )
        
        # Test mocked behavior
        response = agent.run(
            recipient=Mock(),
            message="test"
        )
        
        self.assertEqual(response.summary, "Test summary")
        self.assertEqual(len(response.chat_history), 2)
```

---

## Deployment Patterns

### Local Deployment

```python
from autogen import ConversableAgent, GroupChat, GroupChatManager, LLMConfig

def deploy_local():
    """Deploy AG2 agents locally for single-machine use."""
    llm_config = LLMConfig.from_json("OAI_CONFIG_LIST")
    
    # Create agents
    agents = [
        ConversableAgent(
            name="agent1",
            system_message="You are helpful.",
            llm_config=llm_config
        ),
        ConversableAgent(
            name="agent2",
            system_message="You are thorough.",
            llm_config=llm_config
        )
    ]
    
    # Create group chat
    groupchat = GroupChat(
        agents=agents,
        messages=[],
        max_round=10
    )
    
    manager = GroupChatManager(groupchat=groupchat, llm_config=llm_config)
    
    # Start conversation
    agents[0].initiate_chat(
        manager,
        message="Start task"
    )

if __name__ == "__main__":
    deploy_local()
```

### Docker Deployment

**Dockerfile:**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port if running web service
EXPOSE 8000

# Run application
CMD ["python", "main.py"]
```

**requirements.txt:**

```
ag2[openai]==0.2.x
pydantic==2.x
python-dotenv==1.x
```

**main.py:**

```python
import os
from dotenv import load_dotenv
from autogen import ConversableAgent, LLMConfig

load_dotenv()

def main():
    llm_config = LLMConfig({
        "model": "gpt-4",
        "api_key": os.getenv("OPENAI_API_KEY")
    })
    
    agent = ConversableAgent(
        name="production_agent",
        llm_config=llm_config
    )
    
    print("Agent deployed in Docker container")

if __name__ == "__main__":
    main()
```

### REST API Deployment with A2A Protocol

```python
from autogen import ConversableAgent, LLMConfig
from autogen.a2a.server import A2aAgentServer

def deploy_as_rest_service():
    """Deploy AG2 agents as REST services."""
    llm_config = LLMConfig.from_json("OAI_CONFIG_LIST")
    
    agent = ConversableAgent(
        name="api_agent",
        system_message="You process requests via API.",
        llm_config=llm_config
    )
    
    # Create REST server
    server = A2aAgentServer(
        agent=agent,
        url="http://0.0.0.0:8000",
        agent_card={
            "name": "api_agent",
            "description": "REST API for agent communication",
            "version": "1.0.0",
            "capabilities": {
                "streaming": True,
                "function_calling": True
            }
        }
    )
    
    # Start server
    server.start()
    print("Server running on http://0.0.0.0:8000")

if __name__ == "__main__":
    deploy_as_rest_service()
```

### Client Code for REST Service

```python
from autogen.a2a.client import A2aRemoteAgent
from autogen import ConversableAgent, LLMConfig

def use_remote_agent():
    """Use remote AG2 agent via REST API."""
    remote_agent = A2aRemoteAgent(
        url="http://localhost:8000",
        name="remote_assistant"
    )
    
    local_coordinator = ConversableAgent(
        name="coordinator",
        llm_config=LLMConfig.from_json("OAI_CONFIG_LIST")
    )
    
    # Use remote agent like local
    response = local_coordinator.initiate_chat(
        remote_agent,
        message="Process this data"
    )
    
    response.process()
    print(response.summary)

if __name__ == "__main__":
    use_remote_agent()
```

---

## Performance Optimisation

### Parallel Execution with Threading

```python
import concurrent.futures
from autogen import ConversableAgent, LLMConfig

def parallel_agents():
    """Run multiple agents in parallel."""
    llm_config = LLMConfig.from_json("OAI_CONFIG_LIST")
    
    agents = [
        ConversableAgent(
            name=f"agent_{i}",
            llm_config=llm_config
        )
        for i in range(3)
    ]
    
    def run_agent(agent):
        recipient = ConversableAgent("dummy", llm_config=False)
        return agent.run(
            recipient=recipient,
            message="Process task",
            max_turns=2
        )
    
    # Run agents in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(run_agent, agent) for agent in agents]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    
    return results

results = parallel_agents()
```

### Connection Pooling for API Calls

```python
from autogen import LLMConfig
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_optimised_config():
    """Create LLM config with connection pooling."""
    # Create session with connection pooling
    session = requests.Session()
    
    retry_strategy = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504],
        method_whitelist=["HEAD", "GET", "POST"],
        backoff_factor=1
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    config = LLMConfig({
        "model": "gpt-4",
        "api_key": "sk-xxx"
    })
    
    return config

config = create_optimised_config()
```

### Caching Responses

```python
from functools import lru_cache
from autogen import ConversableAgent, LLMConfig

class CachingAgent(ConversableAgent):
    def __init__(self, *args, cache_size=128, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_size = cache_size
        self._response_cache = {}
    
    @lru_cache(maxsize=128)
    def get_response(self, message: str):
        """Get cached response or compute new."""
        # Check cache
        if message in self._response_cache:
            return self._response_cache[message]
        
        # Compute response (simplified)
        response = f"Response to: {message}"
        self._response_cache[message] = response
        
        return response

# Usage
caching = CachingAgent(
    name="cached",
    llm_config=LLMConfig.from_json("OAI_CONFIG_LIST")
)
```

---

## Security Considerations

### API Key Management

```python
import os
from dotenv import load_dotenv
from autogen import LLMConfig

# Load from environment variables
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set in environment")

config = LLMConfig({
    "model": "gpt-4",
    "api_key": api_key
})
```

### Input Validation

```python
from autogen import ConversableAgent, LLMConfig
import re

class SecureAgent(ConversableAgent):
    MAX_MESSAGE_LENGTH = 10000
    
    def validate_message(self, message: str) -> bool:
        """Validate message for security."""
        # Length check
        if len(message) > self.MAX_MESSAGE_LENGTH:
            return False
        
        # Pattern check (prevent injection)
        dangerous_patterns = [
            r'<script',
            r'javascript:',
            r'onerror=',
            r'onclick='
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, message, re.IGNORECASE):
                return False
        
        return True
    
    def send(self, message, recipient, request_reply=True, silent=False):
        if not self.validate_message(message):
            raise ValueError("Message failed validation")
        return super().send(message, recipient, request_reply, silent)

# Usage
secure_agent = SecureAgent(
    name="secure",
    llm_config=LLMConfig.from_json("OAI_CONFIG_LIST")
)
```

### Rate Limiting

```python
import time
from autogen import ConversableAgent, LLMConfig

class RateLimitedAgent(ConversableAgent):
    def __init__(self, *args, requests_per_minute=60, **kwargs):
        super().__init__(*args, **kwargs)
        self.requests_per_minute = requests_per_minute
        self.request_times = []
    
    def check_rate_limit(self):
        """Check if rate limit exceeded."""
        now = time.time()
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        if len(self.request_times) >= self.requests_per_minute:
            wait_time = 60 - (now - self.request_times[0])
            raise Exception(f"Rate limit exceeded. Wait {wait_time:.1f}s")
        
        self.request_times.append(now)
    
    def send(self, message, recipient, request_reply=True, silent=False):
        self.check_rate_limit()
        return super().send(message, recipient, request_reply, silent)

# Usage
rate_limited = RateLimitedAgent(
    name="rate_limited",
    llm_config=LLMConfig.from_json("OAI_CONFIG_LIST"),
    requests_per_minute=30
)
```

---

## Monitoring and Observability

### Metrics Collection

```python
from dataclasses import dataclass
from datetime import datetime
from autogen import ConversableAgent, LLMConfig

@dataclass
class AgentMetrics:
    agent_name: str
    messages_sent: int = 0
    messages_received: int = 0
    total_tokens: int = 0
    errors: int = 0
    start_time: datetime = None
    end_time: datetime = None
    
    def duration(self):
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0

class MetricsCollector:
    def __init__(self):
        self.metrics = {}
    
    def record_message(self, agent_name, tokens=0):
        if agent_name not in self.metrics:
            self.metrics[agent_name] = AgentMetrics(agent_name)
        
        self.metrics[agent_name].messages_sent += 1
        self.metrics[agent_name].total_tokens += tokens
    
    def get_report(self):
        return self.metrics

# Usage
collector = MetricsCollector()
collector.record_message("agent1", tokens=150)
collector.record_message("agent2", tokens=200)
print(collector.get_report())
```

### Health Checks

```python
from enum import Enum
from autogen import ConversableAgent, LLMConfig

class AgentHealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"

class HealthCheckAgent(ConversableAgent):
    def health_check(self) -> AgentHealthStatus:
        """Perform health check."""
        try:
            # Check LLM connectivity
            if not self.llm_config:
                return AgentHealthStatus.UNHEALTHY
            
            # Check agent responsiveness
            if len(self.chat_messages) > 1000:
                return AgentHealthStatus.DEGRADED
            
            return AgentHealthStatus.HEALTHY
        except Exception:
            return AgentHealthStatus.UNHEALTHY

# Usage
health_agent = HealthCheckAgent(
    name="health",
    llm_config=LLMConfig.from_json("OAI_CONFIG_LIST")
)
status = health_agent.health_check()
print(f"Agent status: {status.value}")
```

---

## Integration with LangChain and LlamaIndex

### LangChain Integration

```python
from autogen import ConversableAgent, LLMConfig
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

def use_langchain_llm():
    """Integrate LangChain LLM with AG2."""
    # Create LangChain LLM
    langchain_llm = OpenAI(temperature=0.7)
    
    # Wrap in AG2 config
    config = LLMConfig({
        "model": "text-davinci-003",
        "api_key": "sk-xxx"
    })
    
    # Create agent
    agent = ConversableAgent(
        name="langchain_agent",
        llm_config=config
    )
    
    return agent

agent = use_langchain_llm()
```

### LlamaIndex Integration

```python
from autogen import ConversableAgent, LLMConfig
from llama_index import Document, GPTVectorStoreIndex

def use_llamaindex_retrieval():
    """Integrate LlamaIndex retrieval with AG2."""
    # Create index
    documents = [
        Document(text="Sample document 1"),
        Document(text="Sample document 2")
    ]
    
    index = GPTVectorStoreIndex.from_documents(documents)
    
    # Use with AG2 agent
    config = LLMConfig.from_json("OAI_CONFIG_LIST")
    
    agent = ConversableAgent(
        name="retrieval_agent",
        llm_config=config,
        system_message="You answer questions using retrieved documents."
    )
    
    return agent, index

agent, index = use_llamaindex_retrieval()
```

---

## Advanced Async Execution

### Async Agents

```python
import asyncio
from autogen import ConversableAgent, LLMConfig

async def async_agent_example():
    """Example of async agent execution."""
    llm_config = LLMConfig.from_json("OAI_CONFIG_LIST")
    
    agent = ConversableAgent(
        name="async_agent",
        llm_config=llm_config
    )
    
    # Simulate async processing
    await asyncio.sleep(1)
    print("Agent processing complete")

# Run async
asyncio.run(async_agent_example())
```

### Concurrent Task Processing

```python
import asyncio
from autogen import ConversableAgent, LLMConfig

async def process_tasks_concurrently():
    """Process multiple tasks concurrently."""
    llm_config = LLMConfig.from_json("OAI_CONFIG_LIST")
    
    agents = [
        ConversableAgent(
            name=f"worker_{i}",
            llm_config=llm_config
        )
        for i in range(3)
    ]
    
    async def process_task(agent, task):
        # Simulate async task
        await asyncio.sleep(1)
        return f"{agent.name} processed {task}"
    
    # Run all tasks concurrently
    tasks = [
        process_task(agent, f"task_{i}")
        for i, agent in enumerate(agents)
    ]
    
    results = await asyncio.gather(*tasks)
    return results

# Run
results = asyncio.run(process_tasks_concurrently())
print(results)
```

---

This production guide covers enterprise-grade patterns for deploying and maintaining AG2 systems at scale. Refer to the recipes guide for practical code examples and tutorials.

