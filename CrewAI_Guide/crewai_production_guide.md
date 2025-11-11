# CrewAI Production Guide
## Deploying, Testing, Optimising, and Maintaining Multi-Agent Systems

---

## Table of Contents

1. [Production Fundamentals](#production-fundamentals)
2. [Testing Strategies](#testing-strategies)
3. [Deployment Patterns](#deployment-patterns)
4. [Monitoring and Logging](#monitoring-and-logging)
5. [Performance Optimisation](#performance-optimisation)
6. [Cost Optimisation](#cost-optimisation)
7. [Error Handling and Recovery](#error-handling-and-recovery)
8. [Security Considerations](#security-considerations)
9. [Scaling Strategies](#scaling-strategies)
10. [Integration Patterns](#integration-patterns)

---

## Production Fundamentals

### Development vs. Production Configuration

**Development Environment:**

```python
crew = Crew(
    agents=agents,
    tasks=tasks,
    verbose=True,  # Enable detailed logging
    memory=True,   # Enable memory for debugging
    max_rpm=100    # Higher limit for testing
)
```

**Production Environment:**

```python
crew = Crew(
    agents=agents,
    tasks=tasks,
    verbose=False,  # Minimal output
    memory=True,    # Maintain history for analysis
    max_rpm=10,     # Respect API rate limits
    share_crew_state=True  # Optimise for reliability
)
```

### Configuration Management

Store configuration in environment variables:

```python
import os
from dotenv import load_dotenv

load_dotenv()

# API Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')

# Model Configuration
PRODUCTION_MODEL = os.getenv('PRODUCTION_MODEL', 'openai/gpt-4-turbo')
FALLBACK_MODEL = os.getenv('FALLBACK_MODEL', 'openai/gpt-3.5-turbo')

# Execution Configuration
MAX_RPM = int(os.getenv('MAX_RPM', '10'))
VERBOSE = os.getenv('VERBOSE', 'False').lower() == 'true'

# Create LLM based on configuration
llm = LLM(
    model=PRODUCTION_MODEL,
    api_key=OPENAI_API_KEY,
    temperature=0.5
)
```

### Project Structure for Production

```
my_crewai_project/
├── .env                          # Environment variables
├── .env.example                  # Template for env vars
├── .gitignore
├── README.md
├── requirements.txt
├── setup.py
├── pyproject.toml
│
├── src/
│   └── my_project/
│       ├── __init__.py
│       ├── main.py              # Entry point
│       ├── config.py            # Configuration management
│       ├── logging_config.py    # Logging setup
│       │
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── base_agent.py
│       │   └── specialized_agents.py
│       │
│       ├── tasks/
│       │   ├── __init__.py
│       │   └── task_definitions.py
│       │
│       ├── crews/
│       │   ├── __init__.py
│       │   └── crew_factory.py
│       │
│       ├── tools/
│       │   ├── __init__.py
│       │   └── custom_tools.py
│       │
│       └── utils/
│           ├── __init__.py
│           ├── error_handling.py
│           ├── monitoring.py
│           └── caching.py
│
├── tests/
│   ├── unit/
│   │   ├── test_agents.py
│   │   ├── test_tasks.py
│   │   └── test_tools.py
│   │
│   ├── integration/
│   │   ├── test_crew_execution.py
│   │   └── test_end_to_end.py
│   │
│   └── performance/
│       ├── test_latency.py
│       └── test_throughput.py
│
└── docs/
    ├── deployment.md
    ├── troubleshooting.md
    └── api_reference.md
```

---

## Testing Strategies

### Unit Testing Agents

```python
import pytest
from crewai import Agent, LLM

def test_agent_creation():
    """Test agent initialisation."""
    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory"
    )
    
    assert agent.role == "Test Agent"
    assert agent.goal == "Test goal"
    assert agent.backstory == "Test backstory"

def test_agent_with_tools():
    """Test agent with tool assignment."""
    from crewai_tools import FileReadTool
    
    reader = FileReadTool()
    agent = Agent(
        role="File Reader",
        goal="Read files",
        tools=[reader]
    )
    
    assert len(agent.tools) == 1
    assert agent.tools[0].name == "read_file"

@pytest.mark.asyncio
async def test_agent_async_execution():
    """Test async agent execution."""
    agent = Agent(role="Async Agent", goal="Test async")
    # Test async functionality
    pass
```

### Unit Testing Tasks

```python
def test_task_creation():
    """Test task initialisation."""
    agent = Agent(role="Test", goal="Test")
    
    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=agent
    )
    
    assert task.description == "Test task"
    assert task.agent == agent

def test_task_with_async_execution():
    """Test async task configuration."""
    agent = Agent(role="Test", goal="Test")
    
    task = Task(
        description="Async task",
        expected_output="Output",
        agent=agent,
        async_execution=True
    )
    
    assert task.async_execution is True
```

### Integration Testing Crews

```python
import pytest
from crewai import Agent, Task, Crew, Process, LLM

@pytest.fixture
def test_crew():
    """Create a test crew."""
    agent1 = Agent(
        role="Test Agent 1",
        goal="Test goal 1",
        backstory="Test"
    )
    agent2 = Agent(
        role="Test Agent 2",
        goal="Test goal 2",
        backstory="Test"
    )
    
    task1 = Task(
        description="Test task 1",
        expected_output="Output 1",
        agent=agent1
    )
    
    task2 = Task(
        description="Test task 2",
        expected_output="Output 2",
        agent=agent2
    )
    
    crew = Crew(
        agents=[agent1, agent2],
        tasks=[task1, task2],
        process=Process.sequential
    )
    
    return crew

def test_crew_execution(test_crew):
    """Test crew execution."""
    result = test_crew.kickoff()
    assert result is not None
    assert isinstance(result, (str, dict))

def test_crew_sequential_process(test_crew):
    """Test sequential process execution."""
    assert test_crew.process == Process.sequential
    # Verify tasks execute in order
```

### Performance Testing

```python
import time
import pytest

def test_crew_execution_time():
    """Test crew execution latency."""
    crew = setup_test_crew()
    
    start_time = time.time()
    result = crew.kickoff()
    execution_time = time.time() - start_time
    
    # Assert execution completes within threshold
    assert execution_time < 30, f"Execution took {execution_time}s, threshold is 30s"

def test_crew_throughput():
    """Test crew throughput."""
    crew = setup_test_crew()
    
    start_time = time.time()
    
    for _ in range(10):
        crew.kickoff()
    
    total_time = time.time() - start_time
    avg_time_per_exec = total_time / 10
    
    assert avg_time_per_exec < 5, f"Average execution time {avg_time_per_exec}s exceeds threshold"

@pytest.mark.performance
def test_memory_usage():
    """Test memory consumption."""
    import psutil
    
    crew = setup_test_crew()
    
    process = psutil.Process()
    memory_before = process.memory_info().rss / 1024 / 1024  # MB
    
    crew.kickoff()
    
    memory_after = process.memory_info().rss / 1024 / 1024
    memory_used = memory_after - memory_before
    
    assert memory_used < 500, f"Memory usage {memory_used}MB exceeds threshold"
```

---

## Deployment Patterns

### Docker Deployment

**Dockerfile:**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY .env .

# Set environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Run application
CMD ["python", "-m", "src.my_project.main"]
```

**Docker Compose:**

```yaml
version: '3.8'

services:
  crewai-app:
    build: .
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - LOG_LEVEL=INFO
    volumes:
      - ./data:/app/data
    ports:
      - "8000:8000"
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  redis_data:
```

### FastAPI Integration

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from crewai import Crew

app = FastAPI(title="CrewAI Service")

class TaskRequest(BaseModel):
    topic: str
    depth: str = "comprehensive"
    preferences: Optional[dict] = None

class TaskResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[str] = None

@app.post("/execute", response_model=TaskResponse)
async def execute_crew(request: TaskRequest):
    """Execute crew task via API."""
    try:
        crew = create_crew_for_task(request.topic, request.depth)
        result = crew.kickoff()
        
        return TaskResponse(
            task_id="generated_id",
            status="completed",
            result=result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
```

### Kubernetes Deployment

**deployment.yaml:**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: crewai-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: crewai
  template:
    metadata:
      labels:
        app: crewai
    spec:
      containers:
      - name: crewai
        image: my-registry/crewai:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: api-keys
              key: openai
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

---

## Monitoring and Logging

### Comprehensive Logging Setup

```python
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crew_execution.log'),
        logging.StreamHandler()
    ]
)

class CrewLogger:
    """Custom logger for CrewAI operations."""
    
    def __init__(self, name):
        self.logger = logging.getLogger(name)
    
    def log_agent_action(self, agent_name, action, details):
        """Log agent actions."""
        self.logger.info(json.dumps({
            'timestamp': datetime.now().isoformat(),
            'type': 'agent_action',
            'agent': agent_name,
            'action': action,
            'details': details
        }))
    
    def log_task_start(self, task_name):
        """Log task start."""
        self.logger.info(f"Task started: {task_name}")
    
    def log_task_complete(self, task_name, duration, success=True):
        """Log task completion."""
        status = "SUCCESS" if success else "FAILED"
        self.logger.info(f"Task {task_name} {status} (Duration: {duration}s)")
    
    def log_error(self, component, error):
        """Log errors."""
        self.logger.error(f"{component} error: {str(error)}", exc_info=True)

# Usage
logger = CrewLogger(__name__)
logger.log_task_start("research_task")
```

### Monitoring Metrics

```python
from dataclasses import dataclass
from typing import Dict, List
import time

@dataclass
class ExecutionMetrics:
    """Metrics for crew execution."""
    crew_name: str
    start_time: float
    end_time: float
    total_agents: int
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    total_tokens_used: int
    total_cost: float
    
    @property
    def execution_time(self) -> float:
        return self.end_time - self.start_time
    
    @property
    def success_rate(self) -> float:
        total = self.successful_tasks + self.failed_tasks
        return (self.successful_tasks / total * 100) if total > 0 else 0
    
    def to_dict(self) -> Dict:
        return {
            'crew_name': self.crew_name,
            'execution_time': self.execution_time,
            'success_rate': self.success_rate,
            'total_agents': self.total_agents,
            'total_tasks': self.total_tasks,
            'total_tokens_used': self.total_tokens_used,
            'total_cost': self.total_cost
        }

class MetricsCollector:
    """Collect and track execution metrics."""
    
    def __init__(self):
        self.metrics: List[ExecutionMetrics] = []
    
    def add_metrics(self, metrics: ExecutionMetrics):
        """Add metrics record."""
        self.metrics.append(metrics)
    
    def get_average_execution_time(self) -> float:
        """Get average execution time."""
        if not self.metrics:
            return 0
        total_time = sum(m.execution_time for m in self.metrics)
        return total_time / len(self.metrics)
    
    def get_overall_success_rate(self) -> float:
        """Get overall success rate."""
        if not self.metrics:
            return 0
        total_successful = sum(m.successful_tasks for m in self.metrics)
        total_tasks = sum(m.successful_tasks + m.failed_tasks for m in self.metrics)
        return (total_successful / total_tasks * 100) if total_tasks > 0 else 0
```

---

## Performance Optimisation

### LLM Caching Strategy

```python
from functools import lru_cache
from typing import Tuple

class LLMCache:
    """Cache LLM responses to reduce API calls."""
    
    def __init__(self, cache_size=128):
        self.cache = {}
        self.cache_size = cache_size
    
    def get_cache_key(self, prompt: str, model: str) -> str:
        """Generate cache key from prompt and model."""
        import hashlib
        combined = f"{model}:{prompt}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def get(self, prompt: str, model: str) -> str | None:
        """Retrieve cached response."""
        key = self.get_cache_key(prompt, model)
        return self.cache.get(key)
    
    def set(self, prompt: str, model: str, response: str):
        """Cache response."""
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        key = self.get_cache_key(prompt, model)
        self.cache[key] = response
    
    def clear(self):
        """Clear cache."""
        self.cache.clear()

# Usage
cache = LLMCache(cache_size=256)

# Before API call
cached_response = cache.get(prompt, model_name)
if cached_response:
    return cached_response

# After API call
response = llm.call(prompt)
cache.set(prompt, model_name, response)
```

### Parallel Task Execution

```python
import asyncio
from typing import List

async def execute_tasks_concurrently(tasks: List[Task], agents: List[Agent]) -> List[str]:
    """Execute multiple tasks concurrently."""
    
    async def run_task(task, agent):
        # Convert synchronous task execution to async
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, task.execute, agent)
    
    # Run all tasks concurrently
    results = await asyncio.gather(
        *[run_task(task, agent) for task, agent in zip(tasks, agents)]
    )
    
    return results

# Usage
async def main():
    tasks = [task1, task2, task3]
    agents = [agent1, agent2, agent3]
    
    results = await execute_tasks_concurrently(tasks, agents)
    print(f"All tasks completed: {len(results)} results")

# Run async function
asyncio.run(main())
```

### Memory Optimisation

```python
class OptimisedMemoryManager:
    """Manage memory efficiently."""
    
    def __init__(self, max_memory_mb=512):
        self.max_memory = max_memory_mb * 1024 * 1024  # Convert to bytes
        self.memory_usage = {}
    
    def should_cleanup(self) -> bool:
        """Check if cleanup is needed."""
        import psutil
        process = psutil.Process()
        current_memory = process.memory_info().rss
        return current_memory > self.max_memory
    
    def cleanup_old_entries(self):
        """Remove oldest memory entries."""
        # Remove least recently used entries
        sorted_entries = sorted(
            self.memory_usage.items(),
            key=lambda x: x[1]['timestamp']
        )
        
        # Remove oldest 25%
        entries_to_remove = len(sorted_entries) // 4
        for key, _ in sorted_entries[:entries_to_remove]:
            del self.memory_usage[key]
    
    def add_memory_entry(self, key: str, value: dict):
        """Add memory entry with cleanup."""
        if self.should_cleanup():
            self.cleanup_old_entries()
        
        from datetime import datetime
        self.memory_usage[key] = {
            **value,
            'timestamp': datetime.now()
        }
```

---

## Cost Optimisation

### Token Cost Tracking

```python
class CostTracker:
    """Track and optimise API costs."""
    
    # Pricing per 1K tokens (as of 2024)
    PRICING = {
        'gpt-4-turbo': {'input': 0.01, 'output': 0.03},
        'gpt-3.5-turbo': {'input': 0.0005, 'output': 0.0015},
        'claude-3-opus': {'input': 0.015, 'output': 0.075},
        'claude-3-sonnet': {'input': 0.003, 'output': 0.015}
    }
    
    def __init__(self):
        self.total_cost = 0.0
        self.token_usage = {}
    
    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for API call."""
        if model not in self.PRICING:
            return 0.0
        
        pricing = self.PRICING[model]
        input_cost = (input_tokens / 1000) * pricing['input']
        output_cost = (output_tokens / 1000) * pricing['output']
        
        total = input_cost + output_cost
        self.total_cost += total
        
        # Track usage
        if model not in self.token_usage:
            self.token_usage[model] = {'input': 0, 'output': 0}
        
        self.token_usage[model]['input'] += input_tokens
        self.token_usage[model]['output'] += output_tokens
        
        return total
    
    def get_cost_report(self) -> dict:
        """Get comprehensive cost report."""
        return {
            'total_cost': f"${self.total_cost:.4f}",
            'by_model': {
                model: {
                    'input_tokens': self.token_usage[model]['input'],
                    'output_tokens': self.token_usage[model]['output'],
                    'cost': f"${self.calculate_model_cost(model):.4f}"
                }
                for model in self.token_usage
            }
        }
    
    def calculate_model_cost(self, model: str) -> float:
        """Calculate cost for specific model."""
        pricing = self.PRICING.get(model, {'input': 0, 'output': 0})
        tokens = self.token_usage.get(model, {'input': 0, 'output': 0})
        
        input_cost = (tokens['input'] / 1000) * pricing['input']
        output_cost = (tokens['output'] / 1000) * pricing['output']
        
        return input_cost + output_cost

# Usage
tracker = CostTracker()

# After each LLM call
cost = tracker.calculate_cost('gpt-4-turbo', input_tokens=500, output_tokens=200)
print(f"Cost for this call: ${cost:.4f}")

# Get overall report
report = tracker.get_cost_report()
print(f"Total cost: {report['total_cost']}")
```

---

## Error Handling and Recovery

### Robust Error Handling

```python
from typing import Optional, Callable
import time

class TaskExecutor:
    """Execute tasks with error handling and retry logic."""
    
    def __init__(self, max_retries=3, backoff_factor=2):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
    
    def execute_with_retry(
        self,
        task: Task,
        on_error: Optional[Callable] = None
    ) -> str:
        """Execute task with automatic retry."""
        
        for attempt in range(self.max_retries):
            try:
                return self.execute_task(task)
            
            except Exception as e:
                if attempt < self.max_retries - 1:
                    # Calculate backoff time
                    wait_time = self.backoff_factor ** attempt
                    logger.warning(
                        f"Task failed (attempt {attempt + 1}/{self.max_retries}). "
                        f"Retrying in {wait_time}s. Error: {str(e)}"
                    )
                    time.sleep(wait_time)
                else:
                    # All retries exhausted
                    if on_error:
                        on_error(e)
                    raise
    
    def execute_task(self, task: Task) -> str:
        """Execute single task."""
        # Implementation
        pass

# Usage
executor = TaskExecutor(max_retries=3)

try:
    result = executor.execute_with_retry(
        task,
        on_error=lambda e: logger.error(f"Task failed: {e}")
    )
except Exception as e:
    logger.critical(f"Task execution failed after retries: {e}")
```

---

## Security Considerations

### API Key Management

```python
import os
from cryptography.fernet import Fernet

class SecureKeyManager:
    """Manage API keys securely."""
    
    def __init__(self, master_key: Optional[str] = None):
        if not master_key:
            master_key = os.getenv('MASTER_KEY')
        
        self.cipher = Fernet(master_key.encode())
    
    def encrypt_key(self, api_key: str) -> str:
        """Encrypt API key."""
        return self.cipher.encrypt(api_key.encode()).decode()
    
    def decrypt_key(self, encrypted_key: str) -> str:
        """Decrypt API key."""
        return self.cipher.decrypt(encrypted_key.encode()).decode()
    
    @staticmethod
    def load_from_env() -> dict:
        """Load keys from environment."""
        return {
            'openai': os.getenv('OPENAI_API_KEY'),
            'anthropic': os.getenv('ANTHROPIC_API_KEY'),
            'serper': os.getenv('SERPER_API_KEY')
        }

# Usage
manager = SecureKeyManager()

# Encrypt keys for storage
encrypted = manager.encrypt_key(api_key)

# Decrypt keys for use
api_key = manager.decrypt_key(encrypted)
```

---

## Scaling Strategies

### Horizontal Scaling with Load Balancing

```python
from typing import List
import random

class CrewLoadBalancer:
    """Distribute tasks across multiple crew instances."""
    
    def __init__(self, crews: List[Crew]):
        self.crews = crews
    
    def execute_balanced(self, task: Task) -> str:
        """Execute task on least loaded crew."""
        # Select crew with lowest recent task count
        selected_crew = min(self.crews, key=lambda c: c.task_count)
        return selected_crew.execute_task(task)
    
    def execute_parallel(self, tasks: List[Task]) -> List[str]:
        """Execute tasks in parallel across crews."""
        results = []
        for task, crew in zip(tasks, self.crews):
            result = crew.execute_task(task)
            results.append(result)
        return results

# Usage
crews = [create_crew() for _ in range(3)]
balancer = CrewLoadBalancer(crews)

results = balancer.execute_parallel([task1, task2, task3])
```

---

## Integration Patterns

### Database Integration

```python
from sqlalchemy import create_engine, Column, String, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class ExecutionRecord(Base):
    """Store crew execution records."""
    __tablename__ = 'executions'
    
    id = Column(Integer, primary_key=True)
    crew_name = Column(String)
    task_name = Column(String)
    status = Column(String)  # success, failed, pending
    result = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)

# Create database session
engine = create_engine('sqlite:///crew_execution.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

class ExecutionDatabase:
    """Manage execution records."""
    
    def __init__(self):
        self.session = Session()
    
    def record_execution(self, crew_name: str, task_name: str, result: str, status: str):
        """Record execution."""
        record = ExecutionRecord(
            crew_name=crew_name,
            task_name=task_name,
            result=result,
            status=status,
            completed_at=datetime.utcnow()
        )
        self.session.add(record)
        self.session.commit()
    
    def get_execution_history(self, crew_name: str) -> List[ExecutionRecord]:
        """Get execution history."""
        return self.session.query(ExecutionRecord).filter(
            ExecutionRecord.crew_name == crew_name
        ).order_by(ExecutionRecord.created_at.desc()).all()
```

This production guide provides comprehensive strategies for deploying, monitoring, and optimising CrewAI systems in production environments.


