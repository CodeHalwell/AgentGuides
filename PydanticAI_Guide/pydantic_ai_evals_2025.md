# Pydantic AI: Powerful Evals Guide (2025)
## Systematic Testing with Pydantic Logfire Integration

**Version:** 1.0.0 (2025)
**Framework:** Pydantic AI Testing & Evaluation
**Focus:** Comprehensive agent evaluation, testing, and quality assurance

---

## Table of Contents

1. [Overview](#overview)
2. [Logfire Integration](#logfire-integration)
3. [Evaluation Frameworks](#evaluation-frameworks)
4. [Unit Testing Agents](#unit-testing-agents)
5. [Integration Testing](#integration-testing)
6. [Performance Benchmarking](#performance-benchmarking)
7. [Quality Metrics](#quality-metrics)
8. [Regression Testing](#regression-testing)
9. [Production Monitoring](#production-monitoring)

---

## Overview

### What are Evals?

**Evaluations (Evals)** are systematic tests that measure AI agent quality across multiple dimensions:

- **Correctness**: Does the agent produce accurate outputs?
- **Consistency**: Are outputs stable across similar inputs?
- **Performance**: Response time, token usage, cost
- **Safety**: Harmful content detection, bias testing
- **Robustness**: Error handling, edge cases

### Why Powerful Evals Matter

```python
# ❌ WITHOUT Evals
# Deploy agent → Hope it works → User complaints → Fix issues → Repeat

# ✅ WITH Evals
# Write evals → Test agent → Measure quality → Fix issues → Deploy confidently
```

**Benefits:**
- Catch issues before production
- Quantify agent improvements
- Prevent regressions
- Build confidence in changes
- Continuous quality monitoring

---

## Logfire Integration

### Automatic Instrumentation

```python
import logfire
from pydantic_ai import Agent
import asyncio

# Configure Logfire
logfire.configure(
    token='your-logfire-token',  # From logfire.pydantic.dev
    service_name='agent-evals',
    environment='testing'
)

# Instrument Pydantic AI automatically
logfire.instrument_pydantic_ai()

# All agent operations are now traced
agent = Agent('openai:gpt-4o')

async def run_eval():
    """Run evaluation with automatic tracing."""

    # Every run is traced to Logfire
    result = await agent.run("What is 2 + 2?")

    # View traces at logfire.pydantic.dev
    return result.output

asyncio.run(run_eval())
```

### Custom Eval Spans

```python
from pydantic_ai import Agent
import logfire
from pydantic import BaseModel
from typing import List

class EvalResult(BaseModel):
    """Evaluation result structure."""
    test_case: str
    expected: str
    actual: str
    passed: bool
    tokens_used: int
    latency_ms: float

@logfire.span("run_eval_suite")
async def run_eval_suite(
    agent: Agent,
    test_cases: List[dict]
) -> List[EvalResult]:
    """Run evaluation suite with Logfire tracing."""

    results = []

    for test_case in test_cases:
        # Create span for each test case
        with logfire.span(
            "eval_test_case",
            test_case=test_case['input']
        ) as span:

            import time
            start = time.time()

            # Run agent
            result = await agent.run(test_case['input'])

            latency = (time.time() - start) * 1000

            # Check result
            passed = result.output.strip() == test_case['expected'].strip()

            # Log metrics
            logfire.info(
                "test_case_complete",
                passed=passed,
                tokens=result.usage().total_tokens,
                latency_ms=latency
            )

            # Store result
            eval_result = EvalResult(
                test_case=test_case['input'],
                expected=test_case['expected'],
                actual=result.output,
                passed=passed,
                tokens_used=result.usage().total_tokens,
                latency_ms=latency
            )

            results.append(eval_result)

    # Log overall metrics
    pass_rate = sum(1 for r in results if r.passed) / len(results)
    avg_latency = sum(r.latency_ms for r in results) / len(results)
    total_tokens = sum(r.tokens_used for r in results)

    logfire.info(
        "eval_suite_complete",
        total_tests=len(results),
        pass_rate=pass_rate,
        avg_latency_ms=avg_latency,
        total_tokens=total_tokens
    )

    return results

# Usage
agent = Agent('openai:gpt-4o')

test_cases = [
    {"input": "What is 2+2?", "expected": "4"},
    {"input": "Capital of France?", "expected": "Paris"},
    {"input": "Largest ocean?", "expected": "Pacific Ocean"}
]

results = await run_eval_suite(agent, test_cases)

# View detailed traces in Logfire dashboard
for result in results:
    status = "✅" if result.passed else "❌"
    print(f"{status} {result.test_case}: {result.actual}")
```

### Logfire Datasets for Evals

```python
import logfire
from pydantic_ai import Agent
from pydantic import BaseModel
from typing import List

class EvalDataset(BaseModel):
    """Structured eval dataset."""
    name: str
    version: str
    test_cases: List[dict]

# Create eval dataset
math_dataset = EvalDataset(
    name="math_reasoning",
    version="1.0",
    test_cases=[
        {
            "input": "If John has 5 apples and gives 2 to Mary, how many does he have?",
            "expected": "3",
            "category": "arithmetic"
        },
        {
            "input": "What is 25% of 80?",
            "expected": "20",
            "category": "percentage"
        }
    ]
)

# Log dataset to Logfire
logfire.info(
    "eval_dataset_created",
    dataset_name=math_dataset.name,
    version=math_dataset.version,
    num_cases=len(math_dataset.test_cases)
)

@logfire.span("run_dataset_eval")
async def evaluate_on_dataset(
    agent: Agent,
    dataset: EvalDataset
) -> dict:
    """Evaluate agent on entire dataset."""

    results_by_category = {}

    for test_case in dataset.test_cases:
        category = test_case.get('category', 'general')

        if category not in results_by_category:
            results_by_category[category] = []

        result = await agent.run(test_case['input'])

        passed = result.output.strip() == test_case['expected']

        results_by_category[category].append({
            'input': test_case['input'],
            'expected': test_case['expected'],
            'actual': result.output,
            'passed': passed
        })

    # Calculate metrics per category
    category_metrics = {}

    for category, results in results_by_category.items():
        pass_rate = sum(1 for r in results if r['passed']) / len(results)

        category_metrics[category] = {
            'pass_rate': pass_rate,
            'total_tests': len(results)
        }

        logfire.info(
            f"category_{category}_results",
            pass_rate=pass_rate,
            total_tests=len(results)
        )

    return category_metrics

# Run evaluation
agent = Agent('openai:gpt-4o')
metrics = await evaluate_on_dataset(agent, math_dataset)

# View in Logfire with filtering by category
print(metrics)
```

---

## Evaluation Frameworks

### Pytest Integration

```python
import pytest
import pytest_asyncio
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel
import logfire

# Configure Logfire for tests
logfire.configure(environment='test')
logfire.instrument_pydantic_ai()

@pytest_asyncio.fixture
async def agent():
    """Create agent for testing."""
    return Agent('openai:gpt-4o')

@pytest_asyncio.fixture
async def test_agent():
    """Create test agent (no API calls)."""
    return Agent(TestModel())

@pytest.mark.asyncio
async def test_simple_query(agent):
    """Test agent with simple query."""

    with logfire.span("test_simple_query"):
        result = await agent.run("What is 2+2?")

        assert "4" in result.output
        assert result.usage().total_tokens > 0

@pytest.mark.asyncio
async def test_structured_output(agent):
    """Test structured output validation."""

    from pydantic import BaseModel

    class MathResult(BaseModel):
        answer: int
        explanation: str

    structured_agent = Agent(
        'openai:gpt-4o',
        output_type=MathResult
    )

    result = await structured_agent.run("What is 10 + 15?")

    assert result.output.answer == 25
    assert len(result.output.explanation) > 0

@pytest.mark.asyncio
async def test_with_mock(test_agent):
    """Test with mocked model (no API calls)."""

    # TestModel returns predefined responses
    result = await test_agent.run("Any query")

    assert result.output is not None
    assert result.usage().total_tokens == 0  # No actual API call

@pytest.mark.parametrize("input,expected", [
    ("2+2", "4"),
    ("5*3", "15"),
    ("10-7", "3"),
])
@pytest.mark.asyncio
async def test_math_operations(agent, input, expected):
    """Parametrized math tests."""

    result = await agent.run(f"What is {input}?")

    assert expected in result.output

# Run tests with: pytest test_agent.py -v
# View test traces in Logfire dashboard
```

### Custom Eval Framework

```python
from pydantic_ai import Agent
from pydantic import BaseModel
from typing import List, Callable, Optional
from enum import Enum
import asyncio
import logfire

class EvalStatus(str, Enum):
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"

class EvalCase(BaseModel):
    """Single evaluation case."""
    name: str
    input: str
    expected: Optional[str] = None
    validator: Optional[str] = None  # Function name for custom validation
    tags: List[str] = []

class EvalSuite(BaseModel):
    """Collection of eval cases."""
    name: str
    description: str
    cases: List[EvalCase]
    validators: dict[str, Callable] = {}

class EvalRunner:
    """Run evaluation suites."""

    def __init__(self, agent: Agent):
        self.agent = agent
        logfire.instrument_pydantic_ai()

    async def run_case(
        self,
        case: EvalCase,
        validators: dict[str, Callable]
    ) -> dict:
        """Run single eval case."""

        with logfire.span("eval_case", case_name=case.name):
            try:
                # Run agent
                result = await self.agent.run(case.input)

                # Validate result
                if case.expected:
                    # Simple string match
                    passed = case.expected.lower() in result.output.lower()
                elif case.validator and case.validator in validators:
                    # Custom validator
                    validator_func = validators[case.validator]
                    passed = await validator_func(case.input, result.output)
                else:
                    # No validation - just check for output
                    passed = len(result.output) > 0

                status = EvalStatus.PASSED if passed else EvalStatus.FAILED

                logfire.info(
                    "eval_case_result",
                    status=status.value,
                    tokens=result.usage().total_tokens
                )

                return {
                    "name": case.name,
                    "status": status,
                    "input": case.input,
                    "output": result.output,
                    "expected": case.expected,
                    "tokens": result.usage().total_tokens
                }

            except Exception as e:
                logfire.error("eval_case_error", error=str(e))

                return {
                    "name": case.name,
                    "status": EvalStatus.ERROR,
                    "error": str(e)
                }

    async def run_suite(self, suite: EvalSuite) -> dict:
        """Run entire eval suite."""

        with logfire.span("eval_suite", suite_name=suite.name):
            logfire.info(
                "eval_suite_start",
                suite_name=suite.name,
                total_cases=len(suite.cases)
            )

            # Run all cases
            results = await asyncio.gather(*[
                self.run_case(case, suite.validators)
                for case in suite.cases
            ])

            # Calculate metrics
            passed = sum(1 for r in results if r.get('status') == EvalStatus.PASSED)
            failed = sum(1 for r in results if r.get('status') == EvalStatus.FAILED)
            errors = sum(1 for r in results if r.get('status') == EvalStatus.ERROR)

            pass_rate = passed / len(results) if results else 0

            summary = {
                "suite_name": suite.name,
                "total_cases": len(results),
                "passed": passed,
                "failed": failed,
                "errors": errors,
                "pass_rate": pass_rate,
                "results": results
            }

            logfire.info(
                "eval_suite_complete",
                **summary
            )

            return summary

# Usage
agent = Agent('openai:gpt-4o')

# Define custom validator
async def validate_code_output(input: str, output: str) -> bool:
    """Validate that output contains valid Python code."""
    return "def " in output or "class " in output

# Create eval suite
suite = EvalSuite(
    name="coding_agent_eval",
    description="Test coding agent capabilities",
    cases=[
        EvalCase(
            name="simple_function",
            input="Write a function to add two numbers",
            validator="validate_code_output",
            tags=["coding", "easy"]
        ),
        EvalCase(
            name="class_definition",
            input="Create a User class with name and email",
            validator="validate_code_output",
            tags=["coding", "medium"]
        ),
        EvalCase(
            name="algorithm",
            input="Implement binary search",
            validator="validate_code_output",
            tags=["coding", "hard"]
        )
    ],
    validators={"validate_code_output": validate_code_output}
)

# Run evaluation
runner = EvalRunner(agent)
results = await runner.run_suite(suite)

print(f"Pass rate: {results['pass_rate']:.1%}")
print(f"Passed: {results['passed']}/{results['total_cases']}")
```

---

## Unit Testing Agents

### TestModel for Unit Tests

```python
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel, FunctionModel
from pydantic import BaseModel
import pytest

# 1. TestModel - Returns simple predefined responses
@pytest.mark.asyncio
async def test_with_test_model():
    """Test agent logic without API calls."""

    agent = Agent(TestModel())

    result = await agent.run("Any query")

    # TestModel returns structured test data
    assert result.output is not None
    assert result.usage().total_tokens == 0  # No actual tokens

# 2. FunctionModel - Custom response logic
@pytest.mark.asyncio
async def test_with_function_model():
    """Test with custom response function."""

    def custom_response(messages):
        """Generate custom response based on input."""
        last_message = messages[-1].content

        if "math" in last_message.lower():
            return "42"
        elif "code" in last_message.lower():
            return "def hello(): print('world')"
        else:
            return "Default response"

    agent = Agent(FunctionModel(custom_response))

    math_result = await agent.run("Solve this math problem")
    assert "42" in math_result.output

    code_result = await agent.run("Write some code")
    assert "def hello" in code_result.output

# 3. Testing Tools
@pytest.mark.asyncio
async def test_agent_tools():
    """Test agent with tools using TestModel."""

    agent = Agent(TestModel())

    @agent.tool
    async def calculate(ctx, a: int, b: int) -> int:
        """Test tool."""
        return a + b

    # Tool logic is tested independently
    from pydantic_ai import RunContext

    result = await calculate(RunContext(deps=None, model=None, messages=[]), 2, 3)
    assert result == 5

# 4. Testing Validators
@pytest.mark.asyncio
async def test_output_validator():
    """Test output validation logic."""

    from pydantic_ai import ModelRetry, RunContext

    class Result(BaseModel):
        value: int

    agent = Agent(TestModel(), output_type=Result)

    @agent.output_validator
    async def validate_positive(ctx: RunContext, output: Result) -> Result:
        """Ensure value is positive."""
        if output.value < 0:
            raise ModelRetry("Value must be positive")
        return output

    # Validator is tested with mock data
    # In production, if model returns negative, it retries
```

### Snapshot Testing

```python
from pydantic_ai import Agent
import pytest
import json
from pathlib import Path

class SnapshotTester:
    """Snapshot testing for agent outputs."""

    def __init__(self, snapshot_dir: Path):
        self.snapshot_dir = snapshot_dir
        self.snapshot_dir.mkdir(exist_ok=True)

    def get_snapshot_path(self, test_name: str) -> Path:
        """Get path for snapshot file."""
        return self.snapshot_dir / f"{test_name}.json"

    async def assert_matches_snapshot(
        self,
        test_name: str,
        actual: str,
        update: bool = False
    ):
        """Assert output matches snapshot."""

        snapshot_path = self.get_snapshot_path(test_name)

        if update or not snapshot_path.exists():
            # Create/update snapshot
            snapshot_path.write_text(json.dumps({"output": actual}, indent=2))
            return

        # Load snapshot
        snapshot = json.loads(snapshot_path.read_text())
        expected = snapshot["output"]

        # Compare
        assert actual == expected, f"Output doesn't match snapshot: {test_name}"

# Usage
@pytest.fixture
def snapshot_tester():
    return SnapshotTester(Path(__file__).parent / "snapshots")

@pytest.mark.asyncio
async def test_consistent_output(snapshot_tester):
    """Test that agent output is consistent."""

    agent = Agent('openai:gpt-4o', instructions="Always respond with 'Hello'")

    result = await agent.run("Say hello")

    # Check against snapshot
    await snapshot_tester.assert_matches_snapshot(
        "test_consistent_output",
        result.output,
        update=False  # Set to True to update snapshots
    )

# Run with: pytest --update-snapshots (custom flag)
```

---

## Integration Testing

### End-to-End Tests

```python
from pydantic_ai import Agent
from pydantic_ai.durable.prefect import PrefectDurableAgent
from pydantic_ai.mcp import MCPClient
import pytest
import logfire

@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_with_mcp_integration():
    """Integration test with MCP server."""

    # Start MCP server
    mcp_client = MCPClient()
    await mcp_client.connect_stdio(
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp/test"]
    )

    # Create agent
    agent = Agent('openai:gpt-4o')

    # Register MCP tools
    tools = await mcp_client.list_tools()
    # ... register tools

    # Test full workflow
    with logfire.span("integration_test_mcp"):
        result = await agent.run("Read file test.txt and summarize")

        assert result.output is not None
        assert len(result.output) > 0

    # Cleanup
    await mcp_client.disconnect()

@pytest.mark.integration
@pytest.mark.asyncio
async def test_durable_agent_integration():
    """Integration test for durable execution."""

    from pydantic_ai.durable import PostgreSQLDurableBackend

    # Initialize backend
    backend = PostgreSQLDurableBackend("postgresql://localhost/test_db")
    await backend.initialize()

    # Create durable agent
    agent = PrefectDurableAgent(
        'openai:gpt-4o',
        backend=backend,
        checkpoint_every_n_steps=1
    )

    workflow_id = "test-workflow-123"

    # Execute workflow
    result = await agent.run(
        "Multi-step task",
        workflow_id=workflow_id
    )

    # Verify checkpoints were created
    checkpoints = await backend.list_checkpoints(workflow_id)
    assert len(checkpoints) > 0

    # Cleanup
    await backend.delete_checkpoints(workflow_id)

@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_multi_agent_coordination():
    """Integration test for multi-agent workflow."""

    from pydantic_ai.a2a import A2AAgentRegistry

    registry = A2AAgentRegistry()

    # Create agents
    researcher = Agent('openai:gpt-4o')
    analyst = Agent('anthropic:claude-3-5-sonnet-latest')
    writer = Agent('openai:gpt-4o')

    # Register agents
    await registry.register_agent(
        "researcher-001",
        ["research"],
        "http://localhost:8001"
    )

    # Execute coordinated workflow
    research_result = await researcher.run("Research AI safety")
    analysis_result = await analyst.run(f"Analyze: {research_result.output}")
    final_report = await writer.run(f"Write report: {analysis_result.output}")

    # Verify workflow completion
    assert len(final_report.output) > 100

# Run with: pytest -m integration
```

---

## Performance Benchmarking

### Latency Benchmarks

```python
from pydantic_ai import Agent
import asyncio
import time
import statistics
import logfire
from typing import List

class LatencyBenchmark:
    """Benchmark agent latency."""

    def __init__(self, agent: Agent):
        self.agent = agent

    async def measure_latency(
        self,
        query: str,
        iterations: int = 10
    ) -> dict:
        """Measure latency over multiple iterations."""

        latencies = []

        with logfire.span("latency_benchmark", iterations=iterations):
            for i in range(iterations):
                start = time.perf_counter()

                result = await self.agent.run(query)

                latency = (time.perf_counter() - start) * 1000  # ms

                latencies.append(latency)

                logfire.info(
                    f"iteration_{i}",
                    latency_ms=latency,
                    tokens=result.usage().total_tokens
                )

        # Calculate statistics
        stats = {
            "mean_ms": statistics.mean(latencies),
            "median_ms": statistics.median(latencies),
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "stddev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            "p95_ms": sorted(latencies)[int(len(latencies) * 0.95)],
            "p99_ms": sorted(latencies)[int(len(latencies) * 0.99)]
        }

        logfire.info("benchmark_results", **stats)

        return stats

# Usage
agent = Agent('openai:gpt-4o')
benchmark = LatencyBenchmark(agent)

results = await benchmark.measure_latency("What is 2+2?", iterations=100)

print(f"Mean latency: {results['mean_ms']:.2f}ms")
print(f"P95 latency: {results['p95_ms']:.2f}ms")
```

### Token Usage Benchmarks

```python
from pydantic_ai import Agent
import logfire

class TokenBenchmark:
    """Benchmark token usage."""

    def __init__(self, agent: Agent):
        self.agent = agent

    async def measure_tokens(
        self,
        queries: List[str]
    ) -> dict:
        """Measure token usage across queries."""

        total_input = 0
        total_output = 0
        total_cost = 0.0

        # Token costs (example for GPT-4o)
        input_cost_per_1k = 0.005
        output_cost_per_1k = 0.015

        with logfire.span("token_benchmark"):
            for query in queries:
                result = await self.agent.run(query)
                usage = result.usage()

                total_input += usage.input_tokens
                total_output += usage.output_tokens

                cost = (
                    (usage.input_tokens / 1000) * input_cost_per_1k +
                    (usage.output_tokens / 1000) * output_cost_per_1k
                )
                total_cost += cost

        avg_input = total_input / len(queries)
        avg_output = total_output / len(queries)
        avg_cost = total_cost / len(queries)

        stats = {
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "avg_input_tokens": avg_input,
            "avg_output_tokens": avg_output,
            "total_cost_usd": total_cost,
            "avg_cost_usd": avg_cost
        }

        logfire.info("token_benchmark_results", **stats)

        return stats

# Usage
queries = [
    "Summarize: " + "long text" * 100,
    "Translate: " + "text" * 50,
    "Analyze: " + "data" * 75
]

benchmark = TokenBenchmark(agent)
results = await benchmark.measure_tokens(queries)

print(f"Total cost: ${results['total_cost_usd']:.4f}")
print(f"Avg cost per query: ${results['avg_cost_usd']:.4f}")
```

---

## Quality Metrics

### Semantic Similarity Eval

```python
from pydantic_ai import Agent
from sentence_transformers import SentenceTransformer
import numpy as np

class SemanticEvaluator:
    """Evaluate semantic similarity of outputs."""

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def cosine_similarity(
        self,
        text1: str,
        text2: str
    ) -> float:
        """Calculate cosine similarity between texts."""

        embeddings = self.model.encode([text1, text2])

        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )

        return float(similarity)

    async def evaluate_similarity(
        self,
        agent: Agent,
        test_cases: List[dict],
        threshold: float = 0.8
    ) -> dict:
        """Evaluate if agent outputs are semantically similar to expected."""

        results = []

        for case in test_cases:
            result = await agent.run(case['input'])

            similarity = self.cosine_similarity(
                result.output,
                case['expected']
            )

            passed = similarity >= threshold

            results.append({
                'input': case['input'],
                'similarity': similarity,
                'passed': passed
            })

        pass_rate = sum(1 for r in results if r['passed']) / len(results)

        return {
            'pass_rate': pass_rate,
            'avg_similarity': np.mean([r['similarity'] for r in results]),
            'results': results
        }

# Usage
evaluator = SemanticEvaluator()

test_cases = [
    {
        "input": "What's the capital of France?",
        "expected": "The capital of France is Paris."
    }
]

results = await evaluator.evaluate_similarity(
    agent,
    test_cases,
    threshold=0.8
)
```

---

## Regression Testing

### Regression Test Suite

```python
from pydantic_ai import Agent
import logfire
import json
from pathlib import Path
from datetime import datetime

class RegressionTester:
    """Detect regressions in agent quality."""

    def __init__(self, baseline_path: Path):
        self.baseline_path = baseline_path

    async def run_regression_tests(
        self,
        agent: Agent,
        test_cases: List[dict]
    ) -> dict:
        """Run tests and compare to baseline."""

        results = []

        for case in test_cases:
            result = await agent.run(case['input'])

            results.append({
                'input': case['input'],
                'output': result.output,
                'tokens': result.usage().total_tokens
            })

        # Load baseline
        if self.baseline_path.exists():
            baseline = json.loads(self.baseline_path.read_text())

            regressions = self._detect_regressions(results, baseline)

            if regressions:
                logfire.error(
                    "regressions_detected",
                    count=len(regressions),
                    regressions=regressions
                )

                return {
                    'status': 'regression',
                    'regressions': regressions
                }

        # Update baseline
        self.baseline_path.write_text(json.dumps({
            'timestamp': datetime.now().isoformat(),
            'results': results
        }, indent=2))

        logfire.info("regression_tests_passed")

        return {'status': 'passed'}

    def _detect_regressions(
        self,
        current: List[dict],
        baseline: dict
    ) -> List[dict]:
        """Compare current results to baseline."""

        regressions = []
        baseline_results = baseline['results']

        for i, (curr, base) in enumerate(zip(current, baseline_results)):
            # Check if output significantly different
            if curr['output'] != base['output']:
                # Could use semantic similarity here
                regressions.append({
                    'test_index': i,
                    'baseline_output': base['output'],
                    'current_output': curr['output']
                })

        return regressions
```

---

## Production Monitoring

### Continuous Evaluation

```python
from pydantic_ai import Agent
import logfire
from fastapi import FastAPI, BackgroundTasks
import asyncio

app = FastAPI()

class ProductionEvaluator:
    """Continuous evaluation in production."""

    def __init__(self, agent: Agent):
        self.agent = agent
        self.eval_queue = asyncio.Queue()

    async def log_production_request(
        self,
        user_query: str,
        agent_response: str,
        metadata: dict
    ):
        """Log production request for eval."""

        # Add to eval queue
        await self.eval_queue.put({
            'query': user_query,
            'response': agent_response,
            'metadata': metadata
        })

    async def run_continuous_eval(self):
        """Continuously evaluate production requests."""

        while True:
            # Get batch of requests
            batch = []
            for _ in range(10):  # Batch size
                if not self.eval_queue.empty():
                    batch.append(await self.eval_queue.get())

            if batch:
                # Run evals on batch
                with logfire.span("production_eval_batch"):
                    for item in batch:
                        # Example: Check for quality issues
                        has_error = "error" in item['response'].lower()
                        too_short = len(item['response']) < 10

                        if has_error or too_short:
                            logfire.warn(
                                "quality_issue_detected",
                                query=item['query'],
                                issue="error" if has_error else "too_short"
                            )

            await asyncio.sleep(60)  # Eval every minute

evaluator = ProductionEvaluator(agent)

@app.on_event("startup")
async def startup():
    """Start continuous evaluation."""
    asyncio.create_task(evaluator.run_continuous_eval())

@app.post("/api/query")
async def query_agent(
    query: str,
    background_tasks: BackgroundTasks
):
    """Handle query with background evaluation."""

    result = await agent.run(query)

    # Log for evaluation in background
    background_tasks.add_task(
        evaluator.log_production_request,
        query,
        result.output,
        {'tokens': result.usage().total_tokens}
    )

    return {"response": result.output}
```

---

## Summary

**Powerful Evals Enable:**
- ✅ Systematic agent quality testing
- ✅ Logfire integration for observability
- ✅ Regression detection
- ✅ Performance benchmarking
- ✅ Production monitoring
- ✅ Continuous improvement

**Best Practices:**
1. Write evals before deploying
2. Use Logfire for comprehensive tracing
3. Test across multiple dimensions (correctness, latency, cost)
4. Maintain baseline results for regression detection
5. Monitor production continuously
6. Update evals as requirements evolve

---

**Next Steps:**
1. Set up Logfire account and instrumentation
2. Create baseline eval suite
3. Run benchmarks for latency and cost
4. Implement regression testing
5. Add production monitoring
6. Iterate based on eval results

**See Also:**
- `pydantic_ai_comprehensive_guide.md` - Core concepts
- `pydantic_ai_production_guide.md` - Production deployment
- `pydantic_ai_durable_execution.md` - Fault tolerance
