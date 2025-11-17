# CrewAI Flows - Comprehensive Guide (2025)
## Event-Driven Workflows for Agentic AI Automation

---

## Table of Contents

1. [Introduction to Flows](#introduction-to-flows)
2. [Core Concepts](#core-concepts)
3. [Getting Started](#getting-started)
4. [Flow Decorators](#flow-decorators)
5. [Integrating Crews with Flows](#integrating-crews-with-flows)
6. [Conditional Logic and Routing](#conditional-logic-and-routing)
7. [Loops and Iterations](#loops-and-iterations)
8. [State Management](#state-management)
9. [Error Handling](#error-handling)
10. [Advanced Patterns](#advanced-patterns)
11. [Production Best Practices](#production-best-practices)
12. [Real-World Examples](#real-world-examples)

---

## Introduction to Flows

### What Are CrewAI Flows?

**CrewAI Flows** is the revolutionary 2025 feature that extends CrewAI beyond traditional task-based agent orchestration into sophisticated event-driven workflow automation. Flows enable you to:

- Build **event-driven applications** that react to triggers and conditions
- Implement **complex control flow** with conditional branching and loops
- Maintain **persistent state** across multiple workflow executions
- Combine **Python code, LLM calls, and crew executions** seamlessly
- Create **modular, reusable** workflow components

### Why Use Flows?

Traditional CrewAI crews execute tasks sequentially. While powerful, they lack:

- **Conditional branching**: Route execution based on results
- **Loops**: Process collections or retry operations
- **Event handling**: Respond to external triggers
- **State persistence**: Maintain context across executions
- **Complex orchestration**: Mix code logic with agent execution

**Flows solve all these limitations.**

### Key Features

1. **Event-Driven Architecture**: Methods trigger based on completion of other methods
2. **Declarative Workflow**: Define workflows using Python decorators
3. **Type Safety**: Full Python type hints and IDE support
4. **Observability**: Built-in state inspection and monitoring
5. **Crew Integration**: Seamlessly incorporate CrewAI crews into flows
6. **Flexibility**: Combine imperative code with declarative agents

---

## Core Concepts

### Flow Class

The `Flow` class is the foundation of all workflows. It provides:

- Event management and dispatching
- State tracking and persistence
- Execution orchestration
- Error handling infrastructure

```python
from crewai.flow.flow import Flow

class MyFlow(Flow):
    """Base flow class for custom workflows"""
    pass
```

### Flow Lifecycle

1. **Initialization**: Flow instance is created
2. **Start**: Entry point method executes (marked with `@start()`)
3. **Event Propagation**: Listeners trigger based on completed methods
4. **State Updates**: Flow state updates as methods complete
5. **Completion**: Final method returns result

### Flow Execution Model

Flows use an **event-driven execution model**:

```
@start() → triggers → @listen(start_method) → triggers → @listen(next_method) → ...
```

Each method can trigger one or more listeners, creating a directed acyclic graph (DAG) of execution.

---

## Getting Started

### Installation

Flows are included in CrewAI 1.4.1+:

```bash
pip install crewai>=1.4.1
# Or with UV:
uv pip install crewai>=1.4.1
```

### Your First Flow

```python
from crewai.flow.flow import Flow, listen, start

class HelloFlow(Flow):
    """Simple introductory flow"""

    @start()
    def greet(self):
        """Entry point - greet the user"""
        print("Hello from CrewAI Flows!")
        return {"message": "Welcome"}

    @listen(greet)
    def respond(self, greeting_data):
        """Respond to greeting"""
        print(f"Received: {greeting_data['message']}")
        return {"response": "Thank you!"}

    @listen(respond)
    def finalize(self, response_data):
        """Final step"""
        print(f"Flow complete: {response_data['response']}")
        return {"status": "completed"}

# Execute the flow
flow = HelloFlow()
result = flow.kickoff()
print(f"Final result: {result}")
```

**Output:**
```
Hello from CrewAI Flows!
Received: Welcome
Flow complete: Thank you!
Final result: {'status': 'completed'}
```

### Flow with Parameters

```python
from crewai.flow.flow import Flow, listen, start

class ParameterizedFlow(Flow):
    """Flow that accepts input parameters"""

    @start()
    def process_input(self):
        """Accept and process input"""
        # You can pass initial data when calling kickoff()
        initial_data = self.initial_state
        name = initial_data.get('name', 'Guest')
        print(f"Processing request for: {name}")
        return {"name": name, "processed": True}

    @listen(process_input)
    def generate_response(self, data):
        """Generate personalized response"""
        greeting = f"Hello, {data['name']}! Welcome to CrewAI Flows."
        return {"greeting": greeting}

# Execute with parameters
flow = ParameterizedFlow()
result = flow.kickoff(inputs={"name": "Alice"})
print(result)
```

---

## Flow Decorators

### @start()

Marks the entry point of a flow. Every flow must have exactly one `@start()` method.

```python
@start()
def initialize(self):
    """Flow entry point"""
    print("Flow starting...")
    return {"status": "initialized"}
```

**Characteristics:**
- Executes first when `flow.kickoff()` is called
- Returns data that flows to listener methods
- Can access `self.initial_state` for input parameters

### @listen()

Creates an event listener that executes when a specified method completes.

```python
@listen(initialize)
def process_data(self, init_data):
    """Executes after initialize() completes"""
    print(f"Received: {init_data}")
    return {"processed": True}
```

**Characteristics:**
- Receives output from the listened-to method as parameter
- Can listen to multiple methods: `@listen(method1, method2)`
- Triggers immediately after listened method completes

### @router()

Implements conditional routing based on method output.

```python
from crewai.flow.flow import router

@router(process_data)
def route_by_status(self, data):
    """Route based on processing result"""
    if data.get('error'):
        return 'handle_error'
    elif data.get('requires_review'):
        return 'manual_review'
    else:
        return 'auto_approve'

@listen('handle_error')
def handle_error(self, data):
    """Handle error condition"""
    print(f"Error: {data.get('error')}")
    return {"status": "error_handled"}

@listen('auto_approve')
def auto_approve(self, data):
    """Auto-approve successful processing"""
    print("Auto-approving...")
    return {"status": "approved"}
```

**Characteristics:**
- Returns string name of next method to execute
- Enables branching logic and conditional flows
- Router method itself doesn't process data, just routes

---

## Integrating Crews with Flows

### Basic Crew Integration

```python
from crewai import Agent, Task, Crew, LLM
from crewai.flow.flow import Flow, listen, start

class CrewIntegrationFlow(Flow):
    """Flow that executes a crew"""

    def __init__(self):
        super().__init__()
        self.llm = LLM(model="openai/gpt-4-turbo")

    @start()
    def prepare_task(self):
        """Define the task for the crew"""
        return {
            "topic": "Artificial Intelligence in Healthcare",
            "requirements": "Write a 1000-word analysis"
        }

    @listen(prepare_task)
    def execute_research_crew(self, task_data):
        """Execute a research crew"""
        # Create agent
        researcher = Agent(
            role="AI Healthcare Researcher",
            goal=f"Research {task_data['topic']} comprehensively",
            backstory="Expert researcher with deep knowledge of AI and healthcare",
            llm=self.llm,
            verbose=True
        )

        # Create task
        research_task = Task(
            description=f"Research {task_data['topic']}. {task_data['requirements']}",
            expected_output="Comprehensive research report",
            agent=researcher
        )

        # Execute crew
        crew = Crew(
            agents=[researcher],
            tasks=[research_task],
            verbose=True
        )

        result = crew.kickoff()
        return {"research": result}

    @listen(execute_research_crew)
    def summarize_findings(self, research_data):
        """Summarize the research findings"""
        print("Research Complete!")
        print(f"Findings: {research_data['research']}")
        return {"summary": "Research completed successfully"}

# Execute
flow = CrewIntegrationFlow()
result = flow.kickoff()
```

### Multi-Crew Workflow

```python
from crewai import Agent, Task, Crew, LLM
from crewai.flow.flow import Flow, listen, start

class MultiCrewFlow(Flow):
    """Coordinate multiple specialized crews"""

    def __init__(self):
        super().__init__()
        self.llm = LLM(model="openai/gpt-4-turbo")

    @start()
    def define_project(self):
        """Define project parameters"""
        return {
            "product": "AI-Powered Task Manager",
            "target_market": "Remote Teams",
            "timeline": "6 months"
        }

    @listen(define_project)
    def market_research_crew(self, project_data):
        """Execute market research crew"""
        researcher = Agent(
            role="Market Research Analyst",
            goal="Analyze market opportunity and competition",
            backstory="Expert market researcher with 10+ years experience",
            llm=self.llm
        )

        task = Task(
            description=f"Research market for {project_data['product']} targeting {project_data['target_market']}",
            expected_output="Market analysis report with TAM, SAM, SOM, and competitive landscape",
            agent=researcher
        )

        crew = Crew(agents=[researcher], tasks=[task], verbose=True)
        result = crew.kickoff()
        return {"market_research": result}

    @listen(market_research_crew)
    def product_planning_crew(self, research_data):
        """Execute product planning crew"""
        product_manager = Agent(
            role="Product Manager",
            goal="Define product requirements and roadmap",
            backstory="Senior product manager with successful product launches",
            llm=self.llm
        )

        task = Task(
            description=f"Based on market research: {research_data['market_research']}, define product requirements and roadmap",
            expected_output="Product requirements document with feature prioritization",
            agent=product_manager
        )

        crew = Crew(agents=[product_manager], tasks=[task], verbose=True)
        result = crew.kickoff()
        return {"product_plan": result}

    @listen(product_planning_crew)
    def technical_design_crew(self, plan_data):
        """Execute technical design crew"""
        architect = Agent(
            role="Technical Architect",
            goal="Design scalable system architecture",
            backstory="Experienced architect with expertise in cloud-native systems",
            llm=self.llm
        )

        task = Task(
            description=f"Design technical architecture for: {plan_data['product_plan']}",
            expected_output="Technical architecture document with system design diagrams",
            agent=architect
        )

        crew = Crew(agents=[architect], tasks=[task], verbose=True)
        result = crew.kickoff()
        return {"technical_design": result}

    @listen(technical_design_crew)
    def finalize_project(self, design_data):
        """Compile final project plan"""
        print("\n=== PROJECT PLANNING COMPLETE ===")
        print("All crews have executed successfully!")
        return {
            "status": "complete",
            "deliverables": [
                "Market Research Report",
                "Product Requirements Document",
                "Technical Architecture Design"
            ]
        }

# Execute multi-crew workflow
flow = MultiCrewFlow()
final_result = flow.kickoff()
print(f"\nFinal Result: {final_result}")
```

---

## Conditional Logic and Routing

### Simple Conditional Routing

```python
from crewai.flow.flow import Flow, listen, start, router

class ConditionalFlow(Flow):
    """Flow with conditional branching"""

    @start()
    def analyze_input(self):
        """Analyze input and determine path"""
        # Simulate analysis
        data_type = "urgent"  # Could be: urgent, normal, low_priority
        return {"type": data_type, "data": "sample data"}

    @router(analyze_input)
    def route_by_priority(self, data):
        """Route based on data type"""
        priority = data['type']

        if priority == 'urgent':
            return 'handle_urgent'
        elif priority == 'normal':
            return 'handle_normal'
        else:
            return 'handle_low_priority'

    @listen('handle_urgent')
    def handle_urgent(self, data):
        """Handle urgent requests"""
        print(f"URGENT HANDLER: Processing {data}")
        # Escalate, notify, fast-track
        return {"status": "escalated", "priority": "urgent"}

    @listen('handle_normal')
    def handle_normal(self, data):
        """Handle normal requests"""
        print(f"NORMAL HANDLER: Processing {data}")
        # Standard processing
        return {"status": "processed", "priority": "normal"}

    @listen('handle_low_priority')
    def handle_low_priority(self, data):
        """Handle low priority requests"""
        print(f"LOW PRIORITY HANDLER: Queueing {data}")
        # Queue for batch processing
        return {"status": "queued", "priority": "low"}

# Execute
flow = ConditionalFlow()
result = flow.kickoff()
```

### Multi-Condition Routing

```python
from crewai.flow.flow import Flow, listen, start, router

class ComplexRoutingFlow(Flow):
    """Flow with complex multi-condition routing"""

    @start()
    def receive_request(self):
        """Receive and classify request"""
        return {
            "type": "support_ticket",
            "severity": "high",
            "category": "technical",
            "customer_tier": "enterprise"
        }

    @router(receive_request)
    def route_request(self, request):
        """Complex routing logic"""
        # Route based on multiple conditions
        if request['customer_tier'] == 'enterprise' and request['severity'] == 'high':
            return 'vip_fast_track'
        elif request['category'] == 'technical' and request['severity'] in ['high', 'critical']:
            return 'technical_escalation'
        elif request['category'] == 'billing':
            return 'billing_department'
        else:
            return 'standard_queue'

    @listen('vip_fast_track')
    def vip_fast_track(self, request):
        """VIP fast-track processing"""
        print(f"VIP FAST TRACK: Immediate attention for {request['customer_tier']} customer")
        return {"sla": "1 hour", "assigned_to": "senior_team"}

    @listen('technical_escalation')
    def technical_escalation(self, request):
        """Technical escalation path"""
        print(f"TECHNICAL ESCALATION: Severity {request['severity']}")
        return {"sla": "4 hours", "assigned_to": "engineering_team"}

    @listen('billing_department')
    def billing_department(self, request):
        """Billing department handling"""
        print("BILLING: Routing to billing specialists")
        return {"sla": "24 hours", "assigned_to": "billing_team"}

    @listen('standard_queue')
    def standard_queue(self, request):
        """Standard queue processing"""
        print("STANDARD: Added to general queue")
        return {"sla": "48 hours", "assigned_to": "support_team"}

# Execute
flow = ComplexRoutingFlow()
result = flow.kickoff()
```

---

## Loops and Iterations

### Processing Collections

```python
from crewai.flow.flow import Flow, listen, start

class IterationFlow(Flow):
    """Flow that processes multiple items"""

    @start()
    def get_items(self):
        """Retrieve items to process"""
        return {
            "items": [
                {"id": 1, "name": "Task A", "priority": "high"},
                {"id": 2, "name": "Task B", "priority": "medium"},
                {"id": 3, "name": "Task C", "priority": "low"},
                {"id": 4, "name": "Task D", "priority": "high"},
                {"id": 5, "name": "Task E", "priority": "medium"}
            ]
        }

    @listen(get_items)
    def process_items(self, data):
        """Process each item iteratively"""
        results = []

        for item in data['items']:
            processed = self.process_single_item(item)
            results.append(processed)
            print(f"Processed: {item['name']} (Priority: {item['priority']})")

        return {"results": results, "total": len(results)}

    def process_single_item(self, item):
        """Process individual item (helper method)"""
        # Processing logic
        return {
            "id": item['id'],
            "name": item['name'],
            "status": "completed",
            "priority": item['priority']
        }

    @listen(process_items)
    def generate_summary(self, processing_data):
        """Generate summary of processed items"""
        total = processing_data['total']
        results = processing_data['results']

        # Count by priority
        priority_counts = {}
        for result in results:
            priority = result['priority']
            priority_counts[priority] = priority_counts.get(priority, 0) + 1

        summary = {
            "total_processed": total,
            "by_priority": priority_counts
        }

        print(f"\n=== SUMMARY ===")
        print(f"Total processed: {total}")
        print(f"By priority: {priority_counts}")

        return summary

# Execute
flow = IterationFlow()
result = flow.kickoff()
```

### Batch Processing with Crews

```python
from crewai import Agent, Task, Crew, LLM
from crewai.flow.flow import Flow, listen, start

class BatchCrewFlow(Flow):
    """Process multiple items using crews"""

    def __init__(self):
        super().__init__()
        self.llm = LLM(model="openai/gpt-4-turbo")

    @start()
    def get_articles_to_review(self):
        """Get list of articles needing review"""
        return {
            "articles": [
                {"title": "AI Trends 2025", "content": "Article 1 content..."},
                {"title": "Machine Learning Best Practices", "content": "Article 2 content..."},
                {"title": "Neural Network Architectures", "content": "Article 3 content..."}
            ]
        }

    @listen(get_articles_to_review)
    def review_articles(self, articles_data):
        """Review each article using a crew"""
        reviewer = Agent(
            role="Content Reviewer",
            goal="Review articles for quality, accuracy, and clarity",
            backstory="Expert editor with 15 years experience",
            llm=self.llm
        )

        reviews = []

        for article in articles_data['articles']:
            # Create task for each article
            review_task = Task(
                description=f"Review this article: {article['title']}\nContent: {article['content']}",
                expected_output="Review feedback with score (1-10) and suggestions",
                agent=reviewer
            )

            crew = Crew(agents=[reviewer], tasks=[review_task])
            review_result = crew.kickoff()

            reviews.append({
                "title": article['title'],
                "review": review_result
            })

            print(f"Reviewed: {article['title']}")

        return {"reviews": reviews, "count": len(reviews)}

    @listen(review_articles)
    def compile_report(self, review_data):
        """Compile final review report"""
        print(f"\n=== REVIEW COMPLETE ===")
        print(f"Reviewed {review_data['count']} articles")
        return {"status": "complete", "reviews": review_data['reviews']}

# Execute
flow = BatchCrewFlow()
result = flow.kickoff()
```

---

## State Management

### Persistent State Across Methods

```python
from crewai.flow.flow import Flow, listen, start

class StatefulFlow(Flow):
    """Flow that maintains state across execution"""

    def __init__(self):
        super().__init__()
        # Initialize persistent state
        self.state = {
            "total_processed": 0,
            "errors": [],
            "results": [],
            "start_time": None,
            "end_time": None
        }

    @start()
    def initialize(self):
        """Initialize flow state"""
        from datetime import datetime
        self.state['start_time'] = datetime.now()
        print(f"Flow started at: {self.state['start_time']}")
        return {"status": "initialized"}

    @listen(initialize)
    def process_batch_1(self, data):
        """Process first batch and update state"""
        try:
            # Simulate processing
            items_processed = 10
            self.state['total_processed'] += items_processed
            self.state['results'].append(f"Batch 1: {items_processed} items")
            print(f"Batch 1 complete. Total: {self.state['total_processed']}")
        except Exception as e:
            self.state['errors'].append(f"Batch 1: {str(e)}")

        return {"batch": 1, "status": "complete"}

    @listen(process_batch_1)
    def process_batch_2(self, data):
        """Process second batch and update state"""
        try:
            items_processed = 15
            self.state['total_processed'] += items_processed
            self.state['results'].append(f"Batch 2: {items_processed} items")
            print(f"Batch 2 complete. Total: {self.state['total_processed']}")
        except Exception as e:
            self.state['errors'].append(f"Batch 2: {str(e)}")

        return {"batch": 2, "status": "complete"}

    @listen(process_batch_2)
    def process_batch_3(self, data):
        """Process third batch and update state"""
        try:
            items_processed = 8
            self.state['total_processed'] += items_processed
            self.state['results'].append(f"Batch 3: {items_processed} items")
            print(f"Batch 3 complete. Total: {self.state['total_processed']}")
        except Exception as e:
            self.state['errors'].append(f"Batch 3: {str(e)}")

        return {"batch": 3, "status": "complete"}

    @listen(process_batch_3)
    def finalize(self, data):
        """Finalize and report final state"""
        from datetime import datetime
        self.state['end_time'] = datetime.now()
        duration = (self.state['end_time'] - self.state['start_time']).total_seconds()

        print(f"\n=== FLOW COMPLETE ===")
        print(f"Total Processed: {self.state['total_processed']}")
        print(f"Batches: {len(self.state['results'])}")
        print(f"Errors: {len(self.state['errors'])}")
        print(f"Duration: {duration:.2f} seconds")

        return {
            "final_state": self.state,
            "duration_seconds": duration
        }

# Execute
flow = StatefulFlow()
result = flow.kickoff()
```

### Shared State Patterns

```python
from crewai.flow.flow import Flow, listen, start
from typing import Dict, List

class SharedStateFlow(Flow):
    """Flow with complex shared state"""

    def __init__(self):
        super().__init__()
        self.shared_context = {
            "user_preferences": {},
            "session_data": {},
            "analytics": {
                "page_views": 0,
                "actions": [],
                "errors": []
            }
        }

    @start()
    def load_user_preferences(self):
        """Load user preferences into shared state"""
        # Simulate loading preferences
        self.shared_context['user_preferences'] = {
            "theme": "dark",
            "language": "en",
            "notifications": True
        }
        print(f"Loaded preferences: {self.shared_context['user_preferences']}")
        return {"loaded": True}

    @listen(load_user_preferences)
    def initialize_session(self, data):
        """Initialize session with user preferences"""
        self.shared_context['session_data'] = {
            "session_id": "sess_12345",
            "user_id": "user_67890",
            "preferences": self.shared_context['user_preferences']
        }
        print(f"Session initialized: {self.shared_context['session_data']['session_id']}")
        return {"session": self.shared_context['session_data']}

    @listen(initialize_session)
    def track_activity(self, session_data):
        """Track user activity"""
        # Simulate tracking
        self.shared_context['analytics']['page_views'] += 5
        self.shared_context['analytics']['actions'].extend([
            "login", "view_dashboard", "update_profile", "save_settings", "logout"
        ])
        print(f"Tracked {len(self.shared_context['analytics']['actions'])} actions")
        return {"tracking": "complete"}

    @listen(track_activity)
    def generate_analytics_report(self, data):
        """Generate report from shared state"""
        report = {
            "user": self.shared_context['session_data']['user_id'],
            "page_views": self.shared_context['analytics']['page_views'],
            "total_actions": len(self.shared_context['analytics']['actions']),
            "errors": len(self.shared_context['analytics']['errors']),
            "preferences": self.shared_context['user_preferences']
        }

        print(f"\n=== ANALYTICS REPORT ===")
        for key, value in report.items():
            print(f"{key}: {value}")

        return {"report": report}

# Execute
flow = SharedStateFlow()
result = flow.kickoff()
```

---

## Error Handling

### Try-Catch in Flows

```python
from crewai.flow.flow import Flow, listen, start

class ErrorHandlingFlow(Flow):
    """Flow with comprehensive error handling"""

    def __init__(self):
        super().__init__()
        self.errors = []

    @start()
    def risky_operation(self):
        """Operation that might fail"""
        try:
            # Simulate risky operation
            import random
            if random.random() < 0.3:
                raise Exception("Random failure occurred")

            print("Operation succeeded")
            return {"status": "success", "data": "processed_data"}

        except Exception as e:
            error_msg = f"Error in risky_operation: {str(e)}"
            self.errors.append(error_msg)
            print(f"ERROR: {error_msg}")
            return {"status": "error", "error": str(e)}

    @listen(risky_operation)
    def handle_result(self, result):
        """Handle operation result"""
        if result['status'] == 'success':
            print("Processing successful result...")
            return {"processed": True, "data": result['data']}
        else:
            print("Handling error...")
            return {"processed": False, "error_handled": True}

# Execute with error handling
flow = ErrorHandlingFlow()
result = flow.kickoff()
print(f"Flow completed with {len(flow.errors)} errors")
```

### Retry Logic

```python
from crewai.flow.flow import Flow, listen, start, router

class RetryFlow(Flow):
    """Flow with automatic retry logic"""

    def __init__(self):
        super().__init__()
        self.max_retries = 3
        self.retry_count = 0

    @start()
    def unstable_operation(self):
        """Operation that might need retries"""
        import random

        self.retry_count += 1
        print(f"Attempt {self.retry_count}/{self.max_retries}")

        try:
            # Simulate unstable operation (70% failure rate)
            if random.random() < 0.7:
                raise Exception("Operation failed")

            print("SUCCESS!")
            return {"status": "success", "attempt": self.retry_count}

        except Exception as e:
            print(f"FAILED: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "attempt": self.retry_count,
                "can_retry": self.retry_count < self.max_retries
            }

    @router(unstable_operation)
    def decide_retry(self, result):
        """Decide whether to retry or fail"""
        if result['status'] == 'success':
            return 'handle_success'
        elif result.get('can_retry'):
            print(f"Retrying... ({self.retry_count}/{self.max_retries})")
            return 'unstable_operation'  # Retry
        else:
            return 'handle_failure'

    @listen('handle_success')
    def handle_success(self, result):
        """Handle successful operation"""
        print(f"Operation succeeded on attempt {result['attempt']}")
        return {"final_status": "success", "attempts": result['attempt']}

    @listen('handle_failure')
    def handle_failure(self, result):
        """Handle permanent failure"""
        print(f"Operation failed after {self.max_retries} attempts")
        return {
            "final_status": "failed",
            "attempts": self.retry_count,
            "error": result['error']
        }

# Execute with retry logic
flow = RetryFlow()
result = flow.kickoff()
print(f"\nFinal result: {result}")
```

---

## Advanced Patterns

### Pattern: Fan-Out / Fan-In

```python
from crewai import Agent, Task, Crew, LLM
from crewai.flow.flow import Flow, listen, start

class FanOutFanInFlow(Flow):
    """Parallel processing that converges to single result"""

    def __init__(self):
        super().__init__()
        self.llm = LLM(model="openai/gpt-4-turbo")
        self.results = []

    @start()
    def distribute_work(self):
        """Fan-out: Distribute work to multiple paths"""
        return {
            "topics": [
                "AI Ethics",
                "Machine Learning",
                "Natural Language Processing"
            ]
        }

    @listen(distribute_work)
    def research_topic_1(self, data):
        """Research first topic"""
        topic = data['topics'][0]
        print(f"Researching: {topic}")

        researcher = Agent(
            role="Researcher",
            goal=f"Research {topic}",
            llm=self.llm
        )

        task = Task(
            description=f"Research {topic} comprehensively",
            expected_output="Research summary",
            agent=researcher
        )

        crew = Crew(agents=[researcher], tasks=[task])
        result = crew.kickoff()

        self.results.append({"topic": topic, "research": result})
        return {"topic": topic, "done": True}

    @listen(distribute_work)
    def research_topic_2(self, data):
        """Research second topic"""
        topic = data['topics'][1]
        print(f"Researching: {topic}")
        # Similar implementation
        self.results.append({"topic": topic, "research": "Summary 2"})
        return {"topic": topic, "done": True}

    @listen(distribute_work)
    def research_topic_3(self, data):
        """Research third topic"""
        topic = data['topics'][2]
        print(f"Researching: {topic}")
        # Similar implementation
        self.results.append({"topic": topic, "research": "Summary 3"})
        return {"topic": topic, "done": True}

    @listen(research_topic_1, research_topic_2, research_topic_3)
    def consolidate_research(self, *args):
        """Fan-in: Consolidate all research results"""
        print(f"\n=== CONSOLIDATING {len(self.results)} RESEARCH RESULTS ===")

        consolidated = {
            "total_topics": len(self.results),
            "topics": [r['topic'] for r in self.results],
            "status": "complete"
        }

        return consolidated

# Execute fan-out/fan-in flow
flow = FanOutFanInFlow()
result = flow.kickoff()
```

### Pattern: Pipeline with Validation

```python
from crewai.flow.flow import Flow, listen, start, router

class ValidationPipelineFlow(Flow):
    """Pipeline with validation at each stage"""

    @start()
    def stage_1_process(self):
        """Stage 1: Initial processing"""
        print("Stage 1: Processing...")
        data = {"value": 42, "valid": True}
        return data

    @router(stage_1_process)
    def validate_stage_1(self, data):
        """Validate stage 1 output"""
        if data.get('valid'):
            return 'stage_2_process'
        else:
            return 'handle_validation_error'

    @listen('stage_2_process')
    def stage_2_process(self, data):
        """Stage 2: Further processing"""
        print("Stage 2: Processing...")
        data['transformed'] = data['value'] * 2
        data['valid'] = True
        return data

    @router(stage_2_process)
    def validate_stage_2(self, data):
        """Validate stage 2 output"""
        if data.get('transformed'):
            return 'stage_3_process'
        else:
            return 'handle_validation_error'

    @listen('stage_3_process')
    def stage_3_process(self, data):
        """Stage 3: Final processing"""
        print("Stage 3: Processing...")
        data['finalized'] = True
        return data

    @listen(stage_3_process)
    def pipeline_complete(self, data):
        """Pipeline completed successfully"""
        print("Pipeline completed successfully!")
        return {"status": "success", "final_data": data}

    @listen('handle_validation_error')
    def handle_validation_error(self, data):
        """Handle validation failures"""
        print("Validation failed!")
        return {"status": "error", "message": "Validation error occurred"}

# Execute
flow = ValidationPipelineFlow()
result = flow.kickoff()
```

---

## Production Best Practices

### 1. Logging and Monitoring

```python
import logging
from crewai.flow.flow import Flow, listen, start

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonitoredFlow(Flow):
    """Flow with comprehensive logging"""

    @start()
    def start_process(self):
        """Start with logging"""
        logger.info("Flow execution started")
        return {"status": "started"}

    @listen(start_process)
    def process_data(self, data):
        """Process with detailed logging"""
        logger.info(f"Processing data: {data}")

        try:
            # Processing logic
            result = {"processed": True}
            logger.info("Processing successful")
            return result
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            raise

    @listen(process_data)
    def finalize(self, data):
        """Finalize with metrics"""
        logger.info("Flow execution completed")
        return {"status": "complete"}
```

### 2. Input Validation

```python
from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel, Field, ValidationError

class FlowInput(BaseModel):
    """Validated input schema"""
    user_id: str = Field(..., min_length=1)
    action: str = Field(..., pattern="^(create|update|delete)$")
    data: dict

class ValidatedFlow(Flow):
    """Flow with input validation"""

    @start()
    def validate_input(self):
        """Validate input data"""
        try:
            # Validate input against schema
            validated = FlowInput(**self.initial_state)
            print(f"Input validated: {validated.user_id}")
            return validated.dict()
        except ValidationError as e:
            print(f"Validation error: {e}")
            return {"error": "Invalid input", "details": str(e)}

    @listen(validate_input)
    def process_validated_input(self, data):
        """Process validated input"""
        if 'error' in data:
            print("Skipping processing due to validation error")
            return data

        print(f"Processing: {data['action']} for user {data['user_id']}")
        return {"status": "processed", "data": data}

# Execute with validation
flow = ValidatedFlow()
result = flow.kickoff(inputs={
    "user_id": "user123",
    "action": "create",
    "data": {"name": "New Item"}
})
```

### 3. Testing Flows

```python
import pytest
from crewai.flow.flow import Flow, listen, start

class TestableFlow(Flow):
    """Flow designed for testing"""

    @start()
    def process(self):
        return {"value": 10}

    @listen(process)
    def transform(self, data):
        return {"value": data['value'] * 2}

def test_flow_execution():
    """Test flow executes correctly"""
    flow = TestableFlow()
    result = flow.kickoff()
    assert result['value'] == 20

def test_flow_with_mock_input():
    """Test flow with custom input"""
    flow = TestableFlow()
    result = flow.kickoff(inputs={"initial": "data"})
    assert 'value' in result
```

---

## Real-World Examples

### Example 1: Customer Support Automation

```python
from crewai import Agent, Task, Crew, LLM
from crewai.flow.flow import Flow, listen, start, router

class CustomerSupportFlow(Flow):
    """Automated customer support workflow"""

    def __init__(self):
        super().__init__()
        self.llm = LLM(model="openai/gpt-4-turbo")

    @start()
    def receive_ticket(self):
        """Receive support ticket"""
        # In production, this would come from ticket system
        return {
            "ticket_id": "SUPP-12345",
            "customer": "Acme Corp",
            "subject": "API integration issue",
            "description": "Getting 401 errors when calling the API",
            "priority": "high"
        }

    @router(receive_ticket)
    def classify_ticket(self, ticket):
        """Classify ticket type"""
        keywords = ticket['description'].lower()

        if 'api' in keywords or 'integration' in keywords:
            return 'technical_support'
        elif 'billing' in keywords or 'invoice' in keywords:
            return 'billing_support'
        else:
            return 'general_support'

    @listen('technical_support')
    def technical_support(self, ticket):
        """Handle technical support ticket"""
        print(f"Technical Support: {ticket['ticket_id']}")

        tech_agent = Agent(
            role="Technical Support Engineer",
            goal="Diagnose and resolve technical issues",
            backstory="Senior engineer with API expertise",
            llm=self.llm
        )

        task = Task(
            description=f"Investigate: {ticket['description']}",
            expected_output="Technical diagnosis and solution steps",
            agent=tech_agent
        )

        crew = Crew(agents=[tech_agent], tasks=[task])
        solution = crew.kickoff()

        return {
            "ticket_id": ticket['ticket_id'],
            "solution": solution,
            "resolved": True
        }

    @listen('billing_support')
    def billing_support(self, ticket):
        """Handle billing ticket"""
        print(f"Billing Support: {ticket['ticket_id']}")
        return {"ticket_id": ticket['ticket_id'], "routed_to": "billing"}

    @listen('general_support')
    def general_support(self, ticket):
        """Handle general ticket"""
        print(f"General Support: {ticket['ticket_id']}")
        return {"ticket_id": ticket['ticket_id'], "routed_to": "general"}

    @listen(technical_support, billing_support, general_support)
    def close_ticket(self, result):
        """Close ticket and notify customer"""
        print(f"\n=== TICKET CLOSED: {result['ticket_id']} ===")
        return {"status": "closed", "result": result}

# Execute support flow
flow = CustomerSupportFlow()
result = flow.kickoff()
```

### Example 2: Content Production Pipeline

```python
from crewai import Agent, Task, Crew, LLM
from crewai.flow.flow import Flow, listen, start

class ContentProductionFlow(Flow):
    """End-to-end content production"""

    def __init__(self):
        super().__init__()
        self.llm = LLM(model="openai/gpt-4-turbo")

    @start()
    def content_brief(self):
        """Define content requirements"""
        return {
            "topic": "Future of Remote Work",
            "target_audience": "Business leaders",
            "word_count": 1500,
            "tone": "professional yet accessible",
            "seo_keywords": ["remote work", "digital transformation", "workplace"]
        }

    @listen(content_brief)
    def research_phase(self, brief):
        """Research topic"""
        researcher = Agent(
            role="Research Specialist",
            goal="Conduct comprehensive research on topics",
            backstory="Expert researcher with business domain knowledge",
            llm=self.llm
        )

        task = Task(
            description=f"Research '{brief['topic']}' for {brief['target_audience']}. Focus on: {brief['seo_keywords']}",
            expected_output="Research findings with statistics and expert quotes",
            agent=researcher
        )

        crew = Crew(agents=[researcher], tasks=[task])
        research = crew.kickoff()

        return {**brief, "research": research}

    @listen(research_phase)
    def writing_phase(self, data):
        """Write content"""
        writer = Agent(
            role="Content Writer",
            goal="Create engaging, SEO-optimized content",
            backstory="Professional writer with 10+ years experience",
            llm=self.llm
        )

        task = Task(
            description=f"""Write {data['word_count']}-word article on '{data['topic']}'.

Tone: {data['tone']}
Target audience: {data['target_audience']}
SEO keywords: {data['seo_keywords']}
Research: {data['research']}""",
            expected_output="Complete article draft",
            agent=writer
        )

        crew = Crew(agents=[writer], tasks=[task])
        draft = crew.kickoff()

        return {**data, "draft": draft}

    @listen(writing_phase)
    def editing_phase(self, data):
        """Edit and refine"""
        editor = Agent(
            role="Editor",
            goal="Polish content to publication quality",
            backstory="Senior editor with keen eye for detail",
            llm=self.llm
        )

        task = Task(
            description=f"Edit this article for clarity, engagement, and SEO:\n\n{data['draft']}",
            expected_output="Publication-ready article with SEO optimization",
            agent=editor
        )

        crew = Crew(agents=[editor], tasks=[task])
        final_article = crew.kickoff()

        return {
            "topic": data['topic'],
            "final_article": final_article,
            "status": "ready_for_publication"
        }

    @listen(editing_phase)
    def publish(self, data):
        """Publish content"""
        print(f"\n=== PUBLISHING: {data['topic']} ===")
        print(f"Status: {data['status']}")

        return {
            "published": True,
            "topic": data['topic'],
            "article": data['final_article']
        }

# Execute content production
flow = ContentProductionFlow()
result = flow.kickoff()
```

---

## Conclusion

CrewAI Flows represents the future of agentic AI automation. By combining event-driven architecture with powerful agent orchestration, Flows enables you to build sophisticated, production-ready applications that scale.

### Key Takeaways

1. **Flows extend CrewAI** with event-driven workflows and control flow
2. **Use @start(), @listen(), and @router()** to define workflow logic
3. **Integrate crews seamlessly** into flows for powerful agentic automation
4. **Manage state explicitly** for complex, multi-step workflows
5. **Implement error handling and retries** for production resilience
6. **Follow best practices** for logging, validation, and testing

### Next Steps

- Explore the [CrewAI Comprehensive Guide](./crewai_comprehensive_guide.md) for foundation concepts
- Build your own flows for real-world use cases
- Join the CrewAI community: https://www.crewai.com/community
- Review production deployment patterns in the [CrewAI Production Guide](./crewai_production_guide.md)

**Happy Building with CrewAI Flows!**
