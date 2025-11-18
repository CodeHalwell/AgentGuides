# Advanced Multi-Agent Patterns for Amazon Bedrock Agents (Python)

This guide explores advanced multi-agent patterns for building sophisticated and scalable agentic systems with Amazon Bedrock.

## 1. Hierarchical Agent Systems

In a hierarchical agent system, agents are organized in a tree-like structure with a top-level supervisor agent that delegates tasks to sub-supervisors or specialist agents.

**Use Cases:**

*   Large-scale enterprise systems with multiple departments or business units.
*   Complex workflows that require multiple levels of abstraction and control.

**Example: A Hierarchical Customer Support System**

```
                      ┌────────────────────┐
                      │  Global Supervisor │
                      └─────────┬──────────┘
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
      ┌───────▼───────┐ ┌───────▼───────┐ ┌───────▼───────┐
      │ Sales         │ │ Support       │ │ Billing       │
      │ Supervisor    │ │ Supervisor    │ │ Supervisor    │
      └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
              │                 │                 │
      ┌───────┴───────┐ ┌───────┴───────┐ ┌───────┴───────┐
      │               │ │               │ │               │
┌─────▼──┐      ┌─────▼──┐      ┌─────▼──┐      ┌─────▼──┐
│ Product│      │ Lead   │      │ Tier 1 │      │ Tier 2 │
│ Agent  │      │ Gen    │      │ Agent  │      │ Agent  │
└────────┘      └────────┘      └────────┘      └────────┘
```

**Implementation:**

```python
class HierarchicalAgentSystem:
    def __init__(self):
        self.bedrock = boto3.client('bedrock')

    def create_system(self):
        # Create the global supervisor
        global_supervisor = self._create_agent("GlobalSupervisor", "Routes requests to the appropriate department.")

        # Create departmental supervisors
        sales_supervisor = self._create_agent("SalesSupervisor", "Manages sales-related tasks.")
        support_supervisor = self._create_agent("SupportSupervisor", "Manages support-related tasks.")

        # Create specialist agents
        product_agent = self._create_agent("ProductAgent", "Provides product information.")
        lead_gen_agent = self._create_agent("LeadGenAgent", "Generates sales leads.")
        tier1_agent = self._create_agent("Tier1SupportAgent", "Handles basic support requests.")
        tier2_agent = self._create_agent("Tier2SupportAgent", "Handles complex support requests.")

        # Associate agents in a hierarchy
        self._associate(global_supervisor, [sales_supervisor, support_supervisor])
        self._associate(sales_supervisor, [product_agent, lead_gen_agent])
        self._associate(support_supervisor, [tier1_agent, tier2_agent])

    def _create_agent(self, name, instruction):
        # ... implementation for creating an agent ...
        pass

    def _associate(self, supervisor, specialists):
        # ... implementation for associating agents ...
        pass
```

## 2. Dynamic Agent Composition

Dynamic agent composition allows a supervisor agent to select and combine the capabilities of different specialist agents at runtime to fulfill a user's request.

**Use Cases:**

*   Handling ad-hoc or unpredictable user requests.
*   Creating highly flexible and adaptable agentic systems.

**Implementation:**

```python
class DynamicAgentComposition:
    def __init__(self):
        self.bedrock = boto3.client('bedrock')
        self.runtime = boto3.client('bedrock-runtime')
        self.agents = self._discover_agents()

    def _discover_agents(self):
        # Discover available agents and their capabilities
        # This could be done by querying a service registry or a database
        return {
            "WeatherAgent": "Provides weather forecasts.",
            "NewsAgent": "Provides the latest news headlines.",
            "TranslationAgent": "Translates text between languages."
        }

    def invoke(self, request):
        # 1. Understand the user's request
        # 2. Identify the required capabilities
        # 3. Select the appropriate agents
        # 4. Orchestrate the interaction between the selected agents
        # 5. Synthesize the final response

        if "weather" in request.lower():
            # Invoke the WeatherAgent
            pass
        elif "news" in request.lower():
            # Invoke the NewsAgent
            pass
        elif "translate" in request.lower():
            # Invoke the TranslationAgent
            pass
        else:
            # Handle requests that don't match any agent's capabilities
            pass
```

## 3. Agent-to-Agent (A2A) Communication with the A2A Protocol

The Agent-to-Agent (A2A) protocol enables seamless communication and coordination between agents built using different frameworks.

**Status**: Generally Available (March 10, 2025)

**Example:**

An Amazon Bedrock Agent can communicate with an agent built using the OpenAI Agents SDK.

```python
# Bedrock Agent (Python)
response = bedrock_runtime.invoke_agent(
    agentId='BEDROCK_AGENT_ID',
    # ...
    a2a_payload={
        'protocol': 'a2a',
        'version': '1.0',
        'recipient': 'OPENAI_AGENT_ID',
        'message': 'Can you summarize the latest news about AI?'
    }
)

# OpenAI Agent (Python)
@tool
def receive_a2a_message(message: str, sender: str):
    """Receives a message from another agent."""
    # Process the message and return a response
    pass
```

For complete A2A protocol documentation, see **bedrock_a2a_protocol_guide.md**

---

## 4. Multi-Agent Collaboration (GA March 10, 2025)

Amazon Bedrock now provides native multi-agent collaboration with two operational modes:

### Supervisor Mode

In supervisor mode, a supervisor agent coordinates specialist agents to handle complex tasks.

```python
import boto3
import json

bedrock = boto3.client('bedrock')

# Create specialist agents
sales_agent = bedrock.create_agent(
    agentName='SalesSpecialist',
    foundationModel='anthropic.claude-3-5-sonnet-20241022-v2:0',
    instruction='You are a sales specialist. Handle product inquiries, pricing, and quotes.',
    tools=[get_product_info, calculate_pricing, generate_quote]
)

support_agent = bedrock.create_agent(
    agentName='SupportSpecialist',
    foundationModel='anthropic.claude-3-5-sonnet-20241022-v2:0',
    instruction='You are a technical support specialist. Handle troubleshooting and technical issues.',
    tools=[check_system_status, create_ticket, search_kb]
)

# Create supervisor agent
supervisor = bedrock.create_agent(
    agentName='CustomerServiceSupervisor',
    foundationModel='anthropic.claude-3-5-sonnet-20241022-v2:0',
    instruction='You coordinate customer service. Delegate to sales or support specialists as needed.',
    multiAgentConfig={
        'mode': 'SUPERVISOR',
        'supervisorConfig': {
            'subAgents': [
                {
                    'agentId': sales_agent['agentId'],
                    'name': 'sales_specialist',
                    'description': 'Handles sales, pricing, and product information'
                },
                {
                    'agentId': support_agent['agentId'],
                    'name': 'support_specialist',
                    'description': 'Handles technical support and troubleshooting'
                }
            ],
            'parallelCommunication': True  # Enable parallel execution
        }
    }
)

# Use supervisor
bedrock_runtime = boto3.client('bedrock-runtime')

response = bedrock_runtime.invoke_agent(
    agentId=supervisor['agentId'],
    agentAliasId='PROD',
    sessionId='session-123',
    inputText='I need pricing for enterprise plan and help with a technical issue'
)

# Supervisor delegates to both sales and support agents in parallel
print(response['completion'])
```

### Supervisor with Routing Mode

In this mode, the supervisor uses explicit routing logic to determine which specialist agent(s) to invoke.

```python
# Create supervisor with routing
supervisor_with_routing = bedrock.create_agent(
    agentName='SmartSupervisor',
    foundationModel='anthropic.claude-3-5-sonnet-20241022-v2:0',
    instruction='Route customer requests intelligently based on intent classification.',
    multiAgentConfig={
        'mode': 'SUPERVISOR_WITH_ROUTING',
        'supervisorConfig': {
            'subAgents': [
                {
                    'agentId': sales_agent['agentId'],
                    'name': 'sales_specialist',
                    'description': 'Handles sales, pricing, and product information',
                    'routingCriteria': {
                        'keywords': ['price', 'buy', 'purchase', 'quote', 'cost'],
                        'intentClassifications': ['SALES_INQUIRY', 'PRICING_REQUEST']
                    }
                },
                {
                    'agentId': support_agent['agentId'],
                    'name': 'support_specialist',
                    'description': 'Handles technical support and troubleshooting',
                    'routingCriteria': {
                        'keywords': ['error', 'issue', 'not working', 'bug', 'help'],
                        'intentClassifications': ['TECHNICAL_SUPPORT', 'TROUBLESHOOTING']
                    }
                }
            ],
            'routingStrategy': 'INTENT_BASED',
            'fallbackAgent': support_agent['agentId']  # Default if no match
        }
    }
)
```

### Complete Multi-Agent System Example

```python
class ProductionMultiAgentSystem:
    """Production-ready multi-agent collaboration system"""

    def __init__(self, region='us-east-1'):
        self.bedrock = boto3.client('bedrock', region_name=region)
        self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=region)
        self.agents = {}

    def create_specialist_agents(self):
        """Create all specialist agents"""

        # Tier 1 Support Agent
        self.agents['tier1_support'] = self.bedrock.create_agent(
            agentName='Tier1Support',
            foundationModel='anthropic.claude-3-haiku-20240307-v1:0',  # Fast, cost-effective
            instruction='Handle basic customer support questions quickly and efficiently.',
            tools=[get_account_info, check_order_status, update_contact_info]
        )

        # Tier 2 Support Agent
        self.agents['tier2_support'] = self.bedrock.create_agent(
            agentName='Tier2Support',
            foundationModel='anthropic.claude-3-5-sonnet-20241022-v2:0',  # More capable
            instruction='Handle complex technical issues requiring deep analysis.',
            tools=[run_diagnostics, access_logs, escalate_ticket, create_bug_report]
        )

        # Sales Agent
        self.agents['sales'] = self.bedrock.create_agent(
            agentName='SalesAgent',
            foundationModel='anthropic.claude-3-5-sonnet-20241022-v2:0',
            instruction='Provide product information, pricing, and generate quotes.',
            tools=[get_product_catalog, calculate_pricing, generate_quote, check_inventory],
            knowledgeBases=[self.create_product_knowledge_base()]
        )

        # Billing Agent
        self.agents['billing'] = self.bedrock.create_agent(
            agentName='BillingAgent',
            foundationModel='anthropic.claude-3-5-sonnet-20241022-v2:0',
            instruction='Handle billing inquiries, invoices, and payment processing.',
            tools=[get_invoice, process_payment, update_billing_info, check_payment_status]
        )

    def create_department_supervisors(self):
        """Create mid-level supervisors for each department"""

        # Support Supervisor
        self.agents['support_supervisor'] = self.bedrock.create_agent(
            agentName='SupportSupervisor',
            foundationModel='anthropic.claude-3-5-sonnet-20241022-v2:0',
            instruction='Coordinate support requests by routing to appropriate tier.',
            multiAgentConfig={
                'mode': 'SUPERVISOR_WITH_ROUTING',
                'supervisorConfig': {
                    'subAgents': [
                        {
                            'agentId': self.agents['tier1_support']['agentId'],
                            'name': 'tier1',
                            'description': 'Basic support questions',
                            'routingCriteria': {
                                'complexity': 'LOW',
                                'keywords': ['account', 'password', 'status', 'contact']
                            }
                        },
                        {
                            'agentId': self.agents['tier2_support']['agentId'],
                            'name': 'tier2',
                            'description': 'Complex technical issues',
                            'routingCriteria': {
                                'complexity': 'HIGH',
                                'keywords': ['error', 'bug', 'crash', 'performance']
                            }
                        }
                    ],
                    'routingStrategy': 'COMPLEXITY_BASED'
                }
            }
        )

    def create_global_supervisor(self):
        """Create top-level supervisor that coordinates all departments"""

        self.agents['global_supervisor'] = self.bedrock.create_agent(
            agentName='GlobalSupervisor',
            foundationModel='anthropic.claude-3-opus-20240229-v1:0',  # Most capable
            instruction='Top-level customer service coordinator. Route to appropriate department.',
            multiAgentConfig={
                'mode': 'SUPERVISOR_WITH_ROUTING',
                'supervisorConfig': {
                    'subAgents': [
                        {
                            'agentId': self.agents['support_supervisor']['agentId'],
                            'name': 'support',
                            'description': 'All technical support requests'
                        },
                        {
                            'agentId': self.agents['sales']['agentId'],
                            'name': 'sales',
                            'description': 'Sales, products, and pricing'
                        },
                        {
                            'agentId': self.agents['billing']['agentId'],
                            'name': 'billing',
                            'description': 'Billing, invoices, and payments'
                        }
                    ],
                    'parallelCommunication': True,
                    'routingStrategy': 'INTENT_BASED'
                }
            }
        )

    def invoke_system(self, user_request: str, session_id: str = None):
        """Invoke the multi-agent system"""

        import uuid
        session_id = session_id or str(uuid.uuid4())

        response = self.bedrock_runtime.invoke_agent(
            agentId=self.agents['global_supervisor']['agentId'],
            agentAliasId='PROD',
            sessionId=session_id,
            inputText=user_request,
            enableTrace=True  # Enable to see routing decisions
        )

        # Parse streaming response
        completion = ''
        for event in response['completion']:
            if 'chunk' in event:
                completion += event['chunk']['bytes'].decode()

        return {
            'response': completion,
            'trace': response.get('trace', []),
            'session_id': session_id
        }

# Usage
system = ProductionMultiAgentSystem()
system.create_specialist_agents()
system.create_department_supervisors()
system.create_global_supervisor()

# Test multi-agent system
result = system.invoke_system(
    "I need help with a billing issue and also want to know about your enterprise pricing"
)

print(f"Response: {result['response']}")
print(f"Routing trace: {result['trace']}")
```

### Monitoring Multi-Agent Systems

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

def monitor_multi_agent_system(supervisor_agent_id: str):
    """Monitor multi-agent system performance"""

    # Get agent invocation metrics
    metrics = cloudwatch.get_metric_statistics(
        Namespace='AWS/Bedrock',
        MetricName='AgentInvocations',
        Dimensions=[
            {'Name': 'AgentId', 'Value': supervisor_agent_id}
        ],
        StartTime=datetime.utcnow() - timedelta(hours=1),
        EndTime=datetime.utcnow(),
        Period=300,
        Statistics=['Sum', 'Average']
    )

    # Get subagent delegation metrics
    delegation_metrics = cloudwatch.get_metric_statistics(
        Namespace='AWS/Bedrock',
        MetricName='SubAgentDelegations',
        Dimensions=[
            {'Name': 'SupervisorAgentId', 'Value': supervisor_agent_id}
        ],
        StartTime=datetime.utcnow() - timedelta(hours=1),
        EndTime=datetime.utcnow(),
        Period=300,
        Statistics=['Sum']
    )

    return {
        'invocations': metrics,
        'delegations': delegation_metrics
    }
```

For complete multi-agent collaboration patterns, see **bedrock_agents_comprehensive_guide.md** and **bedrock_strands_sdk_guide.md**