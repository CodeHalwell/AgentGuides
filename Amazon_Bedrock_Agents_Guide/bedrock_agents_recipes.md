# Amazon Bedrock Agents: Recipes and Common Patterns

A collection of practical, production-ready recipes and patterns for implementing Amazon Bedrock Agents across various business scenarios.

---

## Table of Contents

1. [Customer Support Agent](#customer-support-agent)
2. [Multi-Agent Customer Service System](#multi-agent-customer-service-system)
3. [Financial Analytics Agent](#financial-analytics-agent)
4. [Document Processing Pipeline](#document-processing-pipeline)
5. [Real-Time Inventory Management](#real-time-inventory-management)
6. [Claims Processing Agent](#claims-processing-agent)
7. [Research and Analysis Agent](#research-and-analysis-agent)
8. [HR and Employee Assistance](#hr-and-employee-assistance)
9. [Code Generation and Debugging](#code-generation-and-debugging)
10. [IoT Data Analysis](#iot-data-analysis)

---

## Customer Support Agent

### Basic Customer Support Implementation

{% raw %}
```python
import boto3
import json
from typing import Dict, Optional
import logging

class CustomerSupportAgent:
    """Production-ready customer support agent"""
    
    def __init__(self, region='us-east-1'):
        self.bedrock = boto3.client('bedrock', region_name=region)
        self.runtime = boto3.client('bedrock-runtime', region_name=region)
        self.s3 = boto3.client('s3')
        self.dynamodb = boto3.resource('dynamodb')
        self.logger = logging.getLogger(__name__)
    
    def create_support_agent(self, agent_name='CustomerSupportAgent') -> str:
        """Create a customer support agent with FAQs and ticket management"""
        
        agent_config = {
            'agentName': agent_name,
            'agentDescription': 'Intelligent customer support agent for ticket management and FAQ',
            'foundationModelId': 'anthropic.claude-3-5-sonnet-20240620-v1:0',
            'instruction': """You are an expert customer support agent. Your responsibilities:
1. Answer frequently asked questions about our products and services
2. Help customers troubleshoot common issues
3. Create and update support tickets
4. Escalate complex issues to human specialists
5. Provide refund and warranty information
6. Maintain professional and empathetic tone

Always:
- Verify customer identity before accessing account information
- Provide clear step-by-step solutions
- Suggest self-service options before escalation
- Document all interactions
- Follow company policies strictly""",
            'idleSessionTTLInSeconds': 900,
            'customerEncryptionKeyArn': 'arn:aws:kms:us-east-1:ACCOUNT:key/KEY_ID'
        }
        
        response = self.bedrock.create_agent(**agent_config)
        agent_id = response['agentId']
        
        # Create action groups for customer support operations
        self._create_ticket_action_group(agent_id)
        self._create_faq_action_group(agent_id)
        self._create_account_action_group(agent_id)
        
        # Create knowledge base with FAQs and policies
        self._create_support_knowledge_base(agent_id)
        
        # Create guardrails for compliance
        self._create_support_guardrails(agent_id)
        
        return agent_id
    
    def _create_ticket_action_group(self, agent_id: str) -> None:
        """Create action group for ticket management"""
        
        ticket_schema = {
            "openapi": "3.0.0",
            "info": {"title": "Support Ticket API", "version": "1.0.0"},
            "paths": {
                "/tickets": {
                    "post": {
                        "operationId": "CreateTicket",
                        "summary": "Create a new support ticket",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "customerId": {"type": "string"},
                                            "category": {"type": "string", "enum": ["BILLING", "TECHNICAL", "GENERAL"]},
                                            "subject": {"type": "string"},
                                            "description": {"type": "string"},
                                            "priority": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]}
                                        },
                                        "required": ["customerId", "category", "subject"]
                                    }
                                }
                            }
                        },
                        "responses": {
                            "201": {
                                "description": "Ticket created",
                                "content": {
                                    "application/json": {
                                        "schema": {
                                            "type": "object",
                                            "properties": {
                                                "ticketId": {"type": "string"},
                                                "status": {"type": "string"},
                                                "createdAt": {"type": "string"}
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "/tickets/{ticketId}": {
                    "get": {
                        "operationId": "GetTicket",
                        "summary": "Retrieve ticket details",
                        "parameters": [
                            {"name": "ticketId", "in": "path", "required": True, "schema": {"type": "string"}}
                        ],
                        "responses": {
                            "200": {"description": "Ticket details"}
                        }
                    }
                }
            }
        }
        
        self.bedrock.create_action_group(
            agentId=agent_id,
            actionGroupName='TicketManagement',
            actionGroupDescription='Manage customer support tickets',
            apiSchema={'payload': json.dumps(ticket_schema)},
            actionGroupExecutor={'lambda': 'arn:aws:lambda:us-east-1:ACCOUNT:function:support-ticket-handler'}
        )
    
    def _create_faq_action_group(self, agent_id: str) -> None:
        """Create action group for FAQ retrieval"""
        
        faq_schema = {
            "openapi": "3.0.0",
            "info": {"title": "FAQ API", "version": "1.0.0"},
            "paths": {
                "/faqs/search": {
                    "get": {
                        "operationId": "SearchFAQ",
                        "summary": "Search frequently asked questions",
                        "parameters": [
                            {"name": "query", "in": "query", "required": True, "schema": {"type": "string"}},
                            {"name": "category", "in": "query", "schema": {"type": "string"}}
                        ],
                        "responses": {
                            "200": {"description": "FAQ results"}
                        }
                    }
                }
            }
        }
        
        self.bedrock.create_action_group(
            agentId=agent_id,
            actionGroupName='FAQRetrieval',
            actionGroupDescription='Search and retrieve FAQ information',
            apiSchema={'payload': json.dumps(faq_schema)},
            actionGroupExecutor={'lambda': 'arn:aws:lambda:us-east-1:ACCOUNT:function:faq-retrieval-handler'}
        )
    
    def _create_account_action_group(self, agent_id: str) -> None:
        """Create action group for account operations"""
        
        account_schema = {
            "openapi": "3.0.0",
            "info": {"title": "Account API", "version": "1.0.0"},
            "paths": {
                "/accounts/{customerId}": {
                    "get": {
                        "operationId": "GetAccount",
                        "summary": "Get customer account details",
                        "parameters": [
                            {"name": "customerId", "in": "path", "required": True, "schema": {"type": "string"}}
                        ],
                        "responses": {
                            "200": {"description": "Account details"}
                        }
                    }
                }
            }
        }
        
        self.bedrock.create_action_group(
            agentId=agent_id,
            actionGroupName='AccountManagement',
            actionGroupDescription='Manage customer account information',
            apiSchema={'payload': json.dumps(account_schema)},
            actionGroupExecutor={'lambda': 'arn:aws:lambda:us-east-1:ACCOUNT:function:account-handler'}
        )
    
    def _create_support_knowledge_base(self, agent_id: str) -> None:
        """Create knowledge base with support documentation"""
        
        kb_response = self.bedrock.create_knowledge_base(
            name=f'{agent_id}-kb',
            description='Support FAQs, policies, and troubleshooting guides',
            roleArn='arn:aws:iam::ACCOUNT:role/BedrockKBRole',
            knowledgeBaseConfiguration={
                'type': 'VECTOR',
                'vectorKnowledgeBaseConfiguration': {
                    'embeddingModel': {
                        'provider': 'BEDROCK',
                        'bedrockEmbeddingModelConfiguration': {
                            'modelId': 'amazon.titan-embed-text-v1'
                        }
                    }
                }
            },
            storageConfiguration={
                'type': 'OPENSEARCH_SERVERLESS',
                'opensearchServerlessConfiguration': {
                    'collectionArn': 'arn:aws:aoss:us-east-1:ACCOUNT:collection/support-kb',
                    'vectorIndexName': 'support-index',
                    'fieldMapping': {
                        'vectorField': 'vector',
                        'textField': 'text',
                        'metadataField': 'metadata'
                    }
                }
            }
        )
        
        # Create data sources
        self.bedrock.create_data_source(
            knowledgeBaseId=kb_response['knowledgeBaseId'],
            name='FAQDocuments',
            dataSourceConfiguration={
                'type': 'S3',
                's3Configuration': {
                    'bucketArn': 'arn:aws:s3:::support-faqs',
                    'inclusionPrefixes': ['faqs/']
                }
            }
        )
    
    def _create_support_guardrails(self, agent_id: str) -> None:
        """Create guardrails for compliance"""
        
        guardrail_response = self.bedrock.create_guardrail(
            name=f'{agent_id}-guardrails',
            description='Guardrails for customer support agent',
            topicPolicyConfig={
                'topicsConfig': [
                    {
                        'name': 'PrivacyPolicy',
                        'definition': 'Discussion of customer privacy policies',
                        'type': 'ALLOW'
                    },
                    {
                        'name': 'UnauthorisedAccess',
                        'definition': 'Attempts to access accounts without proper verification',
                        'type': 'DENY'
                    }
                ]
            },
            contentPolicyConfig={
                'filtersConfig': [
                    {'type': 'HATE', 'inputStrength': 'HIGH', 'outputStrength': 'HIGH'},
                    {'type': 'SEXUAL', 'inputStrength': 'HIGH', 'outputStrength': 'HIGH'},
                    {'type': 'VIOLENCE', 'inputStrength': 'HIGH', 'outputStrength': 'HIGH'}
                ]
            }
        )
        
        # Associate guardrail with agent
        self.bedrock.update_agent(
            agentId=agent_id,
            guardrailConfiguration={
                'guardrailId': guardrail_response['guardrailId']
            }
        )
    
    def invoke_support_agent(self, agent_id: str, customer_query: str, customer_id: str) -> Dict:
        """Invoke the support agent"""
        
        response = self.runtime.invoke_agent(
            agentId=agent_id,
            agentAliasId='LFSTG5EXAMPLE',
            inputText=customer_query,
            sessionId=customer_id
        )
        
        return {
            'agent_response': response['outputText'],
            'trace_information': response.get('traceInformation', {})
        }


# Lambda handler for ticket creation
def ticket_handler(event, context):
    """Lambda function for ticket management actions"""
    
    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('SupportTickets')
    
    action = event.get('function')
    parameters = event.get('parameters', {})
    
    if action == 'CreateTicket':
        ticket_id = f"TICKET-{int(time.time())}"
        
        item = {
            'ticketId': ticket_id,
            'customerId': parameters.get('customerId'),
            'category': parameters.get('category'),
            'subject': parameters.get('subject'),
            'description': parameters.get('description'),
            'priority': parameters.get('priority', 'MEDIUM'),
            'status': 'OPEN',
            'createdAt': datetime.utcnow().isoformat(),
            'updatedAt': datetime.utcnow().isoformat()
        }
        
        table.put_item(Item=item)
        
        return {
            'ticketId': ticket_id,
            'status': 'OPEN',
            'createdAt': item['createdAt']
        }
    
    elif action == 'GetTicket':
        response = table.get_item(Key={'ticketId': parameters.get('ticketId')})
        return response.get('Item', {})
```
{% endraw %}

---

## Multi-Agent Customer Service System

### Orchestrated Multi-Agent System

```python
class MultiAgentCustomerServiceSystem:
    """Orchestrated multi-agent system with supervisor coordination"""
    
    def __init__(self):
        self.bedrock = boto3.client('bedrock')
        self.runtime = boto3.client('bedrock-runtime')
    
    def create_multi_agent_system(self) -> Dict:
        """Create integrated multi-agent customer service system"""
        
        # Create Supervisor Agent
        supervisor_agent = self._create_supervisor_agent()
        
        # Create Specialist Agents
        support_agent = self._create_support_specialist_agent()
        billing_agent = self._create_billing_specialist_agent()
        technical_agent = self._create_technical_specialist_agent()
        sales_agent = self._create_sales_specialist_agent()
        
        # Associate collaborators with supervisor
        self.bedrock.associate_agent_collaborator(
            supervisorAgentId=supervisor_agent['agentId'],
            agentCollaborators=[
                {
                    'agentId': support_agent['agentId'],
                    'collaboratorName': 'SupportSpecialist',
                    'collaboratorDescription': 'Handles general support issues'
                },
                {
                    'agentId': billing_agent['agentId'],
                    'collaboratorName': 'BillingSpecialist',
                    'collaboratorDescription': 'Handles billing and account questions'
                },
                {
                    'agentId': technical_agent['agentId'],
                    'collaboratorName': 'TechnicalSpecialist',
                    'collaboratorDescription': 'Handles technical issues'
                },
                {
                    'agentId': sales_agent['agentId'],
                    'collaboratorName': 'SalesSpecialist',
                    'collaboratorDescription': 'Handles sales inquiries'
                }
            ]
        )
        
        return {
            'supervisor_agent_id': supervisor_agent['agentId'],
            'specialist_agents': {
                'support': support_agent['agentId'],
                'billing': billing_agent['agentId'],
                'technical': technical_agent['agentId'],
                'sales': sales_agent['agentId']
            }
        }
    
    def _create_supervisor_agent(self) -> Dict:
        """Create supervisor agent for routing requests"""
        
        instructions = """You are the primary customer service supervisor responsible for:
1. Understanding customer inquiries
2. Classifying requests into appropriate categories
3. Routing to the right specialist agent
4. Monitoring responses
5. Ensuring customer satisfaction

Always greet customers professionally and clarify their needs before routing.
Categories:
- GENERAL_SUPPORT: General questions, troubleshooting
- BILLING: Invoices, payments, refunds
- TECHNICAL: System issues, integration problems
- SALES: New products, upgrades, inquiries"""
        
        response = self.bedrock.create_agent(
            agentName='CustomerServiceSupervisor',
            agentDescription='Supervisor agent for multi-agent customer service',
            foundationModelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
            instruction=instructions,
            agentCollaboratorMode='SUPERVISOR'
        )
        
        return response
    
    def _create_support_specialist_agent(self) -> Dict:
        """Create support specialist agent"""
        
        instructions = """You are a Support Specialist agent. Your responsibilities:
1. Answer general customer questions
2. Provide troubleshooting guidance
3. Create and manage support tickets
4. Escalate complex issues

Focus on: Common issues, FAQs, ticket management"""
        
        return self.bedrock.create_agent(
            agentName='SupportSpecialist',
            agentDescription='Handles general customer support',
            foundationModelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
            instruction=instructions
        )
    
    def _create_billing_specialist_agent(self) -> Dict:
        """Create billing specialist agent"""
        
        instructions = """You are a Billing Specialist agent. Responsibilities:
1. Manage billing inquiries
2. Process refunds
3. Handle payment issues
4. Review invoices

Focus on: Account management, billing, payments"""
        
        return self.bedrock.create_agent(
            agentName='BillingSpecialist',
            agentDescription='Handles billing and account questions',
            foundationModelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
            instruction=instructions
        )
    
    def _create_technical_specialist_agent(self) -> Dict:
        """Create technical specialist agent"""
        
        instructions = """You are a Technical Specialist agent. Responsibilities:
1. Troubleshoot system issues
2. Assist with integrations
3. Debug technical problems
4. Provide system status updates

Focus on: Technical issues, API problems, system errors"""
        
        return self.bedrock.create_agent(
            agentName='TechnicalSpecialist',
            agentDescription='Handles technical issues and troubleshooting',
            foundationModelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
            instruction=instructions
        )
    
    def _create_sales_specialist_agent(self) -> Dict:
        """Create sales specialist agent"""
        
        instructions = """You are a Sales Specialist agent. Responsibilities:
1. Answer product inquiries
2. Provide pricing information
3. Handle upgrade requests
4. Process new subscriptions

Focus on: Product features, pricing, sales opportunities"""
        
        return self.bedrock.create_agent(
            agentName='SalesSpecialist',
            agentDescription='Handles sales inquiries and product information',
            foundationModelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
            instruction=instructions
        )
```

---

## Multi-Agent System with A2A Protocol

### Orchestrated Multi-Agent System with Agent-to-Agent Communication

This recipe demonstrates how to build a multi-agent system where agents can communicate with each other using the Agent-to-Agent (A2A) protocol.

```python
class A2AMultiAgentSystem:
    """Multi-agent system with A2A communication"""

    def __init__(self):
        self.bedrock = boto3.client('bedrock')
        self.runtime = boto3.client('bedrock-runtime')

    def create_a2a_system(self):
        """Create a multi-agent system with A2A capabilities"""

        # Create a supervisor agent
        supervisor = self._create_supervisor_agent()

        # Create specialist agents
        researcher = self._create_researcher_agent()
        writer = self._create_writer_agent()

        # Enable A2A communication between the agents
        self._enable_a2a_communication(supervisor['agentId'], [researcher['agentId'], writer['agentId']])

        return {
            'supervisor': supervisor['agentId'],
            'researcher': researcher['agentId'],
            'writer': writer['agentId']
        }

    def _create_supervisor_agent(self):
        # ... implementation for creating a supervisor agent ...
        pass

    def _create_researcher_agent(self):
        # ... implementation for creating a researcher agent ...
        pass

    def _create_writer_agent(self):
        # ... implementation for creating a writer agent ...
        pass

    def _enable_a2a_communication(self, supervisor_id, specialist_ids):
        """Enable A2A communication between agents"""
        # In a real implementation, this would involve configuring the agents
        # to allow them to invoke each other. This is a conceptual example.
        print(f"Enabling A2A communication between supervisor {supervisor_id} and specialists {specialist_ids}")

```

---

## Financial Analytics Agent

### Financial Data Analysis and Reporting

{% raw %}
```python
class FinancialAnalyticsAgent:
    """Agent for financial data analysis and reporting"""
    
    def __init__(self):
        self.bedrock = boto3.client('bedrock')
        self.runtime = boto3.client('bedrock-runtime')
    
    def create_financial_agent(self) -> str:
        """Create financial analysis agent"""
        
        instructions = """You are a Financial Analytics Agent specialising in:
1. Financial data analysis
2. Report generation
3. Trend analysis
4. Anomaly detection
5. Forecasting

Always:
- Provide accurate calculations
- Cite data sources
- Highlight significant trends
- Identify risks and opportunities
- Present data clearly with visualisations"""
        
        agent_response = self.bedrock.create_agent(
            agentName='FinancialAnalytics',
            agentDescription='Advanced financial analysis and reporting',
            foundationModelId='anthropic.claude-3-opus-20240229-v1:0',
            instruction=instructions
        )
        
        agent_id = agent_response['agentId']
        
        # Create action groups for financial operations
        self._create_financial_action_groups(agent_id)
        
        # Create knowledge base with financial data
        self._create_financial_knowledge_base(agent_id)
        
        return agent_id
    
    def _create_financial_action_groups(self, agent_id: str) -> None:
        """Create action groups for financial operations"""
        
        # Action group for financial data retrieval
        financial_schema = {
            "openapi": "3.0.0",
            "info": {"title": "Financial Data API", "version": "1.0.0"},
            "paths": {
                "/financials/{period}": {
                    "get": {
                        "operationId": "GetFinancialData",
                        "summary": "Retrieve financial data for period",
                        "parameters": [
                            {"name": "period", "in": "path", "required": True, "schema": {"type": "string"}},
                            {"name": "dataType", "in": "query", "schema": {"type": "string", 
                                "enum": ["REVENUE", "EXPENSES", "PROFIT", "CASHFLOW"]}}
                        ],
                        "responses": {"200": {"description": "Financial data"}}
                    }
                },
                "/forecasts": {
                    "post": {
                        "operationId": "GenerateForecast",
                        "summary": "Generate financial forecasts",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "historicalDataPeriods": {"type": "integer"},
                                            "forecastMonths": {"type": "integer"},
                                            "confidenceLevel": {"type": "number"}
                                        }
                                    }
                                }
                            }
                        },
                        "responses": {"200": {"description": "Forecast data"}}
                    }
                }
            }
        }
        
        self.bedrock.create_action_group(
            agentId=agent_id,
            actionGroupName='FinancialDataRetrieval',
            actionGroupDescription='Retrieve and analyse financial data',
            apiSchema={'payload': json.dumps(financial_schema)},
            actionGroupExecutor={'lambda': 'arn:aws:lambda:us-east-1:ACCOUNT:function:financial-data-handler'}
        )
```
{% endraw %}

---

## Data Analysis Agent with Code Interpretation

### Agent that uses Code Interpretation for Data Analysis

This recipe demonstrates how to create an agent that uses the code interpretation feature to perform data analysis on a CSV file.

```python
class DataAnalysisAgent:
    """Agent for data analysis using code interpretation"""

    def __init__(self):
        self.bedrock = boto3.client('bedrock')
        self.runtime = boto3.client('bedrock-runtime')

    def create_data_analysis_agent(self):
        """Create a data analysis agent with code interpretation enabled"""

        instructions = """You are a data analyst. You can write and execute Python code to answer questions about data.
When you are asked to analyze a file, you can use the `inputFiles` parameter to access the file."""

        response = self.bedrock.create_agent(
            agentName='DataAnalysisAgent',
            foundationModelId='anthropic.claude-3-5-sonnet-20240620-v1:0',
            instruction=instructions,
            enableCodeInterpreter=True
        )

        return response['agentId']

    def invoke_data_analysis_agent(self, agent_id, s3_uri):
        """Invoke the data analysis agent with a CSV file"""

        response = self.runtime.invoke_agent(
            agentId=agent_id,
            agentAliasId='YOUR_AGENT_ALIAS_ID',
            inputText='Analyze the attached CSV file and provide a summary of the data.',
            sessionId='session-1',
            inputFiles=[
                {
                    'name': 'sales_data.csv',
                    'source': {
                        's3': {
                            'uri': s3_uri
                        }
                    }
                }
            ]
        )

        return response['outputText']

```

---

## Document Processing Pipeline

### Automated Document Analysis and Processing

```python
class DocumentProcessingAgent:
    """Agent for document processing and analysis"""
    
    def __init__(self):
        self.bedrock = boto3.client('bedrock')
        self.s3 = boto3.client('s3')
        self.textract = boto3.client('textract')
    
    def create_document_processing_agent(self) -> str:
        """Create document processing agent"""
        
        instructions = """You are a Document Processing Agent capable of:
1. Document text extraction
2. Key information extraction
3. Document classification
4. Data validation
5. Report generation

For each document:
- Extract key information
- Classify document type
- Validate data quality
- Flag anomalies or missing information
- Generate processing report"""
        
        agent_response = self.bedrock.create_agent(
            agentName='DocumentProcessing',
            agentDescription='Automated document analysis and processing',
            foundationModelId='anthropic.claude-3-sonnet-20240229-v1:0',
            instruction=instructions
        )
        
        return agent_response['agentId']
```

---

This recipes document provides practical, production-ready implementations for common Bedrock Agents use cases. Each recipe can be customised for specific requirements and scaled for enterprise deployments.

