---
title: "Amazon Bedrock Agents: Comprehensive Technical Guide"
description: "End-to-end reference for Amazon Bedrock Agents (Agents Classic): IAM, action groups, knowledge bases, guardrails, flows, models, regions and infrastructure as code, with boto3 examples."
framework: amazon-bedrock-agents
---

# Amazon Bedrock Agents: Comprehensive Technical Guide

> **Verified against:** `boto3` %%version=boto3%% (checked %%versions-checked%%). Service clients and operation names in the examples were checked against that release.

> **Maintenance mode.** Amazon Bedrock Agents — now called *Agents Classic* — stopped accepting new customers on 30 July 2026. Existing customers can keep using it, and this guide remains accurate for them. For new work AWS recommends [Amazon Bedrock AgentCore](/amazon-bedrock-agents-guide/bedrock_agentcore_comprehensive_guide/); see AWS's [migration guidance](https://docs.aws.amazon.com/bedrock/latest/userguide/agents-classic-maintenance-mode.html).

**Which boto3 client?** Agent, action-group, knowledge-base and flow *management* uses `bedrock-agent`; *invoking* agents and querying knowledge bases uses `bedrock-agent-runtime`; guardrail management and model listing use `bedrock`; direct model calls and `ApplyGuardrail` use `bedrock-runtime`.

A definitive, end-to-end technical reference covering Amazon Bedrock Agents from foundational concepts through enterprise-scale production deployment. This guide provides exhaustive coverage of architecture patterns, implementation strategies, AWS integration techniques, cost optimisation, security best practices, and advanced agentic patterns.

---

## Table of Contents

1. [Core Fundamentals](#core-fundamentals)
   - [Amazon Bedrock AgentCore](#amazon-bedrock-agentcore)
   - [Best Practices for Building Bedrock Agents](#best-practices-for-building-bedrock-agents)
   - [AWS Bedrock Setup and IAM Permissions](#aws-bedrock-setup-and-iam-permissions)
   - [Bedrock Agents Architecture Overview](#bedrock-agents-architecture-overview)
   - [Core Components: Action Groups, Knowledge Bases, Guardrails, Prompt Flows](#core-components-action-groups-knowledge-bases-guardrails-prompt-flows)
   - [Supported Foundation Models](#supported-foundation-models)
   - [Region Availability and Configuration](#region-availability-and-configuration)

2. [Multi-Agent Systems (MAS)](#multi-agent-systems-mas)
   - [Creating Basic Bedrock Agents via Console](#creating-basic-bedrock-agents-via-console)
   - [Creating Basic Bedrock Agents via CloudFormation and Terraform](#creating-basic-bedrock-agents-via-cloudformation-and-terraform)

**Continue elsewhere in this guide:** [AgentCore services](/amazon-bedrock-agents-guide/bedrock_agentcore_comprehensive_guide/) · [Advanced multi-agent & supervisor patterns](/amazon-bedrock-agents-guide/bedrock_agents_advanced_multi_agent_python/) · [A2A protocol](/amazon-bedrock-agents-guide/bedrock_a2a_protocol_guide/) · [Strands SDK](/amazon-bedrock-agents-guide/bedrock_strands_sdk_guide/) · [Security & IAM](/amazon-bedrock-agents-guide/bedrock_security_iam_examples/) · [Error handling](/amazon-bedrock-agents-guide/bedrock_agents_error_handling_python/) · [Observability](/amazon-bedrock-agents-guide/bedrock_agents_observability_python/) · [Production guide](/amazon-bedrock-agents-guide/bedrock_agents_production_guide/) · [Recipes](/amazon-bedrock-agents-guide/bedrock_agents_recipes/).

---

## Core Fundamentals

### Amazon Bedrock AgentCore

Amazon Bedrock AgentCore (generally available since October 2025) is AWS's platform for deploying and operating agents built with any framework and model. It is the recommended successor to Bedrock Agents for new work.

**Key services of AgentCore:**

*   **Runtime:** Session-isolated, serverless hosting for agents and MCP/A2A servers.
*   **Harness:** A managed agent loop declared as configuration (model, prompt, tools).
*   **Gateway:** Exposes Lambda functions, APIs and MCP servers to agents as MCP tools.
*   **Memory:** Short-term conversation events and long-term extracted memories.
*   **Identity:** Inbound JWT/IAM auth and outbound OAuth/API-key credentials for agents.
*   **Code Interpreter and Browser:** Managed sandboxes for code execution and web automation.
*   **Observability:** OpenTelemetry traces and metrics in CloudWatch.

Details and examples are in the [AgentCore guide](/amazon-bedrock-agents-guide/bedrock_agentcore_comprehensive_guide/).

### Best Practices for Building Bedrock Agents

*   **Define Clear Use Cases and Objectives:** Establish a solid foundation with high-quality ground truth data and meticulously defined instructions.
*   **Build Small and Focused Agents:** Design agents to be modular and specialized, allowing them to collaborate for more complex tasks.
*   **Integrate Actions and APIs Thoughtfully:** Use Infrastructure as Code (IaC) to create and deploy reusable action groups.
*   **Optimize Your Knowledge Base Design:** Integrate agents with existing organizational knowledge bases to provide accurate, context-aware responses.
*   **Implement Robust Security and Access Control:** Use Guardrails to avoid sensitive topics, filter harmful content, and redact sensitive information.
*   **Test and Iterate in Real-World Scenarios:** Begin with a comprehensive set of test cases derived from actual user interactions.
*   **Leverage Infrastructure as Code (IaC):** Use IaC (CloudFormation, AWS CDK, Terraform) to create and deploy agents, guardrails, and knowledge bases for reusability and consistency.
*   **Continuous Optimization:** Continuously optimize for cost and performance as your agents evolve and scale.

### AWS Bedrock Setup and IAM Permissions

Amazon Bedrock Agents require comprehensive AWS environment setup with granular Identity and Access Management (IAM) configurations to ensure secure, scalable agent operations.

#### Setting Up AWS Account and Bedrock Access

**Prerequisites:**
- Active AWS Account with administrative or appropriate delegated access
- Region selection supporting Amazon Bedrock (us-east-1, us-west-2, eu-west-1, ap-southeast-1)
- AWS CLI v2 installed and configured
- Python 3.10+ (required by current boto3) or Node.js 18+ for SDK operations

**AWS Bedrock Console Access:**

```bash
# Verify Bedrock service availability in your region
aws bedrock list-foundation-models --region us-east-1

# List the Anthropic models offered in this Region
aws bedrock list-foundation-models --by-provider anthropic --region us-east-1 \
  --query 'modelSummaries[].modelId'

# List cross-Region inference profiles (IDs such as us.anthropic.claude-sonnet-5-5)
aws bedrock list-inference-profiles --region us-east-1
```

#### Comprehensive IAM Role Configuration

**Trust Policy Document:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": [
          "bedrock.amazonaws.com",
          "lambda.amazonaws.com",
          "states.amazonaws.com",
          "events.amazonaws.com"
        ]
      },
      "Action": "sts:AssumeRole"
    },
    {
      "Effect": "Allow",
      "Principal": {
        "AWS": "arn:aws:iam::ACCOUNT_ID:root"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
```

**Granular Permissions Policy for Agent Operations:**

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "BedrockAgentPermissions",
      "Effect": "Allow",
      "Action": [
        "bedrock:CreateAgent",
        "bedrock:UpdateAgent",
        "bedrock:PrepareAgent",
        "bedrock:GetAgent",
        "bedrock:ListAgents",
        "bedrock:DeleteAgent",
        "bedrock:InvokeAgent",
        "bedrock:CreateAgentAlias",
        "bedrock:UpdateAgentAlias",
        "bedrock:GetAgentAlias",
        "bedrock:CreateAgentActionGroup",
        "bedrock:UpdateAgentActionGroup",
        "bedrock:GetAgentActionGroup",
        "bedrock:DeleteAgentActionGroup",
        "bedrock:AssociateAgentKnowledgeBase",
        "bedrock:CreateKnowledgeBase",
        "bedrock:UpdateKnowledgeBase",
        "bedrock:GetKnowledgeBase",
        "bedrock:DeleteKnowledgeBase",
        "bedrock:CreateDataSource",
        "bedrock:UpdateDataSource",
        "bedrock:GetDataSource",
        "bedrock:DeleteDataSource",
        "bedrock:StartIngestionJob",
        "bedrock:Retrieve",
        "bedrock:RetrieveAndGenerate",
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": "arn:aws:bedrock:*:ACCOUNT_ID:*"
    },
    {
      "Sid": "BedrockModelAccess",
      "Effect": "Allow",
      "Action": [
        "bedrock:GetFoundationModel",
        "bedrock:ListFoundationModels"
      ],
      "Resource": "*"
    },
    {
      "Sid": "S3DataSourceAccess",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:ListBucket",
        "s3:ListBucketVersions"
      ],
      "Resource": [
        "arn:aws:s3:::knowledge-base-bucket/*",
        "arn:aws:s3:::knowledge-base-bucket"
      ]
    },
    {
      "Sid": "OpenSearchAccess",
      "Effect": "Allow",
      "Action": [
        "aoss:ApiAccessAll"
      ],
      "Resource": "arn:aws:aoss:*:ACCOUNT_ID:collection/*"
    },
    {
      "Sid": "LambdaInvocation",
      "Effect": "Allow",
      "Action": [
        "lambda:InvokeFunction"
      ],
      "Resource": "arn:aws:lambda:*:ACCOUNT_ID:function:bedrock-*"
    },
    {
      "Sid": "SecretsManagerAccess",
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue",
        "secretsmanager:DescribeSecret"
      ],
      "Resource": "arn:aws:secretsmanager:*:ACCOUNT_ID:secret:bedrock/*"
    },
    {
      "Sid": "CloudWatchLogsAccess",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:ACCOUNT_ID:log-group:/aws/bedrock/*"
    },
    {
      "Sid": "KMSEncryptionAccess",
      "Effect": "Allow",
      "Action": [
        "kms:Decrypt",
        "kms:GenerateDataKey",
        "kms:DescribeKey"
      ],
      "Resource": "arn:aws:kms:*:ACCOUNT_ID:key/*"
    }
  ]
}
```

**IAM Role Creation via AWS CLI:**

```bash
# Create the role
aws iam create-role \
  --role-name BedrockAgentRole \
  --assume-role-policy-document file://trust-policy.json \
  --description "Role for Amazon Bedrock Agents operations"

# Create and attach the policy
aws iam put-role-policy \
  --role-name BedrockAgentRole \
  --policy-name BedrockAgentPermissions \
  --policy-document file://permissions-policy.json

# Verify role creation
aws iam get-role --role-name BedrockAgentRole

# Get the role ARN
aws iam get-role --role-name BedrockAgentRole --query 'Role.Arn' --output text
```

#### Python SDK IAM Configuration

```python
import boto3
import json
from botocore.exceptions import ClientError

class BedrockIAMManager:
    """Manages IAM configurations for Bedrock Agents"""
    
    def __init__(self, region_name='us-east-1'):
        self.iam_client = boto3.client('iam', region_name=region_name)
        self.region = region_name
    
    def create_bedrock_agent_role(self, role_name, s3_bucket_arn=None, kms_key_arn=None):
        """
        Create a comprehensive IAM role for Bedrock Agents
        
        Args:
            role_name (str): Name of the IAM role
            s3_bucket_arn (str): ARN of S3 bucket for knowledge bases
            kms_key_arn (str): ARN of KMS key for encryption
        
        Returns:
            dict: Role creation response
        """
        
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "bedrock.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        try:
            # Create the role
            response = self.iam_client.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description="Role for Amazon Bedrock Agents",
                MaxSessionDuration=3600
            )
            
            print(f"✓ Created IAM role: {response['Role']['Arn']}")
            return response
        
        except ClientError as e:
            if e.response['Error']['Code'] == 'EntityAlreadyExists':
                print(f"✓ Role already exists: {role_name}")
                return self.iam_client.get_role(RoleName=role_name)
            else:
                raise
    
    def attach_bedrock_permissions(self, role_name, s3_bucket_arn=None, lambda_function_arns=None):
        """
        Attach comprehensive permissions to Bedrock agent role
        
        Args:
            role_name (str): Name of the role
            s3_bucket_arn (str): S3 bucket for knowledge bases
            lambda_function_arns (list): Lambda functions for action groups
        """
        
        # Build resource ARNs
        bedrock_resource = f"arn:aws:bedrock:{self.region}:*:*"
        s3_resources = [s3_bucket_arn] if s3_bucket_arn else ["arn:aws:s3:::*"]
        lambda_resources = lambda_function_arns or ["arn:aws:lambda:*:*:function:bedrock-*"]
        
        permissions_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "BedrockModelInvocation",
                    "Effect": "Allow",
                    "Action": [
                        "bedrock:InvokeModel",
                        "bedrock:InvokeModelWithResponseStream",
                        "bedrock:GetFoundationModel",
                        "bedrock:GetInferenceProfile"
                    ],
                    "Resource": bedrock_resource
                },
                {
                    "Sid": "S3AccessForKnowledgeBases",
                    "Effect": "Allow",
                    "Action": ["s3:GetObject", "s3:ListBucket"],
                    "Resource": s3_resources
                },
                {
                    "Sid": "LambdaActionGroups",
                    "Effect": "Allow",
                    "Action": ["lambda:InvokeFunction"],
                    "Resource": lambda_resources
                },
                {
                    "Sid": "CloudWatchLogs",
                    "Effect": "Allow",
                    "Action": [
                        "logs:CreateLogGroup",
                        "logs:CreateLogStream",
                        "logs:PutLogEvents"
                    ],
                    "Resource": f"arn:aws:logs:{self.region}:*:log-group:/aws/bedrock/*"
                }
            ]
        }
        
        try:
            self.iam_client.put_role_policy(
                RoleName=role_name,
                PolicyName='BedrockAgentPermissions',
                PolicyDocument=json.dumps(permissions_policy)
            )
            print(f"✓ Attached permissions to role: {role_name}")
        except ClientError as e:
            print(f"✗ Error attaching permissions: {e}")
            raise

# Usage
manager = BedrockIAMManager()
manager.create_bedrock_agent_role('BedrockAgentRole')
manager.attach_bedrock_permissions('BedrockAgentRole')
```

### Bedrock Agents Architecture Overview

Amazon Bedrock Agents orchestrate sophisticated interactions between foundation models, external data sources, APIs, and user inputs through a hierarchical orchestration framework.

#### Core Architecture Components

**Architectural Layers:**

```
┌─────────────────────────────────────────────────────────────┐
│                      User Interface Layer                    │
│              (Console, API, SDK, Chat Interface)             │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    Agent Orchestration Layer                  │
│    (Agent State Machine, Request Processing, Routing)        │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
┌───────▼─────┐ ┌─────▼──────┐ ┌────▼────────┐
│  Foundation │ │   Prompt   │ │  Action     │
│   Model     │ │   Engine   │ │  Groups     │
└───────┬─────┘ └─────┬──────┘ └────┬────────┘
        │              │             │
        └──────────────┼─────────────┘
                       │
        ┌──────────────┼──────────────┐
        │              │              │
┌───────▼─────┐ ┌─────▼──────┐ ┌────▼────────┐
│ Knowledge   │ │ Guardrails │ │  External   │
│ Bases       │ │ & Policies │ │  APIs/Tools │
└─────────────┘ └────────────┘ └─────────────┘
```

#### Request Processing Workflow

**Step-by-Step Agent Invocation Flow:**

1. **Request Reception**: User input received via SDK, API, or console
2. **Pre-Processing**: Input validation, guardrails pre-check, context loading
3. **Model Invocation**: Request sent to foundation model with instructions and context
4. **Reasoning Loop**: Model generates intermediate steps (ReAct-style reasoning)
5. **Action Selection**: Model identifies required actions from available action groups
6. **Tool Execution**: Action groups execute APIs, Lambda functions, or database queries
7. **Knowledge Integration**: Knowledge bases queried for relevant context
8. **Response Generation**: Model generates final response based on tool results
9. **Post-Processing**: Guardrails enforcement, output formatting, citation generation
10. **Response Return**: Final response returned to user with metadata

#### Cost Structure Analysis

**Pricing Components:**

| Component | Unit | Approximate Cost | Notes |
|-----------|------|------------------|-------|
| Foundation Model Invocation | Per million input/output tokens | Varies widely by model | The main cost driver; see [Bedrock pricing](https://aws.amazon.com/bedrock/pricing/) |
| Action Group Invocation | Per invocation | No additional charge | Charged via underlying service (Lambda, API) |
| Knowledge Base Query | Per query | Embedding + vector store + model tokens | Vector store (e.g. OpenSearch Serverless) is billed separately |
| Storage (S3) | Per GB/month | $0.023 | Standard S3 rates |
| Agent Version/Alias Management | Per version | Free | No charge for creating versions/aliases |

**Cost Optimisation Strategies:**

- Implement batching for multiple user requests
- Use model selection based on task complexity
- Cache frequently accessed knowledge base results
- Monitor token consumption per invocation
- Implement rate limiting and throttling

#### Enterprise Deployment Considerations

**Multi-Account Strategy:**

```bash
# Create agents in multiple accounts within an AWS Organization
# Use cross-account IAM roles for centralized management

# Central account role
aws iam create-role \
  --role-name CentralBedrockAgentRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"AWS": "arn:aws:iam::MEMBER_ACCOUNT:root"},
      "Action": "sts:AssumeRole"
    }]
  }'

# Member account can then assume central role for agent operations
```

**High Availability Pattern:**

```python
import boto3
from botocore.exceptions import ClientError
import time

class HighAvailabilityBedrockManager:
    """Implements HA patterns for Bedrock Agents"""
    
    def __init__(self, primary_region, fallback_region, agents):
        """agents maps region -> (agent_id, agent_alias_id); agent IDs differ per Region."""
        self.primary_client = boto3.client('bedrock-agent-runtime', region_name=primary_region)
        self.fallback_client = boto3.client('bedrock-agent-runtime', region_name=fallback_region)
        self.primary_region = primary_region
        self.fallback_region = fallback_region
        self.agents = agents
    
    def invoke_with_failover(self, session_id, input_text, max_retries=3):
        """Invoke agent with automatic failover to secondary region"""
        
        retries = 0
        while retries < max_retries:
            try:
                # Try primary region
                agent_id, alias_id = self.agents[self.primary_region]
                response = self.primary_client.invoke_agent(
                    agentId=agent_id,
                    agentAliasId=alias_id,
                    sessionId=session_id,
                    inputText=input_text,
                    enableTrace=True
                )
                print(f"✓ Successfully invoked agent in {self.primary_region}")
                return response
            
            except ClientError as e:
                if e.response['Error']['Code'] in ['ServiceUnavailableException', 'ThrottlingException']:
                    print(f"⚠ Primary region unavailable, attempting fallback...")
                    try:
                        # Try fallback region (a separate agent deployed there)
                        agent_id, alias_id = self.agents[self.fallback_region]
                        response = self.fallback_client.invoke_agent(
                            agentId=agent_id,
                            agentAliasId=alias_id,
                            sessionId=session_id,
                            inputText=input_text,
                            enableTrace=True
                        )
                        print(f"✓ Successfully invoked agent in {self.fallback_region}")
                        return response
                    except ClientError as fallback_error:
                        print(f"✗ Fallback also failed: {fallback_error}")
                        retries += 1
                        if retries < max_retries:
                            time.sleep(2 ** retries)  # Exponential backoff
                else:
                    raise

# Usage
ha_manager = HighAvailabilityBedrockManager(
    'us-east-1', 'us-west-2',
    agents={'us-east-1': ('AGENT_ID_EAST', 'ALIAS_ID_EAST'), 'us-west-2': ('AGENT_ID_WEST', 'ALIAS_ID_WEST')},
)
response = ha_manager.invoke_with_failover('session-1', 'What is the status of my account?')

# invoke_agent returns an event stream; join the chunks for the final text
answer = ''.join(
    event['chunk']['bytes'].decode() for event in response['completion'] if 'chunk' in event
)
```

### Core Components: Action Groups, Knowledge Bases, Guardrails, Prompt Flows

#### Action Groups: API-Driven Agent Capabilities

**Action groups** define the set of APIs and functions an agent can invoke to accomplish tasks. They serve as the interface between the agent's reasoning and external systems.

**Comprehensive Action Group Architecture:**

```python
import json
import boto3
from typing import Dict, List, Optional

class ActionGroupManager:
    """Comprehensive management of Bedrock Agent Action Groups"""
    
    def __init__(self, agent_id: str, region_name='us-east-1'):
        self.bedrock_client = boto3.client('bedrock-agent', region_name=region_name)
        self.agent_id = agent_id
        self.region_name = region_name
    
    def create_lambda_action_group(self, action_group_name: str, 
                                   lambda_function_arn: str,
                                   description: str,
                                   api_schema: Dict) -> Dict:
        """
        Create action group that invokes Lambda functions
        
        Args:
            action_group_name: Name of the action group
            lambda_function_arn: ARN of Lambda function to invoke
            description: Description of the action group
            api_schema: OpenAPI schema defining the function interface
        
        Returns:
            Created action group response
        """
        
        try:
            response = self.bedrock_client.create_agent_action_group(
                agentId=self.agent_id,
                agentVersion='DRAFT',
                actionGroupName=action_group_name,
                description=description,
                actionGroupExecutor={
                    'lambda': lambda_function_arn
                },
                apiSchema={
                    'payload': json.dumps(api_schema)
                }
            )
            print(f"✓ Created Lambda action group: {action_group_name}")
            return response
        except Exception as e:
            print(f"✗ Error creating action group: {e}")
            raise
    
    def create_return_control_action_group(self, action_group_name: str,
                                           description: str,
                                           api_schema: Dict) -> Dict:
        """
        Create an action group whose calls are returned to your application
        (return of control) instead of being executed by Lambda.
        
        Bedrock Agents has no built-in HTTP executor: to call an external API,
        either put the call in a Lambda function, or use RETURN_CONTROL and make
        the call from your application, sending the result back with
        sessionState['returnControlInvocationResults'] on the next invoke_agent.
        
        Args:
            action_group_name: Name of the action group
            description: Description
            api_schema: OpenAPI specification
        
        Returns:
            Created action group response
        """
        
        try:
            response = self.bedrock_client.create_agent_action_group(
                agentId=self.agent_id,
                agentVersion='DRAFT',
                actionGroupName=action_group_name,
                description=description,
                actionGroupExecutor={'customControl': 'RETURN_CONTROL'},
                apiSchema={
                    'payload': json.dumps(api_schema)
                }
            )
            print(f"✓ Created return-of-control action group: {action_group_name}")
            return response
        except Exception as e:
            print(f"✗ Error creating action group: {e}")
            raise
```

**OpenAPI Schema Example for Customer Management:**

```json
{
  "openapi": "3.0.0",
  "info": {
    "title": "Customer Management API",
    "version": "1.0.0",
    "description": "APIs for managing customer accounts and profiles"
  },
  "servers": [
    {
      "url": "https://api.example.com/v1",
      "description": "Production API server"
    }
  ],
  "paths": {
    "/customers/{customerId}": {
      "get": {
        "operationId": "GetCustomerById",
        "summary": "Retrieve customer information by ID",
        "parameters": [
          {
            "name": "customerId",
            "in": "path",
            "required": true,
            "schema": {
              "type": "string",
              "description": "Unique customer identifier"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Customer found",
            "content": {
              "application/json": {
                "schema": {
                  "type": "object",
                  "properties": {
                    "customerId": {"type": "string"},
                    "firstName": {"type": "string"},
                    "lastName": {"type": "string"},
                    "email": {"type": "string", "format": "email"},
                    "phone": {"type": "string"},
                    "accountStatus": {
                      "type": "string",
                      "enum": ["ACTIVE", "SUSPENDED", "CLOSED"]
                    },
                    "createdDate": {"type": "string", "format": "date-time"},
                    "lastOrderDate": {"type": "string", "format": "date-time"},
                    "totalOrderValue": {"type": "number", "format": "double"}
                  },
                  "required": ["customerId", "email", "accountStatus"]
                }
              }
            }
          },
          "404": {
            "description": "Customer not found"
          }
        }
      },
      "put": {
        "operationId": "UpdateCustomer",
        "summary": "Update customer information",
        "parameters": [
          {
            "name": "customerId",
            "in": "path",
            "required": true,
            "schema": {"type": "string"}
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "firstName": {"type": "string"},
                  "lastName": {"type": "string"},
                  "email": {"type": "string"},
                  "phone": {"type": "string"},
                  "accountStatus": {
                    "type": "string",
                    "enum": ["ACTIVE", "SUSPENDED", "CLOSED"]
                  }
                }
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Customer updated successfully"
          }
        }
      }
    },
    "/customers": {
      "post": {
        "operationId": "CreateCustomer",
        "summary": "Create a new customer",
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "type": "object",
                "properties": {
                  "firstName": {"type": "string"},
                  "lastName": {"type": "string"},
                  "email": {"type": "string"},
                  "phone": {"type": "string"}
                },
                "required": ["firstName", "lastName", "email"]
              }
            }
          }
        },
        "responses": {
          "201": {
            "description": "Customer created successfully"
          }
        }
      }
    }
  }
}
```

#### Knowledge Bases: Contextual Information Integration

**Knowledge bases** enable agents to augment responses with current, enterprise-specific information through semantic search and retrieval-augmented generation (RAG).

**Knowledge Base Architecture:**

```python
import boto3
from typing import List, Dict, Optional
import uuid

class KnowledgeBaseManager:
    """Comprehensive management of Bedrock Knowledge Bases"""
    
    def __init__(self, region_name='us-east-1'):
        self.bedrock_client = boto3.client('bedrock-agent', region_name=region_name)
        self.opensearch_client = boto3.client('opensearchserverless', region_name=region_name)
        self.s3_client = boto3.client('s3', region_name=region_name)
    
    def create_knowledge_base(self, kb_name: str,
                             description: str,
                             s3_bucket_arn: str,
                             opensearch_collection_arn: str,
                             role_arn: str,
                             embedding_model_arn: str = 'arn:aws:bedrock:us-east-1::foundation-model/amazon.titan-embed-text-v2:0') -> Dict:
        """
        Create a comprehensive knowledge base with OpenSearch backend
        
        Args:
            kb_name: Name of the knowledge base
            description: Description
            s3_bucket_arn: ARN of S3 bucket for source documents
            opensearch_collection_arn: ARN of OpenSearch Serverless collection
            role_arn: IAM role ARN with necessary permissions
            embedding_model_arn: ARN of the embedding model
        
        Returns:
            Created knowledge base response
        """
        
        try:
            response = self.bedrock_client.create_knowledge_base(
                name=kb_name,
                description=description,
                roleArn=role_arn,
                knowledgeBaseConfiguration={
                    'type': 'VECTOR',
                    'vectorKnowledgeBaseConfiguration': {
                        'embeddingModelArn': embedding_model_arn
                    }
                },
                storageConfiguration={
                    'type': 'OPENSEARCH_SERVERLESS',
                    'opensearchServerlessConfiguration': {
                        'collectionArn': opensearch_collection_arn,
                        'vectorIndexName': f'bedrock-index-{uuid.uuid4().hex[:8]}',
                        'fieldMapping': {
                            'vectorField': 'bedrock-vector',
                            'textField': 'AMAZON_BEDROCK_TEXT_CHUNK',
                            'metadataField': 'AMAZON_BEDROCK_METADATA'
                        }
                    }
                }
            )
            print(f"✓ Created knowledge base: {kb_name}")
            return response
        except Exception as e:
            print(f"✗ Error creating knowledge base: {e}")
            raise
    
    def create_data_source(self, knowledge_base_id: str,
                          data_source_name: str,
                          s3_bucket_arn: str,
                          s3_prefix: str = '',
                          chunking_strategy: str = 'FIXED_SIZE',
                          chunk_size: int = 1000,
                          chunk_overlap: int = 100) -> Dict:
        """
        Create a data source for ingesting documents into knowledge base
        
        Args:
            knowledge_base_id: ID of the knowledge base
            data_source_name: Name of the data source
            s3_bucket_arn: ARN of the bucket holding the documents (arn:aws:s3:::bucket)
            s3_prefix: Optional key prefix to restrict ingestion to
            chunking_strategy: Strategy for chunking documents
            chunk_size: Size of each chunk in tokens
            chunk_overlap: Overlap between chunks
        
        Returns:
            Created data source response
        """
        
        try:
            response = self.bedrock_client.create_data_source(
                knowledgeBaseId=knowledge_base_id,
                name=data_source_name,
                description=f"Data source for {data_source_name}",
                dataSourceConfiguration={
                    'type': 'S3',
                    's3Configuration': {
                        'bucketArn': s3_bucket_arn,
                        **({'inclusionPrefixes': [s3_prefix]} if s3_prefix else {})
                    }
                },
                vectorIngestionConfiguration={
                    'chunkingConfiguration': {
                        'chunkingStrategy': chunking_strategy,
                        'fixedSizeChunkingConfiguration': {
                            'maxTokens': chunk_size,
                            'overlapPercentage': int((chunk_overlap / chunk_size) * 100)
                        }
                    }
                }
            )
            print(f"✓ Created data source: {data_source_name}")
            return response
        except Exception as e:
            print(f"✗ Error creating data source: {e}")
            raise
```

**Knowledge Base Chunking Strategies:**

```python
class ChunkingStrategyManager:
    """Advanced chunking strategies for different content types"""
    
    @staticmethod
    def fixed_size_chunking(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """
        Fixed-size chunking: Splits text into fixed-size chunks with overlap
        
        Best for: Dense technical documentation, legal documents
        """
        chunks = []
        step = chunk_size - overlap
        
        for i in range(0, len(text), step):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks
    
    @staticmethod
    def semantic_chunking(text: str, sentences_per_chunk: int = 5) -> List[str]:
        """
        Semantic chunking: Splits text at sentence boundaries
        
        Best for: News articles, blog posts, general content
        """
        import re
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        
        for sentence in sentences:
            current_chunk.append(sentence)
            if len(current_chunk) >= sentences_per_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    @staticmethod
    def hierarchical_chunking(text: str, max_chunk_size: int = 1000) -> Dict[str, List[str]]:
        """
        Hierarchical chunking: Preserves document structure (sections, subsections)
        
        Best for: Structured documents with clear hierarchy
        """
        import re
        
        chunks_by_level = {'level1': [], 'level2': [], 'level3': []}
        
        # Parse document structure
        level1_sections = re.split(r'^## ', text, flags=re.MULTILINE)
        
        for section in level1_sections:
            if not section.strip():
                continue
            
            level2_sections = re.split(r'^### ', section, flags=re.MULTILINE)
            
            for subsection in level2_sections:
                if len(subsection) > max_chunk_size:
                    # Break into smaller chunks
                    for chunk in ChunkingStrategyManager.fixed_size_chunking(subsection, max_chunk_size):
                        chunks_by_level['level3'].append(chunk)
                else:
                    chunks_by_level['level2'].append(subsection)
        
        return chunks_by_level
```

#### Guardrails: Safety and Compliance

**Guardrails** provide multi-layered protection against inappropriate model behaviour, ensuring compliance with organisational policies and regulatory requirements.

```python
class GuardrailsManager:
    """Comprehensive Bedrock Guardrails management"""
    
    def __init__(self, region_name='us-east-1'):
        self.bedrock_client = boto3.client('bedrock', region_name=region_name)
    
    def create_comprehensive_guardrail(self, guardrail_name: str,
                                      description: str) -> Dict:
        """
        Create a comprehensive guardrail with multiple safety policies
        """
        
        try:
            response = self.bedrock_client.create_guardrail(
                name=guardrail_name,
                description=description,
                blockedInputMessaging="Sorry, I can't help with that request.",
                blockedOutputsMessaging="Sorry, I can't provide that response.",
                topicPolicyConfig={
                    'topicsConfig': [
                        {
                            'name': 'Financial Fraud',
                            'definition': 'Anything related to financial fraud, money laundering, or illegal financial activities',
                            'examples': [
                                'How to commit wire fraud',
                                'Ways to launder money',
                                'How to create fake invoices'
                            ],
                            'type': 'DENY'
                        },
                        {
                            'name': 'Illegal Activities',
                            'definition': 'Requests for help with illegal activities',
                            'examples': [
                                'How to make explosives',
                                'How to hack a system',
                                'How to create counterfeit currency'
                            ],
                            'type': 'DENY'
                        }
                    ]
                },
                contentPolicyConfig={
                    'filtersConfig': [
                        {
                            'type': 'HATE',
                            'inputStrength': 'HIGH',
                            'outputStrength': 'HIGH'
                        },
                        {
                            'type': 'INSULTS',
                            'inputStrength': 'MEDIUM',
                            'outputStrength': 'MEDIUM'
                        },
                        {
                            'type': 'SEXUAL',
                            'inputStrength': 'HIGH',
                            'outputStrength': 'HIGH'
                        },
                        {
                            'type': 'VIOLENCE',
                            'inputStrength': 'HIGH',
                            'outputStrength': 'HIGH'
                        }
                    ]
                },
                wordPolicyConfig={
                    'wordsConfig': [
                        {
                            'text': 'proprietary',
                            'action': 'BLOCK'
                        },
                        {
                            'text': 'confidential',
                            'action': 'BLOCK'
                        }
                    ],
                    'managedWordListConfig': [
                        {
                            'type': 'PROFANITY'
                        }
                    ]
                },
                sensitiveInformationPolicyConfig={
                    'piiEntitiesConfig': [
                        {
                            'type': 'EMAIL',
                            'action': 'ANONYMIZE'
                        },
                        {
                            'type': 'PHONE',
                            'action': 'ANONYMIZE'
                        },
                        {
                            'type': 'NAME',
                            'action': 'ANONYMIZE'
                        },
                        {
                            'type': 'ADDRESS',
                            'action': 'ANONYMIZE'
                        },
                        {
                            'type': 'CREDIT_DEBIT_CARD_NUMBER',
                            'action': 'BLOCK'
                        },
                        {
                            'type': 'BANK_ACCOUNT_NUMBER',
                            'action': 'BLOCK'
                        }
                    ],
                    'regexesConfig': [
                        {
                            'name': 'AWS_SECRET_KEY',
                            'pattern': r'AKIA[0-9A-Z]{16}',
                            'action': 'BLOCK'
                        }
                    ]
                }
            )
            print(f"✓ Created guardrail: {guardrail_name}")
            return response
        except Exception as e:
            print(f"✗ Error creating guardrail: {e}")
            raise
```

Guardrails are managed with the `bedrock` client (as above). Attach one to an agent with `guardrailConfiguration={'guardrailIdentifier': ..., 'guardrailVersion': ...}` on `bedrock-agent` `create_agent` / `update_agent`.

#### Applying Guardrails Programmatically

In addition to associating guardrails with agents at creation time, you can also apply guardrails programmatically to individual API requests using the `ApplyGuardrail` API. This is useful for multi-tenant scenarios or situations requiring dynamic moderation.

```python
import boto3

bedrock_runtime = boto3.client('bedrock-runtime')

response = bedrock_runtime.apply_guardrail(
    guardrailIdentifier='YOUR_GUARDRAIL_ID',
    guardrailVersion='YOUR_GUARDRAIL_VERSION',
    source='USER_INPUT',
    content=[
        {
            'text': {
                'text': 'User input to be evaluated'
            }
        }
    ]
)

# Process the response to check for any violations
if response['action'] == 'GUARDRAIL_INTERVENED':
    print("Input violates guardrails:", response['outputs'])
else:
    # Proceed with invoking the agent
    pass
```

#### Prompt Flows: Orchestrated Conversation Flows

**Amazon Bedrock Flows** (formerly Prompt Flows) orchestrate prompts, agents, knowledge bases and Lambda functions as a graph of nodes. Flows are created with the `bedrock-agent` client (`create_flow`, `prepare_flow`, `create_flow_version`, `create_flow_alias`) and run with `bedrock-agent-runtime` `invoke_flow`. The class below sketches the *shape* of a routing flow as plain data; the real `create_flow` definition uses typed node configurations (`Input`, `Prompt`, `Condition`, `Agent`, `Output`) — see the [Flows API reference](https://docs.aws.amazon.com/bedrock/latest/userguide/flows-create.html).

```python
class PromptFlowManager:
    """Management of Bedrock Prompt Flows"""
    
    def __init__(self, region_name='us-east-1'):
        self.bedrock_client = boto3.client('bedrock-agent', region_name=region_name)
    
    def create_conditional_prompt_flow(self, flow_name: str,
                                      agent_id: str) -> Dict:
        """
        Create a prompt flow with conditional routing logic
        
        Flow structure:
        1. Classify user intent
        2. Route to appropriate specialist agent
        3. Process request
        4. Format and return response
        """
        
        flow_definition = {
            "name": flow_name,
            "nodes": [
                {
                    "id": "input_node",
                    "type": "INPUT",
                    "properties": {
                        "label": "User Input"
                    }
                },
                {
                    "id": "classify_intent",
                    "type": "INVOKE_MODEL",
                    "properties": {
                        "modelId": "us.anthropic.claude-haiku-5-5",
                        "prompt": "Classify the user intent as one of: SUPPORT, SALES, BILLING, TECHNICAL. Respond with ONLY the classification."
                    },
                    "connections": [
                        {"source": "input_node", "target": "classify_intent"}
                    ]
                },
                {
                    "id": "route_decision",
                    "type": "CONDITIONAL",
                    "properties": {
                        "conditions": [
                            {
                                "expression": "intent == 'SUPPORT'",
                                "nextNode": "support_agent"
                            },
                            {
                                "expression": "intent == 'SALES'",
                                "nextNode": "sales_agent"
                            },
                            {
                                "expression": "intent == 'BILLING'",
                                "nextNode": "billing_agent"
                            },
                            {
                                "expression": "intent == 'TECHNICAL'",
                                "nextNode": "technical_agent"
                            }
                        ],
                        "defaultNode": "support_agent"
                    },
                    "connections": [
                        {"source": "classify_intent", "target": "route_decision"}
                    ]
                },
                {
                    "id": "support_agent",
                    "type": "INVOKE_AGENT",
                    "properties": {
                        "agentId": agent_id,
                        "agentName": "Support Agent",
                        "instructions": "You are a support specialist. Help resolve customer issues."
                    }
                },
                {
                    "id": "sales_agent",
                    "type": "INVOKE_AGENT",
                    "properties": {
                        "agentId": agent_id,
                        "agentName": "Sales Agent",
                        "instructions": "You are a sales specialist. Help customers with product information and purchases."
                    }
                },
                {
                    "id": "billing_agent",
                    "type": "INVOKE_AGENT",
                    "properties": {
                        "agentId": agent_id,
                        "agentName": "Billing Agent",
                        "instructions": "You are a billing specialist. Help customers with invoices and payments."
                    }
                },
                {
                    "id": "technical_agent",
                    "type": "INVOKE_AGENT",
                    "properties": {
                        "agentId": agent_id,
                        "agentName": "Technical Agent",
                        "instructions": "You are a technical specialist. Help customers with technical issues."
                    }
                },
                {
                    "id": "output_node",
                    "type": "OUTPUT",
                    "properties": {
                        "label": "Response"
                    }
                }
            ],
            "connections": [
                {"source": "support_agent", "target": "output_node"},
                {"source": "sales_agent", "target": "output_node"},
                {"source": "billing_agent", "target": "output_node"},
                {"source": "technical_agent", "target": "output_node"}
            ]
        }
        
        # Translate this outline into a typed definition for self.bedrock_client.create_flow(...)
        print(f"✓ Defined prompt flow: {flow_name}")
        return flow_definition
```

### Supported Foundation Models

Amazon Bedrock provides access to leading foundation models, each optimised for specific capabilities and use cases.

#### Model Comparison Matrix

The current Claude models on Bedrock are the Claude 5.5 family. Bedrock model IDs have no date suffix; prefix them with a geography (`us.`, `eu.`, `jp.`, `au.`) or `global.` to use a cross-Region inference profile, which most agent workloads should do.

| Model | Provider | Bedrock model ID | Inference profile example | Best for | Relative cost |
|-------|----------|------------------|---------------------------|----------|---------------|
| Claude Opus 5.5 | Anthropic | `anthropic.claude-opus-5-5` | `us.anthropic.claude-opus-5-5` | Hardest reasoning, long-horizon agentic work, complex code | Highest |
| Claude Sonnet 5.5 | Anthropic | `anthropic.claude-sonnet-5-5` | `us.anthropic.claude-sonnet-5-5` | Default for most agents: tool use, coding, analysis | Medium |
| Claude Haiku 5.5 | Anthropic | `anthropic.claude-haiku-5-5` | `us.anthropic.claude-haiku-5-5` | Routing, classification, extraction, high volume | Lowest |
| Amazon Nova family | Amazon | see model cards | — | AWS-native, multimodal, cost-sensitive workloads | Low |
| Llama family | Meta | see model cards | — | Open-weight preference | Low |
| Mistral family | Mistral AI | see model cards | — | Open-weight / European provider preference | Low–medium |

Context windows, maximum output, Region availability and prices change between releases; check the [Bedrock model cards](https://docs.aws.amazon.com/bedrock/latest/userguide/model-cards.html) and [Bedrock pricing](https://aws.amazon.com/bedrock/pricing/) rather than hard-coding them. Older Claude 3.x and 4.x model IDs remain in many tutorials; prefer the current IDs above for new agents.

#### Model Selection Framework

```python
class ModelSelectionFramework:
    """Pick a Bedrock model (inference profile) from coarse task requirements."""

    # Relative 1-10 scores; tune them against your own evaluations
    MODEL_CAPABILITIES = {
        'us.anthropic.claude-opus-5-5': {'reasoning_quality': 10, 'speed': 4, 'cost': 8},
        'us.anthropic.claude-sonnet-5-5': {'reasoning_quality': 9, 'speed': 7, 'cost': 4},
        'us.anthropic.claude-haiku-5-5': {'reasoning_quality': 7, 'speed': 10, 'cost': 1},
    }

    def select_optimal_model(self,
                            required_reasoning: int = 5,  # 1-10
                            latency_requirements: str = 'MEDIUM',  # LOW, MEDIUM, HIGH
                            budget_constraints: str = 'MEDIUM') -> str:  # LOW, MEDIUM, HIGH
        """Return the inference profile ID with the best composite score."""

        latency_weights = {'LOW': 1, 'MEDIUM': 5, 'HIGH': 10}
        cost_weights = {'LOW': 10, 'MEDIUM': 5, 'HIGH': 1}

        scores = {}
        for model, caps in self.MODEL_CAPABILITIES.items():
            if caps['reasoning_quality'] < required_reasoning:
                continue
            speed_score = caps['speed'] * latency_weights[latency_requirements] / 10
            cost_score = (10 - caps['cost']) * cost_weights[budget_constraints] / 10
            scores[model] = caps['reasoning_quality'] + speed_score + cost_score

        recommended = max(scores, key=scores.get)
        print(f"✓ Recommended model: {recommended} (score {scores[recommended]:.2f})")
        return recommended

    def model_fallback_chain(self, primary_model: str) -> list:
        """Fallback order for throttling or capacity errors."""
        return {
            'us.anthropic.claude-opus-5-5': ['us.anthropic.claude-sonnet-5-5'],
            'us.anthropic.claude-sonnet-5-5': ['global.anthropic.claude-sonnet-5-5', 'us.anthropic.claude-haiku-5-5'],
            'us.anthropic.claude-haiku-5-5': ['global.anthropic.claude-haiku-5-5'],
        }.get(primary_model, [])
```

### Code Interpretation

Bedrock Agents can generate and run Python in a managed sandbox for analysis, charts and calculations. It is enabled by adding an action group with the built-in `AMAZON.CodeInterpreter` signature (there is no `enableCodeInterpreter` flag on the agent):

```python
import boto3

agent_client = boto3.client('bedrock-agent')

agent = agent_client.create_agent(
    agentName='CodeInterpretingAgent',
    foundationModel='us.anthropic.claude-sonnet-5-5',
    instruction='You are a data analyst. Write and run Python code to answer questions about data.',
    agentResourceRoleArn='arn:aws:iam::111122223333:role/BedrockAgentRole',
)['agent']

agent_client.create_agent_action_group(
    agentId=agent['agentId'],
    agentVersion='DRAFT',
    actionGroupName='CodeInterpreterAction',
    parentActionGroupSignature='AMAZON.CodeInterpreter',
    actionGroupState='ENABLED',
)

agent_client.prepare_agent(agentId=agent['agentId'])
```

**Example: Using Code Interpretation for Data Analysis**

Files are passed in `sessionState`, and the answer arrives as an event stream:

```python
runtime = boto3.client('bedrock-agent-runtime')

response = runtime.invoke_agent(
    agentId='YOUR_AGENT_ID',
    agentAliasId='YOUR_AGENT_ALIAS_ID',
    sessionId='session-1',
    inputText='Analyse the attached CSV file and summarise the data.',
    sessionState={
        'files': [
            {
                'name': 'sales_data.csv',
                'source': {
                    'sourceType': 'S3',
                    's3Location': {'uri': 's3://your-bucket/sales_data.csv'},
                },
                'useCase': 'CODE_INTERPRETER',
            }
        ]
    },
)

for event in response['completion']:
    if 'chunk' in event:
        print(event['chunk']['bytes'].decode(), end='')
    elif 'files' in event:  # charts or files produced by the generated code
        for f in event['files']['files']:
            open(f['name'], 'wb').write(f['bytes'])
```

### AWS CLI and SDK Setup

#### Installing and Configuring AWS CLI

```bash
# Install AWS CLI v2 (latest)
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Or using package managers
# macOS
brew install awscli

# Ubuntu/Debian
sudo apt-get install awscli

# Verify installation
aws --version

# Configure AWS credentials
aws configure

# Verify Bedrock access
aws bedrock list-foundation-models --region us-east-1
```

#### Python SDK (Boto3) Setup

```bash
# Install the boto3 release this guide was checked against (botocore is pulled in automatically)
pip install "boto3==%%version=boto3%%"

# Verify installation
python -c "import boto3; print(boto3.__version__)"
```

**Comprehensive Boto3 Client Initialization:**

```python
import boto3
from botocore.config import Config

# Create Bedrock client with custom configuration
bedrock_config = Config(
    region_name='us-east-1',
    max_pool_connections=10,
    retries={'max_attempts': 3, 'mode': 'standard'},
    connect_timeout=5,
    read_timeout=60,
    signature_version='v4'
)

bedrock_client = boto3.client('bedrock', config=bedrock_config)                      # models, guardrails
agent_client = boto3.client('bedrock-agent', config=bedrock_config)                  # build agents, KBs, flows
agent_runtime = boto3.client('bedrock-agent-runtime', config=bedrock_config)         # invoke agents, query KBs
bedrock_runtime = boto3.client('bedrock-runtime', config=bedrock_config)             # direct model calls

# Verify connectivity
try:
    response = bedrock_client.list_foundation_models()
    print(f"✓ Connected to Bedrock. Available models: {len(response['modelSummaries'])}")
except Exception as e:
    print(f"✗ Connection failed: {e}")
```

#### JavaScript/Node.js SDK Setup

```bash
# AWS SDK for JavaScript v3 (the v2 "aws-sdk" package is end-of-support)
npm install @aws-sdk/client-bedrock @aws-sdk/client-bedrock-agent-runtime
```

**Node.js SDK Usage:**

```javascript
const { BedrockClient, ListFoundationModelsCommand } = require("@aws-sdk/client-bedrock");
const { BedrockAgentRuntimeClient, InvokeAgentCommand } = require("@aws-sdk/client-bedrock-agent-runtime");

// Create Bedrock client
const bedrockClient = new BedrockClient({ region: "us-east-1" });

// List available models
async function listModels() {
    try {
        const command = new ListFoundationModelsCommand({});
        const response = await bedrockClient.send(command);
        console.log(`✓ Available models: ${response.modelSummaries.length}`);
        return response.modelSummaries;
    } catch (error) {
        console.error("✗ Error listing models:", error);
        throw error;
    }
}

// Invoke agent
async function invokeAgent(agentId, inputText) {
    const runtimeClient = new BedrockAgentRuntimeClient({ region: "us-east-1" });
    
    try {
        const command = new InvokeAgentCommand({
            agentId: agentId,
            agentAliasId: "LFSTG5EXAMPLE", // Production alias
            inputText: inputText,
            sessionId: Date.now().toString()
        });
        
        const response = await runtimeClient.send(command);
        let text = "";
        for await (const event of response.completion) {
            if (event.chunk?.bytes) text += new TextDecoder().decode(event.chunk.bytes);
        }
        console.log("✓ Agent invocation successful");
        return text;
    } catch (error) {
        console.error("✗ Error invoking agent:", error);
        throw error;
    }
}

// Usage
listModels();
```

### Region Availability and Configuration

#### Regional Availability

At the time of checking, Bedrock Agents (Classic) is available in ap-northeast-1, ap-northeast-2, ap-south-1, ap-southeast-1, ap-southeast-2, ca-central-1, eu-central-1, eu-central-2, eu-west-1, eu-west-2, eu-west-3, sa-east-1, us-east-1, us-east-2, us-gov-west-1 and us-west-2. Model availability varies by Region and is wider through cross-Region inference profiles. Check [Supported Regions for Amazon Bedrock Agents](https://docs.aws.amazon.com/bedrock/latest/userguide/agents-supported.html) and the model cards before choosing a Region.

#### Multi-Region Agent Deployment

```python
import boto3
from concurrent.futures import ThreadPoolExecutor, as_completed

class MultiRegionAgentDeployment:
    """Deploy agents across multiple regions for global availability"""
    
    def __init__(self, regions=['us-east-1', 'eu-west-1', 'ap-southeast-1']):
        self.regions = regions
        self.clients = {region: boto3.client('bedrock-agent', region_name=region) for region in regions}
    
    def deploy_agent_to_all_regions(self, agent_config):
        """Deploy identical agent configuration across all regions"""
        
        def deploy_to_region(region):
            try:
                client = self.clients[region]
                agent_id = client.create_agent(**agent_config)['agent']['agentId']
                print(f"✓ Deployed agent to {region}: {agent_id}")
                return {region: agent_id}
            except Exception as e:
                print(f"✗ Deployment failed in {region}: {e}")
                return {region: None}
        
        # Deploy in parallel
        results = {}
        with ThreadPoolExecutor(max_workers=len(self.regions)) as executor:
            futures = {executor.submit(deploy_to_region, region): region for region in self.regions}
            
            for future in as_completed(futures):
                results.update(future.result())
        
        return results
    
    def get_regional_endpoints(self, agent_id_mapping):
        """Get regional endpoints for agents"""
        
        endpoints = {}
        for region, agent_id in agent_id_mapping.items():
            if agent_id:
                endpoints[region] = {
                    'endpoint': f'https://bedrock-agent-runtime.{region}.amazonaws.com',
                    'agent_id': agent_id
                }
        
        return endpoints
```

---

## Multi-Agent Systems (MAS)

**Note:** Multi-agent collaboration in Bedrock Agents (a supervisor agent with collaborator agents, configured with `agentCollaboration` on the supervisor and `associate_agent_collaborator`) became generally available in March 2025. It does not use the A2A protocol; A2A interoperability is provided by AgentCore Runtime (see the [A2A guide](/amazon-bedrock-agents-guide/bedrock_a2a_protocol_guide/)). Supervisor patterns are covered in [Advanced multi-agent patterns](/amazon-bedrock-agents-guide/bedrock_agents_advanced_multi_agent_python/).

### Creating Basic Bedrock Agents via Console

1. Open the Amazon Bedrock console and choose **Agents** → **Create agent** (existing Agents Classic customers only).
2. Give the agent a name, choose a service role (or let the console create one), and pick a model — for example the `us.anthropic.claude-sonnet-5-5` inference profile.
3. Write the agent's instructions.
4. Add action groups (Lambda or return of control, defined by an OpenAPI or function schema) and associate knowledge bases if needed.
5. Choose **Save**, then **Prepare**, and test in the console's test pane against the `DRAFT` version.
6. Create an alias (which snapshots a numbered version) and invoke that alias from your application.

### Creating Basic Bedrock Agents via CloudFormation and Terraform

**CloudFormation Template (YAML):**

```yaml
AWSTemplateFormatVersion: '2010-09-09'
Description: 'CloudFormation template for creating a basic Bedrock Agent'

Parameters:
  AgentName:
    Type: String
    Default: BasicBedrockAgent
    Description: Name of the Bedrock Agent

  FoundationModel:
    Type: String
    Default: us.anthropic.claude-sonnet-5-5
    Description: Foundation model or inference profile to use for the agent

Resources:
  BedrockAgentRole:
    Type: AWS::IAM::Role
    Properties:
      RoleName: !Sub '${AgentName}-role'
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: bedrock.amazonaws.com
            Action: sts:AssumeRole
      Policies:
        - PolicyName: BedrockAgentPolicy
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action:
                  - bedrock:InvokeModel
                  - bedrock:InvokeModelWithResponseStream
                  - bedrock:GetInferenceProfile
                Resource:
                  - !Sub 'arn:aws:bedrock:*::foundation-model/*'
                  - !Sub 'arn:aws:bedrock:*:${AWS::AccountId}:inference-profile/*'

  BedrockAgent:
    Type: AWS::Bedrock::Agent
    Properties:
      AgentName: !Ref AgentName
      Description: Basic Bedrock Agent for demonstration
      FoundationModel: !Ref FoundationModel
      AutoPrepare: true
      Instruction: |
        You are a helpful customer support assistant. Your role is to:
        1. Understand customer inquiries
        2. Provide accurate, helpful information
        3. Escalate complex issues when necessary
        4. Maintain professional and courteous tone
      AgentResourceRoleArn: !GetAtt BedrockAgentRole.Arn
      ActionGroups:
        - ActionGroupName: CustomerSupport
          Description: Basic customer support actions
          ApiSchema:
            S3:
              S3BucketName: !Sub '${AWS::StackName}-schemas'
              S3ObjectKey: customer-support-schema.json
          ActionGroupExecutor:
            Lambda: !GetAtt CustomerSupportLambda.Arn

  CustomerSupportLambda:
    Type: AWS::Lambda::Function
    Properties:
      FunctionName: !Sub '${AgentName}-support-function'
      Runtime: python3.13
      Handler: index.handler
      Role: !GetAtt CustomerSupportLambdaRole.Arn
      Code:
        ZipFile: |
          import json
          
          def handler(event, context):
              """Action group Lambda for an OpenAPI-schema action group."""
              params = {p['name']: p['value'] for p in event.get('parameters', [])}
              api_path = event['apiPath']

              if api_path == '/customers/{customerId}':
                  status, body = 200, {'customerId': params.get('customerId'),
                                       'name': 'John Doe', 'accountStatus': 'ACTIVE'}
              else:
                  status, body = 404, {'error': f'Unknown path {api_path}'}

              # Bedrock Agents expects this envelope, not an API Gateway-style response
              return {
                  'messageVersion': '1.0',
                  'response': {
                      'actionGroup': event['actionGroup'],
                      'apiPath': api_path,
                      'httpMethod': event['httpMethod'],
                      'httpStatusCode': status,
                      'responseBody': {'application/json': {'body': json.dumps(body)}},
                  },
              }

  CustomerSupportLambdaRole:
    Type: AWS::IAM::Role
    Properties:
      AssumeRolePolicyDocument:
        Version: '2012-10-17'
        Statement:
          - Effect: Allow
            Principal:
              Service: lambda.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

  CustomerSupportLambdaPermission:
    Type: AWS::Lambda::Permission
    Properties:
      FunctionName: !Ref CustomerSupportLambda
      Action: lambda:InvokeFunction
      Principal: bedrock.amazonaws.com
      SourceArn: !GetAtt BedrockAgent.AgentArn

  AgentAlias:
    Type: AWS::Bedrock::AgentAlias
    Properties:
      AgentId: !GetAtt BedrockAgent.AgentId
      AgentAliasName: production
      Description: Production alias for the Bedrock Agent

Outputs:
  AgentId:
    Description: ID of the created Bedrock Agent
    Value: !GetAtt BedrockAgent.AgentId
    Export:
      Name: !Sub '${AWS::StackName}-AgentId'

  AgentAliasId:
    Description: Production alias for the agent
    Value: !GetAtt AgentAlias.AgentAliasId
    Export:
      Name: !Sub '${AWS::StackName}-AgentAliasId'

  RoleArn:
    Description: ARN of the agent IAM role
    Value: !GetAtt BedrockAgentRole.Arn
    Export:
      Name: !Sub '${AWS::StackName}-RoleArn'
```

**Terraform Configuration:**

```hcl
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  default = "us-east-1"
}

variable "agent_name" {
  default = "BasicBedrockAgent"
}

variable "foundation_model" {
  default = "us.anthropic.claude-sonnet-5-5"
}

# IAM Role for Bedrock Agent
resource "aws_iam_role" "bedrock_agent_role" {
  name = "${var.agent_name}-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "bedrock.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

# Attach basic Bedrock permissions
resource "aws_iam_role_policy" "bedrock_agent_policy" {
  name = "${var.agent_name}-policy"
  role = aws_iam_role.bedrock_agent_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream",
          "bedrock:GetInferenceProfile"
        ]
        Resource = "*"  # narrow to the specific foundation-model and inference-profile ARNs in production
      }
    ]
  })
}

# Lambda function for agent actions
resource "aws_lambda_function" "customer_support" {
  filename      = "lambda_function.zip"
  function_name = "${var.agent_name}-support-function"
  role          = aws_iam_role.lambda_role.arn
  handler       = "index.handler"
  runtime       = "python3.13"
}

resource "aws_lambda_permission" "allow_bedrock" {
  statement_id  = "AllowBedrockAgent"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.customer_support.function_name
  principal     = "bedrock.amazonaws.com"
  source_arn    = aws_bedrockagent_agent.this.agent_arn
}

resource "aws_iam_role" "lambda_role" {
  name = "${var.agent_name}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

# Lambda basic execution policy
resource "aws_iam_role_policy_attachment" "lambda_basic_execution" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

# Bedrock Agent, action group and alias
resource "aws_bedrockagent_agent" "this" {
  agent_name              = var.agent_name
  agent_resource_role_arn = aws_iam_role.bedrock_agent_role.arn
  foundation_model        = var.foundation_model
  instruction             = "You are a helpful customer support assistant. Answer accurately and escalate complex issues."
}

resource "aws_bedrockagent_agent_action_group" "support" {
  agent_id          = aws_bedrockagent_agent.this.agent_id
  agent_version     = "DRAFT"
  action_group_name = "CustomerSupport"
  action_group_executor {
    lambda = aws_lambda_function.customer_support.arn
  }
  api_schema {
    payload = file("${path.module}/customer-support-schema.json")
  }
}

resource "aws_bedrockagent_agent_alias" "production" {
  agent_id         = aws_bedrockagent_agent.this.agent_id
  agent_alias_name = "production"
  depends_on       = [aws_bedrockagent_agent_action_group.support]
}

output "agent_role_arn" {
  value = aws_iam_role.bedrock_agent_role.arn
}

output "lambda_function_arn" {
  value = aws_lambda_function.customer_support.arn
}
```

---

