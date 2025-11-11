# Amazon Bedrock Agents: Comprehensive Technical Guide

**Framework Name:** bedrock_agents

A definitive, end-to-end technical reference covering Amazon Bedrock Agents from foundational concepts through enterprise-scale production deployment. This guide provides exhaustive coverage of architecture patterns, implementation strategies, AWS integration techniques, cost optimisation, security best practices, and advanced agentic patterns.

---

## Table of Contents

1. [Core Fundamentals](#core-fundamentals)
2. [Simple Agents](#simple-agents)
3. [Multi-Agent Systems (MAS)](#multi-agent-systems)
4. [AgentCore Services](#agentcore-services)
5. [Action Groups](#action-groups)
6. [Knowledge Bases](#knowledge-bases)
7. [Tools Integration](#tools-integration)
8. [Structured Output](#structured-output)
9. [Model Context Protocol (MCP)](#model-context-protocol)
10. [Agentic Patterns](#agentic-patterns)
11. [Guardrails](#guardrails)
12. [Prompt Flows](#prompt-flows)
13. [Memory Systems](#memory-systems)
14. [Context Engineering](#context-engineering)
15. [Multi-Model Support](#multi-model-support)
16. [AWS Integrations](#aws-integrations)
17. [Supervisor Architecture](#supervisor-architecture)
18. [Advanced Topics](#advanced-topics)

---

## Core Fundamentals

### AWS Bedrock Setup and IAM Permissions

Amazon Bedrock Agents require comprehensive AWS environment setup with granular Identity and Access Management (IAM) configurations to ensure secure, scalable agent operations.

#### Setting Up AWS Account and Bedrock Access

**Prerequisites:**
- Active AWS Account with administrative or appropriate delegated access
- Region selection supporting Amazon Bedrock (us-east-1, us-west-2, eu-west-1, ap-southeast-1)
- AWS CLI v2 installed and configured
- Python 3.8+ or Node.js 14+ for SDK operations

**AWS Bedrock Console Access:**

```bash
# Verify Bedrock service availability in your region
aws bedrock list-foundation-models --region us-east-1

# Check account's foundation model access
aws bedrock get-foundation-model-availability-in-region --model-identifier anthropic.claude-3-sonnet-20240229-v1:0
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
        "bedrock:InvokeAgent",
        "bedrock:CreateAgentAlias",
        "bedrock:UpdateAgentAlias",
        "bedrock:GetAgentAlias",
        "bedrock:DeleteAgent",
        "bedrock:CreateActionGroup",
        "bedrock:UpdateActionGroup",
        "bedrock:GetActionGroup",
        "bedrock:DeleteActionGroup",
        "bedrock:CreateKnowledgeBase",
        "bedrock:UpdateKnowledgeBase",
        "bedrock:GetKnowledgeBase",
        "bedrock:DeleteKnowledgeBase",
        "bedrock:CreateDataSource",
        "bedrock:UpdateDataSource",
        "bedrock:GetDataSource",
        "bedrock:DeleteDataSource",
        "bedrock:Invoke"
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
        self.bedrock_client = boto3.client('bedrock', region_name=region_name)
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
                    "Sid": "BedrockCorePermissions",
                    "Effect": "Allow",
                    "Action": [
                        "bedrock:CreateAgent",
                        "bedrock:UpdateAgent",
                        "bedrock:GetAgent",
                        "bedrock:InvokeAgent",
                        "bedrock:ListFoundationModels",
                        "bedrock:GetFoundationModel"
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
| Foundation Model Invocation | Per 1K input/output tokens | $0.003-$0.15 | Varies by model (Claude, Llama, Titan) |
| Action Group Invocation | Per invocation | No additional charge | Charged via underlying service (Lambda, API) |
| Knowledge Base Query | Per query | $0.10 per retrieve | OpenSearch Serverless pricing applies |
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
    
    def __init__(self, primary_region, fallback_region):
        self.primary_client = boto3.client('bedrock', region_name=primary_region)
        self.fallback_client = boto3.client('bedrock', region_name=fallback_region)
        self.primary_region = primary_region
        self.fallback_region = fallback_region
    
    def invoke_with_failover(self, agent_id, input_text, max_retries=3):
        """Invoke agent with automatic failover to secondary region"""
        
        retries = 0
        while retries < max_retries:
            try:
                # Try primary region
                response = self.primary_client.invoke_agent(
                    agentId=agent_id,
                    inputText=input_text,
                    enableTrace=True
                )
                print(f"✓ Successfully invoked agent in {self.primary_region}")
                return response
            
            except ClientError as e:
                if e.response['Error']['Code'] in ['ServiceUnavailable', 'ThrottlingException']:
                    print(f"⚠ Primary region unavailable, attempting fallback...")
                    try:
                        # Try fallback region
                        response = self.fallback_client.invoke_agent(
                            agentId=agent_id,
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
ha_manager = HighAvailabilityBedrockManager('us-east-1', 'us-west-2')
response = ha_manager.invoke_with_failover('agent-123', 'What is the status of my account?')
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
        self.bedrock_client = boto3.client('bedrock', region_name=region_name)
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
            response = self.bedrock_client.create_action_group(
                agentId=self.agent_id,
                actionGroupName=action_group_name,
                actionGroupDescription=description,
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
    
    def create_api_action_group(self, action_group_name: str,
                                api_endpoint: str,
                                description: str,
                                api_schema: Dict,
                                auth_type='API_KEY',
                                auth_secret_arn: Optional[str] = None) -> Dict:
        """
        Create action group that invokes external APIs
        
        Args:
            action_group_name: Name of the action group
            api_endpoint: Base URL of the API
            description: Description
            api_schema: OpenAPI specification
            auth_type: Authentication type (API_KEY, OAuth2, Basic)
            auth_secret_arn: ARN of Secrets Manager secret with credentials
        
        Returns:
            Created action group response
        """
        
        executor_config = {
            'customHttpExecutor': {
                'endpoint': api_endpoint
            }
        }
        
        if auth_type == 'API_KEY' and auth_secret_arn:
            executor_config['customHttpExecutor']['authType'] = 'API_KEY'
            executor_config['customHttpExecutor']['authSecretArn'] = auth_secret_arn
        
        try:
            response = self.bedrock_client.create_action_group(
                agentId=self.agent_id,
                actionGroupName=action_group_name,
                actionGroupDescription=description,
                actionGroupExecutor=executor_config,
                apiSchema={
                    'payload': json.dumps(api_schema)
                }
            )
            print(f"✓ Created API action group: {action_group_name}")
            return response
        except Exception as e:
            print(f"✗ Error creating API action group: {e}")
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
        self.bedrock_client = boto3.client('bedrock', region_name=region_name)
        self.opensearch_client = boto3.client('opensearchserverless', region_name=region_name)
        self.s3_client = boto3.client('s3', region_name=region_name)
    
    def create_knowledge_base(self, kb_name: str,
                             description: str,
                             s3_bucket_arn: str,
                             opensearch_collection_arn: str,
                             role_arn: str,
                             embedding_model_id: str = 'amazon.titan-embed-text-v1') -> Dict:
        """
        Create a comprehensive knowledge base with OpenSearch backend
        
        Args:
            kb_name: Name of the knowledge base
            description: Description
            s3_bucket_arn: ARN of S3 bucket for source documents
            opensearch_collection_arn: ARN of OpenSearch Serverless collection
            role_arn: IAM role ARN with necessary permissions
            embedding_model_id: Model ID for text embeddings
        
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
                        'embeddingModel': {
                            'provider': 'BEDROCK',
                            'bedrockEmbeddingModelConfiguration': {
                                'modelId': embedding_model_id
                            }
                        }
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
                          s3_location_arn: str,
                          chunking_strategy: str = 'FIXED_SIZE',
                          chunk_size: int = 1000,
                          chunk_overlap: int = 100) -> Dict:
        """
        Create a data source for ingesting documents into knowledge base
        
        Args:
            knowledge_base_id: ID of the knowledge base
            data_source_name: Name of the data source
            s3_location_arn: S3 location containing documents (s3://bucket/prefix)
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
                        'bucketArn': s3_location_arn.split('s3://')[1].split('/')[0],
                        'inclusionPrefixes': [s3_location_arn.split('s3://')[1].split('/')[1:]]
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

#### Prompt Flows: Orchestrated Conversation Flows

**Prompt Flows** enable visual, no-code orchestration of complex agent interactions with conditional logic and state management.

```python
class PromptFlowManager:
    """Management of Bedrock Prompt Flows"""
    
    def __init__(self, region_name='us-east-1'):
        self.bedrock_client = boto3.client('bedrock', region_name=region_name)
    
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
                        "modelId": "anthropic.claude-3-sonnet",
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
        
        # Store flow definition (implementation depends on Bedrock Prompt Flows API)
        print(f"✓ Defined prompt flow: {flow_name}")
        return flow_definition
```

### Supported Foundation Models

Amazon Bedrock provides access to leading foundation models, each optimised for specific capabilities and use cases.

#### Model Comparison Matrix

| Model | Provider | Input Tokens | Output Tokens | Use Cases | Strengths | Cost |
|-------|----------|--------------|---------------|-----------|-----------|------|
| Claude 3 Opus | Anthropic | 200K | 4K | Complex reasoning, analysis | Superior reasoning, instruction following | Higher |
| Claude 3 Sonnet | Anthropic | 200K | 4K | Balanced tasks, general purpose | Fast, cost-effective, high quality | Medium |
| Claude 3 Haiku | Anthropic | 200K | 4K | Quick responses, high volume | Fastest, most cost-effective | Lowest |
| Llama 2 70B | Meta | 4K | 4K | General purpose, code | Open source, flexible | Low |
| Llama 2 13B | Meta | 4K | 4K | Lightweight applications | Efficient for edge deployment | Very Low |
| Titan Text Large | Amazon | 8K | 8K | Document summarisation | Native AWS integration | Low |
| Titan Multimodal | Amazon | N/A | N/A | Image + text processing | Multimodal capabilities | Medium |
| Mistral Large | Mistral AI | 32K | 8K | Long context tasks | Excellent long-range reasoning | Medium |

#### Model Selection Framework

```python
class ModelSelectionFramework:
    """Framework for optimal model selection based on task requirements"""
    
    MODEL_CAPABILITIES = {
        'claude-3-opus': {
            'reasoning_quality': 10,
            'speed': 3,
            'cost': 3,
            'context_window': 200000,
            'best_for': ['complex analysis', 'code generation', 'multi-step reasoning']
        },
        'claude-3-sonnet': {
            'reasoning_quality': 9,
            'speed': 7,
            'cost': 2,
            'context_window': 200000,
            'best_for': ['balanced tasks', 'general conversation', 'document analysis']
        },
        'claude-3-haiku': {
            'reasoning_quality': 7,
            'speed': 10,
            'cost': 1,
            'context_window': 200000,
            'best_for': ['high-volume tasks', 'quick responses', 'cost-sensitive applications']
        },
        'llama-2-70b': {
            'reasoning_quality': 7,
            'speed': 5,
            'cost': 1,
            'context_window': 4096,
            'best_for': ['open source preference', 'custom fine-tuning', 'on-premises deployment']
        },
        'titan-text-large': {
            'reasoning_quality': 6,
            'speed': 8,
            'cost': 1,
            'context_window': 8000,
            'best_for': ['AWS-native applications', 'summarisation', 'entity extraction']
        }
    }
    
    def select_optimal_model(self, 
                            required_reasoning: int = 5,  # 1-10
                            latency_requirements: str = 'MEDIUM',  # LOW, MEDIUM, HIGH
                            budget_constraints: str = 'MEDIUM',  # LOW, MEDIUM, HIGH
                            context_needed: int = 8000) -> str:
        """
        Select optimal model based on task requirements
        
        Args:
            required_reasoning: Quality of reasoning needed (1-10)
            latency_requirements: Speed requirements (LOW, MEDIUM, HIGH)
            budget_constraints: Budget constraints (LOW, MEDIUM, HIGH)
            context_needed: Required context window size
        
        Returns:
            Recommended model identifier
        """
        
        latency_weights = {'LOW': 1, 'MEDIUM': 5, 'HIGH': 10}
        cost_weights = {'LOW': 10, 'MEDIUM': 5, 'HIGH': 1}
        
        scores = {}
        
        for model, capabilities in self.MODEL_CAPABILITIES.items():
            # Calculate composite score
            reasoning_score = capabilities['reasoning_quality'] * (required_reasoning / 10)
            speed_score = capabilities['speed'] * latency_weights[latency_requirements] / 10
            cost_score = (10 - capabilities['cost']) * cost_weights[budget_constraints] / 10
            
            # Context window compatibility
            context_score = 1 if capabilities['context_window'] >= context_needed else -10
            
            total_score = reasoning_score + speed_score + cost_score + context_score
            scores[model] = total_score
        
        recommended_model = max(scores, key=scores.get)
        print(f"✓ Recommended model: {recommended_model} (Score: {scores[recommended_model]:.2f})")
        
        return recommended_model
    
    def model_fallback_chain(self, primary_model: str) -> List[str]:
        """
        Define fallback model chain for robustness
        
        Returns: List of models in fallback order
        """
        fallback_chains = {
            'claude-3-opus': ['claude-3-sonnet', 'claude-3-haiku', 'titan-text-large'],
            'claude-3-sonnet': ['claude-3-haiku', 'claude-3-opus', 'mistral-large'],
            'llama-2-70b': ['llama-2-13b', 'titan-text-large', 'claude-3-haiku'],
            'titan-text-large': ['claude-3-haiku', 'mistral-7b', 'llama-2-13b']
        }
        
        return fallback_chains.get(primary_model, [])
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
# Install boto3 latest version
pip install --upgrade boto3

# Install Bedrock-specific dependencies
pip install boto3 botocore --upgrade

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

bedrock_client = boto3.client('bedrock', config=bedrock_config)

# Create Bedrock Runtime client (for agent invocation)
bedrock_runtime = boto3.client('bedrock-runtime', config=bedrock_config)

# Verify connectivity
try:
    response = bedrock_client.list_foundation_models()
    print(f"✓ Connected to Bedrock. Available models: {len(response['modelSummaries'])}")
except Exception as e:
    print(f"✗ Connection failed: {e}")
```

#### JavaScript/Node.js SDK Setup

```bash
# Install AWS SDK for JavaScript v3
npm install @aws-sdk/client-bedrock @aws-sdk/client-bedrock-runtime

# Or install full SDK
npm install aws-sdk
```

**Node.js SDK Usage:**

```javascript
const { BedrockClient, ListFoundationModelsCommand } = require("@aws-sdk/client-bedrock");
const { BedrockRuntimeClient, InvokeAgentCommand } = require("@aws-sdk/client-bedrock-runtime");

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
    const runtimeClient = new BedrockRuntimeClient({ region: "us-east-1" });
    
    try {
        const command = new InvokeAgentCommand({
            agentId: agentId,
            agentAliasId: "LFSTG5EXAMPLE", // Production alias
            inputText: inputText,
            sessionId: Date.now().toString()
        });
        
        const response = await runtimeClient.send(command);
        console.log("✓ Agent invocation successful");
        return response;
    } catch (error) {
        console.error("✗ Error invoking agent:", error);
        throw error;
    }
}

// Usage
listModels();
```

### Region Availability and Configuration

#### Regional Availability Matrix

| Region | Bedrock | Agents | Knowledge Bases | Guardrails | Multi-Agent |
|--------|---------|--------|-----------------|------------|------------|
| us-east-1 | ✓ | ✓ | ✓ | ✓ | ✓ |
| us-west-2 | ✓ | ✓ | ✓ | ✓ | ✓ |
| eu-west-1 | ✓ | ✓ | ✓ | ✓ | ✓ |
| ap-southeast-1 | ✓ | ✓ | ✓ | ✓ | ✓ |
| ap-northeast-1 | ✓ | Limited | Limited | Limited | Limited |

#### Multi-Region Agent Deployment

```python
import boto3
from concurrent.futures import ThreadPoolExecutor, as_completed

class MultiRegionAgentDeployment:
    """Deploy agents across multiple regions for global availability"""
    
    def __init__(self, regions=['us-east-1', 'eu-west-1', 'ap-southeast-1']):
        self.regions = regions
        self.clients = {region: boto3.client('bedrock', region_name=region) for region in regions}
    
    def deploy_agent_to_all_regions(self, agent_config):
        """Deploy identical agent configuration across all regions"""
        
        def deploy_to_region(region):
            try:
                client = self.clients[region]
                response = client.create_agent(**agent_config)
                print(f"✓ Deployed agent to {region}: {response['agentId']}")
                return {region: response['agentId']}
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
                    'endpoint': f'https://bedrock-runtime.{region}.amazonaws.com',
                    'agent_id': agent_id
                }
        
        return endpoints
```

---

## Simple Agents

### Creating Basic Bedrock Agents via Console

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
    Default: anthropic.claude-3-sonnet-20240229-v1:0
    Description: Foundation model to use for the agent

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
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/AmazonBedrockFullAccess
      InlinePolicies:
        - PolicyName: BedrockAgentPolicy
          PolicyDocument:
            Version: '2012-10-17'
            Statement:
              - Effect: Allow
                Action:
                  - logs:CreateLogGroup
                  - logs:CreateLogStream
                  - logs:PutLogEvents
                Resource: !Sub 'arn:aws:logs:${AWS::Region}:${AWS::AccountId}:log-group:/aws/bedrock/*'

  BedrockAgent:
    Type: AWS::Bedrock::Agent
    Properties:
      AgentName: !Ref AgentName
      Description: Basic Bedrock Agent for demonstration
      FoundationModel: !Ref FoundationModel
      AgentInstructions: |
        You are a helpful customer support assistant. Your role is to:
        1. Understand customer inquiries
        2. Provide accurate, helpful information
        3. Escalate complex issues when necessary
        4. Maintain professional and courteous tone
      AgentRoleArn: !GetAtt BedrockAgentRole.Arn
      ActionGroupConfigs:
        - ActionGroupName: CustomerSupport
          ActionGroupDescription: Basic customer support actions
          ApiSchemaS3Location:
            S3BucketName: !Sub '${AWS::StackName}-schemas'
            S3ObjectKey: customer-support-schema.json
          ActionGroupExecutor:
            LambdaArn: !GetAtt CustomerSupportLambda.Arn

  CustomerSupportLambda:
    Type: AWS::Lambda::Function
    Properties:
      FunctionName: !Sub '${AgentName}-support-function'
      Runtime: python3.11
      Handler: index.handler
      Code:
        ZipFile: |
          import json
          
          def handler(event, context):
              """Lambda function for customer support actions"""
              
              action_type = event.get('actionGroup')
              operation = event.get('function')
              parameters = event.get('parameters', {})
              
              # Process customer support requests
              if operation == 'GetCustomerInfo':
                  customer_id = parameters.get('customerId')
                  return {
                      'statusCode': 200,
                      'body': json.dumps({
                          'customerId': customer_id,
                          'name': 'John Doe',
                          'email': 'john@example.com',
                          'accountStatus': 'ACTIVE'
                      })
                  }
              
              elif operation == 'UpdateTicket':
                  ticket_id = parameters.get('ticketId')
                  status = parameters.get('status')
                  return {
                      'statusCode': 200,
                      'body': json.dumps({
                          'ticketId': ticket_id,
                          'status': status,
                          'updatedAt': '2024-01-10T10:00:00Z'
                      })
                  }
              
              return {'statusCode': 400, 'body': json.dumps({'error': 'Unknown operation'})}

  AgentAlias:
    Type: AWS::Bedrock::AgentAlias
    Properties:
      AgentId: !Ref BedrockAgent
      AgentAliasName: production
      Description: Production alias for the Bedrock Agent

Outputs:
  AgentId:
    Description: ID of the created Bedrock Agent
    Value: !Ref BedrockAgent
    Export:
      Name: !Sub '${AWS::StackName}-AgentId'

  AgentAliasId:
    Description: Production alias for the agent
    Value: !Ref AgentAlias
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
      version = "~> 5.0"
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
  default = "anthropic.claude-3-sonnet-20240229-v1:0"
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
          "bedrock:*",
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
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
  runtime       = "python3.11"
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

# Bedrock Agent (Note: Direct Bedrock Agent creation via Terraform may require custom resources)
# For now, use AWS CLI or boto3 for agent creation after infrastructure is provisioned

output "agent_role_arn" {
  value = aws_iam_role.bedrock_agent_role.arn
}

output "lambda_function_arn" {
  value = aws_lambda_function.customer_support.arn
}
```

---

Due to the extremely verbose nature of this request and token limitations, I'll create the remaining files with comprehensive content. Let me create all remaining guides now:

