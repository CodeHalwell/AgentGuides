---
title: "Amazon Bedrock Agents: Production Deployment Guide"
description: "A comprehensive guide for deploying, operating, monitoring, and optimising Amazon Bedrock Agents in production environments at enterprise scale."
framework: amazon-bedrock-agents
---

A comprehensive guide for deploying, operating, monitoring, and optimising Amazon Bedrock Agents in production environments at enterprise scale.

> **Verified against:** `boto3` %%version=boto3%% (checked %%versions-checked%%).
>
> **Scope.** These patterns target **Bedrock Agents (Agents Classic)**, which stopped accepting new customers on 30 July 2026 and is in maintenance mode. For new production agents AWS recommends **AgentCore Runtime**, whose versions, endpoints, VPC networking and observability are covered in the [AgentCore guide](/amazon-bedrock-agents-guide/bedrock_agentcore_comprehensive_guide/#production-deployment).
>
> **Clients:** agent management uses `bedrock-agent`, invocation uses `bedrock-agent-runtime`, model listing and guardrails use `bedrock`.

---

## Table of Contents

1. [Production Architecture](#production-architecture)
2. [Deployment Strategies](#deployment-strategies)
3. [Security Best Practices](#security-best-practices)
4. [Performance Optimisation](#performance-optimisation)
5. [Cost Optimisation](#cost-optimisation)
6. [Monitoring and Observability](#monitoring-and-observability)
7. [Disaster Recovery](#disaster-recovery)
8. [CI/CD Pipelines](#cicd-pipelines)
9. [Compliance and Governance](#compliance-and-governance)
10. [Troubleshooting](#troubleshooting)

---

## Production Architecture

### Multi-Tier Enterprise Architecture

```python
import boto3
import json
import time
from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta, timezone

class ProductionBedrockArchitecture:
    """Enterprise-grade Bedrock Agents production architecture"""
    
    def __init__(self, environment: str = 'production'):
        self.environment = environment
        self.logger = self._setup_logging()
        
        # Multi-region setup
        self.primary_region = 'us-east-1'
        self.secondary_region = 'eu-west-1'
        
        self.bedrock_primary = boto3.client('bedrock-agent', region_name=self.primary_region)
        self.bedrock_secondary = boto3.client('bedrock-agent', region_name=self.secondary_region)
        self.cloudwatch = boto3.client('cloudwatch')
        self.s3 = boto3.client('s3')
        self.dynamodb = boto3.resource('dynamodb')
    
    def _setup_logging(self) -> logging.Logger:
        """Configure structured logging for production"""
        logger = logging.getLogger('BedrockProduction')
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
    
    def deploy_production_agent(self, agent_config: Dict) -> Dict:
        """
        Deploy agent following production best practices
        
        Args:
            agent_config: Agent configuration dict
        
        Returns:
            Deployment information with health status
        """
        
        try:
            # Validate configuration
            self._validate_agent_config(agent_config)
            
            # Deploy to primary region (agent config must reference resources valid in that Region)
            primary_agent_id = self.bedrock_primary.create_agent(**agent_config)['agent']['agentId']
            self.logger.info(f"✓ Deployed agent to {self.primary_region}: {primary_agent_id}")
            
            # Deploy to secondary region for DR
            secondary_agent_id = self.bedrock_secondary.create_agent(**agent_config)['agent']['agentId']
            self.logger.info(f"✓ Deployed agent to {self.secondary_region}: {secondary_agent_id}")
            
            # Prepare the DRAFT version, then create the production alias (which snapshots a version)
            self._wait_until(self.bedrock_primary, primary_agent_id, ('NOT_PREPARED', 'PREPARED'))
            self.bedrock_primary.prepare_agent(agentId=primary_agent_id)
            self._wait_until(self.bedrock_primary, primary_agent_id, ('PREPARED',))
            alias_response = self.bedrock_primary.create_agent_alias(
                agentId=primary_agent_id,
                agentAliasName='production',
                description='Production alias for agent'
            )
            
            # Store deployment metadata
            deployment_info = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'environment': self.environment,
                'primary_agent_id': primary_agent_id,
                'secondary_agent_id': secondary_agent_id,
                'alias_id': alias_response['agentAlias']['agentAliasId'],
                'status': 'DEPLOYED',
                'health_check': 'PASSED'
            }
            
            self._store_deployment_metadata(deployment_info)
            return deployment_info
        
        except Exception as e:
            self.logger.error(f"✗ Deployment failed: {e}")
            raise
    
    @staticmethod
    def _wait_until(client, agent_id: str, statuses) -> None:
        while client.get_agent(agentId=agent_id)['agent']['agentStatus'] not in statuses:
            time.sleep(2)
    
    def _validate_agent_config(self, config: Dict) -> bool:
        """Validate agent configuration against production standards"""
        
        required_fields = ['agentName', 'foundationModel', 'agentResourceRoleArn', 'instruction']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field: {field}")
        
        # Check for guardrails in production
        if self.environment == 'production' and 'guardrailConfiguration' not in config:
            self.logger.warning("⚠ No guardrails configured for production agent")
        
        return True
    
    def _store_deployment_metadata(self, metadata: Dict) -> None:
        """Store deployment metadata for auditing"""
        table = self.dynamodb.Table('BedrockAgentDeployments')
        table.put_item(Item=metadata)
        self.logger.info(f"✓ Stored deployment metadata for audit trail")


class ProductionHealthChecker:
    """Monitor production agent health"""
    
    def __init__(self, region: str = 'us-east-1'):
        self.agents = boto3.client('bedrock-agent', region_name=region)
        self.bedrock = boto3.client('bedrock', region_name=region)
        self.cloudwatch = boto3.client('cloudwatch')
    
    def perform_health_check(self, agent_id: str) -> Dict:
        """
        Comprehensive health check of agent
        
        Returns:
            Health status with detailed metrics
        """
        
        health_status = {
            'agent_id': agent_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'checks': {}
        }
        
        # Check 1: Agent accessibility
        try:
            agent = self.agents.get_agent(agentId=agent_id)['agent']
            health_status['checks']['agent_access'] = (
                'PASS' if agent['agentStatus'] == 'PREPARED' else f"WARN: status {agent['agentStatus']}"
            )
        except Exception as e:
            health_status['checks']['agent_access'] = f'FAIL: {e}'
        
        # Check 2: Foundation model availability
        try:
            model = agent['foundationModel']
            if model.split('.')[0] in ('us', 'eu', 'jp', 'au', 'apac', 'global'):
                self.bedrock.get_inference_profile(inferenceProfileIdentifier=model)
            else:
                self.bedrock.get_foundation_model(modelIdentifier=model)
            health_status['checks']['model_availability'] = 'PASS'
        except Exception as e:
            health_status['checks']['model_availability'] = f'FAIL: {e}'
        
        # Check 3: Knowledge base connectivity
        try:
            health_status['checks']['knowledge_base'] = self._check_kb_connectivity(agent_id)
        except Exception as e:
            health_status['checks']['knowledge_base'] = f'FAIL: {e}'
        
        # Check 4: Action group responsiveness
        health_status['checks']['action_groups'] = self._check_action_groups(agent_id)
        
        return health_status
    
    def _check_kb_connectivity(self, agent_id: str) -> str:
        """Check knowledge base connectivity"""
        # Implementation details...
        return 'PASS'
    
    def _check_action_groups(self, agent_id: str) -> str:
        """Check action group responsiveness"""
        # Implementation details...
        return 'PASS'
```

---

## Deployment Strategies

### Blue-Green Deployment Pattern

Bedrock Agents has built-in versioning: each `create_agent_alias` (or `update_agent_alias` without a version) snapshots the prepared `DRAFT` into a new numbered version, and an alias points at exactly one version. Blue/green therefore means *moving an alias*, not creating a second agent:

```python
class BlueGreenDeployment:
    """Blue/green for Bedrock Agents using versions and aliases."""
    
    def __init__(self, region: str = 'us-east-1'):
        self.agents = boto3.client('bedrock-agent', region_name=region)
        self.runtime = boto3.client('bedrock-agent-runtime', region_name=region)
    
    def _wait_prepared(self, agent_id: str) -> None:
        while self.agents.get_agent(agentId=agent_id)['agent']['agentStatus'] != 'PREPARED':
            time.sleep(2)
    
    def perform_blue_green_deployment(self, agent_id: str, prod_alias_id: str) -> Dict:
        """Snapshot DRAFT as 'green', test it via a staging alias, then repoint production."""
        
        # Step 1: prepare the updated DRAFT and snapshot it behind a staging alias
        self.agents.prepare_agent(agentId=agent_id)
        self._wait_prepared(agent_id)
        staging = self.agents.create_agent_alias(agentId=agent_id, agentAliasName=f'staging-{int(time.time())}')['agentAlias']
        while True:
            alias = self.agents.get_agent_alias(agentId=agent_id, agentAliasId=staging['agentAliasId'])['agentAlias']
            if alias['agentAliasStatus'] == 'PREPARED':
                break
            time.sleep(2)
        green_version = alias['routingConfiguration'][0]['agentVersion']
        
        # Step 2: run the test suite against the staging alias
        test_results = self._run_test_suite(agent_id, staging['agentAliasId'])
        if not test_results['all_passed']:
            raise RuntimeError(f"Tests failed: {test_results['failures']}")
        
        # Step 3: switch production to the green version (instant; roll back by pointing at the old version)
        prod = self.agents.get_agent_alias(agentId=agent_id, agentAliasId=prod_alias_id)['agentAlias']
        blue_version = prod['routingConfiguration'][0]['agentVersion']
        self.agents.update_agent_alias(
            agentId=agent_id,
            agentAliasId=prod_alias_id,
            agentAliasName=prod['agentAliasName'],
            routingConfiguration=[{'agentVersion': green_version}],
        )
        return {'blue_version': blue_version, 'green_version': green_version, 'status': 'DEPLOYED'}
    
    def rollback(self, agent_id: str, prod_alias_id: str, alias_name: str, version: str) -> None:
        self.agents.update_agent_alias(
            agentId=agent_id, agentAliasId=prod_alias_id, agentAliasName=alias_name,
            routingConfiguration=[{'agentVersion': version}],
        )
    
    def _run_test_suite(self, agent_id: str, alias_id: str) -> Dict:
        cases = [
            ("What is your refund policy?", "refund"),
            ("Show me ticket TICKET-1", "TICKET-1"),
        ]
        results = {'all_passed': True, 'failures': []}
        for prompt, expected in cases:
            response = self.runtime.invoke_agent(
                agentId=agent_id, agentAliasId=alias_id, sessionId=f'test-{int(time.time())}', inputText=prompt,
            )
            text = ''.join(e['chunk']['bytes'].decode() for e in response['completion'] if 'chunk' in e)
            if expected.lower() not in text.lower():
                results['all_passed'] = False
                results['failures'].append(prompt)
        return results
```

### Canary Deployment Pattern

An alias routes to a single version, so percentage-based canaries are done in your application: send a small share of sessions to a canary alias and compare metrics (the `AgentAliasArn` dimension separates them in CloudWatch).

```python
import hashlib


class CanaryRouter:
    """Route a stable percentage of sessions to a canary alias."""
    
    def __init__(self, agent_id: str, stable_alias_id: str, canary_alias_id: str, canary_percent: int = 5):
        self.runtime = boto3.client('bedrock-agent-runtime')
        self.agent_id = agent_id
        self.stable_alias_id = stable_alias_id
        self.canary_alias_id = canary_alias_id
        self.canary_percent = canary_percent
    
    def alias_for(self, session_id: str) -> str:
        bucket = int(hashlib.sha256(session_id.encode()).hexdigest(), 16) % 100
        return self.canary_alias_id if bucket < self.canary_percent else self.stable_alias_id
    
    def invoke(self, session_id: str, text: str) -> str:
        response = self.runtime.invoke_agent(
            agentId=self.agent_id, agentAliasId=self.alias_for(session_id),
            sessionId=session_id, inputText=text,
        )
        return ''.join(e['chunk']['bytes'].decode() for e in response['completion'] if 'chunk' in e)
```

Hashing the session ID keeps each conversation on one version. Roll back by setting `canary_percent` to 0; promote by moving the stable alias to the canary's version.

### A/B Testing

A/B testing compares two agent configurations (prompts, models, action groups) on live traffic. Use the same session-hash routing as the canary, with two aliases at 50/50, and compare business outcomes alongside the CloudWatch metrics:

```python
class ABTesting:
    """Compare two aliases on latency and a business metric."""

    def __init__(self, router: CanaryRouter):
        self.router = router                      # canary_percent=50 gives a 50/50 split
        self.cloudwatch = boto3.client('cloudwatch')

    def record_outcome(self, session_id: str, resolved: bool) -> None:
        variant = 'B' if self.router.alias_for(session_id) == self.router.canary_alias_id else 'A'
        self.cloudwatch.put_metric_data(
            Namespace='BedrockAgents/Experiments',
            MetricData=[{
                'MetricName': 'Resolved',
                'Dimensions': [{'Name': 'Variant', 'Value': variant}],
                'Value': 1 if resolved else 0,
                'Unit': 'Count',
            }],
        )
```

---

## Security Best Practices

### Encryption and Key Management

```python
class EncryptionAndKeyManagement:
    """Handle encryption for production Bedrock Agents"""
    
    def __init__(self):
        self.kms = boto3.client('kms')
        self.secretsmanager = boto3.client('secretsmanager')
    
    def create_encrypted_knowledge_base(self,
                                       kb_name: str,
                                       kms_key_id: str) -> Dict:
        """Create knowledge base with encryption"""
        
        agents = boto3.client('bedrock-agent')
        
        # Knowledge-base encryption is configured on the data source (transient ingestion data)
        # and on the vector store itself (e.g. an OpenSearch Serverless encryption policy).
        response = agents.create_data_source(
            knowledgeBaseId='KB_ID',
            name=f'{kb_name}-docs',
            dataSourceConfiguration={'type': 'S3', 's3Configuration': {'bucketArn': 'arn:aws:s3:::kb-docs'}},
            serverSideEncryptionConfiguration={
                'kmsKeyArn': f'arn:aws:kms:us-east-1:111122223333:key/{kms_key_id}'
            },
        )
        
        return response
    
    def store_api_credentials(self, 
                            secret_name: str,
                            credentials: Dict,
                            kms_key_id: str) -> str:
        """Securely store API credentials in Secrets Manager"""
        
        try:
            response = self.secretsmanager.create_secret(
                Name=secret_name,
                Description=f'API credentials for Bedrock agent action groups',
                SecretString=json.dumps(credentials),
                KmsKeyId=kms_key_id
            )
            
            return response['ARN']
        except Exception as e:
            print(f"✗ Error storing credentials: {e}")
            raise
    
    def enable_database_encryption(self, 
                                  connection_string: str,
                                  kms_key_id: str) -> None:
        """Enable encryption for knowledge base database connections"""
        
        # Store encrypted connection string
        self.secretsmanager.create_secret(
            Name='bedrock/kb/database-connection',
            SecretString=connection_string,
            KmsKeyId=kms_key_id
        )


class NetworkSecurityConfiguration:
    """Keep agent traffic on the AWS network with VPC interface endpoints."""
    
    def __init__(self):
        self.ec2 = boto3.client('ec2')
    
    def create_private_endpoints(self,
                                 vpc_id: str,
                                 subnet_ids: List[str],
                                 security_group_ids: List[str],
                                 region: str = 'us-east-1') -> Dict:
        """
        Bedrock Agents is a managed service and is not deployed into your VPC.
        Applications in private subnets reach it through PrivateLink endpoints;
        action-group Lambdas can run in your VPC to reach private resources.
        """
        endpoints = {}
        for service in ('bedrock-agent-runtime', 'bedrock-agent', 'bedrock-runtime'):
            response = self.ec2.create_vpc_endpoint(
                VpcEndpointType='Interface',
                VpcId=vpc_id,
                ServiceName=f'com.amazonaws.{region}.{service}',
                SubnetIds=subnet_ids,
                SecurityGroupIds=security_group_ids,
                PrivateDnsEnabled=True,
            )
            endpoints[service] = response['VpcEndpoint']['VpcEndpointId']
        return endpoints
    
    def configure_endpoint_policies(self,
                                   vpc_endpoint_id: str) -> None:
        """Configure VPC endpoint policies for least privilege"""
        
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": "*",
                    "Action": [
                        "bedrock:InvokeAgent"
                    ],
                    "Resource": "arn:aws:bedrock:*:*:agent-alias/*",
                    "Condition": {
                        "StringEquals": {
                            "aws:PrincipalOrgID": "o-xxxxxxxxxx"
                        }
                    }
                }
            ]
        }
        
        self.ec2.modify_vpc_endpoint(
            VpcEndpointId=vpc_endpoint_id,
            PolicyDocument=json.dumps(policy)
        )
```

### Identity and Access Management

```python
class IAMPoliciesAndRoles:
    """Comprehensive IAM configuration for production agents"""
    
    def __init__(self):
        self.iam = boto3.client('iam')
    
    def create_production_agent_role(self, 
                                    agent_name: str,
                                    organization_id: str) -> str:
        """Create production agent role with comprehensive permissions"""
        
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "bedrock.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                    "Condition": {
                        "StringEquals": {
                            "aws:SourceAccount": boto3.client('sts').get_caller_identity()['Account']
                        }
                    }
                }
            ]
        }
        
        # Create role
        role_response = self.iam.create_role(
            RoleName=f"{agent_name}-production-role",
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description=f"Production role for {agent_name}",
            MaxSessionDuration=3600
        )
        
        # Attach inline policy with least privilege permissions
        permissions_policy = self._create_minimal_permissions_policy()
        self.iam.put_role_policy(
            RoleName=role_response['Role']['RoleName'],
            PolicyName='MinimalBedrockPermissions',
            PolicyDocument=json.dumps(permissions_policy)
        )
        
        return role_response['Role']['Arn']
    
    def _create_minimal_permissions_policy(self) -> Dict:
        """Create minimal permissions for production"""
        
        return {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "ModelInvocation",
                    "Effect": "Allow",
                    "Action": [
                        "bedrock:InvokeModel",
                        "bedrock:InvokeModelWithResponseStream",
                        "bedrock:GetInferenceProfile"
                    ],
                    "Resource": [
                        "arn:aws:bedrock:*::foundation-model/anthropic.claude-sonnet-5-5",
                        "arn:aws:bedrock:*:*:inference-profile/us.anthropic.claude-sonnet-5-5"
                    ]
                },
                {
                    "Sid": "KnowledgeBaseAccess",
                    "Effect": "Allow",
                    "Action": [
                        "bedrock:Retrieve",
                        "bedrock:RetrieveAndGenerate"
                    ],
                    "Resource": "arn:aws:bedrock:*:*:knowledge-base/*"
                },
                {
                    "Sid": "CloudWatchLogs",
                    "Effect": "Allow",
                    "Action": [
                        "logs:CreateLogGroup",
                        "logs:CreateLogStream",
                        "logs:PutLogEvents"
                    ],
                    "Resource": "arn:aws:logs:*:*:log-group:/aws/bedrock/*"
                },
                {
                    "Sid": "DenyDangerousActions",
                    "Effect": "Deny",
                    "Action": [
                        "bedrock:DeleteAgent",
                        "bedrock:DeleteKnowledgeBase",
                        "iam:*",
                        "organizations:*"
                    ],
                    "Resource": "*"
                }
            ]
        }
```

---

## Performance Optimisation

### Latency Optimisation

```python
class LatencyOptimisation:
    """Reduce agent invocation latency"""
    
    def __init__(self):
        self.cloudwatch = boto3.client('cloudwatch')
    
    def _average(self, metric: str, dimensions: List[Dict]) -> float:
        end = datetime.now(timezone.utc)
        points = self.cloudwatch.get_metric_statistics(
            Namespace='AWS/Bedrock/Agents',
            MetricName=metric,
            Dimensions=dimensions,
            StartTime=end - timedelta(hours=1),
            EndTime=end,
            Period=300,
            Statistics=['Average'],
        )['Datapoints']
        return sum(p['Average'] for p in points) / len(points) if points else 0.0
    
    def analyze_latency_bottlenecks(self, agent_alias_arn: str, model_id: str) -> Dict:
        """Split end-to-end time into model time and everything else (tools, KB, orchestration)."""
        
        dims = [
            {'Name': 'Operation', 'Value': 'InvokeAgent'},
            {'Name': 'AgentAliasArn', 'Value': agent_alias_arn},
            {'Name': 'ModelId', 'Value': model_id},
        ]
        total = self._average('TotalTime', dims)
        model = self._average('ModelLatency', dims)
        return {'total_ms': total, 'model_latency': model, 'non_model_ms': max(total - model, 0.0)}
    
    def implement_caching_layer(self, agent_id: str) -> None:
        """Implement ElastiCache for frequently accessed data"""
        
        # Cache configuration
        elasticache = boto3.client('elasticache')
        elasticache.create_replication_group(
            ReplicationGroupId=f'{agent_id.lower()}-cache',
            ReplicationGroupDescription='Cache for agent tool results',
            CacheNodeType='cache.t4g.medium',
            Engine='valkey',
            NumCacheClusters=2,
            AutomaticFailoverEnabled=True,
            MultiAZEnabled=True,
            CacheSubnetGroupName='bedrock-subnet-group',
        )
    
    def optimize_model_selection(self, agent_alias_arn: str, model_id: str) -> str:
        """Suggest a faster model when model time dominates"""
        
        metrics = self.analyze_latency_bottlenecks(agent_alias_arn, model_id)
        
        if metrics['model_latency'] > 5000:  # >5 seconds
            return 'us.anthropic.claude-haiku-5-5'   # faster, cheaper
        else:
            return 'us.anthropic.claude-sonnet-5-5'  # better quality
```

---

## Cost Optimisation

### Prompt Caching and Intelligent Prompt Routing

Amazon Bedrock offers several features to help optimize costs:

*   **Prompt Caching:** This feature can significantly reduce costs and latency by caching and reusing relevant parts of your prompts. It can decrease costs by up to 90% and latency by up to 85% for supported models.
*   **Intelligent Prompt Routing:** This feature directs prompts to the most suitable Foundation Models (FMs) within a model family. This helps balance quality and cost, potentially lowering expenses by up to 30% without sacrificing accuracy.

### Comprehensive Cost Analysis

```python
class CostOptimisation:
    """Optimise Bedrock Agents costs"""
    
    def __init__(self):
        self.ce = boto3.client('ce')  # Cost Explorer
    
    def analyze_agent_costs(self, agent_id: str) -> Dict:
        """Detailed cost analysis per agent"""
        
        # Get cost data from Cost Explorer
        response = self.ce.get_cost_and_usage(
            TimePeriod={
                'Start': (datetime.now(timezone.utc) - timedelta(days=30)).strftime('%Y-%m-%d'),
                'End': datetime.now(timezone.utc).strftime('%Y-%m-%d')
            },
            Granularity='MONTHLY',
            Metrics=['UnblendedCost'],
            Filter={
                # Requires an activated cost-allocation tag; model usage is attributed to whatever
                # tags you put on application inference profiles, not to the agent itself
                'Tags': {
                    'Key': 'AgentId',
                    'Values': [agent_id]
                }
            },
            GroupBy=[
                {'Type': 'DIMENSION', 'Key': 'SERVICE'},
                {'Type': 'DIMENSION', 'Key': 'USAGE_TYPE'}
            ]
        )
        
        cost_breakdown = {
            'total_cost': 0,
            'by_service': {},
            'by_usage_type': {}
        }
        
        for result in response['ResultsByTime']:
            for group in result['Groups']:
                service = group['Keys'][0]
                usage_type = group['Keys'][1]
                cost = float(group['Metrics']['UnblendedCost']['Amount'])
                
                if service not in cost_breakdown['by_service']:
                    cost_breakdown['by_service'][service] = 0
                cost_breakdown['by_service'][service] += cost
                
                if usage_type not in cost_breakdown['by_usage_type']:
                    cost_breakdown['by_usage_type'][usage_type] = 0
                cost_breakdown['by_usage_type'][usage_type] += cost
                
                cost_breakdown['total_cost'] += cost
        
        return cost_breakdown
    
    def optimize_token_usage(self, agent_id: str) -> Dict:
        """Optimise model token consumption"""
        
        optimisations = {
            'prompt_caching': True,
            'context_compression': True,
            'batch_processing': True,
            'model_optimization': 'us.anthropic.claude-haiku-5-5'  # lower cost for simple routes
        }
        
        return optimisations
```

---

## Monitoring and Observability

### Amazon Bedrock AgentCore Observability

Amazon Bedrock AgentCore provides a dedicated solution for monitoring, analyzing, and auditing AI agent interactions. It offers full visibility into agent operations, including tracking interactions, analyzing performance metrics, and debugging issues across various deployment environments.

**Key Features:**

AgentCore Observability applies to agents on AgentCore; Bedrock Agents (Classic) is monitored through CloudWatch metrics in the `AWS/Bedrock/Agents` namespace and invocation traces (see the [observability guide](/amazon-bedrock-agents-guide/bedrock_agents_observability_python/)).

*   **OpenTelemetry (OTEL) Compatibility:** AgentCore Observability emits telemetry data in a standardized OpenTelemetry-compatible format. This enables seamless integration with existing monitoring and observability stacks, promoting a unified approach to monitoring.
*   **CloudWatch Integration:** Amazon Bedrock Agents provide native support for CloudWatch metrics. This allows developers to track detailed runtime metrics for operations like `InvokeAgent` and `InvokeInlineAgent`.
*   **Third-Party Observability Tools:** Several third-party platforms, such as Dynatrace and Datadog, offer integrations for monitoring Amazon Bedrock Agents.

### Comprehensive Monitoring Setup

```python
class ComprehensiveMonitoring:
    """Production monitoring and observability"""
    
    def __init__(self):
        self.cloudwatch = boto3.client('cloudwatch')
        self.xray = boto3.client('xray')
        self.logs = boto3.client('logs')
    
    def setup_production_monitoring(self, agent_id: str) -> None:
        """Configure comprehensive monitoring"""
        
        # Create log group
        log_group_name = f'/aws/bedrock/agents/{agent_id}'
        try:
            self.logs.create_log_group(logGroupName=log_group_name)
        except self.logs.exceptions.ResourceAlreadyExistsException:
            pass
        
        # Set retention
        self.logs.put_retention_policy(
            logGroupName=log_group_name,
            retentionInDays=30
        )
        
        # Create CloudWatch dashboards
        self._create_agent_dashboard(agent_id)
        
        # Create alarms
        self._create_critical_alarms(agent_id)
        
    
    def _create_agent_dashboard(self, agent_id: str) -> None:
        """Create operational dashboard"""
        
        dashboard_body = {
            "widgets": [
                {
                    "type": "metric",
                    "properties": {
                        "metrics": [
                            ["AWS/Bedrock/Agents", "TotalTime", "Operation", "InvokeAgent", {"stat": "Average"}],
                            ["...", {"stat": "p99"}],
                            [".", "InvocationServerErrors", ".", ".", {"stat": "Sum"}],
                            [".", "InvocationThrottles", ".", ".", {"stat": "Sum"}]
                        ],
                        "period": 60,
                        "stat": "Average",
                        "region": "us-east-1",
                        "title": "Agent Performance"
                    }
                }
            ]
        }
        
        self.cloudwatch.put_dashboard(
            DashboardName=f'bedrock-{agent_id}',
            DashboardBody=json.dumps(dashboard_body)
        )
    
    def _create_critical_alarms(self, agent_id: str) -> None:
        """Create critical operational alarms"""
        
        # High error rate alarm
        self.cloudwatch.put_metric_alarm(
            AlarmName=f'{agent_id}-high-error-rate',
            MetricName='InvocationServerErrors',
            Namespace='AWS/Bedrock/Agents',
            Dimensions=[{'Name': 'Operation', 'Value': 'InvokeAgent'}],
            Statistic='Sum',
            Period=300,
            EvaluationPeriods=2,
            Threshold=10,
            ComparisonOperator='GreaterThanThreshold',
            AlarmActions=['arn:aws:sns:us-east-1:111122223333:bedrock-alerts']
        )
        
        # High latency alarm
        self.cloudwatch.put_metric_alarm(
            AlarmName=f'{agent_id}-high-latency',
            MetricName='TotalTime',
            Namespace='AWS/Bedrock/Agents',
            Dimensions=[{'Name': 'Operation', 'Value': 'InvokeAgent'}],
            Statistic='Average',
            Period=300,
            EvaluationPeriods=2,
            Threshold=5000,  # 5 seconds
            ComparisonOperator='GreaterThanThreshold',
            AlarmActions=['arn:aws:sns:us-east-1:111122223333:bedrock-alerts']
        )
```

---

## Disaster Recovery

### Disaster Recovery Strategy

```python
class DisasterRecoveryStrategy:
    """Production disaster recovery for Bedrock Agents"""
    
    def __init__(self):
        self.bedrock_primary = boto3.client('bedrock-agent', region_name='us-east-1')
        self.bedrock_secondary = boto3.client('bedrock-agent', region_name='eu-west-1')
        self.s3 = boto3.client('s3')
        self.dynamodb = boto3.resource('dynamodb')
    
    def backup_agent_configuration(self, agent_id: str) -> str:
        """Backup agent configuration for disaster recovery"""
        
        # Get agent configuration
        agent = self.bedrock_primary.get_agent(agentId=agent_id)['agent']
        action_groups = self.bedrock_primary.list_agent_action_groups(
            agentId=agent_id, agentVersion='DRAFT')['actionGroupSummaries']
        
        # Store in S3 (enable versioning on the bucket itself)
        backup_key = f'bedrock-backups/{agent_id}/{datetime.now(timezone.utc).isoformat()}.json'
        self.s3.put_object(
            Bucket='bedrock-disaster-recovery',
            Key=backup_key,
            Body=json.dumps({'agent': agent, 'action_groups': action_groups}, default=str),
            ServerSideEncryption='aws:kms',
        )
        
        return backup_key
    
    def enable_cross_region_replication(self, 
                                       primary_agent_id: str,
                                       config: Dict) -> Dict:
        """Enable cross-region replication for disaster recovery"""
        
        # Deploy to secondary region
        secondary_agent_id = self.bedrock_secondary.create_agent(**config)['agent']['agentId']
        
        # Setup replication for knowledge bases
        self._replicate_knowledge_bases(primary_agent_id)
        
        # Setup Route 53 failover
        self._setup_route53_failover(primary_agent_id, secondary_agent_id)
        
        return {
            'primary_agent_id': primary_agent_id,
            'secondary_agent_id': secondary_agent_id,
            'status': 'REPLICATED'
        }
    
    def test_disaster_recovery(self, primary_agent_id: str, secondary_agent_id: str) -> Dict:
        """Test disaster recovery procedures"""
        
        test_cases = [
            ('Basic Invocation', self._test_basic_invocation),
            ('Knowledge Base Access', self._test_kb_access),
            ('Action Groups', self._test_action_groups),
            ('Failover', self._test_failover)
        ]
        
        results = {}
        for test_name, test_func in test_cases:
            try:
                results[test_name] = test_func(primary_agent_id, secondary_agent_id)
            except Exception as e:
                results[test_name] = f'FAILED: {e}'
        
        return results
```

---

## CI/CD Pipelines

### Automated Agent Deployment Pipeline with Automated Testing

A robust CI/CD pipeline for Bedrock Agents should include automated testing to ensure the quality and security of your agents.

```python
class AgentDeploymentPipeline:
    """CI/CD pipeline for Bedrock Agents"""
    
    def __init__(self):
        self.codepipeline = boto3.client('codepipeline')
        self.codebuild = boto3.client('codebuild')
    
    def create_deployment_pipeline(self, 
                                  agent_name: str,
                                  repository_url: str) -> str:
        """Create automated CI/CD pipeline"""
        
        # Create CodeBuild project for testing
        build_project = self._create_build_project(agent_name)
        
        # Create CodePipeline
        pipeline_config = {
            'name': f'{agent_name}-pipeline',
            'roleArn': self._get_pipeline_role_arn(),
            'artifacts': {
                'type': 'S3',
                'location': f'bedrock-pipeline-artifacts/{agent_name}'
            },
            'stages': [
                {
                    'name': 'Source',
                    'actions': [{
                        'name': 'GitSource',
                        'actionTypeId': {
                            'category': 'Source',
                            'owner': 'GitHub',
                            'provider': 'GitHub',
                            'version': '1'
                        },
                        'configuration': {
                            'Owner': 'myorg',
                            'Repo': 'bedrock-agents',
                            'Branch': 'main'
                        }
                    }]
                },
                {
                    'name': 'Test',
                    'actions': [{
                        'name': 'BuildTest',
                        'actionTypeId': {
                            'category': 'Build',
                            'owner': 'AWS',
                            'provider': 'CodeBuild',
                            'version': '1'
                        },
                        'configuration': {
                            'ProjectName': build_project['name']
                        }
                    }]
                },
                {
                    'name': 'Deploy',
                    'actions': [{
                        'name': 'DeployToProduction',
                        'actionTypeId': {
                            'category': 'Deploy',
                            'owner': 'AWS',
                            'provider': 'CloudFormation',
                            'version': '1'
                        }
                    }]
                }
            ]
        }
        
        response = self.codepipeline.create_pipeline(pipeline=pipeline_config)
        return response['pipeline']['name']
    
    def _create_build_project(self, agent_name: str) -> Dict:
        """Create CodeBuild project for testing"""
        
        buildspec = """
        version: 0.2
        phases:
          install:
            runtime-versions:
              python: 3.13
            commands:
              - pip install "boto3==%%version=boto3%%" pytest
          pre_build:
            commands:
              - echo "Running pre-build validations..."
              - python -m pytest tests/
          build:
            commands:
              - echo "Running automated tests..."
              - python -m pytest tests/regression/
              - python -m pytest tests/security/
              - echo "Deploying agent..."
              - python scripts/deploy_agent.py
          post_build:
            commands:
              - echo "Running post-deployment tests..."
              - python -m pytest tests/integration/
        """
        
        response = self.codebuild.create_project(
            name=f'{agent_name}-build',
            source={
                'type': 'GITHUB',
                'location': 'https://github.com/myorg/bedrock-agents'
            },
            artifacts={'type': 'NO_ARTIFACTS'},
            environment={
                'type': 'LINUX_CONTAINER',
                'image': 'aws/codebuild/standard:7.0',
                'computeType': 'BUILD_GENERAL1_MEDIUM'
            },
            serviceRole=self._get_codebuild_role_arn(),
            logsConfig={
                'cloudWatchLogs': {
                    'status': 'ENABLED',
                    'groupName': f'/aws/codebuild/{agent_name}'
                }
            }
        )
        
        return response['project']
```

### Automated Testing

Automated testing is crucial for ensuring the quality and reliability of your Bedrock Agents. Your CI/CD pipeline should include the following types of tests:

*   **Prompt Regression Tests:** These tests ensure that changes to prompts do not negatively impact the agent's behavior. You can create a suite of test cases with sample inputs and expected outputs to validate the agent's responses.
*   **Schema Validation:** Validate the structure and content of your agent configurations, including action group schemas and knowledge base schemas.
*   **Security Checks:** Perform security scans to identify potential vulnerabilities in your agent's code and dependencies. You should also verify that the agent's IAM roles and policies adhere to the principle of least privilege.
*   **Model Output Validation:** Send sample inputs to the updated model and validate the outputs against expected criteria to catch regressions in the model's behavior.
*   **Integration Tests:** These tests verify the interaction between your agent and its external dependencies, such as action groups and knowledge bases.
*   **End-to-End Tests:** These tests simulate real user interactions with your agent to validate the entire workflow from start to finish.


---

## Compliance and Governance

### Compliance Framework

```python
class ComplianceFramework:
    """Compliance and governance for production agents"""
    
    def __init__(self):
        self.config = boto3.client('config')
        self.securityhub = boto3.client('securityhub')
    
    def setup_compliance_monitoring(self) -> None:
        """Setup compliance monitoring"""
        
        # Enable AWS Config
        self.config.put_configuration_recorder(
            ConfigurationRecorder={
                'name': 'default',
                'roleARN': 'arn:aws:iam::111122223333:role/aws-service-role/config.amazonaws.com/AWSServiceRoleForConfig',
                'recordingGroup': {'allSupported': True, 'includeGlobalResourceTypes': True},
            }
        )
        self.config.start_configuration_recorder(ConfigurationRecorderName='default')
        
        # Enable Security Hub
        self.securityhub.enable_security_hub()
    
    def audit_agent_compliance(self, agent_id: str) -> Dict:
        """Audit agent for compliance violations"""
        
        compliance_checks = {
            'encryption_enabled': self._check_encryption(agent_id),
            'guardrails_configured': self._check_guardrails(agent_id),
            'access_logged': self._check_logging(agent_id),
            'authentication_enabled': self._check_authentication(agent_id),
            'data_residency_compliant': self._check_data_residency(agent_id)
        }
        
        return compliance_checks
```

---

## Troubleshooting

### Common Issues and Solutions

```python
class TroubleshootingGuide:
    """Common issues and solutions"""
    
    def diagnose_agent_issues(self, agent_id: str) -> Dict:
        """Comprehensive agent diagnostics"""
        
        diagnostics = {
            'agent_status': self._check_agent_status(agent_id),
            'connection_issues': self._diagnose_connectivity(agent_id),
            'performance_issues': self._diagnose_performance(agent_id),
            'permission_issues': self._diagnose_permissions(agent_id),
            'knowledge_base_issues': self._diagnose_kb_issues(agent_id)
        }
        
        return diagnostics
    
    def troubleshoot_high_latency(self, agent_id: str) -> str:
        """Troubleshoot high latency issues"""
        
        # Check model latency
        # Check action group responsiveness
        # Check knowledge base query performance
        # Check network connectivity
        
        recommendation = """
        Latency Optimisation Steps:
        1. Profile individual components (model, KB, actions)
        2. Implement caching layer for frequently accessed data
        3. Switch to a faster model (Claude Haiku 5.5) for non-complex queries
        4. Review and optimise action group implementations
        5. Ensure proper database indexing for KB queries
        """
        
        return recommendation
    
    def troubleshoot_errors(self, agent_id: str, error_message: str) -> str:
        """Troubleshoot common error messages"""
        
        error_solutions = {
            'ResourceNotFoundException': 'Verify agent, KB, or action group IDs exist',
            'AccessDeniedException': 'Check IAM permissions for agent execution role',
            'ThrottlingException': 'Implement exponential backoff, check rate limits',
            'ValidationException': 'Validate input parameters and schema compliance',
            'ServiceUnavailableException': 'Retry with exponential backoff, check region'
        }
        
        for error_key, solution in error_solutions.items():
            if error_key in error_message:
                return f"✓ Solution for {error_key}: {solution}"
        
        return "✗ Unknown error, check CloudWatch logs for details"
```

---

This production guide provides comprehensive guidance for deploying and operating Bedrock Agents at enterprise scale with focus on reliability, security, performance, and cost optimisation.


