# Amazon Bedrock Agents: Production Deployment Guide

A comprehensive guide for deploying, operating, monitoring, and optimising Amazon Bedrock Agents in production environments at enterprise scale.

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
from typing import Dict, List, Optional
import logging
from datetime import datetime

class ProductionBedrockArchitecture:
    """Enterprise-grade Bedrock Agents production architecture"""
    
    def __init__(self, environment: str = 'production'):
        self.environment = environment
        self.logger = self._setup_logging()
        
        # Multi-region setup
        self.primary_region = 'us-east-1'
        self.secondary_region = 'eu-west-1'
        
        self.bedrock_primary = boto3.client('bedrock', region_name=self.primary_region)
        self.bedrock_secondary = boto3.client('bedrock', region_name=self.secondary_region)
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
            
            # Deploy to primary region
            primary_response = self.bedrock_primary.create_agent(**agent_config)
            primary_agent_id = primary_response['agentId']
            self.logger.info(f"✓ Deployed agent to {self.primary_region}: {primary_agent_id}")
            
            # Deploy to secondary region for DR
            secondary_response = self.bedrock_secondary.create_agent(**agent_config)
            secondary_agent_id = secondary_response['agentId']
            self.logger.info(f"✓ Deployed agent to {self.secondary_region}: {secondary_agent_id}")
            
            # Create production alias
            alias_response = self.bedrock_primary.create_agent_alias(
                agentId=primary_agent_id,
                agentAliasName='production',
                description='Production alias for agent'
            )
            
            # Store deployment metadata
            deployment_info = {
                'timestamp': datetime.utcnow().isoformat(),
                'environment': self.environment,
                'primary_agent_id': primary_agent_id,
                'secondary_agent_id': secondary_agent_id,
                'alias_id': alias_response['agentAliasId'],
                'status': 'DEPLOYED',
                'health_check': 'PASSED'
            }
            
            self._store_deployment_metadata(deployment_info)
            return deployment_info
        
        except Exception as e:
            self.logger.error(f"✗ Deployment failed: {e}")
            raise
    
    def _validate_agent_config(self, config: Dict) -> bool:
        """Validate agent configuration against production standards"""
        
        required_fields = ['agentName', 'foundationModelId', 'agentRoleArn']
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
            'timestamp': datetime.utcnow().isoformat(),
            'checks': {}
        }
        
        # Check 1: Agent accessibility
        try:
            agent = self.bedrock.get_agent(agentId=agent_id)
            health_status['checks']['agent_access'] = 'PASS'
        except Exception as e:
            health_status['checks']['agent_access'] = f'FAIL: {e}'
        
        # Check 2: Foundation model availability
        try:
            models = self.bedrock.list_foundation_models()
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

```python
class BlueGreenDeployment:
    """Blue-Green deployment strategy for Bedrock Agents"""
    
    def __init__(self):
        self.bedrock = boto3.client('bedrock')
        self.route53 = boto3.client('route53')
    
    def perform_blue_green_deployment(self, 
                                      new_agent_config: Dict,
                                      current_agent_id: str) -> Dict:
        """
        Deploy new agent (Green) alongside current (Blue)
        with automatic failover
        """
        
        # Step 1: Deploy green agent
        green_response = self.bedrock.create_agent(**new_agent_config)
        green_agent_id = green_response['agentId']
        
        # Step 2: Run comprehensive tests
        test_results = self._run_test_suite(green_agent_id)
        
        if not test_results['all_passed']:
            raise Exception(f"Tests failed: {test_results['failures']}")
        
        # Step 3: Gradual traffic shift
        self._gradual_traffic_shift(current_agent_id, green_agent_id)
        
        # Step 4: Monitor and validate
        monitoring_results = self._monitor_transition(green_agent_id)
        
        return {
            'blue_agent_id': current_agent_id,
            'green_agent_id': green_agent_id,
            'status': 'DEPLOYED',
            'monitoring': monitoring_results
        }
    
    def _run_test_suite(self, agent_id: str) -> Dict:
        """Run comprehensive test suite"""
        tests = [
            self._test_basic_invocation,
            self._test_action_groups,
            self._test_knowledge_bases,
            self._test_guardrails
        ]
        
        results = {'all_passed': True, 'failures': []}
        for test in tests:
            try:
                test(agent_id)
            except Exception as e:
                results['all_passed'] = False
                results['failures'].append(str(e))
        
        return results
    
    def _gradual_traffic_shift(self, blue_id: str, green_id: str, duration_minutes: int = 30):
        """Gradually shift traffic from blue to green"""
        traffic_distribution = [
            (100, 0),    # All to blue
            (90, 10),
            (75, 25),
            (50, 50),
            (25, 75),
            (10, 90),
            (0, 100)     # All to green
        ]
        
        for blue_percent, green_percent in traffic_distribution:
            # Update Route 53 weighted routing
            # Wait 5 minutes between shifts
            print(f"Traffic: {blue_percent}% → {green_percent}%")
            time.sleep(300)
```

### Canary Deployment Pattern

```python
class CanaryDeployment:
    """Canary deployment for risk mitigation"""
    
    def __init__(self):
        self.bedrock = boto3.client('bedrock')
        self.cloudwatch = boto3.client('cloudwatch')
    
    def deploy_with_canary(self, 
                          new_agent_config: Dict,
                          current_agent_id: str,
                          canary_percentage: int = 5) -> Dict:
        """
        Deploy new agent to small user segment (canary)
        before full rollout
        """
        
        # Deploy new agent
        canary_response = self.bedrock.create_agent(**new_agent_config)
        canary_agent_id = canary_response['agentId']
        
        # Configure canary traffic
        self._configure_canary_routing(current_agent_id, canary_agent_id, canary_percentage)
        
        # Monitor canary metrics
        canary_metrics = {
            'error_rate': None,
            'latency': None,
            'user_satisfaction': None
        }
        
        # Wait 1 hour for canary monitoring
        monitoring_duration = 3600
        start_time = datetime.utcnow()
        
        while (datetime.utcnow() - start_time).seconds < monitoring_duration:
            canary_metrics = self._collect_canary_metrics(canary_agent_id)
            
            # Check for issues
            if canary_metrics['error_rate'] > 0.05:  # >5% error rate
                print("⚠ Canary error rate exceeded threshold, rolling back...")
                self._rollback_canary(current_agent_id, canary_agent_id)
                return {'status': 'ROLLED_BACK'}
            
            time.sleep(60)  # Check every minute
        
        # If canary successful, proceed with full deployment
        return {
            'status': 'CANARY_PASSED',
            'canary_agent_id': canary_agent_id,
            'final_metrics': canary_metrics
        }
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
        
        bedrock = boto3.client('bedrock')
        
        response = bedrock.create_knowledge_base(
            name=kb_name,
            encryptionConfiguration={
                'kmsKeyArn': f'arn:aws:kms:us-east-1:ACCOUNT:key/{kms_key_id}'
            },
            # ... other config
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
    """Configure VPC and network security for agents"""
    
    def __init__(self):
        self.ec2 = boto3.client('ec2')
        self.bedrock = boto3.client('bedrock')
    
    def create_private_agent(self,
                            agent_name: str,
                            vpc_id: str,
                            subnet_ids: List[str],
                            security_group_ids: List[str]) -> Dict:
        """Deploy agent in private VPC without internet access"""
        
        response = self.bedrock.create_agent(
            agentName=agent_name,
            networkConfiguration={
                'vpcConfiguration': {
                    'vpcId': vpc_id,
                    'subnetIds': subnet_ids,
                    'securityGroupIds': security_group_ids,
                    'enablePrivateEndpoint': True
                }
            }
        )
        
        return response
    
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
                        "bedrock:InvokeAgent",
                        "bedrock:GetAgent"
                    ],
                    "Resource": "arn:aws:bedrock:*:*:agent/*",
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
                    "Sid": "BedrockAgentInvocation",
                    "Effect": "Allow",
                    "Action": [
                        "bedrock:InvokeAgent",
                        "bedrock:GetAgent"
                    ],
                    "Resource": [
                        "arn:aws:bedrock:us-east-1:*:agent/*",
                        "arn:aws:bedrock:eu-west-1:*:agent/*"
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
                        "organisations:*"
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
        self.bedrock = boto3.client('bedrock')
        self.cloudwatch = boto3.client('cloudwatch')
    
    def analyze_latency_bottlenecks(self, agent_id: str) -> Dict:
        """Identify latency bottlenecks"""
        
        metrics = self.cloudwatch.get_metric_statistics(
            Namespace='AWS/Bedrock',
            MetricName='AgentInvocationLatency',
            Dimensions=[{'Name': 'AgentId', 'Value': agent_id}],
            StartTime=datetime.utcnow() - timedelta(hours=1),
            EndTime=datetime.utcnow(),
            Period=60,
            Statistics=['Average', 'Maximum', 'Minimum']
        )
        
        latency_breakdown = {
            'model_latency': self._measure_model_latency(agent_id),
            'action_group_latency': self._measure_action_group_latency(agent_id),
            'knowledge_base_latency': self._measure_kb_latency(agent_id),
            'network_latency': self._measure_network_latency()
        }
        
        return latency_breakdown
    
    def implement_caching_layer(self, agent_id: str) -> None:
        """Implement ElastiCache for frequently accessed data"""
        
        # Cache configuration
        cache_config = {
            'CacheClusterId': f'{agent_id}-cache',
            'CacheNodeType': 'cache.t3.medium',
            'Engine': 'redis',
            'NumCacheNodes': 2,
            'AutomaticFailover': True,
            'MultiAZEnabled': True,
            'CacheSubnetGroupName': 'bedrock-subnet-group'
        }
        
        elasticache = boto3.client('elasticache')
        elasticache.create_cache_cluster(**cache_config)
    
    def optimize_model_selection(self, agent_id: str) -> str:
        """Select appropriate model based on performance requirements"""
        
        metrics = self.analyze_latency_bottlenecks(agent_id)
        
        if metrics['model_latency'] > 5000:  # >5 seconds
            return 'claude-3-haiku'  # Faster model
        else:
            return 'claude-3-sonnet'  # Better quality
```

---

## Cost Optimisation

### Comprehensive Cost Analysis

```python
class CostOptimisation:
    """Optimise Bedrock Agents costs"""
    
    def __init__(self):
        self.ce = boto3.client('ce')  # Cost Explorer
        self.bedrock = boto3.client('bedrock')
    
    def analyze_agent_costs(self, agent_id: str) -> Dict:
        """Detailed cost analysis per agent"""
        
        # Get cost data from Cost Explorer
        response = self.ce.get_cost_and_usage(
            TimePeriod={
                'Start': (datetime.utcnow() - timedelta(days=30)).strftime('%Y-%m-%d'),
                'End': datetime.utcnow().strftime('%Y-%m-%d')
            },
            Granularity='MONTHLY',
            Metrics=['UnblendedCost'],
            Filter={
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
    
    def implement_reserved_capacity(self) -> None:
        """Reserve capacity for predictable workloads"""
        
        # Implementation for reserved capacity
        pass
    
    def optimize_token_usage(self, agent_id: str) -> Dict:
        """Optimise model token consumption"""
        
        optimisations = {
            'prompt_caching': True,
            'context_compression': True,
            'batch_processing': True,
            'model_optimization': 'claude-3-haiku'  # Lower cost
        }
        
        return optimisations
```

---

## Monitoring and Observability

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
        
        # Enable X-Ray tracing
        self._enable_xray_tracing(agent_id)
    
    def _create_agent_dashboard(self, agent_id: str) -> None:
        """Create operational dashboard"""
        
        dashboard_body = {
            "widgets": [
                {
                    "type": "metric",
                    "properties": {
                        "metrics": [
                            ["AWS/Bedrock", "AgentInvocationLatency", 
                             {"stat": "Average"}],
                            [".", ".", {"stat": "p99"}],
                            [".", "AgentInvocationErrors", {"stat": "Sum"}]
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
            MetricName='AgentInvocationErrors',
            Namespace='AWS/Bedrock',
            Statistic='Sum',
            Period=300,
            EvaluationPeriods=2,
            Threshold=10,
            ComparisonOperator='GreaterThanThreshold',
            AlarmActions=['arn:aws:sns:us-east-1:ACCOUNT:bedrock-alerts']
        )
        
        # High latency alarm
        self.cloudwatch.put_metric_alarm(
            AlarmName=f'{agent_id}-high-latency',
            MetricName='AgentInvocationLatency',
            Namespace='AWS/Bedrock',
            Statistic='Average',
            Period=300,
            EvaluationPeriods=2,
            Threshold=5000,  # 5 seconds
            ComparisonOperator='GreaterThanThreshold',
            AlarmActions=['arn:aws:sns:us-east-1:ACCOUNT:bedrock-alerts']
        )
```

---

## Disaster Recovery

### Disaster Recovery Strategy

```python
class DisasterRecoveryStrategy:
    """Production disaster recovery for Bedrock Agents"""
    
    def __init__(self):
        self.bedrock_primary = boto3.client('bedrock', region_name='us-east-1')
        self.bedrock_secondary = boto3.client('bedrock', region_name='eu-west-1')
        self.s3 = boto3.client('s3')
        self.dynamodb = boto3.resource('dynamodb')
    
    def backup_agent_configuration(self, agent_id: str) -> str:
        """Backup agent configuration for disaster recovery"""
        
        # Get agent configuration
        agent = self.bedrock_primary.get_agent(agentId=agent_id)
        
        # Store in S3
        backup_key = f'bedrock-backups/{agent_id}/{datetime.utcnow().isoformat()}.json'
        self.s3.put_object(
            Bucket='bedrock-disaster-recovery',
            Key=backup_key,
            Body=json.dumps(agent),
            ServerSideEncryption='AES256',
            VersioningConfiguration={'Status': 'Enabled'}
        )
        
        return backup_key
    
    def enable_cross_region_replication(self, 
                                       primary_agent_id: str,
                                       config: Dict) -> Dict:
        """Enable cross-region replication for disaster recovery"""
        
        # Deploy to secondary region
        secondary_response = self.bedrock_secondary.create_agent(**config)
        
        # Setup replication for knowledge bases
        self._replicate_knowledge_bases(primary_agent_id)
        
        # Setup Route 53 failover
        self._setup_route53_failover(primary_agent_id, secondary_response['agentId'])
        
        return {
            'primary_agent_id': primary_agent_id,
            'secondary_agent_id': secondary_response['agentId'],
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

### Automated Agent Deployment Pipeline

```python
class AgentDeploymentPipeline:
    """CI/CD pipeline for Bedrock Agents"""
    
    def __init__(self):
        self.codepipeline = boto3.client('codepipeline')
        self.codebuild = boto3.client('codebuild')
        self.bedrock = boto3.client('bedrock')
    
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
              python: 3.11
            commands:
              - pip install boto3 pytest
          pre_build:
            commands:
              - echo "Running pre-build validations..."
              - python -m pytest tests/
          build:
            commands:
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
        self.config.put_config_recorder(
            ConfigRecorderName='bedrock-recorder',
            RoleARN='arn:aws:iam::ACCOUNT:role/aws-service-role/config.amazonaws.com/AWSServiceRoleForConfig',
            RecordingGroup={
                'allSupported': True,
                'includedResources': [
                    'AWS::Bedrock::Agent',
                    'AWS::Bedrock::KnowledgeBase'
                ]
            }
        )
        
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
        3. Switch to faster model (Haiku) for non-complex queries
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


