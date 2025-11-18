# Microsoft Agent Framework - Enterprise Features 2025
## October 2025 Release - Production-Grade Enterprise Capabilities

**Document Version:** 1.0
**Last Updated:** November 2025
**Target Audience:** Enterprise Architects, Security Teams, Compliance Officers
**Preview Status:** October 2025 Public Preview

---

## Table of Contents

1. [OpenTelemetry Instrumentation](#opentelemetry-instrumentation)
2. [Azure AI Content Safety Integration](#azure-ai-content-safety-integration)
3. [Microsoft Entra ID Authentication Patterns](#microsoft-entra-id-authentication-patterns)
4. [Regulated Industry Compliance](#regulated-industry-compliance)
5. [Enterprise Governance](#enterprise-governance)
6. [Advanced Security Features](#advanced-security-features)
7. [Audit and Compliance Logging](#audit-and-compliance-logging)

---

## OpenTelemetry Instrumentation

The October 2025 release includes comprehensive OpenTelemetry (OTel) instrumentation for distributed tracing, metrics, and logging.

### Automatic Instrumentation

All agent operations are automatically instrumented with OpenTelemetry spans:

```python
from agent_framework import ChatAgent
from agent_framework.telemetry import configure_telemetry
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure OpenTelemetry
def setup_telemetry():
    """Configure OpenTelemetry for agent operations"""

    # Create tracer provider
    tracer_provider = TracerProvider()
    trace.set_tracer_provider(tracer_provider)

    # Configure OTLP exporter (to Azure Monitor, Datadog, etc.)
    otlp_exporter = OTLPSpanExporter(
        endpoint="https://otel-collector.contoso.com:4317",
        headers={
            "api-key": os.getenv("OTEL_API_KEY")
        }
    )

    # Add span processor
    span_processor = BatchSpanProcessor(otlp_exporter)
    tracer_provider.add_span_processor(span_processor)

    # Configure agent framework telemetry
    configure_telemetry(
        service_name="customer-support-agents",
        service_version="1.0.0",
        environment="production",
        enable_auto_instrumentation=True
    )

# All agent operations are now traced
async def traced_agent_execution():
    setup_telemetry()

    async with DefaultAzureCredential() as credential:
        async with AzureAIAgentClient(async_credential=credential) as client:
            agent = ChatAgent(
                chat_client=client,
                instructions="You are helpful."
            )

            # This operation creates spans automatically
            response = await agent.run("Hello!")
            # Spans created:
            # - agent.run
            # - llm.request
            # - tool.execution (if tools used)
            # - memory.read / memory.write
```

### Custom Spans and Metrics

```python
from opentelemetry import trace, metrics
from agent_framework import ChatAgent

tracer = trace.get_tracer(__name__)
meter = metrics.get_meter(__name__)

# Custom metrics
agent_requests_counter = meter.create_counter(
    name="agent.requests",
    description="Total agent requests",
    unit="1"
)

agent_latency_histogram = meter.create_histogram(
    name="agent.latency",
    description="Agent request latency",
    unit="ms"
)

async def execute_with_custom_telemetry():
    """Add custom spans and metrics"""

    # Custom span
    with tracer.start_as_current_span("custom_agent_workflow") as span:
        span.set_attribute("workflow.name", "customer_support")
        span.set_attribute("customer.id", "cust_12345")

        start_time = time.time()

        # Execute agent
        agent = ChatAgent(instructions="You help customers.")
        response = await agent.run("What's my order status?")

        # Record metrics
        latency = (time.time() - start_time) * 1000
        agent_requests_counter.add(1, {"agent": "customer_support"})
        agent_latency_histogram.record(latency, {"agent": "customer_support"})

        span.set_attribute("response.length", len(response.text))
        span.add_event("agent_response_received")

        return response
```

### .NET OpenTelemetry Integration

```csharp
using OpenTelemetry;
using OpenTelemetry.Trace;
using OpenTelemetry.Metrics;
using OpenTelemetry.Resources;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Telemetry;

public class TelemetryConfiguration
{
    public static void ConfigureOpenTelemetry()
    {
        // Configure OpenTelemetry
        var serviceName = "enterprise-agents";
        var serviceVersion = "1.0.0";

        using var tracerProvider = Sdk.CreateTracerProviderBuilder()
            .SetResourceBuilder(ResourceBuilder.CreateDefault()
                .AddService(serviceName: serviceName, serviceVersion: serviceVersion))
            .AddSource("Microsoft.Agents.AI.*")
            .AddOtlpExporter(options =>
            {
                options.Endpoint = new Uri("https://otel-collector.contoso.com:4317");
                options.Headers = $"api-key={Environment.GetEnvironmentVariable("OTEL_API_KEY")}";
            })
            .AddAzureMonitorTraceExporter(options =>
            {
                options.ConnectionString = Environment.GetEnvironmentVariable("APPLICATIONINSIGHTS_CONNECTION_STRING");
            })
            .Build();

        using var meterProvider = Sdk.CreateMeterProviderBuilder()
            .SetResourceBuilder(ResourceBuilder.CreateDefault()
                .AddService(serviceName: serviceName, serviceVersion: serviceVersion))
            .AddMeter("Microsoft.Agents.AI.*")
            .AddOtlpExporter()
            .AddAzureMonitorMetricExporter()
            .Build();

        // Enable auto-instrumentation for agents
        AgentTelemetry.EnableAutoInstrumentation(new AgentTelemetryOptions
        {
            TraceAllOperations = true,
            IncludeMessageContent = false, // PII safety
            CustomTags = new Dictionary<string, string>
            {
                ["environment"] = "production",
                ["region"] = "eastus"
            }
        });
    }

    public static async Task ExecuteWithTelemetry()
    {
        ConfigureOpenTelemetry();

        var agent = new ChatAgent(
            instructions: "You are a helpful assistant."
        );

        // Automatically instrumented
        using var activity = AgentTelemetry.StartActivity("custom_workflow");
        activity?.SetTag("workflow.type", "customer_query");

        var result = await agent.RunAsync("Hello!");

        activity?.SetTag("response.length", result.Length);
    }
}
```

### Azure Monitor Integration

```python
from azure.monitor.opentelemetry import configure_azure_monitor
from agent_framework import ChatAgent

# Configure Azure Monitor
configure_azure_monitor(
    connection_string=os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING"),
    enable_live_metrics=True,
    instrumentation_options={
        "azure_sdk": {"enabled": True},
        "http_client": {"enabled": True},
        "logging": {"enabled": True}
    }
)

# All agent operations automatically flow to Azure Monitor
agent = ChatAgent(instructions="You are helpful.")
response = await agent.run("Query")

# View in Azure Monitor:
# - Application Map: See agent dependencies
# - Transaction search: Find specific agent requests
# - Performance: Analyze latency distributions
# - Failures: Track error rates
```

### Distributed Tracing Across Agents

```python
from opentelemetry import trace
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

async def multi_agent_traced_workflow():
    """Distributed tracing across multiple agents"""

    tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("multi_agent_workflow") as parent_span:
        # Extract context for propagation
        context = {}
        TraceContextTextMapPropagator().inject(context)

        # Agent 1 - with propagated context
        agent1 = ChatAgent(instructions="You research.")
        research = await agent1.run(
            "Research AI trends",
            trace_context=context
        )

        # Agent 2 - continues the trace
        agent2 = ChatAgent(instructions="You analyze.")
        analysis = await agent2.run(
            f"Analyze: {research.text}",
            trace_context=context
        )

        # Both operations appear in the same distributed trace
        parent_span.set_attribute("agents.count", 2)
        return analysis
```

---

## Azure AI Content Safety Integration

Azure AI Content Safety provides real-time content moderation for harmful content, ensuring compliance with content policies.

### Enabling Content Safety

```python
from agent_framework import ChatAgent
from agent_framework.safety import ContentSafetyConfig
from azure.ai.contentsafety import ContentSafetyClient
from azure.identity import DefaultAzureCredential

async def create_safe_agent():
    """Create agent with Content Safety integration"""

    # Configure Content Safety
    content_safety_config = ContentSafetyConfig(
        endpoint=os.getenv("CONTENT_SAFETY_ENDPOINT"),
        credential=DefaultAzureCredential(),
        categories={
            "hate": {"severity_threshold": 2, "block": True},
            "sexual": {"severity_threshold": 2, "block": True},
            "violence": {"severity_threshold": 4, "block": True},
            "self_harm": {"severity_threshold": 2, "block": True}
        },
        check_input=True,  # Check user input
        check_output=True,  # Check agent output
        action_on_violation="block_and_log"
    )

    # Create agent with Content Safety
    agent = ChatAgent(
        instructions="You are a customer support agent.",
        content_safety=content_safety_config
    )

    return agent

# Usage
async def safe_interaction():
    agent = await create_safe_agent()

    try:
        # Content Safety checks input and output
        response = await agent.run("User query here")
        print(response.text)

    except ContentSafetyViolation as e:
        # Harmful content detected
        print(f"Content violation: {e.category}")
        print(f"Severity: {e.severity}")
        print(f"Message blocked")
```

### Content Safety with Custom Actions

```python
from agent_framework.safety import ContentSafetyConfig, SafetyViolationHandler

class CustomSafetyHandler(SafetyViolationHandler):
    """Custom handler for content safety violations"""

    async def handle_input_violation(self, violation):
        """Handle violations in user input"""
        # Log to security system
        await self.log_security_event({
            "type": "input_violation",
            "category": violation.category,
            "severity": violation.severity,
            "user_id": violation.context.get("user_id"),
            "timestamp": datetime.utcnow()
        })

        # Return sanitized response
        return {
            "blocked": True,
            "message": "Your message violates our content policy.",
            "reference_id": violation.id
        }

    async def handle_output_violation(self, violation):
        """Handle violations in agent output"""
        # Log incident
        await self.log_security_event({
            "type": "output_violation",
            "category": violation.category,
            "agent_id": violation.agent_id
        })

        # Return safe fallback response
        return {
            "blocked": True,
            "message": "I apologize, but I cannot provide that information.",
            "fallback": True
        }

# Configure with custom handler
content_safety_config = ContentSafetyConfig(
    endpoint=os.getenv("CONTENT_SAFETY_ENDPOINT"),
    credential=DefaultAzureCredential(),
    violation_handler=CustomSafetyHandler()
)
```

### .NET Content Safety Integration

```csharp
using Azure.AI.ContentSafety;
using Azure.Identity;
using Microsoft.Agents.AI;
using Microsoft.Agents.AI.Safety;

public class SafeAgentConfiguration
{
    public static ChatAgent CreateSafeAgent()
    {
        // Configure Content Safety
        var contentSafetyConfig = new ContentSafetyConfig
        {
            Endpoint = new Uri(Environment.GetEnvironmentVariable("CONTENT_SAFETY_ENDPOINT")),
            Credential = new DefaultAzureCredential(),
            Categories = new Dictionary<string, CategoryConfig>
            {
                ["Hate"] = new CategoryConfig { SeverityThreshold = 2, BlockOnViolation = true },
                ["Sexual"] = new CategoryConfig { SeverityThreshold = 2, BlockOnViolation = true },
                ["Violence"] = new CategoryConfig { SeverityThreshold = 4, BlockOnViolation = true },
                ["SelfHarm"] = new CategoryConfig { SeverityThreshold = 2, BlockOnViolation = true }
            },
            CheckInput = true,
            CheckOutput = true,
            ActionOnViolation = ViolationAction.BlockAndLog
        };

        // Create agent
        var agent = new ChatAgent(
            instructions: "You are a helpful assistant.",
            contentSafety: contentSafetyConfig
        );

        return agent;
    }

    public static async Task SafeExecutionAsync()
    {
        var agent = CreateSafeAgent();

        try
        {
            var result = await agent.RunAsync("User query");
            Console.WriteLine(result);
        }
        catch (ContentSafetyViolationException ex)
        {
            Console.WriteLine($"Content violation: {ex.Category} (Severity: {ex.Severity})");
            // Handle violation appropriately
        }
    }
}
```

### PII Detection and Redaction

```python
from agent_framework.safety import PIIDetectionConfig

# Configure PII detection
pii_config = PIIDetectionConfig(
    enabled=True,
    detect_categories=[
        "email",
        "phone_number",
        "ssn",
        "credit_card",
        "ip_address",
        "person_name",
        "address"
    ],
    action="redact",  # Options: redact, block, log
    redaction_pattern="[REDACTED]"
)

agent = ChatAgent(
    instructions="You help with customer data.",
    pii_detection=pii_config
)

# PII is automatically detected and redacted
response = await agent.run("My email is user@example.com and my SSN is 123-45-6789")
# Output: "My email is [REDACTED] and my SSN is [REDACTED]"
```

---

## Microsoft Entra ID Authentication Patterns

### Managed Identity for Production

```python
from azure.identity.aio import ManagedIdentityCredential
from agent_framework import ChatAgent
from agent_framework.azure import AzureAIAgentClient

async def production_agent_with_managed_identity():
    """Production agent using Managed Identity"""

    # Use system-assigned managed identity
    async with ManagedIdentityCredential() as credential:
        async with AzureAIAgentClient(
            endpoint=os.getenv("AZURE_AI_ENDPOINT"),
            async_credential=credential
        ) as client:
            agent = ChatAgent(
                chat_client=client,
                instructions="Production agent"
            )

            response = await agent.run("Query")
            return response

# Assign roles to the managed identity:
# az role assignment create \
#   --assignee <managed-identity-principal-id> \
#   --role "Cognitive Services OpenAI User" \
#   --scope /subscriptions/<sub-id>/resourceGroups/<rg>/providers/Microsoft.CognitiveServices/accounts/<openai-account>
```

### .NET Managed Identity

```csharp
using Azure.Identity;
using Microsoft.Agents.AI;

public class ProductionAgent
{
    public static async Task<string> ExecuteWithManagedIdentity()
    {
        // System-assigned managed identity
        var credential = new ManagedIdentityCredential();

        var agentClient = new AzureAIAgentClient(
            endpoint: new Uri(Environment.GetEnvironmentVariable("AZURE_AI_ENDPOINT")),
            credential: credential
        );

        var agent = new ChatAgent(
            chatClient: agentClient,
            instructions: "Production agent"
        );

        var result = await agent.RunAsync("Query");
        return result;
    }
}
```

### Workload Identity for Kubernetes

```python
from azure.identity.aio import DefaultAzureCredential
from agent_framework import ChatAgent

async def kubernetes_agent_with_workload_identity():
    """Agent running in AKS with Workload Identity"""

    # DefaultAzureCredential automatically detects workload identity
    async with DefaultAzureCredential() as credential:
        agent = ChatAgent(
            instructions="Kubernetes agent",
            credential=credential
        )

        return agent

# Kubernetes deployment with workload identity:
```

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: agent-service-account
  namespace: agents
  annotations:
    azure.workload.identity/client-id: "<client-id>"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agent-deployment
spec:
  template:
    metadata:
      labels:
        azure.workload.identity/use: "true"
    spec:
      serviceAccountName: agent-service-account
      containers:
        - name: agent
          image: myagent:latest
          env:
            - name: AZURE_CLIENT_ID
              value: "<client-id>"
            - name: AZURE_TENANT_ID
              value: "<tenant-id>"
```

### User-Delegated Authentication

```python
from azure.identity.aio import DeviceCodeCredential, InteractiveBrowserCredential

async def user_delegated_agent():
    """Agent with user-delegated authentication"""

    # For desktop applications
    credential = InteractiveBrowserCredential(
        tenant_id=os.getenv("AZURE_TENANT_ID"),
        client_id=os.getenv("AZURE_CLIENT_ID")
    )

    # For command-line applications
    # credential = DeviceCodeCredential(
    #     tenant_id=os.getenv("AZURE_TENANT_ID"),
    #     client_id=os.getenv("AZURE_CLIENT_ID")
    # )

    async with credential:
        agent = ChatAgent(
            instructions="User-scoped agent",
            credential=credential
        )

        # Agent operates with user's permissions
        response = await agent.run("Access user data")
        return response
```

### Multi-Tenant Authentication

```python
from agent_framework.auth import MultiTenantAuthConfig

async def multi_tenant_agent_system():
    """Support multiple tenants with isolated authentication"""

    # Configure multi-tenant auth
    auth_config = MultiTenantAuthConfig(
        tenants={
            "tenant1": {
                "tenant_id": "tenant1-id",
                "client_id": "client1-id",
                "client_secret": os.getenv("TENANT1_SECRET")
            },
            "tenant2": {
                "tenant_id": "tenant2-id",
                "client_id": "client2-id",
                "client_secret": os.getenv("TENANT2_SECRET")
            }
        },
        isolation_mode="strict"  # Prevent cross-tenant data access
    )

    # Create tenant-specific agent
    async def create_tenant_agent(tenant_id: str):
        credential = auth_config.get_credential_for_tenant(tenant_id)

        agent = ChatAgent(
            instructions=f"Agent for {tenant_id}",
            credential=credential,
            tenant_isolation=True
        )

        return agent

    # Each tenant gets isolated agent
    tenant1_agent = await create_tenant_agent("tenant1")
    tenant2_agent = await create_tenant_agent("tenant2")
```

---

## Regulated Industry Compliance

### HIPAA Compliance Configuration

```python
from agent_framework.compliance import HIPAAConfig

# Configure HIPAA compliance
hipaa_config = HIPAAConfig(
    enabled=True,
    encrypt_at_rest=True,
    encrypt_in_transit=True,
    phi_detection=True,
    audit_all_access=True,
    data_retention_days=2555,  # 7 years
    backup_encryption=True,
    access_logging={
        "log_all_requests": True,
        "log_phi_access": True,
        "log_destination": "azure-storage",
        "storage_account": os.getenv("HIPAA_AUDIT_STORAGE")
    },
    business_associate_agreement=True
)

agent = ChatAgent(
    instructions="You handle healthcare data.",
    compliance=hipaa_config
)
```

### GDPR Compliance

```python
from agent_framework.compliance import GDPRConfig

# Configure GDPR compliance
gdpr_config = GDPRConfig(
    enabled=True,
    data_minimization=True,
    purpose_limitation=True,
    storage_limitation_days=365,
    right_to_erasure=True,
    right_to_access=True,
    consent_management={
        "require_explicit_consent": True,
        "consent_tracking": True
    },
    data_portability=True,
    cross_border_transfers={
        "allowed_regions": ["EU", "EEA"],
        "require_adequacy_decision": True
    },
    privacy_by_design=True
)

agent = ChatAgent(
    instructions="You handle EU customer data.",
    compliance=gdpr_config
)
```

### Financial Services Compliance (SOX, PCI-DSS)

```python
from agent_framework.compliance import FinancialServicesConfig

# Configure financial services compliance
financial_config = FinancialServicesConfig(
    sox_compliance={
        "enabled": True,
        "audit_trail": True,
        "change_management": True,
        "separation_of_duties": True
    },
    pci_dss_compliance={
        "enabled": True,
        "card_data_encryption": True,
        "tokenization": True,
        "network_segmentation": True,
        "vulnerability_management": True
    },
    audit_logging={
        "immutable_logs": True,
        "log_retention_years": 7,
        "real_time_monitoring": True
    }
)

agent = ChatAgent(
    instructions="You handle payment processing.",
    compliance=financial_config
)
```

### Government Cloud (FedRAMP)

```python
from agent_framework.compliance import FedRAMPConfig

# Configure FedRAMP compliance
fedramp_config = FedRAMPConfig(
    impact_level="moderate",  # low, moderate, high
    boundary_protection=True,
    continuous_monitoring=True,
    incident_response={
        "enabled": True,
        "notification_endpoints": ["security@agency.gov"]
    },
    configuration_management=True,
    cryptography={
        "fips_140_2_validated": True,
        "approved_algorithms_only": True
    },
    us_cloud_regions_only=True
)

agent = ChatAgent(
    instructions="You handle government data.",
    compliance=fedramp_config,
    cloud_regions=["usgovvirginia", "usgovarizona"]
)
```

---

## Enterprise Governance

### Policy Enforcement

```python
from agent_framework.governance import PolicyEngine, Policy

# Define policies
policy_engine = PolicyEngine()

# Data access policy
data_access_policy = Policy(
    name="data_access_restrictions",
    rules=[
        {
            "condition": "agent.accesses_pii == true",
            "require": ["role:data_handler", "clearance:level2"],
            "action": "allow_with_audit"
        },
        {
            "condition": "agent.accesses_financial_data == true",
            "require": ["role:financial_analyst", "mfa:enabled"],
            "action": "allow_with_monitoring"
        }
    ]
)

policy_engine.add_policy(data_access_policy)

# Apply policies to agent
agent = ChatAgent(
    instructions="You handle sensitive data.",
    policy_engine=policy_engine
)
```

### Cost Management and Budgets

```python
from agent_framework.governance import CostManagementConfig

# Configure cost controls
cost_config = CostManagementConfig(
    enabled=True,
    budget_limits={
        "daily_usd": 100.0,
        "monthly_usd": 2000.0
    },
    alerts=[
        {
            "threshold_percent": 80,
            "action": "notify",
            "recipients": ["finance@contoso.com"]
        },
        {
            "threshold_percent": 100,
            "action": "throttle",
            "throttle_percentage": 50
        }
    ],
    cost_allocation_tags={
        "department": "customer_service",
        "project": "ai_agents",
        "cost_center": "12345"
    },
    usage_tracking=True
)

agent = ChatAgent(
    instructions="Cost-controlled agent",
    cost_management=cost_config
)
```

### Model Governance

```python
from agent_framework.governance import ModelGovernanceConfig

# Configure model governance
model_governance = ModelGovernanceConfig(
    approved_models=[
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo"
    ],
    model_selection_policy="manual_approval",
    version_control=True,
    model_registry={
        "enabled": True,
        "registry_endpoint": "https://models.contoso.com"
    },
    performance_monitoring={
        "track_accuracy": True,
        "track_latency": True,
        "track_cost": True,
        "alert_on_degradation": True
    },
    fallback_models={
        "gpt-4o": ["gpt-4o-mini", "gpt-4-turbo"]
    }
)

agent = ChatAgent(
    instructions="Governed agent",
    model_governance=model_governance
)
```

---

## Advanced Security Features

### End-to-End Encryption

```python
from agent_framework.security import EncryptionConfig

# Configure end-to-end encryption
encryption_config = EncryptionConfig(
    enabled=True,
    algorithm="AES-256-GCM",
    key_management={
        "provider": "azure-key-vault",
        "key_vault_url": os.getenv("KEY_VAULT_URL"),
        "key_name": "agent-encryption-key",
        "key_version": "latest"
    },
    encrypt_messages=True,
    encrypt_state=True,
    encrypt_memory=True,
    encrypt_logs=True
)

agent = ChatAgent(
    instructions="Encrypted agent",
    encryption=encryption_config
)
```

### Network Security

```python
from agent_framework.security import NetworkSecurityConfig

# Configure network security
network_security = NetworkSecurityConfig(
    enable_private_endpoints=True,
    allowed_ip_ranges=[
        "10.0.0.0/8",  # Internal network
        "52.168.0.0/16"  # Azure services
    ],
    deny_public_access=True,
    require_tls_1_3=True,
    certificate_pinning=True,
    egress_control={
        "allowed_domains": [
            "*.openai.azure.com",
            "*.cosmos.azure.com",
            "api.contoso.com"
        ],
        "deny_by_default": True
    }
)

agent = ChatAgent(
    instructions="Network-secured agent",
    network_security=network_security
)
```

### Zero Trust Architecture

```python
from agent_framework.security import ZeroTrustConfig

# Configure zero trust
zero_trust_config = ZeroTrustConfig(
    verify_all_requests=True,
    never_trust_always_verify=True,
    least_privilege_access=True,
    assume_breach=True,
    microsegmentation={
        "enabled": True,
        "segment_by": ["tenant", "user", "data_classification"]
    },
    continuous_validation={
        "enabled": True,
        "revalidate_interval_minutes": 5
    },
    conditional_access={
        "require_mfa": True,
        "require_compliant_device": True,
        "trusted_locations_only": False
    }
)

agent = ChatAgent(
    instructions="Zero-trust agent",
    zero_trust=zero_trust_config
)
```

---

## Audit and Compliance Logging

### Comprehensive Audit Logging

```python
from agent_framework.audit import AuditConfig

# Configure audit logging
audit_config = AuditConfig(
    enabled=True,
    log_destination="azure-storage",
    storage_account=os.getenv("AUDIT_STORAGE_ACCOUNT"),
    container="audit-logs",
    log_levels={
        "agent_creation": "INFO",
        "agent_execution": "INFO",
        "tool_usage": "INFO",
        "data_access": "WARNING",
        "security_events": "CRITICAL"
    },
    include_in_logs=[
        "timestamp",
        "user_id",
        "agent_id",
        "operation",
        "result",
        "ip_address",
        "user_agent",
        "correlation_id"
    ],
    exclude_from_logs=[
        "passwords",
        "api_keys",
        "pii",
        "phi"
    ],
    immutable_logs=True,
    log_retention_years=7,
    compliance_format="json",
    real_time_monitoring=True
)

agent = ChatAgent(
    instructions="Audited agent",
    audit=audit_config
)
```

### Security Information and Event Management (SIEM)

```python
from agent_framework.audit import SIEMIntegration

# Configure SIEM integration
siem_config = SIEMIntegration(
    provider="microsoft-sentinel",
    workspace_id=os.getenv("SENTINEL_WORKSPACE_ID"),
    shared_key=os.getenv("SENTINEL_SHARED_KEY"),
    log_type="AgentSecurityEvents",
    send_security_events=True,
    send_audit_events=True,
    event_types=[
        "authentication_failure",
        "unauthorized_access_attempt",
        "content_policy_violation",
        "anomalous_behavior",
        "privilege_escalation_attempt"
    ],
    alert_rules=[
        {
            "name": "multiple_failed_auth",
            "condition": "failed_auth_count > 5 in 5 minutes",
            "severity": "high",
            "action": "block_and_alert"
        }
    ]
)

agent = ChatAgent(
    instructions="SIEM-integrated agent",
    siem=siem_config
)
```

---

## Best Practices

### 1. Observability
- Enable OpenTelemetry from day one
- Export to multiple backends for redundancy
- Set up dashboards and alerts
- Monitor cost and performance metrics

### 2. Content Safety
- Enable for all user-facing agents
- Configure appropriate severity thresholds
- Implement custom violation handlers
- Regular review of blocked content

### 3. Authentication
- Use Managed Identity in production
- Implement least privilege access
- Regular credential rotation
- Multi-factor authentication where possible

### 4. Compliance
- Understand regulatory requirements
- Enable compliance features early
- Regular compliance audits
- Document compliance measures

### 5. Security
- Encrypt data at rest and in transit
- Implement zero trust principles
- Regular security assessments
- Incident response planning

---

## Migration Checklist

- [ ] Enable OpenTelemetry instrumentation
- [ ] Configure Azure Monitor integration
- [ ] Enable Content Safety for user-facing agents
- [ ] Configure PII detection and redaction
- [ ] Migrate to Managed Identity authentication
- [ ] Implement compliance requirements (GDPR, HIPAA, etc.)
- [ ] Enable comprehensive audit logging
- [ ] Configure SIEM integration
- [ ] Set up cost management and budgets
- [ ] Implement policy enforcement
- [ ] Configure network security
- [ ] Enable end-to-end encryption
- [ ] Set up monitoring and alerting
- [ ] Document security architecture
- [ ] Conduct security assessment

---

## Resources

### Documentation
- [Azure Monitor OpenTelemetry](https://learn.microsoft.com/azure/azure-monitor/app/opentelemetry-overview)
- [Azure AI Content Safety](https://learn.microsoft.com/azure/ai-services/content-safety/)
- [Microsoft Entra ID](https://learn.microsoft.com/entra/identity/)
- [Azure Compliance](https://learn.microsoft.com/azure/compliance/)

### Tools
- [Azure Policy](https://learn.microsoft.com/azure/governance/policy/)
- [Azure Security Center](https://learn.microsoft.com/azure/security-center/)
- [Microsoft Sentinel](https://learn.microsoft.com/azure/sentinel/)

---

**Last Updated:** November 2025
**Document Version:** 1.0
**Preview Status:** October 2025 Public Preview
**Next Review:** Q2 2026
