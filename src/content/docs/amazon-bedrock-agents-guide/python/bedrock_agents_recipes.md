---
title: "Amazon Bedrock Agents: Recipes and Common Patterns"
description: "Practical recipes for Amazon Bedrock Agents (Agents Classic) and AgentCore: support agents, supervisor multi-agent systems, A2A, financial analytics, code interpretation and document processing."
framework: "amazon-bedrock-agents"
language: "python"
sidebar:
  label: "Recipes"
  order: 9
---

Practical recipes for common Bedrock agent scenarios.

> **Verified against:** `boto3` %%version=boto3%%, `strands-agents` %%version=strands-agents%% and `bedrock-agentcore` %%version=bedrock-agentcore%% (checked %%versions-checked%%).
>
> Recipes 1, 2, 4, 5 and 6 use **Bedrock Agents (Agents Classic)**, which no longer accepts new customers (since 30 July 2026) and is in maintenance mode; they remain valid for existing accounts. Recipe 3 uses **AgentCore**, AWS's recommended platform for new agents.

**Conventions used throughout:**

- `bedrock-agent` is the build-time client (agents, action groups, knowledge bases); `bedrock-agent-runtime` invokes agents; `bedrock` manages guardrails.
- `AGENT_ROLE_ARN` is a service role trusted by `bedrock.amazonaws.com` with `bedrock:InvokeModel*` on the chosen model; each action-group Lambda needs a resource policy allowing `bedrock.amazonaws.com` to invoke it.
- Models are referenced through cross-Region inference profiles such as `us.anthropic.claude-sonnet-5-5`.
- New agents must be **prepared** and given an **alias** before applications can invoke them; the helper below does that.

```python
import time

import boto3

AGENT_ROLE_ARN = "arn:aws:iam::111122223333:role/BedrockAgentRole"
SONNET = "us.anthropic.claude-sonnet-5-5"
HAIKU = "us.anthropic.claude-haiku-5-5"
OPUS = "us.anthropic.claude-opus-5-5"


def wait_for_agent(client, agent_id: str) -> str:
    """Wait until the agent is no longer CREATING, UPDATING or PREPARING; return its status."""
    while True:
        status = client.get_agent(agentId=agent_id)["agent"]["agentStatus"]
        if status not in ("CREATING", "UPDATING", "PREPARING"):
            return status
        time.sleep(2)


def prepare_and_alias(client, agent_id: str, alias_name: str = "live") -> str:
    """Prepare the DRAFT version and create an alias pointing at a new numbered version."""
    wait_for_agent(client, agent_id)
    client.prepare_agent(agentId=agent_id)  # status becomes PREPARING straight away
    if wait_for_agent(client, agent_id) != "PREPARED":
        raise RuntimeError(f"agent {agent_id} failed to prepare")
    alias_id = client.create_agent_alias(agentId=agent_id, agentAliasName=alias_name)["agentAlias"]["agentAliasId"]
    while (status := client.get_agent_alias(agentId=agent_id, agentAliasId=alias_id)["agentAlias"]["agentAliasStatus"]) != "PREPARED":
        if status == "FAILED":
            raise RuntimeError(f"alias {alias_id} failed")
        time.sleep(2)
    return alias_id


def read_completion(response) -> str:
    """invoke_agent returns an event stream; join the text chunks."""
    return "".join(
        event["chunk"]["bytes"].decode() for event in response["completion"] if "chunk" in event
    )
```

---

## Table of Contents

1. [Customer Support Agent](#customer-support-agent)
2. [Multi-Agent Customer Service System](#multi-agent-customer-service-system)
3. [Multi-Agent System with A2A Protocol](#multi-agent-system-with-a2a-protocol)
4. [Financial Analytics Agent](#financial-analytics-agent)
5. [Data Analysis Agent with Code Interpretation](#data-analysis-agent-with-code-interpretation)
6. [Document Processing Pipeline](#document-processing-pipeline)

---

## Customer Support Agent

### Basic Customer Support Implementation

```python
import json
import logging
from typing import Dict

import boto3


class CustomerSupportAgent:
    """Customer support agent with action groups, a knowledge base and a guardrail."""

    def __init__(self, region="us-east-1"):
        self.agents = boto3.client("bedrock-agent", region_name=region)
        self.runtime = boto3.client("bedrock-agent-runtime", region_name=region)
        self.bedrock = boto3.client("bedrock", region_name=region)  # guardrails
        self.logger = logging.getLogger(__name__)

    def create_support_agent(self, agent_name="CustomerSupportAgent") -> Dict[str, str]:
        guardrail = self._create_support_guardrail(agent_name)

        agent = self.agents.create_agent(
            agentName=agent_name,
            description="Customer support agent for tickets, FAQs and accounts",
            foundationModel=SONNET,
            agentResourceRoleArn=AGENT_ROLE_ARN,
            instruction="""You are an expert customer support agent. Your responsibilities:
1. Answer frequently asked questions about our products and services
2. Help customers troubleshoot common issues
3. Create and look up support tickets
4. Escalate complex issues to human specialists

Always:
- Verify customer identity before accessing account information
- Provide clear step-by-step solutions
- Suggest self-service options before escalation""",
            idleSessionTTLInSeconds=900,
            customerEncryptionKeyArn="arn:aws:kms:us-east-1:111122223333:key/KEY_ID",
            guardrailConfiguration={
                "guardrailIdentifier": guardrail["guardrailId"],
                "guardrailVersion": guardrail["version"],
            },
        )["agent"]
        agent_id = agent["agentId"]

        self._create_ticket_action_group(agent_id)
        self._create_account_action_group(agent_id)
        self._attach_support_knowledge_base(agent_id, kb_id="KB_ID")  # see the note below

        alias_id = prepare_and_alias(self.agents, agent_id)
        return {"agent_id": agent_id, "alias_id": alias_id}

    def _create_ticket_action_group(self, agent_id: str) -> None:
        ticket_schema = {
            "openapi": "3.0.0",
            "info": {"title": "Support Ticket API", "version": "1.0.0"},
            "paths": {
                "/tickets": {
                    "post": {
                        "operationId": "CreateTicket",
                        "description": "Create a new support ticket",
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
                                            "priority": {"type": "string", "enum": ["LOW", "MEDIUM", "HIGH"]},
                                        },
                                        "required": ["customerId", "category", "subject"],
                                    }
                                }
                            },
                        },
                        "responses": {"201": {"description": "Ticket created"}},
                    }
                },
                "/tickets/{ticketId}": {
                    "get": {
                        "operationId": "GetTicket",
                        "description": "Retrieve ticket details",
                        "parameters": [
                            {"name": "ticketId", "in": "path", "required": True, "schema": {"type": "string"}}
                        ],
                        "responses": {"200": {"description": "Ticket details"}},
                    }
                },
            },
        }

        self.agents.create_agent_action_group(
            agentId=agent_id,
            agentVersion="DRAFT",
            actionGroupName="TicketManagement",
            description="Manage customer support tickets",
            apiSchema={"payload": json.dumps(ticket_schema)},
            actionGroupExecutor={"lambda": "arn:aws:lambda:us-east-1:111122223333:function:support-ticket-handler"},
        )

    def _create_account_action_group(self, agent_id: str) -> None:
        """Function-schema action groups avoid writing OpenAPI for simple operations."""
        self.agents.create_agent_action_group(
            agentId=agent_id,
            agentVersion="DRAFT",
            actionGroupName="AccountManagement",
            description="Look up customer account information",
            functionSchema={
                "functions": [
                    {
                        "name": "get_account",
                        "description": "Get customer account details",
                        "parameters": {
                            "customerId": {"type": "string", "description": "Customer ID", "required": True}
                        },
                    }
                ]
            },
            actionGroupExecutor={"lambda": "arn:aws:lambda:us-east-1:111122223333:function:account-handler"},
        )

    def _attach_support_knowledge_base(self, agent_id: str, kb_id: str) -> None:
        self.agents.associate_agent_knowledge_base(
            agentId=agent_id,
            agentVersion="DRAFT",
            knowledgeBaseId=kb_id,
            description="Support FAQs, policies and troubleshooting guides. Use for product and policy questions.",
        )

    def _create_support_guardrail(self, name: str) -> Dict:
        guardrail = self.bedrock.create_guardrail(
            name=f"{name}-guardrail",
            description="Guardrail for the customer support agent",
            blockedInputMessaging="Sorry, I can't help with that request.",
            blockedOutputsMessaging="Sorry, I can't share that.",
            topicPolicyConfig={
                "topicsConfig": [
                    {
                        "name": "UnauthorisedAccess",
                        "definition": "Attempts to access another customer's account or bypass identity verification",
                        "type": "DENY",
                    }
                ]
            },
            contentPolicyConfig={
                "filtersConfig": [
                    {"type": "HATE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
                    {"type": "SEXUAL", "inputStrength": "HIGH", "outputStrength": "HIGH"},
                    {"type": "VIOLENCE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
                ]
            },
        )
        version = self.bedrock.create_guardrail_version(guardrailIdentifier=guardrail["guardrailId"])
        return {"guardrailId": guardrail["guardrailId"], "version": version["version"]}

    def invoke_support_agent(self, agent_id: str, alias_id: str, customer_query: str, session_id: str) -> Dict:
        response = self.runtime.invoke_agent(
            agentId=agent_id,
            agentAliasId=alias_id,
            sessionId=session_id,
            inputText=customer_query,
            enableTrace=True,
        )
        text, traces = "", []
        for event in response["completion"]:
            if "chunk" in event:
                text += event["chunk"]["bytes"].decode()
            elif "trace" in event:
                traces.append(event["trace"])
        return {"agent_response": text, "traces": traces}
```

Knowledge bases (and their vector store and S3 data source) are usually created once, separately from the agent; see the [comprehensive guide](/amazon-bedrock-agents-guide/python/bedrock_agents_comprehensive_guide/#knowledge-bases-contextual-information-integration) for `create_knowledge_base` and `create_data_source`.

**Action-group Lambda (OpenAPI schema).** Bedrock sends `apiPath`, `httpMethod`, `parameters` and `requestBody`, and expects a specific response envelope:

```python
import json
import time
from datetime import datetime, timezone

import boto3

table = boto3.resource("dynamodb").Table("SupportTickets")


def ticket_handler(event, context):
    api_path, method = event["apiPath"], event["httpMethod"]
    params = {p["name"]: p["value"] for p in event.get("parameters", [])}

    if api_path == "/tickets" and method == "POST":
        props = event["requestBody"]["content"]["application/json"]["properties"]
        body = {p["name"]: p["value"] for p in props}
        now = datetime.now(timezone.utc).isoformat()
        ticket = {"ticketId": f"TICKET-{int(time.time())}", "status": "OPEN", "createdAt": now, **body}
        table.put_item(Item=ticket)
        status, result = 201, ticket
    elif api_path == "/tickets/{ticketId}" and method == "GET":
        status, result = 200, table.get_item(Key={"ticketId": params["ticketId"]}).get("Item", {})
    else:
        status, result = 404, {"error": f"Unsupported {method} {api_path}"}

    return {
        "messageVersion": "1.0",
        "response": {
            "actionGroup": event["actionGroup"],
            "apiPath": api_path,
            "httpMethod": method,
            "httpStatusCode": status,
            "responseBody": {"application/json": {"body": json.dumps(result, default=str)}},
        },
    }
```

---

## Multi-Agent Customer Service System

### Orchestrated Multi-Agent System

Bedrock Agents multi-agent collaboration: specialist agents are created and aliased first, then associated with a supervisor whose `agentCollaboration` is `SUPERVISOR` (synthesises answers) or `SUPERVISOR_ROUTER` (routes each request to one collaborator).

```python
from typing import Dict

import boto3


class MultiAgentCustomerServiceSystem:
    """Supervisor agent with four collaborator agents."""

    SPECIALISTS = {
        "SupportSpecialist": ("Handles general support and troubleshooting",
                              "You are a Support Specialist. Answer general questions, guide troubleshooting and manage tickets."),
        "BillingSpecialist": ("Handles invoices, payments and refunds",
                              "You are a Billing Specialist. Handle invoices, refunds and payment issues precisely."),
        "TechnicalSpecialist": ("Handles technical and integration issues",
                                "You are a Technical Specialist. Debug system errors, API problems and integrations."),
        "SalesSpecialist": ("Handles product, pricing and upgrade questions",
                            "You are a Sales Specialist. Answer product and pricing questions and handle upgrades."),
    }

    def __init__(self, region="us-east-1"):
        self.agents = boto3.client("bedrock-agent", region_name=region)
        self.runtime = boto3.client("bedrock-agent-runtime", region_name=region)

    def _create_agent(self, name: str, instruction: str, model: str, **extra) -> str:
        return self.agents.create_agent(
            agentName=name,
            foundationModel=model,
            instruction=instruction,
            agentResourceRoleArn=AGENT_ROLE_ARN,
            **extra,
        )["agent"]["agentId"]

    def create_multi_agent_system(self) -> Dict:
        # 1. Specialists, each prepared and aliased
        specialists = {}
        for name, (description, instruction) in self.SPECIALISTS.items():
            agent_id = self._create_agent(name, instruction, HAIKU, description=description)
            alias_id = prepare_and_alias(self.agents, agent_id)
            alias_arn = self.agents.get_agent_alias(agentId=agent_id, agentAliasId=alias_id)["agentAlias"]["agentAliasArn"]
            specialists[name] = {"agent_id": agent_id, "alias_arn": alias_arn, "description": description}

        # 2. Supervisor
        supervisor_id = self._create_agent(
            "CustomerServiceSupervisor",
            """You are the customer service supervisor. Understand each request, delegate it to the right
specialist (support, billing, technical or sales), and combine their answers into one reply.
Greet customers professionally and clarify their needs before delegating.""",
            SONNET,
            agentCollaboration="SUPERVISOR",
        )

        # 3. Associate collaborators (one call per collaborator)
        for name, spec in specialists.items():
            self.agents.associate_agent_collaborator(
                agentId=supervisor_id,
                agentVersion="DRAFT",
                agentDescriptor={"aliasArn": spec["alias_arn"]},
                collaboratorName=name,
                collaborationInstruction=f"Use {name} when the request concerns: {spec['description'].lower()}.",
                relayConversationHistory="TO_COLLABORATOR",
            )

        supervisor_alias = prepare_and_alias(self.agents, supervisor_id)
        return {
            "supervisor_agent_id": supervisor_id,
            "supervisor_alias_id": supervisor_alias,
            "specialist_agents": {k: v["agent_id"] for k, v in specialists.items()},
        }

    def ask(self, system: Dict, question: str, session_id: str) -> str:
        response = self.runtime.invoke_agent(
            agentId=system["supervisor_agent_id"],
            agentAliasId=system["supervisor_alias_id"],
            sessionId=session_id,
            inputText=question,
        )
        return read_completion(response)
```

---

## Multi-Agent System with A2A Protocol

### Orchestrated Multi-Agent System with Agent-to-Agent Communication

Bedrock Agents (Classic) collaborators talk to each other through the built-in supervisor mechanism above, not the A2A protocol. To use [A2A](https://a2a-protocol.org/) — the open agent-interoperability protocol, which lets agents built in different frameworks call each other — host the agents on **AgentCore Runtime** with the A2A server protocol.

**1. A research agent served over A2A** (`researcher.py`, deployed to AgentCore Runtime with `protocolConfiguration={"serverProtocol": "A2A"}` or `agentcore create` choosing the A2A protocol):

```python
# pip install "bedrock-agentcore[a2a]" "strands-agents[a2a]"
from bedrock_agentcore.runtime import serve_a2a
from strands import Agent
from strands.multiagent.a2a import StrandsA2AExecutor

researcher = Agent(
    model="us.anthropic.claude-sonnet-5-5",
    name="researcher",
    description="Researches a topic and returns sourced bullet points.",
    system_prompt="You are a meticulous researcher. Return concise, sourced bullet points.",
)

serve_a2a(StrandsA2AExecutor(researcher))
```

**2. A writer agent elsewhere that calls the researcher as a remote A2A agent:**

```python
from strands import Agent
from strands.agent.a2a_agent import A2AAgent

remote_researcher = A2AAgent(
    endpoint="https://researcher.example.com/",  # the A2A endpoint of the deployed researcher
    name="researcher",
    description="Researches a topic and returns sourced bullet points.",
)

writer = Agent(
    model="us.anthropic.claude-sonnet-5-5",
    system_prompt="You write clear briefings. Ask the researcher for facts before writing.",
    tools=[remote_researcher],
)

print(writer("Write a one-page briefing on heat-pump adoption in the UK."))
```

Agents on AgentCore Runtime are authenticated (SigV4 or OAuth); configure the A2A client with the matching credentials. The [A2A guide](/amazon-bedrock-agents-guide/python/bedrock_a2a_protocol_guide/) covers authentication, agent cards and the protocol in more detail.

---

## Financial Analytics Agent

### Financial Data Analysis and Reporting

```python
import json

import boto3


class FinancialAnalyticsAgent:
    """Agent for financial data analysis and reporting."""

    def __init__(self, region="us-east-1"):
        self.agents = boto3.client("bedrock-agent", region_name=region)

    def create_financial_agent(self) -> str:
        agent_id = self.agents.create_agent(
            agentName="FinancialAnalytics",
            description="Financial analysis and reporting",
            foundationModel=OPUS,   # complex, multi-step quantitative reasoning
            agentResourceRoleArn=AGENT_ROLE_ARN,
            instruction="""You are a Financial Analytics Agent specialising in analysis, reporting,
trend analysis, anomaly detection and forecasting.

Always:
- Show calculations and cite the data you used
- Highlight significant trends, risks and opportunities
- State assumptions behind any forecast""",
        )["agent"]["agentId"]

        self._create_financial_action_group(agent_id)
        self.agents.associate_agent_knowledge_base(
            agentId=agent_id,
            agentVersion="DRAFT",
            knowledgeBaseId="FINANCE_KB_ID",
            description="Annual reports, accounting policies and board packs.",
        )
        return agent_id

    def _create_financial_action_group(self, agent_id: str) -> None:
        financial_schema = {
            "openapi": "3.0.0",
            "info": {"title": "Financial Data API", "version": "1.0.0"},
            "paths": {
                "/financials/{period}": {
                    "get": {
                        "operationId": "GetFinancialData",
                        "description": "Retrieve financial data for a period such as 2026-Q3",
                        "parameters": [
                            {"name": "period", "in": "path", "required": True, "schema": {"type": "string"}},
                            {"name": "dataType", "in": "query", "schema": {"type": "string",
                                "enum": ["REVENUE", "EXPENSES", "PROFIT", "CASHFLOW"]}},
                        ],
                        "responses": {"200": {"description": "Financial data"}},
                    }
                },
                "/forecasts": {
                    "post": {
                        "operationId": "GenerateForecast",
                        "description": "Generate financial forecasts",
                        "requestBody": {
                            "required": True,
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "historicalDataPeriods": {"type": "integer"},
                                            "forecastMonths": {"type": "integer"},
                                            "confidenceLevel": {"type": "number"},
                                        },
                                    }
                                }
                            },
                        },
                        "responses": {"200": {"description": "Forecast data"}},
                    }
                },
            },
        }

        self.agents.create_agent_action_group(
            agentId=agent_id,
            agentVersion="DRAFT",
            actionGroupName="FinancialDataRetrieval",
            description="Retrieve and analyse financial data",
            apiSchema={"payload": json.dumps(financial_schema)},
            actionGroupExecutor={"lambda": "arn:aws:lambda:us-east-1:111122223333:function:financial-data-handler"},
        )
```

---

## Data Analysis Agent with Code Interpretation

### Agent that uses Code Interpretation for Data Analysis

Code interpretation is enabled by adding the built-in `AMAZON.CodeInterpreter` action group; files are passed in `sessionState`.

```python
import boto3


class DataAnalysisAgent:
    """Agent that writes and runs Python to analyse uploaded data."""

    def __init__(self, region="us-east-1"):
        self.agents = boto3.client("bedrock-agent", region_name=region)
        self.runtime = boto3.client("bedrock-agent-runtime", region_name=region)

    def create_data_analysis_agent(self) -> dict:
        agent_id = self.agents.create_agent(
            agentName="DataAnalysisAgent",
            foundationModel=SONNET,
            agentResourceRoleArn=AGENT_ROLE_ARN,
            instruction="You are a data analyst. Write and run Python code to answer questions about the files you are given.",
        )["agent"]["agentId"]

        self.agents.create_agent_action_group(
            agentId=agent_id,
            agentVersion="DRAFT",
            actionGroupName="CodeInterpreterAction",
            parentActionGroupSignature="AMAZON.CodeInterpreter",
            actionGroupState="ENABLED",
        )
        return {"agent_id": agent_id, "alias_id": prepare_and_alias(self.agents, agent_id)}

    def invoke_data_analysis_agent(self, agent: dict, s3_uri: str) -> str:
        response = self.runtime.invoke_agent(
            agentId=agent["agent_id"],
            agentAliasId=agent["alias_id"],
            sessionId="analysis-session-1",
            inputText="Analyse the attached CSV file and summarise the data.",
            sessionState={
                "files": [
                    {
                        "name": "sales_data.csv",
                        "source": {"sourceType": "S3", "s3Location": {"uri": s3_uri}},
                        "useCase": "CODE_INTERPRETER",
                    }
                ]
            },
        )
        return read_completion(response)
```

For new work, the same capability on AgentCore is the [Code Interpreter](/amazon-bedrock-agents-guide/python/bedrock_agentcore_comprehensive_guide/#code-interpretation) service, which any framework can call.

---

## Document Processing Pipeline

### Automated Document Analysis and Processing

Textract extracts the text; the agent classifies and validates it.

```python
import hashlib

import boto3


class DocumentProcessingAgent:
    """Classifies and validates documents after Textract extraction."""

    def __init__(self, region="us-east-1"):
        self.agents = boto3.client("bedrock-agent", region_name=region)
        self.runtime = boto3.client("bedrock-agent-runtime", region_name=region)
        self.textract = boto3.client("textract", region_name=region)

    def create_document_processing_agent(self) -> dict:
        agent_id = self.agents.create_agent(
            agentName="DocumentProcessing",
            description="Automated document analysis and processing",
            foundationModel=HAIKU,  # high-volume classification and extraction
            agentResourceRoleArn=AGENT_ROLE_ARN,
            instruction="""You are a Document Processing Agent. For each document:
- Classify it as INVOICE, CONTRACT, REPORT or OTHER
- Extract the key fields for that type
- Flag anomalies or missing information
- Reply with a short JSON processing report""",
        )["agent"]["agentId"]
        return {"agent_id": agent_id, "alias_id": prepare_and_alias(self.agents, agent_id)}

    def process(self, agent: dict, bucket: str, key: str) -> str:
        blocks = self.textract.detect_document_text(
            Document={"S3Object": {"Bucket": bucket, "Name": key}}
        )["Blocks"]
        text = "\n".join(b["Text"] for b in blocks if b["BlockType"] == "LINE")

        response = self.runtime.invoke_agent(
            agentId=agent["agent_id"],
            agentAliasId=agent["alias_id"],
            sessionId="doc-" + hashlib.sha256(f"{bucket}/{key}".encode()).hexdigest()[:32],  # [0-9a-zA-Z._:-] only
            inputText=f"Process this document (s3://{bucket}/{key}):\n\n{text[:20000]}",
        )
        return read_completion(response)
```

`detect_document_text` is synchronous and suits single-page images; use `start_document_analysis` (asynchronous) for multi-page PDFs and forms or tables.

---

Each recipe is a starting point: scope the IAM roles to the specific models, Lambdas and knowledge bases used, and add the error handling from the [error handling guide](/amazon-bedrock-agents-guide/python/bedrock_agents_error_handling_python/).
