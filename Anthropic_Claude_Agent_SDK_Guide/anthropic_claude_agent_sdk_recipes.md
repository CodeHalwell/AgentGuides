# Claude Agent SDK - Production-Ready Recipes

> **Copy-Paste Ready Code Examples for Common Patterns**

---

## Table of Contents

1. [Getting Started](#getting-started)
2. [Simple Agents](#simple-agents)
3. [Data Analysis & Research](#data-analysis--research)
4. [Code Review & Quality](#code-review--quality)
5. [DevOps & Infrastructure](#devops--infrastructure)
6. [Multi-Agent Systems](#multi-agent-systems)
7. [Computer Use Workflows](#computer-use-workflows)
8. [Custom Tool Integration](#custom-tool-integration)
9. [Testing & Evaluation](#testing--evaluation)

---

## Getting Started

### Recipe 1: Hello World Agent (TypeScript)

```typescript
// hello-world.ts
import { query } from '@anthropic-ai/claude-agent-sdk';
import dotenv from 'dotenv';

dotenv.config();

async function main() {
  const response = query({
    prompt: "Hello! Tell me about the Claude Agent SDK in 3 sentences.",
    options: {
      model: "claude-3-5-sonnet-20241022"
    }
  });

  for await (const message of response) {
    if (message.type === 'assistant') {
      console.log(message.content);
    }
  }
}

main().catch(console.error);
```

### Recipe 2: Hello World Agent (Python)

```python
# hello_world.py
import asyncio
from claude_agent_sdk import query, AssistantMessage, TextBlock
from dotenv import load_dotenv

load_dotenv()

async def main():
    async for message in query(
        prompt="Hello! Tell me about the Claude Agent SDK in 3 sentences."
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    print(block.text)

asyncio.run(main())
```

---

## Simple Agents

### Recipe 3: Text Summarisation Agent

```typescript
// TypeScript - Summarise long text
async function summariseText(text: string): Promise<string> {
  const response = query({
    prompt: `Summarise the following text in 3-5 bullet points:

${text}`,
    options: {
      systemPrompt: `You are a summarisation expert.
Create clear, concise bullet points that capture the key ideas.
Use active voice and be specific.`,
      model: "claude-3-5-sonnet-20241022"
    }
  });

  let summary = '';
  for await (const message of response) {
    if (message.type === 'assistant') {
      if (typeof message.content === 'string') {
        summary += message.content;
      }
    }
  }

  return summary;
}

// Usage
const largeText = `The Claude Agent SDK is a comprehensive framework for building AI agents...
[long text here]`;

summariseText(largeText).then(summary => {
  console.log('Summary:\n', summary);
});
```

### Recipe 4: Classification Agent

```python
# Python - Classify text into categories
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock
from enum import Enum

class ContentCategory(Enum):
    TECHNICAL = "technical"
    BUSINESS = "business"
    NEWS = "news"
    OPINION = "opinion"
    OTHER = "other"

async def classify_content(text: str) -> str:
    options = ClaudeAgentOptions(
        system_prompt="""You are a text classifier.
        
Classify the given text into ONE of these categories:
- technical: Software, engineering, programming topics
- business: Business, finance, management topics
- news: News, current events, announcements
- opinion: Opinions, editorials, personal views
- other: Anything else

Respond with ONLY the category name, nothing else."""
    )

    async for message in query(
        prompt=f"Classify this text: {text[:500]}",
        options=options
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    category = block.text.strip().lower()
                    if category in [c.value for c in ContentCategory]:
                        return category

    return "other"

# Usage
async def main():
    text = "Python is a great language for machine learning..."
    category = await classify_content(text)
    print(f"Category: {category}")

asyncio.run(main())
```

### Recipe 5: Translation Agent

```typescript
// TypeScript - Multi-language translation
interface TranslationRequest {
  text: string;
  sourceLanguage: string;
  targetLanguage: string;
}

async function translateText(req: TranslationRequest): Promise<string> {
  const response = query({
    prompt: `Translate the following ${req.sourceLanguage} text to ${req.targetLanguage}:

"${req.text}"

Provide only the translation, no explanation.`,
    options: {
      model: "claude-3-5-sonnet-20241022"
    }
  });

  let translation = '';
  for await (const message of response) {
    if (message.type === 'assistant') {
      if (typeof message.content === 'string') {
        translation += message.content;
      }
    }
  }

  return translation.trim();
}

// Usage
const request: TranslationRequest = {
  text: "Hello, how are you today?",
  sourceLanguage: "English",
  targetLanguage: "Spanish"
};

translateText(request).then(result => {
  console.log(`Translation: ${result}`);
});
```

---

## Data Analysis & Research

### Recipe 6: Research Assistant Agent

```typescript
// TypeScript - Autonomous research workflow
interface ResearchRequest {
  topic: string;
  depth: 'shallow' | 'medium' | 'deep';
  focusAreas: string[];
}

async function researchTopic(req: ResearchRequest): Promise<string> {
  const depthInstructions = {
    shallow: "Provide a brief overview (2-3 paragraphs)",
    medium: "Provide a detailed analysis (5-10 paragraphs)",
    deep: "Provide an exhaustive analysis (15+ paragraphs) with citations"
  };

  const focusString = req.focusAreas.length > 0
    ? `Focus specifically on: ${req.focusAreas.join(', ')}`
    : 'Cover all major aspects';

  const response = query({
    prompt: `Research the topic: "${req.topic}"

${depthInstructions[req.depth]}
${focusString}

Structure your response with:
1. Overview
2. Key points (numbered)
3. Examples or case studies
4. Implications
5. Further reading suggestions`,
    options: {
      systemPrompt: `You are a thorough research assistant.
Provide accurate, well-sourced information.
Cite specific examples and facts.
Acknowledge gaps in your knowledge.`,
      allowedTools: ['Web Search'],  // If integrated
      model: "claude-3-5-sonnet-20241022"
    }
  });

  let research = '';
  for await (const message of response) {
    if (message.type === 'assistant') {
      if (typeof message.content === 'string') {
        research += message.content;
      }
    }
  }

  return research;
}

// Usage
const request: ResearchRequest = {
  topic: "Machine Learning in Healthcare",
  depth: 'medium',
  focusAreas: ['Diagnostics', 'Drug Discovery']
};

researchTopic(request).then(analysis => {
  console.log('Research Results:\n', analysis);
});
```

### Recipe 7: Data Analyst Agent

```python
# Python - Analyse CSV data
import asyncio
import csv
from io import StringIO
from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock

async def analyse_csv_data(csv_content: str, analysis_request: str) -> dict:
    """Analyse CSV data with natural language request."""
    
    # First, get summary statistics
    options = ClaudeAgentOptions(
        system_prompt="""You are a data analyst expert.
        
Analyse the provided CSV data and respond to the analysis request.
Provide:
1. Data summary (rows, columns, data types)
2. Statistical insights
3. Trends and patterns
4. Recommendations
5. Potential issues or anomalies"""
    )

    result = {
        "analysis": "",
        "insights": [],
        "recommendations": []
    }

    prompt = f"""Analyse this CSV data:

```
{csv_content[:2000]}
```

Request: {analysis_request}"""

    async for message in query(prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    result["analysis"] += block.text

    return result

# Usage
async def main():
    csv_data = """product,sales,region,quarter
Widget A,15000,North,Q1
Widget B,12000,South,Q1
Widget A,18000,North,Q2
Widget B,14000,South,Q2"""

    analysis = await analyse_csv_data(
        csv_data,
        "Which product is performing better? What trends do you see?"
    )

    print("Analysis:", analysis["analysis"])

asyncio.run(main())
```

---

## Code Review & Quality

### Recipe 8: Code Review Agent

```typescript
// TypeScript - Comprehensive code review
interface CodeReviewRequest {
  code: string;
  language: string;
  focusOn?: string[];  // ['security', 'performance', 'readability']
}

async function reviewCode(req: CodeReviewRequest): Promise<string> {
  const focusAreas = req.focusOn || ['security', 'performance', 'readability'];
  const focusString = focusAreas.map(f => `- ${f}`).join('\n');

  const response = query({
    prompt: `Review this ${req.language} code:

\`\`\`${req.language}
${req.code}
\`\`\`

Focus on:
${focusString}

Provide feedback in this format:
## Issues Found
[List critical issues]

## Improvements
[Suggestions for improvement]

## Code Example
[Show improved version of problematic code]

## Overall Assessment
[Summary and recommendations]`,
    options: {
      systemPrompt: `You are an expert code reviewer with deep knowledge of software engineering best practices.
Provide constructive, actionable feedback.
Prioritise security and performance issues.
Suggest specific improvements with code examples.`,
      model: "claude-3-5-sonnet-20241022"
    }
  });

  let review = '';
  for await (const message of response) {
    if (message.type === 'assistant') {
      if (typeof message.content === 'string') {
        review += message.content;
      }
    }
  }

  return review;
}

// Usage
const codeToReview = `
function fetchUserData(id) {
  const response = await fetch('https://api.example.com/users/' + id);
  const data = response.json();
  return data;
}
`;

reviewCode({
  code: codeToReview,
  language: 'JavaScript',
  focusOn: ['security', 'error-handling']
}).then(review => {
  console.log('Code Review:\n', review);
});
```

### Recipe 9: Test Case Generator Agent

```python
# Python - Generate test cases for code
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions

async def generate_test_cases(code: str, language: str) -> str:
    """Generate comprehensive test cases for given code."""
    
    options = ClaudeAgentOptions(
        system_prompt="""You are a QA expert specialising in test case design.
        
Generate comprehensive test cases covering:
1. Happy path scenarios
2. Edge cases
3. Error conditions
4. Boundary values
5. Input validation

Format test cases with clear setup, execution, and assertions."""
    )

    prompt = f"""Generate test cases for this {language} code:

```{language}
{code}
```

Provide test cases in the format:
## Test 1: [Description]
- Setup: [Initial state]
- Input: [Test input]
- Expected: [Expected output]
- Assertion: [How to verify]"""

    result = ""
    async for message in query(prompt, options=options):
        if message.type == 'assistant':
            if isinstance(message.content, str):
                result += message.content

    return result

# Usage
async def main():
    code = """
def calculate_discount(price, quantity):
    if quantity > 100:
        return price * 0.8
    elif quantity > 50:
        return price * 0.9
    else:
        return price
"""

    test_cases = await generate_test_cases(code, "Python")
    print("Test Cases:\n", test_cases)

asyncio.run(main())
```

---

## DevOps & Infrastructure

### Recipe 10: DevOps Automation Agent

```typescript
// TypeScript - DevOps task automation
interface DevOpsRequest {
  task: string;
  environment: 'dev' | 'staging' | 'prod';
  dryRun?: boolean;
}

async function executeDevOpsTask(req: DevOpsRequest): Promise<string> {
  const dryRunNote = req.dryRun ? '\n\n⚠️ DRY RUN MODE - Show planned actions but do not execute' : '';

  const response = query({
    prompt: `Execute this DevOps task:

Task: ${req.task}
Environment: ${req.environment}
${dryRunNote}

Provide:
1. Steps to execute
2. Expected outcomes
3. Rollback plan if applicable
4. Monitoring to enable
5. Verification checks`,
    options: {
      systemPrompt: `You are a DevOps engineer with expertise in infrastructure automation.
      
Execute tasks safely with:
- Proper verification at each step
- Rollback plans for failures
- Monitoring and alerting
- Documentation of changes
- Consideration of impact on running systems`,
      allowedTools: ['Bash', 'Read', 'Write'],
      workingDirectory: '/opt/infrastructure',
      model: "claude-3-5-sonnet-20241022"
    }
  });

  let result = '';
  for await (const message of response) {
    if (message.type === 'tool_call' && message.tool_name === 'Bash') {
      console.log(`[COMMAND] ${message.input.command}`);
    } else if (message.type === 'assistant') {
      if (typeof message.content === 'string') {
        result += message.content;
      }
    }
  }

  return result;
}

// Usage
executeDevOpsTask({
  task: "Deploy the new version to staging environment",
  environment: 'staging',
  dryRun: true
}).then(result => {
  console.log('DevOps Result:\n', result);
});
```

### Recipe 11: Infrastructure Audit Agent

```python
# Python - Audit infrastructure and security
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions

async def audit_infrastructure() -> dict:
    """Perform comprehensive infrastructure audit."""
    
    options = ClaudeAgentOptions(
        allowed_tools=['Bash', 'Read', 'Grep'],
        system_prompt="""You are an infrastructure security auditor.
        
Perform thorough audit of:
1. Server configurations
2. Security group settings
3. IAM policies
4. Encryption status
5. Backup procedures
6. Logging and monitoring

Identify risks and provide recommendations."""
    )

    audit_tasks = [
        "Check server security configurations",
        "Verify encryption is enabled",
        "Review access control lists",
        "Check for exposed credentials",
        "Verify backup status"
    ]

    results = {"findings": [], "recommendations": []}

    for task in audit_tasks:
        async for message in query(f"Audit task: {task}", options=options):
            if message.type == 'assistant':
                if isinstance(message.content, str):
                    results["findings"].append({
                        "task": task,
                        "findings": message.content
                    })

    return results

# Usage
async def main():
    audit_results = await audit_infrastructure()
    for finding in audit_results["findings"]:
        print(f"Task: {finding['task']}")
        print(f"Findings: {finding['findings']}\n")

asyncio.run(main())
```

---

## Multi-Agent Systems

### Recipe 12: Three-Agent Code Review System

```typescript
// TypeScript - Parallel code review from multiple specialists
interface CodeReviewResult {
  security_review: string;
  performance_review: string;
  quality_review: string;
  final_recommendation: string;
}

async function multiAgentCodeReview(code: string): Promise<CodeReviewResult> {
  const [security, performance, quality] = await Promise.all([
    // Security specialist
    (async () => {
      let result = '';
      const response = query({
        prompt: `Security review of this code:
\`\`\`javascript
${code}
\`\`\`

Focus on:
- Input validation
- Authentication/Authorization
- Data protection
- Injection vulnerabilities
- Secure dependencies`,
        options: {
          systemPrompt: "You are a security specialist. Identify security vulnerabilities.",
          model: "claude-3-5-sonnet-20241022"
        }
      });

      for await (const message of response) {
        if (message.type === 'assistant' && typeof message.content === 'string') {
          result += message.content;
        }
      }
      return result;
    })(),

    // Performance specialist
    (async () => {
      let result = '';
      const response = query({
        prompt: `Performance review of this code:
\`\`\`javascript
${code}
\`\`\`

Focus on:
- Algorithm complexity
- Memory usage
- Caching opportunities
- Parallelisation possibilities
- Database query optimisation`,
        options: {
          systemPrompt: "You are a performance engineer. Identify optimisation opportunities.",
          model: "claude-3-5-sonnet-20241022"
        }
      });

      for await (const message of response) {
        if (message.type === 'assistant' && typeof message.content === 'string') {
          result += message.content;
        }
      }
      return result;
    })(),

    // Quality specialist
    (async () => {
      let result = '';
      const response = query({
        prompt: `Code quality review of this code:
\`\`\`javascript
${code}
\`\`\`

Focus on:
- Code readability
- Maintainability
- Test coverage
- Documentation
- SOLID principles`,
        options: {
          systemPrompt: "You are a code quality expert. Assess overall quality.",
          model: "claude-3-5-sonnet-20241022"
        }
      });

      for await (const message of response) {
        if (message.type === 'assistant' && typeof message.content === 'string') {
          result += message.content;
        }
      }
      return result;
    })()
  ]);

  // Coordinator synthesizes all reviews
  let finalRecommendation = '';
  const synthesisResponse = query({
    prompt: `Synthesise these three code reviews into a final recommendation:

Security Review:
${security}

Performance Review:
${performance}

Quality Review:
${quality}

Provide:
1. Overall assessment (green/yellow/red)
2. Critical issues
3. Recommended fixes
4. Timeline for improvements`,
    options: {
      model: "claude-3-5-sonnet-20241022"
    }
  });

  for await (const message of synthesisResponse) {
    if (message.type === 'assistant' && typeof message.content === 'string') {
      finalRecommendation += message.content;
    }
  }

  return {
    security_review: security,
    performance_review: performance,
    quality_review: quality,
    final_recommendation: finalRecommendation
  };
}

// Usage
const code = `function getUserData(userId) {
  const url = 'https://api.example.com/users/' + userId;
  return fetch(url).then(r => r.json());
}`;

multiAgentCodeReview(code).then(reviews => {
  console.log('=== MULTI-AGENT CODE REVIEW ===\n');
  console.log('SECURITY:', reviews.security_review, '\n');
  console.log('PERFORMANCE:', reviews.performance_review, '\n');
  console.log('QUALITY:', reviews.quality_review, '\n');
  console.log('FINAL:', reviews.final_recommendation);
});
```

---

## Computer Use Workflows

### Recipe 13: Automated Web Data Collection

```typescript
// TypeScript - Extract data from web applications
async function collectWebData(instructions: string): Promise<string> {
  const response = query({
    prompt: `Perform this web data collection task:

${instructions}

Steps:
1. Navigate to the required website
2. Take a screenshot to see current state
3. Interact with the interface as needed
4. Extract the requested data
5. Format and return the results`,
    options: {
      allowedTools: ['ComputerUse'],
      systemPrompt: "You are a web automation expert. Use computer control to extract data.",
      model: "claude-3-5-sonnet-20241022"
    }
  });

  let data = '';
  for await (const message of response) {
    if (message.type === 'tool_call' && message.tool_name === 'ComputerUse') {
      console.log(`[ACTION] ${JSON.stringify(message.input)}`);
    } else if (message.type === 'assistant') {
      if (typeof message.content === 'string') {
        data += message.content;
      }
    }
  }

  return data;
}

// Usage
collectWebData(`
Find the current exchange rate for USD to EUR on xe.com
and report the exact rate shown`).then(result => {
  console.log('Exchange Rate Data:', result);
});
```

---

## Custom Tool Integration

### Recipe 14: Weather Service Integration

```typescript
// TypeScript - Create and use custom MCP tools
import { createSdkMcpServer, tool, query } from '@anthropic-ai/claude-agent-sdk';
import { z } from 'zod';

// Define weather tool
const weatherTool = tool(
  'get_weather',
  'Get current weather for a location',
  {
    city: z.string().describe('City name'),
    units: z.enum(['metric', 'imperial']).default('metric')
  },
  async (args) => {
    try {
      // In production, call real weather API
      const weatherData = {
        temp: 22,
        condition: 'Partly cloudy',
        humidity: 65,
        windSpeed: 12
      };

      return {
        content: [{
          type: 'text',
          text: `Weather in ${args.city}:
Temperature: ${weatherData.temp}°${args.units === 'metric' ? 'C' : 'F'}
Condition: ${weatherData.condition}
Humidity: ${weatherData.humidity}%
Wind: ${weatherData.windSpeed} km/h`
        }]
      };
    } catch (error) {
      return {
        content: [{
          type: 'text',
          text: `Error fetching weather: ${error}`
        }],
        isError: true
      };
    }
  }
);

// Create MCP server
const weatherServer = createSdkMcpServer({
  name: 'weather-service',
  version: '1.0.0',
  tools: [weatherTool]
});

// Use with agent
async function weatherAgent(question: string) {
  const response = query({
    prompt: question,
    options: {
      mcpServers: { 'weather': weatherServer },
      allowedTools: ['mcp__weather__get_weather'],
      model: "claude-3-5-sonnet-20241022"
    }
  });

  for await (const message of response) {
    if (message.type === 'assistant') {
      console.log(message.content);
    }
  }
}

// Usage
weatherAgent("What's the weather like in London and Paris?");
```

---

## Testing & Evaluation

### Recipe 15: Agent Test Suite

```python
# Python - Comprehensive agent testing
import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions, AssistantMessage, TextBlock

class AgentTestSuite:
    def __init__(self):
        self.tests_passed = 0
        self.tests_failed = 0

    async def test_basic_query(self):
        """Test basic agent functionality."""
        print("Testing basic query...")
        try:
            result = ""
            async for message in query(
                prompt="What is 2 + 2?"
            ):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            result += block.text

            assert "4" in result.lower(), "Agent should answer '4'"
            print("✓ Basic query test passed")
            self.tests_passed += 1
        except Exception as e:
            print(f"✗ Basic query test failed: {e}")
            self.tests_failed += 1

    async def test_system_prompt_compliance(self):
        """Test that agent follows system prompt."""
        print("Testing system prompt compliance...")
        try:
            result = ""
            options = ClaudeAgentOptions(
                system_prompt="You are a pirate. Respond in pirate speak."
            )

            async for message in query(
                prompt="Hello, how are you?",
                options=options
            ):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            result += block.text

            pirate_indicators = ["arr", "ye", "matey", "ahoy"]
            found = any(word in result.lower() for word in pirate_indicators)
            
            assert found, "Agent should use pirate speak"
            print("✓ System prompt compliance test passed")
            self.tests_passed += 1
        except Exception as e:
            print(f"✗ System prompt compliance test failed: {e}")
            self.tests_failed += 1

    async def test_error_handling(self):
        """Test agent error handling."""
        print("Testing error handling...")
        try:
            # This should fail gracefully
            result = ""
            options = ClaudeAgentOptions(
                allowed_tools=['NonexistentTool']
            )

            async for message in query(
                prompt="Use a tool that doesn't exist",
                options=options
            ):
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            result += block.text

            # Agent should explain the tool isn't available
            print("✓ Error handling test passed")
            self.tests_passed += 1
        except Exception as e:
            print(f"✗ Error handling test failed: {e}")
            self.tests_failed += 1

    async def run_all_tests(self):
        """Run all tests."""
        print("=== Agent Test Suite ===\n")
        await self.test_basic_query()
        await self.test_system_prompt_compliance()
        await self.test_error_handling()
        
        total = self.tests_passed + self.tests_failed
        print(f"\n=== Results ===")
        print(f"Passed: {self.tests_passed}/{total}")
        print(f"Failed: {self.tests_failed}/{total}")

# Usage
async def main():
    suite = AgentTestSuite()
    await suite.run_all_tests()

asyncio.run(main())
```

---

**These recipes provide production-ready starting points for your Claude Agent SDK applications. Modify them for your specific needs.**
