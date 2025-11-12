# Claude Agent SDK (TypeScript) - Practical Recipes and Examples

## Production-Ready Code Recipes

### Recipe 1: Multi-Turn Code Review Agent

```typescript
import { query } from '@anthropic-ai/claude-agent-sdk';
import fs from 'fs';
import path from 'path';

interface CodeReviewFinding {
  file: string;
  line: number;
  severity: 'critical' | 'high' | 'medium' | 'low';
  category: string;
  description: string;
  suggestion: string;
}

class CodeReviewAgent {
  private sessionId?: string;

  async reviewDirectory(
    dirPath: string,
    filePattern: RegExp = /\.(ts|tsx|js|jsx)$/
  ): Promise<CodeReviewFinding[]> {
    const files = this.getFiles(dirPath, filePattern);

    const findings: CodeReviewFinding[] = [];

    for (const file of files) {
      const content = fs.readFileSync(file, 'utf-8');
      const localFindings = await this.reviewFile(content, file);
      findings.push(...localFindings);
    }

    return findings;
  }

  private async reviewFile(content: string, filePath: string): Promise<CodeReviewFinding[]> {
    const prompt = `Review this TypeScript/JavaScript file for issues:

File: ${filePath}

\`\`\`typescript
${content}
\`\`\`

Analyse for:
1. Type safety issues
2. Performance problems
3. Security vulnerabilities
4. Best practice violations
5. Error handling gaps

Provide findings in JSON format.`;

    const response = query({
      prompt,
      options: {
        model: 'claude-sonnet-4-5',
        resume: this.sessionId,
        forkSession: false
      }
    });

    let output = '';

    for await (const message of response) {
      if (message.type === 'system' && message.subtype === 'init' && message.session_id) {
        this.sessionId = message.session_id;
      }

      if (message.type === 'assistant') {
        output += message.content;
      }
    }

    return this.parseFindings(output, filePath);
  }

  private parseFindings(output: string, filePath: string): CodeReviewFinding[] {
    const jsonMatch = output.match(/```json\n([\s\S]*?)\n```/);

    if (!jsonMatch) {
      return [];
    }

    const data = JSON.parse(jsonMatch[1]);
    const findings: CodeReviewFinding[] = [];

    for (const item of data.findings || []) {
      findings.push({
        file: filePath,
        line: item.line || 1,
        severity: item.severity || 'medium',
        category: item.category || 'general',
        description: item.description,
        suggestion: item.suggestion
      });
    }

    return findings;
  }

  private getFiles(dirPath: string, pattern: RegExp): string[] {
    const files: string[] = [];

    const traverse = (currentPath: string) => {
      const entries = fs.readdirSync(currentPath);

      for (const entry of entries) {
        const fullPath = path.join(currentPath, entry);
        const stat = fs.statSync(fullPath);

        if (stat.isDirectory()) {
          // Skip node_modules and dist
          if (!['node_modules', 'dist', '.git'].includes(entry)) {
            traverse(fullPath);
          }
        } else if (pattern.test(entry)) {
          files.push(fullPath);
        }
      }
    };

    traverse(dirPath);
    return files;
  }
}

// Usage
const reviewer = new CodeReviewAgent();

const findings = await reviewer.reviewDirectory('./src');

console.log('Code Review Findings:');
findings.forEach(finding => {
  console.log(`\n${finding.file}:${finding.line}`);
  console.log(`[${finding.severity.toUpperCase()}] ${finding.category}`);
  console.log(`Issue: ${finding.description}`);
  console.log(`Suggestion: ${finding.suggestion}`);
});
```

### Recipe 2: Research and Analysis Pipeline

```typescript
import { query } from '@anthropic-ai/claude-agent-sdk';

interface ResearchResult {
  topic: string;
  research: string;
  analysis: string;
  synthesis: string;
  recommendations: string[];
  sources: string[];
}

class ResearchPipeline {
  async conduct(topic: string): Promise<ResearchResult> {
    console.log(`Starting research on: ${topic}`);

    // Step 1: Research
    const research = await this.research(topic);

    // Step 2: Analyse
    const analysis = await this.analyse(research);

    // Step 3: Synthesise
    const synthesis = await this.synthesise(research, analysis);

    // Step 4: Extract recommendations
    const recommendations = await this.extractRecommendations(synthesis);

    // Step 5: Find sources
    const sources = await this.findSources(topic);

    return {
      topic,
      research,
      analysis,
      synthesis,
      recommendations,
      sources
    };
  }

  private async research(topic: string): Promise<string> {
    const prompt = `Research the topic: "${topic}"

Gather information about:
- Definition and background
- Current state and trends
- Key players and developments
- Statistics and data points
- Recent news and updates

Provide comprehensive research findings.`;

    const response = query({
      prompt,
      options: {
        model: 'claude-sonnet-4-5',
        temperature: 0.7
      }
    });

    let result = '';

    for await (const message of response) {
      if (message.type === 'assistant') {
        result += message.content;
      }
    }

    return result;
  }

  private async analyse(research: string): Promise<string> {
    const prompt = `Analyse this research data:

${research}

Provide deep analysis covering:
- Key patterns and trends
- Strengths and weaknesses
- Opportunities and threats
- Root causes of observed phenomena
- Predictive insights

Format as structured analysis.`;

    const response = query({
      prompt,
      options: {
        model: 'claude-sonnet-4-5',
        temperature: 0.5
      }
    });

    let result = '';

    for await (const message of response) {
      if (message.type === 'assistant') {
        result += message.content;
      }
    }

    return result;
  }

  private async synthesise(research: string, analysis: string): Promise<string> {
    const prompt = `Synthesise research and analysis:

RESEARCH:
${research}

ANALYSIS:
${analysis}

Create a cohesive synthesis that:
- Connects findings to broader context
- Identifies key insights
- Highlights implications
- Suggests future directions

Provide executive summary style synthesis.`;

    const response = query({
      prompt,
      options: {
        model: 'claude-sonnet-4-5',
        temperature: 0.6
      }
    });

    let result = '';

    for await (const message of response) {
      if (message.type === 'assistant') {
        result += message.content;
      }
    }

    return result;
  }

  private async extractRecommendations(synthesis: string): Promise<string[]> {
    const prompt = `Extract actionable recommendations from this synthesis:

${synthesis}

Provide 5-10 specific, actionable recommendations.
Format as JSON array of strings.`;

    const response = query({
      prompt,
      options: {
        model: 'claude-sonnet-4-5'
      }
    });

    let output = '';

    for await (const message of response) {
      if (message.type === 'assistant') {
        output += message.content;
      }
    }

    const jsonMatch = output.match(/\[[\s\S]*?\]/);
    return jsonMatch ? JSON.parse(jsonMatch[0]) : [];
  }

  private async findSources(topic: string): Promise<string[]> {
    const prompt = `Find authoritative sources on: "${topic}"

Suggest 5-10 credible sources (books, papers, websites, organisations)
Format as JSON array of strings.`;

    const response = query({
      prompt,
      options: {
        model: 'claude-sonnet-4-5'
      }
    });

    let output = '';

    for await (const message of response) {
      if (message.type === 'assistant') {
        output += message.content;
      }
    }

    const jsonMatch = output.match(/\[[\s\S]*?\]/);
    return jsonMatch ? JSON.parse(jsonMatch[0]) : [];
  }
}

// Usage
const pipeline = new ResearchPipeline();

const result = await pipeline.conduct('The Future of TypeScript');

console.log(`\nResearch Results for: ${result.topic}\n`);
console.log(`Research:\n${result.research}\n`);
console.log(`Analysis:\n${result.analysis}\n`);
console.log(`Synthesis:\n${result.synthesis}\n`);
console.log('Recommendations:');
result.recommendations.forEach((rec, i) => {
  console.log(`${i + 1}. ${rec}`);
});
console.log('\nSources:');
result.sources.forEach((source, i) => {
  console.log(`${i + 1}. ${source}`);
});
```

### Recipe 3: Autonomous Testing and QA Agent

```typescript
import { query } from '@anthropic-ai/claude-agent-sdk';
import fs from 'fs';
import path from 'path';

interface TestCase {
  name: string;
  description: string;
  input: Record<string, unknown>;
  expectedOutput: Record<string, unknown>;
  edgeCases: string[];
}

interface TestResult {
  testCase: TestCase;
  passed: boolean;
  actualOutput?: Record<string, unknown>;
  error?: string;
  coverage?: number;
}

class AutomatedQAAgent {
  async analyseCodeAndGenerateTests(codeFile: string): Promise<TestCase[]> {
    const content = fs.readFileSync(codeFile, 'utf-8');

    const prompt = `Analyse this code and generate comprehensive test cases:

\`\`\`typescript
${content}
\`\`\`

Generate test cases covering:
1. Normal operation paths
2. Edge cases and boundary conditions
3. Error handling
4. Performance scenarios
5. Integration points

Format as JSON array with: name, description, input, expectedOutput, edgeCases`;

    const response = query({
      prompt,
      options: {
        model: 'claude-sonnet-4-5'
      }
    });

    let output = '';

    for await (const message of response) {
      if (message.type === 'assistant') {
        output += message.content;
      }
    }

    const jsonMatch = output.match(/```json\n([\s\S]*?)\n```/);
    return jsonMatch ? JSON.parse(jsonMatch[1]) : [];
  }

  async generateTestFile(codeFile: string, testFramework: 'jest' | 'vitest' = 'jest'): Promise<string> {
    const content = fs.readFileSync(codeFile, 'utf-8');
    const testCases = await this.analyseCodeAndGenerateTests(codeFile);

    const prompt = `Generate ${testFramework} test code for these test cases:

Original Code:
\`\`\`typescript
${content}
\`\`\`

Test Cases:
${JSON.stringify(testCases, null, 2)}

Generate complete, runnable ${testFramework} test file with:
- All test cases implemented
- Setup and teardown
- Mock implementations where needed
- Assertions for all scenarios`;

    const response = query({
      prompt,
      options: {
        model: 'claude-sonnet-4-5',
        temperature: 0.5
      }
    });

    let testCode = '';

    for await (const message of response) {
      if (message.type === 'assistant') {
        testCode += message.content;
      }
    }

    // Extract code block
    const codeMatch = testCode.match(/```typescript\n([\s\S]*?)\n```/);
    return codeMatch ? codeMatch[1] : testCode;
  }

  async assessTestCoverage(codeFile: string): Promise<{
    overallCoverage: number;
    lineCoverage: number;
    branchCoverage: number;
    functionCoverage: number;
    recommendations: string[];
  }> {
    const content = fs.readFileSync(codeFile, 'utf-8');

    const prompt = `Assess test coverage needs for this code:

\`\`\`typescript
${content}
\`\`\`

Analyse:
1. Lines that should be tested
2. Branches and conditionals
3. Functions and methods
4. Edge cases
5. Error paths

Provide coverage assessment with recommendations in JSON format.`;

    const response = query({
      prompt,
      options: {
        model: 'claude-sonnet-4-5'
      }
    });

    let output = '';

    for await (const message of response) {
      if (message.type === 'assistant') {
        output += message.content;
      }
    }

    const jsonMatch = output.match(/```json\n([\s\S]*?)\n```/);

    if (jsonMatch) {
      return JSON.parse(jsonMatch[1]);
    }

    return {
      overallCoverage: 0,
      lineCoverage: 0,
      branchCoverage: 0,
      functionCoverage: 0,
      recommendations: []
    };
  }
}

// Usage
const qaAgent = new AutomatedQAAgent();

const codeFile = './src/utils/calculator.ts';

// Generate test cases
const testCases = await qaAgent.analyseCodeAndGenerateTests(codeFile);
console.log('Generated test cases:', testCases);

// Generate test file
const testCode = await qaAgent.generateTestFile(codeFile, 'jest');
fs.writeFileSync('./src/utils/calculator.test.ts', testCode);
console.log('Test file generated');

// Assess coverage
const coverage = await qaAgent.assessTestCoverage(codeFile);
console.log('Coverage assessment:', coverage);
```

### Recipe 4: Documentation Auto-Generator

```typescript
import { query } from '@anthropic-ai/claude-agent-sdk';
import fs from 'fs';
import path from 'path';

interface DocSection {
  name: string;
  content: string;
  examples: string[];
  relatedSections: string[];
}

class DocumentationGenerator {
  async generateFromCode(codeFile: string): Promise<string> {
    const content = fs.readFileSync(codeFile, 'utf-8');
    const fileName = path.basename(codeFile);

    const prompt = `Generate comprehensive documentation for this code file (${fileName}):

\`\`\`typescript
${content}
\`\`\`

Documentation should include:
1. Overview and purpose
2. Installation/import instructions
3. API reference with all functions/classes
4. Parameter descriptions and types
5. Return values
6. Usage examples
7. Error handling
8. Best practices
9. Common pitfalls
10. Related documentation

Format as Markdown.`;

    const response = query({
      prompt,
      options: {
        model: 'claude-sonnet-4-5',
        temperature: 0.5
      }
    });

    let documentation = '';

    for await (const message of response) {
      if (message.type === 'assistant') {
        documentation += message.content;
      }
    }

    return documentation;
  }

  async generateAPIReference(codeFile: string): Promise<string> {
    const content = fs.readFileSync(codeFile, 'utf-8');

    const prompt = `Generate an API reference for this code file:

\`\`\`typescript
${content}
\`\`\`

For each exported function, class, or interface, provide:
- Name and signature
- Description
- Parameters with types
- Return type
- Exceptions/errors
- Example usage

Format as detailed Markdown API reference.`;

    const response = query({
      prompt,
      options: {
        model: 'claude-sonnet-4-5'
      }
    });

    let reference = '';

    for await (const message of response) {
      if (message.type === 'assistant') {
        reference += message.content;
      }
    }

    return reference;
  }

  async generateUsageExamples(codeFile: string): Promise<string> {
    const content = fs.readFileSync(codeFile, 'utf-8');

    const prompt = `Generate comprehensive usage examples for this code:

\`\`\`typescript
${content}
\`\`\`

Create examples covering:
1. Basic usage
2. Advanced patterns
3. Error handling
4. Integration with other modules
5. Performance considerations
6. Common use cases

Format with code blocks and explanations.`;

    const response = query({
      prompt,
      options: {
        model: 'claude-sonnet-4-5'
      }
    });

    let examples = '';

    for await (const message of response) {
      if (message.type === 'assistant') {
        examples += message.content;
      }
    }

    return examples;
  }

  async generateCompleteDocumentation(projectDir: string): Promise<void> {
    const files = this.findTypeScriptFiles(projectDir);

    console.log(`Generating documentation for ${files.length} files...`);

    for (const file of files) {
      const relativePath = path.relative(projectDir, file);
      const docFileName = `${path.basename(file, '.ts')}.md`;
      const docPath = path.join(projectDir, 'docs', docFileName);

      console.log(`Processing: ${relativePath}`);

      const documentation = await this.generateFromCode(file);

      // Ensure docs directory exists
      const docsDir = path.dirname(docPath);
      if (!fs.existsSync(docsDir)) {
        fs.mkdirSync(docsDir, { recursive: true });
      }

      fs.writeFileSync(docPath, documentation);
      console.log(`Documentation saved: ${docPath}`);
    }
  }

  private findTypeScriptFiles(dir: string): string[] {
    const files: string[] = [];

    const traverse = (currentPath: string) => {
      const entries = fs.readdirSync(currentPath);

      for (const entry of entries) {
        const fullPath = path.join(currentPath, entry);
        const stat = fs.statSync(fullPath);

        if (stat.isDirectory()) {
          if (!['node_modules', 'dist', '.git', 'docs'].includes(entry)) {
            traverse(fullPath);
          }
        } else if (entry.endsWith('.ts') && !entry.endsWith('.test.ts')) {
          files.push(fullPath);
        }
      }
    };

    traverse(dir);
    return files;
  }
}

// Usage
const docGenerator = new DocumentationGenerator();

// Generate docs for single file
const docs = await docGenerator.generateFromCode('./src/utils/auth.ts');
fs.writeFileSync('./docs/auth.md', docs);

// Generate API reference
const apiRef = await docGenerator.generateAPIReference('./src/services/UserService.ts');
fs.writeFileSync('./docs/api/UserService.md', apiRef);

// Generate for entire project
await docGenerator.generateCompleteDocumentation('./src');
```

### Recipe 5: Performance Analysis and Optimisation Agent

```typescript
import { query } from '@anthropic-ai/claude-agent-sdk';
import fs from 'fs';

interface PerformanceIssue {
  type: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  location: string;
  description: string;
  impact: string;
  solution: string;
  example: string;
}

class PerformanceAnalysisAgent {
  async analysePerformance(codeFile: string): Promise<PerformanceIssue[]> {
    const content = fs.readFileSync(codeFile, 'utf-8');

    const prompt = `Analyse this code for performance issues:

\`\`\`typescript
${content}
\`\`\`

Identify:
1. Algorithm inefficiencies
2. N+1 query problems
3. Memory leaks
4. Unnecessary re-renders (React)
5. Synchronous operations that should be async
6. Missing caching opportunities
7. Inefficient loops or iterations
8. Large bundle size issues

For each issue, provide: type, severity, location, description, impact, solution, example

Format as JSON array.`;

    const response = query({
      prompt,
      options: {
        model: 'claude-sonnet-4-5',
        temperature: 0.5
      }
    });

    let output = '';

    for await (const message of response) {
      if (message.type === 'assistant') {
        output += message.content;
      }
    }

    const jsonMatch = output.match(/```json\n([\s\S]*?)\n```/);
    return jsonMatch ? JSON.parse(jsonMatch[1]) : [];
  }

  async suggestOptimisations(codeFile: string): Promise<string> {
    const content = fs.readFileSync(codeFile, 'utf-8');
    const issues = await this.analysePerformance(codeFile);

    const prompt = `Based on these performance issues:

${JSON.stringify(issues, null, 2)}

For the code:
\`\`\`typescript
${content}
\`\`\`

Generate optimised version of the code addressing all issues.
Provide refactored code with explanations of optimisations.`;

    const response = query({
      prompt,
      options: {
        model: 'claude-sonnet-4-5',
        temperature: 0.3
      }
    });

    let optimisedCode = '';

    for await (const message of response) {
      if (message.type === 'assistant') {
        optimisedCode += message.content;
      }
    }

    return optimisedCode;
  }

  async estimatePerformanceGain(originalFile: string, optimisedFile: string): Promise<{
    estimatedSpeedup: string;
    memoryImprovement: string;
    cpuReduction: string;
    recommendations: string[];
  }> {
    const original = fs.readFileSync(originalFile, 'utf-8');
    const optimised = fs.readFileSync(optimisedFile, 'utf-8');

    const prompt = `Compare performance impact of these versions:

ORIGINAL:
\`\`\`typescript
${original}
\`\`\`

OPTIMISED:
\`\`\`typescript
${optimised}
\`\`\`

Estimate performance gains in terms of:
- Execution speed (X times faster)
- Memory usage (percentage reduction)
- CPU usage (percentage reduction)
- Additional benefits

Format as JSON.`;

    const response = query({
      prompt,
      options: {
        model: 'claude-sonnet-4-5'
      }
    });

    let output = '';

    for await (const message of response) {
      if (message.type === 'assistant') {
        output += message.content;
      }
    }

    const jsonMatch = output.match(/```json\n([\s\S]*?)\n```/);

    if (jsonMatch) {
      return JSON.parse(jsonMatch[1]);
    }

    return {
      estimatedSpeedup: 'unknown',
      memoryImprovement: 'unknown',
      cpuReduction: 'unknown',
      recommendations: []
    };
  }
}

// Usage
const perfAgent = new PerformanceAnalysisAgent();

const issues = await perfAgent.analysePerformance('./src/data-processor.ts');
console.log('Performance Issues:');
issues.forEach(issue => {
  console.log(`\n[${issue.severity.toUpperCase()}] ${issue.type}`);
  console.log(`Description: ${issue.description}`);
  console.log(`Impact: ${issue.impact}`);
  console.log(`Solution: ${issue.solution}`);
});

const optimised = await perfAgent.suggestOptimisations('./src/data-processor.ts');
fs.writeFileSync('./src/data-processor.optimised.ts', optimised);

const gains = await perfAgent.estimatePerformanceGain(
  './src/data-processor.ts',
  './src/data-processor.optimised.ts'
);
console.log('\nEstimated Performance Gains:', gains);
```

### Recipe 6: Security Audit Agent

```typescript
import { query } from '@anthropic-ai/claude-agent-sdk';
import fs from 'fs';

interface SecurityFinding {
  type: string;
  severity: 'critical' | 'high' | 'medium' | 'low';
  cwe: string;
  description: string;
  remediation: string;
  references: string[];
}

class SecurityAuditAgent {
  async auditCode(codeFile: string): Promise<SecurityFinding[]> {
    const content = fs.readFileSync(codeFile, 'utf-8');

    const prompt = `Perform comprehensive security audit of this code:

\`\`\`typescript
${content}
\`\`\`

Check for:
1. Injection vulnerabilities (SQL, command, code)
2. Broken authentication
3. Sensitive data exposure
4. XML external entity (XXE)
5. Broken access control
6. Security misconfiguration
7. Cross-site scripting (XSS)
8. Insecure deserialization
9. Using components with known vulnerabilities
10. Insufficient logging and monitoring

For each finding: type, severity, CWE, description, remediation, references

Format as JSON array.`;

    const response = query({
      prompt,
      options: {
        model: 'claude-sonnet-4-5',
        temperature: 0.3
      }
    });

    let output = '';

    for await (const message of response) {
      if (message.type === 'assistant') {
        output += message.content;
      }
    }

    const jsonMatch = output.match(/```json\n([\s\S]*?)\n```/);
    return jsonMatch ? JSON.parse(jsonMatch[1]) : [];
  }

  async auditDependencies(packageJsonFile: string): Promise<{
    vulnerablePackages: Array<{ name: string; version: string; vulnerability: string; severity: string }>;
    recommendations: string[];
  }> {
    const content = fs.readFileSync(packageJsonFile, 'utf-8');

    const prompt = `Analyse this package.json for security vulnerabilities:

\`\`\`json
${content}
\`\`\`

Check for:
1. Known vulnerable package versions
2. Deprecated packages
3. Packages with abandoned maintenance
4. Packages with security issues
5. Dependency conflicts
6. Out-of-date packages

Format as JSON with vulnerablePackages array and recommendations.`;

    const response = query({
      prompt,
      options: {
        model: 'claude-sonnet-4-5'
      }
    });

    let output = '';

    for await (const message of response) {
      if (message.type === 'assistant') {
        output += message.content;
      }
    }

    const jsonMatch = output.match(/```json\n([\s\S]*?)\n```/);

    if (jsonMatch) {
      return JSON.parse(jsonMatch[1]);
    }

    return {
      vulnerablePackages: [],
      recommendations: []
    };
  }

  async generateSecurityReport(projectDir: string): Promise<string> {
    const findings: SecurityFinding[] = [];

    const tsFiles = this.findTypeScriptFiles(projectDir);

    for (const file of tsFiles) {
      const fileFindings = await this.auditCode(file);
      findings.push(...fileFindings);
    }

    const packageJsonPath = `${projectDir}/package.json`;
    const depVulnerabilities = fs.existsSync(packageJsonPath)
      ? await this.auditDependencies(packageJsonPath)
      : { vulnerablePackages: [], recommendations: [] };

    const prompt = `Generate security audit report:

Code Findings: ${JSON.stringify(findings, null, 2)}
Dependency Issues: ${JSON.stringify(depVulnerabilities, null, 2)}

Create executive summary with:
1. Critical issues needing immediate attention
2. High priority items
3. Medium priority improvements
4. Low priority recommendations
5. Compliance considerations
6. Action plan with timeline

Format as detailed Markdown report.`;

    const response = query({
      prompt,
      options: {
        model: 'claude-sonnet-4-5'
      }
    });

    let report = '';

    for await (const message of response) {
      if (message.type === 'assistant') {
        report += message.content;
      }
    }

    return report;
  }

  private findTypeScriptFiles(dir: string): string[] {
    const files: string[] = [];

    const traverse = (currentPath: string) => {
      const entries = fs.readdirSync(currentPath);

      for (const entry of entries) {
        const fullPath = `${currentPath}/${entry}`;
        const stat = fs.statSync(fullPath);

        if (stat.isDirectory()) {
          if (!['node_modules', 'dist', '.git'].includes(entry)) {
            traverse(fullPath);
          }
        } else if (entry.endsWith('.ts')) {
          files.push(fullPath);
        }
      }
    };

    traverse(dir);
    return files;
  }
}

// Usage
const securityAudit = new SecurityAuditAgent();

const findings = await securityAudit.auditCode('./src/api/endpoints.ts');
console.log('Security Findings:', findings);

const report = await securityAudit.generateSecurityReport('./src');
fs.writeFileSync('./security-report.md', report);
console.log('Security report generated');
```

These recipes provide production-ready patterns for common agent use cases. Each can be customised and extended for specific requirements.
