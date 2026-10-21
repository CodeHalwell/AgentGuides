import versions from './versions.json';

export type Framework = {
  id: string;
  name: string;
  /** Two-letter element symbol for the periodic-table tile. */
  symbol: string;
  tagline: string;
  href: string;
  languages: Array<'Python' | 'TypeScript' | '.NET' | 'Go'>;
  category:
    | 'Multi-Agent'
    | 'Data & RAG'
    | 'Cloud-Native'
    | 'Model-Specific'
    | 'Enterprise'
    | 'Lightweight';
  difficulty: 'Beginner' | 'Intermediate' | 'Advanced';
  useCases: string[];
  version?: string;
};

/**
 * Periodic groups. Five hues plus ink, per the brand: each category has its own
 * colour, and cobalt is reserved for languages.
 */
export const categoryGroup: Record<Framework['category'], 'oxide' | 'verdigris' | 'ink' | 'permanganate' | 'sulfur'> = {
  'Multi-Agent': 'oxide',
  'Data & RAG': 'verdigris',
  'Cloud-Native': 'ink',
  Enterprise: 'ink',
  'Model-Specific': 'permanganate',
  Lightweight: 'sulfur',
};

export const languageSymbol: Record<Framework['languages'][number], string> = {
  Python: 'Py',
  TypeScript: 'Ts',
  '.NET': 'C#',
  Go: 'Go',
};

/** Current version of a package, from versions.json (the single source of truth). */
const v = (pkg: keyof typeof versions.packages): string => `v${versions.packages[pkg].version}`;

export const frameworks: Framework[] = [
  {
    id: 'smolagents',
    symbol: 'Sa',
    name: 'SmolAgents',
    tagline: 'Minimalist, code-first agents.',
    href: '/smolagents-guide/',
    languages: ['Python'],
    category: 'Lightweight',
    difficulty: 'Beginner',
    useCases: ['chatbot', 'automation', 'prototyping'],
    version: v('smolagents'),
  },
  {
    id: 'pydanticai',
    symbol: 'Pa',
    name: 'PydanticAI',
    tagline: 'Type-safe agents with validated outputs and graphs.',
    href: '/pydanticai-guide/',
    languages: ['Python'],
    category: 'Lightweight',
    difficulty: 'Beginner',
    useCases: ['typed-apis', 'structured-output', 'validation'],
    version: v('pydantic-ai'),
  },
  {
    id: 'openai-agents',
    symbol: 'Oa',
    name: 'OpenAI Agents SDK',
    tagline: 'Production Swarm replacement — handoffs, guardrails, sessions.',
    href: '/openai-agents-sdk-guides/',
    languages: ['Python', 'TypeScript'],
    category: 'Multi-Agent',
    difficulty: 'Intermediate',
    useCases: ['chatbot', 'multi-agent', 'customer-service'],
    version: v('openai-agents'),
  },
  {
    id: 'openai-agents-ts',
    symbol: 'Ot',
    name: 'OpenAI Agents SDK (TypeScript)',
    tagline: 'The TS port with HITL approvals and browser support.',
    href: '/openai-agents-sdk-typescript-guide/',
    languages: ['TypeScript'],
    category: 'Multi-Agent',
    difficulty: 'Intermediate',
    useCases: ['chatbot', 'multi-agent', 'web'],
    version: v('@openai/agents'),
  },
  {
    id: 'crewai',
    symbol: 'Cr',
    name: 'CrewAI',
    tagline: 'Role-based agent teams with Flows and checkpoints.',
    href: '/crewai-guide/',
    languages: ['Python'],
    category: 'Multi-Agent',
    difficulty: 'Intermediate',
    useCases: ['team-simulation', 'customer-service', 'orchestration'],
    version: v('crewai'),
  },
  {
    id: 'autogen',
    symbol: 'Au',
    name: 'AutoGen / AG2',
    tagline: 'Research-grade multi-agent with GroupChat and A2A.',
    href: '/autogen-guide/',
    languages: ['Python'],
    category: 'Multi-Agent',
    difficulty: 'Advanced',
    useCases: ['research', 'multi-agent', 'experimentation'],
    version: `AG2 ${v('ag2')} / AutoGen ${v('autogen-agentchat')}`,
  },
  {
    id: 'langgraph-py',
    symbol: 'Lg',
    name: 'LangGraph (Python)',
    tagline: 'Graph-based state machines with type-safe v2 API.',
    href: '/langgraph-guide/python/',
    languages: ['Python'],
    category: 'Multi-Agent',
    difficulty: 'Advanced',
    useCases: ['workflows', 'cycles', 'conditional-routing'],
    version: v('langgraph'),
  },
  {
    id: 'langgraph-ts',
    symbol: 'Lt',
    name: 'LangGraph (TypeScript)',
    tagline: 'Type-safe streaming workflows for the TS stack.',
    href: '/langgraph-guide/typescript/',
    languages: ['TypeScript'],
    category: 'Multi-Agent',
    difficulty: 'Advanced',
    useCases: ['workflows', 'cycles', 'web'],
    version: v('@langchain/langgraph'),
  },
  {
    id: 'llamaindex-py',
    symbol: 'Li',
    name: 'LlamaIndex (Python)',
    tagline: 'RAG + AgentWorkflow 1.0. The knowledge-retrieval default.',
    href: '/llamaindex-guide/python/',
    languages: ['Python'],
    category: 'Data & RAG',
    difficulty: 'Intermediate',
    useCases: ['rag', 'search', 'knowledge-retrieval'],
    version: v('llama-index'),
  },
  {
    id: 'llamaindex-ts',
    symbol: 'Lj',
    name: 'LlamaIndex (TypeScript)',
    tagline: 'Deprecated upstream; final release 0.12.1. Kept for existing projects.',
    href: '/llamaindex-guide/typescript/',
    languages: ['TypeScript'],
    category: 'Data & RAG',
    difficulty: 'Intermediate',
    useCases: ['rag', 'search', 'web'],
    version: v('llamaindex'),
  },
  {
    id: 'haystack',
    symbol: 'Hs',
    name: 'Haystack',
    tagline: 'Pipelines and agents for production RAG and search.',
    href: '/haystack-guide/',
    languages: ['Python'],
    category: 'Data & RAG',
    difficulty: 'Advanced',
    useCases: ['search', 'qa', 'rag'],
    version: v('haystack-ai'),
  },
  {
    id: 'bedrock',
    symbol: 'Br',
    name: 'Amazon Bedrock Agents',
    tagline: 'Multi-agent collaboration GA, AgentCore, AWS-native.',
    href: '/amazon-bedrock-agents-guide/',
    languages: ['Python', 'TypeScript'],
    category: 'Cloud-Native',
    difficulty: 'Intermediate',
    useCases: ['aws', 'multi-agent', 'managed'],
    version: `Strands ${v('strands-agents')} / AgentCore ${v('bedrock-agentcore')} (Python)`,
  },
  {
    id: 'ms-agent-fx',
    symbol: 'Mf',
    name: 'Microsoft Agent Framework',
    tagline: 'Unifies Semantic Kernel + AutoGen. A2A, graphs, declarative.',
    href: '/microsoft-agent-framework-guide/',
    languages: ['Python', '.NET', 'Go'],
    category: 'Cloud-Native',
    difficulty: 'Intermediate',
    useCases: ['azure', 'enterprise', 'a2a'],
    version: `Python ${v('agent-framework')} / .NET ${v('Microsoft.Agents.AI')} / Go ${v('github.com/microsoft/agent-framework-go')}`,
  },
  {
    id: 'google-adk',
    symbol: 'Gk',
    name: 'Google ADK',
    tagline: 'Gemini-first agents with multi-language SDKs.',
    href: '/google-adk-guide/',
    languages: ['Python', 'TypeScript', 'Go'],
    category: 'Cloud-Native',
    difficulty: 'Intermediate',
    useCases: ['gcp', 'gemini', 'multi-agent'],
    version: `Python ${v('google-adk')} / TS ${v('@google/adk')} / Go ${v('google.golang.org/adk')}`,
  },
  {
    id: 'claude-sdk',
    symbol: 'Cl',
    name: 'Anthropic Claude Agent SDK',
    tagline: 'Subagents, hooks, checkpointing and session rewind.',
    href: '/anthropic-claude-agent-sdk-guide/',
    languages: ['Python'],
    category: 'Model-Specific',
    difficulty: 'Intermediate',
    useCases: ['claude', 'multi-agent', 'hooks'],
    version: v('claude-agent-sdk'),
  },
  {
    id: 'claude-sdk-ts',
    symbol: 'Ct',
    name: 'Anthropic Claude SDK (TypeScript)',
    tagline: 'The TS counterpart with the same primitives.',
    href: '/anthropic-claude-agent-sdk-typescript-guide/',
    languages: ['TypeScript'],
    category: 'Model-Specific',
    difficulty: 'Intermediate',
    useCases: ['claude', 'web', 'node'],
    version: v('@anthropic-ai/claude-agent-sdk'),
  },
  {
    id: 'mistral',
    symbol: 'Mi',
    name: 'Mistral Agents API',
    tagline: 'Hosted agents with built-in tools, MCP connectors and handoffs.',
    href: '/mistral-agents-api-guide/',
    languages: ['Python', 'TypeScript'],
    category: 'Model-Specific',
    difficulty: 'Intermediate',
    useCases: ['mistral', 'connectors', 'eu-hosted'],
    version: `Python ${v('mistralai')} / TS ${v('@mistralai/mistralai')}`,
  },
  {
    id: 'semantic-kernel',
    symbol: 'Sk',
    name: 'Semantic Kernel',
    tagline: 'Microsoft plugin architecture — MCP and A2A native.',
    href: '/semantic-kernel-guide/',
    languages: ['Python', '.NET'],
    category: 'Enterprise',
    difficulty: 'Intermediate',
    useCases: ['enterprise', 'microsoft', 'plugins'],
    version: `Python ${v('semantic-kernel')} / .NET ${v('Microsoft.SemanticKernel')}`,
  },
];

export const useCaseLabels: Record<string, string> = {
  chatbot: 'A chatbot or assistant',
  'multi-agent': 'A multi-agent system',
  'customer-service': 'Customer service + handoffs',
  rag: 'A RAG / knowledge system',
  search: 'A search or Q&A system',
  workflows: 'A complex workflow with cycles',
  research: 'Research / experimentation',
  automation: 'Simple automation',
  'team-simulation': 'A team-of-experts simulation',
  aws: 'AWS-native deployment',
  azure: 'Azure-native deployment',
  gcp: 'GCP-native deployment',
  claude: 'A Claude-specific app',
  mistral: 'A Mistral-specific app',
  typed: 'Type-safe, validated outputs',
  web: 'A browser / Node app',
};
