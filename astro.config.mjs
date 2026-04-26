// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import react from '@astrojs/react';
import tailwindcss from '@tailwindcss/vite';
import remarkBasePath from './src/plugins/remark-base-path.mjs';

const base = '/AgentGuides';

// https://astro.build/config
export default defineConfig({
  site: 'https://codehalwell.github.io',
  base,
  trailingSlash: 'ignore',
  markdown: {
    remarkPlugins: [[remarkBasePath, { base }]],
  },
  integrations: [
    react(),
    starlight({
      title: 'Agent Guides',
      description:
        'Comprehensive, production-ready documentation for building AI agents across 18+ frameworks.',
      logo: {
        src: './src/assets/logo.svg',
        replacesTitle: false,
      },
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/CodeHalwell/AgentGuides',
        },
      ],
      customCss: ['./src/styles/global.css'],
      editLink: {
        baseUrl: 'https://github.com/CodeHalwell/AgentGuides/edit/main/',
      },
      lastUpdated: true,
      pagination: true,
      tableOfContents: { minHeadingLevel: 2, maxHeadingLevel: 4 },
      sidebar: [
        {
          label: 'Start Here',
          items: [
            { label: 'Introduction', link: '/' },
            { label: 'Quick Start', link: '/quick-start/' },
            { label: 'Choose a Framework', link: '/frameworks/' },
            { label: 'All Guides', link: '/guides/' },
          ],
        },
        {
          label: 'Multi-Agent Orchestration',
          collapsed: true,
          items: [
            { label: 'OpenAI Agents SDK (Python)', link: '/openai-agents-sdk-guides/' },
            { label: 'OpenAI Agents SDK (TypeScript)', link: '/openai-agents-sdk-typescript-guide/' },
            { label: 'CrewAI', link: '/crewai-guide/' },
            {
              label: 'AutoGen / AG2',
              collapsed: true,
              items: [
                { label: 'Overview', link: '/autogen-guide/' },
                { label: 'AutoGen / AG2 (Python)', link: '/autogen-guide/python/' },
                { label: 'AG2 (features-focused)', link: '/ag2-guide/' },
              ],
            },
            {
              label: 'LangGraph (Python)',
              collapsed: true,
              items: [
                { label: 'Overview', link: '/langgraph-guide/python/' },
                {
                  label: 'Zero → Hero',
                  collapsed: false,
                  items: [
                    { label: '1 · Setup & core concepts', link: '/langgraph-guide/python/chapter-01-setup-and-core-concepts/' },
                    { label: '2 · Your first agent', link: '/langgraph-guide/python/chapter-02-simple-agents/' },
                    { label: '3 · Multi-agent systems', link: '/langgraph-guide/python/chapter-03-multi-agent/' },
                    { label: '4 · Tools', link: '/langgraph-guide/python/chapter-04-tools/' },
                    { label: '5 · Memory & persistence', link: '/langgraph-guide/python/chapter-05-memory/' },
                    { label: '6 · Streaming & debugging', link: '/langgraph-guide/python/chapter-06-streaming-and-debugging/' },
                    { label: '7 · Human-in-the-loop', link: '/langgraph-guide/python/chapter-07-human-in-the-loop/' },
                    { label: '8 · Middleware', link: '/langgraph-guide/python/chapter-08-middleware-hooks/' },
                    { label: '9 · Advanced patterns', link: '/langgraph-guide/python/chapter-09-advanced-patterns/' },
                    { label: '10 · Production & troubleshooting', link: '/langgraph-guide/python/chapter-10-production/' },
                  ],
                },
                {
                  label: 'Reference',
                  collapsed: true,
                  items: [
                    { label: 'Single-page comprehensive', link: '/langgraph-guide/python/langgraph_comprehensive_guide/' },
                    { label: 'Recipes', link: '/langgraph-guide/python/langgraph_recipes/' },
                    { label: 'Diagrams', link: '/langgraph-guide/python/langgraph_diagrams/' },
                    { label: 'Production guide', link: '/langgraph-guide/python/langgraph_production_guide/' },
                    { label: 'Advanced error recovery', link: '/langgraph-guide/python/langgraph_advanced_error_recovery/' },
                    { label: 'Observability', link: '/langgraph-guide/python/langgraph_observability_python/' },
                    { label: 'Performance optimization', link: '/langgraph-guide/python/langgraph_performance_optimization/' },
                    { label: 'Streaming server (FastAPI)', link: '/langgraph-guide/python/langgraph_streaming_server_fastapi/' },
                  ],
                },
              ],
            },
            { label: 'LangGraph (TypeScript)', link: '/langgraph-guide/typescript/' },
          ],
        },
        {
          label: 'Data & Knowledge',
          collapsed: true,
          items: [
            {
              label: 'LlamaIndex',
              collapsed: true,
              items: [
                { label: 'Overview', link: '/llamaindex-guide/' },
                { label: 'Python', link: '/llamaindex-guide/python/' },
                { label: 'TypeScript', link: '/llamaindex-guide/typescript/' },
              ],
            },
            { label: 'Haystack', link: '/haystack-guide/' },
            { label: 'PydanticAI', link: '/pydanticai-guide/' },
          ],
        },
        {
          label: 'Cloud-Native',
          collapsed: true,
          items: [
            { label: 'Amazon Bedrock Agents', link: '/amazon-bedrock-agents-guide/' },
            {
              label: 'Microsoft Agent Framework',
              collapsed: true,
              items: [
                { label: 'Overview', link: '/microsoft-agent-framework-guide/' },
                {
                  label: 'Python',
                  collapsed: false,
                  items: [
                    { label: 'Overview', link: '/microsoft-agent-framework-guide/python/' },
                    { label: 'Comprehensive guide', link: '/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_comprehensive_guide/' },
                    { label: '2025 features', link: '/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_2025_features/' },
                    { label: 'Advanced', link: '/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_advanced/' },
                    { label: 'Sessions & context providers', link: '/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_sessions/' },
                    { label: 'Functional workflows', link: '/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_functional_workflows/' },
                    { label: 'Recipes', link: '/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_recipes/' },
                    { label: 'Diagrams', link: '/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_diagrams/' },
                    { label: 'Production guide', link: '/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_production_guide/' },
                  ],
                },
                { label: '.NET', link: '/microsoft-agent-framework-guide/dotnet/' },
              ],
            },
            {
              label: 'Google ADK',
              collapsed: true,
              items: [
                { label: 'Overview', link: '/google-adk-guide/' },
                { label: 'Python', link: '/google-adk-guide/python/' },
                { label: 'Go', link: '/google-adk-guide/go/' },
                { label: 'TypeScript', link: '/google-adk-guide/typescript/' },
              ],
            },
          ],
        },
        {
          label: 'Model-Specific',
          collapsed: true,
          items: [
            { label: 'Anthropic Claude SDK (Python)', link: '/anthropic-claude-agent-sdk-guide/' },
            { label: 'Anthropic Claude SDK (TypeScript)', link: '/anthropic-claude-agent-sdk-typescript-guide/' },
            { label: 'Mistral Agents API', link: '/mistral-agents-api-guide/' },
          ],
        },
        {
          label: 'Enterprise & Lightweight',
          collapsed: true,
          items: [
            {
              label: 'Semantic Kernel',
              collapsed: true,
              items: [
                { label: 'Overview', link: '/semantic-kernel-guide/' },
                { label: 'Python', link: '/semantic-kernel-guide/python/' },
                { label: '.NET', link: '/semantic-kernel-guide/dotnet/' },
              ],
            },
            { label: 'SmolAgents', link: '/smolagents-guide/' },
          ],
        },
      ],
    }),
  ],
  vite: {
    plugins: [tailwindcss()],
  },
});
