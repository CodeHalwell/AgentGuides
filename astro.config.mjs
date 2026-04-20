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
            { label: 'AutoGen / AG2', link: '/autogen-guide/' },
            { label: 'LangGraph (Python)', link: '/langgraph-guide/python/' },
            { label: 'LangGraph (TypeScript)', link: '/langgraph-guide/typescript/' },
          ],
        },
        {
          label: 'Data & Knowledge',
          collapsed: true,
          items: [
            { label: 'LlamaIndex (Python)', link: '/llamaindex-guide/python/' },
            { label: 'LlamaIndex (TypeScript)', link: '/llamaindex-guide/typescript/' },
            { label: 'Haystack', link: '/haystack-guide/' },
            { label: 'PydanticAI', link: '/pydanticai-guide/' },
          ],
        },
        {
          label: 'Cloud-Native',
          collapsed: true,
          items: [
            { label: 'Amazon Bedrock Agents', link: '/amazon-bedrock-agents-guide/' },
            { label: 'Microsoft Agent Framework', link: '/microsoft-agent-framework-guide/' },
            { label: 'Google ADK', link: '/google-adk-guide/' },
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
            { label: 'Semantic Kernel', link: '/semantic-kernel-guide/' },
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
