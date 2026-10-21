// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import react from '@astrojs/react';
import tailwindcss from '@tailwindcss/vite';
import remarkBasePath from './src/plugins/remark-base-path.mjs';
import remarkVersions from './src/plugins/remark-versions.mjs';
import { readFileSync } from 'node:fs';
import { ExpressiveCodeTheme } from '@astrojs/starlight/expressive-code';

// Code panels are ink in both themes, so one dark theme serves both.
const codeTheme = ExpressiveCodeTheme.fromJSONString(
  readFileSync(new URL('./src/styles/code-theme.json', import.meta.url), 'utf8'),
);

// 3px reading-progress bar under the header on article pages.
const readingProgress = `
document.addEventListener('DOMContentLoaded', () => {
  if (!document.querySelector('.sl-markdown-content') || document.querySelector('.hero')) return;
  const bar = document.createElement('div');
  bar.id = 'reading-progress';
  bar.setAttribute('aria-hidden', 'true');
  const fill = document.createElement('span');
  bar.append(fill);
  document.body.append(bar);
  const update = () => {
    const max = document.documentElement.scrollHeight - innerHeight;
    fill.style.width = (max > 0 ? Math.min(100, (scrollY / max) * 100) : 0) + '%';
  };
  addEventListener('scroll', update, { passive: true });
  update();
});`;

const base = '/AgentGuides';

// https://astro.build/config
export default defineConfig({
  site: 'https://codehalwell.github.io',
  base,
  trailingSlash: 'ignore',
  // Retired "class deep dive" pages; their content now lives in the guide pages.
  redirects: {
    '/google-adk-guide/python/google_adk_class_deep_dives_v2_8': '/AgentGuides/google-adk-guide/python/tools/',
    '/google-adk-guide/python/google_adk_class_deep_dives_vol3': '/AgentGuides/google-adk-guide/python/callbacks-and-plugins/',
    '/google-adk-guide/python/google_adk_class_deep_dives_vol4': '/AgentGuides/google-adk-guide/python/google_adk_comprehensive_guide/',
    '/google-adk-guide/python/google_adk_class_deep_dives_vol5': '/AgentGuides/google-adk-guide/python/runner-and-sessions/',
    '/google-adk-guide/python/google_adk_class_deep_dives_vol6': '/AgentGuides/google-adk-guide/python/workflows/',
    '/google-adk-guide/python/google_adk_class_deep_dives_vol7': '/AgentGuides/google-adk-guide/python/google_adk_comprehensive_guide/',
    '/langgraph-guide/python/reference-class-deep-dives-vol1': '/AgentGuides/langgraph-guide/python/reference-command-and-send/',
    '/langgraph-guide/python/reference-class-deep-dives-vol2': '/AgentGuides/langgraph-guide/python/reference-channels/',
    '/langgraph-guide/python/reference-class-deep-dives-vol3': '/AgentGuides/langgraph-guide/python/reference-channels/',
    '/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives': '/AgentGuides/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_harness/',
    '/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v2': '/AgentGuides/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_comprehensive_guide/',
    '/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v3': '/AgentGuides/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_compaction/',
    '/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v4': '/AgentGuides/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_comprehensive_guide/',
    '/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v5': '/AgentGuides/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_sessions/',
    '/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v6': '/AgentGuides/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_comprehensive_guide/',
    '/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v7': '/AgentGuides/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_checkpointing/',
    '/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v8': '/AgentGuides/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_skills/',
    '/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_class_deep_dives_v9': '/AgentGuides/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_sessions/',
    '/pydanticai-guide/pydantic_ai_class_deep_dives_v2_36': '/AgentGuides/pydanticai-guide/pydantic_ai_comprehensive_guide/',
    '/pydanticai-guide/pydantic_ai_class_deep_dives_v2_40': '/AgentGuides/pydanticai-guide/pydantic_ai_comprehensive_guide/',
    '/pydanticai-guide/pydantic_ai_class_deep_dives_v2_43': '/AgentGuides/pydanticai-guide/pydantic_ai_comprehensive_guide/',
    '/pydanticai-guide/pydantic_ai_class_deep_dives_v2_46': '/AgentGuides/pydanticai-guide/pydantic_ai_comprehensive_guide/',
    '/pydanticai-guide/pydantic_ai_class_deep_dives_v2_51': '/AgentGuides/pydanticai-guide/pydantic_ai_comprehensive_guide/',
    '/pydanticai-guide/pydantic_ai_class_deep_dives_v2_54': '/AgentGuides/pydanticai-guide/pydantic_ai_comprehensive_guide/',
    '/pydanticai-guide/pydantic_ai_class_deep_dives_v2_55': '/AgentGuides/pydanticai-guide/pydantic_ai_comprehensive_guide/',
    '/pydanticai-guide/pydantic_ai_class_deep_dives_v2_57': '/AgentGuides/pydanticai-guide/pydantic_ai_comprehensive_guide/',
    '/pydanticai-guide/pydantic_ai_class_examples_2026_08': '/AgentGuides/pydanticai-guide/pydantic_ai_comprehensive_guide/',
  },
  markdown: {
    remarkPlugins: [remarkVersions, [remarkBasePath, { base }]],
  },
  integrations: [
    react(),
    starlight({
      title: 'Agent Guides',
      description:
        'Guides to building AI agents with eighteen frameworks in Python, TypeScript, .NET and Go.',
      favicon: '/favicon.svg',
      components: {
        SiteTitle: './src/components/SiteTitle.astro',
        PageTitle: './src/components/PageTitle.astro',
      },
      head: [{ tag: 'script', content: readingProgress }],
      social: [
        {
          icon: 'github',
          label: 'GitHub',
          href: 'https://github.com/CodeHalwell/AgentGuides',
        },
      ],
      customCss: [
        '@fontsource/newsreader/400.css',
        '@fontsource/newsreader/400-italic.css',
        '@fontsource/hanken-grotesk/400.css',
        '@fontsource/hanken-grotesk/600.css',
        '@fontsource/jetbrains-mono/400.css',
        '@fontsource/jetbrains-mono/500.css',
        './src/styles/global.css',
      ],
      expressiveCode: {
        themes: [codeTheme],
        useStarlightUiThemeColors: false,
        useStarlightDarkModeSwitch: false,
        styleOverrides: {
          borderRadius: '0',
          borderWidth: '1px',
          borderColor: 'var(--rl)',
          codeBackground: 'var(--code)',
          codeFontFamily: 'var(--sl-font-mono)',
          uiFontFamily: 'var(--sl-font-mono)',
          frames: {
            frameBoxShadowCssValue: 'none',
            editorActiveTabIndicatorTopColor: '#E0602F',
            editorTabBarBackground: 'var(--code)',
            editorActiveTabBackground: 'var(--code)',
            terminalTitlebarBackground: 'var(--code)',
            terminalBackground: 'var(--code)',
            terminalTitlebarDotsOpacity: '0',
          },
        },
      },
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
              label: 'AutoGen & AG2',
              collapsed: true,
              items: [
                { label: 'Overview', link: '/autogen-guide/' },
                { label: 'Microsoft AutoGen (Python)', link: '/autogen-guide/python/' },
                { label: 'Microsoft AutoGen (.NET, archived)', link: '/autogen-guide/dotnet/' },
                { label: 'AG2', link: '/ag2-guide/' },
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
                {
                  label: 'API Reference',
                  collapsed: true,
                  items: [
                    { label: 'StateGraph API', link: '/langgraph-guide/python/reference-state-graph/' },
                    { label: 'Checkpointers', link: '/langgraph-guide/python/reference-checkpointers/' },
                    { label: 'Store (long-term memory)', link: '/langgraph-guide/python/reference-store/' },
                    { label: 'Functional API', link: '/langgraph-guide/python/reference-functional-api/' },
                    { label: 'Command / Send', link: '/langgraph-guide/python/reference-command-and-send/' },
                    { label: 'Streaming modes', link: '/langgraph-guide/python/reference-streaming-modes/' },
                    { label: 'Prebuilt nodes', link: '/langgraph-guide/python/reference-prebuilt-nodes/' },
                    { label: 'Channels', link: '/langgraph-guide/python/reference-channels/' },
                    { label: 'Managed values', link: '/langgraph-guide/python/reference-managed-values/' },
                    { label: 'Runtime & ToolRuntime', link: '/langgraph-guide/python/reference-runtime-and-managed-values/' },
                    { label: 'RemoteGraph', link: '/langgraph-guide/python/reference-remote-graph/' },
                    { label: 'Graceful shutdown', link: '/langgraph-guide/python/reference-graceful-shutdown/' },
                    { label: 'TracePolicy', link: '/langgraph-guide/python/reference-trace-policy/' },
                    { label: 'Cache backends', link: '/langgraph-guide/python/reference-cache-backends/' },
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
            {
              label: 'Amazon Bedrock Agents',
              collapsed: true,
              items: [
                { label: 'Overview (Python)', link: '/amazon-bedrock-agents-guide/' },
                {
                  label: 'TypeScript',
                  collapsed: true,
                  items: [
                    { label: 'Overview', link: '/amazon-bedrock-agents-guide/typescript/' },
                    { label: 'Strands Agents', link: '/amazon-bedrock-agents-guide/typescript/strands_typescript_guide/' },
                    { label: 'AgentCore SDK', link: '/amazon-bedrock-agents-guide/typescript/agentcore_typescript_guide/' },
                    { label: 'Deploying to AgentCore Runtime', link: '/amazon-bedrock-agents-guide/typescript/agentcore_typescript_production/' },
                  ],
                },
              ],
            },
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
                    { label: 'Recipes', link: '/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_recipes/' },
                    { label: 'Diagrams', link: '/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_diagrams/' },
                    { label: 'Production guide', link: '/microsoft-agent-framework-guide/python/microsoft_agent_framework_python_production_guide/' },
                  ],
                },
                { label: '.NET', link: '/microsoft-agent-framework-guide/dotnet/' },
                { label: 'Go (preview)', link: '/microsoft-agent-framework-guide/go/' },
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
            {
              label: 'Mistral Agents API',
              collapsed: true,
              items: [
                { label: 'Overview (Python)', link: '/mistral-agents-api-guide/' },
                { label: 'TypeScript', link: '/mistral-agents-api-guide/typescript/' },
              ],
            },
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
