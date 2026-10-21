// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import react from '@astrojs/react';
import tailwindcss from '@tailwindcss/vite';
import remarkBasePath from './src/plugins/remark-base-path.mjs';
import remarkVersions from './src/plugins/remark-versions.mjs';
import { guideSidebar } from './src/data/sidebar.mjs';
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
        ...guideSidebar(),
        {
          label: 'Updates',
          collapsed: true,
          autogenerate: { directory: 'updates' },
        },
      ],
    }),
  ],
  vite: {
    plugins: [tailwindcss()],
  },
});
