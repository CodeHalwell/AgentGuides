// Builds each guide's sidebar group from its folder, so every page is listed
// without hand-maintaining links (hand-maintained lists left most topic pages
// out of the sidebar).
//
//   guide('langgraph-guide', 'LangGraph')
//
// `sections` below lists the guides; astro.config.mjs and the All Guides page
// (GuideIndex.astro) both build from it.
//
// Ordering: the folder's index first (as "Overview"), then frontmatter
// `sidebar.order`, then a default rank from the file name (tutorial chapters,
// comprehensive guide, topic pages, recipes, production, diagrams). Language
// subfolders become groups ("Python", "TypeScript", ".NET", "Go"), and three or
// more files sharing a `chapter-` or `reference-` prefix get their own group.
// Labels come from frontmatter `sidebar.label`, else the title with the
// framework's name removed.

import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import yaml from 'js-yaml';

// From the project root: this module is also bundled into the All Guides page,
// where import.meta.url no longer points at src/data.
const DOCS = join(process.cwd(), 'src/content/docs');
const LANGS = { python: 'Python', typescript: 'TypeScript', dotnet: '.NET', go: 'Go' };
const GROUPS = { chapter: 'Tutorial', reference: 'API reference' };

function frontmatter(file) {
  const text = readFileSync(file, 'utf8');
  const m = text.match(/^---\n([\s\S]*?)\n---/);
  return m ? (yaml.load(m[1]) ?? {}) : {};
}

function rank(name) {
  const n = name.toLowerCase();
  const chapter = n.match(/chapter-(\d+)/);
  if (chapter) return 100 + Number(chapter[1]);
  if (/comprehensive/.test(n)) return 200;
  if (/start_here|getting.started|quick/.test(n)) return 150;
  if (/migration|migrating/.test(n)) return 700;
  if (/recipe/.test(n)) return 800;
  if (/production|deploy/.test(n)) return 850;
  if (/diagram/.test(n)) return 900;
  if (/update|archive|launch|validation_report|notice/.test(n)) return 950;
  return 500;
}

/** Shorten a page title for the sidebar by dropping the framework name it repeats. */
function shortLabel(title, guideName) {
  let t = String(title).trim();
  const stem = guideName.replace(/\s*\(.*\)\s*$/, '');
  const parts = t.split(/\s+[—–-]\s+|:\s+/);
  if (parts.length > 1 && parts[0].toLowerCase().includes(stem.toLowerCase().split(' ')[0])) {
    t = parts.slice(1).join(' — ');
  }
  t = t
    .replace(new RegExp(`^${stem.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\s*`, 'i'), '')
    .replace(/\((Python|TypeScript|\.NET|C#|Go)\)\s*/g, '')
    .replace(/^[—–:\-\s]+/, '')
    .trim();
  return t ? t[0].toUpperCase() + t.slice(1) : String(title);
}

function pages(dir, guideName) {
  const abs = join(DOCS, dir);
  const out = [];
  for (const entry of readdirSync(abs)) {
    const full = join(abs, entry);
    if (statSync(full).isDirectory()) continue;
    const m = entry.match(/^(.*)\.mdx?$/);
    if (!m) continue;
    const fm = frontmatter(full);
    if (fm.sidebar?.hidden) continue;
    const isIndex = m[1] === 'index';
    out.push({
      slug: isIndex ? `/${dir}/` : `/${dir}/${m[1]}/`,
      name: m[1],
      isIndex,
      order: isIndex ? -1 : (fm.sidebar?.order ?? rank(m[1])),
      label: isIndex ? 'Overview' : (fm.sidebar?.label ?? shortLabel(fm.title ?? m[1], guideName)),
    });
  }
  out.sort((a, b) => a.order - b.order || a.label.localeCompare(b.label));
  return out;
}

function items(dir, guideName) {
  const list = pages(dir, guideName);
  const result = [];
  for (const [prefix, label] of Object.entries(GROUPS)) {
    const members = list.filter((p) => p.name?.startsWith(`${prefix}-`));
    if (members.length >= 3) {
      for (const p of members) list.splice(list.indexOf(p), 1);
      list.push({ group: { label, collapsed: true, items: members.map(link) }, order: members[0].order });
    }
  }
  list.sort((a, b) => a.order - b.order);
  for (const p of list) result.push(p.group ?? link(p));

  // Language folders first, in LANGS order, then any other subfolder (such as
  // ag2-guide/classic), labelled from its index's `sidebar.label` or title.
  const abs = join(DOCS, dir);
  const subs = readdirSync(abs).filter((e) => statSync(join(abs, e)).isDirectory());
  const langs = Object.keys(LANGS).filter((l) => subs.includes(l));
  for (const sub of [...langs, ...subs.filter((s) => !LANGS[s]).sort()]) {
    result.push({ label: LANGS[sub] ?? folderLabel(join(abs, sub), sub), collapsed: true, items: items(`${dir}/${sub}`, guideName) });
  }
  return result;
}

function folderLabel(abs, sub) {
  for (const ext of ['mdx', 'md']) {
    try {
      const fm = frontmatter(join(abs, `index.${ext}`));
      return fm.sidebar?.group ?? fm.title ?? sub;
    } catch {
      // no index with this extension
    }
  }
  return sub[0].toUpperCase() + sub.slice(1);
}

function link(p) {
  return { label: p.label, link: p.slug };
}

/** A collapsed sidebar group listing every page under `dir`. */
export function guide(dir, label) {
  return { label, collapsed: true, items: items(dir, label) };
}

/** Guides by category, in sidebar order. The sidebar and the All Guides page both read this. */
export const sections = [
  {
    label: 'Multi-Agent Orchestration',
    guides: [
      ['openai-agents-sdk-guides', 'OpenAI Agents SDK (Python)'],
      ['openai-agents-sdk-typescript-guide', 'OpenAI Agents SDK (TypeScript)'],
      ['crewai-guide', 'CrewAI'],
      ['ag2-guide', 'AG2'],
      ['autogen-guide', 'Microsoft AutoGen'],
      ['langgraph-guide', 'LangGraph'],
    ],
  },
  {
    label: 'Data & Knowledge',
    guides: [
      ['llamaindex-guide', 'LlamaIndex'],
      ['haystack-guide', 'Haystack'],
      ['pydanticai-guide', 'PydanticAI'],
    ],
  },
  {
    label: 'Cloud-Native',
    guides: [
      ['amazon-bedrock-agents-guide', 'Amazon Bedrock Agents'],
      ['microsoft-agent-framework-guide', 'Microsoft Agent Framework'],
      ['google-adk-guide', 'Google ADK'],
      ['strands-agents-guide', 'Strands Agents'],
    ],
  },
  {
    label: 'Model-Specific',
    guides: [
      ['anthropic-claude-agent-sdk-guide', 'Claude Agent SDK (Python)'],
      ['anthropic-claude-agent-sdk-typescript-guide', 'Claude Agent SDK (TypeScript)'],
      ['mistral-agents-api-guide', 'Mistral Agents API'],
    ],
  },
  {
    label: 'Enterprise & Lightweight',
    guides: [
      ['semantic-kernel-guide', 'Semantic Kernel'],
      ['smolagents-guide', 'SmolAgents'],
      ['agno-guide', 'Agno'],
    ],
  },
];

/** The guide categories as Starlight sidebar groups. */
export function guideSidebar() {
  return sections.map((s) => ({
    label: s.label,
    collapsed: true,
    items: s.guides.map(([dir, label]) => guide(dir, label)),
  }));
}
