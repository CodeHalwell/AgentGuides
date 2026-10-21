// Builds each guide's sidebar group from its folder, so every page is listed
// without hand-maintaining links (hand-maintained lists left most topic pages
// out of the sidebar).
//
//   guide('langgraph-guide/python', 'LangGraph (Python)')
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

const DOCS = new URL('../content/docs/', import.meta.url).pathname;
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
    const members = list.filter((p) => p.name.startsWith(`${prefix}-`));
    if (members.length >= 3) {
      for (const p of members) list.splice(list.indexOf(p), 1);
      list.push({ group: { label, collapsed: true, items: members.map(link) }, order: members[0].order });
    }
  }
  list.sort((a, b) => a.order - b.order);
  for (const p of list) result.push(p.group ?? link(p));

  const abs = join(DOCS, dir);
  for (const sub of Object.keys(LANGS)) {
    try {
      if (statSync(join(abs, sub)).isDirectory()) {
        result.push({ label: LANGS[sub], collapsed: true, items: items(`${dir}/${sub}`, guideName) });
      }
    } catch {
      // no such language folder
    }
  }
  return result;
}

function link(p) {
  return { label: p.label, link: p.slug };
}

/** A collapsed sidebar group listing every page under `dir`. */
export function guide(dir, label) {
  return { label, collapsed: true, items: items(dir, label) };
}
