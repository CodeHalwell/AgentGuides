#!/usr/bin/env node
// Scan src/content/docs for internal markdown links and report any that
// don't resolve to a real file/folder in the collection.
//
// For Starlight:
// - /foo/bar/ resolves to src/content/docs/foo/bar/index.md[x] or foo/bar.md[x]
// - ./baz/ inside src/content/docs/foo/ resolves the same way relative to foo/
//
// Anchors (#) and external URLs are ignored. Query strings (?) are stripped.

import { promises as fs } from 'node:fs';
import path from 'node:path';
import url from 'node:url';

const __filename = url.fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const repoRoot = path.resolve(__dirname, '..');
const docsRoot = path.join(repoRoot, 'src', 'content', 'docs');

async function walk(dir) {
  const out = [];
  async function rec(d) {
    const entries = await fs.readdir(d, { withFileTypes: true });
    for (const e of entries) {
      const full = path.join(d, e.name);
      if (e.isDirectory()) await rec(full);
      else if (/\.(md|mdx)$/i.test(e.name)) out.push(full);
    }
  }
  await rec(dir);
  return out;
}

function* extractLinks(md) {
  // Inline markdown links: [text](href) — href can't contain unbalanced parens.
  // Also handle [...]: ref-style.
  const inline = /\[([^\]]*?)\]\(\s*([^)\s]+?)(?:\s+"[^"]*")?\s*\)/g;
  let m;
  while ((m = inline.exec(md))) {
    yield { text: m[1], href: m[2], index: m.index };
  }
  const ref = /^\s*\[[^\]]+\]:\s*(\S+)/gm;
  while ((m = ref.exec(md))) {
    yield { text: '', href: m[1], index: m.index };
  }
}

async function exists(absPath) {
  try {
    await fs.access(absPath);
    return true;
  } catch {
    return false;
  }
}

async function resolveLink(href, fromFile) {
  // Returns {ok, target} where target is the canonical file path.
  const [bare] = href.split(/[#?]/);
  if (!bare) return { ok: true, target: null }; // pure anchor
  if (/^(https?:|mailto:|tel:)/i.test(bare)) return { ok: true, target: null };
  if (/\.(png|jpe?g|gif|svg|webp|pdf|zip|ico|json|xml)$/i.test(bare))
    return { ok: true, target: null };

  // Resolve to absolute within docsRoot.
  let absDir;
  if (bare.startsWith('/')) {
    // Root-relative, strip leading slash to join with docsRoot.
    // If it starts with /AgentGuides/, strip that too.
    const stripped = bare.replace(/^\/AgentGuides\//, '/').replace(/^\//, '');
    absDir = path.join(docsRoot, stripped);
  } else {
    absDir = path.resolve(path.dirname(fromFile), bare);
  }
  // Trim trailing slash for candidate building.
  const trimmed = absDir.replace(/\/+$/, '');

  const candidates = [
    trimmed + '.md',
    trimmed + '.mdx',
    path.join(trimmed, 'index.md'),
    path.join(trimmed, 'index.mdx'),
  ];
  for (const c of candidates) {
    if (await exists(c)) return { ok: true, target: c };
  }
  return { ok: false, target: trimmed };
}

async function main() {
  const files = await walk(docsRoot);
  const broken = [];
  let totalLinks = 0;
  for (const file of files) {
    const md = await fs.readFile(file, 'utf8');
    // Skip frontmatter and all fenced code blocks (avoid false-positive
    // matches on TS type syntax like `: any;`).
    const body = md
      .replace(/^---\n[\s\S]*?\n---\n/, '')
      .replace(/```[\s\S]*?```/g, '')
      .replace(/`[^`\n]*`/g, '');
    for (const { href } of extractLinks(body)) {
      totalLinks++;
      const r = await resolveLink(href, file);
      if (!r.ok) broken.push({ file: path.relative(docsRoot, file), href, expected: r.target });
    }
  }
  console.log(`Scanned ${files.length} files, ${totalLinks} links, ${broken.length} broken.\n`);
  // Group by href pattern
  const byHref = new Map();
  for (const b of broken) {
    const key = b.href;
    if (!byHref.has(key)) byHref.set(key, []);
    byHref.get(key).push(b.file);
  }
  const sorted = [...byHref.entries()].sort((a, b) => b[1].length - a[1].length);
  for (const [href, filesList] of sorted.slice(0, 30)) {
    console.log(`  ${filesList.length.toString().padStart(3)}×  ${href}`);
    for (const f of filesList.slice(0, 3)) console.log(`         ${f}`);
    if (filesList.length > 3) console.log(`         …and ${filesList.length - 3} more`);
  }
  if (sorted.length > 30) console.log(`\n(${sorted.length - 30} more unique broken hrefs)`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
