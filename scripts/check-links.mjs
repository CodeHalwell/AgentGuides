#!/usr/bin/env node
// Scan src/content/docs for internal links and report any that don't resolve
// to a real page in the collection.
//
// Links are resolved the way the browser resolves them on the deployed site,
// i.e. against the page's URL rather than its source folder:
// - src/content/docs/foo/bar.md       is served at /foo/bar/
// - src/content/docs/foo/index.md     is served at /foo/
// So `./baz/` in foo/bar.md points at /foo/bar/baz/ (usually a 404), while in
// foo/index.md it points at /foo/baz/. Use `../baz/` or `/foo/baz/` from a
// non-index page.
//
// Checks markdown links, reference definitions and MDX href="..." attributes.
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
  // MDX/JSX attributes such as <LinkCard href="./foo/" />.
  const attr = /\bhref=["']([^"']+)["']/g;
  while ((m = attr.exec(md))) {
    yield { text: '', href: m[1], index: m.index };
  }
}

// Site URL path (without base) that Starlight serves a docs file at.
function pageUrl(file) {
  const rel = path.relative(docsRoot, file).split(path.sep).join('/');
  const noExt = rel.replace(/\.(md|mdx)$/i, '');
  const segs = noExt.split('/');
  if (segs[segs.length - 1].toLowerCase() === 'index') segs.pop();
  const slug = segs.map((seg) => seg.toLowerCase().replace(/\s+/g, '-')).join('/');
  return slug ? `/${slug}/` : '/';
}

async function resolveLink(href, fromFile, pages) {
  // Returns {ok, target} where target is the site path the link points at.
  const [bare] = href.split(/[#?]/);
  if (!bare) return { ok: true, target: null }; // pure anchor
  if (/^(https?:|mailto:|tel:|data:)/i.test(bare) || bare.startsWith('//'))
    return { ok: true, target: null };
  if (/\.(png|jpe?g|gif|svg|webp|pdf|zip|ico|json|xml|txt)$/i.test(bare))
    return { ok: true, target: null };

  // Resolve like a browser, against the page URL (base prefix stripped).
  const from = new URL(pageUrl(fromFile), 'https://site.invalid');
  let target = new URL(bare.replace(/^\/AgentGuides(?=\/|$)/, '') || '/', from).pathname;
  target = decodeURIComponent(target).toLowerCase();
  if (!target.endsWith('/')) target += '/';
  return { ok: pages.has(target), target };
}

async function main() {
  const files = await walk(docsRoot);
  const pages = new Set(files.map(pageUrl));
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
      const r = await resolveLink(href, file, pages);
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
  if (broken.length) process.exitCode = 1;
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
