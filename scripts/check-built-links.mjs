#!/usr/bin/env node
// Check every internal link in the built site (dist/) resolves to a real page.
//
// check-links.mjs reads the markdown sources and resolves relative links the way
// authors mean them. This one checks what the browser actually gets, after the
// base path and remark plugins have run, so it catches anything those get wrong.
// Run `npm run build` first. Anchors are not checked.

import { readFileSync, readdirSync, statSync, existsSync } from 'node:fs';
import { join, relative } from 'node:path';

const root = new URL('../dist/', import.meta.url).pathname;
const base = '/AgentGuides';

if (!existsSync(root)) {
  console.error('dist/ not found: run `npm run build` first.');
  process.exit(2);
}

function* walk(dir) {
  for (const entry of readdirSync(dir)) {
    const p = join(dir, entry);
    if (statSync(p).isDirectory()) yield* walk(p);
    else if (entry.endsWith('.html')) yield p;
  }
}

const exists = (sitePath) => {
  const p = decodeURIComponent(sitePath.slice(base.length)).replace(/^\//, '');
  return p === '' || existsSync(join(root, p, 'index.html')) || (existsSync(join(root, p)) && statSync(join(root, p)).isFile());
};

const broken = [];
let checked = 0;
for (const file of walk(root)) {
  const rel = relative(root, file);
  const html = readFileSync(file, 'utf8');
  // Redirect stubs and the 404 page aren't content.
  if (html.startsWith('<!doctype html><title>Redirecting')) continue;
  const content = html.match(/<div class="sl-markdown-content">([\s\S]*?)<\/div>\s*<\/div>\s*<footer/)?.[1] ?? html;
  const pageUrl = `${base}/${rel.replace(/index\.html$/, '')}`;
  for (const [, raw] of content.matchAll(/href="([^"]+)"/g)) {
    const href = raw.replace(/&amp;/g, '&');
    if (/^(https?:|mailto:|tel:|data:|#)/.test(href)) continue;
    checked++;
    const target = new URL(href, `https://x${pageUrl}`).pathname;
    if (!target.startsWith(base) || !exists(target)) broken.push(`${pageUrl}  →  ${href}`);
  }
}

for (const b of broken) console.log(b);
console.log(`\nChecked ${checked} internal links in dist/: ${broken.length} broken.`);
process.exit(broken.length ? 1 : 0);
