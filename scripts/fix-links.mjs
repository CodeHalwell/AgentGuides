#!/usr/bin/env node
// Post-port fixup: the initial port script ate 2 chars off plain filename
// links (because of a slice bug). Scan every content file, find broken
// internal links, and when a sibling file's basename *ends with* the
// broken slug, rewrite the link to the real file.
//
// Scope: src/content/docs only. Leaves external / anchor / asset links
// untouched.

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

async function exists(p) {
  try {
    await fs.access(p);
    return true;
  } catch {
    return false;
  }
}

/**
 * Given a broken link target like "./nggraph_foo/" resolved from
 * fromFile=".../langgraph-guide/python/index.md", try to find the real
 * file in the same directory whose basename ends with "nggraph_foo".
 */
async function suggest(fromFile, href) {
  // Same-dir: "./foo/" — match files/subdirs whose basename ends with slug.
  let m = href.match(/^\.\/([a-z0-9_\-]+)\/?(#.*)?$/i);
  if (m) {
    const slug = m[1].toLowerCase();
    const frag = m[2] ?? '';
    // Special case: "adme" is the port's bug for "readme" — point to folder index.
    if (slug === 'adme' || slug === 'eadme') return `./${frag}`;
    const dir = path.dirname(fromFile);
    const picked = await pickSibling(dir, slug);
    if (picked) return `./${picked}/${frag}`;
    // Fallback: look one level down in immediate subdirs (handles landing
    // pages that linked to per-language files assuming flat layout).
    let entries;
    try {
      entries = await fs.readdir(dir, { withFileTypes: true });
    } catch {
      return null;
    }
    for (const e of entries) {
      if (!e.isDirectory()) continue;
      const picked = await pickSibling(path.join(dir, e.name), slug);
      if (picked) return `./${e.name}/${picked}/${frag}`;
    }
    return null;
  }
  // Sub-dir: "./foo/bar/" — first segment might be the 2-char-eaten subdir name.
  m = href.match(/^\.\/([a-z0-9_\-]+)\/(.+?)(#.*)?$/i);
  if (m) {
    const firstSeg = m[1].toLowerCase();
    const rest = m[2];
    const frag = m[3] ?? '';
    // Try to find a subdir in cwd whose name ends with firstSeg (≤4 lost chars).
    const dir = path.dirname(fromFile);
    let entries;
    try {
      entries = await fs.readdir(dir, { withFileTypes: true });
    } catch {
      return null;
    }
    const candidates = entries
      .filter((e) => e.isDirectory())
      .map((e) => e.name)
      .filter((n) => n.toLowerCase() === firstSeg || (n.toLowerCase().endsWith(firstSeg) && n.length - firstSeg.length <= 4));
    if (candidates.length === 1) {
      return `./${candidates[0]}/${rest}${frag}`;
    }
    // Special cases for common segments.
    const specials = { thon: 'python', pescript: 'typescript', tnet: 'dotnet' };
    if (specials[firstSeg] && entries.some((e) => e.isDirectory() && e.name === specials[firstSeg])) {
      return `./${specials[firstSeg]}/${rest}${frag}`;
    }
  }
  return null;
}

async function pickSibling(dir, slug) {
  let entries;
  try {
    entries = await fs.readdir(dir, { withFileTypes: true });
  } catch {
    return null;
  }
  const names = [];
  for (const e of entries) {
    const base = e.name.replace(/\.mdx?$/i, '').toLowerCase();
    if (base === slug) names.push(e.name.replace(/\.mdx?$/i, ''));
    else if (base.endsWith(slug) && base.length - slug.length <= 4) {
      names.push(e.name.replace(/\.mdx?$/i, ''));
    }
  }
  const unique = [...new Set(names)];
  if (unique.length === 1) return unique[0];
  if (unique.length > 1) {
    unique.sort(
      (a, b) => a.length - slug.length - (b.length - slug.length) || a.localeCompare(b),
    );
    return unique[0];
  }
  return null;
}

async function resolveLink(href, fromFile) {
  const [bare] = href.split(/[#?]/);
  if (!bare) return { ok: true };
  if (/^(https?:|mailto:|tel:)/i.test(bare)) return { ok: true };
  if (/\.(png|jpe?g|gif|svg|webp|pdf|zip|ico|json|xml)$/i.test(bare))
    return { ok: true };

  let abs;
  if (bare.startsWith('/')) {
    const stripped = bare.replace(/^\/AgentGuides\//, '/').replace(/^\//, '');
    abs = path.join(docsRoot, stripped);
  } else {
    abs = path.resolve(path.dirname(fromFile), bare);
  }
  const trimmed = abs.replace(/\/+$/, '');
  const candidates = [
    trimmed + '.md',
    trimmed + '.mdx',
    path.join(trimmed, 'index.md'),
    path.join(trimmed, 'index.mdx'),
  ];
  for (const c of candidates) {
    if (await exists(c)) return { ok: true };
  }
  return { ok: false };
}

const inlineRe = /(\[[^\]]*?\])\((\s*)([^)\s]+?)((?:\s+"[^"]*")?\s*)\)/g;
const refRe = /^(\s*\[[^\]]+\]:\s*)(\S+)(.*)$/gm;

async function fixFile(file) {
  const raw = await fs.readFile(file, 'utf8');
  // Split frontmatter
  const fmMatch = raw.match(/^---\n[\s\S]*?\n---\n/);
  const fm = fmMatch ? fmMatch[0] : '';
  const body = fmMatch ? raw.slice(fm.length) : raw;

  let fixed = body;
  const rewrites = [];

  const inlineMatches = [...body.matchAll(inlineRe)];
  for (const m of inlineMatches) {
    const [full, txt, ws1, href, ws2] = m;
    const res = await resolveLink(href, file);
    if (res.ok) continue;
    const sug = await suggest(file, href);
    if (sug && sug !== href) {
      rewrites.push({ from: full, to: `${txt}(${ws1}${sug}${ws2})` });
    }
  }
  const refMatches = [...body.matchAll(refRe)];
  for (const m of refMatches) {
    const [full, pre, href, post] = m;
    const res = await resolveLink(href, file);
    if (res.ok) continue;
    const sug = await suggest(file, href);
    if (sug && sug !== href) {
      rewrites.push({ from: full, to: `${pre}${sug}${post}` });
    }
  }
  if (rewrites.length === 0) return 0;

  for (const { from, to } of rewrites) {
    fixed = fixed.split(from).join(to);
  }
  await fs.writeFile(file, fm + fixed);
  return rewrites.length;
}

async function main() {
  const files = await walk(docsRoot);
  let total = 0;
  let touched = 0;
  for (const f of files) {
    const n = await fixFile(f);
    if (n > 0) {
      touched++;
      total += n;
    }
  }
  console.log(`Patched ${total} broken links across ${touched} files.`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
