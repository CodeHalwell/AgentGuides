/**
 * Remark plugin: prefix site-absolute markdown links/images with Astro's
 * `base` path. Astro does not automatically rewrite `[text](/foo)` in
 * markdown content — only rendered Starlight chrome (sidebar, nav) — so
 * without this the homepage and all ported cross-links 404 under a
 * subpath deploy like `/AgentGuides/`.
 *
 * Relative links (`./sibling/`, `../other/`) are resolved against the source
 * file's folder, which is how authors write them and how scripts/check-links.mjs
 * checks them. Left alone, the browser resolves them against the page URL, and
 * a page `foo.md` is served at `/foo/`, so `./bar/` would become `/foo/bar/`.
 *
 * Skips: external URLs, mailto/tel/anchors, already-prefixed links,
 * protocol-relative URLs.
 */
import path from 'node:path';

const DOCS = `${path.sep}src${path.sep}content${path.sep}docs${path.sep}`;

/** Site-absolute URL of the folder a docs source file lives in, e.g. `/langgraph-guide/python/`. */
function sourceDir(file) {
  const p = file?.path ?? file?.history?.[0];
  if (!p || !p.includes(DOCS)) return null;
  const rel = p.slice(p.indexOf(DOCS) + DOCS.length);
  const dir = path.posix.dirname(rel.split(path.sep).join('/'));
  return dir === '.' ? '/' : `/${dir}/`;
}

function resolveRelative(url, dir) {
  if (!dir || typeof url !== 'string' || !/^\.{1,2}\//.test(url)) return url;
  const [pathPart, hash = ''] = url.split('#');
  let resolved = path.posix.normalize(dir + pathPart).replace(/\.mdx?(?=\/?$)/, '');
  if (!resolved.endsWith('/') && !path.posix.extname(resolved)) resolved += '/';
  return resolved + (hash ? `#${hash}` : '');
}
export default function remarkBasePath(options = {}) {
  const base = options.base ?? '/';
  const prefix = base.replace(/\/$/, '');
  // No-op if no base (root deploy).
  if (!prefix) return () => {};

  const shouldRewrite = (url) => {
    if (typeof url !== 'string' || !url) return false;
    if (!url.startsWith('/')) return false;
    if (url.startsWith('//')) return false; // protocol-relative
    if (url.startsWith(prefix + '/') || url === prefix) return false;
    return true;
  };

  return function transformer(tree, file) {
    const dir = sourceDir(file);
    const visit = (node) => {
      if (!node || typeof node !== 'object') return;
      // Links only: Astro resolves relative image paths itself for optimisation.
      if (node.type === 'link' || node.type === 'definition') {
        node.url = resolveRelative(node.url, dir);
      }
      if (
        (node.type === 'link' ||
          node.type === 'image' ||
          node.type === 'definition' ||
          node.type === 'linkReference') &&
        shouldRewrite(node.url)
      ) {
        node.url = prefix + node.url;
      }
      if (node.type === 'mdxJsxFlowElement' || node.type === 'mdxJsxTextElement') {
        for (const attr of node.attributes ?? []) {
          if (
            attr &&
            attr.type === 'mdxJsxAttribute' &&
            attr.name === 'href' &&
            typeof attr.value === 'string'
          ) {
            attr.value = resolveRelative(attr.value, dir);
          }
          if (
            attr &&
            attr.type === 'mdxJsxAttribute' &&
            (attr.name === 'href' || attr.name === 'src') &&
            typeof attr.value === 'string' &&
            shouldRewrite(attr.value)
          ) {
            attr.value = prefix + attr.value;
          }
        }
      }
      if (Array.isArray(node.children)) {
        for (const child of node.children) visit(child);
      }
    };
    visit(tree);
  };
}
