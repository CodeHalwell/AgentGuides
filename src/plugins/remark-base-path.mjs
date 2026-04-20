/**
 * Remark plugin: prefix site-absolute markdown links/images with Astro's
 * `base` path. Astro does not automatically rewrite `[text](/foo)` in
 * markdown content — only rendered Starlight chrome (sidebar, nav) — so
 * without this the homepage and all ported cross-links 404 under a
 * subpath deploy like `/AgentGuides/`.
 *
 * Skips: external URLs, mailto/tel/anchors, already-prefixed links,
 * protocol-relative URLs.
 */
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

  return function transformer(tree) {
    const visit = (node) => {
      if (!node || typeof node !== 'object') return;
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
