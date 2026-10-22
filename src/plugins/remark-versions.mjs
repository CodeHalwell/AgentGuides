/**
 * Remark plugin: fill in package versions from src/data/versions.json, so
 * the current version of each framework is written in exactly one place.
 *
 *   %%version=agent-framework%%   -> 1.22.0
 *   %%versions-checked%%          -> 2026-10-21 (when versions.json was last checked)
 *   %%model=openai%%              -> the current model ID for that role (versions.json `models`)
 *
 * Applies to prose, inline code, fenced code, raw HTML, link URLs and MDX
 * attribute values. The syntax avoids braces and angle brackets (MDX) and
 * colons (which remark-directive would claim as `:name` directives). It
 * cannot reach YAML frontmatter, which the content collection parses
 * separately. Avoid it in headings too: the anchor slug would change with
 * every version bump.
 *
 * An unknown package key fails the build rather than rendering a placeholder.
 */
import { readFileSync } from 'node:fs';

const PLACEHOLDER = /%%(?:version=([^%\s]+)|model=([^%\s]+)|versions-checked)%%/g;

export default function remarkVersions(options = {}) {
  const path = options.path ?? new URL('../data/versions.json', import.meta.url);
  const { packages, models, checked } = JSON.parse(readFileSync(path, 'utf8'));

  const fill = (text, file) =>
    text.replace(PLACEHOLDER, (match, pkg, role) => {
      if (role) {
        if (!models[role] || role.startsWith('$')) {
          throw new Error(`${file.path}: unknown model role "${role}" in ${match} (not in src/data/versions.json models)`);
        }
        return models[role];
      }
      if (!pkg) return checked;
      const entry = packages[pkg];
      if (!entry) {
        throw new Error(`${file.path}: unknown package "${pkg}" in ${match} (not in src/data/versions.json)`);
      }
      return entry.version;
    });

  return function transformer(tree, file) {
    const visit = (node) => {
      if (!node || typeof node !== 'object') return;
      if (typeof node.value === 'string' && node.value.includes('%%')) {
        node.value = fill(node.value, file);
      }
      if (typeof node.url === 'string' && node.url.includes('%%')) {
        node.url = fill(node.url, file);
      }
      if (node.type === 'mdxJsxFlowElement' || node.type === 'mdxJsxTextElement') {
        for (const attr of node.attributes ?? []) {
          if (typeof attr?.value === 'string' && attr.value.includes('%%')) {
            attr.value = fill(attr.value, file);
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
