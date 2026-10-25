#!/usr/bin/env node
// Work around an Astro 5 content-store race (present in 5.18.1 and 5.18.2).
//
// MutableDataStore#writeFileAtomic drops a write that arrives while another is
// in progress: it marks it "pending", returns immediately, and later replays the
// *first* caller's data. So when the sync's final save lands during a debounced
// save of a partial store, `waitUntilSaveComplete()` resolves early, the build
// reads the partial snapshot, and whole guides silently go missing. With this
// site's ~60 MB store the overlap happens on most builds.
//
// The patch serialises writes per file: each call waits for the previous write,
// then writes its own (latest) data, and its promise resolves only once that
// write is on disk.
//
// Runs on `postinstall`; safe to run repeatedly. Remove once Astro fixes it.

import { readFileSync, writeFileSync, existsSync } from 'node:fs';

const file = new URL('../node_modules/astro/dist/content/mutable-data-store.js', import.meta.url);
if (!existsSync(file)) process.exit(0);

let src = readFileSync(file, 'utf8');
const MARK = '/* agentguides: serialised atomic writes */';
if (src.includes(MARK)) process.exit(0);

const fn = /  async #writeFileAtomic\(filePath, data, depth = 0\) \{[\s\S]*?\n  \}\n(?=  scopedStore\()/;
if (!fn.test(src)) {
  console.warn('patch-astro: Astro changed; content-store patch not applied. Check whether the write race is fixed upstream.');
  process.exit(0);
}
src = src.replace(
  fn,
  `  ${MARK}
  #writeChains = /* @__PURE__ */ new Map();
  async #writeFileAtomic(filePath, data) {
    const fileKey = filePath.toString();
    const tempFile = filePath instanceof URL ? new URL(\`\${filePath.href}.tmp\`) : \`\${filePath}.tmp\`;
    const previous = this.#writeChains.get(fileKey) ?? Promise.resolve();
    const current = previous.catch(() => {}).then(async () => {
      const oldData = await fs.readFile(filePath, "utf-8").catch(() => "");
      if (oldData === data) return;
      await fs.writeFile(tempFile, data);
      await fs.rename(tempFile, filePath);
    });
    this.#writeChains.set(fileKey, current);
    try {
      await current;
    } finally {
      if (this.#writeChains.get(fileKey) === current) this.#writeChains.delete(fileKey);
    }
  }
`,
);
writeFileSync(file, src);
console.log('patch-astro: content-store write race patched.');
