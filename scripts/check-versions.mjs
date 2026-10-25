#!/usr/bin/env node
// The current version of every package lives only in src/data/versions.json.
// Docs refer to it as %%version=<package>%% and src/plugins/remark-versions.mjs
// fills it in at build time. This script keeps it that way.
//
//   node scripts/check-versions.mjs            scan the docs for hard-coded versions and unknown placeholders
//   node scripts/check-versions.mjs --latest   compare versions.json with the package registries
//   node scripts/check-versions.mjs --latest --write   ...and bump versions.json (the docs follow automatically)
//
// Errors (exit 1): an exact install pin of a tracked package written as a literal number,
// or a placeholder naming a package or model role that isn't in versions.json.
// Warnings also cover stale chat-model IDs inside code blocks.
// Warnings: a literal copy of a current version, or a "targets vX" banner with a literal
// number. These can be legitimate history ("added in 1.17.0"), so they don't fail.
//
// History is exempt: dated release notes under updates/, and revision-history table
// rows (a row whose first cell is a version or a date).

import { readFileSync, writeFileSync, readdirSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';

const root = new URL('..', import.meta.url).pathname;
const versionsPath = join(root, 'src/data/versions.json');
const docsDir = join(root, 'src/content/docs');
const data = JSON.parse(readFileSync(versionsPath, 'utf8'));
const args = new Set(process.argv.slice(2));

const esc = (s) => s.replace(/[.*+?^${}()|[\]\\/]/g, '\\$&');
const SEMVER = String.raw`(\d+\.\d+\.\d+(?:[-.]?(?:a|b|rc|alpha|beta|preview)[.\d-]*)?)`;

// Patterns that pin an exact version of a package, per registry.
function pinPatterns(name, registry) {
  const n = esc(name);
  switch (registry) {
    case 'pypi':
      // pip install pkg==1.2.3, pkg[extra]==1.2.3, "pkg==1.2.3" in requirements/pyproject
      return [new RegExp(String.raw`(?<![\w.-])${n}(?:\[[^\]]*\])?\s*==\s*${SEMVER}`, 'gi')];
    case 'npm':
      // npm install @scope/pkg@1.2.3, "@scope/pkg": "^1.2.3"
      return [
        new RegExp(String.raw`(?<![\w/-])${n}@\^?~?${SEMVER}`, 'g'),
        new RegExp(String.raw`"${n}"\s*:\s*"[\^~]?${SEMVER}"`, 'g'),
      ];
    case 'nuget':
      // dotnet add package X --version 1.2.3, <PackageReference Include="X" Version="1.2.3" />
      return [
        new RegExp(String.raw`(?<![\w.])${n}\s+--version\s+${SEMVER}`, 'gi'),
        new RegExp(String.raw`Include="${n}"\s+Version="${SEMVER}"`, 'gi'),
      ];
    case 'go':
      return [new RegExp(String.raw`${n}@v${SEMVER}`, 'g')];
    case 'crates':
      // cargo add pkg@1.2.3, pkg = "1.2.3" or pkg = { version = "1.2.3" } in Cargo.toml
      return [
        new RegExp(String.raw`cargo add\s+${n}@${SEMVER}`, 'g'),
        new RegExp(String.raw`^\s*${n}\s*=\s*(?:\{[^}\n]*version\s*=\s*)?"[=^~]?${SEMVER}"`, 'gm'),
      ];
    default:
      return [];
  }
}

const BANNER = new RegExp(
  String.raw`(?:targets?|tested (?:with|against)|verified against|current(?:ly)?(?: release| version)?|latest(?: release| version)?|version)\s*[:=]?\s*\**\s*v?${SEMVER}`,
  'gi',
);
const HISTORY = /added|since|introduced|new in|changelog|before|prior|from v?\d|first (?:written|documented)|deprecated|removed|until|up to/i;
const PLACEHOLDER = /%%(?:version=([^%\s]+)|model=([^%\s]+)|versions-checked)%%/g;
// Chat-model IDs that have a current replacement in versions.json `models`; flagged inside code.
const STALE_MODEL = /["'`/:.](gpt-4o(?:-mini)?|gpt-4(?:\.1)?(?:-mini|-nano|-turbo)?|gpt-3\.5-turbo|gpt-5(?:\.\d)?(?:-mini|-nano|-sol|-luna)?|claude-(?:3|3-5|3-7)-(?:opus|sonnet|haiku)[\w-]*|claude-(?:opus|sonnet|haiku)-4(?:-\d)?(?:-\d{8})?|gemini-(?:1\.5|2\.0|2\.5)-(?:pro|flash)(?:-lite)?)["'`]/;
const HISTORY_ROW = /^\|\s*(?:v?\d+\.\d+(?:\.\d+)?|\d{4}-\d{2}(?:-\d{2})?|[A-Z][a-z]+ \d{1,2},? \d{4}|[A-Z][a-z]+ \d{4})\b/;
const ARCHIVE = new Set(['updates']);

function* walk(dir) {
  for (const entry of readdirSync(dir)) {
    const p = join(dir, entry);
    if (statSync(p).isDirectory()) yield* walk(p);
    else if (/\.mdx?$/.test(entry)) yield p;
  }
}

function scanDocs() {
  const errors = [];
  const warnings = [];
  const current = {};
  for (const [name, pkg] of Object.entries(data.packages)) {
    (current[pkg.guide] ??= new Map()).set(pkg.version, name);
  }
  for (const file of walk(docsDir)) {
    const rel = relative(root, file);
    const guide = relative(docsDir, file).split('/')[0];
    if (ARCHIVE.has(guide)) continue;
    const lines = readFileSync(file, 'utf8').split('\n');
    let frontmatter = lines[0] === '---';
    let inCode = false;
    lines.forEach((line, i) => {
      const at = `${rel}:${i + 1}`;
      if (frontmatter && i > 0 && line === '---') frontmatter = false;
      if (/^\s*(```|~~~)/.test(line)) inCode = !inCode;
      if (inCode && !HISTORY.test(line)) {
        const stale = line.match(STALE_MODEL);
        if (stale) warnings.push(`${at}  stale model ID ${stale[1]}; use %%model=<role>%%`);
      }
      for (const m of line.matchAll(PLACEHOLDER)) {
        if (m[1] && !data.packages[m[1]]) errors.push(`${at}  unknown package in ${m[0]}`);
        if (m[2] && (!data.models?.[m[2]] || m[2].startsWith('$'))) errors.push(`${at}  unknown model role in ${m[0]}`);
        if (frontmatter) errors.push(`${at}  ${m[0]} in frontmatter is not substituted`);
        else if (!inCode && /^#{1,6}\s/.test(line)) errors.push(`${at}  ${m[0]} in a heading changes its anchor on every bump`);
      }
      if (HISTORY_ROW.test(line)) return;
      for (const [name, pkg] of Object.entries(data.packages)) {
        for (const re of pinPatterns(pkg.name ?? name, pkg.registry)) {
          for (const m of line.matchAll(re)) {
            errors.push(`${at}  ${name} pinned to literal ${m[1]}; use %%version=${name}%%`);
          }
        }
      }
      const here = current[guide];
      if (!here || HISTORY.test(line)) return;
      for (const m of line.matchAll(BANNER)) {
        warnings.push(`${at}  "${m[0].trim()}" written literally`);
      }
      for (const [version, name] of here) {
        const pkgName = data.packages[name].name ?? name;
        if (!line.includes(pkgName.split('/').at(-1))) continue;
        if (new RegExp(String.raw`(?<![\d.])v?${esc(version)}(?![\d.])`).test(line) && !BANNER.test(line)) {
          warnings.push(`${at}  literal ${version} (current ${name}); use %%version=${name}%%`);
        }
        BANNER.lastIndex = 0;
      }
    });
  }
  for (const w of warnings) console.log(`warn   ${w}`);
  for (const e of errors) console.log(`error  ${e}`);
  console.log(`\n${errors.length} error(s), ${warnings.length} warning(s).`);
  return errors.length === 0;
}

async function latest(name, registry) {
  const get = async (url) => {
    // crates.io rejects requests without a User-Agent.
    const res = await fetch(url, { headers: { 'User-Agent': 'AgentGuides check-versions' } });
    if (!res.ok) throw new Error(`${url} -> ${res.status}`);
    return res.json();
  };
  const stable = (vs) =>
    vs
      .filter((v) => /^\d+\.\d+\.\d+$/.test(v))
      .sort((a, b) => {
        const [x, y] = [a, b].map((v) => v.split('.').map(Number));
        return x[0] - y[0] || x[1] - y[1] || x[2] - y[2];
      })
      .at(-1);
  switch (registry) {
    case 'pypi':
      return (await get(`https://pypi.org/pypi/${name}/json`)).info.version;
    case 'npm':
      return (await get(`https://registry.npmjs.org/${name}/latest`)).version;
    case 'nuget':
      return stable((await get(`https://api.nuget.org/v3-flatcontainer/${name.toLowerCase()}/index.json`)).versions);
    case 'go':
      return (await get(`https://proxy.golang.org/${name}/@latest`)).Version.replace(/^v/, '');
    case 'crates':
      return (await get(`https://crates.io/api/v1/crates/${name}`)).crate.max_stable_version;
  }
}

async function checkLatest() {
  let behind = 0;
  let failed = 0;
  for (const [name, pkg] of Object.entries(data.packages)) {
    try {
      const now = await latest(pkg.name ?? name, pkg.registry);
      if (now !== pkg.version) {
        behind++;
        console.log(`${name.padEnd(32)} ${pkg.version.padEnd(10)} -> ${now}   (${pkg.guide})`);
        if (args.has('--write')) pkg.version = now;
      }
    } catch (err) {
      failed++;
      console.log(`${name.padEnd(32)} lookup failed: ${err.message}`);
    }
  }
  // A partial check must not stamp a new checked date or pass the job.
  if (failed) {
    console.log(`\n${failed} lookup(s) failed; versions.json not updated.`);
    return false;
  }
  if (args.has('--write') && behind) {
    data.checked = new Date().toISOString().slice(0, 10);
    writeFileSync(versionsPath, JSON.stringify(data, null, 2) + '\n');
    console.log(`\nUpdated ${relative(root, versionsPath)}. The docs pick it up on the next build; review the changelogs for breaking changes.`);
  }
  console.log(behind ? `\n${behind} package(s) behind the registries.` : 'All packages current.');
  return behind === 0 || args.has('--write');
}

const ok = args.has('--latest') ? await checkLatest() : scanDocs();
process.exit(ok ? 0 : 1);
