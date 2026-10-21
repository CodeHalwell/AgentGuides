#!/usr/bin/env node
// Keeps the version numbers in the guides honest.
//
//   node scripts/check-versions.mjs            scan the docs for pins/banners that disagree with src/data/versions.json
//   node scripts/check-versions.mjs --latest   compare versions.json with the package registries
//   node scripts/check-versions.mjs --latest --write   ...and update versions.json in place
//
// Exit code is 1 when a stale install pin is found (or, with --latest, when a registry is ahead).
// "Targets vX" style banners are reported as warnings only, as the heuristic can't tell
// a stale banner from a legitimate historical note such as "added in 1.17.0".

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
    default:
      return [];
  }
}

const BANNER = new RegExp(
  String.raw`(?:targets?|tested (?:with|against)|verified against|current(?:ly)?(?: release| version)?|latest(?: release| version)?|version)\s*[:=]?\s*\**\s*v?${SEMVER}`,
  'gi',
);

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
  const byGuide = {};
  for (const [name, pkg] of Object.entries(data.packages)) {
    (byGuide[pkg.guide] ??= []).push(pkg.version);
  }
  for (const file of walk(docsDir)) {
    const rel = relative(root, file);
    const lines = readFileSync(file, 'utf8').split('\n');
    const guide = relative(docsDir, file).split('/')[0];
    lines.forEach((line, i) => {
      for (const [name, pkg] of Object.entries(data.packages)) {
        for (const re of pinPatterns(name, pkg.registry)) {
          for (const m of line.matchAll(re)) {
            if (m[1] !== pkg.version) {
              errors.push(`${rel}:${i + 1}  ${name} pinned to ${m[1]} (current ${pkg.version})`);
            }
          }
        }
      }
      const current = byGuide[guide];
      if (!current) return;
      for (const m of line.matchAll(BANNER)) {
        if (!current.includes(m[1]) && !/added|since|introduced|new in|changelog|before|prior|from v?\d/i.test(line)) {
          warnings.push(`${rel}:${i + 1}  "${m[0].trim()}" (guide targets ${current.join(', ')})`);
        }
      }
    });
  }
  for (const w of warnings) console.log(`warn   ${w}`);
  for (const e of errors) console.log(`error  ${e}`);
  console.log(`\n${errors.length} stale pin(s), ${warnings.length} banner warning(s).`);
  return errors.length === 0;
}

async function latest(name, registry) {
  const get = async (url) => {
    const res = await fetch(url);
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
  }
}

async function checkLatest() {
  let behind = 0;
  for (const [name, pkg] of Object.entries(data.packages)) {
    try {
      const now = await latest(name, pkg.registry);
      if (now !== pkg.version) {
        behind++;
        console.log(`${name.padEnd(32)} ${pkg.version.padEnd(10)} -> ${now}   (${pkg.guide})`);
        if (args.has('--write')) pkg.version = now;
      }
    } catch (err) {
      console.log(`${name.padEnd(32)} lookup failed: ${err.message}`);
    }
  }
  if (args.has('--write') && behind) {
    data.checked = new Date().toISOString().slice(0, 10);
    writeFileSync(versionsPath, JSON.stringify(data, null, 2) + '\n');
    console.log(`\nUpdated ${relative(root, versionsPath)}. Now update the guides and re-run without --latest.`);
  }
  console.log(behind ? `\n${behind} package(s) behind the registries.` : 'All packages current.');
  return behind === 0 || args.has('--write');
}

const ok = args.has('--latest') ? await checkLatest() : scanDocs();
process.exit(ok ? 0 : 1);
