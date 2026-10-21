import { useMemo, useState } from 'react';
import { frameworks, categoryGroup, languageSymbol, type Framework } from '../data/frameworks';
import { withBase } from './FrameworkGrid';

type Lang = Framework['languages'][number];
const LANGS: Lang[] = ['Python', 'TypeScript', '.NET', 'Go'];

const groupVar: Record<string, [string, string]> = {
  oxide: ['var(--ac)', 'var(--onac)'],
  verdigris: ['var(--vd)', 'var(--onvd)'],
  sulfur: ['var(--su)', 'var(--brand-ink)'],
  permanganate: ['var(--pm)', 'var(--onpm)'],
  cobalt: ['var(--co)', 'var(--onco)'],
  ink: ['var(--tx)', 'var(--bg)'],
};
const groupText: Record<string, string> = {
  oxide: 'var(--ac)',
  verdigris: 'var(--vd)',
  sulfur: 'var(--su-tx)',
  permanganate: 'var(--pm)',
  cobalt: 'var(--co)',
  ink: 'var(--tx)',
};

const shortName = (f: Framework) => f.name.replace(/\s*\((Python|TypeScript)\)$/, '').replace('Anthropic ', '');

function Tile({
  n,
  sub,
  symbol,
  label,
  group,
  pressed,
  onToggle,
  title,
}: {
  n: number;
  sub: string;
  symbol: string;
  label: string;
  group: string;
  pressed: boolean;
  onToggle: () => void;
  title: string;
}) {
  const [bg, fg] = groupVar[group];
  return (
    <button
      type="button"
      className="pt-tile"
      aria-pressed={pressed}
      title={title}
      onClick={onToggle}
      style={
        pressed
          ? { background: bg, color: fg, borderColor: 'var(--tx)', boxShadow: `5px 5px 0 ${group === 'ink' ? 'var(--ac)' : 'var(--tx)'}` }
          : undefined
      }
    >
      <span className="pt-n">
        {n} · {sub}
      </span>
      <span className="pt-sym" style={pressed ? undefined : { color: groupText[group] }}>
        {symbol}
      </span>
      <span className="pt-label">{label}</span>
    </button>
  );
}

export default function PeriodicTable({ pages }: { pages: Record<string, number> }) {
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const toggle = (id: string) => {
    const next = new Set(selected);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setSelected(next);
  };

  const pickedFrameworks = frameworks.filter((f) => selected.has(f.id));
  const pickedLangs = LANGS.filter((l) => selected.has(`lang:${l}`));

  const results = useMemo(
    () =>
      (pickedFrameworks.length ? pickedFrameworks : frameworks).filter((f) =>
        pickedLangs.every((l) => f.languages.includes(l)),
      ),
    [selected],
  );

  const formula = [...pickedFrameworks.map((f) => f.symbol), ...pickedLangs.map((l) => languageSymbol[l])];

  return (
    <div className="not-content pt">
      <section className="pt-hero">
        <div className="pt-intro">
          <span className="pt-eyebrow">The periodic table of agent frameworks</span>
          <h1 id="_top" className="pt-title">Pick an element.</h1>
          <p className="pt-lead">
            Eighteen frameworks across Python, TypeScript, .NET and Go. Every example is checked against the current
            release. Select tiles to combine them like a formula.
          </p>
          <div className="pt-formula" aria-live="polite">
            <span className="pt-formula-label">Formula</span>
            <span className="pt-formula-value">{formula.length ? formula.join(' · ') : 'Ag'}</span>
            <span className="pt-formula-note">
              {formula.length
                ? `${results.length} guide${results.length === 1 ? '' : 's'}`
                : 'Nothing selected: showing all eighteen guides'}
            </span>
          </div>
        </div>

        <div className="pt-grid" role="group" aria-label="Frameworks and languages">
          {frameworks.map((f, i) => (
            <Tile
              key={f.id}
              n={i + 1}
              sub={`${pages[f.id] ?? 0}p`}
              symbol={f.symbol}
              label={shortName(f)}
              group={categoryGroup[f.category]}
              pressed={selected.has(f.id)}
              onToggle={() => toggle(f.id)}
              title={`${f.name}: ${f.category}, ${pages[f.id] ?? 0} pages`}
            />
          ))}
          {LANGS.map((l, i) => (
            <Tile
              key={l}
              n={frameworks.length + i + 1}
              sub={`${frameworks.filter((f) => f.languages.includes(l)).length}f`}
              symbol={languageSymbol[l]}
              label={l}
              group="cobalt"
              pressed={selected.has(`lang:${l}`)}
              onToggle={() => toggle(`lang:${l}`)}
              title={`${l}: ${frameworks.filter((f) => f.languages.includes(l)).length} frameworks`}
            />
          ))}
        </div>
      </section>

      <p className="pt-key brand-annotation">
        n · pages in the guide. Groups: <span style={{ color: 'var(--ac)' }}>multi-agent</span> ·{' '}
        <span style={{ color: 'var(--vd)' }}>data & RAG</span> · <span style={{ color: 'var(--tx)' }}>cloud & enterprise</span> ·{' '}
        <span style={{ color: 'var(--pm)' }}>model-specific</span> · <span style={{ color: 'var(--su-tx)' }}>lightweight</span> ·{' '}
        <span style={{ color: 'var(--co)' }}>languages</span>
      </p>

      <section className="pt-results" aria-label="Matching guides">
        {results.map((f) => (
          <a key={f.id} href={withBase(f.href)} className="pt-result">
            <span className="pt-result-eyebrow">
              {f.category} · {f.symbol} {f.languages.map((l) => languageSymbol[l]).join(' ')}
            </span>
            <span className="pt-result-title">{f.name}</span>
            <span className="pt-result-text">{f.tagline}</span>
            <span className="pt-result-meta">{f.version}</span>
          </a>
        ))}
        {results.length === 0 && (
          <p className="brand-annotation">No guide covers that combination. Deselect a language.</p>
        )}
      </section>
    </div>
  );
}
