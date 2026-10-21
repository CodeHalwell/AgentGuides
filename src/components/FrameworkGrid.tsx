import { useMemo, useState } from 'react';
import { frameworks, categoryGroup, languageSymbol, type Framework } from '../data/frameworks';

const categories = ['All', 'Multi-Agent', 'Data & RAG', 'Cloud-Native', 'Model-Specific', 'Enterprise', 'Lightweight'] as const;
const languages = ['All', 'Python', 'TypeScript', '.NET', 'Go'] as const;
const difficulties = ['All', 'Beginner', 'Intermediate', 'Advanced'] as const;

type Category = (typeof categories)[number];
type Language = (typeof languages)[number];
type Difficulty = (typeof difficulties)[number];

/** Element chips: the framework's symbol in its group colour, then one ink chip per language. */
export function Chips({ f }: { f: Framework }) {
  return (
    <div className="flex flex-wrap gap-1.5" aria-label={`${f.category}; ${f.languages.join(', ')}`}>
      <span className="chip" data-group={categoryGroup[f.category]} title={f.category}>
        {f.symbol}
      </span>
      {f.languages.map((l) => (
        <span key={l} className="chip" data-group="cobalt" title={l}>
          {languageSymbol[l]}
        </span>
      ))}
    </div>
  );
}

export function withBase(href: string) {
  const base = (import.meta.env.BASE_URL ?? '/').replace(/\/$/, '');
  if (href.startsWith('http')) return href;
  return `${base}${href}`;
}

export default function FrameworkGrid() {
  const [query, setQuery] = useState('');
  const [category, setCategory] = useState<Category>('All');
  const [language, setLanguage] = useState<Language>('All');
  const [difficulty, setDifficulty] = useState<Difficulty>('All');

  const filtered = useMemo(() => {
    return frameworks.filter((f: Framework) => {
      if (category !== 'All' && f.category !== category) return false;
      if (language !== 'All' && !f.languages.includes(language as never)) return false;
      if (difficulty !== 'All' && f.difficulty !== difficulty) return false;
      if (query) {
        const q = query.toLowerCase();
        if (
          !f.name.toLowerCase().includes(q) &&
          !f.tagline.toLowerCase().includes(q) &&
          !f.useCases.some((u) => u.includes(q))
        ) {
          return false;
        }
      }
      return true;
    });
  }, [query, category, language, difficulty]);

  return (
    <div className="not-content">
      <div className="mb-4 flex flex-wrap gap-1.5" role="group" aria-label="Category">
        {categories.map((c) => (
          <button key={c} type="button" className="ag-chip" aria-pressed={category === c} onClick={() => setCategory(c)}>
            {c}
          </button>
        ))}
      </div>

      <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-center">
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search frameworks, tags, or use cases…"
          aria-label="Search frameworks"
          className="ag-input flex-1"
        />
        <Select label="Language" value={language} onChange={(v) => setLanguage(v as Language)} options={languages} />
        <Select label="Level" value={difficulty} onChange={(v) => setDifficulty(v as Difficulty)} options={difficulties} />
      </div>

      <p className="brand-label mb-3">
        {filtered.length} of {frameworks.length} frameworks
      </p>

      <div className="ag-cells">
        {filtered.map((f) => (
          <a key={f.id} href={withBase(f.href)} className="ag-cell">
            <Chips f={f} />
            <h3>{f.name}</h3>
            <p>{f.tagline}</p>
            <span className="ag-metric">
              {f.version ?? 'managed service'} · {f.difficulty.toLowerCase()}
            </span>
          </a>
        ))}
      </div>

      {filtered.length === 0 && (
        <div className="brand-annotation mt-6 border border-dashed border-[var(--rl)] p-8 text-center">
          No frameworks match those filters. Try broadening them.
        </div>
      )}
    </div>
  );
}

function Select({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: readonly string[];
}) {
  return (
    <label className="brand-label flex items-center gap-2">
      <span className="sr-only sm:not-sr-only">{label}</span>
      <select value={value} onChange={(e) => onChange(e.target.value)} className="ag-input">
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </label>
  );
}
