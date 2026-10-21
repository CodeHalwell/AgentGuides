import { useMemo, useState } from 'react';
import { frameworks, type Framework } from '../data/frameworks';

const categories = ['All', 'Multi-Agent', 'Data & RAG', 'Cloud-Native', 'Model-Specific', 'Enterprise', 'Lightweight'] as const;
const languages = ['All', 'Python', 'TypeScript', '.NET', 'Go'] as const;
const difficulties = ['All', 'Beginner', 'Intermediate', 'Advanced'] as const;

type Category = (typeof categories)[number];
type Language = (typeof languages)[number];
type Difficulty = (typeof difficulties)[number];

// Four hues only (Ink → Oxide → Verdigris → Sulfur); the category label disambiguates.
const groupOf: Record<Framework['category'], string> = {
  'Multi-Agent': 'oxide',
  'Data & RAG': 'verdigris',
  'Cloud-Native': 'ink',
  Enterprise: 'ink',
  'Model-Specific': 'sulfur',
  Lightweight: 'sulfur',
};

const shortLabel: Record<Framework['category'], string> = {
  'Multi-Agent': 'Multi',
  'Data & RAG': 'RAG',
  'Cloud-Native': 'Cloud',
  Enterprise: 'Ent.',
  'Model-Specific': 'Model',
  Lightweight: 'Lite',
};

export function ElementTile({ f, n }: { f: Framework; n: number }) {
  return (
    <span className="ag-tile" data-group={groupOf[f.category]} aria-hidden="true">
      <span>{String(n).padStart(2, '0')}</span>
      <span className="sym">{f.symbol}</span>
      <span className="lab">{shortLabel[f.category]}</span>
    </span>
  );
}

function withBase(href: string) {
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
      <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-center">
        <input
          type="search"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search frameworks, tags, or use cases…"
          className="ag-input flex-1"
        />
        <Select label="Category" value={category} onChange={(v) => setCategory(v as Category)} options={categories} />
        <Select label="Language" value={language} onChange={(v) => setLanguage(v as Language)} options={languages} />
        <Select label="Level" value={difficulty} onChange={(v) => setDifficulty(v as Difficulty)} options={difficulties} />
      </div>

      <p className="brand-label mb-4 text-[var(--sl-color-gray-3)]">
        {filtered.length} of {frameworks.length} frameworks
      </p>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {filtered.map((f) => (
          <a key={f.id} href={withBase(f.href)} className="ag-card group">
            <div className="flex items-start gap-3">
              <ElementTile f={f} n={frameworks.indexOf(f) + 1} />
              <div className="min-w-0">
                <h3 className="m-0 text-lg leading-tight">{f.name}</h3>
                {f.version && <div className="ag-version mt-1">{f.version}</div>}
              </div>
            </div>
            <p className="m-0 text-sm leading-relaxed">{f.tagline}</p>
            <div className="mt-auto flex flex-wrap gap-1.5 pt-1">
              <span className="ag-tag">{f.category}</span>
              <span className="ag-tag">{f.difficulty}</span>
              {f.languages.map((lang) => (
                <span key={lang} className="ag-tag">
                  {lang}
                </span>
              ))}
            </div>
          </a>
        ))}
      </div>

      {filtered.length === 0 && (
        <div className="mt-6 border border-dashed border-[var(--sl-color-gray-5)] p-8 text-center text-[var(--sl-color-gray-3)]">
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
    <label className="brand-label flex items-center gap-2 text-[var(--sl-color-gray-3)]">
      <span className="sr-only sm:not-sr-only">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="ag-input"
      >
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </label>
  );
}
