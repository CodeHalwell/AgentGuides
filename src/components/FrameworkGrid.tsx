import { useMemo, useState } from 'react';
import { frameworks, type Framework } from '../data/frameworks';

const categories = ['All', 'Multi-Agent', 'Data & RAG', 'Cloud-Native', 'Model-Specific', 'Enterprise', 'Lightweight'] as const;
const languages = ['All', 'Python', 'TypeScript', '.NET', 'Go'] as const;
const difficulties = ['All', 'Beginner', 'Intermediate', 'Advanced'] as const;

type Category = (typeof categories)[number];
type Language = (typeof languages)[number];
type Difficulty = (typeof difficulties)[number];

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
          className="flex-1 rounded-lg border border-[var(--sl-color-gray-5)] bg-[var(--sl-color-gray-6)] px-4 py-2 text-sm text-[var(--sl-color-white)] placeholder:text-[var(--sl-color-gray-3)] focus:border-[var(--sl-color-accent)] focus:outline-none focus:ring-2 focus:ring-[var(--sl-color-accent)]/30"
        />
        <Select label="Category" value={category} onChange={(v) => setCategory(v as Category)} options={categories} />
        <Select label="Language" value={language} onChange={(v) => setLanguage(v as Language)} options={languages} />
        <Select label="Level" value={difficulty} onChange={(v) => setDifficulty(v as Difficulty)} options={difficulties} />
      </div>

      <p className="mb-4 text-sm text-[var(--sl-color-gray-3)]">
        Showing <strong>{filtered.length}</strong> of {frameworks.length} frameworks
      </p>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {filtered.map((f) => (
          <a
            key={f.id}
            href={withBase(f.href)}
            className="ag-card group flex flex-col gap-3 no-underline"
          >
            <div className="flex items-start justify-between gap-2">
              <h3 className="m-0 text-base font-semibold text-[var(--sl-color-white)] group-hover:text-[var(--sl-color-accent-high)]">
                {f.name}
              </h3>
              {f.version && (
                <span className="rounded-md border border-[var(--sl-color-gray-5)] bg-[var(--sl-color-gray-5)]/40 px-2 py-0.5 font-mono text-[11px] text-[var(--sl-color-gray-2)]">
                  {f.version}
                </span>
              )}
            </div>
            <p className="m-0 text-sm leading-relaxed text-[var(--sl-color-gray-2)]">
              {f.tagline}
            </p>
            <div className="mt-auto flex flex-wrap gap-1.5 pt-2">
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
        <div className="mt-6 rounded-lg border border-dashed border-[var(--sl-color-gray-5)] p-8 text-center text-[var(--sl-color-gray-3)]">
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
    <label className="flex items-center gap-2 text-xs font-medium text-[var(--sl-color-gray-2)]">
      <span className="sr-only sm:not-sr-only">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-lg border border-[var(--sl-color-gray-5)] bg-[var(--sl-color-gray-6)] px-3 py-2 text-sm text-[var(--sl-color-white)] focus:border-[var(--sl-color-accent)] focus:outline-none focus:ring-2 focus:ring-[var(--sl-color-accent)]/30"
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
