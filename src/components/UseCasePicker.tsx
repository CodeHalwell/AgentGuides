import { useMemo, useState } from 'react';
import { frameworks, useCaseLabels } from '../data/frameworks';

function withBase(href: string) {
  const base = (import.meta.env.BASE_URL ?? '/').replace(/\/$/, '');
  if (href.startsWith('http')) return href;
  return `${base}${href}`;
}

const allUseCases = Array.from(new Set(frameworks.flatMap((f) => f.useCases))).sort();

export default function UseCasePicker() {
  const [selected, setSelected] = useState<Set<string>>(new Set());

  const toggle = (u: string) => {
    const next = new Set(selected);
    if (next.has(u)) next.delete(u);
    else next.add(u);
    setSelected(next);
  };

  const matches = useMemo(() => {
    if (selected.size === 0) return [];
    return frameworks
      .map((f) => {
        const hits = f.useCases.filter((u) => selected.has(u)).length;
        return { f, hits };
      })
      .filter((m) => m.hits > 0)
      .sort((a, b) => b.hits - a.hits || a.f.name.localeCompare(b.f.name))
      .slice(0, 6);
  }, [selected]);

  return (
    <div className="not-content">
      <div className="mb-4 flex flex-wrap gap-2">
        {allUseCases.map((u) => {
          const active = selected.has(u);
          return (
            <button
              key={u}
              type="button"
              onClick={() => toggle(u)}
              className={[
                'rounded-full border px-3 py-1.5 text-xs font-medium transition-all',
                active
                  ? 'border-[var(--sl-color-accent)] bg-[var(--sl-color-accent)]/20 text-[var(--sl-color-accent-high)]'
                  : 'border-[var(--sl-color-gray-5)] bg-[var(--sl-color-gray-6)] text-[var(--sl-color-gray-2)] hover:border-[var(--sl-color-accent)] hover:text-[var(--sl-color-white)]',
              ].join(' ')}
            >
              {useCaseLabels[u] ?? u}
            </button>
          );
        })}
      </div>

      {matches.length === 0 ? (
        <p className="m-0 text-sm italic text-[var(--sl-color-gray-3)]">
          Pick one or more tags above to see recommended frameworks.
        </p>
      ) : (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {matches.map(({ f, hits }) => (
            <a
              key={f.id}
              href={withBase(f.href)}
              className="ag-card group flex flex-col gap-2 no-underline"
            >
              <div className="flex items-start justify-between">
                <h4 className="m-0 text-sm font-semibold text-[var(--sl-color-white)] group-hover:text-[var(--sl-color-accent-high)]">
                  {f.name}
                </h4>
                <span className="rounded-md bg-[var(--sl-color-accent)]/20 px-2 py-0.5 text-[11px] font-semibold text-[var(--sl-color-accent-high)]">
                  {hits} match{hits > 1 ? 'es' : ''}
                </span>
              </div>
              <p className="m-0 text-xs leading-relaxed text-[var(--sl-color-gray-2)]">
                {f.tagline}
              </p>
            </a>
          ))}
        </div>
      )}
    </div>
  );
}
