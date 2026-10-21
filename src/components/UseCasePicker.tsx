import { useMemo, useState } from 'react';
import { frameworks, useCaseLabels } from '../data/frameworks';
import { ElementTile } from './FrameworkGrid';

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
              aria-pressed={active}
              className="ag-chip"
            >
              {useCaseLabels[u] ?? u}
            </button>
          );
        })}
      </div>

      {matches.length === 0 ? (
        <p className="brand-annotation m-0">
          Pick one or more tags above to see recommended frameworks.
        </p>
      ) : (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {matches.map(({ f, hits }) => (
            <a key={f.id} href={withBase(f.href)} className="ag-card group">
              <div className="flex items-start gap-3">
                <ElementTile f={f} n={frameworks.indexOf(f) + 1} />
                <div className="min-w-0">
                  <h4 className="m-0 text-base leading-tight">{f.name}</h4>
                  <div className="brand-annotation mt-1">
                    {hits} match{hits > 1 ? 'es' : ''}
                  </div>
                </div>
              </div>
              <p className="m-0 text-xs leading-relaxed">{f.tagline}</p>
            </a>
          ))}
        </div>
      )}
    </div>
  );
}
