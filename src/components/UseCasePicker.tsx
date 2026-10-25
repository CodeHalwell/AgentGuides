import { useMemo, useState } from 'react';
import { frameworks, useCaseLabels } from '../data/frameworks';
import { Chips, withBase } from './FrameworkGrid';

// Only the curated tags in useCaseLabels are offered; other useCases tags are search keywords.
const used = new Set(frameworks.flatMap((f) => f.useCases));
const allUseCases = Object.keys(useCaseLabels).filter((u) => used.has(u));

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
      <div className="mb-4 flex flex-wrap gap-1.5">
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
              {useCaseLabels[u]}
            </button>
          );
        })}
      </div>

      {matches.length === 0 ? (
        <p className="brand-annotation m-0">
          Pick one or more tags above to see recommended frameworks.
        </p>
      ) : (
        <div className="ag-cells">
          {matches.map(({ f, hits }) => (
            <a key={f.id} href={withBase(f.href)} className="ag-cell">
              <Chips f={f} />
              <h4>{f.name}</h4>
              <p>{f.tagline}</p>
              <span className="ag-metric">
                {hits} of {selected.size} matched
              </span>
            </a>
          ))}
        </div>
      )}
    </div>
  );
}
