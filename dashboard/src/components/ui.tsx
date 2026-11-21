"use client";

import { useState, type ReactNode } from "react";

/** A chart card with a chart/table switch: every value stays reachable without hover. */
export function Card({
  title,
  sub,
  table,
  loading,
  children,
}: {
  title: string;
  sub?: string;
  table?: ReactNode;
  loading?: boolean;
  children: ReactNode;
}) {
  const [showTable, setShowTable] = useState(false);
  return (
    <div className={`card${loading ? " loading" : ""}`} aria-busy={loading}>
      <div className="card-head">
        <div>
          <h3>{title}</h3>
          {sub && <p className="sub">{sub}</p>}
        </div>
        {table && (
          <button className="linkish" onClick={() => setShowTable((v) => !v)} aria-pressed={showTable}>
            {showTable ? "Show chart" : "Show table"}
          </button>
        )}
      </div>
      {showTable ? <div className="table-wrap">{table}</div> : children}
    </div>
  );
}

export function Legend({ items, square }: { items: { label: string; color: string }[]; square?: boolean }) {
  return (
    <div className="legend">
      {items.map((it) => (
        <span key={it.label}>
          <i className={square ? "sq" : undefined} style={{ background: it.color }} />
          {it.label}
        </span>
      ))}
    </div>
  );
}

type TooltipEntry = { name?: string | number; value?: unknown; color?: string; dataKey?: unknown };

/** Tooltip body in the dashboard's tokens; text in ink, identity carried by the dot. */
export function TooltipBox({
  active,
  title,
  payload,
  format,
}: {
  active?: boolean;
  title: ReactNode;
  payload?: readonly TooltipEntry[];
  format: (v: number, key: string) => string;
}) {
  if (!active || !payload?.length) return null;
  return (
    <div className="tt">
      <div className="tt-title">{title}</div>
      {payload.map((p) => (
        <div className="tt-row" key={String(p.dataKey)}>
          <span>
            <i style={{ background: p.color }} />
            {p.name}
          </span>
          <b>{format(Number(p.value), String(p.dataKey))}</b>
        </div>
      ))}
    </div>
  );
}

export const AXIS = { stroke: "var(--axis)", tickLine: false } as const;
export const GRID = { stroke: "var(--grid)", vertical: false } as const;

export const fmt = {
  num: (v: number, d = 1) => v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d }),
  pct: (v: number, d = 1) => `${fmt.num(v, d)}%`,
  signed: (v: number, d = 1) => `${v > 0 ? "+" : v < 0 ? "−" : "±"}${fmt.num(Math.abs(v), d)}`,
  ms: (v: number) => `${Math.round(v).toLocaleString("en-US")} ms`,
  usd: (v: number) => `$${v.toFixed(3)}`,
};

export function QueryError({ error }: { error: { message: string } | null }) {
  return error ? <p className="error">Could not load: {error.message}</p> : null;
}
