"use client";

import { useQuery } from "@tanstack/react-query";
import { useTRPC } from "@/trpc/client";
import { Card, QueryError, fmt } from "./ui";

const show = (v: number, unit: string) => (unit === "%" ? fmt.pct(v) : fmt.num(v));

export function Overview() {
  const trpc = useTRPC();
  const { data, error } = useQuery(trpc.headline.queryOptions());
  if (!data) return <QueryError error={error} />;

  const [hero, ...rest] = data;
  const tiles = rest.slice(0, 3);

  return (
    <>
      <div className="kpis">
        <div className="card tile hero">
          <div className="label">{hero.label} (test split)</div>
          <div className="value">{fmt.num(hero.system)}</div>
          <Delta row={hero} />
        </div>
        {tiles.map((r) => (
          <div className="card tile" key={r.label}>
            <div className="label">{r.label}</div>
            <div className="value">{show(r.system, r.unit)}</div>
            <Delta row={r} />
          </div>
        ))}
      </div>

      <div style={{ marginTop: 16 }}>
        <Card title="Untrained model vs this system" sub="VizWiz test split; a decline counts as “unanswerable”">
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Measure</th>
                  <th className="num">Untrained model</th>
                  <th className="num">This system</th>
                  <th className="num">Change</th>
                  <th>Source</th>
                </tr>
              </thead>
              <tbody>
                {data.map((r) => (
                  <tr key={r.label}>
                    <td>{r.label}</td>
                    <td className="num">{show(r.baseline, r.unit)}</td>
                    <td className="num">
                      <b>{show(r.system, r.unit)}</b>
                    </td>
                    <td className="num">
                      <Change row={r} />
                    </td>
                    <td style={{ color: "var(--text-muted)" }}>{r.source}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="footnote">
            The cost of the safety-first setting is the “answerable questions declined” row: those replies are a
            decline plus retake tips, not a wrong answer.
          </p>
        </Card>
      </div>
    </>
  );
}

type Row = { baseline: number; system: number; unit: string; higherIsBetter: boolean };

function Change({ row }: { row: Row }) {
  const d = row.system - row.baseline;
  const good = row.higherIsBetter ? d > 0 : d < 0;
  const suffix = row.unit === "%" ? " pts" : "";
  return <span className={`delta ${good ? "good" : "bad"}`}>{`${fmt.signed(d)}${suffix}`}</span>;
}

function Delta({ row }: { row: Row }) {
  return (
    <div className="delta">
      <Change row={row} /> vs untrained ({show(row.baseline, row.unit)})
    </div>
  );
}
