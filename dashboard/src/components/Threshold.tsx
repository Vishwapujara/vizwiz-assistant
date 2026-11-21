"use client";

import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useTRPC } from "@/trpc/client";
import { AXIS, Card, GRID, Legend, QueryError, TooltipBox, fmt } from "./ui";

const SERIES = [
  { key: "vizwiz_accuracy", label: "Official accuracy", color: "var(--series-1)" },
  { key: "wrong_answers_pct_of_all", label: "Wrong answers (% of all)", color: "var(--series-2)" },
  { key: "wrongly_declined_pct_of_answerable", label: "Answerable declined (%)", color: "var(--series-3)" },
  { key: "missed_unanswerable_pct", label: "Unanswerable answered (%)", color: "var(--series-4)" },
] as const;

export function Threshold() {
  const trpc = useTRPC();
  const sweep = useQuery(trpc.threshold.sweep.queryOptions());
  const meta = useQuery(trpc.meta.queryOptions());
  const [picked, setPicked] = useState<number | null>(null);

  if (!sweep.data || !meta.data) return <QueryError error={sweep.error ?? meta.error} />;
  const rows = sweep.data;
  const chosen = meta.data.chosenThreshold;
  const idx = picked ?? Math.max(0, rows.findIndex((r) => r.threshold === chosen));
  const at = rows[idx];
  const best = rows.reduce((a, b) => (b.vizwiz_accuracy > a.vizwiz_accuracy ? b : a));

  return (
    <div className="grid-2">
      <Card
        title="What each decline threshold trades"
        sub={`Validation split, n = ${rows[0].n.toLocaleString("en-US")}, served PNG path`}
        table={
          <table>
            <thead>
              <tr>
                <th className="num">Threshold</th>
                {SERIES.map((s) => (
                  <th className="num" key={s.key}>
                    {s.label}
                  </th>
                ))}
                <th className="num">Answered (%)</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.threshold} className={r.threshold === chosen ? "em" : undefined}>
                  <td className="num">{r.threshold.toFixed(2)}</td>
                  {SERIES.map((s) => (
                    <td className="num" key={s.key}>
                      {fmt.num(r[s.key])}
                    </td>
                  ))}
                  <td className="num">{fmt.num(r.answered_pct)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        }
      >
        <Legend items={SERIES.map((s) => ({ label: s.label, color: s.color }))} />
        <div className="chart">
          <ResponsiveContainer>
            <LineChart data={rows} margin={{ top: 16, right: 16, bottom: 4, left: -8 }}>
              <CartesianGrid {...GRID} />
              <XAxis
                dataKey="threshold"
                type="number"
                domain={[0, 1]}
                ticks={[0, 0.2, 0.4, 0.6, 0.8, 1]}
                tickFormatter={(v: number) => v.toFixed(1)}
                {...AXIS}
              />
              <YAxis domain={[0, 100]} ticks={[0, 25, 50, 75, 100]} {...AXIS} />
              <ReferenceLine
                x={chosen}
                stroke="var(--text-muted)"
                label={{ value: `served ${chosen.toFixed(2)}`, position: "top", fill: "var(--text-secondary)", fontSize: 12 }}
              />
              <Tooltip
                content={(p) => (
                  <TooltipBox
                    active={p.active}
                    payload={p.payload}
                    title={`P(unanswerable) ≥ ${Number(p.label).toFixed(2)} declines`}
                    format={(v) => fmt.num(v)}
                  />
                )}
              />
              {SERIES.map((s) => (
                <Line
                  key={s.key}
                  dataKey={s.key}
                  name={s.label}
                  stroke={s.color}
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4, stroke: "var(--surface)", strokeWidth: 2 }}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Card>

      <Card title="Try a threshold" sub={`Rule: ${meta.data.thresholdRule}`}>
        <div className="controls">
          <label>
            Decline when P(unanswerable) ≥ <b style={{ color: "var(--text-primary)" }}>{at.threshold.toFixed(2)}</b>
            <input
              type="range"
              min={0}
              max={rows.length - 1}
              step={1}
              value={idx}
              onChange={(e) => setPicked(Number(e.target.value))}
              aria-valuetext={at.threshold.toFixed(2)}
            />
          </label>
        </div>
        <div className="kpis">
          <Stat label="Official accuracy" value={fmt.num(at.vizwiz_accuracy)} ref_={best.vizwiz_accuracy} cur={at.vizwiz_accuracy} />
          <Stat label="Questions answered" value={fmt.pct(at.answered_pct)} />
          <Stat label="Wrong answers, share of all" value={fmt.pct(at.wrong_answers_pct_of_all)} />
          <Stat label="Answerable questions declined" value={fmt.pct(at.wrongly_declined_pct_of_answerable)} />
          <Stat label="Unanswerable questions answered" value={fmt.pct(at.missed_unanswerable_pct)} />
        </div>
        <p className="footnote">
          Served value {chosen.toFixed(2)} maximises official accuracy on val. Lower thresholds decline more: fewer wrong
          answers, more answerable questions sent back with retake tips.
        </p>
      </Card>
    </div>
  );
}

function Stat({ label, value, ref_, cur }: { label: string; value: string; ref_?: number; cur?: number }) {
  return (
    <div className="tile" style={{ padding: "4px 0" }}>
      <div className="label">{label}</div>
      <div className="value" style={{ fontSize: 22 }}>
        {value}
      </div>
      {ref_ !== undefined && cur !== undefined && (
        <div className="delta">{cur === ref_ ? "best on val" : `${fmt.signed(cur - ref_)} vs best`}</div>
      )}
    </div>
  );
}
