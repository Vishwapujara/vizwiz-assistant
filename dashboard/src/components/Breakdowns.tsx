"use client";

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { ERROR_DIMENSIONS } from "@/lib/constants";
import { useTRPC } from "@/trpc/client";
import { AXIS, Card, Legend, QueryError, TooltipBox, fmt } from "./ui";

const BEFORE = { label: "Untrained (E1)", color: "var(--neutral-mark)" };
const AFTER = { label: "Fine-tuned (E5)", color: "var(--series-1)" };

type Dimension = (typeof ERROR_DIMENSIONS)[number];
const DIM_LABEL: Record<Dimension, string> = {
  question_kind: "Question kind",
  quality_label: "Photo-quality label",
  answerer_agreement: "Answerer agreement",
  photo: "Clean photo vs photo issue",
};
const pretty = (s: string) => s.replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());

function HBars({
  rows,
  beforeKey,
  afterKey,
  unit,
}: {
  rows: Record<string, string | number>[];
  beforeKey: string;
  afterKey: string;
  unit: string;
}) {
  return (
    <>
      <Legend square items={[BEFORE, AFTER]} />
      <div className="chart" style={{ height: 40 * rows.length + 40 }}>
        <ResponsiveContainer>
          <BarChart data={rows} layout="vertical" barGap={2} barCategoryGap="22%" margin={{ top: 4, right: 24, bottom: 4, left: 8 }}>
            <CartesianGrid stroke="var(--grid)" horizontal={false} />
            <XAxis type="number" domain={[0, 100]} ticks={[0, 25, 50, 75, 100]} {...AXIS} />
            <YAxis type="category" dataKey="name" width={130} {...AXIS} />
            <Tooltip
              cursor={{ fill: "var(--grid)", opacity: 0.5 }}
              content={(p) => (
                <TooltipBox
                  active={p.active}
                  payload={p.payload}
                  title={`${p.label} (n = ${Number(p.payload?.[0]?.payload?.n ?? 0).toLocaleString("en-US")})`}
                  format={(v) => `${fmt.num(v)}${unit}`}
                />
              )}
            />
            <Bar dataKey={beforeKey} name={BEFORE.label} fill={BEFORE.color} maxBarSize={12} radius={[0, 4, 4, 0]} isAnimationActive={false} />
            <Bar dataKey={afterKey} name={AFTER.label} fill={AFTER.color} maxBarSize={12} radius={[0, 4, 4, 0]} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </>
  );
}

export function Breakdowns() {
  const trpc = useTRPC();
  const [dimension, setDimension] = useState<Dimension>("question_kind");
  const errors = useQuery({ ...trpc.errors.queryOptions({ dimension }), placeholderData: keepPreviousData });
  const quality = useQuery(trpc.quality.queryOptions());

  const errRows = (errors.data ?? []).map((r) => ({ ...r, name: pretty(r.grp) }));
  const qRows = (quality.data ?? []).map((r) => ({ ...r, name: pretty(r.label) }));

  return (
    <>
      <div className="controls">
        <label>
          Answering accuracy by
          <select value={dimension} onChange={(e) => setDimension(e.target.value as Dimension)}>
            {ERROR_DIMENSIONS.map((d) => (
              <option key={d} value={d}>
                {DIM_LABEL[d]}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="grid-2">
        <Card
          title={`Answering accuracy by ${DIM_LABEL[dimension].toLowerCase()}`}
          sub="Answerable test questions, VizWiz accuracy"
          loading={errors.isPlaceholderData}
          table={
            <table>
              <thead>
                <tr>
                  <th>Group</th>
                  <th className="num">n</th>
                  <th className="num">{BEFORE.label}</th>
                  <th className="num">{AFTER.label}</th>
                  <th className="num">Change</th>
                </tr>
              </thead>
              <tbody>
                {errRows.map((r) => (
                  <tr key={r.grp}>
                    <td>{r.name}</td>
                    <td className="num">{r.n.toLocaleString("en-US")}</td>
                    <td className="num">{fmt.num(r.accuracy_before)}</td>
                    <td className="num">{fmt.num(r.accuracy_after)}</td>
                    <td className="num">{fmt.signed(r.accuracy_after - r.accuracy_before)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          }
        >
          <QueryError error={errors.error} />
          <HBars rows={errRows} beforeKey="accuracy_before" afterKey="accuracy_after" unit="" />
          <p className="footnote">A photo can carry several quality labels, so those groups overlap.</p>
        </Card>

        <Card
          title="Photo-quality detection, F1 per label"
          sub="Test split, labels ordered by how common they are"
          table={
            <table>
              <thead>
                <tr>
                  <th>Label</th>
                  <th className="num">Support</th>
                  <th className="num">F1 before</th>
                  <th className="num">F1 after</th>
                  <th className="num">Precision</th>
                  <th className="num">Recall</th>
                </tr>
              </thead>
              <tbody>
                {qRows.map((r) => (
                  <tr key={r.label}>
                    <td>{r.name}</td>
                    <td className="num">{r.support.toLocaleString("en-US")}</td>
                    <td className="num">{fmt.num(r.f1_before)}</td>
                    <td className="num">{fmt.num(r.f1_after)}</td>
                    <td className="num">{fmt.num(r.precision)}</td>
                    <td className="num">{fmt.num(r.recall)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          }
        >
          <QueryError error={quality.error} />
          <HBars rows={qRows.map((r) => ({ ...r, n: r.support }))} beforeKey="f1_before" afterKey="f1_after" unit="" />
          <p className="footnote">Rare labels (obstruction, dark, bright) are found less often: recall stays under 50%.</p>
        </Card>
      </div>
    </>
  );
}
