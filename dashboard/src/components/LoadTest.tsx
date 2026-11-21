"use client";

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { CONFIGS, LOAD_METRICS } from "@/lib/constants";
import { useTRPC } from "@/trpc/client";
import { AXIS, Card, GRID, Legend, QueryError, TooltipBox, fmt } from "./ui";

type Metric = (typeof LOAD_METRICS)[number];
const METRIC_LABEL: Record<Metric, string> = {
  p95_ms: "p95 latency",
  p50_ms: "Median latency",
  p99_ms: "p99 latency",
  rps: "Throughput (req/s)",
};
const COLORS = ["var(--series-1)", "var(--series-2)", "var(--series-3)", "var(--series-4)"];

export function LoadTest() {
  const trpc = useTRPC();
  const [metric, setMetric] = useState<Metric>("p95_ms");
  const sweep = useQuery({ ...trpc.load.sweep.queryOptions({ metric }), placeholderData: keepPreviousData });
  const capacity = useQuery(trpc.load.capacity.queryOptions());
  const meta = useQuery(trpc.meta.queryOptions());

  const isLatency = metric !== "rps";
  const format = (v: number) => (isLatency ? fmt.ms(v) : `${fmt.num(v, 2)} req/s`);
  const slo = meta.data?.sloP95Ms ?? 2000;

  return (
    <>
      <div className="controls">
        <label>
          Measure
          <select value={metric} onChange={(e) => setMetric(e.target.value as Metric)}>
            {LOAD_METRICS.map((m) => (
              <option key={m} value={m}>
                {METRIC_LABEL[m]}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div style={{ display: "grid", gap: 16 }}>
        <Card
          title={`${METRIC_LABEL[metric]} by concurrent users`}
          sub="Locust on one L4, 0 failures in all runs"
          loading={sweep.isPlaceholderData}
          table={
            sweep.data && (
              <table>
                <thead>
                  <tr>
                    <th className="num">Users</th>
                    {CONFIGS.map((c) => (
                      <th className="num" key={c.id}>
                        {c.label}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {sweep.data.rows.map((r) => (
                    <tr key={r.users}>
                      <td className="num">{r.users}</td>
                      {CONFIGS.map((c) => (
                        <td className="num" key={c.id}>
                          {r[c.id] !== undefined ? format(r[c.id]) : "—"}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            )
          }
        >
          <QueryError error={sweep.error} />
          <Legend items={CONFIGS.map((c, i) => ({ label: c.label, color: COLORS[i] }))} />
          <div className="chart">
            {sweep.data && (
              <ResponsiveContainer>
                <LineChart data={sweep.data.rows} margin={{ top: 16, right: 16, bottom: 4, left: 4 }}>
                  <CartesianGrid {...GRID} />
                  <XAxis dataKey="users" {...AXIS} />
                  <YAxis
                    {...AXIS}
                    tickFormatter={(v: number) => (isLatency ? `${v / 1000} s` : String(v))}
                    width={44}
                  />
                  {metric === "p95_ms" && (
                    <ReferenceLine
                      y={slo}
                      stroke="var(--text-muted)"
                      label={{ value: `target p95 < ${slo / 1000} s`, position: "insideBottomRight", fill: "var(--text-secondary)", fontSize: 12 }}
                    />
                  )}
                  <Tooltip
                    content={(p) => (
                      <TooltipBox active={p.active} payload={p.payload} title={`${p.label} concurrent users`} format={format} />
                    )}
                  />
                  {CONFIGS.map((c, i) => (
                    <Line
                      key={c.id}
                      dataKey={c.id}
                      name={c.label}
                      stroke={COLORS[i]}
                      strokeWidth={2}
                      dot={{ r: 4, fill: COLORS[i], stroke: "var(--surface)", strokeWidth: 2 }}
                      activeDot={{ r: 5, stroke: "var(--surface)", strokeWidth: 2 }}
                      isAnimationActive={false}
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
          <p className="footnote">Users double from 1 to 32; points are evenly spaced.</p>
        </Card>

        <Card title="Capacity and cost" sub={`Cost assumes a fully busy L4 at $${meta.data?.l4PricePerHour ?? "—"}/hour`}>
          <QueryError error={capacity.error} />
          {capacity.data && (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Configuration</th>
                    <th className="num">1 user p50 / p95</th>
                    <th className="num">req/s at p95 &lt; {slo / 1000} s</th>
                    <th className="num">$ per 1,000</th>
                    <th className="num">Peak req/s</th>
                  </tr>
                </thead>
                <tbody>
                  {capacity.data.map((c, i) => (
                    <tr key={c.id} className={i === 0 ? "em" : undefined}>
                      <td>{c.label}</td>
                      <td className="num">
                        {c.singleP50} / {c.singleP95} ms
                      </td>
                      <td className="num">
                        {c.sloRps !== null ? `${fmt.num(c.sloRps, 2)} (${c.sloUsers} users)` : "—"}
                      </td>
                      <td className="num">{c.sloCostPer1k !== null ? fmt.usd(c.sloCostPer1k) : "—"}</td>
                      <td className="num">
                        {c.peakRps !== null ? `${fmt.num(c.peakRps, 2)} (${c.peakUsers})` : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <p className="footnote">
            One multi-task adapter serves as fast as three specialists: vLLM batches LoRA requests, and KV-cache use
            peaks near {capacity.data ? Math.round(Math.max(...capacity.data.map((c) => c.kvCachePeakPct))) : "—"}%,
            so the limit is compute, not memory.
          </p>
        </Card>
      </div>
    </>
  );
}
