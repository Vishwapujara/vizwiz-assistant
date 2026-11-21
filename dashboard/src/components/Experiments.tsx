"use client";

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { EXPERIMENTS } from "@/lib/constants";
import { useTRPC } from "@/trpc/client";
import { Card, QueryError, fmt } from "./ui";

type Experiment = (typeof EXPERIMENTS)[number];

const INFO: Record<Experiment, { label: string; question: string; reference: string; margin?: number; verdict: string }> = {
  "e5-multitask": {
    label: "E5: one multi-task adapter vs three specialists",
    question: "Does one shared adapter lose anything against one adapter per task?",
    reference: "E2 specialists",
    verdict: "It matches all three, so one adapter is served (D-056).",
  },
  "e4-fp8": {
    label: "E4: FP8 base model vs bf16",
    question: "Does FP8 keep quality within the pre-set −1 point margin?",
    reference: "bf16",
    margin: -1,
    verdict: "Quality F1's interval reaches −1.16, past the margin, so bf16 stays (D-058).",
  },
  resolution: {
    label: "Image resolution (zero-shot, val)",
    question: "How much does image size matter?",
    reference: "0.52 MP",
    verdict: "0.52 MP matches 1.05 MP at twice the speed, so it is the default (D-050).",
  },
};

const TASK: Record<string, string> = { answer: "Answering", answerability: "Answerability", quality: "Quality" };
const RUN: Record<string, string> = {
  e5: "E5 multi-task",
  e5_fp8: "FP8",
  "262144": "0.26 MP",
  "1048576": "1.05 MP",
};
const METRIC: Record<string, string> = { vqa_accuracy: "accuracy", average_precision: "AP", macro_f1: "macro-F1" };

export function Experiments() {
  const trpc = useTRPC();
  const [experiment, setExperiment] = useState<Experiment>("e5-multitask");
  const { data, error, isPlaceholderData } = useQuery({
    ...trpc.experiments.compare.queryOptions({ experiment }),
    placeholderData: keepPreviousData,
  });
  const info = INFO[experiment];

  // The forest plot shows the contrast under test; the untrained baseline (E1) is in the table.
  const points = (data ?? [])
    .filter((r) => r.run !== r.reference && r.run !== "e1")
    .map((r) => ({
      ...r,
      name: `${TASK[r.task]} · ${RUN[r.run] ?? r.run}`,
      err: [r.diff - r.ci_lo, r.ci_hi - r.diff] as [number, number],
    }));
  const reach = Math.max(1.5, ...points.map((p) => Math.max(Math.abs(p.ci_lo), Math.abs(p.ci_hi))));
  const lim = Math.ceil(reach * 2) / 2;

  return (
    <>
      <div className="controls">
        <label>
          Experiment
          <select value={experiment} onChange={(e) => setExperiment(e.target.value as Experiment)}>
            {EXPERIMENTS.map((x) => (
              <option key={x} value={x}>
                {INFO[x].label}
              </option>
            ))}
          </select>
        </label>
      </div>
      <Card
        title={info.question}
        sub={`Paired difference vs ${info.reference}, points with bootstrap 95% intervals`}
        loading={isPlaceholderData}
        table={
          data && (
            <table>
              <thead>
                <tr>
                  <th>Task</th>
                  <th>Run</th>
                  <th className="num">Value</th>
                  <th className="num">Difference</th>
                  <th className="num">95% interval</th>
                  <th className="num">P(better)</th>
                  <th className="num">n</th>
                </tr>
              </thead>
              <tbody>
                {data.map((r) => (
                  <tr key={r.task + r.run} className={r.run === r.reference ? "em" : undefined}>
                    <td>
                      {TASK[r.task]} ({METRIC[r.metric] ?? r.metric}, {r.split})
                    </td>
                    <td>{r.run === r.reference ? `${info.reference} (reference)` : r.run === "e1" ? "Untrained (E1)" : RUN[r.run] ?? r.run}</td>
                    <td className="num">{fmt.num(r.value, 2)}</td>
                    <td className="num">{r.run === r.reference ? "—" : fmt.signed(r.diff, 2)}</td>
                    <td className="num">
                      {r.run === r.reference ? "—" : `[${fmt.signed(r.ci_lo, 2)}, ${fmt.signed(r.ci_hi, 2)}]`}
                    </td>
                    <td className="num">{r.run === r.reference ? "—" : fmt.num(r.p_better, 2)}</td>
                    <td className="num">{r.n.toLocaleString("en-US")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )
        }
      >
        <QueryError error={error} />
        <Forest points={points} lim={lim} margin={info.margin} />
        <p className="footnote">{info.verdict} Right of zero is better than the reference.</p>
      </Card>
    </>
  );
}

type Point = { name: string; diff: number; ci_lo: number; ci_hi: number; value: number };

/** Forest plot: one row per contrast, dot = paired difference, whisker = 95% interval. */
function Forest({ points, lim, margin }: { points: Point[]; lim: number; margin?: number }) {
  const x = (v: number) => `${((v + lim) / (2 * lim)) * 100}%`;
  const step = lim <= 2 ? 0.5 : 1;
  const ticks: number[] = [];
  for (let t = -lim; t <= lim + 1e-9; t += step) ticks.push(Math.round(t * 10) / 10);
  return (
    <div className="forest" role="list">
      {points.map((p) => {
        const desc = `${p.name}: ${fmt.signed(p.diff, 2)}, 95% interval ${fmt.signed(p.ci_lo, 2)} to ${fmt.signed(p.ci_hi, 2)}`;
        return (
          <div className="forest-row" role="listitem" key={p.name}>
            <div className="forest-label">{p.name}</div>
            <div className="forest-track" tabIndex={0} aria-label={desc}>
              {ticks.map((t) => (
                <span key={t} className={`forest-grid${t === 0 ? " zero" : ""}`} style={{ left: x(t) }} />
              ))}
              {margin !== undefined && <span className="forest-margin" style={{ left: x(margin) }} />}
              <span className="forest-ci" style={{ left: x(p.ci_lo), width: `calc(${x(p.ci_hi)} - ${x(p.ci_lo)})` }} />
              <span className="forest-dot" style={{ left: x(p.diff) }} />
              <span className="forest-tip" style={{ left: x(p.diff) }}>
                <b>{fmt.signed(p.diff, 2)}</b> [{fmt.signed(p.ci_lo, 2)}, {fmt.signed(p.ci_hi, 2)}]
              </span>
            </div>
          </div>
        );
      })}
      <div className="forest-row">
        <div className="forest-label" />
        <div className="forest-axis">
          {ticks.map((t) => (
            <span key={t} style={{ left: x(t) }}>
              {t === 0 ? "0" : fmt.signed(t, 1)}
            </span>
          ))}
          {margin !== undefined && (
            <span className="forest-margin-label" style={{ left: x(margin) }}>
              margin
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
