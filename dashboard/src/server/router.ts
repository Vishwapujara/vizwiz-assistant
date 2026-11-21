import { z } from "zod";
import { all, get } from "./db";
import { publicProcedure, router } from "./trpc";
import { CONFIGS, ERROR_DIMENSIONS, EXPERIMENTS, LOAD_METRICS } from "@/lib/constants";


type Meta = { key: string; value: string };
type LoadRow = { config: string; users: number; rps: number; p50_ms: number; p95_ms: number; p99_ms: number; failures: number; requests: number; kv_cache_peak_pct: number };

const meta = () => Object.fromEntries(all<Meta>("SELECT key, value FROM meta").map((m) => [m.key, m.value]));

export const appRouter = router({
  meta: publicProcedure.query(() => {
    const m = meta();
    return {
      model: m.model,
      gpu: m.gpu,
      chosenThreshold: Number(m.chosen_threshold),
      thresholdRule: m.threshold_rule,
      l4PricePerHour: Number(m.l4_price_per_hour),
      sloP95Ms: Number(m.slo_p95_ms),
      builtUtc: m.built_utc,
    };
  }),

  headline: publicProcedure.query(() =>
    all<{ label: string; baseline: number; system: number; unit: "score" | "%"; higher_is_better: number; source: string }>(
      "SELECT label, baseline, system, unit, higher_is_better, source FROM headline ORDER BY pos",
    ).map((r) => ({ ...r, higherIsBetter: r.higher_is_better === 1 })),
  ),

  threshold: router({
    sweep: publicProcedure.query(() =>
      all<{
        threshold: number; n: number; vizwiz_accuracy: number; answered_pct: number;
        wrong_answers_pct_of_all: number; wrongly_declined_pct_of_answerable: number; missed_unanswerable_pct: number;
      }>(
        `SELECT threshold, n, vizwiz_accuracy, answered_pct, wrong_answers_pct_of_all,
                wrongly_declined_pct_of_answerable, missed_unanswerable_pct
           FROM threshold_sweep ORDER BY threshold`,
      ),
    ),
  }),

  load: router({
    // Pivoted for a line chart: one row per user count, one column per configuration.
    sweep: publicProcedure.input(z.object({ metric: z.enum(LOAD_METRICS) })).query(({ input }) => {
      const rows = all<{ config: string; users: number; value: number }>(
        `SELECT config, users, ${input.metric} AS value FROM load_sweep ORDER BY users`,
      );
      const byUsers = new Map<number, Record<string, number>>();
      for (const r of rows) {
        const row = byUsers.get(r.users) ?? { users: r.users };
        row[r.config] = r.value;
        byUsers.set(r.users, row);
      }
      return { metric: input.metric, rows: [...byUsers.values()] };
    }),

    // Capacity within the latency target and the cost per 1,000 requests it implies.
    capacity: publicProcedure.query(() => {
      const m = meta();
      const slo = Number(m.slo_p95_ms);
      const price = Number(m.l4_price_per_hour);
      return CONFIGS.map(({ id, label }) => {
        const rows = all<LoadRow>("SELECT * FROM load_sweep WHERE config = ? ORDER BY users", id);
        const atSlo = rows.filter((r) => r.p95_ms < slo).sort((a, b) => b.rps - a.rps)[0];
        const peak = [...rows].sort((a, b) => b.rps - a.rps)[0];
        const single = rows.find((r) => r.users === 1);
        const costPer1k = (rps: number) => (price / (rps * 3600)) * 1000;
        return {
          id,
          label,
          singleP50: single?.p50_ms ?? null,
          singleP95: single?.p95_ms ?? null,
          sloRps: atSlo?.rps ?? null,
          sloUsers: atSlo?.users ?? null,
          sloCostPer1k: atSlo ? costPer1k(atSlo.rps) : null,
          peakRps: peak?.rps ?? null,
          peakUsers: peak?.users ?? null,
          peakCostPer1k: peak ? costPer1k(peak.rps) : null,
          requests: rows.reduce((s, r) => s + r.requests, 0),
          failures: rows.reduce((s, r) => s + r.failures, 0),
          kvCachePeakPct: Math.max(...rows.map((r) => r.kv_cache_peak_pct)),
        };
      });
    }),
  }),

  experiments: router({
    compare: publicProcedure.input(z.object({ experiment: z.enum(EXPERIMENTS) })).query(({ input }) =>
      all<{ task: string; metric: string; split: string; n: number; reference: string; run: string; value: number; diff: number; ci_lo: number; ci_hi: number; p_better: number }>(
        `SELECT task, metric, split, n, reference, run, value, diff, ci_lo, ci_hi, p_better
           FROM comparisons WHERE experiment = ?
          ORDER BY CASE task WHEN 'answer' THEN 0 WHEN 'answerability' THEN 1 ELSE 2 END, run`,
        input.experiment,
      ),
    ),
  }),

  quality: publicProcedure.query(() =>
    all<{ label: string; support: number; f1_before: number; f1_after: number; precision: number; recall: number }>(
      `SELECT a.label, a.support, ROUND(b.f1 * 100, 1) AS f1_before, ROUND(a.f1 * 100, 1) AS f1_after,
              ROUND(a.precision * 100, 1) AS precision, ROUND(a.recall * 100, 1) AS recall
         FROM quality_labels a JOIN quality_labels b ON b.label = a.label AND b.run = 'e1'
        WHERE a.run = 'e5' ORDER BY a.support DESC`,
    ),
  ),

  errors: publicProcedure.input(z.object({ dimension: z.enum(ERROR_DIMENSIONS) })).query(({ input }) =>
    all<{ grp: string; n: number; accuracy_before: number; accuracy_after: number }>(
      "SELECT grp, n, accuracy_before, accuracy_after FROM error_groups WHERE dimension = ? ORDER BY n DESC",
      input.dimension,
    ),
  ),

  runs: publicProcedure.query(() =>
    all<{ run_id: string; created_utc: string; git_commit: string; split: string; dtype: string }>(
      "SELECT run_id, created_utc, git_commit, split, dtype FROM runs ORDER BY created_utc",
    ),
  ),

  health: publicProcedure.query(() => ({ ok: get<{ n: number }>("SELECT COUNT(*) AS n FROM headline")!.n > 0 })),
});

export type AppRouter = typeof appRouter;
