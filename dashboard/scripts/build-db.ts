// Builds data/results.db from the committed results/*.json (D-067).
// Runs before `next dev` and `next build`; the app only ever reads the database.
//   node scripts/build-db.ts [results-dir] [db-path]
import { existsSync, mkdirSync, readFileSync, readdirSync, rmSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { DatabaseSync } from "node:sqlite";

const here = dirname(new URL(import.meta.url).pathname);
const resultsDir = resolve(process.argv[2] ?? join(here, "..", "..", "results"));
const dbPath = resolve(process.argv[3] ?? join(here, "..", "data", "results.db"));

// Assumption used for cost per 1,000 requests (D-066): GCP g2-standard-4 on demand.
const L4_PRICE_PER_HOUR = 0.707;
const SLO_P95_MS = 2000;

type Json = any; // results files are trusted, committed artifacts

const readJson = (rel: string): Json => JSON.parse(readFileSync(join(resultsDir, rel), "utf8"));

if (!existsSync(join(resultsDir, "eval"))) {
  // A CLI deploy from dashboard/ uploads no ../results: keep the database built locally.
  if (existsSync(dbPath)) {
    console.log(`No results at ${resultsDir}; keeping the existing ${dbPath}`);
    process.exit(0);
  }
  console.error(`No results found at ${resultsDir}`);
  process.exit(1);
}

mkdirSync(dirname(dbPath), { recursive: true });
rmSync(dbPath, { force: true });
const db = new DatabaseSync(dbPath);

db.exec(`
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE runs (
  run_id TEXT PRIMARY KEY, created_utc TEXT, git_commit TEXT, model TEXT,
  adapter TEXT, split TEXT, gpu TEXT, dtype TEXT, max_pixels INTEGER);
CREATE TABLE headline (
  pos INTEGER PRIMARY KEY, label TEXT NOT NULL, baseline REAL NOT NULL, system REAL NOT NULL,
  unit TEXT NOT NULL, higher_is_better INTEGER NOT NULL, source TEXT NOT NULL);
CREATE TABLE comparisons (
  experiment TEXT NOT NULL, task TEXT NOT NULL, metric TEXT NOT NULL, split TEXT NOT NULL,
  n INTEGER NOT NULL, reference TEXT NOT NULL, run TEXT NOT NULL, value REAL NOT NULL,
  diff REAL NOT NULL, ci_lo REAL NOT NULL, ci_hi REAL NOT NULL, p_better REAL NOT NULL);
CREATE TABLE threshold_sweep (
  threshold REAL PRIMARY KEY, n INTEGER, vizwiz_accuracy REAL, answered_pct REAL,
  wrong_answers_pct_of_all REAL, wrong_among_answered_pct REAL,
  wrongly_declined_pct_of_answerable REAL, missed_unanswerable_pct REAL);
CREATE TABLE load_sweep (
  config TEXT NOT NULL, users INTEGER NOT NULL, requests INTEGER, failures INTEGER, rps REAL,
  mean_ms REAL, p50_ms REAL, p90_ms REAL, p95_ms REAL, p99_ms REAL, kv_cache_peak_pct REAL,
  stage_quality_ms REAL, stage_answerability_ms REAL, stage_answer_ms REAL,
  PRIMARY KEY (config, users));
CREATE TABLE quality_labels (
  run TEXT NOT NULL, label TEXT NOT NULL, precision REAL, recall REAL, f1 REAL, support INTEGER,
  PRIMARY KEY (run, label));
CREATE TABLE demo_samples (
  pos INTEGER NOT NULL, id TEXT PRIMARY KEY, file TEXT NOT NULL, question TEXT NOT NULL,
  reference TEXT, shows TEXT, source TEXT, source_url TEXT, author TEXT, license TEXT, changes TEXT);
CREATE TABLE demo_answers (
  id TEXT PRIMARY KEY, status TEXT NOT NULL, message TEXT NOT NULL, answer TEXT, issues TEXT NOT NULL,
  unanswerable_score REAL, tips TEXT NOT NULL, latency_ms REAL, created_utc TEXT, git_commit TEXT, setup TEXT);
CREATE TABLE error_groups (
  dimension TEXT NOT NULL, grp TEXT NOT NULL, n INTEGER NOT NULL,
  accuracy_before REAL NOT NULL, accuracy_after REAL NOT NULL, PRIMARY KEY (dimension, grp));
`);

const insert = (table: string, row: Record<string, string | number | null>) => {
  const cols = Object.keys(row);
  db.prepare(`INSERT INTO ${table} (${cols.join(",")}) VALUES (${cols.map(() => "?").join(",")})`).run(
    ...Object.values(row),
  );
};

db.exec("BEGIN");

// --- runs -----------------------------------------------------------------
for (const dir of readdirSync(join(resultsDir, "eval"))) {
  const file = join("eval", dir, "metrics.json");
  if (!existsSync(join(resultsDir, file))) continue;
  const m = readJson(file);
  if (!m.run_id) continue;
  insert("runs", {
    run_id: m.run_id, created_utc: m.created_utc ?? null, git_commit: m.git_commit ?? null,
    model: m.model ?? null, adapter: m.adapter ?? null, split: m.split ?? null,
    gpu: m.gpu ?? null, dtype: m.dtype ?? null, max_pixels: m.max_pixels ?? null,
  });
}

// --- headline: untrained model vs the served system, test split ----------
// End-to-end numbers come from the serving check (tune_report.json, test). "This system"
// is the served setting: threshold 0.10, quality as tips only (D-063, D-064).
const tune = readJson("serve/serve-check/tune_report.json");
const e1 = readJson("eval/e1-baseline/metrics.json").tasks;
const e5 = readJson("eval/e5-multi-lora/metrics.json").tasks;
const base = tune.test.never_decline;
const sys = tune.test.without_quality_rule;
const headline: [string, number, number, string, number, string][] = [
  ["Official VizWiz accuracy, end to end", base.vizwiz_accuracy, sys.vizwiz_accuracy, "score", 1, "serve-check (test)"],
  ["Wrong answers, share of all questions", base.wrong_answers_pct_of_all, sys.wrong_answers_pct_of_all, "%", 0, "serve-check (test)"],
  ["Unanswerable questions answered anyway", base.missed_unanswerable_pct, sys.missed_unanswerable_pct, "%", 0, "serve-check (test)"],
  ["Answerable questions declined", base.wrongly_declined_pct_of_answerable, sys.wrongly_declined_pct_of_answerable, "%", 0, "serve-check (test)"],
  ["Answering accuracy on answerable questions", e1.answer.metrics.vqa_accuracy, e5.answer.metrics.vqa_accuracy, "score", 1, "E1 vs E5 (test)"],
  ["Detecting unanswerable questions (AP)", e1.answerability.metrics.average_precision, e5.answerability.metrics.average_precision, "score", 1, "E1 vs E5 (test)"],
  ["Photo-quality issues (macro-F1, 7 labels)", e1.quality.metrics.macro_f1, e5.quality.metrics.macro_f1, "score", 1, "E1 vs E5 (test)"],
];
headline.forEach(([label, b, s, unit, hib, source], pos) =>
  insert("headline", { pos, label, baseline: b, system: s, unit, higher_is_better: hib, source }),
);

// --- paired comparisons with bootstrap CIs --------------------------------
const addCompare = (experiment: string, c: Json) => {
  for (const [run, r] of Object.entries<Json>(c.runs)) {
    insert("comparisons", {
      experiment, task: c.task, metric: c.metric, split: c.split, n: c.n, reference: c.reference,
      run, value: r.value, diff: r.diff_vs_reference, ci_lo: r.diff_ci95[0], ci_hi: r.diff_ci95[1],
      p_better: r.p_better,
    });
  }
};
for (const task of ["answer", "answerability", "quality"]) {
  addCompare("e5-multitask", readJson(`eval/e5-multi-lora/compare_${task}.json`));
  addCompare("e4-fp8", readJson(`eval/e4-fp8-e5/compare_${task}.json`));
}
const resolution = readJson("eval/resolution/resolution_summary.json").compare;
for (const c of Object.values<Json>(resolution)) addCompare("resolution", c);

// --- decline-threshold sweep (val, served PNG path, D-066) ----------------
const tunePng = readJson("serve/serve-check/tune_report_png.json");
for (const r of tunePng.val.sweep) {
  if (!r.use_quality_rule) continue;
  insert("threshold_sweep", {
    threshold: r.threshold, n: r.n, vizwiz_accuracy: r.vizwiz_accuracy, answered_pct: r.answered_pct,
    wrong_answers_pct_of_all: r.wrong_answers_pct_of_all, wrong_among_answered_pct: r.wrong_among_answered_pct,
    wrongly_declined_pct_of_answerable: r.wrongly_declined_pct_of_answerable,
    missed_unanswerable_pct: r.missed_unanswerable_pct,
  });
}

// --- load-test sweep (Locust on one L4) ------------------------------------
const sweep = readFileSync(join(resultsDir, "bench/load-test/sweep.jsonl"), "utf8")
  .split("\n").filter(Boolean).map((l) => JSON.parse(l));
for (const r of sweep) {
  insert("load_sweep", {
    config: r.label, users: r.users, requests: r.requests, failures: r.failures, rps: r.rps,
    mean_ms: r.mean_ms, p50_ms: r.p50_ms, p90_ms: r.p90_ms, p95_ms: r.p95_ms, p99_ms: r.p99_ms,
    kv_cache_peak_pct: r.kv_cache_peak_pct, stage_quality_ms: r.stage_mean_ms?.quality ?? null,
    stage_answerability_ms: r.stage_mean_ms?.answerability ?? null, stage_answer_ms: r.stage_mean_ms?.answer ?? null,
  });
}

// --- per-label quality F1, untrained vs E5 --------------------------------
for (const [run, tasks] of [["e1", e1], ["e5", e5]] as const) {
  for (const [label, s] of Object.entries<Json>(tasks.quality.metrics.per_label)) {
    insert("quality_labels", { run, label, precision: s.precision, recall: s.recall, f1: s.f1, support: s.support });
  }
}

// --- error analysis: answering accuracy before/after, by group -------------
const errors = readJson("error_analysis/errors_test.json").answer;
for (const dimension of ["by_question_kind", "by_quality_label", "by_answerer_agreement", "by_photo"]) {
  for (const [grp, g] of Object.entries<Json>(errors[dimension])) {
    insert("error_groups", {
      dimension: dimension.replace(/^by_/, ""), grp, n: g.n,
      accuracy_before: g.accuracy_before, accuracy_after: g.accuracy_after,
    });
  }
}

// --- try-it samples (openly licensed photos, D-068) and their saved answers ----
const samples: Json[] = JSON.parse(readFileSync(join(here, "..", "public", "samples", "samples.json"), "utf8"));
samples.forEach((x, pos) =>
  insert("demo_samples", {
    pos, id: x.id, file: x.file, question: x.question, reference: x.reference ?? null, shows: x.shows ?? null,
    source: x.source, source_url: x.source_url, author: x.author, license: x.license, changes: x.changes ?? null,
  }),
);
// Written by `python -m vizwiz.serve.demo` (notebook 16); absent until that has run.
let demoAnswers = 0;
if (existsSync(join(resultsDir, "demo", "answers.json"))) {
  const demo = readJson("demo/answers.json");
  const known = new Set(samples.map((x) => x.id));
  for (const a of demo.answers) {
    if (!known.has(a.id)) continue;
    insert("demo_answers", {
      id: a.id, status: a.status, message: a.message, answer: a.answer ?? null,
      issues: JSON.stringify(a.issues ?? []), unanswerable_score: a.unanswerable_score ?? null,
      tips: JSON.stringify(a.tips ?? []), latency_ms: a.latency_ms ?? null,
      created_utc: demo.created_utc ?? null, git_commit: demo.git_commit ?? null, setup: demo.setup ?? null,
    });
    demoAnswers++;
  }
}

// --- meta -----------------------------------------------------------------
const meta: Record<string, string> = {
  model: readJson("eval/e5-multi-lora/metrics.json").model,
  gpu: readJson("eval/e5-multi-lora/metrics.json").gpu,
  chosen_threshold: String(tunePng.chosen_threshold),
  threshold_rule: tunePng.rule,
  l4_price_per_hour: String(L4_PRICE_PER_HOUR),
  slo_p95_ms: String(SLO_P95_MS),
  load_test_commit: readJson("bench/load-test/server_info.json").git_commit,
  built_utc: new Date().toISOString(),
};
for (const [key, value] of Object.entries(meta)) insert("meta", { key, value });

db.exec("COMMIT");
db.close();

console.log(
  `results.db: ${sweep.length} load-test rows, ${headline.length} headline rows, ` +
    `${samples.length} try-it samples (${demoAnswers} with saved answers) -> ${dbPath}`,
);
