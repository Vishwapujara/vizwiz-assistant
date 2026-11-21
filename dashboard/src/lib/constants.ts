// Shared by the tRPC router (input validation) and the UI (controls, series colors).
// Load-test configurations in fixed order: the order sets each one's series color.
export const CONFIGS = [
  { id: "e5-parallel_checks", label: "E5, parallel checks (served)" },
  { id: "e5-sequential", label: "E5, sequential" },
  { id: "e5-speculative", label: "E5, speculative" },
  { id: "specialists-parallel_checks", label: "3 specialists, parallel" },
] as const;

export const LOAD_METRICS = ["p95_ms", "p50_ms", "p99_ms", "rps"] as const;
export const EXPERIMENTS = ["e5-multitask", "e4-fp8", "resolution"] as const;
export const ERROR_DIMENSIONS = ["question_kind", "quality_label", "answerer_agreement", "photo"] as const;
