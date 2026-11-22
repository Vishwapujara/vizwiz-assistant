import "server-only";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { TRPCError } from "@trpc/server";
import { z } from "zod";
import { all, get } from "./db";
import { publicProcedure, router } from "./trpc";

// Try-it section (D-068): five openly licensed photos, answers saved from the
// fine-tuned model, and an optional live call to the AWS gateway (D-069).
// The live call only ever sends one of these five photos: no visitor uploads.

type SampleRow = {
  id: string; file: string; question: string; reference: string | null; shows: string | null;
  source: string; source_url: string; author: string; license: string; changes: string | null;
};
type AnswerRow = {
  id: string; status: string; message: string; answer: string | null; issues: string;
  unanswerable_score: number | null; tips: string; latency_ms: number | null;
  created_utc: string | null; git_commit: string | null;
};

const reply = z.object({
  status: z.enum(["answered", "retake", "unanswerable"]),
  message: z.string(),
  answer: z.string().nullable(),
  issues: z.array(z.string()),
  unanswerable_score: z.number().nullable(),
  tips: z.array(z.string()),
  latency_ms: z.number(),
});
export type Reply = z.infer<typeof reply>;

const gateway = () => {
  const url = process.env.VIZWIZ_GATEWAY_URL?.replace(/\/+$/, "");
  return url ? { url, key: process.env.VIZWIZ_API_KEY ?? "" } : null;
};

export const demoRouter = router({
  samples: publicProcedure.query(() => {
    const answers = new Map(all<AnswerRow>("SELECT * FROM demo_answers").map((a) => [a.id, a]));
    return all<SampleRow>("SELECT * FROM demo_samples ORDER BY pos").map((s) => {
      const a = answers.get(s.id);
      return {
        ...s,
        saved: a
          ? {
              status: a.status as Reply["status"],
              message: a.message,
              answer: a.answer,
              issues: JSON.parse(a.issues) as string[],
              unanswerable_score: a.unanswerable_score,
              tips: JSON.parse(a.tips) as string[],
              latency_ms: a.latency_ms,
              createdUtc: a.created_utc,
              gitCommit: a.git_commit,
            }
          : null,
      };
    });
  }),

  // Is the AWS GPU server up? Checked from the server, never from the visitor's browser.
  status: publicProcedure.query(async (): Promise<{ state: "not_configured" | "offline" | "starting" | "online" }> => {
    const gw = gateway();
    if (!gw) return { state: "not_configured" };
    try {
      const r = await fetch(`${gw.url}/health`, { signal: AbortSignal.timeout(3000), cache: "no-store" });
      return { state: r.ok ? "online" : "starting" };
    } catch {
      return { state: "offline" };
    }
  }),

  runLive: publicProcedure
    .input(z.object({ id: z.string().min(1).max(64) }))
    .mutation(async ({ input }): Promise<Reply> => {
      const sample = get<SampleRow>("SELECT * FROM demo_samples WHERE id = ?", input.id);
      if (!sample) throw new TRPCError({ code: "NOT_FOUND", message: "Unknown sample" });
      const gw = gateway();
      if (!gw) throw new TRPCError({ code: "PRECONDITION_FAILED", message: "The live model is not configured" });

      const bytes = await readFile(join(process.cwd(), "public", "samples", sample.file));
      const form = new FormData();
      form.append("image", new Blob([bytes], { type: "image/jpeg" }), sample.file);
      form.append("question", sample.question);

      let res: Response;
      try {
        res = await fetch(`${gw.url}/v1/assist`, {
          method: "POST",
          body: form,
          headers: gw.key ? { "X-API-Key": gw.key } : {},
          signal: AbortSignal.timeout(60_000),
          cache: "no-store",
        });
      } catch {
        throw new TRPCError({ code: "SERVICE_UNAVAILABLE", message: "The live model is off or unreachable" });
      }
      if (!res.ok) {
        throw new TRPCError({
          code: res.status === 401 ? "UNAUTHORIZED" : "INTERNAL_SERVER_ERROR",
          message: res.status === 401 ? "The dashboard's API key was refused" : `The model answered HTTP ${res.status}`,
        });
      }
      return reply.parse(await res.json());
    }),
});
