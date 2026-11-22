"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { useTRPC } from "@/trpc/client";
import { QueryError, fmt } from "./ui";

type Reply = {
  status: "answered" | "retake" | "unanswerable";
  message: string;
  answer: string | null;
  issues: string[];
  unanswerable_score: number | null;
  tips: string[];
  latency_ms: number | null;
};

const STATUS: Record<Reply["status"], { icon: string; label: string }> = {
  answered: { icon: "✓", label: "Answered" },
  unanswerable: { icon: "↺", label: "Declined: can't answer from this photo" },
  retake: { icon: "↺", label: "Asked for a retake" },
};

const LIVE_LABEL = {
  not_configured: "Live model not set up: showing saved answers",
  offline: "Live model is off: showing saved answers",
  starting: "Live model is starting up",
  online: "Live model is on (AWS, one L4)",
} as const;

export function TryIt() {
  const trpc = useTRPC();
  const samples = useQuery(trpc.demo.samples.queryOptions());
  const status = useQuery({ ...trpc.demo.status.queryOptions(), staleTime: 30_000, refetchInterval: 60_000 });
  const [picked, setPicked] = useState(0);
  const [live, setLive] = useState<Record<string, Reply>>({});
  const run = useMutation(
    trpc.demo.runLive.mutationOptions({ onSuccess: (r, v) => setLive((m) => ({ ...m, [v.id]: r })) }),
  );

  if (!samples.data) return <QueryError error={samples.error} />;
  const s = samples.data[picked];
  const state = status.data?.state ?? "not_configured";
  const liveReply = live[s.id];

  return (
    <div className="card tryit">
      <div className="tryit-status" role="status">
        <span className={`dot ${state === "online" ? "on" : ""}`} aria-hidden="true" />
        {LIVE_LABEL[state]}
      </div>

      <div className="thumbs" role="radiogroup" aria-label="Sample photo">
        {samples.data.map((x, i) => (
          <button
            key={x.id}
            role="radio"
            aria-checked={i === picked}
            className={`thumb${i === picked ? " picked" : ""}`}
            onClick={() => setPicked(i)}
            title={x.question}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={`/samples/${x.file}`} alt="" />
            <span>{x.shows?.split(":")[0]}</span>
          </button>
        ))}
      </div>

      <div className="tryit-main">
        <figure>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={`/samples/${s.file}`} alt={`Sample photo: ${s.source}`} />
          <figcaption>
            <a href={s.source_url} target="_blank" rel="noreferrer">
              {s.source}
            </a>{" "}
            by {s.author}, {s.license}. {s.changes}
          </figcaption>
        </figure>

        <div className="tryit-side">
          <div className="q">
            <div className="label">Question</div>
            <div className="qtext">“{s.question}”</div>
            {s.reference && <div className="ref">Reference answer: {s.reference}</div>}
          </div>

          {liveReply ? (
            <ReplyCard title="Live answer" reply={liveReply} />
          ) : s.saved ? (
            <ReplyCard
              title="Saved answer from the fine-tuned model"
              reply={{ ...s.saved, latency_ms: null }}
              note={savedNote(s.saved)}
            />
          ) : (
            <div className="reply empty">
              No saved answer yet. Run <code>notebooks/16_demo_answers.ipynb</code> on a Colab L4 to compute them.
            </div>
          )}

          <div className="live-row">
            <button
              className="btn"
              disabled={state !== "online" || run.isPending}
              onClick={() => run.mutate({ id: s.id })}
            >
              {run.isPending && run.variables?.id === s.id ? "Asking the model…" : "Run live on AWS"}
            </button>
            {state !== "online" && <span className="hint">Available while the AWS GPU server is on.</span>}
          </div>
          {run.error && run.variables?.id === s.id && <p className="error">{run.error.message}</p>}
        </div>
      </div>
    </div>
  );
}

function savedNote(s: { createdUtc: string | null; gitCommit: string | null; setup: string | null }) {
  const parts = [
    s.createdUtc && `Computed ${s.createdUtc.slice(0, 10)}`,
    s.setup && `on ${s.setup}`,
    s.gitCommit && `at ${s.gitCommit}`,
  ].filter(Boolean);
  return parts.length ? parts.join(" ") : undefined;
}

// Saved answers carry no time: they may come from a different GPU or a cold start.
function ReplyCard({ title, reply, note }: { title: string; reply: Reply; note?: string }) {
  const st = STATUS[reply.status];
  return (
    <div className="reply">
      <div className="label">{title}</div>
      <div className={`badge ${reply.status}`}>
        <span aria-hidden="true">{st.icon}</span> {st.label}
      </div>
      <p className="spoken">{reply.message}</p>
      {/* A decline already speaks its tips; an answer can still carry one. */}
      {reply.status === "answered" && reply.tips.length > 0 && (
        <p className="tip">
          <span aria-hidden="true">💡</span> Tip: {reply.tips.join(" ")}
        </p>
      )}
      <dl>
        {reply.unanswerable_score !== null && (
          <>
            <dt>P(unanswerable)</dt>
            <dd>{fmt.num(reply.unanswerable_score, 2)} (declines at ≥ 0.10)</dd>
          </>
        )}
        <dt>Photo issues found</dt>
        <dd>{reply.issues.length ? reply.issues.join(", ") : "none"}</dd>
        {reply.latency_ms !== null && (
          <>
            <dt>Time</dt>
            <dd>{fmt.ms(reply.latency_ms)}</dd>
          </>
        )}
      </dl>
      {note && <div className="hint">{note}</div>}
    </div>
  );
}
