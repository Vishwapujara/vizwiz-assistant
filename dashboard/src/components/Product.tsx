"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { useTRPC } from "@/trpc/client";

// Product home: a phone that shows real replies from the model. Recorded replies are
// labelled as such; when the GPU server is on, "Ask" goes to the live model.

type Reply = {
  status: "answered" | "retake" | "unanswerable";
  message: string;
  tips: string[];
};
type Sample = {
  id: string;
  file: string;
  question: string;
  source: string;
  source_url: string;
  author: string;
  license: string;
  shows: string | null;
  saved: Reply | null;
};

const sentence = (s: string) => (s ? s[0].toUpperCase() + s.slice(1) : s);
const spokenText = (r: Reply) =>
  r.status === "answered" && r.tips.length ? `${sentence(r.message)}. ${r.tips.join(" ")}` : sentence(r.message);

function useSamples() {
  const trpc = useTRPC();
  return useQuery(trpc.demo.samples.queryOptions());
}

function usePrefersReducedMotion() {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const m = window.matchMedia("(prefers-reduced-motion: reduce)");
    setReduced(m.matches);
    const on = () => setReduced(m.matches);
    m.addEventListener("change", on);
    return () => m.removeEventListener("change", on);
  }, []);
  return reduced;
}

function speak(text: string) {
  if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(new SpeechSynthesisUtterance(text));
}

/** The phone screen: photo, the question as a sent message, the reply below it. */
function Phone({
  sample,
  reply,
  asking,
  label,
  footer,
}: {
  sample: Sample;
  reply: Reply | null;
  asking?: boolean;
  label?: string;
  footer?: React.ReactNode;
}) {
  return (
    <div className="phone">
      <div className="phone-screen">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img className="phone-photo" src={`/samples/${sample.file}`} alt="" />
        <div className="bubble me">{sample.question}</div>
        <div className="bubble them" aria-live="polite">
          {asking ? (
            <span className="typing" aria-label="Thinking">
              <i />
              <i />
              <i />
            </span>
          ) : reply ? (
            <>
              <span>{sentence(reply.message)}</span>
              {reply.status === "answered" && reply.tips.length > 0 && <span className="bubble-tip">{reply.tips[0]}</span>}
            </>
          ) : (
            <span className="muted">Tap Ask</span>
          )}
        </div>
        {label && reply && !asking && <div className="bubble-label">{label}</div>}
        {footer}
      </div>
    </div>
  );
}

/** Hero: cycles through the recorded replies. Decorative; the interactive version is below. */
export function HeroPhone() {
  const { data } = useSamples();
  const reduced = usePrefersReducedMotion();
  const [i, setI] = useState(0);
  const [paused, setPaused] = useState(false);
  const withAnswers = (data ?? []).filter((s) => s.saved);
  useEffect(() => {
    if (reduced || paused || withAnswers.length < 2) return;
    const t = setInterval(() => setI((x) => (x + 1) % withAnswers.length), 4200);
    return () => clearInterval(t);
  }, [reduced, paused, withAnswers.length]);
  if (!withAnswers.length) return <div className="phone phone-skeleton" aria-hidden="true" />;
  const s = withAnswers[i % withAnswers.length];
  return (
    <div aria-hidden="true" onMouseEnter={() => setPaused(true)} onMouseLeave={() => setPaused(false)} key={s.id} className="hero-phone">
      <Phone sample={s} reply={s.saved} />
    </div>
  );
}

/** "Try it": pick a photo, tap Ask, hear the reply. */
export function TryApp() {
  const trpc = useTRPC();
  const { data, error } = useSamples();
  const status = useQuery({ ...trpc.demo.status.queryOptions(), staleTime: 30_000 });
  const live = status.data?.state === "online";
  const [picked, setPicked] = useState(0);
  const [shown, setShown] = useState<Record<string, { reply: Reply; live: boolean }>>({});
  const [asking, setAsking] = useState(false);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const run = useMutation(trpc.demo.runLive.mutationOptions());

  useEffect(() => () => {
    if (timer.current) clearTimeout(timer.current);
  }, []);

  if (error) return <p className="error">Could not load the photos.</p>;
  if (!data) return <div className="tryapp" aria-busy="true" />;
  const s = data[picked];
  const current = shown[s.id];

  const ask = async () => {
    setAsking(true);
    try {
      if (live) {
        const r = await run.mutateAsync({ id: s.id });
        setShown((m) => ({ ...m, [s.id]: { reply: r, live: true } }));
      } else if (s.saved) {
        const saved = s.saved;
        await new Promise((res) => (timer.current = setTimeout(res, 700)));
        setShown((m) => ({ ...m, [s.id]: { reply: saved, live: false } }));
      }
    } catch {
      if (s.saved) setShown((m) => ({ ...m, [s.id]: { reply: s.saved!, live: false } }));
    } finally {
      setAsking(false);
    }
  };

  return (
    <div className="tryapp">
      <div className="picker" role="radiogroup" aria-label="Choose a photo">
        {data.map((x, i) => (
          <button
            key={x.id}
            role="radio"
            aria-checked={i === picked}
            className={`pick${i === picked ? " on" : ""}`}
            onClick={() => setPicked(i)}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={`/samples/${x.file}`} alt="" />
            <span>
              {x.question}
              {x.shows && <small>{x.shows.split(":")[0]}</small>}
            </span>
          </button>
        ))}
      </div>

      <div className="tryapp-phone">
        <Phone
          sample={s}
          reply={current?.reply ?? null}
          asking={asking}
          label={current ? (current.live ? "Live answer" : "Recorded answer") : undefined}
          footer={
            <div className="phone-actions">
              <button className="btn btn-lg" onClick={ask} disabled={asking || (!live && !s.saved)}>
                {asking ? "Asking…" : current ? "Ask again" : "Ask"}
              </button>
              <button
                className="btn btn-ghost"
                onClick={() => current && speak(spokenText(current.reply))}
                disabled={!current || asking}
                aria-label="Read the answer aloud"
              >
                🔊 Listen
              </button>
            </div>
          }
        />
        <p className="credit">
          Photo:{" "}
          <a href={s.source_url} target="_blank" rel="noreferrer">
            {s.author}
          </a>
          , {s.license}
        </p>
      </div>
    </div>
  );
}
