import Link from "next/link";

// How a reply is made: the three-stage cascade, drawn as a flow (no numbers needed).

const CHECKS = [
  {
    n: "1",
    title: "Photo quality",
    body: "Looks for problems a blind photographer can't see: blur, bad framing, rotation, too dark or bright, a finger over the lens, or nothing recognisable.",
    out: "Problems become retake tips, e.g. “The photo is too dark. Turn on a light or move near a window.”",
  },
  {
    n: "2",
    title: "Can this be answered?",
    body: "Estimates how likely it is that the question can't be answered from this photo at all.",
    out: "A probability between 0 and 1. At 0.10 or more, the assistant declines.",
  },
];

export function HowItWorks() {
  return (
    <div className="how">
      <ol className="flow" aria-label="How a reply is made">
        <li className="step card">
          <div className="step-kicker">You send</div>
          <div className="step-title">A photo and a question</div>
          <p>The photo is turned upright and resized, the same way as during training.</p>
        </li>

        <li className="step-arrow" aria-hidden="true" />

        <li className="step-group">
          <div className="step-kicker group-kicker">Two checks, at the same time</div>
          <div className="checks">
            {CHECKS.map((c) => (
              <div className="step card" key={c.n}>
                <div className="step-title">
                  <span className="num" aria-hidden="true">
                    {c.n}
                  </span>
                  {c.title}
                </div>
                <p>{c.body}</p>
                <p className="out">{c.out}</p>
              </div>
            ))}
          </div>
        </li>

        <li className="step-arrow" aria-hidden="true" />

        <li className="step-group">
          <div className="step-kicker group-kicker">Then one of two replies</div>
          <div className="checks">
            <div className="step card outcome">
              <div className="step-title">
                <span className="num" aria-hidden="true">
                  3
                </span>
                Answer
              </div>
              <p>If the question looks answerable, the model gives a short answer, like a sighted helper would.</p>
              <p className="out">“Bananas.” · “Chapel Lane.”</p>
            </div>
            <div className="step card outcome">
              <div className="step-title">
                <span className="num" aria-hidden="true">
                  ↺
                </span>
                Decline, with tips
              </div>
              <p>If not, it says so instead of guessing, and explains how to take a better photo.</p>
              <p className="out">“I can't answer that from this photo. The photo is blurry. Hold the phone steady, elbows on a table if you can.”</p>
            </div>
          </div>
        </li>
      </ol>

      <div className="under-hood">
        <div>
          <h3>Why decline?</h3>
          <p>
            For someone who can't check the answer, a confident wrong answer is worse than “I can't tell”. The assistant
            is tuned to prefer declining to guessing.
          </p>
        </div>
        <div>
          <h3>Under the hood</h3>
          <p>
            One small vision-language model (Qwen3-VL, 4B parameters) with one LoRA adapter fine-tuned on VizWiz, real
            photos and questions from blind users. The same adapter does all three jobs, served by vLLM on one NVIDIA L4
            GPU.
          </p>
        </div>
        <div>
          <h3>How well it works</h3>
          <p>
            Accuracy, safety, speed and cost are on the <Link href="/results">Results</Link> page.
          </p>
        </div>
      </div>
    </div>
  );
}
