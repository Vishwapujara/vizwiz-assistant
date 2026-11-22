import type { Metadata } from "next";
import { Breakdowns } from "@/components/Breakdowns";
import { Experiments } from "@/components/Experiments";
import { LoadTest } from "@/components/LoadTest";
import { Overview } from "@/components/Overview";
import { Threshold } from "@/components/Threshold";

export const metadata: Metadata = { title: "VizWiz Results" };

export default function ResultsPage() {
  return (
    <main className="page">
      <header className="page-header">
        <h1>Results</h1>
        <p>
          How accurate, how safe, how fast and how cheap the assistant is. Every number is read from the committed{" "}
          <code>results/</code> files of the experiments; no VizWiz photos are shown.
        </p>
      </header>

      <section aria-labelledby="h-overview">
        <h2 id="h-overview">Overview</h2>
        <p className="lede">The served system against the untrained model on the held-out test split.</p>
        <Overview />
      </section>

      <section aria-labelledby="h-threshold">
        <h2 id="h-threshold">Decline threshold</h2>
        <p className="lede">
          The answerability stage gives P(unanswerable). Above the threshold the reply is a decline plus retake tips.
          One config value sets the trade-off between wrong answers and unnecessary declines.
        </p>
        <Threshold />
      </section>

      <section aria-labelledby="h-load">
        <h2 id="h-load">Load test and cost</h2>
        <p className="lede">Four cascade configurations under increasing concurrency on a single L4.</p>
        <LoadTest />
      </section>

      <section aria-labelledby="h-exp">
        <h2 id="h-exp">Experiments</h2>
        <p className="lede">Paired comparisons on the same examples; reading rules were fixed before each run.</p>
        <Experiments />
      </section>

      <section aria-labelledby="h-break">
        <h2 id="h-break">Where it improved</h2>
        <p className="lede">Error analysis of the fine-tuned model against the untrained one.</p>
        <Breakdowns />
      </section>
    </main>
  );
}
