import { HowItWorks } from "@/components/HowItWorks";
import { TryIt } from "@/components/TryIt";

export default function Home() {
  return (
    <main className="page">
      <header className="page-header hero-head">
        <h1>A photo assistant that knows when to say “I can't tell”</h1>
        <p>
          Blind and low-vision people take a photo and ask a question: “What does this sign say?”, “What time is it?”.
          This assistant answers briefly, or, when the photo can't answer the question, says so and explains how to take
          a better one.
        </p>
      </header>

      <section aria-labelledby="h-how">
        <h2 id="h-how">How it works</h2>
        <p className="lede">Every reply goes through three small steps, all done by the same fine-tuned model.</p>
        <HowItWorks />
      </section>

      <section aria-labelledby="h-try">
        <h2 id="h-try">Try it yourself</h2>
        <p className="lede">
          Pick a photo to see what the fine-tuned model says. Answers were saved from a real run, so this works with
          the GPU off; when the AWS server is on you can also ask the model live. The photos are openly licensed,
          not from VizWiz; the last two are blurred or darkened copies, to show how bad photos are handled.
        </p>
        <TryIt />
      </section>

    </main>
  );
}
