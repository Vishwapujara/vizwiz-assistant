import { HeroPhone, TryApp } from "@/components/Product";
import { BRAND } from "@/lib/brand";

const FEATURES = [
  {
    icon: "💬",
    title: "Ask in your own words",
    body: "“What does this sign say?” “What time is it?” Answers come back short, ready to be read aloud.",
  },
  {
    icon: "✋",
    title: "Honest when it can't tell",
    body: "A wrong answer you can't check is worse than none. If the photo can't answer your question, it says so.",
  },
  {
    icon: "📷",
    title: "Helps you retake",
    body: "Blurry, dark or cut off? You hear what went wrong and how to fix it: “Turn on a light or move near a window.”",
  },
];

export default function Home() {
  return (
    <main>
      <section className="lp-hero" aria-labelledby="hero-title">
        <div className="lp-hero-inner">
          <div className="lp-hero-copy">
            <p className="eyebrow">For blind and low-vision people</p>
            <h1 id="hero-title">Ask about anything in front of you.</h1>
            <p className="lp-hero-sub">
              Take a photo, ask a question, hear a short answer. When the photo can't answer it, {BRAND} tells you how to
              take a better one.
            </p>
            <div className="lp-ctas">
              <a className="btn btn-lg" href="#try">
                Try it now
              </a>
              <a className="btn btn-ghost btn-lg" href="/research">
                See the research
              </a>
            </div>
          </div>
          <HeroPhone />
        </div>
      </section>

      <section className="features page" aria-label="What it does">
        {FEATURES.map((f) => (
          <div className="feature" key={f.title}>
            <div className="feature-icon" aria-hidden="true">
              {f.icon}
            </div>
            <h2>{f.title}</h2>
            <p>{f.body}</p>
          </div>
        ))}
      </section>

      <section id="try" className="try page" aria-labelledby="try-title">
        <h2 id="try-title">Try it</h2>
        <p className="try-sub">Pick a photo and tap Ask.</p>
        <TryApp />
      </section>
    </main>
  );
}
