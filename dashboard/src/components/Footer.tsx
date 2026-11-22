import Link from "next/link";
import { BRAND, REPO_URL } from "@/lib/brand";

export function Footer() {
  return (
    <footer className="footer">
      <div className="footer-inner">
        <p>
          <b>{BRAND}</b> is a research prototype, not a safety device. Don't rely on it for medication, money, or
          anything where a wrong answer could hurt.
        </p>
        <p className="footer-links">
          <Link href="/research">Research</Link>
          <a href={REPO_URL} target="_blank" rel="noreferrer">
            Source code
          </a>
          <span>Demo photos: CC0, Wikimedia Commons</span>
        </p>
      </div>
    </footer>
  );
}
