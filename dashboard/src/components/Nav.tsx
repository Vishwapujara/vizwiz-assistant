"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { BRAND } from "@/lib/brand";

export function Nav() {
  const path = usePathname();
  return (
    <nav className="nav" aria-label="Main">
      <div className="nav-inner">
        <Link href="/" className="brand">
          <span className="brand-mark" aria-hidden="true" />
          {BRAND}
        </Link>
        <div className="nav-links">
          <Link href="/#try">Try it</Link>
          <Link href="/research" aria-current={path === "/research" ? "page" : undefined}>
            Research
          </Link>
        </div>
      </div>
    </nav>
  );
}
