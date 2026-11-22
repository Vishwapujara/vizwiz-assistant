import type { Metadata } from "next";
import { Nav } from "@/components/Nav";
import { Providers } from "@/trpc/client";
import "./globals.css";

export const metadata: Metadata = {
  title: "VizWiz Assistant",
  description: "How the VizWiz photo assistant works, a try-it demo, and its measured accuracy, latency and cost.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Providers>
          <Nav />
          {children}
        </Providers>
      </body>
    </html>
  );
}
