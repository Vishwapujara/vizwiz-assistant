import type { Metadata } from "next";
import { Providers } from "@/trpc/client";
import "./globals.css";

export const metadata: Metadata = {
  title: "VizWiz Results",
  description: "Accuracy, safety, latency and cost of the VizWiz visual assistant, from the committed results.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
