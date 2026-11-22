import type { NextConfig } from "next";

const config: NextConfig = {
  // Standalone output for the Docker image (docker/dashboard.Dockerfile).
  output: "standalone",
  // The API route opens data/results.db and, for live try-it calls, the sample photos at
  // runtime: ship them with the function (Vercel) and
  // with the standalone server (Docker).
  outputFileTracingIncludes: { "/api/trpc/[trpc]": ["./data/results.db", "./public/samples/*.jpg"] },
  poweredByHeader: false,
};

export default config;
