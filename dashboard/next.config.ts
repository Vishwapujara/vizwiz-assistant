import type { NextConfig } from "next";

const config: NextConfig = {
  // Standalone output for the Docker image (docker/dashboard.Dockerfile).
  output: "standalone",
  // The API route opens data/results.db at runtime: ship it with the function (Vercel) and
  // with the standalone server (Docker).
  outputFileTracingIncludes: { "/api/trpc/[trpc]": ["./data/results.db"] },
  poweredByHeader: false,
};

export default config;
