# Results dashboard (Next.js, D-067). Build context is the repo root: the build step
# turns results/*.json into dashboard/data/results.db, which the server reads.
FROM node:24-slim AS build
WORKDIR /repo/dashboard
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY results /repo/results
COPY dashboard ./
RUN npm run build

FROM node:24-slim
# Set VIZWIZ_GATEWAY_URL and VIZWIZ_API_KEY at run time to enable live try-it calls (D-069).
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1 PORT=8501 HOSTNAME=0.0.0.0
WORKDIR /app
COPY --from=build --chown=node:node /repo/dashboard/.next/standalone ./
COPY --from=build --chown=node:node /repo/dashboard/.next/static ./.next/static
COPY --from=build --chown=node:node /repo/dashboard/public ./public
USER node
EXPOSE 8501
HEALTHCHECK --interval=10s --timeout=3s --retries=5 \
    CMD node -e "fetch('http://127.0.0.1:8501/api/trpc/health').then(r=>process.exit(r.ok?0:1),()=>process.exit(1))"
CMD ["node", "server.js"]
