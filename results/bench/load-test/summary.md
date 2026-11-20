| Configuration | 1 user p50 / p95 (ms) | Best req/s with p95 < 2 s | Peak req/s | Failures |
|---|---|---|---|---|
| E5, parallel checks (served) | 610 / 880 | 2.92 at 4 users (p95 1900) | 4.12 at 16 | 0 of 1141 |
| E5, sequential | 720 / 910 | 2.02 at 2 users (p95 1400) | 3.81 at 16 | 0 of 1040 |
| E5, speculative | 580 / 860 | 2.37 at 2 users (p95 1400) | 3.64 at 16 | 0 of 1081 |
| 3 specialists, parallel | 630 / 840 | 3.04 at 4 users (p95 1900) | 4.30 at 32 | 0 of 1174 |

| Users | E5, parallel checks (served) | E5, sequential | E5, speculative | 3 specialists, parallel |
|---|---|---|---|---|
| 1 | 1.66 req/s, p50 610, p95 880 | 1.44 req/s, p50 720, p95 910 | 1.70 req/s, p50 580, p95 860 | 1.71 req/s, p50 630, p95 840 |
| 2 | 2.35 req/s, p50 840, p95 1200 | 2.02 req/s, p50 990, p95 1400 | 2.37 req/s, p50 840, p95 1400 | 2.35 req/s, p50 860, p95 1300 |
| 4 | 2.92 req/s, p50 1400, p95 1900 | 2.67 req/s, p50 1400, p95 2100 | 2.94 req/s, p50 1300, p95 2400 | 3.04 req/s, p50 1300, p95 1900 |
| 8 | 3.83 req/s, p50 2000, p95 3200 | 3.36 req/s, p50 2300, p95 3100 | 3.51 req/s, p50 2100, p95 3800 | 3.83 req/s, p50 2000, p95 3400 |
| 16 | 4.12 req/s, p50 3700, p95 5900 | 3.81 req/s, p50 4000, p95 5700 | 3.64 req/s, p50 3700, p95 7100 | 4.15 req/s, p50 3700, p95 5800 |
| 32 | 4.05 req/s, p50 7900, p95 11000 | 3.73 req/s, p50 8100, p95 11000 | 3.49 req/s, p50 8100, p95 12000 | 4.30 req/s, p50 7500, p95 10000 |
