# Training setup checks (Step 3.1)

Colab L4 (22.0 GiB usable), commit `e6284f5`, `05_train_setup.ipynb`. Answer task,
LoRA r = 16 on the language model only (252 modules, none in the vision tower;
33.0 M trainable parameters).

## Checks

| Check | Result |
|---|---|
| W&B login from Colab Secrets | OK (project `vizwiz-assistant`) |
| QLoRA smoke run, 200 examples, stopped on purpose after the step-20 checkpoint | stopped with exit 3; `checkpoint-10`, `checkpoint-20` saved |
| Resume | continued from `checkpoint-20`, ran 10 more steps, finished at step 30 |
| LoRA (bf16) smoke run, 30 steps | OK; val loss 0.42 after 30 steps |
| Loss masking | prefix check passed on every example (a failure raises) |

## Smoke-run speed (includes data loading; effective batch 16)

| Mode | per-device batch | s / optimizer step | examples / s | peak memory |
|---|---|---|---|---|
| LoRA bf16 | 2 | 9.9 | 1.61 | 11.1 GB |
| QLoRA NF4 | 2 | 13.7 | 1.17 | 7.2 GB |

QLoRA saves 35% memory and is 27% slower here, a first data point for E3.

## Profile: LoRA bf16, gradient checkpointing (`train_profile_l4.json`)

Timed forward + backward + optimizer step, median of 8 steps after 2 warm-up steps.

| max_pixels | batch 1 | batch 2 | batch 4 | tokens / example |
|---|---|---|---|---|
| 262,144 | 1.66 ex/s, 9.2 GB | **2.82 ex/s**, 9.7 GB | 2.83 ex/s, 11.0 GB | ~260 |
| 524,288 | **1.46 ex/s**, 9.7 GB | 1.41 ex/s, 10.7 GB | 1.29 ex/s, 12.9 GB | ~480 |
| 1,048,576 | **0.77 ex/s**, 10.6 GB | 0.65 ex/s, 12.6 GB | 0.58 ex/s, 16.6 GB | ~870 |

- Memory is not the limit: every setting fits, the largest at 16.6 GB of 22.
- Throughput is compute-bound and roughly proportional to tokens: doubling the
  image budget halves the speed. Above 262k, a bigger per-device batch does not
  help (padding and the longer sequences cost more than the parallelism gains).
- The smoke run at 524k (1.61 ex/s) matched the profile (1.41–1.46), so data
  loading is not a bottleneck.

## Hours per training epoch on the L4 (from the best setting per budget)

| max_pixels | answer (13,134) | answerability (17,967) | quality (20,525) | all three |
|---|---|---|---|---|
| 262,144 | 1.3 h | 1.8 h | 2.0 h | **5.1 h** |
| 524,288 | 2.5 h | 3.4 h | 3.9 h | **9.8 h** |
| 1,048,576 | 4.7 h | 6.5 h | 7.4 h | **18.6 h** |

E5 (one multi-task adapter) costs about as much again as the three adapters
together, and E3 (QLoRA) about 1.4× one adapter.
