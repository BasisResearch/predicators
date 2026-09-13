# Bridge three-to-four-span comparison

The primary comparison is complete across distinct seeds 0, 1, and 2: MB and MF each solved the entire run in 1/3 seeds.
This variant did not produce an observed solve-rate advantage for MB.

| Arm | Whole-run successes | Mean successful steps (n) | Mean resets |
|---|---:|---:|---:|
| MB | 1/3 | 2,939 (n=1) | 0 |
| MF | 1/3 | 2,634 (n=1) | 1 |

Mean steps includes only whole-run successful seeds; mean resets includes all three finalized agent seeds.
With only one whole-run success per arm, these step means do not establish a reliable sample-efficiency difference.
MB used fewer resets in this small sample, but both arms failed the test on seeds 1 and 2.
Both successful seed-0 pilot runs are reused, and the unnecessary additional MF seed-0 failure is retained separately rather than treated as another independent seed.
The MF pilot and follow-up have identical agent runtime files and experiment flags despite differing documentation/configuration commits.

MB seed 1 ended voluntarily at 1/2 levels, 2,808 steps, and zero resets after reporting the span blocks out of reach on the floor.
MB seed 2 ended at 1/2 levels, 2,392 steps, and zero resets when the build failed the settling check.
Neither is an infrastructure or usage-limit failure.
The [seed-2 analysis](bridge-span-mb-seed2-analysis.md) documents its unchanged no-op residual and separates verified outcome evidence from the agent's proposed mechanical explanation.

The [per-seed table](bridge-span-mb-mf-results.md) and [final audit](/home/ycliang/predicators/logs/bridge_mb_extension_20260913/final-comparison-20260913.json) contain the underlying scorecards.
The user clarified on 2026-09-13 that Bridge baseline experiments must remain paused until the domain variant is chosen.
Fourteen queued tasks were held, and the three unfinished running tasks were requeued on hold after verifying saved level checkpoints.
The completed oracle-scene seed 2 result is retained; its remaining video processing does not release the held successor.
Resumption requires the variant decision, and uses the existing checkpoint replay verification; interrupted in-flight reasoning may need repeating.
Non-Bridge experiments are unaffected.
The operational record is `logs/bridge_span_followup_20260912/pause-20260913.json`.
