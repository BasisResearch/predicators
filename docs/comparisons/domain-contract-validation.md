# Cross-domain comparison tool validation

The standalone and zero-shot action-boundary tests now use the real noisy cohort configurations in all five domains.
Their earlier tests exercised these contracts only in Boil.
The other four comparison methods already had five-domain play-tool coverage.

All ten extended checks passed in array `22650730`.
Type checking and lint passed in job `22650731`.
The tests use scripted agent tool calls and do not call Claude or produce solve-rate outcomes.

| Domain | Runtime | Check job | Tests passed |
|---|---|---|---:|
| Boil | `59336397069a` | 22650730_0 | 2 |
| Bridge, 3-to-4 span | `73b5e517bf18` | 22650730_1 | 2 |
| Fan | `59336397069a` | 22650730_2 | 2 |
| Domino | `59336397069a` | 22650730_3 | 2 |
| Original Balloons | `59336397069a` | 22650730_4 | 2 |

The standalone check edits its program during a real continual play session and verifies that subsequent predictions load the changed program.
It checks the supplied model interface has no engine instance or base-simulator reference files, rejects engine-diagnostic rollout modes, and records a charged real action.
This is an interface check, not an exhaustive sandbox security audit.

The zero-shot check refuses the first charged action without a model, seals a valid model before taking that action, and rejects later source edits and fitting routes.
It restores the sealed source from a saved approach state and verifies that the next action is charged correctly.
The separate-process standalone resume test remains covered by the earlier 25-test suite in job `22648953`.

An immutable copy of the extended test module was run against each cohort's existing frozen production package.
The runner checks the imported package path and test-file digest before executing the tests.
No frozen experiment code was edited and no agent run was restarted for this validation.
The source paths, digests, job records, and verified pass summaries are recorded in `/home/ycliang/predicators/logs/comparison_domain_contracts_20260912/validated.json` and its adjacent validation manifest.
