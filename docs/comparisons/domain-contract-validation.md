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


## Standalone package access

The experiment environment includes PyBullet, NumPy, and SciPy.
The standalone approach omits base-simulator reference files, removes the engine-backed evaluator, and routes supplied predictions through the learned skill-transition program.
The original frozen experiment prompt forbids importing an environment or physics engine inside a prediction.
The import guard screens hidden `predicators.envs` and `predicators.ground_truth_models` modules; it does not prohibit the `pybullet` package itself.
Thus the implemented isolation covers the supplied prediction interface, not all physics packages accessible to arbitrary agent code.
A scan of the two started Boil standalone runs on 2026-09-13 found PyBullet import matches in copied skill-controller references and a transcript displaying one such reference.
That scan did not find an agent-written PyBullet simulator, but it is not an exhaustive proof of engine-free behavior or import isolation.
No experiment runtime or package installation was changed for this audit.


## Physics libraries permitted, 2026-09-13

The user clarified that the standalone agent may use PyBullet or other available simulation libraries.
The intended comparison withholds a prepared scene and base simulator, while allowing the agent to construct its own predictive world from public observations and recorded interactions.
The development prompt now states this explicitly and retains the prohibition on inspecting the task environment, ground-truth mechanisms, or live simulator state.
The original frozen experiments retain their stricter prompt and must not be presented as a cohort run under the revised instruction.
Replacement configurations cover twelve original non-Bridge seeds and three Bridge span-transfer seeds, keeping their respective domain runtimes and task settings.
Whether to replace the existing standalone cohort or retain it as the stricter comparison is pending the user's preference; no replacement experiment has been submitted.
Compute array 22672397 passed all ten domain checks for agent-owned PyBullet predictions, live-environment isolation, and ordinary program predictions.
Static job 22672446 passed mypy and pylint for both changed Python files after correcting a test type annotation reported by 22672398.
Their manifests and outputs are under `/home/ycliang/predicators/logs/standalone_engine_contract_20260913/`; these are mechanical checks, not agent solve-rate seeds.
