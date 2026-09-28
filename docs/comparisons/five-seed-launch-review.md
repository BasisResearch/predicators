# Five-seed benchmark extension

The requested extension adds seeds 3 and 4 for six comparison arms across Boil, Domino, Balloons, Bridge, and Fan: 60 additional runs.
EMPIRIC seeds 3 and 4 already belong to the separately launched r2 cohort and are not relaunched here.
All outcomes count, including failures.

## Runtime and comparability

The launch snapshot starts from `f2ed37aef`, the repaired EMPIRIC r2 runtime, at `/home/ycliang/predicators-five-seeds-frozen-20260919`.
This includes task-certification repairs, independent simulation trials, Oracle attachment restoration, and the learned-model restoration interface.
It deliberately excludes the subsequent Balloons layout change `a25cd54c4`, which moves the payload/chute and changes the displayed target band.
The original environment geometry and task-generation settings therefore remain consistent with the existing benchmark.
The main checkout's layout changes are preserved.

All new runs use Opus, the shared composite skills, the same observation noise and real-step allowances, and the 48-hour active wall-clock allowance.
Legacy blocking preflight and the nonblocking audit are explicitly disabled for these six comparison arms.
The already-running EMPIRIC r2 cohort retains its nonblocking audit; historical EMPIRIC also has mixed preflight settings, so the pooled figure is not a matched preflight experiment.
The new comparison seeds receive shared simulator fixes that were absent from some historical seeds; source cohorts and runtime versions must remain disclosed.

## Implementation review

- **Oracle dynamics:** fixed supplied mechanisms and parameters, hidden model source, no fitting or alternative parameter edits, and no privileged execution state or oracle controller.
  State is still reconstructed from observations.
- **Direct agent:** no supplied simulator or model-building gate; uses the shared real-action tools and its own sandbox code.
- **Direct + scene:** the same direct-agent implementation, with read-only engine wrapper, scene manifest, URDFs, and meshes.
  The prompt asks it to solve tasks, not build a simulator; no domain-specific dynamics source or harness simulator is supplied.
- **Standalone sim.:** agent-written program model, including an optional agent-written PyBullet implementation, without a supplied physics twin.
  The probe permits data scoring and single plan rollouts, but withholds harness fitting, repeated trials, refinement/search, and engine-backed evaluation/rendering.
- **No harness fitting:** learned simulator with agent-declared parameter values/ranges; harness estimation is disabled, but agent-written estimation and uncertainty-aware rollouts remain available.
- **No explicit uncertainty:** fitting remains available on raw observations, while noise declaration, smoothing, belief frames/draws, parameter uncertainty, and uncertainty-based decision tools are disabled.

The review inspected approach implementations, runtime tool restrictions, inherited configuration flags, scene-package contents, and frozen-model protections.
Compute-node regression tests exercise the real continual play loop with scripted agents, Oracle mechanism fidelity, Bridge restoration, prompt contracts, and result-cohort selection.
These checks do not establish future solve rates or resolve the previously documented exact Bridge replay mismatch.

## Launch and reporting

The launcher is [continual_benchmark_five_seeds.yaml](https://github.com/BasisResearch/predicators/blob/iclr-empiric-submission/scripts/configs/predicatorv3/continual_benchmark_five_seeds.yaml).
All 60 destination seed directories were checked absent before submission.
Jobs use `mit_preemptable`, checkpoint resume, automatic requeue, and account labels `a,b,c,d`, all confirmed usable by the user during this launch review.
The user's account e corresponds to the launcher's `dat` label and is reserved as backup, not included in the normal rotation.
Each array contains seeds 3 and 4.

The full benchmark keeps Oracle r2 and EMPIRIC r2 identifiable as separate cohorts.
The paper selection pools original seeds 0-2 with new seeds 3-4, using repaired Oracle seeds 0-2 for Domino/Bridge.
Every selected result retains its source path and source arm.
The original six-run Oracle diagnostic figure remains a fixed pilot comparison.
Overleaf is not automatically updated by this launch.

## Submission receipt

Submitted September 19, 2026, from frozen commit `c9eaeaaa1`.
All 30 arrays were accepted and all 60 requested seed tasks were verified in the scheduler.
Each task requests eight CPUs and 16 GB, with a 12-hour allocation and pre-timeout self-requeue.

| Agent | Balloons | Bridge | Boil | Fan | Domino |
|---|---|---|---|---|---|
| Oracle dynamics | `23126476` | `23126477` | `23126478` | `23126479` | `23126480` |
| Direct agent | `23126481` | `23126482` | `23126483` | `23126484` | `23126485` |
| Direct + scene | `23126486` | `23126487` | `23126488` | `23126489` | `23126498` |
| Standalone sim. | `23126499` | `23126500` | `23126501` | `23126502` | `23126503` |
| No harness fitting | `23126504` | `23126505` | `23126506` | `23126507` | `23126508` |
| No explicit uncertainty | `23126509` | `23126510` | `23126511` | `23126512` | `23126513` |

Monitor `23126258` replaces monitor `23111801` and refreshes the main report/figure every 60 seconds when finished results change.
The existing experiments were not restarted or modified.

## Validation receipts

- Repair, Oracle mechanisms, trial isolation, restoration, and prompt suite: 62 passed (`23125725`).
- Direct, scene-assets, Standalone, and resume/capability suite: 29 passed (`23125870`).
- Configuration and report-selection suite: 8 passed, and static checks passed for three source files (`23125773`).
- Corrected launch-contract test inside the final runtime snapshot: 1 passed (`23126061`); subsequent commits only changed account comments.
- The initial launch-contract assertion incorrectly counted Direct + scene as a separate implementation class; it was corrected and verified before submission.
- The initial reporting test exposed a hard-coded Oracle cohort denominator; the report now computes expected counts from registered seed directories, and its tests pass.

The broader overlapping suite `23125478` reached the 15-minute allocation limit; its Oracle/ablation cases passed, while repeated Domino task generation consumed much of the quick-test allocation.
Do not describe that broader invocation as a fully passing suite.
At the post-submission check, all 60 new tasks were queued for scheduler priority, rather than blocked on account usage.

## September 20 checkpoint resume

Fifty of the sixty new comparison seeds finished; ten remaining seeds terminated on account a's weekly limit.
At the user's request, these ten were resubmitted from the same frozen checkout and experiment IDs, with `auto_resume` and existing checkpoints verified.
Minimal tool-free Opus requests confirmed b and d working (`23199515`); the usage endpoint itself returned HTTP 403.
Accounts a and c were excluded because of known weekly-limit failures; dat remains backup only.
Round-robin assignment splits the resumed runs evenly between b and d.

| Run | Seed | Job | Account |
|---|---:|---|---|
| Oracle, Balloons | 4 | `23199623` | b |
| Oracle, Boil | 4 | `23199624` | d |
| Direct + scene, Bridge | 3 | `23199626` | b |
| Direct + scene, Boil | 4 | `23199627` | d |
| Standalone, Boil | 3 | `23199629` | b |
| No fitting, Fan | 3 | `23199630` | d |
| No uncertainty, Balloons | 3 | `23199632` | b |
| No uncertainty, Bridge | 4 | `23199633` | d |
| No uncertainty, Boil | 3 | `23199634` | b |
| No uncertainty, Domino | 3 | `23199635` | d |
