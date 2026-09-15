# Boil complete unheld-scene prior audit

This extends the independently verified [fixture proposal](boil-fixture-proposal.md) and [articulated-state component](boil-articulated-prior.md) into a declared full scene for the fixed historical Boil program.
It is part of Stage A physical support work needed for Stage B inference comparisons.
The production agent and its uncertainty handling remain unchanged.

## Information and model boundary

The development input is the first 264-action training recording under `logs/uncertainty_recording_audit_20260912/inputs/Boil/L01/`.
Only reconstructed public observations and actions enter inference.
The frozen runtime is `b09217bb38f2c3082136994ae43fbef2eb590e82`, with explicitly captured inference-component overlays.
The historical learned filling/heating program has SHA-256 `b123f544d835e745d2c740d527ebb6b80d20e9b8bb412b1c6429f8de11afe172`.
Its eight original development parameter bounds define independent uniform priors; an optimizer's logarithmic coordinate flag does not change that prior.
This is a transferred historical program, separate from the later sweep's parameter-free Boil model.

The initial case has one unheld jug, conditioned on its public exact held-state reading.
Known shapes, object identities and colors, fixed robot base placement, and the two-table geometry are conditioned interface inputs.
Missing joint velocities, object orientations and motion do not imply rest.
The fresh native initializer supplies cold heat and zero spill, and the literal learned program starts with `LATENT_INIT={}`.
No evaluator-private heat or saved inferred memory initializes the candidate.
Jug volume is sampled explicitly and initializes the learned volume memory through the ordinary rule path.

## Original scene components

| Component | Declared distribution |
| --- | --- |
| Four fixed fixture base positions | Independent uniform x in [0.35, 1.15], y in [1.05, 1.65], z in [0.35, 0.8] meters |
| Fixture orientations | Probability 0.8: all upright with independent uniform yaw; probability 0.2: independent uniform SO(3) orientations |
| Robot joint positions | Original zero-centered Gaussian reset law; exact controlled positions are conditioned with their density retained, four unobserved positions remain uncertain |
| Robot motion | Probability 0.8: all joint velocities zero; otherwise each of thirteen movable joint velocities uniform in [-0.1, 0.1] native units |
| Jug pose and motion | Probability 0.8: upright, resting on the support plane with uncertain xy/yaw; otherwise free position/orientation with linear velocities in [-0.1, 0.1] m/s and angular velocities in [-0.2, 0.2] rad/s |
| Jug water | Probability 0.5 at zero; otherwise uniform over [0, 1.3] |
| Two switch sliders | Validated endpoint-rest/uniform-motion law, conditioned on exact initial flags with both event probabilities retained |
| Unobserved faucet hinge | Validated endpoint-rest/uniform-motion law over [0, pi/2], without an invented switch observation |

The jug placement cell is x in [0.3, 1.2], y in [1.1, 1.85], z in [0.4, 1.2] meters.
The existing rigid-assembly component erodes its free position bounds by a conservative enclosing radius derived from native geometry.
The resting component uses the actual collision support depth.
The geometry predicate rejects the whole candidate if the robot or jug penetrates another collision body or any pair of fixed fixtures interpenetrates.
Fixed wheel/ground contacts, fixture mounting intersections with tables, and fixed table overlap are explicit exceptions.
Visual-only liquid bodies do not participate in this predicate.

The target is proportional to `p(theta) p0(scene) I[C(scene)]` times the full observation/transition likelihood.
The geometric normalizer is independent of theta for this fixed program: the eight parameters modify only reported features and latent memory, with no physical commands or parameter-dependent collision shape.
Its unknown common normalizer can therefore be omitted from posterior ratios, but not from absolute model evidence.
This justification does not apply automatically to other programs.

## Observation-guided proposals

All proposal guides retain a 0.25 original-prior component and a 0.75 normalized truncated-Gaussian component.
Every guide retains the full original-prior/proposal density correction.
Fixture xyz and observed upright yaw use the first 65 public observations; jug position, resting yaw and positive water volume use the first observation.
Fixture xyz guidance includes the declared correlated output discrepancy.
Unobserved fixture angles and moving SO(3) branches do not acquire a fabricated observation guide.
All original observation factors remain in the likelihood exactly once.
Unused coordinates in mixture cases integrate out under the original unit-coordinate representation.

## Native support correction

The first complete-scene attempt, job `22688683_0`, used the jug AABB lower edge as its support depth.
That edge includes 1 mm of collision padding: 0.061 m instead of the actual 0.060 m support depth.
The job was cancelled after identifying this mistake; its partial records remain in `logs/uncertainty_boil_full_scene_20260913/` as an invalid-prior diagnostic.
They are not an agent failure or accepted posterior input.

The corrected initializer queries native jug-to-table separation from an elevated identity-orientation pose, derives the support depth, and checks it again at zero separation before applying the candidate.
The corrected native audit, `22688726_0`, completed in 7:02 on one compute CPU.
Its bundle is `logs/uncertainty_boil_full_scene_v2_20260913/`.

Sixteen independently drawn scenes use seed 402, with two parameter settings per scene: historical defaults and an independent original-uniform parameter draw.
Two identical point controls provide the previously verified reference trajectory.
Three scenes fail the declared initial geometry predicate and are rejected before any action.
Thirteen scenes are geometrically feasible; two have finite full-recording likelihood under both parameter settings, while the other eleven retain exact-event contradictions.
That gives 28 completed histories and 7,392 native actions including the point controls.
These counts measure support in this proposal audit, not agent solve rates or numerically adequate posterior coverage.

Every paired physical history is identical across its two parameter settings, including native pre-rule frames and joint-transition factors.
The rule-generated observations and model memory may differ with parameters.
This is the required physical independence check before considering reuse of native histories during parameter inference.

Fresh geometry verification `22689157` completed on one compute CPU in 24 seconds.
It reconstructs all sixteen saved scenes in new native worlds, independently enumerates 464 eligible body pairs, and reproduces all rejected and explicitly allowed intersections.
It also repeats both support-depth probes for each scene without executing environment actions.

Independent density verification `22689045` completed in 5:51 on one CPU.
It verifies all 66,528 joint factors to maximum absolute error 3.638e-12, the Gaussian guide calculations and inverse proposal coordinates, all conditioned initial-state factors and complete output likelihoods.
Ten corrupted transition, output, proposal and initial-state inputs are rejected.
Additional coordinate verification `22689232` checks all sixteen scenes against the original bounds, auxiliary-coordinate layout, moving-jug orientation/velocity map and mixture branches, rejecting five corruption classes.

Literal-rule verification `22689096` completed in 28 allocation seconds, with 15.143 seconds in its reader and no native actions.
All 7,392 rule-generated frames and model-memory states match exactly under both default and independently sampled parameter settings.
The native physical histories are therefore reusable across the tested parameter changes, while each parameter-dependent observation and memory trajectory is still evaluated anew.
The bundle's `verified-inputs.json` pins the source report, scripts and all four verification reports.

## Remaining gate

A finite candidate establishes likelihood support, not useful parameter uncertainty.
The next experiment must cover the original scene and parameter law, preserve all initial conditioning factors, and assess independent-run agreement, held-out predictions and compute cost.
Cached physical histories may reduce repeated work only under the fixed program's verified parameter-independence boundary.
No arbitrary reuse across physical parameters, state changes, different programs or different observation phases is justified.
Stage A closure, Stage B acceptance and live-agent non-regression remain unproven.

## Subsequent runtime-closure finding

The [prefix-inference integration](boil-joint-inference.md#initialization-order-obstruction-and-correction) exposes an additional limitation of this audit.
Removing its intermediate initialization operations changes some contact-rich continuations despite exact agreement in initial public state and saved scene geometry.
The original operation sequence still repeats exactly, and its density, geometry and literal-rule checks remain valid for that sequence.
Those checks do not establish that the final physical coordinates alone define an observation-independent generative initialization.
The subsequent fixed-template experiment removes the observation-derived intermediate numeric anchor and is being validated as a separately identified runtime before posterior fitting.
