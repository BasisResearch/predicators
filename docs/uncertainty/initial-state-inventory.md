# Initial-state and exact-observation inventory

September 12, 2026.
This is a source and recording inventory for Stage A of the [simplification proposal](simplification-proposal.md), including the unresolved prior choices.
It does not certify a feasible physical prior or an implemented five-domain posterior.
The acting MB agent continues to use legacy inference.

## Version and information boundary

The inventory uses the five frozen first training levels selected in `logs/uncertainty_recording_audit_v2_20260912/plan.json`.
Each is one actual reset episode from historical MB seed 0; subsequent observations belong to the same initial state and full action history.
The accompanying report in `job-22625932/report.json` identifies every data and sensor digest.
The associated task and model runtime is `b09217bb38f2c3082136994ae43fbef2eb590e82`, with the corrected offline observation reader overlaid.
Recorded sanitized truth is transformed through the original keyed noise channel before becoming inference evidence.
Body velocities, privileged heat/cure state, and weld metadata are excluded from that evidence.

| Domain | Training actions | Observations | Noisy scalar readings per frame | Exact scalar readings per frame | Frozen cycle-000 program versions |
| --- | ---: | ---: | ---: | ---: | --- |
| Bridge | 1,186 | 1,187 | 40 | 67 | 1 |
| Fan | 132 | 133 | 59 | 49 | 1, 2, 3 |
| Domino | 161 | 162 | 30 | 40 | 1, 2 |
| Boil | 264 | 265 | 22 | 24 | 1 |
| Original non-hatch balloons | 235 | 236 | 26 | 32 | 1, 2, 3, 4, 5 |

These are observation counts, not independent physical coordinates or posterior dimensions.
For example, robot Cartesian pose and joint positions describe the same mechanism, and a balloon's tied flag does not independently specify its weld frame.
All five recordings include nine exact controlled joint positions and seven exact robot features per frame.
The runtime defaults to fixed-base Fetch; nine observed joints must not be mistaken for the whole URDF joint state.
The later recorded-action cache audit `22633028` establishes an additional timing distinction: the Cartesian robot pose comes from PyBullet's cached link transform, while joint positions are read from the current joint state.
After a physics step, an explicit forward-kinematics query or an identical joint reset can change the reported Cartesian pose while leaving joint positions and velocities unchanged.
On eight recorded Balloons actions, the native cached poses match the historical observations exactly; refreshed poses differ by up to 1.094 mm in a coordinate and 0.00634 radians in an angle.
The Cartesian readings therefore cannot be eliminated as instantaneous functions of the observed joint vector without preserving this historical observation phase.
This is engine observation timing, not evidence for increasing sensor noise.
See the [cache audit](../../logs/uncertainty_robot_cache_20260912/assessment.json) and [transition reference](transition-discrepancy.md).
The finger readout has a different result: all 1,983 recorded frames match the fixed endpoint interpolation applied after float32 conversion of the exact left-finger joint observation.
The [checked readout reduction](observation-reductions.md) therefore retains the joint's likelihood and verifies the extra readout without treating it as independent evidence.
Omitting the float32 conversion produces 1,967 false mismatches; incompatible or source-missing readouts are not silently dropped.
The compute audit in `logs/uncertainty_state_inventory_v2_20260912` enumerates the actual visible-model joints without generating evaluator tasks.
Job `22627492` confirms the same layout in all five domains: 24 URDF joints, comprising nine observed movable joints, four unobserved movable joints, and eleven fixed joints.
The unobserved movable joints are the two wheels and head pan/tilt.
Before justified reductions, robot motion therefore contributes nine unknown controlled velocities plus four unobserved positions and four unobserved velocities per episode.
This is 17 possible continuous robot coordinates per root, not an assertion that all affect the likelihood.

The frozen Bridge and Boil cycle-000 programs are no-op models with empty parameter declarations.
They are useful incomplete-program controls, but cannot by themselves establish predictive performance of the final learned models.
Later cycle versions exist in the source run archives; selecting them requires recording their training-information boundary and parameter sidecars before comparison.
The final archived Bridge and Boil `simulator.py` files were also checked: their byte hashes match their cycle-000 no-op versions.
For these two particular seed-0 runs, selecting a later cycle does not supply a more complete model.
The frozen programs declare no `MODEL_STATE_INIT` quantities.
That gives these particular programs zero declared memory dimensions; it does not establish that real heat, curing, or other hidden processes are absent.

## Common state contract

Sources are the [noise injector](../../predicators/observation_noise.py), [recording projection](../../predicators/code_sim_learning/inference_recording.py), [robot wrapper](../../predicators/pybullet_helpers/robots/single_arm.py), [visible simulator loader](../../predicators/code_sim_learning/base_simulator.py), and [model-memory contract](../../predicators/agent_sdk/prompts/subclass_model.md).
The rows below apply separately to every domain inventory that follows.

| Quantity and source | What the interface establishes | What remains unknown | Prior and feasible representation | Conditioning or elimination | Remaining continuous dimensions and discrete cases |
| --- | --- | --- | --- | --- | --- |
| Controlled robot joint positions, public observation | Nine exact positions at each observed step | Their relation to contacts and subsequent dynamics | Initial positions are coordinate constraints; future outputs must follow candidate dynamics | Eliminate nine initial position coordinates; do not overwrite later predictions | Zero sampled initial controlled positions; trajectory constraints remain |
| Robot Cartesian pose and fingers, public features | Seven exact values, partly redundant with joint positions | Consistency with fixed kinematics and any unobserved joints | Check the actual observation map and its dependent coordinates | Remove only proven deterministic redundancy; retain any independent constraint | Count depends on the robot map, not seven additional free coordinates |
| Robot motion and non-controlled movable joints, URDF and public observations | Controlled joint velocities are not measured; the URDF establishes which joints can move | Initial velocities and any unobserved movable positions | Explicit initial-position and motion prior; URDF limits alone are not a reset guarantee, and zero velocity needs a guarantee or prior atom | Fixed URDF joints and a fixed base have no physical degrees of freedom; missing movable values are not zero by omission | Nine controlled velocities, plus unknown movable-joint coordinates pending reduction |
| Object geometry and static descriptors | Exact sizes, colors, labels, and selected device links | Whether a value is an immutable input or a predicted output | Fixed known geometry; represent true static scene placement once per episode | Condition on immutable descriptors after verifying the program uses them as inputs | No dimensions for conditioned descriptors; scene pose uncertainty remains |
| Object pose and motion | Selected noisy pose coordinates; most velocities are unobserved | Omitted rotations, linear/angular motion, and dependencies through support contacts | A normalized prior on feasible assemblies, with explicit contact/rest/moving cases; support and case masses still to be specified | Derive constrained coordinates only with the induced density; no noisy-pose plug-in initialization | Domain rows below count known observation coordinates; final free dimensions remain unresolved |
| Grasp/attachment state | Some exact flags, but no complete original constraint frames | Endpoint frames, pending commands, and native attachment history | A coherent assembly representation; simulate all recorded actions from its root | Root flags constrain cases; later weld frames and command queues come from candidate history | Continuous frame coordinates and discrete alternatives depend on the root assembly |
| Subclass memory | Declared initialization is part of each candidate program | Undeclared Python state or external mutable resources | Use `MODEL_STATE_INIT` for the model's own reset memory; reject incomplete runtime capture | Derive parameter-dependent declared initialization; continue memory across the full prefix | Zero declared memory dimensions in this frozen program set; future programs require new rows |

The environment reset tool promises a restart of the level, not a universal observation of every initial velocity or attachment frame.
Reading a default in a privileged restoration path is not evidence that the agent observed that value.
Similarly, `rollout_states` starting a simulated trial at rest is a legacy fitting assumption, not a reason to assign all physical initial velocities probability one at zero.

## Bridge

The recording contains five blocks, one bottle, two site markers, and the robot.
The [environment source](../../predicators/envs/pybullet_bridge.py) separates partially observed block features from hidden curing counters and attached partners.
Its hidden task generator and evaluator restoration code are diagnostic sources, not an agent prior.

| Quantity and source | What the interface establishes | What remains unknown | Prior and feasible representation | Conditioning or elimination | Remaining continuous dimensions and discrete cases |
| --- | --- | --- | --- | --- | --- |
| Five block poses and geometry, public features | Noisy xyz/roll/pitch/yaw; exact half-extents and colors | True poses and their support contacts | Feasible oriented boxes under known geometry; contact-case density not yet specified | Condition on size/color, score all noisy pose readings | 30 noisy pose coordinates before support reduction |
| Bottle and two sites, public features | Bottle noisy xyz/rot; sites noisy xyz | Bottle's omitted rotations; true scene locations | Bottle geometry plus static site locations; avoid treating each repeated reading as a new location | Condition only on descriptors established as fixed inputs | 10 noisy coordinates; omitted bottle orientation still needs treatment |
| Motion, holding, glue, and attachments | Exact holding and glue flags | Body velocities, cure state, weld topology/frames | Common motion and coherent-assembly contract; incomplete no-op program has no invented cure-memory variables | Keep predicted glue/holding transitions as exact constraints | Six potentially movable bodies; root attachment cases unresolved |
| Frozen program memory | No declared memory | Missing mechanism is a model-adequacy issue | No-op model remains an explicitly incomplete control | No evaluator cure counters supplied to the candidate | Zero declared memory coordinates |

## Fan

The recording contains one ball, four fans, four switches, four boundary objects, one wall, one target, and the robot.
The supplied [visible base source](../../predicators/envs/pybullet_fan_base.py) establishes geometry and device accessors without exposing the hidden airflow task generator.

| Quantity and source | What the interface establishes | What remains unknown | Prior and feasible representation | Conditioning or elimination | Remaining continuous dimensions and discrete cases |
| --- | --- | --- | --- | --- | --- |
| Ball, public features | Noisy xyz and exact radius | True position, velocity, spin, contact state | Sphere pose/motion under known geometry; eliminate rotational gauge only after showing the program and contacts are invariant to it | Fix radius; retain xyz likelihood | Three noisy coordinates; omitted motion and rotational relevance unresolved |
| Fans, switches, boundaries, wall, and target | Noisy xyz/rot; exact extents and selected device links | True fixed layout and any relevant articulated state | One static scene representation per episode with geometric/device dependencies | Condition on immutable extents/links, not noisy layout | 56 noisy coordinates before shared-layout reductions |
| Switch and target events | Exact `is_on` and `is_hit` | Event-consistent dynamics and initial articulated configuration | Discrete cases consistent with observed flags and public mechanisms | Indicators for predicted events; observed initial flags constrain cases | Compatible event cases, not extra continuous noise |
| Frozen program memory | No declared memory | Invalid parameter declarations in the earliest version | Keep invalid logarithmic bounds classified as setup failures | Do not repair bounds silently for an inference comparison | Zero declared memory coordinates |

The later [prior-provenance audit](offline-fitter-comparison.md#fan-prior-provenance) verifies identical dynamics methods across all three saved Fan versions and distinguishes optimizer declarations from probability densities.
The planned posterior comparison uses an explicitly declared uniform prior on the earliest [0, 1] support with the latest executable program; the invalid earliest artifact remains unchanged.
This resolves that parameter-prior choice for the development experiment, but not the static-layout, contact or articulated-state rows above.

The [articulated-state follow-up](fan-articulated-prior.md) implements exact Boolean conditioning of a declared switch rest/motion prior and verifies 2,048 native state readbacks.
The four prismatic switches have 29.6 mm enforced travel and a 14.8 mm on threshold; unchanged URDF metadata reports a larger nominal interval.
Four public fan objects represent twenty physical fan bodies with continuous, collision-bearing rotor joints.
The later articulated replay correction preserves all these nonrobot joint states, which earlier snapshots omitted.
A declared uniform rotor position/velocity component adds forty continuous coordinates; eight joint slider/rotor samples restore exactly and repeat sixteen recorded actions in fresh worlds.
The articulated portion has 40 to 48 continuous dimensions across sixteen switch rest/moving combinations.
The subsequent [full Fan root law](fan-initial-scene.md) combines those variables with fixture placement, box orientation cases, ball pose/motion and conditioned robot state under an explicit fixed-mounting and contact policy.
That development representation has 82 to 112 continuous state coordinates, or 83 to 113 with its airflow parameter, across 65,536 case combinations before support and remaining observations.
Eight sampled whole scenes pass the declared support gate and repeat exactly, but none of the earlier eight saved candidates satisfies the entire 132-action event sequence.
The subsequent placement/contact audit finds a full-recording positive witness and three nearby positive perturbations, all reproduced exactly in fresh worlds.
Numerical posterior exploration and predictive adequacy remain open; these selected support witnesses are not a calibrated task prior or a production posterior.

## Domino

The recording contains six dominoes and the robot.
The [domain implementation](../../predicators/envs/pybullet_domino) and observation schema show that each domino exposes roll and yaw but not pitch.
The task-cache repair preserves exact robot initialization; it does not reveal missing object coordinates to the agent.

| Quantity and source | What the interface establishes | What remains unknown | Prior and feasible representation | Conditioning or elimination | Remaining continuous dimensions and discrete cases |
| --- | --- | --- | --- | --- | --- |
| Six domino poses, public features | Noisy xyz/roll/yaw; exact colors and held flags | Six omitted pitches, true poses, contact modes | Oriented rigid bodies using known geometry; stable support and moving cases need normalized prior mass | Condition on colors; retain holding constraints and pose likelihood | 30 noisy pose coordinates plus up to six omitted rotation coordinates before constraints |
| Domino motion and grasp state | No direct body velocity measurement | Linear/angular velocities and grasp frames | Common motion/assembly representation; contacts couple pose and motion | Reconstruct all later states from one episode root | Six potentially movable bodies; initial grasp alternatives unresolved |
| Frozen program memory | Five physical parameter declarations, no memory | Physical parameter/initial-state tradeoffs | Preserve the chosen original prior for each fixed program | Do not turn fitted bounds or optimizer output into a new prior | Zero declared memory coordinates |

## Boil

The recording contains one jug, one burner, one faucet, two switches, and the robot.
The [environment source](../../predicators/envs/pybullet_boil.py) stores partially observed heating state privately; it is not a recorded scalar sensor.

| Quantity and source | What the interface establishes | What remains unknown | Prior and feasible representation | Conditioning or elimination | Remaining continuous dimensions and discrete cases |
| Jug pose and liquid readings | Noisy xyz/rot, water volume, bubbling level; exact color/held flag | True liquid quantities, omitted orientation, motion, and heat | Feasible jug/liquid state under the candidate program; heating memory must be declared by a revised program | Score the declared 0.07 scalar noise; never initialize heat from evaluator metadata | Six noisy coordinates, including two liquid readings; omitted orientation/motion remain |
| Burner, faucet, and switches | Noisy positions/rotations and spilled level; exact on/off flags | True fixture poses, spilled amount, event state | Shared static fixture geometry plus valid liquid/event support | Condition on fixed scene descriptors; retain event constraints | 16 noisy coordinates before scene reductions |
| Jug motion, grasp, and model memory | Holding flag is exact; no declared memory in frozen no-op model | Real heat dynamics are missing from this program | Keep no-op as an inadequacy control; a later heat model needs its own initialization contract | Do not add hidden evaluator heat to make this program fit | One potentially movable jug; zero declared memory coordinates |

The no-op artifact also contains an optional geometry-dump side effect guarded by `BOIL_DUMP_GEOM`.
A replay identity must record its setting and external-file policy even if disabled in the comparison.
The [scalar inadequacy control](boil-incomplete-control.md) now quantifies the frozen program's constant-output limitation using all 265 noisy public frames.
Even the best unrestricted constants leave bubbling and water-volume RMSE at 5.51 and 5.93 times the declared sensor sigma, while spill remains near the noise scale.
This is a noisy predictive failure, not an exact contradiction or a reason to invent hidden heat initialization from evaluator metadata.

The later [Boil articulated-state component](boil-articulated-prior.md) identifies an additional unobserved revolute joint in the faucet asset, separate from the two observed on/off switches.
Its original position/motion law must remain unconditioned by those flags; the switch laws retain their actual event probabilities.
The component implementation passes numerical references, the native inventory confirms all three joints, and independent checks verify 48 initial joint draws plus all 18 complete conditional histories.
This adds explicit missing joint coordinates without declaring the rest of the Boil scene inventory complete.

The subsequent [full unheld-scene composition](boil-full-scene-prior.md) declares fixture xyz/orientation, all robot nuisance joints and motion, jug pose/motion/water and the three articulated components together.
Its 84 unit inputs include mixture selectors and unused case auxiliaries as well as eight parameter inputs; this is a computational map size, not a claim of 84 independent physical degrees of freedom.
The whole-scene geometry predicate has a parameter-independent normalizer only for the frozen historical feature-only program.
Independent density, coordinate and fresh-native geometry checks pass, including two full-recording supported sampled scenes and exact rule reuse across changed parameters.
This closes a declared prior/support construction for that unheld development case, while broader attachment cases, posterior adequacy and all-domain runtime closure remain separate requirements.

## Original non-hatch balloons

The recording contains one box, three balloons, three clips, one band, and the robot.
The supplied [visible base](../../predicators/envs/pybullet_balloons_base.py) exposes box speed as the norm of its three-dimensional linear velocity.
The hatch layout and acceptance rules are outside this inventory.

| Quantity and source | What the interface establishes | What remains unknown | Prior and feasible representation | Conditioning or elimination | Remaining continuous dimensions and discrete cases |
| Box and balloon poses | Noisy xyz; exact material/gas labels | Full box orientation, balloon orientations where attachments make them relevant, true pose | Rigid bodies and a coherent tied assembly under the visible geometry | Condition on labels; do not derive hidden orientations from archived metadata | 12 noisy pose coordinates; omitted rotations require an assembly chart |
| Clips and band | Clip noisy xyz/rot and exact on/off; band noisy xy and exact lo/hi | True placement and possible placement dependencies | Static scene/clip mechanism from the public base | Fix band bounds, not its noisy placement | 14 noisy coordinates before layout reductions |
| Box speed | Exact nonnegative speed, not a noisy pose channel | Velocity direction at positive speed; support of the zero-speed event | Positive speed gives a sphere constraint; exact zero requires explicit treatment of rest support and its prior mass | Do not merely keep one velocity component or add a noise floor; affine elimination alone does not handle the norm constraint | Two directional dimensions at positive speed; zero-speed conditional prior needs a separate construction |
| Ties, pops, releases, and forces | Exact tied/popped and clip states | Original weld frames, velocities, pending attachment/lift commands | Coherent assemblies initialized from the prior; full-history replay supplies subsequent commands and original frames | Indicators for discrete predictions; do not reattach at a resumed deflected pose | Four potentially moving bodies plus articulated clips; compatible initial assembly cases unresolved |
| Program memory and external input | No declared memory; an instance cache contains body IDs | Optional `model_params.json` and its precedence over runtime parameters | Body-ID cache is derived per world; freeze sidecar bytes or declared absence | Do not treat a cached body ID as a latent physical quantity | Zero declared memory coordinates; sidecar is a runtime dependency |

The audited first balloon observation has exact box speed `0.0`.
Consequently, the positive-speed sphere construction alone would not cover even this first development episode.
An explicitly declared rest component is one candidate prior design; its mass must be specified rather than inferred from an omitted velocity field.
The offline `RestOrGaussianVelocityPrior` now implements that component: probability `rho` at zero velocity and probability `1-rho` in a zero-mean isotropic Gaussian with declared per-axis standard deviation `sigma`.
Its exact zero-speed conditional selects the rest component, has no free velocity direction, and retains the observation mass `rho`.
At positive speed, two uniform sphere coordinates describe direction and the Maxwell radial density retains the information about `rho` and `sigma`.
Zero speed without a declared atom remains unsupported unless a separate conditional extension is specified.
This resolves the mathematical support construction for this candidate velocity component; it does not choose its hyperparameters or establish independence from pose, attachment, and robot state in a full physical prior.

The balloon program versions also change parameter names and narrow some bounds after observing transients.
A fixed-original-prior comparison cannot silently reuse the latest narrowed bounds as though they preceded the data.
Record parameter meanings, units, prior provenance, and sidecar precedence for each chosen fixed-program target.
The later [composed target and prior audit](balloons-composed-inference.md) records those five-version changes, declares a fixed ten-parameter development prior, and verifies every lower/middle/upper program override.
It also gives finite complete-recording conditional path factors under an explicit stochastic extension for one sampled root, while retaining another root's exact event failures.
The resulting joint target has 522-553 active continuous coordinates, including two conditional velocity directions at each of 235 actions, across the existing sixteen initial motion cases.
The later [native sphere audit](balloons-composed-inference.md#native-justification-and-quotient-representation) justifies a fixed-program quotient over the nine initial balloon-orientation coordinates, reducing that target to 513-544 active coordinates while retaining every motion case and all world-frame angular velocities.
This is a defined candidate inference problem, not a numerically adequate posterior or support for the deterministic sensor-only model.

## What this permits next

The later public-candidate map audit `22631317` constructs all five visible worlds using only projected feature values and fresh model-owned body handles.
At the initial boundary, all five reproduce every exact recorded feature when feature values retain float64 precision.
This is an initialization-map check, not a geometric-prior or trajectory-feasibility certificate.
The first audit's float32 cast introduced artificial descriptor mismatches; that audit implementation was corrected without changing the agent or recorded data.

The audit independently perturbs each of the 177 noisy coordinates in both directions, for 354 probes.
All sixteen Fan pose coordinates are reset to configured fan placements by the visible base and therefore cannot be independent initial-state coordinates under that initialization protocol.
Boil likewise resets spilled level and derives bubbling from its initialized hidden heat; a noisy observation of either is not itself a writable latent quantity.
Bridge's Euler-angle probes exhibit canonicalization and coupling near a pitch pole, so independent noisy Euler coordinates cannot be confused with independent physical rotations.
These are properties of the tested visible initialization map; reductions in a learned-program prior still require checking that program's initialization and output overrides.
See [the map audit](../../logs/uncertainty_scene_map_v3_20260912/reference-22631317.json).

The first complete Balloons candidate-scene reference now has an explicit conditional root law with 42 through 73 continuous dimensions across sixteen motion cases.
Its sixteen accepted samples satisfy the declared geometry policy, reproduce every exact initial feature, and give identical repeated fresh-world continuations.
They still contradict later exact observations, starting with moving robot joints or box speed at action one.
The [root-prior definition](balloons-initial-scene.md) records its assumptions, the reset-dependent fixture geometry, the supported-rest diagnostic, and the remaining exact-trajectory constraint.

The measured-coordinate counts sum to 177 noisy scalars per initial frame across five reset episodes.
They are an audit of evidence, not a 177-dimensional independent box prior.
The actual posterior dimension and discrete case count remain undefined until normalized physical support, robot-state reductions, assembly charts, and memory initialization are fixed.
Reporting a numerical joint dimension now would conceal those unresolved choices.

The [affine conditioning reference](../../predicators/code_sim_learning/inference_conditioning.py) implements one restricted reduction correctly: a square nonsingular equation `y = A(u) z + b(u)` eliminates `z` and retains `1 / abs(det(A(u)))` in the density on free coordinates `u`.
It supports exact initial coordinate elimination as a special case.
It does not solve overdetermined robot-contact trajectories or arbitrary nonlinear observations.
The separate velocity-prior construction handles a speed-norm observation only under its explicit rest/isotropic-Gaussian assumption.
Unsupported charts, rejected individual prior points, and numerical solve errors remain separate outcomes.
The offline batch sampler now retains these conditional-base factors throughout tempering and Metropolis moves, and returns full joint samples including eliminated coordinates.
Its integrated affine-dynamics references pass at the tested larger budget; they do not resolve the physical rows above.
The reviewed constant-output checker separately represents the frozen Bridge contradiction without requiring an invented physical prior or treating failed candidate search as proof.

The [rigid-assembly component](assembly-prior.md) now supplies one explicit generative representation for fixed relative geometry.
It derives all body poses and weld frames from a root pose, and child linear velocities include the shared angular motion around that root.
Free/rest, free/moving, and planar-support components have 6, 12, and 3 continuous coordinates, independent of the number of bodies whose relative poses are fixed by that model.
These dimensions describe those declared components only; the recordings do not establish fixed relative geometry or identify a support case merely by omitting fields.
Normalized case masses, uncertain relative transforms, scene regions, and robot motion still require specification before reporting a full task-prior dimension.
The generated physical reference passed all 12 trials, including plane contact and full-prefix replay, without loading evaluator tasks or recorded hidden state.

The [joint-state component](robot-state-prior.md) now conditions exact initial positions while retaining their original prior density and all unknown joint motion.
For the audited Fetch schema, compatible rest and moving components have 4 and 17 free coordinates respectively.
The initial-position check passed for Bridge, Fan, Domino, and Boil under the declared bounds, with exact restoration of 256 sampled joint states.
The original balloons initial shoulder-lift angle is -1.5119263 rad, outside the assumed URDF lower bound -1.221 rad, so that component returns no conditional samples for balloons.
This is an incompatible prior assumption, not an agent failure or grounds to clip the observation.
The same visible-geometry audit finds identical public robot features but different head collision envelopes in all five domains, ruling out kinematics alone as a justification for eliminating head state.
Scene-specific irrelevance, initialization-law support, and joint/body collision compatibility still require resolution.

The subsequent [composed reference](scene-prior-composition.md) adds an explicit Gaussian joint initialization law, preserving the bounded-prior rejection as a control.
The robot wrapper reproduces the exact recorded joint vector, including the balloons angle outside the URDF interval.
Global rejection then combines conditioned joints with generated rigid assemblies, retaining the declared base measure rather than resampling only a colliding body.
The support predicate explicitly permits the supplied fixed wheel/plane fixture contacts, whose -9.675 mm signed distance follows from the checked wheel radii and fixed centers; all other tested intersections are rejected.
It produced 80 accepted generated scenes in 82 draws across all five visible environments.
This closes those component-support obstructions under a declared model, not the historical scene-layout, case-mass, full-runtime, or exact-trajectory requirements.

Before the physical comparison, close the unresolved rows with an explicit generative initial-state model and its normalizing/case factors.
Then validate its support on known-model recordings and measure exact-output feasibility on frozen learned programs.
Use the existing complete-prefix replay; do not promote arbitrary mid-run snapshots to exact engine state.

The subsequent [Domino integration](domino-joint-inference.md) closes a restricted positive case with a declared global rest/moving mixture, full robot state and six uncertain body poses.
Its active dimensions are 27 at rest and 94 when moving, including five physical parameters, plus one discrete case.
It conditions on the observed unheld initial case; this does not close other attachment cases or the inventories of the other four domains.
All eight preflight roots were geometrically feasible and repeated exactly over 64 actions, with finite full-output likelihood for two roots.
The supported-root construction is now available for inference testing; reliable physical posterior sampling is still a separate requirement.
