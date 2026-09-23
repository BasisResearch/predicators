# EMPIRIC related-work reading audit

## Scope and standard

This is an author-facing working record, not a claim that the literature search is exhaustive.
The review began on 2026-09-22 with the current related-work section and then expanded through primary-source searches and reference lists.
Only a paper marked **complete** has had every page of the retrieved PDF read, including its embedded appendices, examples, and references.
Downloading a PDF, inspecting its abstract, or finding it in another bibliography does not qualify as complete reading.
Separate supplementary files and project videos are not included in that label unless explicitly listed.
This review does not reproduce experiments or independently prove authors' theoretical claims.
Descriptions below distinguish reported evidence from our interpretation of relevance to EMPIRIC.
Temporary PDFs, page-labelled text, source URLs, page counts, and SHA-256 fingerprints are cached under `/tmp/empiric-literature-20260922/`.
The initial pass completed 35 full-paper readings before revising the manuscript.
The 2026-09-23 follow-up completed four additional user-suggested papers, bringing the total to 39 complete readings.
The current related-work section cites 36 of these papers.
WorldTest and KinDER were cited outside related work in an earlier revision, but those mentions are absent from the current manuscript; their bibliography entries remain available.
MPMWorlds remains in this audit but is omitted from the manuscript to keep the physical-program discussion focused.
Additional discovered candidates remain explicitly unread below and do not support new manuscript claims.

## Findings to carry into the revision

- Do not describe all prior simulator-learning work as merely tuning existing physical parameters.
  NeuralSim explicitly learns forces absent from its analytical engine.
- Do not claim that all program-world-model methods must build everything from scratch.
  WorldCoder transfers and edits earlier programs, and its introduction explicitly discusses that reuse.
- Do not define coding agents as incapable of explicit predictive modeling.
  EMPIRIC's own baseline audit contains agents that write predictive programs.
- Separate scene reconstruction, parameter identification within a model family, and writing new mechanism code.
  These are related but different tasks, not a ranking of whole fields.
- Keep claims about partial observability and uncertainty specific to each method and its experiments.
  A program can encode stochastic dynamics without representing epistemic uncertainty over its own parameters or structure.

## Completed readings

### NeuralSim: Augmenting Differentiable Simulators with Neural Networks

- Source: [arXiv:2011.04217v2](https://arxiv.org/abs/2011.04217v2), 8/8 PDF pages, complete.
- Method and evidence: neural components can be inserted within a differentiable rigid-body engine, not just added to its final predicted state (Sections III and IV).
  Experiments cover planar-pushing friction, previously absent viscous forces for a swimmer, sparse selection of augmentation inputs, and replacement of an MPC component.
- Relevance: a direct antecedent for retaining analytical mechanics while learning missing effects.
  EMPIRIC's distinction should be its agent-written mechanism programs and interaction/inference workflow, not the invention of hybrid physics or the first ability to add an unmodeled force.
- Decision: add prominently to the physical-model paragraph.
- Avoid: saying NeuralSim only calibrates parameters, always uses fixed hand-selected inputs, or has demonstrated EMPIRIC's continual mechanism-discovery setting.
- Claim locations: Sections III-A, IV-A through IV-D; especially IV-B's swimmer and IV-C's sparse-input discovery.

### WorldCoder

- Source: [arXiv:2402.12275v3](https://arxiv.org/abs/2402.12275v3), 65/65 PDF pages, complete, including proof, generated code, prompts, and checklist.
- Method and evidence: synthesizes Python transition and reward functions from object-structured interaction data, repairs counterexamples, and imposes a planner-linked optimism condition (Section 2).
  Experiments use Sokoban, MiniGrid, and a fully observed fluent representation of ALFWorld (Section 3).
  The curriculum transfers earlier code to new dynamics and goals.
- Relevance: direct antecedent for executable world models learned through interaction and revised using prediction errors.
  Its evaluated deterministic symbolic models differ from EMPIRIC's physical-engine extensions fitted to noisy continuous observations.
- Decision: retain as a central comparison.
- Avoid: claiming it has no exploration principle, cannot reuse mechanisms, handles partially observed ALFWorld as evaluated here, or proves general sample-efficiency guarantees without its assumptions.
- Claim locations: Sections 2.1-2.5, 3, 5; Appendices A, B, D-F.

### Agentic Real2Sim

- Source: [arXiv:2607.19190v4](https://arxiv.org/abs/2607.19190v4), 8/8 PDF pages, complete.
- Method and evidence: converts recorded interaction episodes into runnable twins through visual reconstruction, physical-property priors, scene alignment, and simulator-guided grasp refinement (Section III).
  DROID conversion, additional embodiments, and policy evaluation are considered (Section IV).
  The limitations explicitly state that object-level system identification and direct physical-parameter validation are not supported in the reported framework (Section V).
- Relevance: supports scene initialization and replay alignment, rather than evidence of autonomous missing-mechanism discovery through new robot experiments.
- Decision: retain in a compact reconstruction paragraph.
- Avoid: dismissing it as geometry-only, claiming it fits all physical parameters from trajectories, or treating successful episode replay as identified dynamics.
- Bibliographic update applied: the v4 author list adds Pengyu Jing, Yixian Cheng, and Yiduo Qu to the previous BibTeX entry.

### TheoryCoder: Synthesizing World Models for Bilevel Planning

- Source: [arXiv:2503.20124v2](https://arxiv.org/abs/2503.20124v2), 35/35 PDF pages, complete.
- Method and evidence: learns low-level Python transition code while supplied PDDL operators and predicate checks guide hierarchical planning and exploration.
  Evaluations cover grid-based games and reuse across levels, including larger instances.
- Relevance: an antecedent for using structured prior knowledge to reduce both model-learning and planning demands.
- Decision: retain beside WorldCoder.
- Avoid: claiming that it synthesizes every level of its representation, lacks exploration, or reports the same interaction-cost metric as EMPIRIC.
- Claim locations: Sections 3-5; the appendix specifies supplied abstractions, prompts, and generated programs.

### GIF-MCTS: Generating Code World Models with Large Language Models Guided by Monte Carlo Tree Search

- Source: [arXiv:2405.15383v2](https://arxiv.org/abs/2405.15383v2), 40/40 PDF pages, complete.
- Method and evidence: searches over code generation, improvement, and repair using supplied descriptions and offline transition data.
  The benchmark includes discrete and continuous-control environments, with downstream MCTS or cross-entropy planning.
  Its evaluated continuous models do not call the original physics simulator; adding such tools is explicitly proposed as future work.
- Relevance: a particularly precise comparison for EMPIRIC's reuse of engine dynamics rather than synthesized approximations of all transitions.
- Decision: retain.
- Avoid: calling the work exclusively symbolic/discrete, conflating offline validation data with autonomous exploration, or asserting that its proposed future integration has already been evaluated.
- Claim locations: Sections 3-6.1; Appendices E, K, and O.

### POMDP Coder

- Source: [CoRL 2025 publisher PDF](https://raw.githubusercontent.com/mlresearch/v305/main/assets/curtis25a/curtis25a.pdf), 48/48 PDF pages, complete.
- Method and evidence: synthesizes Pyro initial-state, transition, observation, and reward models using coverage-based repair; combines them with particle-filter belief updates and online planning.
  Its principal experiments start with ten demonstrations and assume post-hoc access to intermediate states.
  Experiments use discrete POMDPs, MiniGrid, and real mobile-robot search.
- Relevance: direct precedent for program learning with partial observability, but with different training information and planning machinery from EMPIRIC.
- Decision: retain; distinguish post-hoc state supervision from inference from noisy recordings.
- Avoid: describing its learned probabilities as a posterior over program structures, denying its supplied geometric helpers, or claiming EMPIRIC performs the same full belief-space planning.
- Claim locations: Sections 3-4, 5.4, 7; Appendices C and E.
- Bibliographic version discrepancy: the retrieved PDF lists six authors, but the publisher's landing page and exported BibTeX list seven, including Joshua Tenenbaum.
  Preserve the publisher's citation metadata rather than silently deleting an author based on the PDF alone.

### Pinductor

- Source: [arXiv:2605.13740v1](https://arxiv.org/abs/2605.13740v1), 58/58 PDF pages, complete.
- Method and evidence: jointly proposes executable POMDP components and repairs them using particle-filtered observation compatibility and committee-disagreement diagnostics.
  Evaluation covers MiniGrid with ten manually collected offline trajectories and additional online experience, without realized hidden-state supervision.
  Its ranking objective is a posterior-expected kernel score, explicitly not marginal likelihood or an ELBO.
- Relevance: a close precedent for observation-grounded program revision under partial observability.
  EMPIRIC's difference is not simply removing state labels: it extends physical-engine mechanics and infers continuous parameters within a selected program.
- Decision: retain prominently with POMDP Coder.
- Avoid: calling committee disagreement a calibrated program posterior, describing its current evaluation as robot manipulation, or equating its belief-space planner with EMPIRIC's point-state rehearsals.
- Claim locations: Sections 4-5; Appendices A, C, D, and I.

### Simulator-Augmented Interaction Networks (SAIN)

- Source: [arXiv:1904.06580v1](https://arxiv.org/abs/1904.06580v1), 7/7 PDF pages, complete.
- Method and evidence: combines Bullet predictions with learned object-interaction residuals and uses the model in receding-horizon search.
  Experiments include indirect multi-object pushing on a real robot and transfer to another surface and disk-size arrangement.
- Relevance: explicit prior evidence for the modeling and generalization benefits of combining learned corrections with engine physics.
- Decision: add alongside NeuralSim; a stronger match for EMPIRIC's physical-inductive-bias discussion than unrelated reconstruction systems.
- Avoid: suggesting hybrid models previously lacked control demonstrations, object compositionality, or multi-step rollout training.
- Claim locations: Sections III and IV.

### Galileo

- Source: [author-hosted NeurIPS 2015 PDF](https://cncl.yale.edu/assets/pdf/2015_YildirimWuGalileo.pdf), 9/9 PDF pages, complete.
- Method and evidence: infers mass and friction using Bullet rollouts and velocity-based likelihoods, and uses inferred properties to train a visual recognition model.
  Although the generative description includes shape and position, the implemented inference fixes these using tracking and samples mass/friction.
  Evaluations concern passive videos and physical predictions, not robot-selected experiments.
- Relevance: foundational simulation-based physical inference and the distinction between perceptual initialization and dynamical identification.
- Decision: retain, concisely.
- Avoid: claiming all scene variables are jointly sampled in its experiments or that all evaluated physical judgments are accurate; the slope-change task is near chance.
- Claim locations: Sections 3.1-3.2, 5, and 6.

### BayesSim

- Source: [RSS 2019 publisher PDF](https://www.roboticsproceedings.org/rss15/p29.pdf), 10/10 PDF pages, complete.
- Method and evidence: learns a conditional mixture density from simulated trajectories to approximate a posterior over simulator parameters.
  Experiments examine parameter ambiguity and posterior-based domain randomization in simulated control and manipulation tasks.
- Relevance: direct precedent for retaining multiple plausible physical parameters rather than relying on one fitted value.
- Decision: add to the inference discussion.
- Avoid: presenting its evaluation as physical-robot experiments or attributing mechanism-code discovery to parameter inference.
- Claim locations: Sections III-V.

### Sim-to-Real Transfer with Neural-Augmented Robot Simulation

- Source: [CoRL 2018 publisher PDF](https://proceedings.mlr.press/v87/golemo18a/golemo18a.pdf), 12/12 PDF pages, complete.
- Method and evidence: learns recurrent corrections between simulated and observed transitions, then trains policies in the augmented simulator.
  Experiments include synthetic backlash and a real Poppy Ergo robot.
- Relevance: history-dependent simulation correction already exists; recurrent memory alone is not EMPIRIC's distinction.
- Decision: add beside NeuralSim and SAIN.
- Avoid: describing prior residual models as necessarily memoryless or incapable of representing unmodeled effects.
- Claim locations: Sections III-V; Appendices A-C.

### Task-Directed Exploration in Continuous POMDPs for Robotic Manipulation of Articulated Objects

- Source: [arXiv:2212.04554](https://arxiv.org/abs/2212.04554), 8/8 PDF pages, complete.
- Method and evidence: STRUG scores uncertainty by compatibility of plans across belief particles and uses it in belief-space exploration.
  Simulated articulated-object manipulation starts with uncertain kinematic parameters; its stated assumptions exclude informative actions that irreversibly destroy goal reachability.
- Relevance: task-relevant uncertainty and exploration are established ideas, distinct from learning new mechanism programs.
- Decision: add to the inference and exploration paragraph.
- Avoid: suggesting EMPIRIC's logs establish the benefit of its information-gain tool or claiming STRUG learns arbitrary mechanism classes.
- Claim locations: Sections III-V.

### PoE-World

- Source: [arXiv:2505.10819v4](https://arxiv.org/abs/2505.10819v4), 30/30 PDF pages, complete.
- Method and evidence: composes small program experts into a probabilistic next-observation model, optimizes their weights, and learns constraints for planning.
  Short demonstrations initialize models that are repaired through interaction; Atari experiments test new object arrangements and multiplicities.
- Relevance: close precedent for modular, probabilistic, reusable program models, not only monolithic transition code.
- Decision: add to executable world models.
- Avoid: claiming EMPIRIC introduces program modularity or history dependence; PoE-World conditions on observation history and uses object-contact helpers.
- Claim locations: Sections 3-4, Appendix A.1-A.4.

### PETS

- Source: [arXiv:1805.12114v2](https://arxiv.org/abs/1805.12114v2), 17/17 PDF pages, complete.
- Method and evidence: bootstrapped probabilistic dynamics networks represent epistemic and aleatoric uncertainty; sampled trajectories support receding-horizon action optimization.
  Evaluation uses four simulated continuous-control tasks and component ablations.
- Relevance: established use of uncertain dynamics for sample-efficient control.
- Decision: retain, concisely.
- Avoid: crediting PETS with evaluated information-seeking exploration, which it leaves for future work, or equating its ensemble with EMPIRIC's conditional parameter posterior.
- Claim locations: Sections 4-7, Appendix A.

### Plan2Explore

- Source: [arXiv:2005.05960v2](https://arxiv.org/abs/2005.05960v2), 13/13 PDF pages, complete.
- Method and evidence: learns an exploration policy in an imagined latent dynamics model using ensemble disagreement as an information-gain proxy.
  Reward-free exploration supports downstream adaptation in pixel-based continuous-control tasks.
- Relevance: precedent for selecting actions using predicted information, distinct from task-specific mechanism-program revision.
- Decision: retain as exploration context, without suggesting EMPIRIC experimentally establishes the same benefit.
- Avoid: calling disagreement exact mutual information or claiming the experiments learn without environment interactions.
- Claim locations: Sections 2-5 and Appendix A.

### TossingBot

- Source: [arXiv:1903.11239v3](https://arxiv.org/abs/1903.11239v3), 13/13 PDF pages, complete (journal revision).
- Method and evidence: learns a correction to the release velocity supplied by a ballistic controller, jointly with grasp selection.
  Simulation and real-robot experiments test throwing new objects and to new target locations.
- Relevance: an important physical-inductive-bias precedent, but the residual is in control parameters, not a learned forward-simulator transition.
- Decision: retain with that distinction explicit.
- Avoid: describing it as learning a general simulator or merely fitting known physical parameters.
- Claim locations: Sections II, IV, VI, and VII.

### Code-as-World

- Source: [arXiv:2608.27549v1](https://arxiv.org/abs/2608.27549v1), 31/31 PDF pages, complete.
- Method and evidence: constructs executable scenes from text or recorded video and iterates using simulation and verification feedback.
  Its SDK supports MuJoCo physics as well as animation-based reconstruction; its main downstream application generates supervision for physical video reasoning.
- Relevance: direct precedent for coding into an existing physics engine, not merely generating a standalone transition function.
- Decision: retain and acknowledge that overlap explicitly.
- Avoid: claiming physics-engine integration is new, treating video realism translation as robot-policy transfer, or attributing autonomous robot experimentation to its reported evaluation.
- Claim locations: Sections 3-5, 7.2; Appendices B.2 and C.2.

### Code World Models for General Game Playing

- Source: [arXiv:2510.04542v1](https://arxiv.org/abs/2510.04542v1), 58/58 PDF pages, complete.
- Method and evidence: learns game simulators from descriptions and offline trajectories, including functions that reconstruct hidden histories for imperfect-information search.
  Closed-deck experiments learn both the simulator and inference function without supplied internal states.
- Relevance: another precedent for executable models with partial observability.
- Decision: retain concisely with probabilistic program approaches.
- Avoid: describing observation-consistent hidden-history sampling as calibrated Bayesian inference, or attributing online model revision to the reported experiments.
- Claim locations: Sections 3-5; Appendices B-E and I.

### MPMWorlds

- Source: [arXiv:2606.01538v2](https://arxiv.org/abs/2606.01538v2), 16/16 PDF pages, complete.
- Method and evidence: benchmarks synthesis of executable Taichi material-point simulations against video continuation on synthetic two-dimensional material dynamics.
  Generated programs are selected by reconstruction of the observed prefix; experiments vary access to scene geometry and material information.
- Relevance: physical program synthesis and long-horizon prediction, but not learning through robot-selected interventions or skill execution.
- Decision: omit from the manuscript in this pass; Code-as-World already provides a fully read physical-program comparison, and robot intervention is more central to the present discussion.
- Avoid: claiming the study demonstrates robot planning or universally superior geometry reconstruction from programs.
- Claim locations: Sections 3-5, Appendix D.

### Scalable Real2Sim

- Source: [arXiv:2503.00370v2](https://arxiv.org/abs/2503.00370v2), 8/8 PDF pages, complete.
- Method and evidence: reconstructs object geometry and identifies inertial properties through robot manipulation and joint-torque measurements.
  Optimized excitation trajectories explicitly seek informative measurements within a rigid-body parameterization.
- Decision: retain as active physical identification, not merely image reconstruction.
- Avoid: claiming EMPIRIC introduces robot-selected identification experiments or that all reported inertial estimates are accurate; rotational inertia remains difficult.
- Claim locations: Sections III-D and V.

### One-Shot Real-to-Sim

- Source: [arXiv:2412.00259v4](https://arxiv.org/abs/2412.00259v4), 8/8 PDF pages, complete.
- Method and evidence: combines differentiable geometry reconstruction and contact simulation to estimate appearance, geometry, and rigid-body physical parameters from an RGB-D interaction sequence.
  Simulated and real pushing experiments evaluate predictions for subsequent pushes.
- Decision: retain; distinguish joint parameter/geometry inference from mechanism-code revision.
- Avoid: calling it geometry-only or assuming accurate recovery despite severe occlusion and rotational ambiguities.
- Claim locations: Sections III-V.

### Code as Policies

- Source: [arXiv:2209.07753v4](https://arxiv.org/abs/2209.07753v4), 16/16 PDF pages, complete.
- Method and evidence: generates hierarchical Python policies that compose perception, geometric computation, and control APIs, including feedback controllers.
  Experiments include simulated tasks and several physical robot embodiments.
- Decision: retain as a foundational coding-agent reference.
- Avoid: defining code policies as open-loop or intrinsically model-free; also do not attribute evaluated execution-driven code repair to this version, whose appendix reports difficulties with that approach.
- Claim locations: Sections III-IV; Appendices A, F, and K.

### CaP-X

- Source: [arXiv:2603.22435v2](https://arxiv.org/abs/2603.22435v2), 58/58 PDF pages, complete.
- Method and evidence: benchmarks manipulation coding agents across feedback and abstraction interfaces; Agent0 combines visual descriptions, a synthesized skill library, and candidate-policy aggregation.
  Its reported library is synthesized in a single pass from successful programs, not continually revised during deployment.
- Decision: retain as the closest coding-agent interface comparison.
- Avoid: denying feedback, geometric planning, or skill reuse, or claiming its benchmark specifically evaluates learned predictive simulators.
- Claim locations: Sections 2-5; Appendices H and K.

### ASPIRE

- Source: [arXiv:2607.00272v1](https://arxiv.org/abs/2607.00272v1), 43/43 PDF pages, complete.
- Method and evidence: converts execution failures and validated repairs into reusable skill guidance, with evolutionary policy search and transfer across tasks.
  It evaluates simulated manipulation and real-robot adaptation.
- Decision: retain as persistent skill and repair learning, distinct from learning physical transition programs.
- Avoid: claiming it starts without human-written skill templates, or describing its real-robot adaptation as frozen-policy zero-shot transfer.
- Claim locations: Sections 2-3; Appendices D-E.

### ENPIRE

- Source: [arXiv:2606.19980v2](https://arxiv.org/abs/2606.19980v2), 28/28 PDF pages, complete.
- Method and evidence: coding agents improve scripted and learned policies through real-robot experiments, sharing code across robot stations.
  Human-guided environment construction establishes safety, rewards, and resets before autonomous policy research; experiments include heuristic control, behavioral cloning, and reinforcement learning.
- Decision: retain as a direct precedent for robot-in-the-loop autonomous experimentation.
- Avoid: claiming real-robot experimental iteration is unique to EMPIRIC, or that ENPIRE specifically learns a predictive physics simulator.
- Claim locations: Sections 2-3; Appendices A-B and D.

### NSRTs

- Source: [arXiv:2105.14074v3](https://arxiv.org/abs/2105.14074v3), 8/8 PDF pages, complete.
- Method and evidence: jointly learns symbolic operators, local continuous neural transition models, and action samplers, then combines symbolic search with simulated continuous refinement.
  Four simulated domains test object-count and task-horizon generalization from supplied transition data.
- Decision: retain as learned representations for bilevel planning.
- Avoid: describing NSRTs as purely symbolic or unable to predict continuous effects; the evaluated setting instead assumes deterministic, fully observed dynamics and given predicates.
- Claim locations: Sections III-VII.

### Integrated Task and Motion Planning

- Source: [arXiv:2010.01083v1](https://arxiv.org/abs/2010.01083v1), 30/30 PDF pages, complete.
- Method and evidence: surveys joint discrete action selection and continuous constraint satisfaction, including how motion-level failures inform task-level search.
  Its extensions explicitly discuss kinodynamics, uncertainty, and learning.
- Decision: retain as background for skill-level and continuous reasoning.
- Avoid: turning the survey's restricted core presentation into an impossibility claim about all TAMP methods.
- Claim locations: Sections 1.2, 3, and 4.

### VisualPredicator

- Source: user-supplied `liang25vp.pdf`, matching [arXiv:2410.23156v2](https://arxiv.org/abs/2410.23156v2), 29/29 PDF pages, complete.
- Method and evidence: proposes and selects Python predicates combining perceptual queries and symbolic computation, while learning high-level operators from online interaction.
  Five simulated domains evaluate generalization to more objects and complex goals.
- Decision: retain as a predecessor in learned planning abstractions.
- Avoid: claiming its reported experiments establish full probabilistic state estimation or flawless visual grounding; Appendix B.5 documents important perception failures and provided predicates.
- Claim locations: Sections 3-6; Appendices B and D.

### ExoPredicator

- Source: user-supplied `liang26ep.pdf`, matching [arXiv:2509.26255v3](https://arxiv.org/abs/2509.26255v3), 41/41 PDF pages, complete.
- Method and evidence: learns abstract exogenous processes with activation conditions and stochastic delays, using language-model proposals and variational inference.
  Training starts with one or two demonstrations and continues through interaction; evaluation uses supplied object features and harder simulated tasks.
- Decision: retain with explicit acknowledgement that delayed physical effects are already modeled.
- Avoid: claiming symbolic methods cannot learn timing, confusing process-parameter fitting with a posterior over programs, or describing the evaluated perception as unrestricted image-grounded inference.
- Claim locations: Sections 3-6; Appendices A.3-A.5 and C.3.

### URDFormer

- Source: [RSS 2024 PDF](https://www.roboticsproceedings.org/rss20/p124.pdf), 18/18 PDF pages, complete.
- Method and evidence: learns to predict articulated scene structure from images using procedurally generated, visually augmented training data.
  Reconstructed environments support randomized simulation training and real articulated-object manipulation.
- Decision: retain as a source of base-scene structure.
- Avoid: attributing identification of mass, inertia, or friction to the reported system; Section VI leaves these properties to future work.
- Claim locations: Sections III-VI.
  Separately linked supplementary material is not covered by this reading.

### Real2Code

- Source: [arXiv:2406.08474v2](https://arxiv.org/abs/2406.08474v2), 17/17 PDF pages, complete.
- Method and evidence: combines part segmentation and shape completion with language-model generation of articulation code from oriented bounding boxes.
  Experiments evaluate geometric and joint reconstruction, including real objects.
- Decision: retain as programmatic scene reconstruction, not new-mechanism learning.
- Avoid: claiming demonstrated robot-policy learning or physical identification of friction from interventions; Section 5 lists additional physical properties as future work.
- Claim locations: Sections 3-5; Appendices A-F.

### SplatSim

- Source: [arXiv:2409.10161v3](https://arxiv.org/abs/2409.10161v3), 8/8 PDF pages, complete.
- Method and evidence: renders physics-simulated robot and object motion with reconstructed Gaussian appearance to train transferable visuomotor policies.
  Real-robot experiments include pushing, rearrangement, and insertion.
- Decision: retain, grouped with Re3Sim.
- Avoid: confusing visual reconstruction with identification of physical dynamics; the system uses an existing physics simulator and aligned object models.
- Claim locations: Sections III-V.

### Re3Sim

- Source: [project PDF](https://re3sim.github.io/re3sim.pdf), arXiv:2502.08645v4, 8/8 PDF pages, complete.
- Method and evidence: combines reconstructed geometry, Gaussian background appearance, foreground rendering, and physical simulation to generate manipulation demonstrations.
  Experiments evaluate real-world policy transfer and simulation/real performance correlation.
- Decision: retain, grouped with SplatSim.
- Avoid: claiming physical system identification; Section III-B uses default parameters, and Section VI explicitly states this limitation.
- Claim locations: Sections III-IV and VI.

### RialTo

- Source: [RSS 2024 PDF](https://www.roboticsproceedings.org/rss20/p015.pdf), 23/23 PDF pages, complete.
- Method and evidence: uses scans and a human-facing articulation interface to construct digital twins, then combines demonstrations, simulation reinforcement learning, and policy distillation.
  Real-robot experiments test robustness to changed poses, distractors, and disturbances.
- Decision: retain as reconstruction used for policy refinement.
- Avoid: claiming autonomous dynamics identification; default mass/friction and relatively quasistatic tasks are explicit choices.
- Claim locations: Sections III-B through III-D, IV, VII; Appendix VIII.

### ASID

- Source: [arXiv:2404.12308v2](https://arxiv.org/abs/2404.12308v2), 20/20 PDF pages, complete; ICLR 2024.
- Method and evidence: trains exploration policies in randomized simulation using a Fisher-information objective, identifies simulator parameters from real interaction, then optimizes a task policy in the updated simulator.
  Experiments include real rod balancing and shuffleboard.
- Decision: add as a particularly close active-identification precedent.
- Avoid: claiming EMPIRIC introduces simulator-designed physical experiments, or reducing ASID's model family to continuous parameters only; it also considers a supplied binary articulation choice.
- Claim locations: Sections 3-5; Appendix A.

## Additional completed readings: 2026-09-23

### WorldTest / AutumnBench: Benchmarking World-Model Learning with Environment-Level Queries

- Source: [arXiv:2510.19788v4](https://arxiv.org/abs/2510.19788v4), 34/34 PDF pages, complete, including the formal protocols, prompts, tables, reference baselines, and full example interaction trace.
  SHA-256: `22293763c6e09d7822102dc048bb77133f974b5c9011d3decd6dea2ba6f5f79c`.
- Method and evidence: WorldTest separates reward-free exploration from evaluation in query-derived challenge environments and scores behavior without requiring a particular internal model representation.
  AutumnBench implements 43 grid-world environments with masked-frame prediction, change detection, and planning challenges.
  The task type is disclosed before exploration, but its specific test parameters are not.
- Relevance: task success alone and broad world-model knowledge are different evaluation targets.
  EMPIRIC tests the usefulness of learning missing mechanisms during goal-directed manipulation, not the full breadth of environment-level queries proposed here.
- Decision: cite briefly in the problem setting, contrasting environment-level queries with task success during continual learning.
  At the author's request, the separate evaluation-related-work paragraph was removed.
  Do not describe the benchmark as a competing model-learning algorithm or suggest EMPIRIC has been evaluated on it.
- Evidence limits: the main reasoning-model comparison uses one completion per problem, while the human reference aggregates the 80th percentile across attempts.
  Reset patterns and action perplexity are behavioral associations, not controlled evidence that a particular exploration rule causes the performance gap.
  The simulator reference uses finite sampling and bounded search; do not repeat a blanket claim that it is a performance upper bound.
  Cross-model price comparisons are not controlled compute-scaling interventions.
- Claim locations: Sections 4.1-4.2 and 5.1; Appendices C, E, and F.
  The manuscript citation concerns the evaluation protocol, not the stronger causal interpretations of the behavioral analyses.
- Metadata: retain the initial preprint year, 2025, with the current title; the PDF spells the second author's name Thanh Dat Nguyen.

### KinDER: A Physical Reasoning Benchmark for Robot Learning and Planning

- Source: [arXiv:2604.25788v2](https://arxiv.org/abs/2604.25788v2), 21/21 PDF pages, complete, including baseline settings, environment tables, prompts, and the noise-wrapper evaluation.
  SHA-256: `39d1283a971b7b903edb7885cabdc9dea9c412bd3cfcc8431f0c39154ba44900`.
- Method and evidence: 25 procedurally generated environments span kinematic/dynamic and 2D/3D settings, targeting spatial relations, nonprehensile manipulation, tool use, geometric constraints, and dynamic constraints.
  The library supplies parameterized skills, predicates, teleoperation, and demonstrations.
  Its main comparison evaluates 13 baselines on eight representative environments, not all 25.
  Baselines differ in their supplied information and learning resources, including ground-truth transitions for MPC and demonstrations for learned models and policies.
- Relevance: a direct precedent for controlled, embodied physical-reasoning evaluation with continuous constraints.
  EMPIRIC focuses on learning mechanisms omitted from its supplied simulator under noisy observations during a continual task sequence.
- Decision: cite alongside the simulated-domain introduction as context for embodied physical reasoning, without comparing success rates across the two studies.
- Avoid: describing KinDER as purely kinematic, without dynamics learning, or incapable of noisy observations.
  Although the main design sets aside partial observability, Appendix E and Table VIII explicitly test observation/action noise wrappers on two environments.
  Its mobile-manipulator Shelf3D demonstration is a real-to-sim-to-real planning example, not evidence of autonomous discovery of hidden mechanism programs.
- Claim locations: Sections II, IV-VI, and VIII; Appendices C-E.
- Metadata: cite as RSS 2026, as stated on the primary arXiv record.

### Coding Agents for Generalized Task and Motion Planning Problems / AgenticGenPlan

- Sources: [project page](https://agenticgentamp.github.io/) and its [linked manuscript](https://agenticgentamp.github.io/assets/paper.pdf?v=20260922), 9/9 PDF pages, complete, including tables, qualitative examples, limitations, and references.
  SHA-256: `9484f20060193914f83fde41496dc7823add88ed09e200671efff5c5ea574f14`.
- Method and evidence: a coding agent chooses simulator tests and synthesizes a stateful programmatic policy for an environment with fully observed, object-centric states.
  The main setting exposes a black-box reset/step/render client but no environment implementation, supplied skills, or TAMP abstractions; the source-access variant additionally exposes implementation helpers and arbitrary-state access during synthesis.
  Policies are frozen for evaluation on held-out instances from the same initial-state distribution, with no further coding-agent or LLM calls.
  Frozen code can still use observations, internal memory, search, retries, and feedback; it is not an open-loop action sequence.
  Section IV-B describes self-directed edge-case testing and fitting a six-parameter kinematic calibration model from observed positions of a grasped block.
- Relevance: especially close evidence that a general coding agent can devise probes, construct models, and calibrate them without a dedicated modeling harness.
  EMPIRIC's distinction is not the first use of self-directed robot experiments or internal models, but continual revision and inference for engine-level mechanism programs under noisy observations, with supplied manipulation skills.
- Decision: add to coding agents for control, explicitly contrasting frozen-policy evaluation with continued model revision during task execution.
  Do not imply that this method is an implemented EMPIRIC baseline or compare its rates to ours.
- Evidence limits: the linked PDF's Table III has newer backend/timing results than the adjacent prose, and the website includes additional experiments beyond the PDF's headline sweep.
  Cite the consistent synthesis protocol and calibration example, not the mismatched timing summaries or aggregate counts.
- Claim locations: Sections II-A through II-C, III-A, IV-A, IV-B (Building Internal Models), and VI.
- Metadata: the linked PDF is anonymized; the public project page supplies the seven authors used in BibTeX.
  Cite as a 2026 manuscript, without inventing an arXiv identifier or publication venue.

### From Pixels to Predicates / pix2pred

- Source: [arXiv:2501.00296v4](https://arxiv.org/abs/2501.00296v4), 37/37 PDF pages, complete, including references, prompts, experimental protocols, and all learned-model listings.
  SHA-256: `cf6b03f0562e7cae9a751e62724ecf9597b8ddf8a755bb46877f40aa915274dc`.
- Method and evidence: a VLM proposes and evaluates visual predicates from skill-annotated demonstrations.
  Predicate selection optimizes a planning objective while inducing symbolic skill operators; continuous skill-parameter samplers are learned where applicable.
  Deployment uses symbolic planning and supplied skills, without requiring a precise physical simulator.
  The evaluation tests compositional generalization in simulated tasks and transfer from human demonstrations to a Spot robot.
- Relevance and decision: add beside VisualPredicator in the planning-abstractions paragraph.
  Contrast demonstration-based skill abstractions with EMPIRIC's simulator-level mechanism learning through its own experiments.
  These are complementary representations and learning settings, not evidence that symbolic models cannot generalize.
- Avoid: describing its input as unannotated video, its predicates as all hand-specified, or its learning procedure as ignoring perceptual noise.
  It uses soft precondition intersection and pruning to accommodate erroneous labels.
  Full object observability is an explicit limitation, but some real-robot executions replan from feedback.
  The real-world comparison scores generated-plan correctness; do not relabel its table as end-to-end physical execution success.
  The real system also uses supplied skills, object descriptions, gripper predicates, and marked juicer regions.
- Claim locations: Sections 3-5 and 7; Appendices A.1-A.4, A.6-A.8.
- Metadata: cite the 2026 RA-L article, not the initial preprint year.
  The published title abbreviates "Vision-Language Models" to "VLMs".
  Volume 11(4), pages 4002-4009 are confirmed by the [coauthor's publication page](https://www.robo.guru/research.html); the [MIT group bibliography](https://lis.csail.mit.edu/publications/) also confirms the journal, title, and year.

## Candidates not fully read

Status here means not yet fully read in this audit; no new manuscript claim should rely on these entries yet.

### Executable and probabilistic world models

- [One Life to Learn](https://openreview.net/forum?id=UQ36IrVCw2), discovered through primary-source search.
- [Mind-Studio](https://arxiv.org/abs/2606.16070), discovered through primary-source search.

### Hybrid physics, inference, and exploration

- [Augmenting Differentiable Simulators with Neural Networks to Close the Sim2Real Gap](https://arxiv.org/abs/2007.06045), earlier NeuralSim workshop paper; check whether separate citation adds anything.
- Augmenting Physical Simulators with Stochastic Neural Networks, referenced by NeuralSim; obtain primary source.

### Reconstruction and policy-transfer context

- [Digital Cousins](https://proceedings.mlr.press/v270/dai25a.html), previously cited; PDF retrieved but not fully read in this pass.
- [PolaRiS](https://arxiv.org/abs/2512.16881), previously cited; PDF retrieved but not fully read in this pass.
- [SimFoundry](https://arxiv.org/abs/2606.28276), previously cited; PDF retrieved but not fully read in this pass.

These three entries are removed from the compact reconstruction discussion, not rejected on technical grounds.
Their bibliography records are preserved for future consideration.
The revised paragraph uses only the six reconstruction papers fully read above.

## Organization and manuscript decisions

Use five paragraphs: code world models; learning physical dynamics; planning abstractions; scene reconstruction; coding agents.
Each paragraph starts with one short framing sentence before the specific papers.
The first paragraph focuses on the model representation; the second combines physical priors, inference from observations, and informative data collection.
Condense the former hybrid-model and physical-inference paragraphs by grouping shared ideas, preserving their twelve citations, and using one concluding comparison with EMPIRIC.
Keep the distinction between posterior parameter draws and the current-state estimate used for rehearsal accurate wherever it is discussed.
WorldTest was initially cited in the problem setting and KinDER in the simulated-domain introduction, without a separate related-work paragraph on evaluation.
Those mentions are absent from the manuscript at the final commit check; this condensation does not restore them.
The closest overlaps come first, while the reconstruction paragraph is shorter than in the original draft.
Add PoE-World, neural-augmented simulation, SAIN, NeuralSim, BayesSim, ASID, and STRUG.
Keep the distinction at the level of EMPIRIC's combined representation and learning workflow rather than claiming novelty for programs, residuals, memory, or informative experiments individually.
The method description must distinguish parameter-sampled rehearsal from full belief-space planning.
The previously agreed related-work wording was: "Execution-time rehearsals sample physical parameters from their posterior, with every rollout initialized from the same estimated current state."
Parameters are not sampled from the current-state estimate.
The 2026-09-23 follow-up adds WorldTest/AutumnBench, KinDER, AgenticGenPlan, and pix2pred after complete reading.
Correct the introduction's claims that abstract models cannot learn delays, that program models always reconstruct everything from scratch, and that coding agents necessarily have no explicit models.
Do not make changes to the empirical claims, abstract, or results in this literature pass.

### One-page condensation: 2026-09-23

At the author's request, condense the section to less than one page using the existing ICLR font size and spacing.
Give the most space to code world models, physical dynamics learning, and coding agents that learn through interaction.
Keep planning abstractions and scene reconstruction as shorter complementary comparisons.
Retain all 36 related-work citations, but group closely related contributions rather than listing every method's implementation details.
Remove the extended bonding example, repeated descriptions of inherited mechanics and skill rehearsal, and the rehearsal-initialization sentence already covered by the method and limitations.
Preserve recognition that prior work already includes partial observability, missing-force learning, self-directed experimentation, and physics-engine integration.
Each paragraph retains a short framing sentence and a concise comparison with EMPIRIC.
The compiled section occupies approximately four-fifths of a page, flowing from page 9 onto page 10 without forced page breaks or typography changes.
The full main text still occupies ten pages; shortening this section alone does not meet the nine-page manuscript target.

## Validation

The revised related-work section contains 36 distinct citation keys, all covered by completed readings above.
WorldTest and KinDER are currently uncited bibliography entries, so the manuscript cites 36 papers from this reading audit in total.
Eleven bibliography entries were added across the initial pass and the 2026-09-23 follow-up; existing entries for deferred papers were retained but are no longer cited in this section.
The manuscript compiled successfully with BibTeX resolving all citations, and the rendered related-work pages were inspected.
Three pre-existing cross-references in the real-robot appendix remain unresolved: `fig:real-button`, `eq:real-depth-likelihood`, and `eq:real-fan-likelihood`.
These are outside this literature revision and were not guessed or replaced.
The paper revisions are committed as `b6bee03` (bibliography) and `c7e86e8` (manuscript wording and condensation).
