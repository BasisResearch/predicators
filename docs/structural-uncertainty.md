# Structural uncertainty in the learned residual model

Status: design proposal, not implemented (2026-08-26, branch `bridge-learning`).
Origin: discussion following the weld-pin work, where the agent's learned model committed silently to one mechanism hypothesis among several the replay buffer could not distinguish.

## Problem

The agent-sim learn step synthesizes `simulator.py` with `RESIDUAL_RULES` plus `PARAM_SPECS`, and `ParamSpec` (`predicators/code_sim_learning/fit_space.py:25`) declares only continuous uncertainty: a bounded float with a linear or log fitting scale.
Uncertainty about model *structure* - which mechanism drives an effect, or whether a hypothesized effect exists at all - has no representation.
The synthesis step must guess, and the guess is invisible: nothing downstream knows an alternative structure was equally consistent with the data.

The motivating confound is real in bridge.
After coverage episodes that always place a block immediately after dabbing glue, two cure mechanisms fit identically, because elapsed time and contact time coincide in every trajectory:

- **clock**: cure progress ticks every step after the dab, contact or not;
- **contact**: cure progress ticks only while a block touches the wet streak.

If the agent guesses wrong, plans certify under a wrong model, fail at execution, and burn exploration cycles on undirected retries.
This works directly against the standing goal of learning in a couple of cycles: the fastest path to a correct model is often one *discriminating* experiment, and today nothing in the system knows which experiment that is.

## Design overview

Structural hypotheses become first-class, enumerated `ParamSpec`s.
The fit produces a posterior over structures alongside the continuous posteriors, and the parameter ensemble (`agent_sim_learning_approach.py:790`) carries a committed structure inside each member.

The invariant is **hard within a member, soft across the ensemble**:

- every simulated rollout runs one committed mechanism end to end, so each imagined future is a self-consistent world;
- uncertainty lives only in *which* member was drawn, expressed as posterior weights over structures.

This is deliberately not PoE-World-style blending (arXiv 2505.10819), where every expert stays active on every prediction and the weighted product can describe a physically incoherent in-between world ("70% cured").
Here a 0.5/0.5 posterior means half the members live in a clock world and half in a contact world, and their disagreement is itself the exploration signal.

## Agent-facing contract change

`ParamSpec` grows two fields:

```python
ParamSpec("cure_mechanism", kind="structural", choices=["clock", "contact"])
```

`kind` defaults to `"continuous"` (today's behavior, unchanged); `choices` is required iff `kind == "structural"` and must be a short list of hashable labels.
A structural spec has no `init_value`/`lo`/`hi`/`scale`; `__post_init__` enforces the split.

Rule code reads the committed choice like any other param:

```python
PARAM_SPECS = [
    ParamSpec("cure_mechanism", kind="structural", choices=["clock", "contact"]),
    ParamSpec("cure_threshold", init_value=25, lo=5, hi=80),
]
LATENT_INIT = {"cure_progress": 0.0}

def cure_rule(observation, latent, history, updates, params):
    ticking = dab_present(observation)
    if params["cure_mechanism"] == "contact":
        ticking = ticking and touching_streak(observation)
    if ticking:
        latent["cure_progress"] += 1.0
    cured = latent["cure_progress"] >= params["cure_threshold"]
    updates[dab]["cured"] = 1.0 if cured else 0.0
    return updates
```

The second structural form is **rule inclusion**: a candidate rule whose existence is itself uncertain declares `ParamSpec("drift_rule_active", kind="structural", choices=[0, 1])` and early-returns when 0.
This is the spike-and-slab analog of PoE-World's per-expert weights, with the hard-within-member semantics kept.

**Value encoding.**
`_fitted_params` stays `Dict[str, float]` (`agent_sim_learning_approach.py:780`): a structural entry stores the *choice index* as a float.
`_ParamsView.__getitem__` resolves structural names through the spec's `choices` so rule code sees the readable label, never the index.
This keeps every numeric consumer untouched - ensemble arithmetic, checkpoint save/load, the sweep's `{k}={v:.4g}` formatting - while giving agent-written rules the ergonomic comparison above.

## Fitting: enumerate and weight

Discrete dimensions do not go inside the MCMC walk; integer-coded categoricals mix terribly in emcee and plateau the sampler.
Instead a wrapper around `fit_rule_parameters` / `fit_rule_parameters_latent` (`predicators/code_sim_learning/fitting.py:670`, `:738`) enumerates:

1. Form the product of structural assignments over all structural specs.
   Hard-cap it (`agent_structural_max_assignments`, default 8); exceeding the cap is a synthesis error the learn message tells the agent to fix by splitting hypotheses across cycles.
2. For each assignment, run the existing continuous fit conditioned on that assignment (full budget per assignment, sequential).
   `FitResult.samples` stays continuous-only; the assignment is conditioning, not a sampled column.
3. Approximate each assignment's log evidence: Laplace when the LM bundle (`jacobian`/`noise_sigma`/`prior_sigma`) is present, BIC-style penalized best log-prob otherwise.
4. Normalize into structure weights.

The weights only gate pruning and stratification proportions; they never blend predictions and never soften the validation gate.
Decision robustness therefore comes from the hard gate, not from evidence-approximation quality, so crude approximations are acceptable.

**Pruning.**
A structure is retired when its weight stays below `agent_structural_prune_weight` (default 0.02, roughly a Bayes factor of 50) for `agent_structural_prune_patience` consecutive fits (default 2).
The next learn rewrite deletes the dead branch from `simulator.py`; once one structure remains, the structural spec disappears and the model is ordinary again.
The GO/NO-GO self-test in the learn discipline covers the rewrite as it covers any other model edit.

## Ensemble

`_select_param_ensemble` (`agent_sim_learning_approach.py:1612`) becomes stratified: the member budget (`agent_explorer_info_ensemble_size`, unchanged) is split across surviving structures proportionally to their weights, with at least one member per surviving structure.
Each member dict gains its structure's index-encoded entries; member 0 remains the MAP of the top-weight structure.
The per-structure members are drawn by the existing dispatch (posterior subsample, Laplace, uniform jitter) from that structure's own `FitResult`.
Total member count does not grow, so downstream costs are re-partitioned, not multiplied.

## Plan validation

No change to the sweep itself.
The rule-parameter margin gate (`predicators/agent_sdk/tools/testing.py:94-118`, inside `_parameter_margin_sweep` at `:42`) already re-runs the validation rollout under each ensemble member by swapping `_fitted_params` wholesale through `_rule_param_override_scope` (`agent_sim_learning_approach.py:1691`).
Structure-carrying members flow through that swap untouched.

Consequences:

- **Certified means structurally robust.**
  The gate stays all-members-must-pass, so a certified plan reaches the goal in every world the agent still believes in - under both clock and contact curing, not just the MAP mechanism.
- **A structural rejection is actionable in two distinct ways.**
  The existing rejection detail names the failing member's params, so the agent sees *which hypothesis* blocks certification.
  It can replan structurally robustly (dab and seat immediately, which cures under both mechanisms), or conclude no robust plan exists, which is exactly when the discriminating experiment has positive value.
- **Physics-margin interaction.**
  The physics sigma grid (`testing.py:75-93`) runs first and is orthogonal: physical overrides are env-construction params, not residual-rule state.
  The one guard needed is provider-side: if a physical dimension is only meaningful under one structure (a force constant belonging to a rule that structure excludes), the provider skips that dimension for members of the excluding structure.

The rejected alternative here is a weight-quorum gate ("pass if members covering >= 90% of posterior mass pass").
A low-weight straggler blocking certification is the prune rule's problem, not the gate's; softening the gate certifies plans that only work if the structural guess happens to be right, which is the exact failure mode this design removes.

## Exploration

Two mechanisms fall out with almost no new code:

- **Refinement-time info-seeking is structure-aware for free.**
  `score_atom_disagreement` (`agent_sim_learning_approach.py:1659`) already scores ensemble disagreement by swapping members through `_fitted_params`.
  Structure-split members disagree hardest exactly at states that discriminate mechanisms, so the agent_bilevel explorer's info-scorer starts steering toward discriminators the moment members carry structures.
- **Explicit discriminator entries in `open_questions.md`.**
  When more than one structure survives above the prune floor, the learn step writes a ranked, runnable experiment spec where the structures' predictions diverge, injected verbatim by the explorer's existing ledger path (`predicators/explorers/agent_bilevel_explorer.py`, `_read_open_questions`).
  Example entry:

  > Structure discriminator (`cure_mechanism`, weights 0.52/0.48): DabGlue(spanA); Wait 40 steps with nothing touching the streak; Place(spanB) on the streak; observe whether the weld appears immediately (clock) or only ~25 steps after contact (contact).

The closed loop: a plan the user cares about fails the gate structurally, the failure feeds the ledger, one discriminator episode collapses the posterior, the refit prunes the loser, and the previously rejected plan certifies on the refit.
Structures that never disagree on any plan that matters coexist indefinitely at zero cycle cost, because robust plans certify straight through them.

## Rejected alternatives

- **Discrete dims inside the MCMC walk**: poor emcee mixing, plateaued acceptance; enumeration is exact and the assignment count is capped small.
- **PoE-World weighted product of experts**: every prediction is a blend that corresponds to no single physical world; imagined rollouts lose internal consistency, and there is no committed world for the margin gate to certify against.
- **Weight-quorum validation gate**: see Plan validation; pruning handles stragglers without weakening certification semantics.
- **One `simulator.py` per structure**: shared continuous params would be re-declared and refit independently, the latent contract would fork, and pruning would be a file merge instead of an `if`-branch deletion.
  A single file with a declared switch keeps the diff small and the collapse mechanical.

## Implementation plan

1. `predicators/code_sim_learning/fit_space.py`: `ParamSpec.kind`/`choices` plus `__post_init__` validation; helpers to split specs into continuous/structural and map choice label to index and back.
2. `predicators/code_sim_learning/fitting.py`: the enumerate-and-weight wrapper (assignment product, conditioned fits, evidence approximation, normalized weights); a small result container bundling per-structure `FitResult`s with weights.
3. `predicators/approaches/agent_sim_learning_approach.py`: `_ParamsView` structural resolution; stratified `_select_param_ensemble`; structure weights and per-structure fit results in the checkpoint save/load dict; prune bookkeeping across fits.
4. Prompts (`agent_sim_learning_approach.py` synthesis sections, `predicators/agent_sdk/sketch_prompts.py`): a "declare structural alternatives you cannot yet distinguish" instruction with the cure example, and the discriminator entry type for `open_questions.md`.
5. `predicators/settings.py`: `agent_structural_uncertainty` (master switch, default False), `agent_structural_max_assignments` (8), `agent_structural_prune_weight` (0.02), `agent_structural_prune_patience` (2).
6. No changes to `_parameter_margin_sweep`, `_rule_param_override_scope`, or the explorer's ledger injection; assert that via tests instead.

## Testing plan

- Unit, fitting: on synthetic data generated by one structure, the wrapper recovers that structure with dominant weight, and the confounded-data case keeps both structures alive with split weights.
- Unit, params view: rule code sees choice labels; `_fitted_params`, checkpoints, and ensemble members hold float indices round-trip.
- Unit, ensemble: stratification respects weights, guarantees one member per surviving structure, and keeps the total budget.
- Unit, validation: a fixture plan that succeeds under one structure and fails under the other is rejected as PARAM-SENSITIVE with the structural member named in the detail; a robust fixture plan certifies with both structures alive.
- E2E, bridge: seed a run with confounded coverage episodes; verify the fit reports split weights, the discriminator lands in `open_questions.md`, the next cycle's episode collapses the posterior, and the pruned rewrite passes GO/NO-GO.
