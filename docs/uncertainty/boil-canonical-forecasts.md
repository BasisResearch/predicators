# Boil forecasts after canonical prefix inference

September 13, 2026.
This continues the [canonical Boil inference experiment](boil-joint-inference.md) in Stage B of the [simplification plan](simplification-proposal.md).
The experiment remains offline, and the production agent keeps its existing estimator.

## Prediction boundary

The two canonical fits use the first 132 recorded actions and 133 observations of the fixed development episode.
The following 132 actions are reserved for prediction assessment.
Each continuation reconstructs the sampled initial scene from the validated fixed template and replays all 264 actions in one fresh world.
It preserves native contact history and the literal simulator program's memory across the fitting boundary.
It does not restore an arbitrary engine snapshot at action 132.

During the fitting prefix, the declared conditional joint-transition model uses the observed joint positions and retains their density factors.
During forecast generation, it draws fresh joint innovations without consulting future observations.
The output-error process conditions on the fitting prefix and propagates forward; future sensor readings are sampled independently under that process.
The separate future-likelihood calculation conditions on the recorded future joints and retains their complete transition density.
Those observation-conditioned paths are used only to evaluate density, never to calculate forecast means or event probabilities.

Every generated continuation checks exact agreement with the original fitting worker on its entire physical prefix, literal-rule prefix and target factors.
No future likelihood reweights the fitted particles.
Every positive-weight particle contributes to the forecast, and particles assigning zero density to the recorded future remain in the mixture with their original weights.
An entirely unsupported empirical future mixture is reported explicitly, without renormalizing a successful subset.

Clean public recordings supply assessment targets only.
The generator receives the fitting observations and future actions, but no future noisy or clean readings.
The fixed scene candidates used to validate the machinery were selected during earlier development audits, so their fixture results are mechanical checks, not an independent predictive-performance result.

## Verified native fixture

Fixture job `22690359` completed in 1:57 on four compute CPUs.
Two supported scene candidates each have two generated futures and one recorded-future density calculation, with additional repeats and density round trips.
The fixture records 4,884 native action steps, including fitting-reference replays.
Generated futures repeat exactly and reproduce their trajectory and complete density when their generated observations are supplied to the density evaluator.
Changing a future scalar reading changes its likelihood without changing the native trajectory or carried model memory.

Independent verifier `22690509` completed in 2:33.
It checks all twelve saved histories and 28,512 scalar joint-transition factors, recomputes literal model memory, independently calculates output moments, and compares event flags with the native environment predicates.
It rejects deliberate corruptions of the generation/density role, fitting observations, corrected joints, transition factors, model memory, forecast moments, goal events and initial likelihood factor.
An earlier verifier attempt, `22690433`, failed while serializing NumPy Boolean predicate outputs; explicit conversion to ordinary Booleans fixes the checker without changing any simulation or factor.

These checks validate the forecast machinery on the selected candidates.
They do not establish posterior exploration, predictive adequacy or an agent solve rate.

## Completed-population adapter

The adapter checks the source program, data, prior, model, runtime and sampler configuration against the frozen fit identity.
It restores a completed checksummed sampler checkpoint through the sampler API with target evaluation prohibited.
The recovered result must exactly match the saved population, including weights and joint coordinates.
Incomplete populations cannot enter this path.

For each positive-weight particle, the planned assessment generates two independent banks of four futures and separately evaluates the recorded future density.
Every history is saved, and every saved record is checked against the original fitting prefix, complete transition factors and model memory.
The first generated and recorded-density paths are also repeated in fresh worlds.
An independent reader reconstructs the weighted summary from the saved histories and original checkpoint.

Reported metrics retain their physical units:

- Output-mean error against noisy future observations for jug position, water volume, bubbling and spilled volume.
- Native prediction error against clean public future readings for those same six features.
- Probability curves and Brier scores for filled, boiled, no spill, burner off and the actual task goal.
- Complete future mixture log density, including unsupported particles.
- Differences between future banks and between independently fitted populations.
- Sampler diagnostics, native simulation steps and allocation time.

This first pair is exploratory and has no declared Boil numerical acceptance margins.
Agreement between its fits would not establish calibration, a matched improvement over the legacy estimator, or live-agent non-regression.
Those remain subsequent Stage B/C/E requirements.

The frozen scripts, inputs, reports and complete forecast histories are in `logs/uncertainty_boil_canonical_forecast_20260913/`.

## Adapter validation and queued forecasts

Adapter check `22690783` completed in 1:40 on a compute node.
Its generated and recorded-density continuations exactly reproduce the previously verified fixture, including fresh repeats and all saved fields.
It independently accumulates mixture means and variances, retains zero future support, and verifies exact recovery of a synthetic completed checkpoint without evaluating its target.
Nine negative controls reject missing or duplicated rows, changed weights, swapped roles, invalid bank indices, negative variance, unfinished checkpoints, altered checkpoint weights and changed inference identity.
The synthetic checkpoint is an API test, not a fitted physical posterior.
An earlier adapter check, `22690760`, failed because that synthetic fixture supplied an inconsistent prior identity; correcting the fixture allowed the unchanged identity guard to pass.

The source identities and completed fixture certificates are pinned in `posterior-plan.json` before submission.
Forecast/verifier jobs `22690858_0` and `22690863_1` wait for successful completion of fits `22690118_0` and `_1` respectively.
Each requests four CPUs, 20 GB and at most two hours in `mit_preemptable`.
The paired report `22690864` waits for both forecast/verifier jobs to succeed.
Both fits were still running at submission, so there is no completed-population forecast result yet.
