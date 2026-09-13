# Balloons prefix-only inference

September 13, 2026.
This is a separate Stage B experiment under the [uncertainty simplification plan](simplification-proposal.md).
It does not change the production agent or reinterpret the earlier [full-recording Balloons pilots](balloons-composed-inference.md).

## Fitting boundary

The earlier numerical seeds 300 and 301 fit all 235 actions in the first training episode.
Their guide center was selected using the complete recording.
The existing 64-action-prefix future-generation fixture uses that previously selected witness, so it validates mechanics rather than a prefix-fitted forecast.

The new experiment fits only the first 64 actions and 65 observations, reserving the following 171 actions for later prediction assessment.
It removes future observations immediately after the trusted recording loader validates the complete file, before constructing conditioning factors, trajectories or a guide.
The target's observation lookup contains only steps 0 through 64.
It does not load the full-recording guide center or fitted populations.

The simulator program remains the same frozen development artifact used by the earlier experiment.
That program itself was historically selected using training experience, so this is a conditional forecast comparison for a fixed development program, not an independent test of program synthesis or calibration on unseen tasks.

## Support and guide construction

The first guide uses original parameter-prior medians and a root centered on the analytically conditioned initial observation.
It retains the existing broad/local mixture, uniform motion-case selectors, inactive auxiliaries, conditional velocity-direction coordinates and density-corrected table-clearance proposal.
No future observation chooses the guide center.

Job `22691470` completed in 1:04 on four compute CPUs.
It tests 32 candidates and repeats each fresh native prefix.
Six candidates fail initial support checks; 26 reach complete native prefix evaluation.
Fifteen have finite complete prefix factors, while eleven retain exact-event/output contradictions.
All repeated evaluations agree exactly, including rejected cases.
The check executes 3,328 native actions.
Finite support alone does not establish accurate predictions or adequate numerical exploration.

The fitting guide is centered on candidate 5, selected by the largest corrected joint prefix target among these 32 candidates.
That selection uses only the fitting prefix, and the new guide retains its complete probability-density correction.
The original parameter and scene priors are unchanged.

## Factorization and independent verification

The fixed representation has 206 joint coordinates: ten parameters, 68 root coordinates and 128 conditional velocity-direction coordinates for the 64 actions.
It retains unused case auxiliaries; this is not a claim of 206 active continuous dimensions in every motion case.
The mixture selector adds one proposal coordinate.

The base factor contains initial analytic conditioning constants and the inverse guide density, together with geometry and exact-event support.
The tempered factor contains the complete remaining output likelihood and all speed and joint-transition density factors.
The earlier full-recording implementation placed those transition density factors in the base.
Moving finite continuous density factors into the annealed term changes the numerical path while preserving the final target at temperature one.
Exact joint and speed observations remain represented through the same conditional coordinates at every temperature; no equality is replaced by a tolerance band.
Every exact event still has an indicator likelihood.

Independent verifier `22691577` completed in 2:01.
It changes every future observation and action and verifies that the resulting fitting data are identical.
Access beyond the target's step-64 observation boundary fails.
It reconstructs all 32 audited candidates under the numerical-runtime overlay and matches every saved result exactly.
It then evaluates sixteen candidates from the new guide, seven of which have finite prefix factors.
The final target agrees exactly under the old and new factorization on these checked points, with maximum observed log-factor difference zero.
Serial and isolated parallel target evaluations also agree exactly.

The initial analytic constant is 3.522918693126006, retaining the location, conditioned-joint, clip-yaw and initial box-rest factors.
The common unknown geometric normalizer is not used to claim absolute model evidence.
These checks validate implementation and support on the selected cases, not posterior adequacy.

## Bounded numerical pilots

Array `22691680` launches numerical seeds 620 and 621 after the verified prefix checks.
Each uses 64 particles, 32 cubic-spaced temperatures, eight moves per temperature and a cap of 16,448 target evaluations.
The blocked proposal covers all 207 proposal coordinates, including the mixture selector and four blocks of conditional velocity directions.
Every target evaluation reconstructs its complete native prefix in a fresh candidate world.
Four isolated workers evaluate each population without sharing simulator instances.

Each pilot requests four CPUs, 20 GB and at most eight hours on `mit_preemptable`, pinned to the previously audited Balloons CPU type.
Complete-stage checkpoints preserve the original prior, numerical configuration, random state, particle weights, coordinates and cumulative evaluation count.
Returned native work and worker time are reported per allocation; interrupted work and previous allocations remain additional cost.
The launcher checks the independent certificate, source hashes and fitting boundary before sampling.

These are numerical inference replicas on one development recording, not agent seeds.
They remain separate from seeds 300/301, hatch experiments and the MB/MF performance sweep.
The 171-action causal forecast path, prediction stability, inference cost and comparison against the incumbent remain unfinished.
No posterior is approved for planning merely because a pilot completes.

Artifacts are in `logs/uncertainty_balloons_prefix64_20260913/`.
