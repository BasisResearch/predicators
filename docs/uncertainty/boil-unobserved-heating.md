# Boil: unobserved heating parameters and conditional marginalization

September 14, 2026.
This is an offline follow-up to the [supported Boil comparison](boil-supported-inference.md).
It separates numerical parameter concentration from the information available in the fitting prefix.
The production estimator is unchanged.

## What the fitting prefix observes

The 132-action fitting prefix never turns on the burner.
The burner state is an exact observation, verified against the frozen sensor contract.
The selected native continuations first turn it on at action 185.

The frozen program initializes heat to zero and increments it only when the burner is on, the jug is near enough, the jug is not held, and sufficient water is present.
Burner radius is used only in that heating condition.
Heating onset and width are used only in the clipped readout of accumulated heat.
Their prior supports are positive, and the physical simulator histories are independent of these feature-only parameters.

Consequently, for this program and any supported history of this all-burners-off prefix, the likelihood does not depend on these three parameters.
The original target uses independent uniform priors on burner radius `[0.04, 0.25]`, onset `[5, 80]`, and width `[1, 40]`.
Writing the remaining unknowns as `phi`, the target therefore factorizes as

```text
p(phi, radius, onset, width | prefix)
  = p(phi | prefix) p(radius) p(onset) p(width).
```

This statement depends on the exact observation contract, the frozen program, zero initial heat, and this particular prefix.
It does not apply after heating observations arrive or to an arbitrary simulator program.
The native transition and output-discrepancy factors on the remaining coordinates are unchanged.

The incumbent's own fitting report identifies burner radius, onset and width as unconstrained and retains defaults `0.12`, `29.5` and `7.0`.
Its good future prediction does not show that this prefix identified those values.
The posterior replacement should retain the declared prior uncertainty rather than reproduce a default point through unexplained concentration.

## Native reproduction and parameter interventions

Bundle `logs/uncertainty_boil_heating_diagnostic_20260914` selects the highest-weight retained particle from each completed numerical fit, 410 and 411, and four archived future random draws per selected particle.
It keeps each physical history and all non-heating parameters fixed.
Profiles are the selected parameters, incumbent onset/width, incumbent onset/width/radius, and two boundary profiles for the three heating parameters.
These are forty fixed-history interventions, not new fits or agent seeds.

Native diagnostic `22715829` completed in 3:55 with 792 simulator actions, exactly reproducing one complete source continuation per selected particle.
All eight archived histories reproduce the selected program's complete readouts and memory.
Every intervention preserves the entire fitting-prefix prediction, memory and likelihood exactly.
Independent reader `22715830` completed in five seconds, reconstructing 10,560 rule frames with literal arithmetic, checking 1,056 integrated future frames and rejecting altered parameters and predictions.
Its source checksum matches the completed diagnostic.

The following RMSE is averaged over the four selected future draws per numerical seed.
All four draws give the same bubbling RMSE within each listed profile.

| Numerical fitting seed | Selected onset/width | Incumbent onset/width | Uniform-prior onset/width integration |
|---|---:|---:|---:|
| 410 | 0.58826 | 0.01915 | 0.28497 |
| 411 | 0.49246 | 0.01915 | 0.28497 |

Changing only onset and width recovers the incumbent's bubbling error on these selected histories, without changing their physical motion or fitting score.
This identifies the retained heating parameters as the source of the selected histories' bubbling error.
It does not show that motion uncertainty is harmless for every particle or future path.
The two-parameter integration in this diagnostic holds the selected radius fixed and is not the complete posterior predictive calculation.

## Integrating all three heating parameters

Bundle `logs/uncertainty_boil_heating_marginalization_20260914` applies the factorization to all 512 generated histories from the two completed 32-particle fits.
Every original positive particle weight and both four-draw forecast banks are preserved.
The approximate posterior marginal over all other parameters and scene coordinates is retained.
The calculation replaces the accidental fitted heating-coordinate values with their declared independent priors.

For each fixed physical history, distances to the burner partition its radius prior into intervals with constant heating decisions.
A single radius interval determines the complete accumulated-heat trajectory, preserving the assumption of fixed parameters through time.
The clipped bubbling readout is integrated over onset analytically and over width by one-dimensional quadrature.
Interval probabilities then integrate burner radius.
Goal probabilities also retain the same history's filling, spilling and burner-off conditions.

Independent verification uses radius order statistics at each future step, rather than the producer's interval enumeration, and a different clipped-ramp integral.
It additionally checks every stored full-trajectory heat-count sequence, the exact burner observation contract, zero initial heat, source weights and aggregate summaries.
The calculation produces means and event probabilities; it does not claim a new joint future-density estimate or refit the remaining unknowns.

Integration job `22716011` completed in 57 seconds, and reader `22716012` completed in 1:24.
Both use compute nodes and require zero additional native simulator actions.
The reader checks all 512 histories, 67,584 future frames and 133 readout-integral values, and rejects a deliberately changed marginal mean.
The verified result checksum is `f918a4b451ec540a746feadc002b79ed6dd411ad4f4cfce525833ca3da87f9f5`.

| Method | Numerical seed | Bubbling RMSE of weighted mean | Final goal probability |
|---|---|---:|---:|
| Original supported fit | 410 | 0.33856 | 0.5637 |
| Heating priors integrated | 410 | 0.28731 | 0.7097 |
| Original supported fit | 411 | 0.31636 | 0.7691 |
| Heating priors integrated | 411 | 0.29648 | 0.6849 |
| Incumbent retained point | N/A | 0.01915 | 1.0000 |

The final-goal disagreement falls from 0.2054 to 0.0247 between the two approximate populations.
Both bubbling means improve, but remain worse than the incumbent's retained point on this recording.
The remaining difference cannot be diagnosed solely as simulator failure: this is also an extrapolation of heating behavior that the fitting prefix has not observed.
The retained posterior over scene and other parameters remains an approximation requiring numerical assessment.
No agent solve rate or Stage B acceptance is established here.

## Implications for the simplification

For a certified independent parameter block, retaining its analytical prior factor is simpler and more reliable than estimating that factor through resampling and limited random-walk moves.
A useful next numerical comparison can remove this block from the fitting coordinates and restore its prior when drawing complete parameter sets or integrating predictions.
That requires explicit factorization provenance and a result representation that does not mislabel fixed placeholder values as posterior samples.
The block must return to joint inference when new observations can constrain it.

Keep the original 132-action extrapolation comparison as a prior-retention and prediction case.
A separate prefix after heating experience can test whether the replacement learns the heating response from informative data.
Neither setting should silently replace the other, and program defaults must not be introduced as a new prior merely to improve the observed future score.
The full five-domain numerical, predictive and live-agent acceptance work remains open.
