# Explicit transition-discrepancy reference

September 12, 2026.
This is an offline model extension under the [simplification proposal](simplification-proposal.md), separate from the deterministic sensor-only reference and the production agent.
The deterministic contact diagnostics found reproducible predictions, occasional exact first-output matches, and unresolved later exact outputs.
They do not prove that every deterministic explanation is impossible.
They do justify evaluating a specified discrepancy model rather than treating a root solver's convergence flag as a conditional posterior.

## Velocity law

The new `VelocityDiscrepancy` class defines a post-transition velocity distribution around the simulator's predicted linear velocity `mu`:

$$
v \sim \rho\,\delta_0 + (1-\rho)\,\mathcal{N}(\mu,\sigma_v^2 I_3).
$$

The rest mass `rho` and correction scale `sigma_v` are declared model hyperparameters.
The correction scale has velocity units per declared transition; it is neither sensor variance nor a numerical solver tolerance.
This model allows an abrupt rest event and otherwise applies an isotropic velocity correction.
Whether that is an adequate representation of missing forces or contact errors must be tested; the mathematical construction does not establish physical adequacy.
Changing transition frequency requires a new interpretation of its hyperparameters.

For exactly observed speed `r=0`, the rest component is selected and contributes probability mass `rho`.
Without a rest atom, conditioning at zero remains unsupported by this construction.
For positive speed, the moving component has the noncentral chi radial density in three dimensions.
Writing `m = norm(mu)`, its density for `m>0` is:

$$
f_R(r) = \frac{r}{\sqrt{2\pi}\sigma_v m}
\exp\left[-\frac{(r-m)^2}{2\sigma_v^2}\right]
\left(1-\exp[-2rm/\sigma_v^2]\right).
$$

The retained transition factor is `(1-rho) * f_R(r)`.
At `m=0`, this reduces to the Maxwell density already used by the central initial-velocity prior.
These factors use the mixed speed measure consisting of an atom at zero and Lebesgue measure on positive speeds.
They are not Cartesian velocity densities.

Given positive speed, velocity direction follows a von Mises-Fisher distribution centered on `mu` with concentration `r*m/sigma_v**2`.
Two unit-uniform coordinates sample that normalized conditional direction law.
The resulting direction remains part of the candidate history because it changes subsequent positions and contacts.
Replacing it with the simulator's original direction, or with its conditional mean, would change the model.
The implementation reports floating-point reconstruction error separately from the likelihood.
It does not accept an arbitrary band around an exact speed observation.

## Recorded Balloons diagnostic

The frozen diagnostic bundle is `logs/uncertainty_balloons_transition_v2_20260912`.
Its program, noisy public observations, and initial-scene law come from the earlier [original non-hatch Balloons reference](balloons-initial-scene.md).
No evaluator-only state supplies an initial candidate.
Two complete candidate roots are drawn using the same first-observation conditional law and geometric rejection policy.

The diagnostic declares corrections after each native environment action and before reading the corrected observation:

1. Each of the nine observed controlled joint positions receives an independent Gaussian correction with standard deviation 0.001 in that joint's coordinate units.
2. Box linear velocity receives the rest/Gaussian mixture correction, with rest mass 0.1 and three separately evaluated scales: 0.0001, 0.001, and 0.01 m/s.
3. Native angular velocities and joint velocities are retained.
   Other body states, discrete events, attachments and program memory continue through the candidate simulation.

These are explicit alternative transition models, not parameter fits or selected deployment settings.
Joint corrections and box-velocity corrections are conditionally independent given the native prediction in this declared model.
Joint positions can be conditioned analytically on their exact readings, retaining the Gaussian transition density at those readings.
Exact box speed uses the radial construction above, retaining uncertainty about direction.
Every other exact output remains an explicit consistency check.
The first observation is already absorbed by the conditional root proposal and is not scored again as independent evidence.

Each root/scale combination has two direction draws and a fresh-world repeat.
The first 32 actions use the conditional transitions; the following 16 actions draw unconditional transitions without reading future observations.
Future observations are compared only after those states have been simulated.
This separates matching an observed speed by construction from predicting an unobserved future speed.
The report retains correction magnitudes, transition density factors, remaining exact-output disagreements, noisy prediction scores, and repeatability.

This small collection of paths is a support and reconstruction diagnostic, not an adequately weighted approximation to a recording posterior.
Its transition factors alone are not model evidence.
Conditional root normalizers, all observation factors, path importance weights and numerical adequacy must be handled before comparing posterior distributions.

## Acceptance limits

The correction law preserves nonnegative speed, but it is not a contact-aware force model.
Joint position corrections can introduce geometry intersections or violate relationships with native velocity and attachment states.
Such limitations must remain visible; matching joint readings does not certify physical support.
No discrete event is corrected to make an incomplete program appear accurate.
In particular, this extension cannot explain the frozen Bridge model's missing glue transitions.

If this representation yields feasible, reproducible paths, the next checks are parameter/state inference under the complete declared transition model, held-out predictive scoring, event predictions and repeatability as the compute budget increases.
Poor future predictions or inappropriate correction sizes are reasons to revise or reject this model, even if its conditional arithmetic is correct.
Its selection must precede matched live comparisons and must not change the incumbent's production behavior before the proposal's remaining gates pass.

## Validation status

Compute job `22632883` passed seventeen functional tests, two-file mypy and lint, and pinned formatting checks.
The numerical tests integrate the Gaussian density over spherical shells independently, check total radial mass and second moments, and verify conditional directional quantiles.
They also cover the central-prior limit, the rest atom, rotation, concentrated directions, malformed inputs and explicit numerical failure.
This verifies the component's conditional arithmetic, not a physical posterior or a live agent result.

The first component check attempt passed functional tests and mypy but found a long line and a test closure lint issue; both were corrected in the frozen v2 checks.
The first physical diagnostic attempt failed while serializing a NumPy boolean to JSON.
That report-writing error was corrected in v2 without changing the transition model, and job `22632884` is the replacement diagnostic.
These are setup failures, not failed agent seeds or evidence against a stochastic model.
