# Uncertainty

[Documentation index](../README.md)

Read the implementation explanation first, then the simplification proposal.
The older design note records how the current features developed and includes plans that should be checked against the implementation.

| Document | Status and scope |
|---|---|
| [What the code does](explained.md) | September 11, 2026 source audit: parameter intervals, information-seeking score, trajectory preparation, carried estimates, fit evidence, execution belief, and `sim.fit()`. |
| [Simplification proposal](simplification-proposal.md) | Proposed fixed-prior parameter inference, uncertain initial states, filtering/smoothing, and staged evaluation; not implemented by this document. |
| [Original uncertainty design](design.md) | Observation-noise channel, feature design, and dated implementation and experiment notes. |

Results are indexed [separately](../uncertainty-results/INDEX.md).
Enabling all six feature flags does not establish that every feature affected every run, and comparing whole configurations does not isolate each feature's contribution.
