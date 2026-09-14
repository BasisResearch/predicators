# Boil heating: verified fixtures and reader recovery

September 14, 2026.
All four short joint-inference fits are complete and independently verified.
Replacement full fits are running, with remaining work queued for resources or prerequisite verification.
Both uniform-arm forecast fixtures have completed; their independent readers are running or queued.
The incumbent selected-point control is also complete and independently verified.
Stage B numerical and predictive acceptance remains open.

## Reader failure and correction

The original readers `22758855`, `22758858` and `22758860` completed their numerical-ledger replay, then failed when comparing a fresh worker result with its serialized representation.
The physical proposal coordinates were a tuple in the fresh result and a list in JSON.
Python's direct container equality rejected them despite identical coordinates.
The fourth pending invocation of that same reader, `22758862`, was cancelled after this failure was reproduced.
Its fitted source was allowed to finish.

The correction runs the unchanged frozen reader with fresh worker results passed through standard JSON serialization and decoding, matching the ledger's transport.
Before conversion, it checks the proposal and joint coordinates with exact equality after explicit container conversion.
It does not change numerical values, weaken tolerances or skip a target check.
The complete numerical replay and all 32 fresh final-target checks then pass for each fit.
Each verification output records both the original checker checksum and the transport adapter's path, checksum and plan identity.

The four fitting reports and checkpoints were reused without refitting.
All frozen scientific inputs remain unchanged.
The reproduction, transport adapter and verification provenance are in `logs/uncertainty_boil_heating_reader_recovery_20260914`.

| Arm | Numerical seed | Fitting evaluations | Reported fit seconds | Fitting native steps | Recovery reader | Fresh reader native steps |
|---|---:|---:|---:|---:|---:|---:|
| Uniform | 410 | 199 | 266.37 | 37,632 | 22762404 | 7,168 |
| Uniform | 411 | 207 | 272.52 | 39,648 | 22762405 | 7,168 |
| Guided | 410 | 199 | 271.99 | 38,304 | 22762406 | 7,168 |
| Guided | 411 | 207 | 230.55 | 39,200 | 22762409 | 7,168 |

These two-temperature fixtures test the implementation and source handoff.
They are deliberately insufficient for accepting the posterior approximation.

## Restored downstream pipeline

The reader failures caused the original dependent full fits, forecasts and collectors to be cancelled before they started.
Their terminal states and zero elapsed times were checked before replacement submission.
Replacement jobs use the same frozen fitting and forecasting scripts, seeds, target identities and numerical budgets.
All fitting readers now use the explicit JSON transport adapter.
Forecast readers retain their existing implementation.

Four verified earlier full fits, with the same 32 particles, 64 temperatures, eight moves and 20,000-evaluation cap, took approximately 5,404 to 5,660 seconds on the shorter prefix.
Scaling their observed rates to the full evaluation cap and the 224/132-action ratio gives at most 12,855 seconds, about 3.6 hours.
This is an empirical estimate, not a guaranteed runtime bound.
Replacement full fits request six hours, and their readers request two hours; the latter's verified earlier timing references were 54 and 66 seconds.
The numerical work is unchanged, and no active full fit was restarted.

| Arm | Seed | Full fit | Full reader | Forecast fixture | Fixture reader | Full forecast | Forecast reader |
|---|---:|---:|---:|---:|---:|---:|---:|
| Uniform | 410 | 22762490 | 22762491 | 22762492 | 22762493 | 22762494 | 22762495 |
| Uniform | 411 | 22762496 | 22762497 | 22762498 | 22762499 | 22762500 | 22762501 |
| Guided | 410 | 22762502 | 22762503 | 22762508 | 22762509 | 22762510 | 22762511 |
| Guided | 411 | 22762512 | 22762513 | 22762514 | 22762515 | 22762516 | 22762517 |

Posterior collector `22762518` follows all four forecast readers, and combined incumbent collector `22762519` follows it.
Their original metric calculations are unchanged.
The collection wrapper additionally validates transport provenance and appends the replacement job mapping and scheduler accounting, retaining the original failed and cancelled allocation records separately.
The sealed submission record and allocation evidence are in `logs/uncertainty_boil_heating_pipeline_recovery_20260914`.

## Verified incumbent result

Incumbent fit `22761363` completed in 19:32, and literal replay reader `22761364` completed in nineteen seconds.
The fit report records 128,208 native steps and 572 rollout calls, including its prediction work; the independent reader adds 264 native steps.
The complete prediction and hidden feature memory reproduce independently.
The fitted settings and parameter specifications remain identical to the earlier incumbent control.

The selected values include fill rate 0.01419785, heating onset 29.95697 and width 6.01631.
On the forty-action suffix, clean water-volume RMSE is 0.008744, bubbling RMSE is zero, and all five event curves match the clean labels, including the final goal.
Bubbling is already saturated at 1 throughout this suffix, so zero error here does not establish general thermal-parameter accuracy.
The clean positional RMSE values are approximately 0.041 mm, 6.502 mm and 8.005 mm for x, y and z.

This is the incumbent's selected-point prediction on one development recording, not a new agent solve-rate seed or a test of its full planning ensemble.
There is still no completed matched posterior performance comparison.
