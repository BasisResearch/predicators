# Historical Bridge and Boil model controls

This is an additional Stage B development audit under the [simplification proposal](simplification-proposal.md).
It does not replace the frozen incomplete-model controls or change the acting agent.
No parameter fitting, posterior assessment or new agent run occurs here.

## Why these controls are needed

The saved Bridge and Boil programs from all three original noisy-sweep seeds declare no learnable parameters.
The audit covers thirteen versioned files across the six runs, including later learning cycles.
Their hidden-dynamics hooks are empty or diagnostic-only.
Those artifacts remain useful tests of explicit model inadequacy and parameter-free dispatch, but cannot measure an advantage of one parameter estimator over another.

Earlier saved noisy runs contain learned programs with explicit filling/heating and glue/bond mechanisms.
This audit freezes the seed-0 cycle-000 version-002 program from `boil-agent_continual_noise_p12_r07` and from `bridge-agent_continual_cross_noise_on`.
The source programs were created on September 8; the target observations are the already-frozen September 10 first-training recordings used by the existing controls.
Selection is based on the presence of the missing dynamics, before evaluating transfer errors.
Defaults, parameter bounds, model memory and command rules are retained from those historical artifacts.
The source program and its chosen bounds are learned artifacts, not evidence of an original prior predating their source data.
A subsequent posterior experiment must declare and freeze its prior separately.

The native replay uses the same frozen `b09217bb3` runtime and public noisy initial state as the original incomplete-model controls.
Historical rules run through their existing post-step interface: feature updates affect reported predictions, while emitted physics commands affect subsequent native steps.
This is a historical-model control, not a conversion of those programs to the current subclass contract or evidence of conversion parity.
The audit retains complete trajectories and hidden model memory and never injects later observations into the rollout.

Each domain has five fresh replays: the original no-op, historical defaults twice, and one parameter at its lower and upper declared bounds.
The perturbed parameter is `fill_rate` for Boil and `bond_dist` for Bridge.
The original no-op trajectory must match its archived control exactly; the two default histories, commands and model memories must match each other exactly.
Errors are reported for all post-action readings and for the suffix after action 64.
The suffix is a diagnostic partition; there is no fit on its preceding prefix in this audit.

## Boil outcome

Job `22685010_1` completes all five 264-action replays in 30 allocation seconds on one CPU, using 1,320 native actions.
Independent reader `22685333` verifies the archived control, complete repetitions, all scalar error calculations, parameter variants and exact-output contradictions.

| Reading | No-op all-action RMSE | Historical defaults all-action RMSE | Historical defaults suffix RMSE |
| --- | ---: | ---: | ---: |
| Water volume | 0.7600 | 0.08506 | 0.08976 |
| Bubbling level | 0.4260 | 0.06786 | 0.06695 |
| Spilled level | 0.07104 | 0.07104 | 0.06909 |

The declared sensor standard deviation for these three channels is 0.07.
The transferred filling/heating program supplies substantially better scalar predictions on this recording without refitting.
Changing fill rate to either declared bound changes 210 predicted frames; the upper bound produces large water-volume error rather than being silently clipped to an acceptable result.

However, the full sensor-only replay still contradicts eighteen exact observed channels.
Joint and related robot readout mismatches begin at action 9; the faucet and its switch disagree on their on/off state at two actions beginning at action 55.
Lower scalar error therefore does not establish a supported complete likelihood or justify dropping the exact observations from inference.
The program is a useful candidate for a separately declared physical-discrepancy comparison, not an approved posterior model.

## Bridge outcome

The first Bridge job, `22685010_0`, failed after 55 allocation seconds because its model memory contains a set and tuple-keyed dictionary that the report serializer could not encode.
This was a diagnostic reporting failure, not a failed agent seed or a physical replay contradiction.
The original script and failed output are preserved.
The replacement serializer preserves dictionaries, sets, tuples and lists with explicit type tags, including tuple keys; a JSON roundtrip verifies those distinctions.
The replacement Bridge job, `22685141_0`, completes five 1,186-action replays in 133 allocation seconds on one CPU, using 5,930 native actions.
The failed allocation remains additional cost; its last saved progress report is not a complete accounting of native work.

| Replay | Glue reading mismatches across all faces and actions | Steps emitting attachment commands |
| --- | ---: | ---: |
| No-op | 2,272 | 0 |
| Historical defaults | 3,695 | 347 |
| Lower bond-distance bound | 4,042 | 0 |
| Upper bond-distance bound | 3,695 | 347 |

The learned program makes six glue channels vary and can emit attachment commands, but its default glue predictions transfer poorly.
The lower bond-distance bound changes 347 complete predicted frames and removes the attachment commands; the upper bound leaves the default history unchanged.
These results expose parameter sensitivity and a flat region, without showing that fitting can recover the observed mechanics.
All exact-output contradictions remain visible; meaningful dynamics alone do not establish model adequacy.
Independent reader `22685371` verifies all five Bridge histories in 23 allocation seconds and retains contradictions in 26 exact observed channels for the default transferred model.

## Artifacts and next decision

The frozen bundle is `logs/uncertainty_bridge_boil_model_controls_20260913/`.
Its plan records the thirteen-file inventory, selected source programs, target data, runtime, variants and original-control hashes.
The reports preserve generated observations and memory; independent verification reloads the original public evidence and checks both noisy errors and exact-channel mismatches.
The existing no-op controls remain unchanged.
The bundle's `verified-inputs.json` pins the completed reports, verification artifacts and script versions, including the original reporting failure.

Use Boil's improved scalar dynamics to define a supported full observation/transition model before attempting a parameter-inference comparison.
For Bridge, first resolve whether model revision or a declared discrepancy model can explain the glue and bond transitions; do not assume a larger sampler will repair the transferred program.
Neither audit completes Stage B, resolves uncertain initial-state inference or permits retiring the incumbent estimator.
