# Public controller feedback boundary

The September 15 MF log review found hidden attachment labels and exact collision geometry in live skill failures.
The fix applies to the shared MB/MF interaction boundary across all domains.

`EpisodeRunner.run_option` converts controller exceptions into public failure feedback before returning an invocation outcome.
`ContinualRun.run_policy` uses the same conversion.
Tool responses and invocation records therefore contain the same public reason, while detailed diagnostics remain in host debug logs.
Unexpected continual tool errors also omit raw exception text.

The shared agent parent exports public skill API documentation rather than controller implementation files.
Skill listings continue to provide parameter names, descriptions, and bounds.
Sandbox setup rebuilds the harness-owned reference directory so retired exports do not remain in that directory after reopening.
The controllers still perform the same collision checks and attachment-aware planning internally.

The public API still reports whether an action succeeded, failed, or ended the episode, along with its charged steps.
Zero-step refusals therefore remain observable; the fix removes detailed collision metrology rather than suppressing all information carried by action outcomes.
Historical agent memory and Git history are not erased, so a clean evaluation requires fresh runs under the new contract.
Existing sweep outcomes remain historical results under their original runtime.

Regression tests inject the recorded Bridge, Fan, Domino, and Boil diagnostics through real continual skill tools and plans.
Additional tests cover policy failures, API references for all five environments, and removal of retired reference files.
Broader regression and type checks are submitted as compute jobs 22776858 and 22776859, currently awaiting the maintenance reservation.
