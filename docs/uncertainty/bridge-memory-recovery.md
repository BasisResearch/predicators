# Bridge inference: checkpoint recovery after memory exhaustion

September 14, 2026.
Both full supported-initialization fits in array `22706932` terminated with scheduler state `OUT_OF_MEMORY` after 2:51:51.
Each allocation reached approximately 64 GiB of resident memory.
These are infrastructure interruptions, not completed inference results or agent failures.

Both last complete checkpoints are at temperature stage 26 of 64.
Seed 810 has 4,535 logical evaluations and seed 811 has 4,348; both retain 32 original particle lineages at that stage.
The original interrupted reports, ledgers and checkpoints remain unchanged.
The recovery bundle, `logs/uncertainty_bridge_memory_resume_20260914`, contains checksummed copies of the complete checkpoint prefixes and explicit links to the interrupted artifacts.
The 38 and 69 evaluations logged after the respective complete checkpoints are retained in the original ledgers and excluded from resumed sampler history.
Their resource cost remains part of the interrupted allocations.

Validation job `22713841` reproduces both entire numerical traces through their saved checkpoints, including sampler random state.
Four fresh native target checks also reproduce their saved joint values and scores exactly, using 2,400 native actions.
This validates the recovery point rather than claiming that the incomplete posterior is adequate.

The first resume array, `22713869`, failed before sampling because its frozen loader compared JSON lists with tuples from dataclass serialization.
The checkpoint, model identity and configuration values were unchanged.
An isolated launcher canonicalizes report-only dataclass serialization before calling the frozen fitter; the probability model and sampler objects remain unchanged.
Both end-to-end loader preflights in array `22714028` pass, reaching the sampler with the exact saved identity, configuration and checkpoint.
The launcher and preflight outputs have separate source hashes in the recovery bundle.
The failed first resume allocations and their cancelled dependent pipelines remain recorded as setup failures.

Array `22714037` resumes these checkpoints with the unchanged frozen fitter, probability model, sampler configuration, 16 ordered workers, and original evaluation budget.
Each new allocation requests 128 GiB instead of 64 GiB on the original compute node.
The underlying memory-growth cause has not been isolated; the larger allocation provides headroom for the remaining stages.
Readers `22714038` and `22714039` follow the fits and will independently replay the combined pre-interruption and resumed ledgers.

The replacement forecast pipeline is frozen in `logs/uncertainty_bridge_resumed_forecasts_v2_20260914`.
Its forecast and summary calculations are unchanged; source paths point to the resumed fits and new output directory.
Forecasts `22714051` and `22714054` depend on their fit readers, followed by readers `22714052` and `22714055` and comparison `22714056`.
Every full forecast reruns the existing summary guards, and its reader checks complete histories and fresh native repetitions.
The old downstream jobs cannot proceed after their failed dependencies and are terminal.

The resumed fit report's elapsed-time field covers only its new allocation.
Total inference cost must also include the interrupted allocation, initialization and recovery validation; do not quote the new elapsed-time field as total cost.
The numerical and predictive gates remain open.
