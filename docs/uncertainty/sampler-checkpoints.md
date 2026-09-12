# Resumable offline inference

September 12, 2026.

The full-recording Domino and Fan fits take hours on preemptable compute nodes.
Previously, interruption lost the sampler population, even when a diagnostic report retained its best candidate.
A best candidate cannot reconstruct the weighted population or its random stream.

`sample_batch` now accepts an optional checkpoint callback and an optional saved continuation record.
It emits a record after successful initialization and after each complete temperature stage.
Each record preserves proposal coordinates, mapped joint states, base-density factors, likelihoods, weights, ancestry, random-generator state, evaluation counts, acceptance counters, resampling counts and stage diagnostics.
Resuming starts at the next stage without evaluating the saved population again.

```python
from pathlib import Path

from predicators.code_sim_learning.inference_checkpoint import SamplerCheckpoint
from predicators.code_sim_learning.inference_sampling import sample_batch

checkpoint_path = Path("/shared/run/sampler.json")
resume = (SamplerCheckpoint.load(checkpoint_path)
          if checkpoint_path.exists() else None)
result = sample_batch(
    prior, identity, likelihood, config, seed,
    condition=condition,
    checkpoint=lambda record: record.save(checkpoint_path),
    resume=resume,
)
```

Use a distinct path for each identified run and serialize writes to that path.
The parent directory must already exist.
Saving writes and flushes a temporary file in the same directory before replacing the previous file atomically.
Loading checks the file checksum and schema and uses JSON rather than executable deserialization.

## Compatibility and result semantics

The continuation signature includes the data, program, sensor model, original/conditional prior, declared runtime, seed, NumPy version, full sampler configuration and checkpoint kernel version.
A mismatch rejects the checkpoint before evaluating a candidate.
Callers remain responsible for identifying their callback code, simulator dependencies and execution controls in the runtime identity.
The checkpoint does not discover missing dependencies or make nondeterministic callbacks reproducible.

The evaluation limit remains cumulative across a resumed numerical run.
An interruption inside a stage loses that unfinished stage and repeats it from the last complete boundary.
The result's evaluation count describes the retained numerical run; job telemetry must separately account for discarded work and repeated initialization of the simulator process.
There is no checkpoint before the initial population has been evaluated successfully.
This first implementation does not continue from the middle of a Metropolis move or automatically extend a budget.

A continuation record is solver state, not an inference result.
It does not expose marginal quantiles, pass numerical assessment, publish parameters, or approve actions.
Budget-exhausted and unsupported results still contain no posterior samples.
Even a completed sampler result must pass the separate numerical assessment before parameter consumers can use it.

Running fits retain their frozen source and do not acquire checkpoint support retroactively.
Future launchers must opt into persistence explicitly; the API does not submit, resume or modify Slurm jobs.

## Validation

Compute job `22646779` completed successfully on `mit_preemptable`, with artifacts in `logs/uncertainty_sampler_checkpoint_20260912`.
All forty functional tests pass, including interrupted disk round trips for ordinary and conditional priors, exact preservation of the final population and diagnostics, compatibility rejection, corrupted files, failed atomic replacement and cumulative budget handling.
Three-file type checking, lint and the pinned formatting checks also pass.

All thirty-two paired comparisons with the frozen pre-change sampler match exactly, including samples, weights, diagnostics and evaluation counts.
These cover eight seeds, ordinary and conditional priors, and complete and budget-exhausted fits, with a multimodal target, zero-support candidates, blocked moves and a nonuniform temperature schedule.
The tests establish continuation and unchanged numerical behavior for these references; they do not establish adequate exploration of the physical-domain posterior targets.
