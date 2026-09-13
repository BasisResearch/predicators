"""Offline tempered batch sampling for small continuous reference problems.

The box reference and the explicit conditional-base extension share the
same tempered kernel. Conditional maps must supply their correct density
factors and full joint coordinates. This does not establish a feasible
physical-state prior for the five domains and is not used by the agent.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Callable, Iterator, List, Literal, Optional, Tuple, Union

import numpy as np

from predicators.code_sim_learning.inference_checkpoint import \
    SamplerCheckpoint
from predicators.code_sim_learning.inference_data import InferenceIdentity, \
    content_digest
from predicators.code_sim_learning.inference_evaluation import BatchedTarget


@dataclass(frozen=True)
class BoxPrior:
    """Fixed normalized independent uniforms in named coordinates.

    Names distinguish shared parameters from individual episode initial
    states. Known exact values should be conditioned and excluded from
    this vector. More general supports require a different prior
    implementation, not rejection followed by an unrecorded change in
    prior density.
    """
    names: Tuple[str, ...]
    bounds: Tuple[Tuple[float, float], ...]

    def __post_init__(self) -> None:
        names = tuple(self.names)
        bounds = tuple((float(lo), float(hi)) for lo, hi in self.bounds)
        if not names or len(names) != len(bounds) or len(
                set(names)) != len(names):
            raise ValueError(
                "Prior requires distinct names and matching bounds")
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("Prior names must be nonempty strings")
        if any(not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi
               or not math.isfinite(hi - lo) for lo, hi in bounds):
            raise ValueError("Prior bounds require finite positive widths")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "bounds", bounds)

    @property
    def digest(self) -> str:
        """Include support, parameter meanings and normalized prior family."""
        return content_digest(
            json.dumps(
                {
                    "schema": 1,
                    "family": "independent_uniform",
                    "prior": asdict(self)
                },
                sort_keys=True).encode("utf-8"))


@dataclass(frozen=True)
class SamplerConfig:
    """Budget, temperature schedule and optional proposal-coordinate blocks.

    Nonempty blocks partition the proposal coordinates and are selected
    uniformly for symmetric moves. Empty blocks preserve full-vector
    moves. An explicit increasing schedule may replace equally spaced
    temperatures, with the same stage count and final target. A fixed
    refresh probability mixes in independent uniform proposals within
    the selected block. Zero preserves the original random-walk
    schedule.
    """
    particles: int = 512
    temperatures: int = 32
    moves: int = 4
    proposal_scale: float = 0.05  # fraction of each prior width
    max_evaluations: int = 100000
    resample_ess_fraction: float = 0.5
    proposal_blocks: Tuple[Tuple[int, ...], ...] = ()
    temperature_schedule: Tuple[float, ...] = ()
    refresh_probability: float = 0.

    def __post_init__(self) -> None:
        for value in (self.particles, self.temperatures, self.moves,
                      self.max_evaluations):
            if not isinstance(value, int) or value <= 0:
                raise ValueError("Sampler counts must be positive integers")
        if not math.isfinite(self.proposal_scale) or self.proposal_scale <= 0:
            raise ValueError("Proposal scale must be finite and positive")
        if not 0 < self.resample_ess_fraction <= 1:
            raise ValueError("Resampling ESS fraction must lie in (0, 1]")
        if not math.isfinite(self.refresh_probability) or not \
                0 <= self.refresh_probability <= 1:
            raise ValueError("Refresh probability must lie in [0, 1]")
        blocks = tuple(tuple(block) for block in self.proposal_blocks)
        members = [index for block in blocks for index in block]
        if any(not block for block in blocks) or any(
                not isinstance(index, int) or isinstance(index, bool) or
                index < 0 for index in members) or \
                len(set(members)) != len(members):
            raise ValueError(
                "Proposal blocks require distinct nonnegative indices")
        object.__setattr__(self, "proposal_blocks", blocks)
        schedule = tuple(self.temperature_schedule)
        if schedule and (len(schedule) != self.temperatures
                         or schedule[-1] != 1. or any(
                             isinstance(beta, bool) or not math.isfinite(beta)
                             or beta <= previous
                             for previous, beta in zip((0., ) +
                                                       schedule, schedule))):
            raise ValueError(
                "Temperature schedule must increase from zero to one "
                "with the declared number of stages")
        object.__setattr__(self, "temperature_schedule", schedule)


@dataclass(frozen=True)
class ConditionedPrior:
    """A declared conditional base measure in explicit proposal coordinates.

    original_prior identifies the fixed generative prior, not a previous
    fit. conditioning identifies the exact observations, coordinate map,
    and density correction. The supplied map must cover the intended
    support and include the density ratio relative to the normalized
    uniform proposal. This declaration does not verify those properties.
    names describe the full joint output, including eliminated
    coordinates.
    """
    names: Tuple[str, ...]
    original_prior: str
    conditioning: str
    proposal: BoxPrior

    def __post_init__(self) -> None:
        names = tuple(self.names)
        if not names or len(set(names)) != len(names) or any(
                not isinstance(n, str) or not n for n in names):
            raise ValueError("Conditional output names must be distinct")
        for digest in (self.original_prior, self.conditioning):
            if len(digest) != 64 or any(c not in "0123456789abcdef"
                                        for c in digest):
                raise ValueError(
                    "Conditional identities must be SHA256 digests")
        object.__setattr__(self, "names", names)

    @property
    def digest(self) -> str:
        """Pin the original prior, conditioning map and proposal
        declaration."""
        return content_digest(
            json.dumps(
                {
                    "schema": 1,
                    "family": "conditional_base_measure",
                    "prior": asdict(self)
                },
                sort_keys=True).encode("utf-8"))


@dataclass(frozen=True)
class PriorPoint:
    """Full joint candidate and log conditional-base/proposal density ratio.

    The ratio includes exact-observation evidence and all coordinate or
    proposal corrections, but no remaining noisy-observation likelihood.
    Negative infinity rejects this point; unsupported maps must raise.
    """
    joint: Tuple[float, ...]
    log_weight: float


@dataclass(frozen=True)
class BatchPosterior:
    """Versioned candidate result; never a publication or adequacy certificate.

    Samples are joint vectors, including uncertain initial states.
    Failure results contain no posterior samples, even if a partial
    temperature was reached. ESS is measured before each resampling;
    retained weights must not disguise earlier weight collapse or
    missing modes.
    """
    identity: InferenceIdentity
    prior: Union[BoxPrior, ConditionedPrior]
    config: SamplerConfig
    seed: int
    status: Literal["complete", "budget_exhausted", "no_particle_support"]
    samples: Tuple[Tuple[float, ...], ...]
    weights: Tuple[float, ...]
    evaluations: int
    completed_temperature: float
    initial_finite: int
    effective_sample_sizes: Tuple[float, ...]
    accepted_moves: int
    attempted_moves: int
    surviving_ancestors: int
    resampling_count: int
    schema_version: Literal[1] = 1
    estimator: Literal[
        "offline_tempered_smc_box",
        "offline_tempered_smc_conditional"] = "offline_tempered_smc_box"

    def marginal_quantiles(
        self, name: str, probabilities: Tuple[float, ...] = (.05, .5, .95)
    ) -> Tuple[float, ...]:
        """Summarize the same joint samples, without a separate width fit.

        These are empirical posterior quantiles, not a calibration
        claim. Failed or incomplete numerical results have no credible
        intervals.
        """
        if self.status != "complete" or not self.samples:
            raise ValueError("No completed posterior samples")
        if not probabilities or any(not math.isfinite(p) or p < 0 or p > 1
                                    for p in probabilities):
            raise ValueError("Quantile probabilities must lie in [0, 1]")
        index = self.prior.names.index(name)
        values = np.asarray(self.samples)[:, index]
        weights = np.asarray(self.weights)
        keep = weights > 0
        values, weights = values[keep], weights[keep]
        order = np.argsort(values, kind="stable")
        values, weights = values[order], weights[order]
        cumulative = np.cumsum(weights / weights.sum())
        cumulative[-1] = 1.0
        # Inverse empirical CDF, including endpoints, with zero-mass samples
        # excluded. This convention does not interpolate across mode gaps.
        return tuple(
            float(values[np.searchsorted(cumulative, p)])
            for p in probabilities)


class _BudgetExceeded(Exception):
    """Internal control flow for an exhausted evaluation allowance."""


def sample_batch(prior: Union[BoxPrior, ConditionedPrior],
                 identity: InferenceIdentity,
                 log_likelihood: Union[Callable[[np.ndarray], float],
                                       BatchedTarget],
                 config: SamplerConfig,
                 seed: int,
                 *,
                 condition: Optional[Callable[[np.ndarray],
                                              PriorPoint]] = None,
                 checkpoint: Optional[Callable[[SamplerCheckpoint],
                                               None]] = None,
                 resume: Optional[SamplerCheckpoint] = None) -> BatchPosterior:
    """Sample a fixed-prior target from scratch, without carried fit weights.

    The callable must evaluate the immutable complete dataset under the
    identified simulator and sensor model. It receives an owned candidate;
    it must not use future observations, mutable hidden caches, or random
    transitions. Program/setup exceptions propagate rather than becoming
    zero likelihood. Negative infinity is legitimate zero support, whereas
    NaN and positive infinity are evaluation errors.

    Initialization draws from the actual prior. Each fixed temperature
    reweights by the likelihood increment, resamples only below the declared
    ESS threshold, then applies symmetric random-walk Metropolis moves on the
    uniform support. Without resampling, importance weights are retained
    through the target-invariant moves. Out-of-support proposals are rejected,
    never clipped. Final weighted particles target beta=1, but diagnostics and
    repeatability
    checks remain necessary; no ESS threshold certifies undiscovered modes.

    For a ConditionedPrior, condition maps proposal coordinates to the full
    joint candidate. Initial weights include its base/proposal correction
    before any likelihood tempering. Metropolis ratios retain that base
    factor at every temperature; only the remaining likelihood is tempered.
    Both callbacks receive owned arrays. The budget counts joint target
    evaluations, including rejected base points; simulator work may be less
    when a prior point is rejected before invoking the likelihood.

    An optional checkpoint callback receives complete initialization and
    temperature-stage boundaries. Resume restores that same run, including
    its RNG and cumulative evaluation budget. Signatures require unchanged
    data, model, prior, runtime, seed, NumPy version and sampler settings.
    Interrupted stages are repeated from their last saved boundary; the
    checkpoint is not a posterior or a numerical adequacy certificate.
    Callers must identify all callback dependencies in identity.runtime.

    Alternatively, a BatchedTarget owns both conditioning and likelihood and
    receives one ordered population per mutation sweep. Workers must evaluate
    whole candidates in isolated processes. This mode pre-draws an acceptance
    uniform for every proposal, including rejected ones, so worker scheduling
    cannot change the random stream. Its separately identified checkpoint
    kernel cannot resume scalar-mode checkpoints. Scalar calls preserve the
    original draw schedule. A nonzero refresh probability mixes independent
    uniform proposals within the selected block into either execution mode.
    Both proposal components are symmetric on the declared box, so the same
    conditional-base and likelihood Metropolis ratio applies. Refreshing is
    a numerical move, not resampling parameters from a new inference prior.
    Both modes target the same declared distribution;
    their random streams and resulting finite populations can differ.
    """
    if identity.prior != prior.digest:
        raise ValueError("Prior differs from immutable inference identity")
    conditional = isinstance(prior, ConditionedPrior)
    batched = isinstance(log_likelihood, BatchedTarget)
    if (batched and condition is not None) or (not batched and conditional !=
                                               (condition is not None)):
        raise ValueError(
            "A conditional prior requires exactly one coordinate map")
    proposal_prior = prior.proposal if isinstance(prior,
                                                  ConditionedPrior) else prior
    if config.proposal_blocks and set(
            i for block in config.proposal_blocks for i in block) != \
            set(range(len(proposal_prior.names))):
        raise ValueError(
            "Proposal blocks must partition all proposal coordinates")
    rng = np.random.default_rng(seed)
    lower, upper = np.asarray(proposal_prior.bounds).T
    count = config.particles
    particles = rng.uniform(lower,
                            upper,
                            size=(count, len(proposal_prior.names)))
    joints = np.zeros((count, len(prior.names)))
    base_weights = np.zeros(count)
    ancestors = np.arange(count)
    likelihoods = np.full(count, -np.inf)
    weights = np.full(count, 1.0 / count)
    resampling_count = 0
    evaluations = 0
    completed = 0.0
    initial_finite = 0
    ess_values: List[float] = []
    accepted = 0
    attempted = 0
    completed_stage = 0
    signature_config = asdict(config)
    if config.refresh_probability == 0.:
        # Keep default checkpoints compatible with the original schedule.
        del signature_config["refresh_probability"]
    signature = content_digest(
        json.dumps(
            {
                "kernel": "tempered_smc_ordered_batch_v1"
                if batched else "tempered_smc_stage_checkpoint_v1",
                "identity": identity.digest,
                "prior": prior.digest,
                "config": signature_config,
                "seed": seed,
                "numpy": np.__version__
            },
            sort_keys=True,
            allow_nan=False).encode("utf-8"))

    if resume is not None:
        if resume.signature != signature:
            raise ValueError("Checkpoint differs from requested inference run")
        state = resume.unpack()
        particles = np.asarray(state["particles"], dtype=float)
        joints = np.asarray(state["joints"], dtype=float)
        base_weights = np.asarray(state["base_weights"], dtype=float)
        likelihoods = np.asarray(state["likelihoods"], dtype=float)
        weights = np.asarray(state["weights"], dtype=float)
        ancestors = np.asarray(state["ancestors"], dtype=int)
        if particles.shape != (count, len(proposal_prior.names)) or \
                joints.shape != (count, len(prior.names)) or any(
                    a.shape != (count,) for a in
                    (base_weights, likelihoods, weights, ancestors)):
            raise ValueError("Invalid checkpoint population shape")
        if not np.all(np.isfinite(particles)) or \
                not np.all(np.isfinite(joints)) or \
                np.any(particles < lower) or np.any(particles > upper) or \
                np.any(np.isnan(base_weights)) or \
                np.any(base_weights == math.inf) or \
                np.any(np.isnan(likelihoods)) or \
                np.any(likelihoods == math.inf) or \
                not np.any(np.isfinite(likelihoods)) or \
                not np.all(np.isfinite(weights)) or np.any(weights < 0) or \
                not np.isclose(weights.sum(), 1., rtol=0., atol=1e-12) or \
                np.any(ancestors < 0) or np.any(ancestors >= count):
            raise ValueError("Invalid checkpoint population values")
        evaluations = state["evaluations"]
        initial_finite = state["initial_finite"]
        completed_stage = state["completed_stage"]
        accepted = state["accepted"]
        attempted = state["attempted"]
        resampling_count = state["resampling_count"]
        ess_values = state["ess_values"]
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in
               (evaluations, initial_finite, completed_stage, accepted,
                attempted, resampling_count)) or \
                not count <= evaluations <= config.max_evaluations or \
                not 0 < initial_finite <= count or \
                completed_stage > config.temperatures or \
                attempted != completed_stage * config.moves * count or \
                accepted > attempted or resampling_count > completed_stage or \
                len(ess_values) != completed_stage or any(
                    not math.isfinite(v) or v <= 0 or v > count + 1e-8
                    for v in ess_values):
            raise ValueError("Invalid checkpoint progress")
        completed = (config.temperature_schedule[completed_stage - 1]
                     if config.temperature_schedule else
                     completed_stage / config.temperatures) \
                     if completed_stage else 0.
        rng.bit_generator.state = state["rng_state"]

    def emit_checkpoint() -> None:
        if checkpoint is None:
            return
        state = {
            "particles": particles.tolist(),
            "joints": joints.tolist(),
            # Explicit strings represent legitimate negative-infinite log
            # densities without nonstandard JSON numeric extensions.
            "base_weights": [str(float(v)) for v in base_weights],
            "likelihoods": [str(float(v)) for v in likelihoods],
            "weights": weights.tolist(),
            "ancestors": ancestors.tolist(),
            "evaluations": evaluations,
            "initial_finite": initial_finite,
            "completed_stage": completed_stage,
            "accepted": accepted,
            "attempted": attempted,
            "resampling_count": resampling_count,
            "ess_values": ess_values,
            "rng_state": rng.bit_generator.state
        }
        checkpoint(
            SamplerCheckpoint(
                signature, json.dumps(state, sort_keys=True, allow_nan=False)))

    def evaluate(candidate: np.ndarray) -> Tuple[float, float, np.ndarray]:
        nonlocal evaluations
        if evaluations >= config.max_evaluations:
            raise _BudgetExceeded
        evaluations += 1
        point = (PriorPoint(tuple(candidate), 0.)
                 if condition is None else condition(candidate.copy()))
        joint = np.asarray(point.joint, dtype=float)
        base = float(point.log_weight)
        if joint.shape != (len(prior.names), ) or not np.all(
                np.isfinite(joint)):
            raise ValueError("Coordinate map returned invalid joint values")
        if math.isnan(base) or base == math.inf:
            raise ValueError("Base weight returned NaN or positive infinity")
        if base == -math.inf:
            return -math.inf, base, joint.copy()
        assert not isinstance(log_likelihood, BatchedTarget)
        value = float(log_likelihood(joint.copy()))
        if math.isnan(value) or value == math.inf:
            raise ValueError("Likelihood returned NaN or positive infinity")
        return value, base, joint.copy()

    def evaluate_many(
        candidates: List[np.ndarray]
    ) -> Iterator[Tuple[float, float, np.ndarray]]:
        nonlocal evaluations
        if not isinstance(log_likelihood, BatchedTarget):
            for candidate in candidates:
                yield evaluate(candidate)
            return
        allowed = min(len(candidates), config.max_evaluations - evaluations)
        proposals = tuple(
            tuple(float(v) for v in candidate)
            for candidate in candidates[:allowed])
        # Reserve the logical budget before any workers can start. A partial
        # stage never emits a checkpoint or usable posterior population.
        evaluations += allowed
        if proposals:
            rows = log_likelihood.evaluate(proposals, len(prior.names),
                                           conditional)
            for row in rows:
                yield row.log_likelihood, row.log_base, np.asarray(row.joint)
        if allowed < len(candidates):
            raise _BudgetExceeded

    def propose(index: int) -> np.ndarray:
        if config.proposal_blocks:
            # A state-independent mixture of symmetric block kernels.
            block = list(config.proposal_blocks[int(
                rng.integers(len(config.proposal_blocks)))])
            proposal = particles[index].copy()
            if config.refresh_probability > 0 and \
                    rng.random() < config.refresh_probability:
                proposal[block] = rng.uniform(lower[block], upper[block])
            else:
                proposal[block] += rng.normal(size=len(block)) * \
                    (upper[block] - lower[block]) * config.proposal_scale
            return proposal
        if config.refresh_probability > 0 and \
                rng.random() < config.refresh_probability:
            return rng.uniform(lower, upper)
        return particles[index] + rng.normal(
            size=len(proposal_prior.names)) * \
            (upper - lower) * config.proposal_scale

    def supported(proposal: np.ndarray) -> bool:
        return bool(
            np.all(np.isfinite(proposal)) and np.all(proposal >= lower)
            and np.all(proposal <= upper))

    def accept(index: int,
               proposal: np.ndarray,
               trial: Tuple[float, float, np.ndarray],
               beta: float,
               uniform: Optional[float] = None) -> None:
        nonlocal accepted
        proposed_likelihood, proposed_base, proposed_joint = trial
        if proposed_likelihood == -math.inf:
            return
        log_ratio = beta * (proposed_likelihood - likelihoods[index])
        if conditional:
            log_ratio += proposed_base - base_weights[index]
        # The original scalar path draws only after a finite evaluation.
        # Batch mode supplies its independently pre-drawn acceptance value.
        draw = rng.random() if uniform is None else uniform
        if math.log(1.0 - draw) < log_ratio:
            particles[index] = proposal
            likelihoods[index] = proposed_likelihood
            base_weights[index] = proposed_base
            joints[index] = proposed_joint
            accepted += 1

    def result(
        status: Literal["complete", "budget_exhausted", "no_particle_support"]
    ) -> BatchPosterior:
        complete = status == "complete"
        return BatchPosterior(
            identity=identity,
            prior=prior,
            config=config,
            seed=seed,
            status=status,
            samples=tuple(tuple(float(v) for v in row)
                          for row in joints) if complete else (),
            weights=tuple(float(w) for w in weights) if complete else (),
            evaluations=evaluations,
            completed_temperature=completed,
            initial_finite=initial_finite,
            effective_sample_sizes=tuple(ess_values),
            accepted_moves=accepted,
            attempted_moves=attempted,
            surviving_ancestors=len(set(ancestors.tolist())),
            resampling_count=resampling_count,
            estimator="offline_tempered_smc_conditional"
            if conditional else "offline_tempered_smc_box")

    try:
        if resume is None:
            for i, trial in enumerate(evaluate_many(list(particles))):
                likelihoods[i], base_weights[i], joints[i] = trial
                initial_finite += int(math.isfinite(likelihoods[i]))
            if not initial_finite:
                # Finite initialization may simply have missed valid support.
                return result("no_particle_support")
            if conditional:
                weights = np.exp(base_weights - np.max(base_weights))
                weights /= weights.sum()
            emit_checkpoint()
        for stage in range(completed_stage + 1, config.temperatures + 1):
            beta = config.temperature_schedule[stage - 1] if \
                config.temperature_schedule else stage / config.temperatures
            # Center before multiplication to avoid loss of stability from
            # large normalizing constants common to every candidate.
            if conditional and stage == 1:
                # Keep tiny base mass in log space until the first likelihood
                # update: a discrete observation may select that component.
                log_weights = base_weights - np.max(base_weights)
            else:
                log_weights = np.full(count, -np.inf)
                np.log(weights, out=log_weights, where=weights > 0)
            log_weights += (beta - completed) * (likelihoods -
                                                 np.max(likelihoods))
            weights = np.exp(log_weights - np.max(log_weights))
            weights /= weights.sum()
            ess = float(1.0 / np.dot(weights, weights))
            ess_values.append(ess)
            if ess < config.resample_ess_fraction * count:
                indices = rng.choice(count,
                                     size=count,
                                     replace=True,
                                     p=weights)
                particles = particles[indices].copy()
                likelihoods = likelihoods[indices].copy()
                base_weights = base_weights[indices].copy()
                joints = joints[indices].copy()
                ancestors = ancestors[indices]
                weights.fill(1.0 / count)
                resampling_count += 1
            for _ in range(config.moves):
                if batched:
                    pending = []
                    for i in range(count):
                        attempted += 1
                        proposal = propose(i)
                        uniform = float(rng.random())
                        if supported(proposal):
                            pending.append((i, proposal, uniform))
                    trials = evaluate_many([row[1] for row in pending])
                    for row, trial in zip(pending, trials):
                        accept(row[0], row[1], trial, beta, row[2])
                else:
                    for i in range(count):
                        attempted += 1
                        proposal = propose(i)
                        if supported(proposal):
                            accept(i, proposal, evaluate(proposal), beta)
            completed = beta
            completed_stage = stage
            emit_checkpoint()
        return result("complete")
    except _BudgetExceeded:
        return result("budget_exhausted")
