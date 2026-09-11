"""Driver-owned execution of agent requests.

Agents select actions, skills or controllers. This executor advances the
real environment through the protocol, preserving its accounting, noise,
recordings and terminal checks. Agent tools only translate requests and
format results; model learning and simulated rollouts are outside this
API.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AsyncContextManager, Awaitable, Callable, \
    List, Optional, Set, Tuple, Union

from predicators.run.episode import EpisodeOver, EpisodeState, StepOutcome
from predicators.structs import Action, GroundAtom, State, _Option

if TYPE_CHECKING:
    from predicators.run.continual import InvocationResult, \
        ProtocolObservation, ProtocolSession


@dataclass(frozen=True)
class PrimitiveAction:
    """Apply one validated action vector."""
    action: Action


@dataclass(frozen=True)
class RequestReset:
    """Request a reset, subject to the protocol's reset rules."""
    note: str = ""


@dataclass(frozen=True)
class ExecuteSkill:
    """Run one skill with the agent's expected outcome."""
    option: _Option
    expected: Set[GroundAtom] = field(default_factory=set)
    note: str = ""
    expected_absent: Set[GroundAtom] = field(default_factory=set)


@dataclass(frozen=True)
class ExecutePlan:
    """Run skills in order until a stop condition is reached."""
    skills: Tuple[ExecuteSkill, ...]
    stop_on_divergence: bool = True


ActionSelector = Callable[["ProtocolObservation"], Awaitable[Optional[Action]]]


@dataclass(frozen=True)
class ExecutePolicy:
    """Open an agent controller and ask it for each primitive action.

    The factory owns controller resources, such as a sandbox subprocess.
    The driver owns observations, stepping and stopping. None yields
    control back to the agent without applying an action.
    """
    open_controller: Callable[[], AsyncContextManager[ActionSelector]]
    max_steps: int


@dataclass(frozen=True)
class GiveUp:
    """End the run after the conversation has saved its final notes."""
    note: str


ExecutionRequest = Union[PrimitiveAction, RequestReset, ExecuteSkill,
                         ExecutePlan, ExecutePolicy]


@dataclass(frozen=True)
class SkillExecution:
    """One completed invocation and the observations used to report it."""
    before: State
    after: State
    result: InvocationResult


@dataclass
class ExecutionProgress:
    """Results accumulated even if execution raises or is cancelled.

    Attempts retains the tool-call counter's existing semantics. Steps
    and resets are the real protocol charges, including a step that hits
    a cap before returning. A fresh instance is used for each request.
    """
    attempts: int = 0
    steps: int = 0
    resets: int = 0
    step_outcome: Optional[StepOutcome] = None
    skills: List[SkillExecution] = field(default_factory=list)
    stop_reason: str = ""


@dataclass
class ExecutionObserver:
    """Adapter hooks at the original action and reporting boundaries.

    Reporting a skill before starting the next preserves the timing of
    agent predicate evaluation and prevents a reporting error from
    letting later actions execute. These callbacks run synchronously.
    """
    on_attempt: Optional[Callable[[], None]] = None
    before_skill: Optional[Callable[[State], None]] = None
    after_skill: Optional[Callable[[SkillExecution], None]] = None

    def attempted(self, progress: ExecutionProgress) -> None:
        """Record an attempt before handing the action to the protocol."""
        progress.attempts += 1
        if self.on_attempt is not None:
            self.on_attempt()


class InteractionBusy(Exception):
    """An execution request would interleave with a running controller."""


class InteractionExecutor:
    """The shared real-environment executor for MB and MF.

    Awaiting execute hands execution to the driver until its request
    ends. Skills and policies may take many physics steps without an LLM
    turn. The executor is run-owned, so separate tool adapters share its
    busy gate.
    """

    def __init__(self, session: ProtocolSession) -> None:
        self._session = session
        self._busy = False

    @property
    def busy(self) -> bool:
        """Whether another request currently owns environment execution."""
        return self._busy

    def finish(self, request: GiveUp) -> None:
        """Apply the deferred give-up request at the round boundary."""
        if self._busy:
            raise InteractionBusy("A policy is running")
        self._session.end_run(request.note)

    async def execute(self,
                      request: ExecutionRequest,
                      progress: ExecutionProgress,
                      observer: Optional[ExecutionObserver] = None) -> None:
        """Execute one request, leaving partial results on errors.

        Protocol exceptions propagate unchanged for the caller to
        report. The finally block also runs on cancellation and
        preemption.
        """
        if self._busy:
            raise InteractionBusy("A policy is running")
        card = self._session.level_card()
        steps, resets = card.steps, card.resets
        observer = observer or ExecutionObserver()
        self._busy = True
        try:
            if isinstance(request, PrimitiveAction):
                observer.attempted(progress)
                progress.step_outcome = self._session.step(request.action)
            elif isinstance(request, RequestReset):
                observer.attempted(progress)
                self._session.reset(request.note)
            elif isinstance(request, ExecuteSkill):
                self._invoke(request, progress, observer)
            elif isinstance(request, ExecutePlan):
                for skill in request.skills:
                    self._invoke(skill, progress, observer)
                    result = progress.skills[-1].result
                    if result.status != "succeeded":
                        progress.stop_reason = result.status
                        break
                    if request.stop_on_divergence and result.diverged:
                        progress.stop_reason = "divergence"
                        break
                    if result.outcome.episode_state is not \
                            EpisodeState.NOT_FINISHED:
                        progress.stop_reason = "terminal"
                        break
            elif isinstance(request, ExecutePolicy):
                await self._run_policy(request, progress, observer)
            else:
                raise TypeError(f"Unknown execution request: {type(request)}")
        finally:
            progress.steps = card.steps - steps
            progress.resets = card.resets - resets
            self._busy = False

    def _invoke(self, request: ExecuteSkill, progress: ExecutionProgress,
                observer: ExecutionObserver) -> None:
        before = self._session.observe().frame
        if observer.before_skill is not None:
            observer.before_skill(before)
        observer.attempted(progress)
        result = self._session.invoke(request.option, request.expected,
                                      request.note, request.expected_absent)
        after = self._session.observe().frame
        execution = SkillExecution(before, after, result)
        progress.skills.append(execution)
        if observer.after_skill is not None:
            observer.after_skill(execution)

    async def _run_policy(self, request: ExecutePolicy,
                          progress: ExecutionProgress,
                          observer: ExecutionObserver) -> None:
        limit = request.max_steps
        if type(limit) is not int or limit <= 0:  # pylint: disable=unidiomatic-typecheck
            raise ValueError("max_steps must be a positive integer")
        obs = self._session.observe()
        if obs.state is not EpisodeState.NOT_FINISHED:
            raise EpisodeOver(f"episode is {obs.state.value}")
        async with request.open_controller() as get_action:
            for _ in range(limit):
                observation = self._session.observe()
                action = await get_action(observation)
                if action is None:
                    progress.stop_reason = "policy_stopped"
                    return
                observer.attempted(progress)
                outcome = self._session.step(action)
                progress.step_outcome = outcome
                if outcome.state is not EpisodeState.NOT_FINISHED:
                    progress.stop_reason = "terminal"
                    return
        progress.stop_reason = "max_steps"
