"""Robot-clearance measurement for the capture gate.

Certification's validation rollouts sample the belief's own execution
variability, but not the gap between the belief's realized poses and
the real executor's: every move phase terminates anywhere within its
pose tolerance, grasp heights vary within the descend tolerance, and
placed objects land with scatter. A plan whose robot passes within
that slop of a bystander certifies 8/8 by luck and loses the ninth
draw for real (2026-09-02 bridge seed3: two belief-certified explore
plans died on 6.5 mm and 9.9 mm real contacts against a block every
rollout had cleared).

:class:`RobotClearanceProbe` measures, along each validation rollout's
low-level trajectory, the minimum distance between the robot's links
and every body the executor would treat as a bystander - excluding the
held object and its welded attachments (the skill factory's own
planning-scene reconstruction decides that), the current option's
argument objects (a pick's fingers straddle their target by design),
the bodies the option's skill declares it contacts (a push strikes its
appliance's switch, a separate body: boil/fan 2026-09-02 refused every
plan at -3 to -14 mm against the switch being pushed) and the object
held when the option starts (a Place releases it and retreats past it
at a distance set by the grasp, not by the executor's slop: domino
2026-09-02 refused every plan at ~5 mm against the domino just placed).
The verdict compares that minimum against the executor's pose slop,
``sqrt(move_to_pose_tol)``, read from the plan's own skills, so the
bar is the executor's and not a domain constant.
"""
from __future__ import annotations

import logging
import math
from typing import Any, List, Optional, Sequence, Tuple

import pybullet as p

from predicators.structs import _Option

# Only bodies within this distance of a robot link are queried; anything
# farther cannot be the minimum that matters.
_QUERY_DISTANCE = 0.05


def phase_skill_of(options: Sequence[_Option]) -> Optional[Any]:
    """The first PhaseSkill (duck-typed) behind a grounded plan's options.

    A PhaseSkill's built option binds its policy as a bound method, so
    the skill - with its planning simulator and executor tolerances -
    is reachable through it. ``None`` when no option is skill-factory
    built or none carries a planning simulator.
    """
    for opt in options:
        skill = getattr(getattr(opt.parent, "policy", None), "__self__", None)
        config = getattr(skill, "_config", None)
        if (skill is not None and hasattr(skill, "_sim_collision_context")
                and getattr(config, "simulator", None) is not None):
            return skill
    return None


class RobotClearanceProbe:
    """Minimum robot-to-bystander clearance over validation rollouts.

    Feed each executed option's low-level trajectory to
    :meth:`observe`; :attr:`min_dist` and :attr:`where` then describe
    the closest approach seen so far. ``stride`` subsamples trajectory
    states (a contact event spans many control steps, so every third
    state loses nothing the verdict needs); the final state of every
    option is always probed - phase goals are where the executor's
    refusals were measured.
    """

    def __init__(self, skill: Any, stride: int = 3) -> None:
        self._skill = skill
        self._stride = max(1, stride)
        self.min_dist = math.inf
        self.where = ""
        self.num_probes = 0

    @property
    def threshold(self) -> float:
        """The executor's pose slop: its move-phase terminal tolerance."""
        return float(math.sqrt(self._skill._config.move_to_pose_tol))  # pylint: disable=protected-access

    def observe(self, label: str, option: _Option,
                states: Sequence[Any]) -> None:
        """Probe one option's trajectory states (best-effort diagnostic)."""
        exempt = {o.name for o in option.objects}
        if states:
            exempt |= self._designed_contacts(option, states[0])
        last = len(states) - 1
        for k, state in enumerate(states):
            if k % self._stride and k != last:
                continue
            try:
                dist, body = self._min_robot_clearance(state, exempt)
            except Exception as e:  # pylint: disable=broad-except
                logging.debug("Clearance probe skipped a state: %s", e)
                continue
            self.num_probes += 1
            if dist < self.min_dist:
                self.min_dist = dist
                self.where = (f"{label}, step {option.name}"
                              f"({', '.join(o.name for o in option.objects)})"
                              f" vs {body}")

    def _designed_contacts(self, option: _Option, state: Any) -> set:
        """Names of the bodies this option touches by design: the ones its
        skill declares (a push's switch, see ``PhaseSkill.contact_objects``)
        and the object it holds when it starts (a Place's fingers open around
        the block they release, and their retreat clears it by the grasp
        geometry, not by the executor's pose slop).

        Best-effort: a failing lookup exempts
        nothing.
        """
        names: set = set()
        skill = getattr(getattr(option.parent, "policy", None), "__self__",
                        None)
        contacts = getattr(skill, "contact_objects", None)
        if contacts is not None:
            try:
                names |= {o.name for o in contacts(state, option.objects)}
            except Exception as e:  # pylint: disable=broad-except
                logging.debug("Clearance probe: contact lookup failed: %s", e)
        try:
            held = self._held_object_name(state)
        except Exception as e:  # pylint: disable=broad-except
            logging.debug("Clearance probe: held lookup failed: %s", e)
            held = None
        if held is not None:
            names.add(held)
        return names

    def _held_object_name(self, state: Any) -> Optional[str]:
        """The name of the object held in ``state``, if any."""
        _, _, names, held, _ = self._skill._sim_collision_context(state)  # pylint: disable=protected-access
        return None if held is None else names.get(held)

    def _min_robot_clearance(self, state: Any,
                             exempt: set) -> Tuple[float, str]:
        skill = self._skill
        sim = skill._config.simulator  # pylint: disable=protected-access
        _, bodies, names, _, _ = skill._sim_collision_context(state)  # pylint: disable=protected-access
        robot = sim._pybullet_robot  # pylint: disable=protected-access
        robot.set_joints(state.joint_positions)
        client = sim._physics_client_id  # pylint: disable=protected-access
        best = _QUERY_DISTANCE
        best_body = ""
        for body in bodies:
            name = names.get(body, str(body))
            if name in exempt:
                continue
            points = p.getClosestPoints(robot.robot_id,
                                        body,
                                        _QUERY_DISTANCE,
                                        physicsClientId=client)
            if not points:
                continue
            dist = min(pt[8] for pt in points)
            if dist < best:
                best, best_body = dist, name
        return best, best_body

    def verdict(self) -> Tuple[bool, str, str]:
        """``(ok, summary line, detail)``.

        ``ok`` is False when the closest approach is inside the
        executor's pose slop; ``detail`` then names it for the refusal
        message (empty when ok or when nothing was probed).
        """
        if self.num_probes == 0:
            return True, "", ""
        thr = self.threshold
        if not math.isfinite(self.min_dist):
            return True, (f"min robot clearance: >{_QUERY_DISTANCE * 1e3:.0f}"
                          " mm"), ""
        summary = (f"min robot clearance: {self.min_dist * 1e3:.1f} mm "
                   f"({self.where}; executor pose slop {thr * 1e3:.0f} mm)")
        if self.min_dist < thr:
            return False, summary, (
                f"the robot passes within {self.min_dist * 1e3:.1f} mm of a "
                f"bystander ({self.where}), inside the executor's "
                f"{thr * 1e3:.0f} mm pose slop")
        return True, summary, ""


def clearance_lines(probe: Optional[RobotClearanceProbe]) -> List[str]:
    """Report lines for a probe, empty when no probe ran."""
    if probe is None:
        return []
    _, summary, _ = probe.verdict()
    return [summary] if summary else []
