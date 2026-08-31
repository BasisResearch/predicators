"""Is the domino-fan wind force recoverable from rollouts? Measure it.

Answers, in about a minute and with no LLM in the loop, the question
that otherwise costs a 40-minute learning run to answer badly: does a
change in ``domino_fan_wind_force`` produce a change a fitter could
see?

It does not, and the reason is the domino rather than the optimizer.
The wind acts only while the start block is upright (a fallen domino
is out of the airstream), which at any force that topples at all is
about two simulation steps. So the entire observation is "tipped or
did not", and the map from force to that observation is a step:

    force (N)   topple step   drift (mm)
         0.10   never                0.02
         0.25   never                0.00
         0.50   never                0.01
         0.75   never                0.03
         1.00   5                    7.70
         1.50   2                    9.49     <- the env's true value
         2.00   2                   19.23
         3.00   1                   11.16

1.5 N and 2.0 N are indistinguishable. Below ~0.8 N nothing moves at
all -- static friction holds the block, so there is no gentle
sub-threshold signal either -- and the post-topple drift is
non-monotonic, being tumbling rather than a function of the parameter.
A run's own sysID diagnostics said the same thing in their own words
("weakly identified", "NOT identified (posterior ~= prior)").

Contrast ``pybullet_fan``, which fits this same parameter happily: its
wind pushes a BALL with no stopping condition, so every timestep's
position is a continuous monotone function of the force. Whether a
wind parameter is fittable is a property of what the wind is pushing.

Run it after any change to the wind, the block geometry, or the
friction, to check whether the answer has moved:

    PYTHONHASHSEED=0 python scripts/domino_debug/probe_wind_identifiability.py
"""

import numpy as np

from predicators import utils


def main() -> None:
    """Sweep the wind force; report what an observer could measure."""
    utils.reset_config({
        "env": "pybullet_domino_fan",
        "seed": 0,
        "num_test_tasks": 1,
        "num_train_tasks": 0,
        "domino_fan_aligned_tasks": True,
    })
    # Imported after reset_config so the env sees the right settings.
    # pylint: disable=import-outside-toplevel
    from predicators.envs.pybullet_domino.components.domino_component import \
        DominoComponent
    from predicators.envs.pybullet_domino.env import PyBulletDominoFanEnv
    from predicators.settings import CFG
    from predicators.structs import Action, EnvironmentTask

    env = PyBulletDominoFanEnv(use_gui=False)
    base = env.get_test_tasks()[0]
    thresh = DominoComponent.domino_roll_threshold

    # The same scene, but with every fan already blowing at t=0: the
    # robot is held still, so the wind is the only thing acting and
    # nothing else can be credited with the topple.
    init = base.init.copy()
    for obj in init:
        if obj.type.name in ("fan", "switch"):
            init.set(obj, "is_on", 1.0)
    green = next(o for o in init
                 if o.type.name == "domino" and all(
                     abs(float(init.get(o, c)) -
                         DominoComponent.start_domino_color[i]) < 1e-3
                     for i, c in enumerate(("r", "g", "b"))))
    task = EnvironmentTask(init, base.goal)

    def drift(state) -> float:
        """How far the block has moved from where it started (m)."""
        return float(
            np.hypot(state.get(green, "x") - init.get(green, "x"),
                     state.get(green, "y") - init.get(green, "y")))

    def probe(force: float):
        """(steps until the block topples, drift) at this wind force."""
        CFG.domino_fan_wind_force = force
        env._current_task = task  # pylint: disable=protected-access
        env._set_state(init)  # pylint: disable=protected-access
        # Hold the arm exactly where it is.
        hold = np.array(env._pybullet_robot.get_joints(),  # pylint: disable=protected-access
                        dtype=np.float32)
        state = init
        for step in range(150):
            env.step(Action(hold))
            state = env._get_state()  # pylint: disable=protected-access
            roll = float(state.get(green, "roll"))
            # Roll is meaningful modulo pi: a box turned 180 degrees
            # about its width axis is the same box.
            roll = (roll + np.pi / 2) % np.pi - np.pi / 2
            if abs(roll) >= thresh:
                return step, drift(state)
        return None, drift(state)

    print(f"\n{'force (N)':>10} {'topple step':>13} {'drift (mm)':>12}")
    for force in (0.10, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00):
        step, dist = probe(force)
        mark = "   <- env's value" if abs(force - 1.5) < 1e-9 else ""
        print(f"{force:>10.2f} {str(step):>13} {1000 * dist:>12.2f}{mark}",
              flush=True)
    print("\nDistinct topple-step values across a 30x range of force is the "
          "\nwhole signal a fitter has. See the module docstring.\n")


if __name__ == "__main__":
    main()
