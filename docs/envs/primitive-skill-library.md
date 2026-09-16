# The primitive skill library

`CFG.skill_library` selects which skills `get_gt_options` builds for the PyBullet continual environments.
`composite` (the default) is each environment's own factory-built set: `PickJug`, `SwitchFaucetOn`, `Push`, `Release`, `PickBlock`, and so on.
`primitive` replaces that set, in every supporting environment, with the same five domain-general skills.

## Why

The composite skills encode task knowledge in the controller: where a jug's handle is, which way a toggle slides, when a clip is released.
A real arm exposes none of that.
What a Franka or UR arm exposes through MoveIt or libfranka is a planned move to a pose, a guarded straight-line move, a gripper, and time.
Under the primitive library the agent supplies the grasp points, push directions and release moments itself, from its own (possibly noisy) observations.
Nothing in the library reads a target object's pose from the true state.

## The skills

Every skill takes only the robot.
Poses and displacements are in the world frame, in metres and radians.

| Skill | Parameters | Ends when |
|---|---|---|
| `MoveTo` | `x, y, z, yaw, tilt` | the planned, collision-free path arrives; fails naming the blocker when the pose is in contact or no path exists |
| `MoveLinear` | `dx, dy, dz, step` | the end effector has moved by the displacement; fails naming the contact when the arm stops making progress |
| `MoveUntilContact` | `dx, dy, dz, step` | the hand or the held object first touches another body, or the displacement is reached |
| `Gripper` | `width, force` | the fingers reach the width, or stall on an object |
| `Wait` | `steps` | the shared wait skill's rules |

`step` is the metres the end effector travels per environment step, so a small value is a gentle touch.
`width` is in the units of the robot's `fingers` observation feature, between the environment's closed and open values.
`tilt` and `yaw` are the end effector's pitch and yaw; the robot's `tilt` and `wrist` features read the current values.

The executor's controller-level rails stay on: the joint-jump guard on every incremental step, the stall abort that names the contact, and validated goal IK for planned moves.
Grasping is unchanged: the simulator attaches an object when the fingers pinch it while closing and detaches it when they open.
`force` is the grip force limit in newtons.
A real gripper's command is a width and a force, so the parameter is in the signature for the plan lines to transfer unchanged; the simulated rigid attachment ignores it, and the parameter description tells the agent so.

## Enabling it

Set `skill_library: primitive` in the run's flags.
Supported environments implement `GroundTruthOptionFactory.get_primitive_skill_context`, returning the same `SkillConfig` their composite skills use, so both libraries share one robot, one planning simulator and one set of tolerances: Boil, Fan, Domino, Bridge and Balloons.
Other environments raise `NotImplementedError` under `primitive`.
The library is sized from the environment class's `x_lb`..`z_ub` workspace bounds and its `closed_fingers`/`open_fingers` feature values.

The composite skills remain available under `composite`, which the oracle and NSRT-based approaches need.

For a skills-against-skills comparison also set `continual_raw_control: false`.
That removes the `env_step` and `env_run_policy` tools from both skill agents, so neither arm can drive the joints by hand around the library it was given.
