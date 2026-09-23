# Skill API

Use `skills_list` for the available names, typed object arguments, parameter order, and parameter bounds in the current environment.
All coordinates are in metres in the world frame, and angles are in radians.
A skill executes real environment steps and may fail before reaching its target.
Use the resulting observations and images to check its effects.

- Pick skills attempt to grasp their named object and lift it.
  A `grasp_z_offset` is measured relative to the object's top face.
- Place targets the held object's centre at `(x, y, release_z)` with the requested yaw, then opens the gripper.
  Its controller may descend to nearby support before releasing.
- MoveTo targets the held object's centre, or the gripper when empty, at the requested world pose.
- Push and switch skills move along the direction and distance named by their parameters.
- Pour targets the specified receiving object with the requested motion parameters.
- Release opens the named clip using the requested reach and stroke parameters.
- Wait holds the current robot joints.
  A positive step-count parameter requests that many environment steps, up to the skill cap.
  A declared subgoal or an episode ending can terminate it sooner.

Skill availability, parameter limits, and these controller conventions do not specify the environment's hidden mechanisms.
Infer their effects from interaction.
