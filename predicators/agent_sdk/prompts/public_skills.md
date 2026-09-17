# Skill API

Use `skills_list` for available names, typed object arguments, parameter order, descriptions, and bounds in the current environment.
Coordinates are in metres in the world frame and angles are in radians.
Skills execute real environment steps and may stop before reaching their targets.
Check their effects using the resulting observations and images.

- Pick skills attempt to grasp their named object and lift it.
  Follow the listed height-offset convention for that skill.
- Place moves to the requested drop pose, opens the gripper, and retreats.
  The listed parameters describe whether a coordinate targets the gripper or the held object.
- MoveTo moves to the requested world pose without releasing.
- Push and switch skills use the listed approach, contact-height, and stroke parameters.
- Pour moves the held container toward the specified receiving object.
- Release opens the named clip with the requested reach and stroke parameters.
- Wait holds the current robot joints.
  A positive step-count parameter requests that many environment steps, subject to the skill cap.
  A declared subgoal or episode termination can stop it sooner.

Some environments expose the primitive library instead of the skills above: `MoveTo`, `MoveLinear`, `MoveUntilContact`, `Gripper` and `Wait`, each taking only the robot.
Those skills know nothing about the objects: you choose grasp points, push directions and release moments from your observations.

- MoveTo plans a collision-free path to a world end-effector pose (x, y, z, yaw, tilt) and keeps the current finger width.
  It fails, naming the blocking body, when the pose is in contact or no collision-free path exists.
- MoveLinear moves the end effector along a straight world-frame displacement (dx, dy, dz) at the given metres per step, through contact.
  It fails, naming the contact, when the arm stops making progress.
- MoveUntilContact is the same stroke but stops at the first contact of the hand or the held object with anything else.
- Gripper opens or closes the fingers to a width in the robot's fingers feature units under a grip force limit in newtons, and stops when they arrive or stall on an object.
  Closing on an object grasps it; opening releases it; the simulated grasp ignores the force limit.

Controller failures report that the requested motion could not be completed.
They do not provide hidden attachment identities, internal contact distances, or an exact collision map.
Infer task mechanisms from interaction and public observations.
