# Continual frozen model comparisons

Each section opens the play prompt of one frozen-dynamics arm, before the
decision workflow. The workflow and workbench that follow are the frozen
variants (play_system.md: workflow_frozen, model_frozen), so nothing later in
the prompt asks the agent to fit, repair, or rewrite the dynamics.

<!-- section: zero_shot -->
## Zero-shot dynamics comparison

Write a valid simulator.py from the task description, initial observation, and supplied base simulator before taking any real action or reset.
You may rehearse and debug that initial model using sandbox computation.
The first real action seals its code, parameter values, and declared parameter ranges for the entire run.
No numerical fitting, data-driven parameter sweeps, or subsequent dynamics edits are permitted, including after resets, level changes, or resume.
Do not route dynamics through mutable helper files, recorded data files, or other changing external inputs.
Continue to plan, infer current model memory from observations, and update predicates and journal using recorded experience.
If a later dynamics edit is attempted, model predictions and real actions refuse until the original source is restored; round completion restores it automatically.

<!-- section: scene_only -->
## Scene-only comparison

The supplied simulator, which runs inside `sim` and is not exposed as source, fixes exact scene geometry, robot articulation, and base body properties, including masses, friction and damping where they were miscalibrated.
It contains no mechanism dynamics, and it is fixed for the entire run.
This is an idealized scene twin, not a learned reconstruction system.
Current object poses and sensor readings remain noisy, and hidden execution state is not provided.
Use the supplied simulator for planning and rehearsal; do not change it, add missing mechanisms, fit parameters, or route predictions through an alternative dynamics model.
You may collect experience, adjust plans, and update predicates and journal.

<!-- section: no_learning -->
## No-learning comparison

Write `./simulator.py` as a `SceneBase` subclass that loads the scene's bodies from the scene manifest and assets and maps the observed features to them.
Write no mechanisms: the file may not override `_domain_specific_step` or the step methods, declare parameters, or queue forces, torques, velocities or attachments, and `sim` refuses a file that does.
Set each body's physical values once, from the manifest, the asset files and general knowledge, and do not change them to match recordings.
Edit the file only to fix the scene, such as a wrong body, joint or feature mapping, or to load a new level's bodies.
`sim` has no world until `./simulator.py` loads; afterwards it runs the real skill controllers inside your scene.
Nothing fits parameters, and the model has no hidden dynamics.
Current object poses and sensor readings remain noisy, and hidden execution state is not provided.
Use the model for planning and rehearsal. You may collect experience, adjust plans, and update predicates and journal.

<!-- section: no_learning_loaded -->
## No-learning comparison

Write `./simulator.py` as a `SceneBase` subclass: the base loads the scene's bodies from the scene manifest and assets, and your file maps the observed features that no body pose carries.
Write no mechanisms: the file may not override `_domain_specific_step` or the step methods, declare parameters, or queue forces, torques, velocities or attachments, and `sim` refuses a file that does.
You may set each body's physical values once, in `_on_scene_loaded` with `p.changeDynamics`, from the asset files and general knowledge; do not change them to match recordings.
Edit the file only to fix the scene, such as a wrong joint or feature mapping.
`sim` has no world until `./simulator.py` loads; afterwards it runs the real skill controllers inside your scene.
Nothing fits parameters, and the model has no hidden dynamics.
Current object poses and sensor readings remain noisy, and hidden execution state is not provided.
Use the model for planning and rehearsal. You may collect experience, adjust plans, and update predicates and journal.

<!-- section: oracle_dynamics -->
## Oracle dynamics comparison

The supplied simulator, which runs inside `sim` and is not exposed as source, contains the environment's mechanism dynamics and correct parameter values.
The model is fixed for the entire run; do not edit it, fit parameters, or route predictions through another dynamics model.
Current observations remain noisy and hidden execution state is not handed to you.
You still choose experiments and actions, plan, rehearse, infer current state, and invent predicates.
An oracle dynamics model does not supply an oracle controller or guarantee a successful plan from an uncertain current state.
