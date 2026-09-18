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

The supplied simulator.py fixes exact scene geometry, robot articulation, and base body properties, including masses, friction and damping where they were miscalibrated.
It contains no mechanism dynamics, and it is fixed for the entire run.
This is an idealized scene twin, not a learned reconstruction system.
Current object poses and sensor readings remain noisy, and hidden execution state is not provided.
Use the supplied simulator for planning and rehearsal; do not change it, add missing mechanisms, fit parameters, or route predictions through an alternative dynamics model.
You may collect experience, adjust plans, and update predicates and journal.

<!-- section: oracle_dynamics -->
## Oracle dynamics comparison

The supplied simulator.py contains the environment's mechanism dynamics and correct parameter values.
The model is fixed for the entire run; do not edit it, fit parameters, or route predictions through another dynamics model.
Current observations remain noisy and hidden execution state is not handed to you.
You still choose experiments and actions, plan, rehearse, infer current state, and invent predicates.
An oracle dynamics model does not supply an oracle controller or guarantee a successful plan from an uncertain current state.
