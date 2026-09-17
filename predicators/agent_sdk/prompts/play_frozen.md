# Continual frozen model comparisons

<!-- section: zero_shot -->
## Zero-shot dynamics comparison

Write a valid simulator.py from the task description, initial observation, and supplied base simulator before taking any real action or reset.
You may rehearse and debug that initial model using sandbox computation.
The first real action seals its code, parameter values, and declared parameter ranges for the entire run.
No numerical fitting, data-driven parameter sweeps, or subsequent dynamics edits are permitted, including after resets, level changes, or resume.
Do not route dynamics through mutable helper files, recorded data files, or other changing external inputs.
These restrictions replace the earlier instructions to learn, repair, or refit dynamics from interaction.
Continue to plan, infer current model memory from observations, and update predicates and journal using recorded experience.
If a later dynamics edit is attempted, model predictions and real actions refuse until the original source is restored; round completion restores it automatically.

<!-- section: oracle_scene -->
## Oracle scene reconstruction comparison

The supplied simulator.py fixes exact scene geometry, robot articulation, and base body properties, including masses, friction and damping where they were miscalibrated.
It contains no added mechanism dynamics.
This is an idealized scene-reconstruction reference, not a learned reconstruction system.
Current object poses and sensor readings remain noisy, and hidden execution state is not provided.
Use the supplied fixed simulator for planning and rehearsal; do not change it, add missing mechanisms, fit parameters, or route predictions through an alternative dynamics model.
These restrictions replace earlier model learning and repair instructions.
You may collect experience, adjust plans, and update predicates and journal.

<!-- section: oracle_dynamics -->
## Oracle dynamics comparison

The supplied simulator.py contains the environment's mechanism dynamics and correct parameter values.
The model is fixed for the entire run; do not edit it, fit parameters, or route predictions through another dynamics model.
These restrictions replace earlier model learning and repair instructions.
Current observations remain noisy and hidden execution state is not handed to you.
You still choose experiments and actions, plan, rehearse, infer current state, and invent predicates.
An oracle dynamics model does not supply an oracle controller or guarantee a successful plan from an uncertain current state.
