# Balloons

Balloons and a box.
A box sits at the left of the table, balloons rest in a rack along the back, each held by a clip in front of it, and a translucent band floats beside the box's column.
The robot pushes a clip open; the freed balloon's string pulls it to the box, and it pulls the box up.
The goal is the box hanging at rest with its centre inside the band, with no balloon burst on the ceiling.

## Visual explanation

Open [the overview slides](overview_slides.html) for the original chute and the new hatch prototype.
The Hatch navigation button jumps to the comparison, collision rules, and two embedded replay videos.
The videos show red then gold jamming and gold then red passing on the same audited hatch task.
They are mechanical demonstrations, separate from MB, MF and continual-oracle results.
The [PDF](../../slides/balloons_domain_slides.pdf) includes stills in place of the videos.
Individual [jam](visuals/hatch-jam.mp4) and [passage](visuals/hatch-pass.mp4) clips, the [height plot](visuals/hatch-height.png), and [reproduction notes](visuals/README.md) are also available.

## What is hidden

- Each balloon colour's lift, which fades linearly with height, so a box with enough balloons rises to the height where the pull matches its weight and hangs there.
  On the base simulator an opened clip frees nothing: the balloon stays in the rack and the box stays put.
- The release itself (an open clip's balloons fly to the end of their strings above the box, a bundle side by side at one tier, later bundles stacking a tier above earlier ones), the mass of each box material, and the air's drag.
- A freed balloon that reaches the ceiling bursts: the level is lost, and a freed balloon cannot be clipped back.

## What the agent controls

`Release(robot, clip)[approach, contact_z]` (the shared push skill on the clip's toggle) and `Wait(robot)`.
The observation carries the box's pose, colour and speed, every balloon's pose, colour, clip index, tied and popped flags, every clip's pose and latched state, and the band's heights.

## Train and test

Train levels have a pine or oak box and two or three balloons; together they show both materials and every balloon colour.
Test levels hold the whole palette, four balloons, on a pine or oak box: a combination training never showed, built from lifts and a mass it did.
Generation version 2 chooses a reference subset whose tested immediate release orders all reach the evaluator's goal.
Test levels also contain a witnessed losing release sequence with an in-band analytic equilibrium.
Other subsets and orders may win too; uniqueness is not claimed.
The generator checks every simulation frame, requires sustained rest before classifying off-target failure, and keeps timeouts unresolved.
A jam label additionally requires the same sequence to win in a diagnostic replay without box-wall collisions.
The oracle plans over clips with `Needed` and `Holds` helpers and a derived `AllNeededOpen`.

## Bundle test levels

`balloons_test_bundle_sizes` (for example `[2, 2, 2, 2]`) ties the test rack's balloons into bundles, one clip per bundle, racked in a row behind the clip; every balloon's `clip` feature names its clip.
One cut frees the whole bundle, so no cut is a small trim and every first cut is an unseen union launched from the table.
The generator accepts a bundle level only when the reference wins with two or more settled cuts and does not start with the weakest bundle, every winning union is a colour multiset no train rack on the same box held, a decoy bundle rests in the band by the analytic law yet bursts when cut first, every winning order starts with the same clip, and, when the draws allow it, a tempting bundle rests below the band above the reference's first cut with an analytic in-band continuation that loses (the generator prefers such a level and falls back to one without, recorded as `bundle_tempting_clip` -1).
Of the three readings of the rest heights (weakest first, the bundle that rests in band, the highest safe bundle then add) the first two always lose and the third loses on a level with a tempting bundle; the winning order is found by rolling the ascent forward.
A rack larger than the palette repeats colours.
The metrics record `bundle_decoy_clip`, `bundle_tempting_clip`, `bundle_reference_cuts` and `bundle_reference_first_clip`.

## What the model must predict

Which clips to open is a small choice, but the height each choice gives is a quantitative composition of per colour lifts, the fade and the box's mass, and one balloon too many bursts on the ceiling.
Release order also changes the ascent: the same pair can burst in one order and succeed in the reverse order.
A model can help compare these choices before irreversible actions, but an advantage over a free-coding MF agent must be measured.
See [the version 2 validation plan](../../uncertainty-results/balloons-v2-plan.md) for the checked release schedules and benchmark scope.

## Files

- `predicators/envs/pybullet_balloons_base.py`: the visible simulator (surfaced to agents as reference source).
- `predicators/envs/pybullet_balloons.py`: release, lift, pop, masses, tasks, evaluator, predicates.
- `predicators/ground_truth_models/balloons/`: options, helper predicates, processes, the oracle's plan.
- `tests/envs/test_pybullet_balloons.py`.
- The video and images below are historical illustrations from an earlier generator, not version 2 results.
- `oracle_solve.mp4` in this directory: the oracle playing a train level then a test level under the continual protocol.
- `train_start.jpg`, `train_won.jpg`, `test_start.jpg`, `test_won.jpg`: historical frames of that video.
- `sweep_20260907/`: the later chute sweep's recorded images and scorecards, retained in the overview slides.
- `visuals/`: current geometry illustrations and audited hatch replay videos.
