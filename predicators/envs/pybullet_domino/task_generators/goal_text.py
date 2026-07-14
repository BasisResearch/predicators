"""Natural-language goal descriptions for min-block domino tasks.

Centralizes the goal-NL strings that were duplicated verbatim across the
plain, turn, and heavy min-block task builders. All share the same
"arrange the blues, push the green, topple the purple, use as few blues
as possible, don't touch the purple" instruction; the heavy variant adds
that only the blues may be moved (the gray obstacle is fixed scenery).
"""

MIN_BLOCK_GOAL_NL = (
    "Arrange the blue dominoes so that when the green domino is pushed, "
    "the purple domino is toppled -- using AS FEW blue dominoes as "
    "possible (possibly none). Do NOT directly push or topple the purple "
    "domino yourself.")

HEAVY_GOAL_NL = (
    "Arrange the blue dominoes so that when the green domino is pushed, "
    "the purple domino is toppled -- using AS FEW blue dominoes as "
    "possible (possibly none). Only the blue dominoes may be moved. Do NOT "
    "directly push or topple the purple domino yourself.")
