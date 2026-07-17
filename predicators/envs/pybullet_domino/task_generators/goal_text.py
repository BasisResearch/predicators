"""Natural-language goal descriptions for min-block domino tasks.

Centralizes the goal-NL strings that were duplicated verbatim across the
plain, turn, and heavy min-block task builders. All share the same
"arrange the blues, push the green, topple the purple, use as few blues
as possible, keep everything else staged and standing until the push"
instruction; the heavy variant also names the gray heavy blocks as fixed
scenery.
"""

MIN_BLOCK_GOAL_NL = (
    "Arrange the blue dominoes so that when the green domino is pushed, "
    "the purple domino is toppled -- using AS FEW blue dominoes as "
    "possible (possibly none). Only the blue dominoes may be rearranged: "
    "the green and purple dominoes must stay untouched at their staged "
    "poses, upright and never held, until the green is pushed, and nothing "
    "may topple before that push. Only the green domino may ever be "
    "pushed.")

HEAVY_GOAL_NL = (
    "Arrange the blue dominoes so that when the green domino is pushed, "
    "the purple domino is toppled -- using AS FEW blue dominoes as "
    "possible (possibly none). Only the blue dominoes may be rearranged: "
    "the green and purple dominoes and the gray heavy blocks must stay "
    "untouched at their staged poses, upright and never held, until the "
    "green is pushed, and nothing may topple before that push. Only the "
    "green domino may ever be pushed.")
