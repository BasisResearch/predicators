"""Natural-language goal descriptions for min-block domino tasks.

Centralizes the goal-NL strings that were duplicated verbatim across the
plain, turn, and heavy min-block task builders. All share the same
"arrange the blues, push the green, topple the purple, use as few blues
as possible, keep everything else staged and standing until the push"
instruction; the heavy variant also names the gray heavy blocks as fixed
scenery.
"""

# The evaluator's counterfactual verification, stated in the goal text
# so every enforced rule is inferable up front: without it, an agent
# whose rollouts pass only with the arm's help sees nothing but opaque
# solved=False verdicts (run_20260718_141716 burned two 45-minute
# attempts theorizing about an "unknown evaluator criterion").
CASCADE_VERIFICATION_NL = (
    " A solve only counts if the push itself causes the cascade: it is "
    "verified by replaying your push with every robot link except the "
    "fingertips made intangible, and the built layout must still cascade "
    "to the goal - topples that needed the arm's body earn nothing.")

# The wind variant of the same rule. There is no counterfactual replay
# to describe: the probe exists to prove the arm's body did not carry a
# cascade its push started, and a switch pressed metres from the chain
# has no such contact to disprove. What is enforced instead is that a
# TurnFanOn step is actually on the record -- otherwise an episode that
# reached the goal by knocking the chain over while placing a block
# would certify.
WIND_VERIFICATION_NL = (
    " A solve only counts if switching the fan on is what starts the "
    "cascade: the episode "
    "must carry a TurnFanOn step, nothing may topple before it, and an "
    "episode that reaches the goal with no TurnFanOn on the record is "
    "rejected.")

MIN_BLOCK_GOAL_NL = (
    "Arrange the blue dominoes so that when the green domino is pushed, "
    "the purple domino is toppled -- using AS FEW blue dominoes as "
    "possible (possibly none). Only the blue dominoes may be rearranged: "
    "the green and purple dominoes must stay untouched at their staged "
    "poses, upright and never held, until the green is pushed, and nothing "
    "may topple before that push. Only the green domino may ever be "
    "pushed." + CASCADE_VERIFICATION_NL)

# The declaration variant. Same rule, different named step: there is
# no counterfactual replay for a skill that moves nothing, so what is
# checked is that the declaration is on the record and that nothing
# fell before it.
DECLARE_VERIFICATION_NL = (
    " A solve only counts if the declaration is what starts the "
    "cascade: the episode "
    "must carry a DeclareFinished step, nothing may topple before it, "
    "and an episode that reaches the goal with no DeclareFinished on "
    "the record is rejected.")

# The min-block instruction for a wind-triggered env. Nothing pairs
# domino_min_block_tasks with a fan today, but the two flags are
# independent and the push wording is unfollowable in a fan env, so the
# builder picks between them rather than leaving a trap set.
MIN_BLOCK_WIND_GOAL_NL = (
    "Arrange the blue dominoes so that when the fan is switched on, the "
    "wind topples the green domino and the purple domino is toppled -- "
    "using AS FEW blue dominoes as possible (possibly none). Only the "
    "blue dominoes may be rearranged: the green and purple dominoes must "
    "stay untouched at their staged poses, upright and never held, until "
    "the fan is switched on, and nothing may topple before that. The "
    "robot must never push a domino - the only way to start the cascade "
    "is to press the fan's switch." + WIND_VERIFICATION_NL)

HEAVY_GOAL_NL = (
    "Arrange the blue dominoes so that when the green domino is pushed, "
    "the purple domino is toppled -- using AS FEW blue dominoes as "
    "possible (possibly none). Only the blue dominoes may be rearranged: "
    "the green and purple dominoes and the gray blocks must stay "
    "untouched at their staged poses, upright and never held, until the "
    "green is pushed, and nothing may topple before that push. Only the "
    "green domino may ever be pushed." + CASCADE_VERIFICATION_NL)
