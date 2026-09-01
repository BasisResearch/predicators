"""Learned predicates for pybullet_domino_fan.

All numeric cutoffs are shared with simulator.py's PARAM_SPECS via the
pre-injected `params` view, so a refit moves rule and predicate together.

Evidence for each cutoff (belief-sim sweeps at the demo layout, which
reproduces both recorded cascades to <= 1 cm on every body):
  * bridge_max_gap 0.13 : centre-to-centre spacings 0.06 / 0.08 / 0.10 /
    0.12 / 0.14 all relay the cascade to the purple target; 0.16 stalls
    (the struck domino only reaches roll 0.12).  0.13 sits inside the
    working bucket with a clear margin to the 0.16 failure.
  * bridge_max_lateral 0.035 : lateral offsets of 0.00 and 0.06 still
    relay, 0.12 does not.  0.035 is deliberately tighter than the
    empirical boundary (tighten, never widen).
  * bridge_yaw_align 0.5 : |a.u| of 0.45 and 0.70 relay, 0.92 and 1.00
    (edge-on bridge) stall the chain and even jam the green start block
    at roll ~0.5.
  * toppled_roll 0.7 : recorded rolls are ~0 while standing and settle at
    1.44-1.57 once down; nothing ever rests between 0.2 and 1.4.
Recorded domino poses are BODY CENTRES (z = 0.475 = table 0.40 + half of
the 0.15 m height), so domino-domino spacing needs no anchor offset - the
recorded origin is the functional point.  The fan, by contrast, is only
ever used through its `rot`-derived facing direction, never its origin
distance, so no fan anchor offset is needed either.
"""


_DEFAULTS = {
    "bridge_max_gap": 0.13,
    "bridge_max_lateral": 0.035,
    "bridge_yaw_align": 0.5,
    "toppled_roll": 0.7,
    "upright_roll": 0.2,
}


def _p(name):
    """Read a shared simulator ParamSpec, falling back to its declared
    init_value when no fit has populated the params view yet (the
    predicate-quality loader can run before the first fit)."""
    try:
        return float(params[name])
    except Exception:
        return _DEFAULTS[name]


def _roll(s, d):
    return abs(float(s.get(d, "roll")))


def _toppled(s, objs, latent=None):
    return _roll(s, objs[0]) >= _p("toppled_roll")


def _upright(s, objs, latent=None):
    return _roll(s, objs[0]) < _p("upright_roll")


def _fan_on(s, objs, latent=None):
    return float(s.get(objs[0], "is_on")) > 0.5


def _fan_off(s, objs, latent=None):
    return float(s.get(objs[0], "is_on")) <= 0.5


def _switch_on(s, objs, latent=None):
    return float(s.get(objs[0], "is_on")) > 0.5


def _xy(s, o):
    return np.array([float(s.get(o, "x")), float(s.get(o, "y"))])


def _bridges_gap(s, objs, latent=None):
    """`bridge` stands between `src` and `tgt` close enough, and squarely
    enough, that a topple travelling src -> tgt relays through it."""
    bridge, src, tgt = objs
    if bridge is src or bridge is tgt or src is tgt:
        return False
    if float(s.get(bridge, "is_held")) > 0.5:
        return False
    p_b, p_s, p_t = _xy(s, bridge), _xy(s, src), _xy(s, tgt)
    span = p_t - p_s
    span_len = float(np.linalg.norm(span))
    if span_len < 1e-6:
        return False
    u = span / span_len
    perp = np.array([-u[1], u[0]])
    along = float((p_b - p_s) @ u)
    # must sit strictly between the two, and split the span into two
    # hops each short enough to carry the topple
    if along <= 0.0 or along >= span_len:
        return False
    if float(np.linalg.norm(p_b - p_s)) > _p("bridge_max_gap"):
        return False
    if float(np.linalg.norm(p_t - p_b)) > _p("bridge_max_gap"):
        return False
    if abs(float((p_b - p_s) @ perp)) > _p("bridge_max_lateral"):
        return False
    # face the oncoming domino squarely: the width axis (cos yaw, sin yaw)
    # must be near-perpendicular to the cascade direction
    yaw = float(s.get(bridge, "yaw"))
    axis = np.array([np.cos(yaw), np.sin(yaw)])
    if abs(float(axis @ u)) > _p("bridge_yaw_align"):
        return False
    return True


LEARNED_PREDICATES = [
    Predicate("DomToppled", [domino_type], _toppled),
    Predicate("DomUpright", [domino_type], _upright),
    Predicate("FanRunning", [fan_type], _fan_on),
    Predicate("FanIdle", [fan_type], _fan_off),
    Predicate("SwitchPressed", [switch_type], _switch_on),
    Predicate("DomBridges", [domino_type, domino_type, domino_type],
              _bridges_gap),
]

