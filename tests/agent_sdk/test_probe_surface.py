"""The ``sim`` probe description offers exactly what each arm accepts."""
from predicators.agent_sdk.tools.exploration import ProbeSurface, \
    belief_probe_blurb


def test_default_surface_offers_everything() -> None:
    """The learning arm fits, edits, scores alternatives and sweeps."""
    text = belief_probe_blurb(synthesis_probe=True)
    for phrase in ("`sim.fit(", "PARAMS UNFITTED", "sweep_params",
                   "phys_params", "belief_draws", "sim.suggest_probes",
                   "fit_params=False"):
        assert phrase in text


def test_supplied_surface_hides_model_and_fitting() -> None:
    """Scene-only and oracle dynamics: a fixed model queried through sim."""
    text = belief_probe_blurb(synthesis_probe=True,
                              surface=ProbeSurface(fit=False,
                                                   edit_model=False,
                                                   alt_params=False))
    assert "not exposed as source" in text
    for phrase in ("sim.fit", "PARAMS UNFITTED", "sweep_params", "phys_params",
                   "params=None", "simulator.py rules"):
        assert phrase not in text
    assert "belief_draws" in text
    assert "`sim.validate(traj_idxs=None)`" in text


def test_sealed_surface_names_the_seal() -> None:
    """Zero-shot: the agent's own file, sealed at the first action."""
    text = belief_probe_blurb(synthesis_probe=True,
                              surface=ProbeSurface(fit=False,
                                                   sealed=True,
                                                   alt_params=False))
    assert "seals it" in text
    assert "sim.fit" not in text
    assert "simulator.py rules" in text


def test_no_fit_surface_keeps_alternative_values() -> None:
    """No harness fitting: declared values, candidates scored by validate."""
    text = belief_probe_blurb(synthesis_probe=True,
                              surface=ProbeSurface(fit=False))
    assert "the harness fits nothing" in text
    assert "sim.fit" not in text
    assert "sweep_params" not in text
    assert "phys_params={name: value} scores ONE" in text
    assert "`sim.validate(traj_idxs=None, params=None)`" in text


def test_point_estimate_surface_drops_uncertainty() -> None:
    """No explicit uncertainty: no draws, beliefs or disagreement probes."""
    text = belief_probe_blurb(synthesis_probe=True,
                              surface=ProbeSurface(uncertainty=False))
    for phrase in ("belief_draws", "sim.belief()", "sim.suggest_probes"):
        assert phrase not in text
    assert "`sim.fit(" in text


def test_solve_surface_unchanged() -> None:
    """The solve-phase probe keeps its wording."""
    text = belief_probe_blurb(synthesis_probe=False)
    assert "over the belief simulator" in text
    assert "sim.fit" not in text
