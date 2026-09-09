"""Build the standalone uncertainty-results HTML deck and its PDF copy.

Uses the saved result snapshot and the six current curve figures. Run on
a compute node with --pdf to render and validate the PDF.
"""
import argparse
import base64
import importlib.util
import json
import re
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "uncertainty-results"
STEM = HERE / "uncertainty_results_slides"
FEATURES = [
    ("Parameter intervals", "code_sim_learning_interval_belief",
     "Retain informative but wide fits and test plans across their intervals."
     ),
    ("Noise-aware probes", "agent_explorer_info_seeking_noise_aware",
     "Value a probe by whether its noisy observation can distinguish models."),
    ("Fit-side filtering", "code_sim_learning_rollout_noise_filter",
     "Find rest and motion relative to sigma; start rollouts from denoised rest windows."
     ),
    ("Carried fit center", "code_sim_learning_carry_posterior",
     "Use the last applied MAP as the next prior center; do not carry its width."
     ),
    ("Laplace evidence", "code_sim_learning_fit_evidence",
     "Report fit quality with a complexity penalty and comparable version deltas."
     ),
    ("Execution belief", "continual_belief_frame",
     "Expose smoothed state, spread and atom fractions for planning and monitoring."
     ),
]

CSS = """
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; font-family: 'DejaVu Sans', sans-serif; color: #182c3b; }
body { background: #172a39; }
.slide { position: relative; width: 1280px; height: 720px; background: white; padding: 48px 64px 45px; overflow: hidden; }
.kicker { font-size: 14px; text-transform: uppercase; letter-spacing: 2px; color: #008577; font-weight: bold; margin: 0 0 18px; }
h1 { font-size: 47px; line-height: 1.12; margin: 0 0 20px; letter-spacing: -1px; }
h2 { font-size: 35px; line-height: 1.18; margin: 0 0 22px; letter-spacing: -0.5px; }
h3 { font-size: 23px; line-height: 1.23; margin: 0 0 13px; }
p { font-size: 22px; line-height: 1.42; margin: 0 0 18px; }
.lead { color: #526475; font-size: 24px; }
.small { font-size: 17px; line-height: 1.45; }
.muted { color: #526475; }
.teal { color: #008577; }
.orange { color: #d77831; }
.blue { color: #4169a1; }
.row { display: flex; width: 100%; margin-top: 24px; }
.col { flex: 1; min-width: 0; margin-right: 24px; }
.col:last-child { margin-right: 0; }
.card { background: #f3f6f8; padding: 24px; border-top: 4px solid #4169a1; }
.card.tealcard { border-top-color: #008577; background: #eef8f5; }
.card.orangecard { border-top-color: #d77831; background: #fbf4ee; }
.execution .tealcard p { font-size: 20px; line-height: 1.35; margin-bottom: 14px; }
.execution .tealcard p.small { font-size: 16px; }
.big { font-size: 66px; font-weight: bold; line-height: 1.08; margin: 13px 0 8px; color: #008577; }
.rule { background: #eef8f5; border-left: 5px solid #008577; padding: 18px 22px; font-size: 22px; margin-top: 22px; }
.rule.small { font-size: 17px; }
.footer { position: absolute; bottom: 18px; left: 64px; right: 64px; color: #657483; font-size: 11px; line-height: 1.4; }
.page { position: absolute; bottom: 18px; right: 25px; color: #657483; font-size: 12px; }
.graphic { padding: 0; }
.graphic img { position: absolute; top: 0; left: 100px; width: 1080px; height: 720px; }
.graphic .page { background: white; padding: 3px; }
.feature-row { display: flex; margin-top: 18px; }
.feature { width: 32%; margin-right: 2%; background: #f3f6f8; padding: 20px; height: 188px; border-top: 4px solid #008577; }
.feature:last-child { margin-right: 0; }
.feature h3 { font-size: 21px; }
.feature p { font-size: 18px; }
.index { color: #008577; font-size: 13px; font-weight: bold; margin-bottom: 9px; }
.equation { color: #008577; font-size: 34px; font-weight: bold; margin: 22px 0; }
.code { font-family: 'DejaVu Sans Mono', monospace; font-size: 17px; color: #4169a1; }
.tool { margin-bottom: 18px; padding-bottom: 16px; border-bottom: 1px solid #dce4ea; }
.tool:last-child { border: none; }
.tool p { font-size: 18px; margin: 6px 0 0; }
table { width: 100%; border-collapse: collapse; font-size: 18px; line-height: 1.4; }
th { background: #eaf0f4; text-align: left; color: #344b60; }
td, th { padding: 10px 12px; border-bottom: 1px solid #dce4ea; vertical-align: top; }
.flags td { padding: 14px 10px; }
.flags .code { font-size: 16px; }
.results { font-size: 13.5px; line-height: 1.18; }
.results td, .results th { padding: 5px 7px; }
.results .num { text-align: right; white-space: nowrap; }
.results .group td { border-top: 2px solid #93a8b8; }
.results .highlight { background: #edf8f4; }
.results .historical { color: #667789; background: #f6f6f6; }
.chip { display: inline-block; font-size: 16px; padding: 5px 10px; background: #e5f3ef; color: #008577; margin-bottom: 14px; }
.diagram { width: 100%; height: 228px; display: block; margin-top: 12px; }
.cover-bottom { font-size: 18px; margin-top: 24px; color: #526475; }
.speaker { display: none; }
.nav { position: fixed; bottom: 12px; left: 50%; transform: translateX(-50%); display: flex; align-items: center; z-index: 30; background: #172a39; border: 1px solid #536777; border-radius: 8px; color: white; padding: 6px; }
.nav button { border: none; background: transparent; color: white; padding: 5px 14px; font: 14px 'DejaVu Sans', sans-serif; cursor: pointer; }
.nav span { font-size: 12px; padding: 0 12px; min-width: 80px; text-align: center; }
.nav button:focus-visible { outline: 2px solid #61c6b5; }
@media screen { #stage { position: absolute; top: 50%; left: 50%; width: 1280px; height: 720px; transform-origin: center center; } .slide { display: none; } .slide.active { display: block; } }
@page { size: 1280px 720px; margin: 0; }
@media print { body { background: white; } #stage { transform: none !important; position: static; } .slide { display: block !important; page-break-after: always; } .slide:last-child { page-break-after: auto; } .nav { display: none; } }
"""

JS = """
const slides = Array.from(document.querySelectorAll('.slide'));
const stage = document.getElementById('stage');
let current = 0;
function show(index) {
  current = Math.max(0, Math.min(slides.length - 1, index));
  slides.forEach((slide, i) => { slide.classList.toggle('active', i === current); slide.setAttribute('aria-hidden', i !== current); });
  document.getElementById('counter').textContent = (current + 1) + ' / ' + slides.length;
  document.getElementById('previous').disabled = current === 0;
  document.getElementById('next').disabled = current === slides.length - 1;
  history.replaceState(null, '', '#slide-' + (current + 1));
}
function resize() {
  const scale = Math.min((innerWidth - 24) / 1280, (innerHeight - 76) / 720);
  stage.style.transform = 'translate(-50%, -50%) scale(' + Math.max(0.1, scale) + ')';
}
function hashIndex() { const n = Number(location.hash.replace('#slide-', '')); return Number.isFinite(n) && n > 0 ? n - 1 : 0; }
document.getElementById('previous').onclick = () => show(current - 1);
document.getElementById('next').onclick = () => show(current + 1);
document.getElementById('fullscreen').onclick = () => { if (document.fullscreenElement) document.exitFullscreen(); else document.documentElement.requestFullscreen(); };
document.addEventListener('keydown', event => {
  if (['ArrowRight', 'PageDown', ' '].includes(event.key)) { event.preventDefault(); show(current + 1); }
  if (['ArrowLeft', 'PageUp'].includes(event.key)) { event.preventDefault(); show(current - 1); }
  if (event.key === 'Home') { event.preventDefault(); show(0); }
  if (event.key === 'End') { event.preventDefault(); show(slides.length - 1); }
});
window.addEventListener('resize', resize);
window.addEventListener('hashchange', () => show(hashIndex()));
show(hashIndex()); resize();
"""


def load_results():
    """Use the same aggregation function as the report and figures."""
    spec = importlib.util.spec_from_file_location("uncertainty_figures",
                                                  RESULTS / "make_figures.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    payload = json.loads((RESULTS / "snapshot.json").read_text())
    return payload, [module.aggregate(row)
                     for row in payload["rows"]], module.number


def image_uri(name):
    """Embed vector charts so the HTML is a portable single file."""
    data = (RESULTS / f"{name}.svg").read_bytes()
    return "data:image/svg+xml;base64," + base64.b64encode(data).decode(
        "ascii")


def build():
    """Build the authored slides around the verified result snapshot."""
    payload, summaries, number = load_results()
    assert len(summaries
               ) == 17, "Review the slide narrative when adding new settings."
    assert all(run["finished_at"] is not None for row in payload["rows"]
               for run in row["runs"])
    headlines = {
        row["domain"]: row
        for row in summaries if row["arm"] == "MB + uncertainty"
    }
    assert set(headlines) == {"Domino", "Boil"}
    assert all(row["successful_seeds"] == row["seeds"] == 2
               for row in headlines.values())
    slides = []

    def add(title,
            body,
            source="Source: docs/continual-uncertainty.md, implementation status (§8).",
            notes="",
            graphic=False):
        page = len(slides) + 1

        # WeasyPrint 52 renders SVG images but not inline SVG elements.
        # Embed authored diagrams as images so screen and PDF use one asset.
        def diagram_image(match):
            svg = match.group(0).encode("utf-8")
            uri = "data:image/svg+xml;base64," + base64.b64encode(svg).decode(
                "ascii")
            return f'<img class="diagram" alt="Parameter belief interval compared with the prior anchor" src="{uri}">'

        body = re.sub(r"<svg\b.*?</svg>", diagram_image, body, flags=re.DOTALL)
        footer = "" if graphic else f'<div class="footer">{escape(source)}</div>'
        html = f'<section class="slide {"graphic" if graphic else ""}" id="slide-{page}" aria-label="{escape(title)}">{body}{footer}<div class="page">{page:02d}</div><aside class="speaker">{escape(notes)}</aside></section>'
        slides.append({"title": title, "html": html, "notes": notes})

    add("Model-based control under observation noise",
        f"""
      <div class="kicker">Continual learning · September 8, 2026</div>
      <h1>Model-based control<br>under observation noise</h1>
      <p class="lead">Domino recovers at 1 cm.<br>Boil succeeds with pose and reading noise.</p>
      <div class="row">
        <div class="col card tealcard"><h3>Domino · 1 cm pose noise</h3><div class="big">{number(headlines['Domino']['solve_rate_pct'])}%</div><p>levels solved with MB features</p><p class="small">Earlier MB: 50% · MF: 50%<br>Successful MB runs: {number(headlines['Domino']['mean_steps_successful'])} mean steps</p></div>
        <div class="col card tealcard"><h3>Boil · pose + reading noise</h3><div class="big">{number(headlines['Boil']['solve_rate_pct'])}%</div><p>levels solved with MB features</p><p class="small">MF: 25% · no matched MB-without run<br>Successful MB runs: {number(headlines['Boil']['mean_steps_successful'])} mean steps</p></div>
      </div>
      <div class="cover-bottom">Two seeds per noisy condition. All selected scorecards are final.</div>
    """,
        source=
        f"Result snapshot: {payload['observed_at']}. These are package comparisons, not feature ablations.",
        notes=
        "Each run has one train level and one test level. The 100% values mean both seeds solved both levels. Steps are averaged only over successful seeds. Earlier MB code predates the feature package, so this does not attribute the recovery to a particular feature."
        )

    add("Experimental setup and metric definitions",
        """
      <div class="kicker">How to read the results</div><h2>The noise channel is shared; the new belief tools are MB-only</h2>
      <div class="row">
        <div class="col card"><h3>Same observation channel</h3><p>Gaussian noise on object pose and, for the combined boil point, scalar readings.</p><p class="small">Sigmas are declared to both arms. Repeated observation at the same step returns the same draw.</p><p class="small">MF stays a plain code agent and may write its own smoothing.</p></div>
        <div class="col card tealcard"><h3>What we measure</h3><p><b>Solve rate:</b> fraction of train + test levels solved, over all seeds.</p><p><b>Steps:</b> whole-run total, averaged only over seeds solving both levels.</p><p><b>Resets:</b> mean across all seeds.</p></div>
      </div>
      <div class="rule">No successful seed means <b>N/A steps</b>, never zero steps.</div>
      <p class="small muted" style="margin-top:16px">Noiseless references: seed 0 only. Noisy conditions: seeds 0 and 1.<br>Skills and evaluators retain true state; this tests the agent's observation and decision channel.</p>
    """,
        source=
        "Sources: observation_noise.py; run/continual.py; uncertainty-results/make_figures.py."
        )

    chart_slides = [
        ("Domino solve rate", "domino_solve_rate",
         "At 1 cm, the earlier MB pair and MF pair each solved only the train level. With all new features enabled, both MB seeds solved train and test. At 2 cm, the feature-enabled arm has not been run."
         ),
        ("Domino steps on successful seeds", "domino_steps",
         "MF has no fully successful seed in the displayed domino settings. Earlier MB has successful seeds only at no noise and 0.5 cm. The new MB 1 cm mean is 376.5 steps across two seeds. The noiseless reference is a single seed at 297; matched seed 0 is 436 versus 297."
         ),
        ("Boil solve rate", "boil_solve_rate",
         "With pose plus scalar-reading noise, MB solves every level and MF solves one of four levels across the two seeds. The combined point is separate from the pose sweep. Both noiseless references used unsanitized observations and are not clean controls."
         ),
        ("Boil steps on successful seeds", "boil_steps",
         "MB succeeds in 728 and 532 steps, averaging 630. Combined-noise MF has no fully successful seed, so there is no step estimate. Boil already tolerated pose noise without the new feature package."
         ),
        ("Domino resets", "domino_resets",
         "Every displayed domino run has zero resets, including the failed runs. Zero resets therefore does not imply success."
         ),
        ("Boil resets", "boil_resets",
         "Resets are averaged over all seeds. Combined-noise MB averages zero; MF averages five. This is a cost of unsuccessful runs as well as successful runs."
         ),
    ]
    for title, filename, notes in chart_slides:
        add(title,
            f'<img alt="{escape(title)}" src="{image_uri(filename)}">',
            notes=notes,
            graphic=True)

    cards = []
    for index, (name, _, description) in enumerate(FEATURES, 1):
        cards.append(
            f'<div class="feature"><div class="index">{index:02d}</div><h3>{name}</h3><p>{description}</p></div>'
        )
    add("Six MB features enabled together",
        f"""
      <div class="kicker">What changed for MB</div><h2>Six flags, enabled together in the validation runs</h2>
      <div class="feature-row">{''.join(cards[:3])}</div><div class="feature-row">{''.join(cards[3:])}</div>
      <div class="rule">The agent still writes the model and chooses actions.<br>The added machinery exposes uncertainty to those decisions.</div>
    """,
        source=
        "All six flags default off. The domino v2 and boil p12_r07 MB configs enable them; MF does not receive the harness belief frame."
        )

    add("Parameter intervals replace a binary rejection",
        """
      <div class="kicker">1 · Parameter intervals / 2 · Noise-aware probes</div><h2>A wide, informative fit should not fall back to an implausible anchor</h2>
      <svg class="diagram" viewBox="0 0 1152 228" xmlns="http://www.w3.org/2000/svg">
        <text x="0" y="28" font-family="DejaVu Sans" font-size="18" fill="#526475">Illustration from the earlier domino friction fit</text>
        <line x1="75" y1="112" x2="1100" y2="112" stroke="#bccbd0" stroke-width="3"/>
        <rect x="357" y="96" width="641" height="32" fill="#d5eee7"/>
        <line x1="203" y1="82" x2="203" y2="145" stroke="#718392" stroke-width="3"/>
        <circle cx="588" cy="112" r="10" fill="#008577"/>
        <line x1="716" y1="82" x2="716" y2="145" stroke="#d77831" stroke-width="3"/>
        <text x="357" y="80" font-family="DejaVu Sans" font-size="19" fill="#008577">Belief interval: about 0.22 to 0.72</text>
        <text x="203" y="180" text-anchor="middle" font-family="DejaVu Sans" font-size="19" fill="#526475">Anchor 0.10</text>
        <text x="588" y="180" text-anchor="middle" font-family="DejaVu Sans" font-size="19" fill="#008577">MAP 0.40</text>
        <text x="746" y="210" text-anchor="middle" font-family="DejaVu Sans" font-size="18" fill="#d77831">Truth 0.50 (analysis only)</text>
      </svg>
      <div class="row" style="margin-top:0">
        <div class="col card"><h3>Retain the estimate</h3><p class="small">A moved, informative but wide posterior gets the WIDE verdict and can deploy its MAP.</p></div>
        <div class="col card tealcard"><h3>Test the interval</h3><p class="small">Plan sweeps report the fraction of parameter settings that pass and where outcomes change.</p></div>
        <div class="col card"><h3>Probe when it matters</h3><p class="small">A mixed sweep can trigger information seeking. Probe value accounts for noisy readings.</p></div>
      </div>
    """,
        source=
        "Sources: identifiability.py; grid_seed.py; belief_probe.py; continual-uncertainty.md §3.7 and §8.",
        notes=
        "The interval illustration is from the earlier failure analysis, not a fitted-parameter trace from the new winning runs. A parameter that did not move or remains wider than its prior still falls back to the anchor. Noise-aware probe scoring uses mutual information over eight noisy reads per ensemble member, not raw prediction spread."
        )

    add("Filtering, carried centers and evidence",
        """
      <div class="kicker">3 · Fit-side filtering / 4 · Carried center / 5 · Evidence</div><h2>Improve what the fit sees and what survives the next fit</h2>
      <div class="row">
        <div class="col card tealcard"><h3>Fit-side filtering</h3><p>Detect motion from window means relative to the declared sigma.</p><p>Start a rest-anchored rollout from an averaged frame.</p><div class="equation">spread ≈ σ / √n</div><p class="small">This reduces initial-state noise while the object is at rest.</p></div>
        <div class="col card"><h3>Carry the prior center</h3><p>The last applied MAP becomes the next fit's prior center.</p><p><b>The width is not carried.</b></p><p class="small">Fits already pool past levels' data; carrying its posterior width would count those data twice.</p></div>
        <div class="col card"><h3>Compare by evidence</h3><p>Report Laplace log evidence from the fit's residuals and local curvature.</p><p>Include a complexity penalty and comparable version deltas.</p><p class="small">This informs the agent's choice; it is not an automatic model-replacement rule.</p></div>
      </div>
    """,
        source=
        "Sources: trajectory_prep.py; agent_sim_learning_approach.py; synthesis_backend.py; evidence.py.",
        notes=
        "Angles use circular means. Rest detection compares windows using their standard errors, with the configured settle tolerance as a floor. Evidence may be unavailable when the fit lacks suitable curvature. Version comparisons are reported only when the implementation considers the scored residual data comparable."
        )

    add("Execution belief and the tool surface",
        """
      <div class="kicker">6 · Execution belief / tools available to the agent</div><h2>Expose spread, test sensitivity, then act</h2>
      <div class="row execution">
        <div class="col card tealcard"><h3>Execution-time belief</h3><p>Average each object's noisy features over a recent window consistent with rest.</p><p>Shorten the window when its two half-means disagree beyond the noise threshold.</p><p>Expose the mean, spread and atom fractions beside the raw observation.</p><p class="small">Slow motion can pass this heuristic and cause lag. Expected-atom checks use fractions over belief draws.</p></div>
        <div class="col" style="flex:1.15">
          <div class="tool"><span class="code">sim.belief()</span><p>Read the current smoothed features, spreads and atom fractions.</p></div>
          <div class="tool"><span class="code">sim.run(..., belief_draws=K)</span><p>Vary the starting state to test sensitivity to observation uncertainty.</p></div>
          <div class="tool"><span class="code">sim.run(..., physics_sweep=True)</span><p>Sweep the learned parameter intervals.</p></div>
          <div class="tool"><span class="code">evaluate_trajectory(...,<br> physics_sweep=True)</span><p>Score the recorded sequence across parameter settings.</p></div>
        </div>
      </div>
      <div class="rule small">Not built: a full particle filter or observation-belief draws in <span class="code">evaluate_trajectory</span>.<br>Atom fractions live in the observation and <span class="code">sim.belief()</span>, not <span class="code">sim.predicates()</span>.</div>
    """,
        source=
        "Sources: observation_belief.py; run/continual.py; agent_sdk/belief_probe.py.",
        notes=
        "The execution belief is a rest-window smoother, not a particle filter tracking moving latent states. The scalar-reading channel is shared by both arms and folds into the same sigma-aware fit and belief calculations. Physics sweeps and start-state belief draws are separate tool modes."
        )

    add("What is established and what remains open",
        """
      <div class="kicker">Interpretation</div><h2>Recovery is observed; the contribution of each feature is still open</h2>
      <div class="row">
        <div class="col card tealcard"><h3>What the runs show</h3><p><b>Domino at 1 cm:</b> both MB seeds now solve both levels; the earlier MB pair did not.</p><p><b>Boil, combined noise:</b> MB solves all four levels across seeds, with zero resets; MF solves one.</p><p class="small">All four new MB runs finish successfully.</p></div>
        <div class="col card"><h3>What they do not isolate</h3><p>Only two noisy seeds and one noiseless reference seed per arm.</p><p>Earlier MB runs use older code; individual feature ablations are still absent.</p><p>Boil has no matched feature-off run with reading noise, and its old noiseless references exposed hidden heat.</p></div>
      </div>
      <div class="rule">To support stronger claims: clean noiseless boil controls, matched feature ablations, and more seeds.</div>
    """,
        source=
        "Sources: result snapshot and validation scope in continual-uncertainty.md."
        )

    flag_rows = "".join(
        f'<tr><td>{escape(name)}</td><td class="code">{escape(flag)}</td></tr>'
        for name, flag, _ in FEATURES)
    add("Appendix: exact flags",
        f"""
      <div class="kicker">Appendix · reproducibility</div><h2>The six MB feature flags</h2>
      <table class="flags"><thead><tr><th>Feature</th><th>Flag enabled in the validation MB configs</th></tr></thead><tbody>{flag_rows}</tbody></table>
      <p class="small" style="margin-top:22px">All six default off.<br>Shared channel: <span class="code">continual_obs_noise_position</span>, <span class="code">continual_obs_noise_orientation</span> and <span class="code">continual_obs_noise_scalar</span>.</p>
    """,
        source=
        "Configs: protocol_continual_noise_domino_10mm_v2.yaml and protocol_continual_noise_boil_p12_r07.yaml."
        )

    table_rows = []
    for row in summaries:
        noise = row["noise"].replace("Pose ", "").replace(" / ", "/").replace(
            " + reading ", " + read ")
        arm = "MB + features" if row["arm"] == "MB + uncertainty" else row[
            "arm"]
        if row["historical_unsanitized"]:
            arm += " †"
        classes = []
        if row["arm"] == "MB + uncertainty":
            classes.append("highlight")
        if row["historical_unsanitized"]:
            classes.append("historical")
        if row["domain"] == "Boil" and row["noise"] == "No noise" and row[
                "arm"] == "MB":
            classes.append("group")
        cells = [
            row["domain"], noise, arm,
            str(row["seeds"]),
            str(row["successful_seeds"]),
            number(row["solve_rate_pct"]) + "%",
            number(row["mean_steps_successful"]),
            number(row["mean_resets"])
        ]
        table_rows.append('<tr class="' + ' '.join(classes) + '">' + ''.join(
            f'<td class="{"num" if i > 2 else ""}">{escape(value)}</td>'
            for i, value in enumerate(cells)) + '</tr>')
    add("Appendix: full numerical results",
        f"""
      <div class="kicker">Appendix · all displayed settings</div><h2>Seed means and successful-run counts</h2>
      <table class="results"><thead><tr><th>Env</th><th>Noise σ</th><th>Arm</th><th>n</th><th>Success n</th><th>Solve</th><th>Steps¹</th><th>Resets</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
      <p class="small muted" style="font-size:14px;margin-top:13px">¹ Steps use only seeds solving both levels; N/A = zero successful seeds. Solve rate and resets use all seeds.<br>† Unsanitized historical boil reference. Position σ is in cm; orientation σ is in radians.</p>
    """,
        source=
        f"Source: uncertainty-results/snapshot.json, captured {payload['observed_at']}; 17 conditions / 30 runs."
        )

    html = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>MB under observation noise</title><style>' + CSS + '</style></head><body><main id="stage">' + ''.join(
        slide["html"] for slide in slides
    ) + '</main><nav class="nav" aria-label="Slide navigation"><button id="previous" aria-label="Previous slide">← Previous</button><span id="counter" aria-live="polite"></span><button id="next" aria-label="Next slide">Next →</button><button id="fullscreen">Fullscreen</button></nav><script>' + JS + '</script></body></html>'
    assert 'src="http' not in html and 'href="http' not in html
    assert "—" not in html
    STEM.with_suffix(".html").write_text(html)
    notes = [
        "# MB under observation noise", "",
        "Generated by `make_uncertainty_results_slides.py`; do not edit generated outputs manually.",
        "",
        "Open `uncertainty_results_slides.html` in a browser and use the arrow keys or navigation buttons.",
        "The HTML embeds all charts and works offline.",
        "The PDF is a fixed 16:9 export.", "",
        "Steps average only completed seeds that solve both levels; solve rate and resets use all seeds.",
        "", "The PDF renderer requires WeasyPrint and PyMuPDF.",
        "On Engaging, the command below uses the matching C++ runtime so it also works on older compute nodes.",
        "", "Regenerate on an Engaging compute node:", "", "```bash",
        "LD_PRELOAD=/orcd/software/core/001/pkg/miniforge/25.11.0-0/lib/libstdc++.so.6 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /orcd/software/core/001/pkg/miniforge/25.11.0-0/bin/python docs/slides/make_uncertainty_results_slides.py --pdf",
        "```", "", "Slide outline and presenter notes:", ""
    ]
    for i, slide in enumerate(slides, 1):
        notes.extend([f"## {i}. {slide['title']}", ""])
        if slide["notes"]:
            notes.extend([slide["notes"].replace(". ", ".\n"), ""])
    STEM.with_suffix(".md").write_text("\n".join(notes).rstrip() + "\n")
    print(f"Built {len(slides)} slides: {STEM.with_suffix('.html')}")
    return len(slides)


def render_pdf(count):
    """Render the printable slides and check page count and page bounds."""
    import pymupdf as fitz
    from weasyprint import HTML

    HTML(filename=str(STEM.with_suffix(".html"))).write_pdf(
        str(STEM.with_suffix(".pdf")))
    doc = fitz.open(STEM.with_suffix(".pdf"))
    preview = HERE.parent.parent / "logs" / "uncertainty_handoff_20260908" / "slide-preview"
    preview.mkdir(parents=True, exist_ok=True)
    for i, page in enumerate(doc):
        assert abs(page.rect.width - 960) < 1 and abs(page.rect.height -
                                                      540) < 1
        for block in page.get_text("blocks"):
            x0, y0, x1, y1 = block[:4]
            assert x0 >= -1 and y0 >= -1 and x1 <= 961 and y1 <= 541, (i + 1,
                                                                       block)
        page.get_pixmap(matrix=fitz.Matrix(1.1, 1.1)).save(
            preview / f"slide-{i+1:02d}.png")
    assert len(doc) == count, (len(doc), count)
    print(
        f"Rendered and checked {count} PDF pages: {STEM.with_suffix('.pdf')}")


def main():
    """Build the deck, optionally including its PDF and preview pages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()
    count = build()
    if args.pdf:
        render_pdf(count)


if __name__ == "__main__":
    main()
