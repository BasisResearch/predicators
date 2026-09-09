"""Build both balloons decks from the saved September 7 uniform-sweep snapshot.

No experiment imports or simulations. Images are embedded so either HTML
file can be shared alone. Optional PDF rendering uses WeasyPrint on a
compute node.
"""

import argparse
import base64
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "envs" / "balloons" / "sweep_20260907"

CSS = """
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; color: #182c3b; font-family: 'DejaVu Sans', sans-serif; }
body { background: #172a39; }
.slide { position: relative; width: 1280px; height: 720px; padding: 45px 62px 52px; background: white; overflow: hidden; }
.kicker { color: #008577; font-size: 14px; font-weight: bold; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 16px; }
h1 { font-size: 50px; line-height: 1.12; margin: 12px 0 24px; }
h2 { font-size: 35px; line-height: 1.15; margin: 0 0 25px; }
h3 { font-size: 23px; margin: 0 0 15px; }
p, li { font-size: 22px; line-height: 1.4; }
p { margin: 0 0 19px; }
ul, ol { margin: 0; padding-left: 27px; }
li { margin-bottom: 14px; }
.row { display: flex; }
.col { flex: 1; min-width: 0; margin-right: 30px; }
.col:last-child { margin-right: 0; }
.card { background: #f0f5f7; padding: 24px; border-top: 4px solid #008577; }
.card p, .card li { font-size: 21px; }
.small, .small li { font-size: 17px; line-height: 1.4; }
.muted { color: #526475; }
.rule { background: #edf7f3; border-left: 5px solid #008577; padding: 18px 22px; margin-top: 22px; }
.rule p { margin: 0; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: .83em; color: #2e6297; }
table { border-collapse: collapse; width: 100%; font-size: 20px; line-height: 1.35; }
td, th { padding: 12px 14px; text-align: left; border-bottom: 1px solid #dce4ea; }
th { background: #eaf0f4; color: #344b60; }
.compact { font-size: 18px; }
.compact td, .compact th { padding: 10px 12px; }
.equation { font-family: Georgia, serif; font-size: 30px; margin: 18px 0; }
.photo { width: 100%; height: 435px; object-fit: contain; }
figure { margin: 0; }
figcaption { font-size: 15px; color: #526475; margin-top: 9px; line-height: 1.35; }
.footer { position: absolute; left: 62px; right: 90px; bottom: 20px; font-size: 11px; color: #657483; }
.page { position: absolute; bottom: 20px; right: 28px; font-size: 12px; color: #657483; }
.metric { color: #008577; font-size: 51px; font-weight: bold; margin: 12px 0; }
.mb { color: #2864a6; } .mf { color: #b85e13; }
.diagram { width: 100%; height: 360px; }
a { color: #2864a6; }
.nav { position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%); background: #172a39; border: 1px solid #536777; color: white; border-radius: 8px; padding: 5px; display: flex; align-items: center; z-index: 10; }
.nav button { font: 14px 'DejaVu Sans', sans-serif; color: white; background: transparent; border: none; padding: 6px 16px; cursor: pointer; }
.nav span { font-size: 12px; min-width: 65px; text-align: center; }
.nav button:focus-visible { outline: 2px solid #61c6b5; }
@media screen { #stage { position: absolute; top: 50%; left: 50%; width: 1280px; height: 720px; transform-origin: center center; } .slide { display: none; } .slide.active { display: block; } }
@page { size: 1280px 720px; margin: 0; }
@media print { body { background: white; } #stage { position: static; transform: none !important; } .slide { display: block !important; page-break-after: always; } .slide:last-child { page-break-after: auto; } .nav { display: none; } }
"""

JS = """
const slides = Array.from(document.querySelectorAll('.slide'));
let index = 0;
function show(i) {
  index = Math.max(0, Math.min(slides.length - 1, i));
  slides.forEach((s, k) => s.classList.toggle('active', k === index));
  document.getElementById('count').textContent = (index + 1) + ' / ' + slides.length;
  history.replaceState(null, '', '#/' + index);
}
function resize() {
  const scale = Math.min(innerWidth / 1280, (innerHeight - 65) / 720);
  document.getElementById('stage').style.transform = 'translate(-50%, -50%) scale(' + scale + ')';
}
document.getElementById('prev').onclick = () => show(index - 1);
document.getElementById('next').onclick = () => show(index + 1);
document.addEventListener('keydown', e => {
  if (['ArrowRight', 'ArrowDown', 'PageDown', ' '].includes(e.key)) { e.preventDefault(); show(index + 1); }
  if (['ArrowLeft', 'ArrowUp', 'PageUp'].includes(e.key)) { e.preventDefault(); show(index - 1); }
  if (e.key === 'Home') show(0);
  if (e.key === 'End') show(slides.length - 1);
});
addEventListener('resize', resize);
show(Number(location.hash.replace('#/', '')) || 0); resize();
"""


def image(name, alt):
    data = base64.b64encode((ASSETS / name).read_bytes()).decode()
    return f'<img class="photo" src="data:image/png;base64,{data}" alt="{alt}">'


def build():
    snapshot = json.loads((ASSETS / "manifest.json").read_text())
    slides = []

    def add(kicker, title, body, source):
        slides.append(f'<section class="slide" aria-label="{title}">'
                      f'<div class="kicker">{kicker}</div><h2>{title}</h2>'
                      f'{body}<div class="footer">{source}</div>'
                      f'<div class="page">{len(slides) + 1}</div></section>')

    start = image(
        "test_start.png",
        "Sweep seed 0 test start: four balloons, an oak box and two chute walls"
    )
    won = image(
        "test_won.png",
        "Sweep seed 0 test win: gold balloon released and the box inside the chute at the target band"
    )

    add(
        "Balloons / uniform sweep / updated 8 September 2026",
        "The Balloons Domain", f'''
    <div class="row"><div class="col"><h1>Float a box<br>through a chute</h1>
    <p>Choose which balloons to release so the box reaches the green height band at low speed, without bursting a balloon.</p>
    <p class="muted">The sweep version combines hidden lift dynamics with tilt and wall contact.</p>
    <p class="small">September 7 uniform sweep<br>Seeds 0, 1, 2 · two train levels + one test level</p></div>
    <figure class="col">{start}<figcaption>Actual sweep frame: model-based seed 0, level 3.</figcaption></figure></div>
    ''',
        'Version: pybullet_balloons, contact-decoy generation; experiment keys balloons-agent_continual_uniform and balloons-agent_continual_model_free_uniform.'
    )

    add(
        "Version used in the sweep", "What changed from the original puzzle",
        '''
    <div class="row"><div class="col card"><h3>Transient motion</h3>
    <p>Drag is <b>2.2</b>, down from 12. The rising box can overshoot its predicted rest height.</p>
    <p>A balloon can burst at the ceiling even when the predicted hover height is inside the band.</p></div>
    <div class="col card"><h3>Tilt and contact</h3>
    <p>Balloon attachments are spread across the box top by clip index.</p>
    <p>Off-centre pulls can tilt or sway the box into the <b>two chute walls</b>.</p></div>
    <div class="col card"><h3>A jamming alternative</h3>
    <p>The test generator requires a subset that looks good by hover height but fails to settle in the band without bursting.</p>
    <p>The intended discriminator is <b>contact during ascent</b>.</p></div></div>
    <div class="rule"><p>Predicted hover height is a useful filter. It does not certify that the ascent will succeed.</p></div>
    ''',
        'Source: settings.py balloons_drag / balloons_require_jam_decoy; pybullet_balloons_base.py chute and attachment geometry.'
    )

    add(
        "The scene", "The box must pass between two walls", '''
    <div class="row"><div class="col">
    <svg class="diagram" viewBox="0 0 540 360" role="img" aria-label="Schematic of a box rising through a narrow chute, with a green target band and a tilted box contacting a wall">
    <rect x="35" y="325" width="470" height="15" fill="#d2b08b"/>
    <rect x="155" y="45" width="24" height="258" fill="#818392"/>
    <rect x="335" y="45" width="24" height="258" fill="#818392"/>
    <rect x="180" y="95" width="154" height="40" fill="#d2eedf"/>
    <text x="375" y="120" font-size="18" fill="#008577">target band</text>
    <rect x="207" y="260" width="100" height="62" fill="#a47944"/>
    <path d="M257 250 L257 155" stroke="#008577" stroke-width="4" stroke-dasharray="8 6"/>
    <path d="M247 169 L257 155 L267 169" fill="none" stroke="#008577" stroke-width="4"/>
    <g transform="rotate(27 283 203)"><rect x="233" y="173" width="100" height="62" fill="#a47944" opacity=".75"/></g>
    <circle cx="335" cy="196" r="9" fill="#c66d26"/>
    <path d="M344 196 L382 249" stroke="#c66d26" stroke-width="2"/>
    <text x="378" y="274" font-size="18" fill="#a65418">tilt / contact</text>
    <text x="32" y="22" font-size="17" fill="#526475">Mechanism schematic, not a recorded trajectory</text>
    </svg></div><div class="col"><ul>
    <li>The slot is <b>9.6 cm wide</b>, from z = 0.50 to 1.00 m.</li>
    <li>Attachment offsets span <b>−2.8 to +2.8 cm</b>, ordered by clip index.</li>
    <li>Walls collide with <b>the box only</b>. The arm and balloons pass through them.</li>
    <li>Balanced net pull alone is not a guarantee: the full motion and contacts determine whether the box clears the slot.</li>
    </ul></div></div>
    ''',
        'Source: pybullet_balloons_base.py, chute_half_gap=0.048, chute_z_lo/hi, attach_span=0.028 and collision filters.'
    )

    add(
        "Observation and control", "Same objects and skills, richer motion",
        '''
    <table class="compact"><tr><th>Object</th><th>Observed features</th><th>Role</th></tr>
    <tr><td>Box</td><td><code>x, y, z, color, speed</code></td><td>Material identity, position and linear speed; box orientation is absent from this feature vector.</td></tr>
    <tr><td>Balloon</td><td><code>x, y, z, color, tied, popped</code></td><td>Gas identity; released/attached status; burst status.</td></tr>
    <tr><td>Clip</td><td><code>x, y, z, rot, is_on</code></td><td>Clip i releases balloon i; an open clip stays open.</td></tr>
    <tr><td>Band</td><td><code>x, y, lo, hi</code></td><td>Acceptable box-centre heights, 5 cm apart.</td></tr>
    <tr><td>Robot</td><td><code>x, y, z, fingers, roll, tilt, wrist</code></td><td>End-effector and gripper state.</td></tr></table>
    <div class="row" style="margin-top:24px"><div class="col"><h3>Release(robot, clip)</h3><p class="small">Push the toggle open. Parameters set approach distance and contact height. A freed balloon cannot be clipped back.</p></div>
    <div class="col"><h3>Wait(robot)</h3><p class="small">Hold still while the physics evolves. A jam is not automatically a terminal loss; it can leave the task unsolved.</p></div></div>
    ''',
        'Source: pybullet_balloons_base.py object types; ground_truth_models/balloons/options.py; sweep observations are noiseless.'
    )

    add(
        "Hidden dynamics", "Height is only the free-ascent prediction", '''
    <div class="equation">L<sub>c</sub>(z) = L<sub>c</sub> · max(0, 1 − (z − z<sub>table</sub>) / h)</div>
    <div class="equation">z* = z<sub>table</sub> + h · (1 − W / Σ L<sub>c</sub>)</div>
    <p class="small muted">W includes the box and freed balloons. The second expression assumes an unconstrained equilibrium; it does not model overshoot or wall contact.</p>
    <table class="compact"><tr><th>Quantity</th><th>Sweep value</th><th>Meaning</th></tr>
    <tr><td>Red / blue / green / gold lift</td><td>0.35 / 0.50 / 0.70 / 1.00 N</td><td>Per-colour lift at table height</td></tr>
    <tr><td>Fade height</td><td>0.80 m</td><td>Lift declines with height</td></tr>
    <tr><td>Pine / oak box mass</td><td>0.05 / 0.08 kg</td><td>Both materials occur in the sweep</td></tr>
    <tr><td>Balloon mass / air drag</td><td>0.005 kg / 2.2</td><td>Freed mass and transient damping</td></tr>
    <tr><td>Ceiling / settle threshold</td><td>z = 1.18 m / speed &lt; 0.01 m/s</td><td>Burst boundary and the evaluator's low-speed check</td></tr></table>
    ''',
        'True constants shown for explanation, not supplied as answers to the agent. Source: settings.py and pybullet_balloons.py hover_height().'
    )

    add(
        "Task generation", "Test levels include a near-height jamming decoy",
        '''
    <ol>
    <li>Draw a rack and box material; enumerate balloon subsets and their unconstrained hover heights.</li>
    <li>Choose a 5 cm band within the chute and simulate the in-band candidates. Keep a level with <b>one safe subset</b> under this check.</li>
    <li>For test, require the candidate <b>nearest the band centre</b> to fail to settle in-band <b>without bursting</b>: the jamming decoy.</li>
    <li>Its predicted hover height must be within <b>2 cm</b> of the safe subset's. Check that the oracle's actual release plan succeeds.</li>
    </ol>
    <div class="rule"><p>Height alone can favour the wrong subset. Release order and the path through the chute still matter during execution.</p></div>
    <p class="small muted" style="margin-top:18px">The generation probe opens the candidate clips and rolls forward for up to 400 steps. A separate oracle check validates a sequence of Release / Wait skills.</p>
    ''',
        'Source: pybullet_balloons.py _make_tasks() / subset_outcome(); balloons_require_jam_decoy=True, contact_height_tol=0.02, max_sampling_attempts=80.'
    )

    add(
        "Worked sweep level / seed 0 / test",
        "Two near-identical height predictions", '''
    <p>Oak box; rack order: green, red, blue, gold.<br>Recorded target band: <b>[0.5023, 0.5523] m</b>.</p>
    <table><tr><th>Subset</th><th>Total lift</th><th>Analytic z*</th><th>Interpretation</th></tr>
    <tr><td>Gold (clip 3)</td><td>1.00 N</td><td>0.5329 m</td><td>In band; the recorded agent released it and won.</td></tr>
    <tr><td>Green + red (clips 0, 1)</td><td>1.05 N</td><td>0.5273 m</td><td>Also in band, closer to its centre; height cannot certify it.</td></tr></table>
    <div class="rule"><p>The predictions differ by only <b>5.6 mm</b>. The old isolated-hover-height example no longer describes this task.</p></div>
    <p class="small muted" style="margin-top:24px">The successful real sequence was Release(clip3), 41 steps, then Wait, 7 steps. The alternative's height above is calculated from the source law, not a newly replayed counterfactual.</p>
    ''',
        'Source: uniform MB seed 0, run_20260907_102820, L03/index.jsonl and level observation; analytic values use the true source constants.'
    )

    add(
        "Recorded sweep frames", "Gold alone solves this test level", f'''
    <div class="row"><figure class="col">{start}<figcaption>Before: four clipped balloons, box below the chute.</figcaption></figure>
    <figure class="col">{won}<figcaption>After: gold released; the evaluator reports WIN at test step 48.</figcaption></figure></div>
    <p class="small muted" style="margin-top:20px">The chute partially hides the box and green band in this camera view. These are preserved experiment frames, not a replay on newer code.</p>
    ''',
        'Source: balloons-agent_continual_uniform/seed0/run_20260907_102820/L03/renders/{ep000_start,ep000_win}.png.'
    )

    add(
        "Train / test and continual protocol",
        "Three levels per seed, one continuing agent", '''
    <table><tr><th></th><th>Training</th><th>Test</th></tr>
    <tr><td>Levels per seed</td><td>2</td><td>1</td></tr>
    <tr><td>Balloons per level</td><td>2 or 3</td><td>4, the full palette</td></tr>
    <tr><td>Box material</td><td>Pine or oak</td><td>Pine or oak</td></tr>
    <tr><td>Resets</td><td>Allowed and counted</td><td>Unavailable</td></tr></table>
    <ul style="margin-top:24px"><li>Training draws prefer colours and materials not yet shown; generation can fall back to a free draw.</li>
    <li>The journal, learned model and recorded data carry forward to each new level. Test composes familiar identities in a new rack.</li>
    <li>The sweep uses seeds 0–2 and noiseless observations. Report solve rate, total real steps and agent resets over <b>all train + test levels</b>.</li></ul>
    ''',
        'Source: protocol_continual_uniform_sweep.yaml and saved scorecards; 5,000 steps per level pooled into a 15,000-step run cap, 48-hour wall-clock cap.'
    )

    add(
        "The two agent arms", "What the simulator adds", '''
    <div class="row"><div class="col card"><h3 class="mb">Model-based (MB)</h3><ul>
    <li>Uses a visible base simulator with the scene, chute geometry and rigid-body contacts.</li>
    <li>Writes residual rules for release, attachment, lift and bursting; fits uncertain physical parameters.</li>
    <li>Can test release plans in its learned simulator before paying real steps.</li></ul></div>
    <div class="col card"><h3 class="mf">Model-free baseline (MF)</h3><ul>
    <li>Has the same real environment, skill tools, observations, journal and recorded experience.</li>
    <li>Has no supplied simulator or fit-and-rollout workbench.</li>
    <li>Still has arbitrary Python: it can fit equations and reason about dynamics by hand.</li></ul></div></div>
    <div class="rule"><p>Contact prediction is the intended benefit. The baseline's name does not mean it must guess or cannot build an analytic model.</p></div>
    ''',
        'Source: uniform-sweep arm configuration and balloons agent tool contract. Sandbox computation does not count as real environment steps.'
    )

    rows = []
    for arm in ("MB", "MF"):
        values = [r for r in snapshot["runs"] if r["arm"] == arm]
        for r in values:
            rows.append(
                f'<tr><td class="{arm.lower()}">{arm}</td><td>{r["seed"]}</td>'
                f'<td>{r["won"]} / {r["levels"]}</td><td>{r["steps"]:,}</td><td>{r["resets"]}</td></tr>'
            )
        n = len(values)
        solve = sum(r["won"] / r["levels"] for r in values) / n * 100
        steps = sum(r["steps"] for r in values) / n
        resets = sum(r["resets"] for r in values) / n
        rows.append(
            f'<tr style="background:#edf7f3;font-weight:bold"><td>{arm}</td><td>Mean</td>'
            f'<td>{solve:.1f}%</td><td>{steps:,.1f}</td><td>{resets:.2f}</td></tr>'
        )
    add(
        "Completed uniform sweep",
        "Balloons: a modest gap in these three seeds", '''
    <table class="compact"><tr><th>Arm</th><th>Seed</th><th>Levels solved</th><th>Total real steps</th><th>Resets</th></tr>
    ''' + "\n".join(rows) + '''</table>
    <p class="small" style="margin-top:20px">MB: <b>9/9</b> levels; MF: <b>8/9</b>. Mean step counts are close (482 vs 496). These runs do not show a large balloons step-efficiency advantage.</p>
    <p class="small muted">Means cover all three levels, including unfinished levels in completed unsuccessful runs. Steps exclude sandbox rollouts; lower counts can reflect stopping before solving.</p>
    ''',
        'Source: six completed *_uniform scorecards pinned in sweep_20260907/manifest.json. These are the uniform sweep, not the earlier contact A/B results.'
    )

    add(
        "Sources and reproducibility",
        "The deck now describes the sweep version", '''
    <table class="compact"><tr><th>Source</th><th>What to inspect</th></tr>
    <tr><td><code>predicators/envs/pybullet_balloons_base.py</code></td><td>Scene, chute collision filters, attachment offsets and observed features.</td></tr>
    <tr><td><code>predicators/envs/pybullet_balloons.py</code></td><td>Lift, attachment, burst, task generation, oracle checks and evaluator.</td></tr>
    <tr><td><code>predicators/settings.py</code></td><td>Drag 2.2; contact-decoy default; 2 cm tolerance; training palette.</td></tr>
    <tr><td><code>protocol_continual_uniform_sweep.yaml</code></td><td>Two training levels, common MB/MF flags and noiseless protocol.</td></tr>
    <tr><td><code>docs/envs/balloons/sweep_20260907/</code></td><td>Saved screenshots, source manifest, per-seed results and hashes.</td></tr>
    <tr><td><code>docs/slides/make_balloons_domain_slides.py</code></td><td>Rebuilds both standalone HTML decks; --pdf also renders the PDF.</td></tr></table>
    <div class="rule"><p>The older September 6 oracle video and stills show the original environment. They are retained as historical assets and are not used in this deck.</p></div>
    ''',
        'Updated 2026-09-08. HTML copies: docs/slides/balloons_domain_slides.html and docs/envs/balloons/overview_slides.html.'
    )

    def embed_svg(match):
        svg = match.group(0).replace(
            '<svg ', '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
        data = base64.b64encode(svg.encode()).decode()
        return (
            '<img class="diagram" alt="Schematic: a box tilts while rising between chute walls toward the green band" '
            f'src="data:image/svg+xml;base64,{data}">')

    content = re.sub(r'<svg\b.*?</svg>',
                     embed_svg,
                     '\n'.join(slides),
                     flags=re.S)
    html = (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>The Balloons Domain - Uniform Sweep</title><style>' + CSS +
        '</style></head>\n'
        '<body><main id="stage">\n' + content + '</main>\n'
        '<nav class="nav" aria-label="Slide navigation"><button id="prev" aria-label="Previous slide">←</button>'
        '<span id="count" aria-live="polite"></span><button id="next" aria-label="Next slide">→</button></nav>'
        '<script>' + JS + '</script></body></html>\n')
    (HERE / "balloons_domain_slides.html").write_text(html)
    (ASSETS.parent / "overview_slides.html").write_text(html)
    return len(slides)


def render_pdf(count):
    import pymupdf
    from weasyprint import HTML

    output = HERE / "balloons_domain_slides.pdf"
    HTML(filename=str(HERE / "balloons_domain_slides.html")).write_pdf(
        str(output))
    doc = pymupdf.open(output)
    assert len(doc) == count, (len(doc), count)
    preview = HERE.parent.parent / "logs" / "analysis" / "balloons_slides_20260908"
    preview.mkdir(parents=True, exist_ok=True)
    for i, page in enumerate(doc):
        assert abs(page.rect.width - 960) < 1 and abs(page.rect.height -
                                                      540) < 1
        for block in page.get_text("blocks"):
            x0, y0, x1, y1 = block[:4]
            assert x0 >= 0 and y0 >= 0 and x1 <= 960 and y1 <= 540, (i + 1,
                                                                     block)
        page.get_pixmap().save(preview / f"slide-{i + 1:02d}.png")
    print(f"Rendered and checked {count} pages: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", action="store_true")
    args = parser.parse_args()
    slide_count = build()
    print(f"Built both standalone HTML decks: {slide_count} slides")
    if args.pdf:
        render_pdf(slide_count)
