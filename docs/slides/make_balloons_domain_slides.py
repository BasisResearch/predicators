"""Build balloons decks from the saved sweep and audited hatch illustrations.

No experiment imports or simulations. Images and videos are embedded so
either HTML file can be shared alone. Optional PDF rendering uses
WeasyPrint on a compute node.
"""

import argparse
import base64
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSETS = HERE.parent / "envs" / "balloons" / "sweep_20260907"
VISUALS = ASSETS.parent / "visuals"

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
.demo-video, .video-print { width: 100%; height: 335px; object-fit: contain; background: #f0f5f7; }
.video-print { display: none; }
.play-pair { font: 16px 'DejaVu Sans', sans-serif; background: #008577; color: white; border: 0; padding: 10px 18px; border-radius: 5px; cursor: pointer; }
.play-pair:focus-visible { outline: 3px solid #2864a6; outline-offset: 3px; }
.wide-plot { width: 100%; height: 360px; object-fit: contain; }
a { color: #2864a6; }
.nav { position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%); background: #172a39; border: 1px solid #536777; color: white; border-radius: 8px; padding: 5px; display: flex; align-items: center; z-index: 10; }
.nav button { font: 14px 'DejaVu Sans', sans-serif; color: white; background: transparent; border: none; padding: 6px 16px; cursor: pointer; }
.nav span { font-size: 12px; min-width: 65px; text-align: center; }
.nav button:focus-visible { outline: 2px solid #61c6b5; }
@media screen { #stage { position: absolute; top: 50%; left: 50%; width: 1280px; height: 720px; transform-origin: center center; } .slide { display: none; } .slide.active { display: block; } }
@page { size: 1280px 720px; margin: 0; }
@media print { body { background: white; } #stage { position: static; transform: none !important; } .slide { display: block !important; page-break-after: always; } .slide:last-child { page-break-after: auto; } .nav { display: none; } }
@media print { video.demo-video, .play-pair { display: none; } .video-print { display: block; } }
"""

JS = """
const slides = Array.from(document.querySelectorAll('.slide'));
let index = 0;
function show(i) {
  index = Math.max(0, Math.min(slides.length - 1, i));
  slides.forEach((s, k) => {
    s.classList.toggle('active', k === index);
    if (k !== index) s.querySelectorAll('video').forEach(v => v.pause());
  });
  document.getElementById('count').textContent = (index + 1) + ' / ' + slides.length;
  history.replaceState(null, '', '#/' + index);
}
function resize() {
  const scale = Math.min(innerWidth / 1280, (innerHeight - 65) / 720);
  document.getElementById('stage').style.transform = 'translate(-50%, -50%) scale(' + scale + ')';
}
document.getElementById('prev').onclick = () => show(index - 1);
document.getElementById('next').onclick = () => show(index + 1);
document.getElementById('hatch').onclick = () => show(3);
document.querySelectorAll('.play-pair').forEach(button => {
  button.onclick = () => {
    button.closest('.slide').querySelectorAll('video').forEach(video => {
      video.currentTime = 0;
      video.play().catch(() => { button.textContent = "Use each video's play control"; });
    });
  };
});
document.addEventListener('keydown', e => {
  if (e.target.closest('video, button, input, select, textarea, a')) return;
  if (['ArrowRight', 'ArrowDown', 'PageDown', ' '].includes(e.key)) { e.preventDefault(); show(index + 1); }
  if (['ArrowLeft', 'ArrowUp', 'PageUp'].includes(e.key)) { e.preventDefault(); show(index - 1); }
  if (e.key === 'Home') show(0);
  if (e.key === 'End') show(slides.length - 1);
});
addEventListener('resize', resize);
addEventListener('hashchange', () => show(Number(location.hash.replace('#/', '')) || 0));
show(Number(location.hash.replace('#/', '')) || 0); resize();
"""


def image(name, alt):
    data = base64.b64encode((ASSETS / name).read_bytes()).decode()
    return f'<img class="photo" src="data:image/png;base64,{data}" alt="{alt}">'


def visual_image(name, alt, css="photo"):
    data = base64.b64encode((VISUALS / name).read_bytes()).decode()
    return f'<img class="{css}" src="data:image/png;base64,{data}" alt="{alt}">'


def video(name, alt):
    data = base64.b64encode((VISUALS / f"{name}.mp4").read_bytes()).decode()
    poster = base64.b64encode(
        (VISUALS / f"{name}-final.png").read_bytes()).decode()
    return (
        f'<video class="demo-video" controls muted playsinline preload="metadata" '
        f'aria-label="{alt}" poster="data:image/png;base64,{poster}">'
        f'<source src="data:video/mp4;base64,{data}" type="video/mp4">'
        'Open the accompanying MP4 to watch this sequence.</video>' +
        visual_image(f"{name}-final.png", alt, "video-print"))


def add_hatch_slides(add):
    """Explain the new geometry using audited physical demonstrations."""
    manifest = json.loads((VISUALS / "manifest.json").read_text())
    jam, passage = [row["outcome"] for row in manifest["runs"]]
    source = ('Mechanical illustrations: audited hatch seed 5, source '
              f'{manifest["source_commit"][:9]}; visuals/manifest.json. '
              'These are not MB, MF or continual-oracle results.')
    add(
        "Chute to hatch / new prototype", "The new hatch changes the path",
        f'''
    <div class="row"><figure class="col"><h3>Original chute</h3>
    {visual_image("chute-initial.png", "A small cube below the original vertical chute walls", "demo-video")}
    <figcaption>A 7 cm cube fits inside a 9.6 cm vertical slot.<br>Tilting or swaying can bring the box into a wall.</figcaption></figure>
    <figure class="col"><h3>New hatch</h3>
    {visual_image("hatch-initial.png", "A long payload below horizontal hatch panels and an elevated green band", "demo-video")}
    <figcaption>A 20 cm-long payload starts below a 17 cm opening.<br>It must pass the panels and reach the green band above.</figcaption></figure></div>
    <div class="rule"><p>Same Release and Wait skills. The hatch makes the box's orientation during ascent central to the task.</p></div>
    ''', source)
    add(
        "Geometry and collision rules",
        "Walls block the box; the ceiling bursts balloons", '''
    <div class="row"><div class="col">
    <svg class="diagram" viewBox="0 0 540 360" role="img" aria-label="Hatch cross-section: ceiling above balloons, green target above two panels, long box below a narrow opening">
    <rect x="30" y="20" width="480" height="14" fill="#b75b38"/>
    <text x="44" y="59" font-size="17" fill="#9e4527">Ceiling height: balloons burst here</text>
    <rect x="170" y="84" width="175" height="36" fill="#c4e6d3"/>
    <text x="359" y="108" font-size="17" fill="#008577">target band</text>
    <rect x="30" y="205" width="168" height="18" fill="#818392"/>
    <rect x="322" y="205" width="188" height="18" fill="#818392"/>
    <path d="M200 191 L320 191 M200 184 L200 198 M320 184 L320 198" stroke="#526475" stroke-width="2"/>
    <text x="213" y="180" font-size="18" fill="#526475">17 cm gap</text>
    <rect x="179" y="285" width="151" height="27" fill="#c9994f"/>
    <text x="168" y="343" font-size="18" fill="#526475">20 cm-long payload</text>
    <path d="M211 285 L211 258 M295 285 L295 258" stroke="#008577" stroke-width="3"/>
    <path d="M205 264 L211 256 L217 264 M289 264 L295 256 L301 264" fill="none" stroke="#008577" stroke-width="3"/>
    <text x="35" y="252" font-size="16" fill="#526475">panels at z = 0.57 m</text>
    </svg></div><div class="col"><ul>
    <li>Both chute walls and hatch panels collide with <b>the box only</b>.</li>
    <li>Balloons and the robot pass through these obstacles. Wall contact does <b>not</b> burst balloons.</li>
    <li>The hatch opening is offset by 1.2 cm. The payload is 20 × 7 × 3.6 cm.</li>
    <li>Off-centre balloon pulls rotate the box. Release order changes its path and contacts.</li>
    </ul></div></div>
    <p class="small muted">Cross-section schematic, not a trajectory. Balloon bursting is a separate ceiling-height rule.</p>
    ''',
        'Source: visible box-wall collision filters and ceiling burst rule in pybullet_balloons_base.py / pybullet_balloons.py.'
    )
    add(
        "Audited hatch example / seed 5",
        "Same two balloons, different release order", f'''
    <div class="row"><figure class="col"><h3 style="color:#b85e13">Red → gold: jam</h3>
    {video("hatch-jam", "Mechanical replay: red then gold jams below the hatch")}
    <figcaption>{jam["steps"]} primitive actions; box centre {jam["height"] * 100:.1f} cm.<br>Persistent support against the hatch; target not reached.</figcaption></figure>
    <figure class="col"><h3 style="color:#008577">Gold → red: passage</h3>
    {video("hatch-pass", "Mechanical replay: gold then red passes the hatch and wins")}
    <figcaption>{passage["steps"]} primitive actions; box centre {passage["height"] * 100:.1f} cm.<br>The evaluator reports a win inside the target band.</figcaption></figure></div>
    <div style="display:flex;align-items:center;gap:22px;margin-top:20px"><button class="play-pair">Replay both</button>
    <p class="small muted" style="margin:0">Playback: 16 primitive actions per second, plus start/end holds.<br>Both clips start from the same task state; no agent chooses these actions.</p></div>
    ''', source)
    add(
        "Measured motion", "The difference appears along the trajectory", f'''
    {visual_image("hatch-height.png", "Box-height curves: red then gold stays below the hatch; gold then red rises into the target band", "wide-plot")}
    <div class="row"><div class="col card"><p class="small"><b>Verified contact failure.</b> The audited red-then-gold sequence succeeds when box-panel collisions are disabled.</p></div>
    <div class="col card"><p class="small"><b>MB hypothesis.</b> A learned simulator could compare release sequences before an irreversible real action. An agent advantage remains unmeasured.</p></div></div>
    ''', source)
    add(
        "Noisy prototype / scope",
        "What stays the same, and what must be learned", '''
    <table class="compact"><tr><th></th><th>Original chute</th><th>Hatch prototype</th></tr>
    <tr><td>Objective</td><td>Hang at rest in a height band</td><td>Hang at rest above the hatch</td></tr>
    <tr><td>Controls</td><td>Release a clip; Wait</td><td>The same two skills</td></tr>
    <tr><td>Training / test</td><td>Two small racks / four-balloon rack</td><td>The same compositional split</td></tr>
    <tr><td>Hidden physics</td><td>Release, colour-dependent lift, mass, drag, burst</td><td>The same laws with a wider payload and new contacts</td></tr>
    <tr><td>Observed orientation</td><td>Absent from the historical box feature vector</td><td>Box and balloon roll, pitch and yaw</td></tr>
    <tr><td>Pilot noise</td><td>Historical sweep below was noiseless</td><td>1 cm position; 0.02 rad orientation</td></tr></table>
    <div class="rule"><p>Both arms receive the same hatch geometry and noisy observations. MB learns a simulator subclass; MF retains its free-coding baseline.</p></div>
    <p class="small muted" style="margin-top:16px">The clips illustrate physical states from mechanical validation. They are not recordings of noisy agent decisions or evidence of solve-rate improvement.</p>
    ''', source)


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
        "Balloons / chute and hatch / updated 9 September 2026",
        "The Balloons Domain", f'''
    <div class="row"><div class="col"><h1>Float a box<br>past obstacles</h1>
    <p>Choose which balloons to release so the box reaches the green height band at low speed, without bursting a balloon.</p>
    <p class="muted">The original chute and the new hatch combine hidden lift dynamics with tilt and contact.</p>
    <p class="small">New: hatch geometry and two mechanical replay videos.<br>Historical sweep results remain in a separate section.</p></div>
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
    <div class="col card"><h3>Contact challenge</h3>
    <p>The historical generator searched for alternatives with similar hover heights and different ascent outcomes.</p>
    <p>Its early stopping could misclassify an oscillation as a jam. Later audits check sustained contact and full release sequences.</p></div></div>
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

    add_hatch_slides(add)

    add(
        "Historical sweep / observation and control",
        "Same objects and skills, richer motion", '''
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
        "Historical generator / corrected after the sweep",
        "A predicted rest height does not prove a jam", '''
    <ol>
    <li>Draw a rack and box material; enumerate balloon subsets and their unconstrained hover heights.</li>
    <li>The sweep's helper could stop at a low-speed turning point before the box finished oscillating.</li>
    <li>Later replay showed that some supposed jamming alternatives actually reached the target. The earlier uniqueness and jam labels are not established.</li>
    <li>The revised checks execute Release sequences, require sustained off-goal rest, and verify that a suspected jam disappears without wall collisions.</li>
    </ol>
    <div class="rule"><p>Height alone can favour the wrong subset. Release order and the path through the chute still matter during execution.</p></div>
    <p class="small muted" style="margin-top:18px">Timeouts remain unresolved. A tested reference can win in both orders while an alternative wins in one order and jams in the other.</p>
    ''',
        'Source: docs/uncertainty-results/balloons-failure-analysis.md and balloons-subclass-followup.md; historical scores are preserved.'
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
        "Agent interfaces / current MB contract", "What the simulator adds",
        '''
    <div class="row"><div class="col card"><h3 class="mb">Model-based (MB)</h3><ul>
    <li>Uses a visible base simulator with the scene, chute geometry and rigid-body contacts.</li>
    <li>Writes a simulator subclass for release, attachment, lift and bursting; explicitly fits physical parameters.</li>
    <li>Can test release plans in its learned simulator before paying real steps.</li></ul></div>
    <div class="col card"><h3 class="mf">Model-free baseline (MF)</h3><ul>
    <li>Has the same real environment, skill tools, observations, journal and recorded experience.</li>
    <li>Has no supplied simulator or fit-and-rollout workbench.</li>
    <li>Still has arbitrary Python: it can fit equations and reason about dynamics by hand.</li></ul></div></div>
    <div class="rule"><p>Contact prediction is the intended benefit. The baseline's name does not mean it must guess or cannot build an analytic model.</p></div>
    ''',
        'The historical sweep used the earlier residual interface. Current MB uses the unified subclass contract; historical results are not reruns of it.'
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
        successful = [r for r in values if r["won"] == r["levels"]]
        mean_steps = (
            f'{sum(r["steps"] for r in successful) / len(successful):,.1f}'
            if successful else 'N/A')
        resets = sum(r["resets"] for r in values) / n
        rows.append(
            f'<tr style="background:#edf7f3;font-weight:bold"><td>{arm}</td><td>Mean</td>'
            f'<td>{solve:.1f}%</td><td>{mean_steps} (n={len(successful)})</td><td>{resets:.2f}</td></tr>'
        )
    add(
        "Historical noiseless uniform sweep / September 7",
        "Balloons: a modest gap in these three seeds", '''
    <table class="compact"><tr><th>Arm</th><th>Seed</th><th>Levels solved</th><th>Total real steps</th><th>Resets</th></tr>
    ''' + "\n".join(rows) + '''</table>
    <p class="small" style="margin-top:20px">MB: <b>9/9</b> levels; MF: <b>8/9</b>. These small samples do not establish a large step-efficiency advantage.</p>
    <p class="small muted">Mean steps include only whole-run successful seeds, including their attempts and resets (qualifying n shown). Solve rate and mean resets include all seeds. Per-seed rows retain actual total steps.</p>
    ''',
        'Source: six completed *_uniform scorecards pinned in sweep_20260907/manifest.json. These are the uniform sweep, not the earlier contact A/B results.'
    )

    add(
        "Sources and reproducibility",
        "Original results and hatch evidence have separate sources", '''
    <table class="compact"><tr><th>Source</th><th>What to inspect</th></tr>
    <tr><td><code>predicators/envs/pybullet_balloons_base.py</code></td><td>Scene, chute collision filters, attachment offsets and observed features.</td></tr>
    <tr><td><code>predicators/envs/pybullet_balloons.py</code></td><td>Lift, attachment, burst, task generation, oracle checks and evaluator.</td></tr>
    <tr><td><code>predicators/settings.py</code></td><td>Drag 2.2; contact-decoy default; 2 cm tolerance; training palette.</td></tr>
    <tr><td><code>protocol_continual_uniform_sweep.yaml</code></td><td>Two training levels, common MB/MF flags and noiseless protocol.</td></tr>
    <tr><td><code>docs/envs/balloons/sweep_20260907/</code></td><td>Saved screenshots, source manifest, per-seed results and hashes.</td></tr>
    <tr><td><code>docs/envs/balloons/visuals/</code></td><td>Hatch videos, frame images, height curves and a replay manifest.</td></tr>
    <tr><td><code>docs/slides/make_balloons_domain_slides.py</code></td><td>Rebuilds both standalone HTML decks; --pdf also renders the PDF.</td></tr></table>
    <p class="small muted" style="margin-top:18px">Hatch videos replay an audited task with fixed release sequences. Physical renders show true poses; the observation-noise channel is not painted into the scene. No hatch MB/MF result is claimed.</p>
    ''',
        'Updated 2026-09-09. Both HTML copies embed their media. The PDF uses the videos\' final frames.'
    )

    def embed_svg(match):
        svg = match.group(0).replace(
            '<svg ', '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
        data = base64.b64encode(svg.encode()).decode()
        alt = re.search(r'aria-label="([^"]+)"', svg).group(1)
        return (f'<img class="diagram" alt="{alt}" '
                f'src="data:image/svg+xml;base64,{data}">')

    content = re.sub(r'<svg\b.*?</svg>',
                     embed_svg,
                     '\n'.join(slides),
                     flags=re.S)
    html = (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Balloons - Original Chute and New Hatch</title><style>' + CSS +
        '</style></head>\n'
        '<body><main id="stage">\n' + content + '</main>\n'
        '<nav class="nav" aria-label="Slide navigation"><button id="prev" aria-label="Previous slide">←</button>'
        '<span id="count" aria-live="polite"></span><button id="next" aria-label="Next slide">→</button>'
        '<button id="hatch" aria-label="Jump to hatch examples">Hatch</button></nav>'
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
    preview = HERE.parent.parent / "logs" / "analysis" / "balloons_slides_20260909"
    preview.mkdir(parents=True, exist_ok=True)
    for i, page in enumerate(doc):
        assert abs(page.rect.width - 960) < 1 and abs(page.rect.height -
                                                      540) < 1
        if i in (3, 5):
            assert len(page.get_images()) >= 2, (i + 1,
                                                 "missing comparison images")
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
