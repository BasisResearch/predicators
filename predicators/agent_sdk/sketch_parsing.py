"""Read/write grammar for agent plan sketches.

Split out of ``bilevel_sketch`` (see that module's docstring for the
full layout); holds both sides of the sketch-line grammar - the
formatters that render steps/sketches/plans as text and the parsers
that turn plan text back into ``SketchStep``s (subgoal annotations,
``~`` ground-sampler annotations, continuous params) - so the two
sides cannot drift apart.
"""
import logging
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from predicators import utils
from predicators.agent_sdk.sketch_types import GroundSampler, SketchStep
from predicators.structs import GroundAtom, Object, ParameterizedOption, \
    ParameterizedSampler, Predicate, Task, Type, _Option


def _fmt_params(opt: _Option) -> str:
    """Compact one-line dump of a grounded option's parameters."""
    return np.array2string(np.asarray(opt.params, dtype=float),
                           precision=4,
                           separator=", ")


def format_step_line(
    idx: int,
    option_name: str,
    objects: Sequence[Object],
    params: Optional[Union[Sequence[float], np.ndarray]] = None,
    subgoal_atoms: Optional[Set[GroundAtom]] = None,
    params_width: Optional[Union[Sequence[float], np.ndarray]] = None,
    sampler_name: Optional[str] = None,
    subgoal_neg_atoms: Optional[Set[GroundAtom]] = None,
) -> str:
    """Format one plan/sketch step as a single indented line.

    ``  <idx>: OptName(obj1, obj2)[p0, p1] ~ [w0, w1] -> {Atom, NOT
    Atom}``

    The ``[params]``, ``~ [widths]`` / ``~ name`` and ``-> {atoms}``
    slots are omitted when their argument is empty/None. Negative
    subgoals render as ``NOT Atom`` inside the same braces - dropping
    them made the logged plan diverge from the sketch monitoring
    actually used (a monitored ``NOT GluedEndB`` Wait logged as bare
    ``Wait(robot)`` in run_20260830_145216's attempts record). Shared
    by the sketch- and plan-formatting helpers below so every per-step
    line reads identically.
    """
    objs = ", ".join(o.name for o in objects)
    line = f"  {idx}: {option_name}({objs})"
    if params is not None and len(params):
        par = ", ".join(f"{p:.4f}" for p in params)
        line += f"[{par}]"
        if params_width is not None and len(params_width):
            wid = ", ".join(f"{w:.4f}" for w in params_width)
            line += f" ~ [{wid}]"
    if sampler_name:
        line += f" ~ {sampler_name}"
    if subgoal_atoms or subgoal_neg_atoms:
        parts = [str(a) for a in sorted(subgoal_atoms or set(), key=str)]
        parts += [
            f"NOT {a}" for a in sorted(subgoal_neg_atoms or set(), key=str)
        ]
        line += f" -> {{{', '.join(parts)}}}"
    return line


def format_sketch_lines(sketch: Sequence[SketchStep]) -> List[str]:
    """Render a plan sketch as one ``format_step_line`` per step.

    Each step shows its ``initial_params`` (if the LLM proposed any),
    its ground-sampler annotation (window half-widths or the referenced
    sampler name) and its ``subgoal_atoms``.
    """
    lines = []
    for i, s in enumerate(sketch):
        gs = s.ground_sampler
        lines.append(
            format_step_line(i,
                             s.option.name,
                             s.objects,
                             params=s.initial_params,
                             subgoal_atoms=s.subgoal_atoms,
                             params_width=gs.width if gs is not None else None,
                             sampler_name=(gs.name if gs is not None
                                           and gs.fn is not None else None),
                             subgoal_neg_atoms=s.subgoal_neg_atoms))
    return lines


def format_plan_lines(
    plan: Sequence[_Option],
    sketch: Optional[Sequence[SketchStep]] = None,
) -> List[str]:
    """Render a grounded option plan as one ``format_step_line`` per step.

    Each step shows its continuous ``params``. When ``sketch`` is given,
    the parallel step's ``subgoal_atoms`` are appended so the log
    mirrors the annotated sketch.
    """
    lines = []
    for i, opt in enumerate(plan):
        step = sketch[i] if sketch and i < len(sketch) else None
        subgoals = step.subgoal_atoms if step is not None else None
        neg_subgoals = step.subgoal_neg_atoms if step is not None else None
        lines.append(
            format_step_line(i,
                             opt.name,
                             opt.objects,
                             params=opt.params,
                             subgoal_atoms=subgoals,
                             subgoal_neg_atoms=neg_subgoals))
    return lines


def strip_code_fences(text: str) -> str:
    """Strip markdown code fences wrapping plan text."""
    lines = text.split('\n')
    while lines and lines[0].strip().startswith('```'):
        lines.pop(0)
    while lines and lines[-1].strip().startswith('```'):
        lines.pop()
    return '\n'.join(lines)


# Matches an atom like ``Pred(a:t, b:t)`` or ``NOT Pred(a)`` in subgoal text.
_ATOM_RE = re.compile(r'(NOT\s+)?(\w+)\(([^)]*)\)')

# Once-per-process memory for tolerant-mode subgoal warnings: annotations
# are re-parsed on every monitored step, so an unknown name used to log
# one warning per parse (hundreds per predicate per run - 2026-08-30
# bridge runs logged ~660/~1460 of them).
_warned_subgoal_issues: Set[str] = set()


def _warn_subgoal_once(msg: str) -> None:
    if msg in _warned_subgoal_issues:
        return
    _warned_subgoal_issues.add(msg)
    logging.warning("%s (skipping this atom; logged once)", msg)


def parse_atoms(
    atoms_text: str,
    predicates: Set[Predicate],
    objects: Sequence[Object],
    strict: bool = False,
) -> Tuple[Set[GroundAtom], Set[GroundAtom]]:
    """Parse atoms like ``Pred(a:t, b:t)`` / ``NOT Pred(a)`` from a string.

    Returns ``(positive_atoms, negative_atoms)``. Any number of atoms
    may appear in ``atoms_text`` (separated by commas or anything else —
    the regex finds each ``Pred(...)``).

    ``strict`` (tool inputs): an atom with an unknown predicate or
    object, or the wrong arity, raises ``ValueError`` naming the
    problem - silently dropping it would monitor a different sketch
    than the agent wrote (a literal ``NotARealPredicate`` annotation
    passed refinement untouched in run_20260830_145226). Tolerant mode
    (re-parsing logged plans) skips such atoms with a once-per-process
    warning.
    """
    pred_map = {p.name: p for p in predicates}
    obj_map = {o.name: o for o in objects}
    pos_atoms: Set[GroundAtom] = set()
    neg_atoms: Set[GroundAtom] = set()

    def _bad(msg: str, hint: str) -> None:
        if strict:
            raise ValueError(f"{msg} in subgoal annotation - {hint}")
        _warn_subgoal_once(f"{msg} in subgoal")

    for atom_match in _ATOM_RE.finditer(atoms_text):
        is_neg = atom_match.group(1) is not None
        pred_name = atom_match.group(2)
        obj_names = [
            n.strip().split(':')[0] for n in atom_match.group(3).split(',')
        ]
        if pred_name not in pred_map:
            _bad(f"Unknown predicate {pred_name!r}",
                 f"available predicates: {sorted(pred_map)}")
            continue
        pred = pred_map[pred_name]
        missing = [n for n in obj_names if n not in obj_map]
        if missing:
            _bad(f"Unknown object(s) {missing} for {pred_name}",
                 f"available objects: {sorted(obj_map)}")
            continue
        objs = [obj_map[n] for n in obj_names]
        if len(objs) != len(pred.types):
            _bad(
                f"Arity mismatch for {pred_name}: expected "
                f"{len(pred.types)}, got {len(objs)}",
                "give one argument per predicate type")
            continue
        (neg_atoms if is_neg else pos_atoms).add(GroundAtom(pred, objs))
    return pos_atoms, neg_atoms


def parse_subgoal_annotations(
    text: str,
    predicates: Set[Predicate],
    objects: Sequence[Object],
    option_names: Set[str],
    strict: bool = False,
) -> List[Optional[Tuple[Set[GroundAtom], Set[GroundAtom]]]]:
    """Parse ``-> {Pred(...), NOT Pred(...)}`` annotations from plan text.

    Returns a list parallel to the option lines in ``text``. Each entry
    is ``None`` for a line with no annotation, or ``(positive_atoms,
    negative_atoms)`` otherwise. ``strict`` makes a malformed atom raise
    instead of being skipped (see ``parse_atoms``).
    """
    subgoal_re = re.compile(r'->\s*\{([^}]*)\}')
    results: List[Optional[Tuple[Set[GroundAtom], Set[GroundAtom]]]] = []

    for line in text.split('\n'):
        # Mirror the enumeration-prefix tolerance in the option-plan
        # parser so the per-line subgoal results stay index-parallel with
        # the parsed options (a numbered "0: Pick(...)" line must be seen
        # as an option line here too, else annotations misalign).
        stripped = utils.strip_enumeration_prefix(line.strip())
        if not stripped:
            continue
        first_token = stripped.split('(')[0]
        if first_token not in option_names:
            continue

        sg_match = subgoal_re.search(stripped)
        if not sg_match:
            results.append(None)
            continue

        pos_atoms, neg_atoms = parse_atoms(sg_match.group(1),
                                           predicates,
                                           objects,
                                           strict=strict)
        if pos_atoms or neg_atoms:
            results.append((pos_atoms, neg_atoms))
        else:
            results.append(None)

    return results


# A `-> {atoms}` subgoal annotation appended to a sketch step line. Stripped
# before the canonical option-plan parser reads the `[params]` block, so a
# `{...}` brace is never mistaken for params text.
_SUBGOAL_ANNOTATION_RE = re.compile(r'\s*->\s*\{[^}]*\}')


def strip_subgoal_annotations(text: str) -> str:
    """Remove ``-> {atoms}`` subgoal annotations from every line."""
    return _SUBGOAL_ANNOTATION_RE.sub('', text)


# A ground-sampler annotation appended after a step's `[params]` block:
# either a window `~ [w1, w2]` (per-dimension half-widths around the
# proposed params) or a named code sampler `~ my_sampler` (a
# GROUND_SAMPLERS key). Stripped before the canonical option-plan parser
# reads the `[params]` block (that parser takes the text after the FIRST
# `[` and would misread the widths as parameter text), and parsed
# separately per line. The raw token keeps its brackets so the resolver
# can tell the two forms apart.
_REGION_ANNOTATION_RE = re.compile(r'\s*~\s*(\[[^\]]*\]|[A-Za-z_]\w*)')


def strip_region_annotations(text: str) -> str:
    """Remove ``~ [widths]`` / ``~ name`` annotations from every line."""
    return _REGION_ANNOTATION_RE.sub('', text)


def parse_region_annotations(
    text: str,
    option_names: Set[str],
) -> List[List[str]]:
    """Extract raw ground-sampler annotation tokens from plan text.

    Returns a list parallel to the option lines in ``text`` (same line
    filter as ``parse_subgoal_annotations``, so the two stay aligned
    with the parsed options). Each entry lists that line's raw ``~``
    tokens - ``[w1, w2]`` with brackets, or a bare sampler name - empty
    when the line has none. Validation happens in
    ``_resolve_ground_sampler``, where the resolved option, strictness,
    and the loaded named samplers are known.
    """
    results: List[List[str]] = []
    for line in text.split('\n'):
        stripped = utils.strip_enumeration_prefix(line.strip())
        if not stripped:
            continue
        first_token = stripped.split('(')[0]
        if first_token not in option_names:
            continue
        results.append(
            [m.group(1) for m in _REGION_ANNOTATION_RE.finditer(stripped)])
    return results


def _resolve_ground_sampler(
    raw_blocks: List[str],
    step_idx: int,
    option: ParameterizedOption,
    center: Optional[np.ndarray],
    strict: bool,
    enabled: bool,
    ground_sampler_fns: Optional[Dict[str, ParameterizedSampler]],
    notices: Optional[List[str]] = None,
) -> Optional[GroundSampler]:
    """Validate one step's ``~`` annotation into a ``GroundSampler``.

    Returns ``None`` when the step carries no annotation. A bad
    annotation raises ``ValueError`` naming the step in strict mode; in
    tolerant mode it is dropped with a warning and the step is kept.
    With ``enabled`` False (``agent_bilevel_ground_samplers`` off) the
    annotation is silently ignored - the step keeps its params as an
    exact seed and refinement uses the default uniform samplers - with
    a line appended to ``notices`` so tool output can say so (an error
    here cost every audited run 1-3 turns of syntax guessing).
    ``ground_sampler_fns`` maps the names that a ``~ my_sampler`` form
    may reference.
    """
    if not raw_blocks:
        return None

    # Explicit final return: pylint calls it useless, mypy requires it.
    # pylint: disable-next=useless-return
    def _bad(reason: str) -> Optional[GroundSampler]:
        msg = (f"step {step_idx} ({option.name}): bad '~' ground-sampler "
               f"annotation - {reason}")
        if strict:
            raise ValueError(msg)
        logging.warning("Dropping ground-sampler annotation: %s", msg)
        return None

    if not enabled:
        note = (f"step {step_idx} ({option.name}): the '~' region "
                "annotation was IGNORED - ground samplers are disabled in "
                "this configuration, so the step's params seed the search "
                "and sampling uses the default uniform samplers.")
        logging.info("Ignoring ground-sampler annotation: %s", note)
        if notices is not None and note not in notices:
            notices.append(note)
        return None
    if len(raw_blocks) > 1:
        return _bad("multiple '~' annotations on one line")
    token = raw_blocks[0]
    if not token.startswith('['):
        # Named code sampler.
        fns = ground_sampler_fns or {}
        if token not in fns:
            avail = (", ".join(sorted(fns))
                     if fns else "none loaded - define GROUND_SAMPLERS in "
                     "ground_samplers.py")
            return _bad(f"unknown ground sampler '{token}' (available: "
                        f"{avail})")
        return GroundSampler(fn=fns[token], name=token)
    if center is None:
        return _bad("'~ [widths]' requires proposed center params in "
                    "'[...]' on the same line")
    tokens = [t.strip() for t in token[1:-1].split(',') if t.strip()]
    if not tokens:
        return _bad("empty '~ []' block; give one half-width per "
                    "parameter or omit the block")
    try:
        widths = np.asarray([float(t) for t in tokens], dtype=np.float32)
    except ValueError:
        return _bad(f"non-numeric half-width in {token!r}")
    expected = option.params_space.shape[0]
    if widths.shape[0] != expected:
        return _bad(f"{widths.shape[0]} half-width(s) but option "
                    f"{option.name} expects {expected}")
    if np.any(widths < 0):
        return _bad("half-widths must be >= 0")
    return GroundSampler(center=center, width=widths)


def parse_sketch_from_text(
    plan_text: str,
    task: Task,
    *,
    predicates: Set[Predicate],
    options: Set[ParameterizedOption],
    types: Set[Type],
    parse_continuous_params: bool = False,
    strict: bool = False,
    parse_ground_samplers: bool = True,
    ground_sampler_fns: Optional[Dict[str, ParameterizedSampler]] = None,
    notices: Optional[List[str]] = None,
) -> List[SketchStep]:
    """Parse plan-sketch text into ``SketchStep``s.

    Applies ``strip_code_fences`` first, then delegates option-plan
    parsing to ``utils.parse_model_output_into_option_plan`` and subgoal
    annotation parsing to ``parse_subgoal_annotations``.

    ``strict`` is for tool inputs that are pure plan text: any line that
    fails to parse raises ``ValueError`` naming the line, instead of the
    default freeform tolerance (skip preamble, drop malformed lines,
    truncate at the first non-option line). Without it, a dropped line
    also silently misaligns the per-line subgoal annotations below.
    Strict mode also validates every subgoal atom: an unknown predicate
    or object, or a wrong arity, raises instead of being silently
    dropped from the monitored sketch (see ``parse_atoms``).

    When ``parse_continuous_params`` is set, each step's ``[p0, p1, ...]``
    block is parsed by the SAME canonical parser the open-loop planner
    uses (``parse_model_output_into_option_plan`` with
    ``parse_continuous_params=True``) and stored as ``initial_params`` for
    the refinement to try first. Sketch lines also carry ``-> {subgoal}``
    annotations and optional ``~`` ground-sampler annotations (a window
    ``~ [w0, w1]`` of per-parameter half-widths, or ``~ my_sampler``
    naming an entry of ``ground_sampler_fns``; refinement tries the
    exact center once, then draws the step from the ground sampler).
    Both would be misread as params text by that parser, so they are
    stripped before parsing and read separately from the original (a
    dropped line in tolerant mode misaligns them the same way;
    ``strict`` tool inputs error instead).

    ``parse_ground_samplers`` is the caller-threaded value of
    ``RefinementConfig.ground_samplers``: when False, any ``~``
    annotation is accepted but ignored (params still seed the search;
    sampling stays uniform), with an explanation appended to
    ``notices`` for the caller to surface in tool output.
    """
    cleaned_text = strip_code_fences(plan_text)
    objects = list(task.init)
    option_names = {o.name for o in options}

    # Strip subgoal and region annotations only when parsing params, so the
    # `[params]` extraction in the canonical parser isn't confused by a
    # `{...}` brace or a second `[...]` block. (With params off the parser
    # never reads past the `)`, so the annotations are inert there.)
    parse_text = (strip_region_annotations(
        strip_subgoal_annotations(cleaned_text))
                  if parse_continuous_params else cleaned_text)
    parsed = utils.parse_model_output_into_option_plan(
        parse_text,
        objects,
        types,
        options,
        parse_continuous_params=parse_continuous_params,
        strict=strict)

    if not parsed:
        return []

    subgoals = parse_subgoal_annotations(cleaned_text,
                                         predicates,
                                         objects,
                                         option_names,
                                         strict=strict)
    regions = (parse_region_annotations(cleaned_text, option_names)
               if parse_continuous_params else [])

    sketch: List[SketchStep] = []
    for i, (option, objs, params) in enumerate(parsed):
        sg = subgoals[i] if i < len(subgoals) else None
        ip = (np.asarray(params, dtype=np.float32)
              if parse_continuous_params else None)
        if ip is not None and ip.size == 0 and \
                option.params_space.shape[0] > 0:
            # Explicit `[]` on a parametrized option: "no seed" - let the
            # refinement search sample the parameters (strict parsing lets
            # the empty list through for exactly this case).
            ip = None
        ground_sampler = _resolve_ground_sampler(
            regions[i] if i < len(regions) else [],
            i,
            option,
            center=ip if ip is not None and ip.size > 0 else None,
            strict=strict,
            enabled=parse_ground_samplers,
            ground_sampler_fns=ground_sampler_fns,
            notices=notices)
        if sg is not None:
            pos, neg = sg
            sketch.append(
                SketchStep(option=option,
                           objects=objs,
                           subgoal_atoms=pos if pos else None,
                           subgoal_neg_atoms=neg if neg else None,
                           initial_params=ip,
                           ground_sampler=ground_sampler))
        else:
            sketch.append(
                SketchStep(option=option,
                           objects=objs,
                           subgoal_atoms=None,
                           initial_params=ip,
                           ground_sampler=ground_sampler))
    # Coverage diagnostic: unannotated steps are invisible to per-step
    # refinement validation, execution monitoring, and suffix replanning.
    unannotated = [
        f"{i}: {s.option.name}" for i, s in enumerate(sketch)
        if s.subgoal_atoms is None and s.subgoal_neg_atoms is None
    ]
    if unannotated:
        logging.info("Sketch subgoal coverage: %d/%d steps unannotated (%s).",
                     len(unannotated), len(sketch), ", ".join(unannotated))
    return sketch
