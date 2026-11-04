"""The claim check of a play session (``continual_claim_check``).

An agent can trust a belief it has barely tested: round 32's Bridge seed
2 read one glue probe as "the vertical face can never be reached" and
reset nine times on it. At set points the skill tools wait for a record
of the belief the next step leans on: the recordings that tested it
directly, what varied between them, a rival explanation, and the
experiment that would separate the two. The harness cannot see a belief
or judge an argument, so it decides when a record is due and checks its
form, and the agent decides what goes in it.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from predicators.settings import CFG

# Why a record is due.
AFTER_RESET = "after_reset"
READY_TO_WIN = "ready_to_win"
LOW_P_HAT = "low_p_hat"

_REASONS = {
    AFTER_RESET: ("this level has been reset {resets} times without a win, "
                  "so a belief behind these attempts may be wrong"),
    READY_TO_WIN: ("this request's rehearsal has P-hat {p_hat:.2f}, and a win "
                   "ends this level and the experiments it allows"),
    LOW_P_HAT: ("this request's rehearsal has P-hat {p_hat:.2f}, so acting on "
                "it dismisses the failing draws"),
}
_TEXT_FIELDS = ("claim", "rules_out", "what_varied", "rival",
                "next_experiment")
_VERDICTS = ("supported", "refuted", "unresolved", "not_run")
# A verdict that reports an experiment needs something run since the record.
_NEEDS_A_RUN = ("supported", "refuted", "unresolved")


@dataclass
class ClaimCheck:
    """When a record is due, and the records of this run so far.

    ``log_path`` is the run's ``claims.jsonl``: every record and
    resolution is appended there, and a resumed run reads them back so a
    restart neither forgets an open claim nor asks again.
    """
    log_path: Optional[str] = None
    records: List[Dict[str, Any]] = field(default_factory=list)
    resolutions: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    # The trigger of the last refusal, until a record answers it.
    pending: Optional[str] = None

    def __post_init__(self) -> None:
        if self.log_path is None or not os.path.isfile(self.log_path):
            return
        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row["event"] == "record":
                    self.records.append(row)
                else:
                    self.resolutions[int(row["id"])] = row

    def _append(self, row: Dict[str, Any]) -> None:
        if self.log_path is None:
            return
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def _answered(self,
                  trigger: str,
                  level: int,
                  resets: Optional[int] = None) -> bool:
        return any(r["trigger"] == trigger and r["level"] == level and (
            resets is None or r["resets"] == resets) for r in self.records)

    def due(self, split: str, level: int, resets: int,
            p_hat: Optional[float]) -> Optional[str]:
        """The trigger a skill request must wait on, if any.

        ``p_hat`` is that of the rehearsal that lets the request
        through, None when no rehearsal does. On a training level a
        record is due at each reset from
        ``continual_claim_check_resets`` on, and once a level when a
        rehearsal reaches ``continual_claim_check_win_p_hat``. On a test
        level one is due once when the agent acts below
        ``continual_claim_check_low_p_hat``.
        """
        if split == "test":
            if (p_hat is not None
                    and p_hat < CFG.continual_claim_check_low_p_hat
                    and not self._answered(LOW_P_HAT, level)):
                return LOW_P_HAT
            return None
        if (resets >= CFG.continual_claim_check_resets
                and not self._answered(AFTER_RESET, level, resets)):
            return AFTER_RESET
        if (p_hat is not None and p_hat >= CFG.continual_claim_check_win_p_hat
                and not self._answered(READY_TO_WIN, level)):
            return READY_TO_WIN
        return None

    def open_ids(self) -> List[int]:
        """The records whose experiment has no resolution yet."""
        return [
            r["id"] for r in self.records if r["id"] not in self.resolutions
        ]

    def refusal(self, trigger: str, resets: int,
                p_hat: Optional[float]) -> str:
        """The text of a refusal for ``trigger``, which it leaves pending."""
        self.pending = trigger
        reason = _REASONS[trigger].format(resets=resets, p_hat=p_hat or 0.0)
        text = (f"Claim check: {reason}. Before this request, call "
                "claims_record with the belief it leans on most.")
        if self.open_ids():
            text += (" First call claims_resolve for the open records: "
                     f"{self.open_ids()}.")
        return text

    def record(self, args: Dict[str, Any], *, recordings: int, level: int,
               resets: int, run_steps: int,
               rehearsals: int) -> Tuple[bool, str]:
        """Add a record; whether it was accepted, and the reply."""
        if self.pending is None:
            return False, "No claim check is due."
        if self.open_ids():
            return False, ("Call claims_resolve first for the open records: "
                           f"{self.open_ids()}.")
        problem = _form_problem(args, recordings)
        if problem is not None:
            return False, problem
        row = {
            "event": "record",
            "id": len(self.records),
            "trigger": self.pending,
            "level": level,
            "resets": resets,
            "run_steps": run_steps,
            "rehearsals": rehearsals,
        }
        row.update({k: args[k] for k in _TEXT_FIELDS})
        row["direct_tests"] = args["direct_tests"]
        row["predictions"] = args["predictions"]
        self.records.append(row)
        self._append(row)
        self.pending = None
        return True, (f"Recorded as {row['id']}. Run the experiment, then "
                      "call claims_resolve with what it showed. Send the "
                      "request again when it is the step you want.")

    def resolve(self, args: Dict[str, Any], *, run_steps: int,
                rehearsals: int) -> Tuple[bool, str]:
        """Close a record; whether it was closed, and the reply."""
        try:
            claim_id = int(args.get("id", -1))
        except (TypeError, ValueError):
            claim_id = -1
        if claim_id not in self.open_ids():
            return False, f"Open records: {self.open_ids()}."
        verdict = str(args.get("verdict", ""))
        result = str(args.get("result", "")).strip()
        if verdict not in _VERDICTS or not result:
            return False, ("claims_resolve needs a result and a verdict, one "
                           f"of {list(_VERDICTS)}.")
        record = self.records[claim_id]
        if verdict in _NEEDS_A_RUN and (run_steps == record["run_steps"] and
                                        rehearsals == record["rehearsals"]):
            return False, (
                "No real step and no rehearsal has run since this record, so "
                "no experiment can have shown this. Run it, or give the "
                "verdict not_run and say why in the result.")
        row = {
            "event": "resolve",
            "id": claim_id,
            "verdict": verdict,
            "result": result,
            "run_steps": run_steps,
        }
        self.resolutions[claim_id] = row
        self._append(row)
        return True, f"Record {claim_id} resolved: {verdict}."


def _form_problem(args: Dict[str, Any], recordings: int) -> Optional[str]:
    """What is wrong with the form of a record, or None."""
    for name in _TEXT_FIELDS:
        if name != "next_experiment" and not str(args.get(name, "")).strip():
            return f"claims_record needs {name}."
    tests = args.get("direct_tests")
    if not isinstance(tests, list) or not tests:
        return ("claims_record needs direct_tests: each with the recording "
                "index that tested the claim directly and what it showed. A "
                "belief no recording tested needs an experiment first.")
    for test in tests:
        if not isinstance(test, dict) or not str(test.get("what", "")).strip():
            return "Each direct test needs a recording index and what."
        index = test.get("recording")
        if not isinstance(index, int) or not 0 <= index < recordings:
            return (f"Direct test recording {index!r} is not one of the "
                    f"{recordings} recordings (indices from 0).")
    predictions = args.get("predictions")
    if not isinstance(predictions, dict) or not all(
            str(predictions.get(k, "")).strip() for k in ("claim", "rival")):
        return ("claims_record needs predictions: what next_experiment "
                "shows if the claim holds (claim) and if the rival does "
                "(rival).")
    if not str(args.get("next_experiment", "")).strip():
        return "claims_record needs next_experiment."
    return None
