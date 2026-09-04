"""Per-level recordings for the continual protocol (section 4.7).

One directory per level holds:

* ``actions.jsonl``: one line per primitive step (``{"ep", "i", "a"}``)
  and one per episode boundary (``{"event": "reset", ...}``), appended
  as the steps are taken. This is the replay source for a resume: the
  env loses nothing on a kill because every applied action is on disk.
* ``index.jsonl``: one line per skill invocation, reset, resume and
  level event, with the agent's expected outcome and note, the observed
  atoms, the episode state and the render path. The viewer reads this.
* ``episodes.pkl``: sanitised states and actions of every episode on
  the level, rewritten at each flush. Learning data and the viewer's
  state inspector both come from here.
* ``checkpoint.pkl``: the last observed state and the flush time, used
  to verify a replay and to measure downtime.
* ``renders/``: PNG renders at invocation boundaries.
"""
from __future__ import annotations

import json
import os
import pickle
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio
import numpy as np

from predicators.structs import Action, State

ACTIONS_FILENAME = "actions.jsonl"
INDEX_FILENAME = "index.jsonl"
EPISODES_FILENAME = "episodes.pkl"
CHECKPOINT_FILENAME = "checkpoint.pkl"
RENDERS_DIRNAME = "renders"


def option_label(action: Action) -> Optional[str]:
    """``Skill(objs)[params]`` for the option an action came from."""
    if not action.has_option():
        return None
    option = action.get_option()
    params = ", ".join(f"{float(v):.4g}" for v in option.params)
    return f"{option.simple_str()}[{params}]"


def sanitize_state(state: State) -> State:
    """A copy carrying only observable ``data`` (no simulator state)."""
    return State({o: np.array(v, copy=True) for o, v in state.data.items()})


def states_close(a: State, b: State, atol: float = 1e-3) -> bool:
    """Feature-by-feature comparison ignoring simulator state."""
    if sorted(a.data) != sorted(b.data):
        return False
    return all(np.allclose(a.data[o], b.data[o], atol=atol) for o in a.data)


class LevelRecording:
    """Writer and reader for one level's recording directory."""

    def __init__(self, level_dir: str) -> None:
        self._dir = level_dir
        os.makedirs(level_dir, exist_ok=True)
        os.makedirs(self.renders_dir, exist_ok=True)
        # Kept open for the level: one line per step, closed at level end.
        # pylint: disable-next=consider-using-with
        self._actions_file = open(self.actions_path, "a", encoding="utf-8")

    # -- Paths -------------------------------------------------------------

    @property
    def dir(self) -> str:
        """The level directory."""
        return self._dir

    @property
    def actions_path(self) -> str:
        """Path of the per-step action log."""
        return os.path.join(self._dir, ACTIONS_FILENAME)

    @property
    def index_path(self) -> str:
        """Path of the invocation index."""
        return os.path.join(self._dir, INDEX_FILENAME)

    @property
    def episodes_path(self) -> str:
        """Path of the sanitised episodes pickle."""
        return os.path.join(self._dir, EPISODES_FILENAME)

    @property
    def checkpoint_path(self) -> str:
        """Path of the replay-verification checkpoint."""
        return os.path.join(self._dir, CHECKPOINT_FILENAME)

    @property
    def renders_dir(self) -> str:
        """Directory of PNG renders."""
        return os.path.join(self._dir, RENDERS_DIRNAME)

    # -- Writing -----------------------------------------------------------

    def begin_episode(self, episode: int, by: str) -> None:
        """Mark an episode boundary in the action log."""
        self._write_action_line({
            "event": "reset",
            "ep": episode,
            "by": by,
            "t": time.time()
        })

    def append_step(self, episode: int, step: int, action: Action) -> None:
        """Log one applied primitive action."""
        self._write_action_line({
            "ep": episode,
            "i": step,
            "a": [float(v) for v in action.arr],
        })

    def append_index(self, entry: Dict[str, Any]) -> None:
        """Append one invocation-level event to the index."""
        entry = dict(entry)
        entry.setdefault("t", time.time())
        with open(self.index_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, sort_keys=True, default=_json_default))
            f.write("\n")

    def flush(self, episodes: Sequence[Dict[str, Any]], last_state: State,
              episode: int, steps: int) -> None:
        """Persist the episodes pickle and the checkpoint; sync the log."""
        self._actions_file.flush()
        os.fsync(self._actions_file.fileno())
        payload = [_sanitize_episode(ep) for ep in episodes]
        _atomic_pickle(self.episodes_path, payload)
        _atomic_pickle(
            self.checkpoint_path, {
                "episode": episode,
                "steps": steps,
                "state": sanitize_state(last_state),
                "ts": time.time(),
            })

    def save_render(self, name: str, image: np.ndarray) -> str:
        """Write one PNG under ``renders/`` and return its path."""
        path = os.path.join(self.renders_dir, f"{name}.png")
        imageio.imwrite(path, np.asarray(image).astype(np.uint8))
        return path

    def close(self) -> None:
        """Close the action log."""
        if not self._actions_file.closed:
            self._actions_file.flush()
            self._actions_file.close()

    # -- Reading -----------------------------------------------------------

    def load_checkpoint(self) -> Optional[Dict[str, Any]]:
        """The last checkpoint, or ``None`` when the level never flushed."""
        if not os.path.isfile(self.checkpoint_path):
            return None
        with open(self.checkpoint_path, "rb") as f:
            data: Dict[str, Any] = pickle.load(f)
        return data

    def read_current_episode(self) -> Tuple[int, List[np.ndarray]]:
        """The last episode's index and its applied action arrays."""
        episode = 0
        actions: List[np.ndarray] = []
        if not os.path.isfile(self.actions_path):
            return episode, actions
        with open(self.actions_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("event") == "reset":
                    episode = int(rec["ep"])
                    actions = []
                else:
                    actions.append(np.array(rec["a"], dtype=np.float32))
        return episode, actions

    def read_episodes(self) -> List[Dict[str, Any]]:
        """The sanitised episodes pickle, or ``[]``."""
        if not os.path.isfile(self.episodes_path):
            return []
        with open(self.episodes_path, "rb") as f:
            episodes: List[Dict[str, Any]] = pickle.load(f)
        return episodes

    def read_index(self) -> List[Dict[str, Any]]:
        """All index entries in order."""
        if not os.path.isfile(self.index_path):
            return []
        entries = []
        with open(self.index_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    # -- Internals ---------------------------------------------------------

    def _write_action_line(self, rec: Dict[str, Any]) -> None:
        self._actions_file.write(json.dumps(rec))
        self._actions_file.write("\n")
        self._actions_file.flush()


def _sanitize_episode(ep: Dict[str, Any]) -> Dict[str, Any]:
    states: Sequence[State] = ep["states"]
    actions: Sequence[Action] = ep["actions"]
    return {
        "episode":
        ep["episode"],
        "end":
        ep.get("end", "in_progress"),
        "states": [sanitize_state(s) for s in states],
        "actions": [{
            "arr": np.array(a.arr, copy=True),
            "option": option_label(a)
        } for a in actions],
    }


def _atomic_pickle(path: str, payload: Any) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (set, frozenset)):
        return sorted(str(x) for x in obj)
    return str(obj)
