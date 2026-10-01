# PyBullet Environments

**RoboDisco** (Robot Model Discovery Benchmark) is a suite of 22 environments for robotic world-model learning and causal discovery.
The envs ship as part of the [predicators](../../README.md) repository and are exposed through a standard [Gymnasium](https://gymnasium.farama.org/) API.

🌐 **Project page:** <https://yichao-liang.github.io/robodisco-site/>

Each environment features a Fetch or Panda robot interacting with objects on a tabletop.
Five of them (Balloons, Boil, Bridge, Domino and Fan) are the benchmark domains of the EMPIRIC experiments; their benchmark settings live in [`scripts/configs/empiric/envs.yaml`](../../scripts/configs/empiric/envs.yaml).
The same envs are used by predicators' planning research code and can be consumed independently of the planner.

## Installation

From the repo root:

```bash
pip install -e .
```

This installs the agent solvers and the RoboDisco envs together.
The package is slightly heavy because it bundles both; a lighter envs-only install is future work.

## Quick Start (Gymnasium API)

```python
from predicators import utils
from predicators.envs import gymnasium_wrapper as robodisco

# Apply parser defaults to predicators' global CFG (only needed when
# consuming the envs as a library rather than via main.py).
utils.reset_config({"num_train_tasks": 1, "num_test_tasks": 1})

robodisco.register_all_environments()
env = robodisco.make("robodisco/Blocks-v0", render_mode="rgb_array")

obs, info = env.reset()
for _ in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break

frame = env.render()  # (H, W, 3) uint8 RGB array
env.close()
```

The Gymnasium wrapper exposes:

- `obs`: a 1-D `float32` numpy array of object features (PyBullet body ids and other `sim_features` are excluded).
  Its layout comes from the objects of the first training task, so objects that only appear in other tasks (the BusyBoard and Laser test tasks add some) are missing from `obs`; read them from `info["state"]`.
- `action_space`: the underlying robot's joint action space, as a `gymnasium.spaces.Box`.
- `reward`: `1.0` when all goal predicates are satisfied, `0.0` otherwise.
- `terminated`: `True` when the goal is reached.
- `truncated`: `True` when the episode hits the 500-step limit.
- `info["state"]`: the full object-centric `predicators.structs.State` for the current step (predicates, types, sim state).
- `info["goal_reached"]`: shortcut for `env.goal_reached()`.
- `env.reset(options={"train_or_test": "test", "task_idx": 0})` selects the task to play (training task 0 by default), and `env.reset(seed=s)` reseeds the environment.

## Walkthroughs

- **Notebook:** [`notebooks/getting_started.ipynb`](../../notebooks/getting_started.ipynb), an interactive walkthrough with rendering.
- **Smoke script:** [`scripts/robodisco_getting_started.py`](../../scripts/robodisco_getting_started.py), a non-interactive smoke test that mirrors the notebook and resets every env to verify installation health.

## Environments

Status legend:
- **Preview** - one task being solved, by planning with ground-truth models or by a scripted skill sequence, from the [RoboDisco site](https://yichao-liang.github.io/robodisco-site/); click it for the environment's page.
- **Tasks** - the task generator varies the initial state across tasks (✅) or starts every task from one layout (❌).
- **Skills** - `predicators/ground_truth_models/<env>/options.py` builds a non-empty skill library (✅) or none (❌); see [Skill libraries](#skill-libraries).
- **Oracle** - the approach that solves the test task with `python predicators/main.py --env <env_name> --approach <approach> --seed 0 --num_train_tasks 1 --num_test_tasks 1 --timeout 60` in the default configuration: `oracle` (bilevel planning with ground-truth operators and samplers) or `oracle_process_planning` (planning with ground-truth process models); ❌ when neither does.
- Names in **bold** are the five benchmark domains.

| Environment | Preview | Description | Tasks | Skills | Oracle |
|---|---|---|:---:|:---:|:---:|
| `robodisco/Ants-v0` | [![](https://yichao-liang.github.io/robodisco-site/assets/gifs/ants.gif)](https://yichao-liang.github.io/robodisco-site/envs/ants.html) | Sort food by whether it attracts ants, stacking same-coloured items | ✅ | ✅ | `oracle` |
| `robodisco/Balance-v0` | [![](https://yichao-liang.github.io/robodisco-site/assets/gifs/balance.gif)](https://yichao-liang.github.io/robodisco-site/envs/balance.html) | Equalise two plates on a beam, then press the button | ✅ | ✅ | `oracle` |
| **`robodisco/Balloons-v0`** | [![](https://yichao-liang.github.io/robodisco-site/assets/gifs/balloons.gif)](https://yichao-liang.github.io/robodisco-site/envs/balloons.html) | Free clipped balloons so they lift a box to hang inside a target band | ✅ | ✅ | ❌ |
| `robodisco/Barrier-v0` | [![](https://yichao-liang.github.io/robodisco-site/assets/gifs/barrier.gif)](https://yichao-liang.github.io/robodisco-site/envs/barrier.html) | Raise and lower barriers with the switches that control them | ✅ | ❌ | ❌ |
| `robodisco/Blocks-v0` | [![](https://yichao-liang.github.io/robodisco-site/assets/gifs/blocks.gif)](https://yichao-liang.github.io/robodisco-site/envs/blocks.html) | Rearrange blocks into the requested towers | ✅ | ✅ | `oracle` |
| **`robodisco/Boil-v0`** | [![](https://yichao-liang.github.io/robodisco-site/assets/gifs/boil.gif)](https://yichao-liang.github.io/robodisco-site/envs/boil.html) | Fill jugs at a faucet and boil them on a burner without spilling | ✅ | ✅ | `oracle_process_planning`\* |
| **`robodisco/Bridge-v0`** | [![](https://yichao-liang.github.io/robodisco-site/assets/gifs/bridge.gif)](https://yichao-liang.github.io/robodisco-site/envs/bridge.html) | Glue identical blocks into an n-shaped bridge whose joints cure over time | ✅ | ✅ | `oracle_process_planning`\* |
| `robodisco/BusyBoard-v0` | [![](https://yichao-liang.github.io/robodisco-site/assets/gifs/busyboard.gif)](https://yichao-liang.github.io/robodisco-site/envs/busyboard.html) | Light the right lamps on a board of switches with hidden wiring | ✅ | ✅ | ❌ |
| `robodisco/Circuit-v0` | [![](https://yichao-liang.github.io/robodisco-site/assets/gifs/circuit.gif)](https://yichao-liang.github.io/robodisco-site/envs/circuit.html) | Wire a battery to a bulb and switch it on | ✅ | ✅ | `oracle` |
| `robodisco/Coffee-v0` | [![](https://yichao-liang.github.io/robodisco-site/assets/gifs/coffee.gif)](https://yichao-liang.github.io/robodisco-site/envs/coffee.html) | Brew a jug of coffee and pour it into every cup | ✅ | ✅ | ❌ |
| `robodisco/Cover-v0` | [![](https://yichao-liang.github.io/robodisco-site/assets/gifs/cover.gif)](https://yichao-liang.github.io/robodisco-site/envs/cover.html) | Place blocks so they cover target regions | ✅ | ✅ | `oracle` |
| `robodisco/Crane-v0` | [![](https://yichao-liang.github.io/robodisco-site/assets/gifs/crane.gif)](https://yichao-liang.github.io/robodisco-site/envs/crane.html) | Swing a hinged ram to knock a crate onto a landing pad | ✅ | ✅ | `oracle` |
| **`robodisco/Domino-v0`** | [![](https://yichao-liang.github.io/robodisco-site/assets/gifs/domino.gif)](https://yichao-liang.github.io/robodisco-site/envs/domino.html) | Place as few dominoes as possible so a pushed chain topples the target | ✅ | ✅ | ❌ |
| **`robodisco/Fan-v0`** | [![](https://yichao-liang.github.io/robodisco-site/assets/gifs/fan.gif)](https://yichao-liang.github.io/robodisco-site/envs/fan.html) | Switch banks of fans to blow a ball through a walled grid to a target | ✅ | ✅ | ❌ |
| `robodisco/Float-v0` | [![](https://yichao-liang.github.io/robodisco-site/assets/gifs/float.gif)](https://yichao-liang.github.io/robodisco-site/envs/float.html) | Raise the water level to lift a floating block within reach | ✅ | ✅ | ❌ |
| `robodisco/Grow-v0` | [![](https://yichao-liang.github.io/robodisco-site/assets/gifs/grow.gif)](https://yichao-liang.github.io/robodisco-site/envs/grow.html) | Grow plants by pouring from the jug of matching colour | ✅ | ✅ | `oracle_process_planning` |
| `robodisco/IceRink-v0` | [![](https://yichao-liang.github.io/robodisco-site/assets/gifs/icerink.gif)](https://yichao-liang.github.io/robodisco-site/envs/icerink.html) | Push sliding tiles onto matching targets across a low-friction rink | ✅ | ✅ | ❌ |
| `robodisco/Laser-v0` | [![](https://yichao-liang.github.io/robodisco-site/assets/gifs/laser.gif)](https://yichao-liang.github.io/robodisco-site/envs/laser.html) | Place mirrors to route a laser beam onto the targets | ✅ | ✅ | ❌ |
| `robodisco/Launcher-v0` | [![](https://yichao-liang.github.io/robodisco-site/assets/gifs/launcher.gif)](https://yichao-liang.github.io/robodisco-site/envs/launcher.html) | Cock a spring launcher to knock only the top block off a tower | ✅ | ✅ | `oracle` |
| `robodisco/MagicBin-v0` | [![](https://yichao-liang.github.io/robodisco-site/assets/gifs/magic-bin.gif)](https://yichao-liang.github.io/robodisco-site/envs/magic-bin.html) | Make blocks vanish by dropping them in a bin that a switch activates | ❌ | ❌ | ❌ |
| `robodisco/Magnets-v0` | [![](https://yichao-liang.github.io/robodisco-site/assets/gifs/magnets.gif)](https://yichao-liang.github.io/robodisco-site/envs/magnets.html) | Steer coloured pieces into slots with a magnetic wand | ✅ | ✅ | `oracle` |
| `robodisco/Switch-v0` | [![](https://yichao-liang.github.io/robodisco-site/assets/gifs/switch.gif)](https://yichao-liang.github.io/robodisco-site/envs/switch.html) | Set a light's colour with a power switch and a colour switch | ✅ | ❌ | ❌ |

All columns were checked on October 1, 2026.
Every environment builds, resets, steps and renders through the Gymnasium wrapper with the Quick Start configuration.
Circuit, Laser and Switch fail with a PyBullet joint error when another environment was built earlier in the same process, so build them in a fresh process.

\* Boil and Bridge solve with `oracle_process_planning` under the settings of their oracle tests ([`test_oracle_process_planning_boil.py`](../../tests/approaches/test_oracle_process_planning_boil.py), [`test_oracle_process_planning_bridge.py`](../../tests/approaches/test_oracle_process_planning_bridge.py)); they do not solve in the default configuration.

Laser and BusyBoard start every training task from one layout and use a different one for testing; MagicBin keeps one layout and varies only the goal.

The other ❌ entries in the Oracle column, in the default configuration:
Barrier, MagicBin and Switch have no ground-truth operators or processes;
Domino and Fan fail an assertion while building their ground-truth processes;
Coffee fails to read a cup's pose;
Balloons, BusyBoard, Float, IceRink and Laser run but do not solve the test task.
The clips on the project page come from the benchmark settings or from scripted skill sequences, as each environment's page says.

## Skill libraries

`CFG.skill_library` selects the skills that `get_gt_options` builds:

- `"composite"` (the default) is each environment's own skills, such as `PickJug`, `SwitchFaucetOn` or `Push`, with task knowledge built in.
  These are the skills of the Skills column.
- `"primitive"` is one domain-general library, identical in every environment that supports it: `MoveTo[x, y, z, yaw, tilt]`, `MoveLinear[dx, dy, dz, step]`, `MoveUntilContact[dx, dy, dz, step]`, `Gripper[width, force]` and `Wait[steps]` (see [`skill_factories/primitives.py`](../ground_truth_models/skill_factories/primitives.py)).
  The agent supplies the grasp points, push strokes and release moments itself.
  Balloons, Boil, Bridge, Domino and Fan support it.

## Per-environment configuration

The RoboDisco envs read from predicators' global `CFG` object, which normally gets populated by `predicators/main.py`'s command-line parser.
For library use, set it explicitly:

```python
from predicators import utils
utils.reset_config({
    "num_train_tasks": 5,
    "num_test_tasks": 5,
    "blocks_num_blocks_train": [3, 4],
    "blocks_num_blocks_test": [4, 5],
})
```

You can also pass overrides per-make via the wrapper:

```python
env = robodisco.make(
    "robodisco/Blocks-v0",
    render_mode="rgb_array",
    cfg_overrides={"blocks_num_blocks_train": [4]},
)
```

See `predicators/settings.py` for the full list of available CFG fields.

## Standalone API (without the gym wrapper)

Each env can be used directly via predicators' `BaseEnv` interface:

```python
from predicators import utils
from predicators.envs.pybullet_blocks import PyBulletBlocksEnv
from predicators.structs import Action

utils.reset_config({"num_train_tasks": 5, "num_test_tasks": 5})
env = PyBulletBlocksEnv(use_gui=False)
state = env.reset("train", 0)
for _ in range(50):
    action = Action(env.action_space.sample())
    state = env.step(action)
```

This gives you direct access to `env.predicates`, `env.types`, `env.goal_predicates`, `env.get_train_tasks()`, etc., without flattening the state into a `Box` observation.

## Developing new envs

For a guide on writing new PyBullet environments, see [`docs/envs/pybullet-guide.md`](../../docs/envs/pybullet-guide.md).

## Predicators planning framework

These envs also power the predicators bilevel-planning research codebase.
See the [top-level README](../../README.md) for details.
