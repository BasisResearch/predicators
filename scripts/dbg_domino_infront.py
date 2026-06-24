"""Crack task-2 spurious InFront: use the EXACT approach machinery."""

from predicators import utils
from predicators.envs.pybullet_domino.env import PyBulletDominoEnv
from predicators.ground_truth_models import augment_task_with_helper_objects, \
    get_gt_helper_predicates
from predicators.structs import Task

utils.reset_config({
    "env": "pybullet_domino",
    "seed": 0,
    "num_train_tasks": 1,
    "num_test_tasks": 5,
    "domino_use_domino_blocks_as_target": True,
    "domino_use_continuous_place": True,
    "domino_restricted_push": True,
    "domino_initialize_at_finished_state": False,
    "domino_has_glued_dominos": False,
})

env = PyBulletDominoEnv()
tasks = env._generate_test_tasks()  # pylint: disable=protected-access
env_task = tasks[1]  # task 2
task = augment_task_with_helper_objects(Task(env_task.init, env_task.goal),
                                        "pybullet_domino")
s = task.init
helper_preds = get_gt_helper_predicates("pybullet_domino")
# How many distinct InFront predicate OBJECTS exist, and from where?
env_infronts = [p for p in env.predicates if p.name == "InFront"]
helper_infronts = [p for p in helper_preds if p.name == "InFront"]
print(f"env InFront objs: {len(env_infronts)} "
      f"derived={[type(p).__name__ for p in env_infronts]}")
print(f"helper InFront objs: {len(helper_infronts)} "
      f"derived={[type(p).__name__ for p in helper_infronts]}")
if env_infronts and helper_infronts:
    print("same object?", env_infronts[0] is helper_infronts[0])
    print("equal (==)?", env_infronts[0] == helper_infronts[0])

# The approach does: helpers | initial_predicates (helpers win on collision).
full_preds = helper_preds | set(env.predicates)
preds = {p.name: p for p in full_preds}
infront_in_full = [p for p in full_preds if p.name == "InFront"]
print(f"InFront objs in (helpers|env): {len(infront_in_full)} "
      f"-> {[type(p).__name__ for p in infront_in_full]}")

# Apply the FIX: drop base predicates whose name a helper already provides.
helper_names = {p.name for p in helper_preds}
fixed_preds = helper_preds | {
    p
    for p in env.predicates if p.name not in helper_names
}
fixed_infront = [p for p in fixed_preds if p.name == "InFront"]
fixed_atoms = utils.abstract(s, fixed_preds)
print(f"FIXED: InFront objs={len(fixed_infront)} "
      f"types={[type(p).__name__ for p in fixed_infront]}")
print("FIXED InFront atoms:",
      sorted(str(a) for a in fixed_atoms if a.predicate.name == "InFront"))

atoms = utils.abstract(s, full_preds)
atpos = {
    a.objects[0].name: a.objects[1]
    for a in atoms if a.predicate.name == "DominoAtPos"
}
print("=== DominoAtPos ===")
for d in sorted(atpos):
    loc = atpos[d]
    print(f"  {d} -> {loc.name}  "
          f"(xx={s.get(loc,'xx'):.4f} yy={s.get(loc,'yy'):.4f})")
print("=== InFront atoms ===")
for a in sorted(str(x) for x in atoms if x.predicate.name == "InFront"):
    print(" ", a)
print("=== InFrontDirection atoms ===")
for a in sorted(
        str(x) for x in atoms if x.predicate.name == "InFrontDirection"):
    print(" ", a)

c0, c1 = atpos["domino_0"], atpos["domino_1"]
n0 = tuple(float(v) for v in c0.name.split("_")[1:])
n1 = tuple(float(v) for v in c1.name.split("_")[1:])
print("=== manual d0 vs d1 ===")
print(f"  d0 cell name->coords: {n0}  feats: "
      f"({s.get(c0,'xx'):.4f},{s.get(c0,'yy'):.4f})")
print(f"  d1 cell name->coords: {n1}  feats: "
      f"({s.get(c1,'xx'):.4f},{s.get(c1,'yy'):.4f})")
print(f"  name dx={abs(n0[0]-n1[0]):.4f} dy={abs(n0[1]-n1[1]):.4f} "
      f"(pos_gap=0.098, tol={0.098*0.3:.4f})")
