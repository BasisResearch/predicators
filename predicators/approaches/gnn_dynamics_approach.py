"""A neural dynamics baseline (paper arm C5): a GNN transition model over the
object-centric state features, trained on option-level transitions from the
same interaction data the agent arms collect, and planned against by random
shooting through the learned model.

The model predicts, for a state and a ground option, the change of every
object feature and the number of low-level steps the option takes. The
last ``gnn_dynamics_history_len`` pre-option states ride along as node
features so a mechanism that is not visible in one observation (a cure
that started a few options ago) can be inferred from the recent past.
Planning samples random applicable options, rolls them through the
model, and accepts the first sequence whose predicted final state
satisfies the goal under the env's (oracle) predicates; by default the
plan is re-shot from the observed state every time an option terminates.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import dill as pkl
import numpy as np
import torch
import torch.nn
import torch.optim
from gym.spaces import Box
from torch.utils.data import DataLoader

from predicators import utils
from predicators.approaches import ApproachFailure, ApproachTimeout, \
    BaseApproach
from predicators.explorers import create_explorer
from predicators.gnn.gnn import EncodeProcessDecode, setup_graph_net
from predicators.gnn.gnn_utils import GraphDictDataset, compute_normalizers, \
    get_single_model_prediction, graph_batch_collate, normalize_graph, \
    train_model
from predicators.nsrt_learning.segmentation import segment_trajectory
from predicators.settings import CFG
from predicators.structs import Action, Dataset, DummyOption, \
    InteractionRequest, InteractionResult, LowLevelTrajectory, \
    ParameterizedOption, Predicate, State, Task, Type, _Option

# One training example: the recent pre-option states (oldest first), the
# pre-option state, the ground option, the post-option state, and the
# option's low-level step count.
_Example = Tuple[List[State], State, _Option, State, int]


class GNNDynamicsShootingApproach(BaseApproach):
    """GNN transition model over object features + shooting planner."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        self._sorted_options = sorted(self._initial_options,
                                      key=lambda o: o.name)
        self._trajectories: List[LowLevelTrajectory] = []
        self._online_learning_cycle = 0
        self._requests_train_task_idxs: List[int] = []
        self._gnn: Optional[EncodeProcessDecode] = None
        self._data_exemplar: Optional[Tuple[Dict, Dict]] = None
        self._input_normalizers: Optional[Dict] = None
        self._target_normalizers: Optional[Dict] = None
        # Node feature layout: type one-hot, current features, history
        # features (per lag), option-argument slot one-hot.
        self._type_to_index: Dict[str, int] = {}
        self._feat_to_index: Dict[str, int] = {}
        self._max_option_objects = 0
        self._max_option_params = 0
        self._mse_loss = torch.nn.MSELoss()

    @classmethod
    def get_name(cls) -> str:
        return "gnn_dynamics_shooting"

    @property
    def is_learning_based(self) -> bool:
        return True

    # ── Data ─────────────────────────────────────────────────────

    def _generate_examples(self) -> List[_Example]:
        """Option-level transitions from every stored trajectory."""
        examples: List[_Example] = []
        history_len = CFG.gnn_dynamics_history_len
        for traj in self._trajectories:
            if not traj.actions or not traj.actions[0].has_option():
                continue
            segments = segment_trajectory(traj, self._initial_predicates)
            pre_states: List[State] = []
            for segment in segments:
                state = segment.states[0]
                history = pre_states[-history_len:] if history_len else []
                examples.append((list(history), state, segment.get_option(),
                                 segment.states[-1], len(segment.actions)))
                pre_states.append(state)
        return examples

    def _setup_fields(self, examples: Sequence[_Example]) -> None:
        types: Set[str] = set()
        feats: Set[str] = set()
        max_objects = 0
        max_params = 0
        for _, state, option, _, _ in examples:
            for obj in state:
                types.add(obj.type.name)
                feats.update(obj.type.feature_names)
            max_objects = max(max_objects, len(option.objects))
            max_params = max(max_params, option.params.shape[0])
        # Every option's argument count and parameter box, not just the
        # ones seen: a test-time sample of an unseen option must graphify.
        for param_opt in self._sorted_options:
            max_objects = max(max_objects, len(param_opt.types))
            max_params = max(max_params, param_opt.params_space.shape[0])
        self._type_to_index = {t: i for i, t in enumerate(sorted(types))}
        self._feat_to_index = {f: i for i, f in enumerate(sorted(feats))}
        self._max_option_objects = max_objects
        self._max_option_params = max_params

    def _feature_row(self, state: State, obj: Any) -> np.ndarray:
        row = np.zeros(len(self._feat_to_index))
        for feat, val in zip(obj.type.feature_names, state[obj]):
            row[self._feat_to_index[feat]] = val
        return row

    def _graphify_input(self, history: Sequence[State], state: State,
                        option: _Option) -> Tuple[Dict, Dict[Any, int]]:
        objects = list(state)
        object_to_node = {o: i for i, o in enumerate(objects)}
        n_obj = len(objects)
        n_types = len(self._type_to_index)
        n_feats = len(self._feat_to_index)
        history_len = CFG.gnn_dynamics_history_len
        n_node_feats = (n_types + n_feats + history_len * n_feats +
                        self._max_option_objects)
        nodes = np.zeros((n_obj, n_node_feats))
        arg_slots = {o: i for i, o in enumerate(option.objects)}
        for obj in objects:
            i = object_to_node[obj]
            nodes[i, self._type_to_index[obj.type.name]] = 1
            current = self._feature_row(state, obj)
            nodes[i, n_types:n_types + n_feats] = current
            # History as differences to the current features, most
            # recent lag first; missing lags stay zero (no change).
            for lag in range(history_len):
                if lag < len(history):
                    past = history[-1 - lag]
                    if obj in past.data:
                        past_row = self._feature_row(past, obj)
                        start = n_types + n_feats * (1 + lag)
                        nodes[i, start:start + n_feats] = past_row - current
            if obj in arg_slots:
                nodes[i, n_types + n_feats * (1 + history_len) +
                      arg_slots[obj]] = 1
        # Fully connected (no self loops) so effects can flow between
        # objects; edge features: constant, both endpoints are arguments.
        senders, receivers, edges = [], [], []
        for s in range(n_obj):
            for r in range(n_obj):
                if s == r:
                    continue
                senders.append(s)
                receivers.append(r)
                both = float(objects[s] in arg_slots
                             and objects[r] in arg_slots)
                edges.append([1.0, both])
        n_edge = len(edges)
        onehot = np.zeros(len(self._sorted_options))
        onehot[self._sorted_options.index(option.parent)] = 1
        params = np.zeros(self._max_option_params)
        params[:option.params.shape[0]] = option.params
        graph = {
            "n_node": np.array(n_obj),
            "nodes": nodes,
            "n_edge": np.reshape(n_edge, [1]).astype(np.int64),
            "edges": np.reshape(edges, [n_edge, 2]),
            "senders": np.reshape(senders, [n_edge]).astype(np.int64),
            "receivers": np.reshape(receivers, [n_edge]).astype(np.int64),
            "globals": np.r_[onehot, params],
        }
        return graph, object_to_node

    def _graphify_target(self, state: State, next_state: State,
                         num_actions: int, graph_input: Dict,
                         object_to_node: Dict[Any, int]) -> Dict:
        n_obj = len(object_to_node)
        nodes = np.zeros((n_obj, len(self._feat_to_index)))
        for obj, i in object_to_node.items():
            nodes[i] = (self._feature_row(next_state, obj) -
                        self._feature_row(state, obj))
        return {
            "n_node": graph_input["n_node"],
            "nodes": nodes,
            "n_edge": graph_input["n_edge"],
            "edges": np.zeros((int(graph_input["n_edge"][0]), 1)),
            "senders": graph_input["senders"],
            "receivers": graph_input["receivers"],
            "globals": np.array([num_actions / float(max(CFG.horizon, 1))]),
        }

    # ── Learning ─────────────────────────────────────────────────

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        self._trajectories = list(dataset.trajectories)
        self._learn_model()
        self._save(None)

    def get_interaction_requests(self) -> List[InteractionRequest]:
        explorer = create_explorer(CFG.explorer, self._initial_predicates,
                                   self._initial_options, self._types,
                                   self._action_space, self._train_tasks)
        requests: List[InteractionRequest] = []
        self._requests_train_task_idxs = []
        for _ in range(CFG.online_nsrt_learning_requests_per_cycle):
            task_idx = int(self._rng.choice(len(self._train_tasks)))
            policy, termination_fn = explorer.get_exploration_strategy(
                task_idx, CFG.timeout)
            requests.append(
                InteractionRequest(train_task_idx=task_idx,
                                   act_policy=policy,
                                   query_policy=lambda s: None,
                                   termination_function=termination_fn))
            self._requests_train_task_idxs.append(task_idx)
        return requests

    def restore_interaction_requests(self, train_task_idxs: List[int]) -> None:
        self._requests_train_task_idxs = list(train_task_idxs)

    def learn_from_interaction_results(
            self, results: Sequence[InteractionResult]) -> None:
        assert len(results) == len(self._requests_train_task_idxs)
        for task_idx, result in zip(self._requests_train_task_idxs, results):
            self._trajectories.append(
                LowLevelTrajectory(result.states,
                                   result.actions,
                                   _is_demo=False,
                                   _train_task_idx=task_idx))
        self._learn_model()
        self._save(self._online_learning_cycle)
        self._online_learning_cycle += 1

    def _learn_model(self) -> None:
        examples = self._generate_examples()
        if not examples:
            logging.warning("GNN dynamics: no option-level transitions yet; "
                            "keeping the previous model.")
            return
        self._setup_fields(examples)
        graph_inputs, graph_targets = [], []
        for history, state, option, next_state, num_actions in examples:
            graph_input, object_to_node = self._graphify_input(
                history, state, option)
            graph_inputs.append(graph_input)
            graph_targets.append(
                self._graphify_target(state, next_state, num_actions,
                                      graph_input, object_to_node))
        self._data_exemplar = (graph_inputs[0], graph_targets[0])
        self._gnn = setup_graph_net(GraphDictDataset([graph_inputs[0]],
                                                     [graph_targets[0]]),
                                    num_steps=CFG.gnn_num_message_passing,
                                    layer_size=CFG.gnn_layer_size)
        if CFG.gnn_do_normalization:
            self._input_normalizers = compute_normalizers(graph_inputs)
            self._target_normalizers = compute_normalizers(graph_targets)
            graph_inputs = [
                normalize_graph(g, self._input_normalizers)
                for g in graph_inputs
            ]
            graph_targets = [
                normalize_graph(g, self._target_normalizers)
                for g in graph_targets
            ]
        num_validation = (max(1, int(len(examples) * 0.1))
                          if CFG.gnn_use_validation_set else 0)
        train_set = GraphDictDataset(graph_inputs[num_validation:],
                                     graph_targets[num_validation:])
        val_set = GraphDictDataset(graph_inputs[:num_validation],
                                   graph_targets[:num_validation])
        dataloaders = {
            "train":
            DataLoader(train_set,
                       batch_size=CFG.gnn_batch_size,
                       shuffle=False,
                       num_workers=0,
                       collate_fn=graph_batch_collate),
            "val":
            DataLoader(val_set,
                       batch_size=CFG.gnn_batch_size,
                       shuffle=False,
                       num_workers=0,
                       collate_fn=graph_batch_collate),
        }
        optimizer = torch.optim.Adam(self._gnn.parameters(),
                                     lr=CFG.gnn_learning_rate,
                                     weight_decay=CFG.gnn_weight_decay)
        logging.info(
            "Training GNN dynamics on %d option transitions from "
            "%d trajectories.", len(examples), len(self._trajectories))
        best = train_model(self._gnn,
                           dataloaders,
                           optimizer=optimizer,
                           criterion=self._mse_loss,
                           global_criterion=self._mse_loss,
                           num_epochs=CFG.gnn_num_epochs,
                           do_validation=CFG.gnn_use_validation_set)
        self._gnn.load_state_dict(best)

    # ── Checkpointing ────────────────────────────────────────────

    def _save(self, online_learning_cycle: Optional[int]) -> None:
        info = {
            "trajectories":
            self._trajectories,
            "online_learning_cycle":
            self._online_learning_cycle,
            "exemplar":
            self._data_exemplar,
            "state_dict":
            (self._gnn.state_dict() if self._gnn is not None else None),
            "type_to_index":
            self._type_to_index,
            "feat_to_index":
            self._feat_to_index,
            "max_option_objects":
            self._max_option_objects,
            "max_option_params":
            self._max_option_params,
            "input_normalizers":
            self._input_normalizers,
            "target_normalizers":
            self._target_normalizers,
        }
        path = (f"{utils.get_approach_save_path_str()}_"
                f"{online_learning_cycle}.gnn")
        with open(path, "wb") as f:
            pkl.dump(info, f)

    def load(self, online_learning_cycle: Optional[int]) -> None:
        path = (f"{utils.get_approach_load_path_str()}_"
                f"{online_learning_cycle}.gnn")
        with open(path, "rb") as f:
            info = pkl.load(f)
        self._trajectories = info["trajectories"]
        self._online_learning_cycle = info["online_learning_cycle"]
        self._data_exemplar = info["exemplar"]
        self._type_to_index = info["type_to_index"]
        self._feat_to_index = info["feat_to_index"]
        self._max_option_objects = info["max_option_objects"]
        self._max_option_params = info["max_option_params"]
        self._input_normalizers = info["input_normalizers"]
        self._target_normalizers = info["target_normalizers"]
        if info["state_dict"] is not None and self._data_exemplar is not None:
            example_input, example_target = self._data_exemplar
            self._gnn = setup_graph_net(GraphDictDataset([example_input],
                                                         [example_target]),
                                        num_steps=CFG.gnn_num_message_passing,
                                        layer_size=CFG.gnn_layer_size)
            self._gnn.load_state_dict(info["state_dict"])

    # ── Prediction ───────────────────────────────────────────────

    def predict_next_state(self, history: Sequence[State], state: State,
                           option: _Option) -> Tuple[State, int]:
        """The model's post-option state and low-level step count."""
        assert self._gnn is not None, "Learn a model before predicting."
        graph_input, object_to_node = self._graphify_input(
            history, state, option)
        if CFG.gnn_do_normalization:
            assert self._input_normalizers is not None
            graph_input = normalize_graph(graph_input, self._input_normalizers)
        out = get_single_model_prediction(self._gnn, graph_input)
        if CFG.gnn_do_normalization:
            assert self._target_normalizers is not None
            out = normalize_graph(out, self._target_normalizers, invert=True)
        next_state = state.copy()
        for obj, i in object_to_node.items():
            delta = out["nodes"][i]
            for feat in obj.type.feature_names:
                next_state.set(
                    obj, feat,
                    state.get(obj, feat) +
                    float(delta[self._feat_to_index[feat]]))
        num_actions = max(
            1, int(round(float(out["globals"][0]) * max(CFG.horizon, 1))))
        return next_state, num_actions

    # ── Planning ─────────────────────────────────────────────────

    def _shoot(self, task: Task, init_state: State, history: Sequence[State],
               deadline: float) -> Optional[List[_Option]]:
        """Random shooting through the learned model.

        Returns the first sampled option sequence whose predicted final
        state satisfies the goal, or None when the tries or the deadline
        run out.
        """
        for _ in range(CFG.gnn_dynamics_shooting_max_tries):
            if time.perf_counter() > deadline:
                return None
            state = init_state
            past = list(history)
            plan: List[_Option] = []
            num_actions = 0
            for _ in range(CFG.gnn_dynamics_max_plan_length):
                if task.goal_holds(state):
                    return plan
                option = utils.sample_applicable_option(
                    self._sorted_options, state, self._rng)
                if option is None:
                    break
                next_state, k = self.predict_next_state(past, state, option)
                plan.append(option)
                past.append(state)
                state = next_state
                num_actions += k
                if num_actions > CFG.horizon:
                    break
            if task.goal_holds(state) and plan:
                return plan
        return None

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        if self._gnn is None:
            raise ApproachFailure("GNN dynamics model has not been learned.")
        deadline = time.perf_counter() + timeout
        observed: List[State] = []
        plan: List[_Option] = []
        cur_option: _Option = DummyOption
        replan = CFG.gnn_dynamics_replan_every_option

        def _next_option(state: State) -> _Option:
            nonlocal plan
            if replan or not plan:
                shot = self._shoot(task, state, observed, deadline)
                if shot is None:
                    if time.perf_counter() > deadline:
                        raise ApproachTimeout(
                            "GNN dynamics shooting timed out.")
                    raise ApproachFailure(
                        "GNN dynamics shooting found no plan that reaches "
                        "the goal under the learned model.")
                plan = list(shot)
            option = plan.pop(0)
            if not option.initiable(state):
                raise ApproachFailure(
                    f"Planned option {option.name} is not initiable in the "
                    "observed state.")
            return option

        def _policy(state: State) -> Action:
            nonlocal cur_option
            if cur_option is DummyOption or cur_option.terminal(state):
                cur_option = _next_option(state)
                observed.append(state)
            return cur_option.policy(state)

        return _policy
