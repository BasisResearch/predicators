"""Safe execution and validation of agent-generated code proposals."""
import logging
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from predicators import utils
from predicators.structs import CausalProcess, ParameterizedOption, \
    Predicate, State, Task, Type


@dataclass
class ProposalBundle:
    """Accumulates proposals made by the agent during a single iteration."""
    proposed_types: Set[Type] = field(default_factory=set)
    proposed_predicates: Set[Predicate] = field(default_factory=set)
    augment_task_fn: Optional[Callable[[Task], Task]] = None
    augment_task_code: Optional[str] = None
    proposed_processes: Set[CausalProcess] = field(default_factory=set)
    proposed_options: Set[ParameterizedOption] = field(default_factory=set)
    errors: List[str] = field(default_factory=list)
    # Retractions: names of previously-proposed abstractions to remove
    retract_type_names: Set[str] = field(default_factory=set)
    retract_predicate_names: Set[str] = field(default_factory=set)
    retract_object_augmentor: bool = False
    retract_process_names: Set[str] = field(default_factory=set)
    retract_option_names: Set[str] = field(default_factory=set)


def exec_code_safely(code: str, context: Dict[str, Any],
                     expected_var: str) -> Tuple[Any, Optional[str]]:
    """Execute code in the given context and return the expected variable.

    Returns (result, None) on success, or (None, error_message) on
    failure.
    """
    try:
        exec(code, context)  # pylint: disable=exec-used
    except Exception:
        return None, traceback.format_exc()

    if expected_var not in context:
        return None, (f"Code executed successfully but did not define "
                      f"'{expected_var}'. Available names: "
                      f"{[k for k in context if not k.startswith('_')]}")
    return context[expected_var], None


def build_exec_context(
        types: Set[Type],
        predicates: Set[Predicate],
        options: Set[ParameterizedOption],
        extra_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Build a namespace for exec() with standard imports and current
    abstractions.

    Args:
        extra_context: Additional bindings to inject (e.g. option builder
            helpers). Merged after standard bindings so it can override them.
    """
    import numpy as np
    import torch
    from gym.spaces import Box

    from predicators.structs import CausalProcess, DerivedPredicate, \
        EndogenousProcess, ExogenousProcess, GroundAtom, LiftedAtom, \
        NSPredicate, Object, ParameterizedOption, Predicate, State, Task, \
        Type, Variable
    from predicators.utils import ConstantDelay, DiscreteGaussianDelay

    context: Dict[str, Any] = {}

    # Standard imports
    context["np"] = np
    context["numpy"] = np
    context["torch"] = torch
    context["Box"] = Box

    # Struct classes
    context["Type"] = Type
    context["Predicate"] = Predicate
    context["DerivedPredicate"] = DerivedPredicate
    context["NSPredicate"] = NSPredicate
    context["Object"] = Object
    context["Variable"] = Variable
    context["LiftedAtom"] = LiftedAtom
    context["GroundAtom"] = GroundAtom
    context["ExogenousProcess"] = ExogenousProcess
    context["EndogenousProcess"] = EndogenousProcess
    context["CausalProcess"] = CausalProcess
    context["ParameterizedOption"] = ParameterizedOption
    context["State"] = State
    context["Task"] = Task
    context["ConstantDelay"] = ConstantDelay
    context["DiscreteGaussianDelay"] = DiscreteGaussianDelay

    # Typing
    from typing import List as ListT
    from typing import Sequence
    from typing import Set as SetT
    context["List"] = ListT
    context["Set"] = SetT
    context["Sequence"] = Sequence

    # All current types as typename_type
    for t in types:
        context[f"{t.name}_type"] = t

    # All current predicates by name
    for p in predicates:
        context[p.name] = p
        # Also expose classifiers
        context[f"_{p.name}_holds"] = p._classifier

    # All current options by name
    for o in options:
        context[o.name] = o

    # Inject any extra bindings (e.g. option builder helpers)
    if extra_context:
        context.update(extra_context)

    return context


def validate_predicate(pred: Predicate, types: Set[Type],
                       example_state: State) -> Optional[str]:
    """Validate a predicate against current types and an example state.

    Returns None on success, or an error message string on failure.
    """
    # Check that all predicate types reference valid types
    for t in pred.types:
        if t not in types:
            return f"Predicate '{pred.name}' references unknown type '{t.name}'"

    # Try to evaluate the predicate on the example state
    try:
        utils.abstract(example_state, {pred})
    except Exception:
        return (f"Predicate '{pred.name}' failed evaluation on example state: "
                f"{traceback.format_exc()}")

    return None
