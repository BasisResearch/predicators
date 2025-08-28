"""Ground-truth processes for the fan environment."""

from typing import Dict, Set

import torch

from predicators.ground_truth_models import GroundTruthProcessFactory
from predicators.settings import CFG
from predicators.structs import CausalProcess, EndogenousProcess, \
    ExogenousProcess, LiftedAtom, ParameterizedOption, Predicate, Type, \
    Variable
from predicators.utils import ConstantDelay, DiscreteGaussianDelay, \
    null_sampler


class PyBulletFanGroundTruthProcessFactory(GroundTruthProcessFactory):
    """Ground-truth processes for the fan environment.

    Endogenous: toggling switches (modeled as turning fans on/off).
    Exogenous: ball moves between grid positions when blown by active fans.
    """

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_fan"}

    @classmethod
    def get_processes(
            cls, env_name: str, types: Dict[str,
                                            Type], predicates: Dict[str,
                                                                    Predicate],
            options: Dict[str, ParameterizedOption]) -> Set[CausalProcess]:
        del env_name  # unused

        # Types
        robot_type = types["robot"]
        switch_type = types["switch"]
        fan_type = types["fan"]
        ball_type = types["ball"]
        location_type = types["loc"]
        side_type = types["side"]

        # Predicates
        FanOn = predicates["FanOn"]
        FanOff = predicates["FanOff"]
        BallAtLoc = predicates["BallAtLoc"]
        ClearPos = predicates["ClearPos"]
        FanFacingSide = predicates["FanOnSide"]
        SideOf = predicates["SideOf"]
        Controls = predicates["Controls"]
        OppositeFan = predicates["OppositeFan"]
        if not CFG.fan_known_controls_relation:
            SwitchOn = predicates["SwitchOn"]
            SwitchOff = predicates["SwitchOff"]

        # Options
        switch_option = None
        switch_on_option = None
        switch_off_option = None
        NoOp = options["NoOp"]

        if CFG.fan_combine_switch_on_off:
            switch_option = options["SwitchOnOff"]
        else:
            switch_on_option = options["SwitchOn"]
            switch_off_option = options["SwitchOff"]

        processes: Set[CausalProcess] = set()

        def _make_fan_toggle_process(
                name: str, start_predicate: Predicate,
                add_predicate: Predicate,
                delete_predicate: Predicate) -> EndogenousProcess:
            """Helper function to create fan on/off toggle processes."""
            robot = Variable("?robot", robot_type)
            if CFG.fan_known_controls_relation:
                controled_obj = Variable("?fan", fan_type)
            else:
                controled_obj = Variable("?switch", switch_type)
            parameters = [robot, controled_obj]
            option_vars = [robot, controled_obj]
            condition_at_start = {
                LiftedAtom(start_predicate, [controled_obj]),
            }
            add_effects: Set[LiftedAtom] = {
                LiftedAtom(add_predicate, [controled_obj]),
            }
            delete_effects = {
                LiftedAtom(delete_predicate, [controled_obj]),
            }
            delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(2.0),
                                                       sigma=torch.tensor(0.1))

            # Select the appropriate option based on configuration
            if CFG.fan_combine_switch_on_off:
                selected_option = switch_option
            else:
                if name in ["TurnFanOn", "TurnSwitchOn"]:
                    selected_option = switch_on_option
                else:
                    selected_option = switch_off_option

            return EndogenousProcess(name, parameters, condition_at_start,
                                     set(), set(), add_effects,
                                     delete_effects, delay_distribution,
                                     torch.tensor(1.0), selected_option,
                                     option_vars, null_sampler)

        # --- Endogenous processes: Switch toggling ---
        # For the harder setting of having to figure out which switch controls
        # which fan, we can have effects to be turn swtch on/off, and have it
        # to invent Control(fan, switch) predicate.
        if CFG.fan_known_controls_relation:
            processes.add(
                _make_fan_toggle_process("TurnFanOn", FanOff, FanOn, FanOff))
            processes.add(
                _make_fan_toggle_process("TurnFanOff", FanOn, FanOff, FanOn))
        else:
            processes.add(
                _make_fan_toggle_process("TurnSwitchOn", SwitchOff, SwitchOn,
                                         SwitchOff))
            processes.add(
                _make_fan_toggle_process("TurnSwitchOff", SwitchOn, SwitchOff,
                                         SwitchOn))

        # Noop
        robot = Variable("?robot", robot_type)
        parameters = [robot]
        option_vars = [robot]
        option = NoOp
        delay_distribution = ConstantDelay(1)
        noop_process = EndogenousProcess("NoOp", parameters, set(), set(),
                                         set(), set(),
                                         set(), delay_distribution,
                                         torch.tensor(1.0), option,
                                         option_vars, null_sampler)
        processes.add(noop_process)

        # --- Exogenous processes: Ball movement due to active fans ---
        fan = Variable("?fan", fan_type)
        op_fan = Variable("?op_fan", fan_type)
        switch = Variable("?switch", switch_type)
        op_switch = Variable("?op_switch", switch_type)
        ball = Variable("?ball", ball_type)
        pos1 = Variable("?pos1", location_type)
        pos2 = Variable("?pos2", location_type)
        dir = Variable("?dir", side_type)
        parameters = [ball, pos1, pos2, dir]
        condition_at_start = {
            LiftedAtom(BallAtLoc, [ball, pos1]),
            LiftedAtom(ClearPos, [pos2]),
            LiftedAtom(SideOf, [pos1, pos2, dir]),
            LiftedAtom(FanFacingSide, [fan, dir]),
        }
        if CFG.fan_known_controls_relation:
            parameters.extend([fan, op_fan])
            condition_at_start.add(LiftedAtom(OppositeFan, [fan, op_fan]))
            condition_at_start.add(LiftedAtom(FanOn, [fan]))
            condition_at_start.add(LiftedAtom(FanOff, [op_fan]))
        else:
            raise NotImplementedError
            # if not known, we add it here and let the agent to potentially
            # learn this
            condition_at_start.add(LiftedAtom(SwitchOn, [switch]))
            condition_at_start.add(LiftedAtom(Controls, [switch, fan]))

        condition_overall = set(condition_at_start)
        add_effects = {
            LiftedAtom(BallAtLoc, [ball, pos2]),
        }
        delete_effects = {
            LiftedAtom(BallAtLoc, [ball, pos1]),
        }
        delay_distribution = DiscreteGaussianDelay(mu=torch.tensor(4.0),
                                                   sigma=torch.tensor(0.1))
        move_to_side = ExogenousProcess("MoveToSide", parameters,
                                        condition_at_start, condition_overall,
                                        set(), add_effects, delete_effects,
                                        delay_distribution, torch.tensor(1.0))
        processes.add(move_to_side)
        return processes
