"""Ground-truth processes for the boil environments."""
from typing import Dict, Set

import numpy as np

from predicators.ground_truth_models import GroundTruthProcessFactory
from predicators.settings import CFG
from predicators.structs import CausalProcess, EndogenousProcess, \
    ExogenousProcess, LiftedAtom, ParameterizedOption, Predicate, Type, \
    Variable
from predicators.utils import CMPDelay, ConstantDelay, GaussianDelay, \
    null_sampler


class PyBulletBoilGroundTruthProcessFactory(GroundTruthProcessFactory):
    """Ground-truth processes for the boil environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_boil"}

    @staticmethod
    def get_processes(
            env_name: str, types: Dict[str, Type], predicates: Dict[str,
                                                                    Predicate],
            options: Dict[str, ParameterizedOption]) -> Set[CausalProcess]:
        # Types
        robot_type = types["robot"]
        jug_type = types["jug"]
        burner_type = types["burner"]
        faucet_type = types["faucet"]
        switch_type = types["switch"]

        # Predicates
        HandEmpty = predicates["HandEmpty"]
        Holding = predicates["Holding"]
        JugOnBurner = predicates["JugOnBurner"]
        JugUnderFaucet = predicates["JugUnderFaucet"]
        JugFilled = predicates["JugFilled"]
        WaterSpilled = predicates["WaterSpilled"]
        NoWaterSpilled = predicates["NoWaterSpilled"]
        NoJugUnderFaucet = predicates["NoJugUnderFaucet"]
        FaucetOn = predicates["FaucetOn"]
        FaucetOff = predicates["FaucetOff"]
        BurnerOn = predicates["BurnerOn"]
        BurnerOff = predicates["BurnerOff"]

        WaterBoiled = predicates["WaterBoiled"]

        # Options
        PickJug = options["PickJug"]
        PlaceOnBurner = options["PlaceOnBurner"]
        PlaceUnderFaucet = options["PlaceUnderFaucet"]
        # Having swtich for each because of the type
        SwitchFaucetOn = options["SwitchFaucetOn"]
        SwitchFaucetOff = options["SwitchFaucetOff"]
        SwitchBurnerOn = options["SwitchBurnerOn"]
        SwitchBurnerOff = options["SwitchBurnerOff"]
        NoOp = options["NoOp"]

        # Create a random number generator
        rng = np.random.default_rng(CFG.seed)

        processes = set()

        # --- Endogenous Processes / Durative Actions ---
        # PickJugFromFaucet
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        faucet = Variable("?faucet", faucet_type)
        parameters = [robot, jug, faucet]
        option_vars = [robot, jug]
        option = PickJug
        condition_at_start = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(JugUnderFaucet, [jug, faucet]),
        }
        add_effects = {
            LiftedAtom(Holding, [robot, jug]),
            LiftedAtom(NoJugUnderFaucet, [faucet]),
        }
        delete_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(JugUnderFaucet, [jug, faucet]),
        }
        if CFG.boil_use_constant_delay:
            delay_distribution = ConstantDelay(4)
        else:
            delay_distribution = CMPDelay(1, 1, rng)
        pick_jug_from_faucet_process = EndogenousProcess(
            "PickJugFromFaucet", parameters, condition_at_start, set(), set(),
            add_effects, delete_effects, delay_distribution, 1.0, option,
            option_vars, null_sampler)
        processes.add(pick_jug_from_faucet_process)

        # PickJugFromOutsideFaucet
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        faucet = Variable("?faucet", faucet_type)
        parameters = [robot, jug, faucet]
        option_vars = [robot, jug]
        option = PickJug
        condition_at_start = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(NoJugUnderFaucet, [faucet]),
        }
        add_effects = {
            LiftedAtom(Holding, [robot, jug]),
        }
        delete_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(NoJugUnderFaucet, [faucet]),
        }
        if CFG.boil_use_constant_delay:
            delay_distribution = ConstantDelay(3)
        else:
            delay_distribution = CMPDelay(1, 1, rng)
        pick_jug_outside_faucet_process = EndogenousProcess(
            "PickJugFromOutsideFaucet", parameters, condition_at_start, set(),
            set(), add_effects, delete_effects, delay_distribution, 1.0,
            option, option_vars, null_sampler)
        processes.add(pick_jug_outside_faucet_process)

        # PlaceOnBurner
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        burner = Variable("?burner", burner_type)
        parameters = [robot, jug, burner]
        option_vars = [robot, burner]
        option = PlaceOnBurner
        condition_at_start = {
            LiftedAtom(Holding, [robot, jug]),
        }
        add_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(JugOnBurner, [jug, burner]),
        }
        delete_effects = {
            LiftedAtom(Holding, [robot, jug]),
        }
        if CFG.boil_use_constant_delay:            
            delay_distribution = ConstantDelay(5)
        else:
            delay_distribution = CMPDelay(1, 1, rng)
        place_on_burner_process = EndogenousProcess(
            "PlaceOnBurner", parameters, condition_at_start, set(), set(),
            add_effects, delete_effects, delay_distribution, 1.0, option,
            option_vars, null_sampler)
        processes.add(place_on_burner_process)

        # PlaceUnderFaucet
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        faucet = Variable("?faucet", faucet_type)
        parameters = [robot, jug, faucet]
        option_vars = [robot, faucet]
        option = PlaceUnderFaucet
        condition_at_start = {
            LiftedAtom(Holding, [robot, jug]),
        }
        add_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(JugUnderFaucet, [jug, faucet]),
        }
        delete_effects = {
            LiftedAtom(Holding, [robot, jug]),
            LiftedAtom(NoJugUnderFaucet, [faucet]),
        }
        if CFG.boil_use_constant_delay:
            delay_distribution = ConstantDelay(3)
        else:
            delay_distribution = CMPDelay(1, 1, rng)
        place_under_faucet_process = EndogenousProcess(
            "PlaceUnderFaucet", parameters, condition_at_start, set(), set(),
            add_effects, delete_effects, delay_distribution, 1.0, option,
            option_vars, null_sampler)
        processes.add(place_under_faucet_process)

        # SwitchFaucetOn
        robot = Variable("?robot", robot_type)
        faucet = Variable("?faucet", faucet_type)
        parameters = [robot, faucet]
        option_vars = [robot, faucet]
        option = SwitchFaucetOn
        condition_at_start = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(FaucetOff, [faucet]),
        }
        add_effects = {
            LiftedAtom(FaucetOn, [faucet]),
        }
        delete_effects = {
            LiftedAtom(FaucetOff, [faucet]),
        }
        if CFG.boil_use_constant_delay:
            delay_distribution = ConstantDelay(1)
        else:
            delay_distribution = CMPDelay(1, 1, rng)
        switch_faucet_on_process = EndogenousProcess(
            "SwitchFaucetOn", parameters, condition_at_start, set(), set(),
            add_effects, delete_effects, delay_distribution, 1.0, option,
            option_vars, null_sampler)
        processes.add(switch_faucet_on_process)

        # SwitchFaucetOff
        robot = Variable("?robot", robot_type)
        faucet = Variable("?faucet", faucet_type)
        parameters = [robot, faucet]
        option_vars = [robot, faucet]
        option = SwitchFaucetOff
        condition_at_start = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(FaucetOn, [faucet]),
        }
        add_effects = {
            LiftedAtom(FaucetOff, [faucet]),
        }
        delete_effects = {
            LiftedAtom(FaucetOn, [faucet]),
        }
        if CFG.boil_use_constant_delay:
            delay_distribution = ConstantDelay(1)
        else:
            delay_distribution = CMPDelay(1, 1, rng)
        switch_faucet_off_process = EndogenousProcess(
            "SwitchFaucetOff", parameters, condition_at_start, set(), set(),
            add_effects, delete_effects, delay_distribution, 1.0, option,
            option_vars, null_sampler)
        processes.add(switch_faucet_off_process)

        # SwitchBurnerOn
        robot = Variable("?robot", robot_type)
        burner = Variable("?burner", burner_type)
        parameters = [robot, burner]
        option_vars = [robot, burner]
        option = SwitchBurnerOn
        condition_at_start = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(BurnerOff, [burner]),
        }
        add_effects = {
            LiftedAtom(BurnerOn, [burner]),
        }
        delete_effects = {
            LiftedAtom(BurnerOff, [burner]),
        }
        if CFG.boil_use_constant_delay:
            delay_distribution = ConstantDelay(3)
        else:
            delay_distribution = CMPDelay(1, 1, rng)
        switch_burner_on_process = EndogenousProcess(
            "SwitchBurnerOn", parameters, condition_at_start, set(), set(),
            add_effects, delete_effects, delay_distribution, 1.0, option,
            option_vars, null_sampler)
        processes.add(switch_burner_on_process)

        # SwitchBurnerOff
        robot = Variable("?robot", robot_type)
        burner = Variable("?burner", burner_type)
        parameters = [robot, burner]
        option_vars = [robot, burner]
        option = SwitchBurnerOff
        condition_at_start = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(BurnerOn, [burner]),
        }
        add_effects = {
            LiftedAtom(BurnerOff, [burner]),
        }
        delete_effects = {
            LiftedAtom(BurnerOn, [burner]),
        }
        if CFG.boil_use_constant_delay:
            delay_distribution = ConstantDelay(1)
        else:
            delay_distribution = CMPDelay(1, 1, rng)
        switch_burner_off_process = EndogenousProcess(
            "SwitchBurnerOff", parameters, condition_at_start, set(), set(),
            add_effects, delete_effects, delay_distribution, 1.0, option,
            option_vars, null_sampler)
        processes.add(switch_burner_off_process)

        # Noop
        robot = Variable("?robot", robot_type)
        parameters = [robot]
        option_vars = [robot]
        option = NoOp
        # delay_distribution = GaussianDelay(3, 2, rng)
        delay_distribution = ConstantDelay(1)
        noop_process = EndogenousProcess("NoOp", parameters, set(), set(),
                                         set(), set(), set(),
                                         delay_distribution, 1.0, option,
                                         option_vars, null_sampler)
        processes.add(noop_process)

        # --- Exogenous Processes ---
        # FillJug
        jug = Variable("?jug", jug_type)
        faucet = Variable("?faucet", faucet_type)
        parameters = [jug, faucet]
        option_vars = [jug]
        condition_at_start = {
            LiftedAtom(JugUnderFaucet, [jug, faucet]),
            LiftedAtom(FaucetOn, [faucet]),
        }
        condition_overall = {
            LiftedAtom(JugUnderFaucet, [jug, faucet]),
            LiftedAtom(FaucetOn, [faucet]),
        }
        add_effects = {
            LiftedAtom(JugFilled, [jug]),
        }
        if CFG.boil_use_constant_delay:
            delay_distribution = ConstantDelay(4)
        else:
            delay_distribution = CMPDelay(1, 1, rng)
        fill_jug_process = ExogenousProcess("FillJug", parameters,
                                            condition_at_start,
                                            condition_overall, set(),
                                            add_effects, {},
                                            delay_distribution, 1.0)
        processes.add(fill_jug_process)

        # OverfillJug
        jug = Variable("?jug", jug_type)
        faucet = Variable("?faucet", faucet_type)
        parameters = [jug, faucet]
        condition_at_start = {
            LiftedAtom(JugUnderFaucet, [jug, faucet]),
            LiftedAtom(FaucetOn, [faucet]),
            LiftedAtom(JugFilled, [jug]),
        }
        condition_overall = {
            LiftedAtom(JugUnderFaucet, [jug, faucet]),
            LiftedAtom(FaucetOn, [faucet]),
            LiftedAtom(JugFilled, [jug]),
        }
        add_effects = {
            LiftedAtom(WaterSpilled, []),
        }
        delete_effects = {
            LiftedAtom(NoWaterSpilled, []),
        }
        if CFG.boil_use_constant_delay:
            delay_distribution = ConstantDelay(3)
        else:
            delay_distribution = CMPDelay(1, 1, rng)
        overfill_jug_process = ExogenousProcess("OverfillJug", parameters,
                                                condition_at_start,
                                                condition_overall, set(),
                                                add_effects, delete_effects,
                                                delay_distribution, 1.0)
        processes.add(overfill_jug_process)

        # Spill
        faucet = Variable("?faucet", faucet_type)
        parameters = [faucet]
        condition_at_start = {
            LiftedAtom(NoJugUnderFaucet, [faucet]),
            LiftedAtom(FaucetOn, [faucet]),
        }
        add_effects = {
            LiftedAtom(WaterSpilled, []),
        }
        delete_effects = {
            LiftedAtom(NoWaterSpilled, []),
        }
        if CFG.boil_use_constant_delay:
            delay_distribution = ConstantDelay(3)
        else:
            delay_distribution = CMPDelay(1, 1, rng)
        spill_process = ExogenousProcess("Spill",
                                         parameters, condition_at_start, set(),
                                         set(), add_effects, delete_effects,
                                         delay_distribution, 1.0)
        processes.add(spill_process)

        # Boil
        burner = Variable("?burner", burner_type)
        jug = Variable("?jug", jug_type)
        parameters = [burner, jug]
        condition_at_start = {
            LiftedAtom(JugOnBurner, [jug, burner]),
            LiftedAtom(JugFilled, [jug]),
            LiftedAtom(BurnerOn, [burner]),
        }
        condition_overall = {
            LiftedAtom(JugOnBurner, [jug, burner]),
            LiftedAtom(JugFilled, [jug]),
            LiftedAtom(BurnerOn, [burner]),
        }
        add_effects = {
            LiftedAtom(WaterBoiled, [jug]),
        }
        if CFG.boil_use_constant_delay:
            delay_distribution = ConstantDelay(5)
        else:
            delay_distribution = CMPDelay(1, 1, rng)
        boil_process = ExogenousProcess("Boil", parameters, condition_at_start,
                                        condition_overall, set(), add_effects,
                                        set(), delay_distribution, 1.0)
        processes.add(boil_process)

        return processes
