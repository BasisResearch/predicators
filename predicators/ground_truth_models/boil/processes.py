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

        # Predicates
        HandEmpty = predicates["HandEmpty"]
        Holding = predicates["Holding"]
        JugAtBurner = predicates["JugAtBurner"]
        JugAtFaucet = predicates["JugAtFaucet"]
        NoJugAtFaucet = predicates["NoJugAtFaucet"]
        JugNotAtBurnerOrFaucet = predicates["JugNotAtBurnerOrFaucet"]
        NoJugAtFaucetOrJugAtFaucetAndFilled = predicates[
            "JugNotAtFaucetOrAtFaucetAndFilled"]
        JugFilled = predicates["JugFilled"]
        JugNotFilled = predicates["JugNotFilled"]
        # WaterSpilled = predicates["WaterSpilled"]
        NoWaterSpilled = predicates["NoWaterSpilled"]
        FaucetOn = predicates["FaucetOn"]
        FaucetOff = predicates["FaucetOff"]
        BurnerOn = predicates["BurnerOn"]
        BurnerOff = predicates["BurnerOff"]
        WaterBoiled = predicates["WaterBoiled"]
        if CFG.boil_goal == "human_happy":
            HumanHappy = predicates["HumanHappy"]
        elif CFG.boil_goal == "task_completed":
            TaskCompleted = predicates["TaskCompleted"]

        # Options
        PickJug = options["PickJug"]
        PlaceOnBurner = options["PlaceOnBurner"]
        PlaceUnderFaucet = options["PlaceUnderFaucet"]
        PlaceOutsideBurnerAndFaucet = options["PlaceOutsideBurnerAndFaucet"]
        # Having swtich for each because of the type
        SwitchFaucetOn = options["SwitchFaucetOn"]
        SwitchFaucetOff = options["SwitchFaucetOff"]
        SwitchBurnerOn = options["SwitchBurnerOn"]
        SwitchBurnerOff = options["SwitchBurnerOff"]
        NoOp = options["NoOp"]
        if CFG.boil_goal == "task_completed":
            DeclareComplete = options["DeclareComplete"]

        # Create a random number generator
        rng = np.random.default_rng(CFG.seed)

        processes: Set[CausalProcess] = set()

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
            LiftedAtom(JugAtFaucet, [jug, faucet]),
        }
        add_effects = {
            LiftedAtom(Holding, [robot, jug]),
            LiftedAtom(NoJugAtFaucet, [faucet]),
        }
        delete_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(JugAtFaucet, [jug, faucet]),
        }
        if CFG.boil_use_constant_delay:
            delay_distribution = ConstantDelay(4)
        elif CFG.boil_use_normal_delay:
            delay_distribution = GaussianDelay(mean=4, std=0.2)
        else:
            delay_distribution = CMPDelay(80, 3)
        pick_jug_from_faucet_process = EndogenousProcess(
            "PickJugFromFaucet", parameters, condition_at_start, set(), set(),
            add_effects, delete_effects, delay_distribution, 1.0, option,
            option_vars, null_sampler)
        processes.add(pick_jug_from_faucet_process)

        # PickJugFromBurner
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        burner = Variable("?burner", burner_type)
        parameters = [robot, jug, burner]
        option_vars = [robot, jug]
        option = PickJug
        condition_at_start = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(JugAtBurner, [jug, burner]),
        }
        add_effects = {
            LiftedAtom(Holding, [robot, jug]),
        }
        delete_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(JugAtBurner, [jug, burner]),
        }
        if CFG.boil_use_constant_delay:
            delay_distribution = ConstantDelay(4)
        elif CFG.boil_use_normal_delay:
            delay_distribution = GaussianDelay(mean=4, std=0.2)
        else:
            delay_distribution = CMPDelay(80, 3)
        pick_jug_from_burner_process = EndogenousProcess(
            "PickJugFromBurner", parameters, condition_at_start, set(), set(),
            add_effects, delete_effects, delay_distribution, 1.0, option,
            option_vars, null_sampler)
        processes.add(pick_jug_from_burner_process)

        # PickJugFromOutsideFaucetAndBurner
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        parameters = [robot, jug]
        option_vars = [robot, jug]
        option = PickJug
        condition_at_start = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(JugNotAtBurnerOrFaucet, [jug]),
        }
        add_effects = {
            LiftedAtom(Holding, [robot, jug]),
        }
        delete_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(JugNotAtBurnerOrFaucet, [jug]),
        }
        if CFG.boil_use_constant_delay:
            delay_distribution = ConstantDelay(3)
        elif CFG.boil_use_normal_delay:
            delay_distribution = GaussianDelay(mean=3, std=0.2, rng=rng)
        else:
            delay_distribution = CMPDelay(55, 3)
        pick_jug_outside_faucet_burner_process = EndogenousProcess(
            "PickJugFromOutsideFaucetAndBurner", parameters,
            condition_at_start, set(), set(), add_effects, delete_effects,
            delay_distribution, 1.0, option, option_vars, null_sampler)
        processes.add(pick_jug_outside_faucet_burner_process)

        # PlaceOnBurner
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        burner = Variable("?burner", burner_type)
        faucet = Variable("?faucet", faucet_type)
        parameters = [robot, jug, burner, faucet]
        option_vars = [robot, burner]
        option = PlaceOnBurner
        condition_at_start = {
            LiftedAtom(Holding, [robot, jug]),
        }
        add_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(JugAtBurner, [jug, burner]),
        }
        delete_effects = {
            LiftedAtom(Holding, [robot, jug]),
        }
        if CFG.boil_use_constant_delay:
            delay_distribution = ConstantDelay(5)
        elif CFG.boil_use_normal_delay:
            delay_distribution = GaussianDelay(mean=5, std=0.2)
        else:
            delay_distribution = CMPDelay(100, 3)
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
            LiftedAtom(JugAtFaucet, [jug, faucet]),
        }
        delete_effects = {
            LiftedAtom(Holding, [robot, jug]),
            LiftedAtom(NoJugAtFaucet, [faucet]),
        }
        if CFG.boil_use_constant_delay:
            delay_distribution = ConstantDelay(3)
        elif CFG.boil_use_normal_delay:
            delay_distribution = GaussianDelay(mean=3, std=0.2)
        else:
            delay_distribution = CMPDelay(55, 3)
        place_under_faucet_process = EndogenousProcess(
            "PlaceUnderFaucet", parameters, condition_at_start, set(), set(),
            add_effects, delete_effects, delay_distribution, 1.0, option,
            option_vars, null_sampler)
        processes.add(place_under_faucet_process)

        # PlaceAtOutsideFaucetAndBurner
        robot = Variable("?robot", robot_type)
        jug = Variable("?jug", jug_type)
        parameters = [robot, jug]
        option_vars = [robot]
        option = PlaceOutsideBurnerAndFaucet
        condition_at_start = {
            LiftedAtom(Holding, [robot, jug]),
        }
        add_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(JugNotAtBurnerOrFaucet, [jug]),
        }
        delete_effects = {
            LiftedAtom(Holding, [robot, jug]),
        }
        if CFG.boil_use_constant_delay:
            delay_distribution = ConstantDelay(3)
        elif CFG.boil_use_normal_delay:
            delay_distribution = GaussianDelay(mean=3, std=0.2)
        else:
            delay_distribution = CMPDelay(55, 3)
        place_at_outside_faucet_burner_process = EndogenousProcess(
            "PlaceOutsideFaucetAndBurner", parameters, condition_at_start,
            set(), set(), add_effects, delete_effects, delay_distribution, 1.0,
            option, option_vars, null_sampler)
        processes.add(place_at_outside_faucet_burner_process)

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
        elif CFG.boil_use_normal_delay:
            delay_distribution = GaussianDelay(mean=1, std=0.2)
        else:
            delay_distribution = CMPDelay(1, 1)
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
        elif CFG.boil_use_normal_delay:
            delay_distribution = GaussianDelay(mean=1, std=0.2)
        else:
            delay_distribution = CMPDelay(1, 1)
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
        elif CFG.boil_use_normal_delay:
            delay_distribution = GaussianDelay(mean=3, std=0.2, rng=rng)
        else:
            delay_distribution = CMPDelay(55, 3)
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
        elif CFG.boil_use_normal_delay:
            delay_distribution = GaussianDelay(mean=1, std=0.2, rng=rng)
        else:
            delay_distribution = CMPDelay(1, 1)
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

        if CFG.boil_goal == "task_completed":
            # DeclareComplete
            robot = Variable("?robot", robot_type)
            parameters = [robot, jug, burner]
            option_vars = [robot]
            option = DeclareComplete
            condition_at_start = {
                LiftedAtom(NoWaterSpilled, []),
                LiftedAtom(WaterBoiled, [jug]),
                LiftedAtom(JugFilled, [jug]),
                LiftedAtom(BurnerOff, [burner]),
            }
            add_effects = {LiftedAtom(TaskCompleted, [])}
            delay_distribution = ConstantDelay(1)
            declare_complete_process = EndogenousProcess(
                "DeclareComplete", parameters, condition_at_start, set(),
                set(), add_effects, set(), delay_distribution, 1.0, option,
                option_vars, null_sampler)
            processes.add(declare_complete_process)

        # --- Exogenous Processes ---
        # FillJug
        jug = Variable("?jug", jug_type)
        faucet = Variable("?faucet", faucet_type)
        parameters = [jug, faucet]
        option_vars = [jug]
        condition_at_start = {
            LiftedAtom(JugAtFaucet, [jug, faucet]),
            LiftedAtom(FaucetOn, [faucet]),
            LiftedAtom(JugNotFilled, [jug]),
        }
        condition_overall = {
            LiftedAtom(JugAtFaucet, [jug, faucet]),
            LiftedAtom(FaucetOn, [faucet]),
        }
        add_effects = {
            LiftedAtom(JugFilled, [jug]),
        }
        delete_effects = {
            LiftedAtom(JugNotFilled, [jug]),
        }
        if CFG.boil_use_constant_delay:
            delay_distribution = ConstantDelay(7)
            delay_distribution = ConstantDelay(
                4)  # temporary for param learning
        elif CFG.boil_use_normal_delay:
            delay_distribution = GaussianDelay(mean=4, std=0.2, rng=rng)
        else:
            delay_distribution = CMPDelay(100, 2.9)
        fill_jug_process = ExogenousProcess("FillJug", parameters,
                                            condition_at_start,
                                            condition_overall, set(),
                                            add_effects, delete_effects,
                                            delay_distribution, 1.0)
        processes.add(fill_jug_process)

        # OverfillJug
        jug = Variable("?jug", jug_type)
        faucet = Variable("?faucet", faucet_type)
        parameters = [jug, faucet]
        condition_at_start = {
            LiftedAtom(FaucetOn, [faucet]),
        }
        if CFG.boil_use_derived_predicates:
            condition_at_start.add(
                LiftedAtom(NoJugAtFaucetOrJugAtFaucetAndFilled, [jug, faucet]))
        else:
            condition_at_start.add(LiftedAtom(JugAtFaucet, [jug, faucet]))
            condition_at_start.add(LiftedAtom(JugFilled, [jug]))
        condition_overall = condition_at_start.copy()
        # add_effects = {
        #     LiftedAtom(WaterSpilled, []),
        # }
        add_effects = set()
        delete_effects = {
            LiftedAtom(NoWaterSpilled, []),
        }
        if CFG.boil_use_constant_delay:
            delay_distribution = ConstantDelay(3)
        elif CFG.boil_use_normal_delay:
            delay_distribution = GaussianDelay(mean=3, std=0.2, rng=rng)
        else:
            delay_distribution = CMPDelay(55, 3)
        overfill_jug_process = ExogenousProcess("OverfillJug", parameters,
                                                condition_at_start,
                                                condition_overall, set(),
                                                add_effects, delete_effects,
                                                delay_distribution, 1.0)
        processes.add(overfill_jug_process)

        # Spill
        if not CFG.boil_use_derived_predicates:
            faucet = Variable("?faucet", faucet_type)
            parameters = [faucet]
            condition_at_start = {
                LiftedAtom(NoJugAtFaucet, [faucet]),
                LiftedAtom(FaucetOn, [faucet]),
            }
            # add_effects = {
            #     LiftedAtom(WaterSpilled, []),
            # }
            add_effects = set()
            delete_effects = {
                LiftedAtom(NoWaterSpilled, []),
            }
            if CFG.boil_use_constant_delay:
                delay_distribution = ConstantDelay(3)
            elif CFG.boil_use_normal_delay:
                delay_distribution = GaussianDelay(mean=3, std=0.2, rng=rng)
            else:
                delay_distribution = CMPDelay(55, 3)
            spill_process = ExogenousProcess("Spill",
                                             parameters, condition_at_start,
                                             set(), set(), add_effects,
                                             delete_effects,
                                             delay_distribution, 1.0)
            processes.add(spill_process)

        # Boil
        burner = Variable("?burner", burner_type)
        jug = Variable("?jug", jug_type)
        parameters = [burner, jug]
        condition_at_start = {
            LiftedAtom(JugAtBurner, [jug, burner]),
            LiftedAtom(JugFilled, [jug]),
            LiftedAtom(BurnerOn, [burner]),
        }
        condition_overall = {
            LiftedAtom(JugAtBurner, [jug, burner]),
            LiftedAtom(JugFilled, [jug]),
            LiftedAtom(BurnerOn, [burner]),
        }
        add_effects = {
            LiftedAtom(WaterBoiled, [jug]),
        }
        if CFG.boil_use_constant_delay:
            delay_distribution = ConstantDelay(5)
        elif CFG.boil_use_normal_delay:
            delay_distribution = GaussianDelay(mean=5, std=0.2, rng=rng)
        else:
            delay_distribution = CMPDelay(100, 3)
        boil_process = ExogenousProcess("Boil", parameters, condition_at_start,
                                        condition_overall, set(), add_effects,
                                        set(), delay_distribution, 1.0)
        processes.add(boil_process)

        if CFG.boil_goal == "human_happy":
            # HumanHappyProcess
            jug = Variable("?jug", jug_type)
            burner = Variable("?burner", burner_type)
            condition_at_start = {
                LiftedAtom(NoWaterSpilled, []),
                LiftedAtom(WaterBoiled, [jug]),
                LiftedAtom(JugFilled, [jug]),
                LiftedAtom(BurnerOff, [burner]),
            }
            add_effects = {LiftedAtom(HumanHappy, [])}
            delay_distribution = ConstantDelay(3)
            human_happy_process = ExogenousProcess("HumanHappy", parameters,
                                                   condition_at_start, set(),
                                                   set(), add_effects, set(),
                                                   delay_distribution, 1.0)
            processes.add(human_happy_process)

        return processes
