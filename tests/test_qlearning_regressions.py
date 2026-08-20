# SPDX-License-Identifier: MIT
import ast
import inspect
import tempfile
from collections import defaultdict
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "src" / "yfs_xrd_refinement" / "qlearning.py"
SOURCE = SOURCE_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE, filename=str(SOURCE_PATH))
PARENT = {
    child: parent
    for parent in ast.walk(TREE)
    for child in ast.iter_child_nodes(parent)
}


def calls(root, owner, method):
    return [
        node for node in ast.walk(root)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == method
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == owner
    ]


def writes_name(node, name):
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    else:
        return False
    return any(
        isinstance(child, ast.Name)
        and isinstance(child.ctx, ast.Store)
        and child.id == name
        for target in targets
        for child in ast.walk(target)
    )


def is_action_branch(node, action):
    if not isinstance(node, ast.If):
        return False
    test = node.test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "action_selected"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == action
    )


OUTER = next(
    node for node in TREE.body
    if isinstance(node, ast.FunctionDef) and node.name == "outer_refine_all"
)
STAGE_LOOP = next(
    node for node in ast.walk(OUTER)
    if isinstance(node, ast.For)
    and any(isinstance(child, ast.Name) and child.id == "stg" for child in ast.walk(node.target))
    and isinstance(node.iter, ast.Call)
    and isinstance(node.iter.func, ast.Name)
    and node.iter.func.id == "enumerate"
)
REFINE_LOOP = next(
    node for node in ast.walk(STAGE_LOOP)
    if isinstance(node, ast.For)
    and isinstance(node.target, ast.Name)
    and node.target.id == "loop"
)


def load_definitions(*names):
    nodes = [
        node for node in TREE.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names
    ]
    namespace = {"np": np, "defaultdict": defaultdict}
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(SOURCE_PATH), "exec"), namespace)
    return [namespace[name] for name in names]


Agent, compose_refinement_loss = load_definitions(
    "QLearningRefineAgent", "compose_refinement_loss"
)


def make_agent(*, epsilon=0.0, seed=20260820):
    return Agent(
        actions=list(range(7)),
        learning_rate=0.5,
        reward_decay=0.9,
        e_greedy=epsilon,
        rng=np.random.default_rng(seed),
    )


class QLearningStructureTests(unittest.TestCase):
    def test_each_iteration_builds_exactly_one_learning_transition(self):
        choose = calls(REFINE_LOOP, "agent", "choose_action")
        append = calls(REFINE_LOOP, "rwp_trend_history", "append")
        transition_writes = [
            node for node in ast.walk(REFINE_LOOP) if writes_name(node, "transition")
        ]
        update_calls = [
            node for node in ast.walk(REFINE_LOOP)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "learn_transition"
        ]
        helper = next(
            node for node in ast.walk(OUTER)
            if isinstance(node, ast.FunctionDef) and node.name == "learn_transition"
        )
        self.assertEqual(len(choose), 1)
        self.assertEqual(len(append), 1)
        self.assertEqual(len(transition_writes), 1)
        self.assertEqual(len(update_calls), 1)
        self.assertEqual(len(calls(helper, "agent", "learn")), 1)
        self.assertLess(choose[0].lineno, append[0].lineno)
        self.assertLess(append[0].lineno, transition_writes[0].lineno)

    def test_agent_and_history_are_not_reset_between_stages(self):
        self.assertEqual(
            [node for node in ast.walk(STAGE_LOOP) if writes_name(node, "agent")], []
        )
        self.assertEqual(
            [node for node in ast.walk(STAGE_LOOP) if writes_name(node, "rwp_trend_history")], []
        )
        history_initializations = [
            node for node in ast.walk(OUTER) if writes_name(node, "rwp_trend_history")
        ]
        self.assertEqual(len(history_initializations), 1)
        self.assertLess(history_initializations[0].lineno, STAGE_LOOP.lineno)
        for call in calls(STAGE_LOOP, "rwp_trend_history", "append"):
            self.assertEqual(ast.unparse(call.args[0]), "r_best")

    def test_action4_reuses_initial_frozen_groups_without_regrouping(self):
        action4 = next(node for node in ast.walk(REFINE_LOOP) if is_action_branch(node, 4))
        action4_body = ast.Module(body=action4.body, type_ignores=[])
        assignments = [
            node for node in ast.walk(action4_body)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "groups" for target in node.targets)
        ]
        self.assertEqual(len(assignments), 1)
        self.assertEqual(ast.unparse(assignments[0].value), "fixed_groups_map[key]")
        forbidden = [
            node for node in ast.walk(action4_body)
            if isinstance(node, ast.Call)
            and (
                isinstance(node.func, ast.Name) and node.func.id == "SpacegroupAnalyzer"
                or isinstance(node.func, ast.Attribute)
                and node.func.attr == "get_symmetrized_structure"
            )
        ]
        self.assertEqual(forbidden, [])
        regroup_inside_stages = [
            node for node in ast.walk(STAGE_LOOP)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "SpacegroupAnalyzer"
        ]
        self.assertEqual(regroup_inside_stages, [])

    def test_stage_boundaries_defer_bootstrap_and_only_final_stage_is_terminal(self):
        final_assign = next(
            node for node in ast.walk(REFINE_LOOP)
            if writes_name(node, "is_final_transition")
        )
        expression = ast.unparse(final_assign.value)
        self.assertIn("stg_index == len(stage_settings) - 1", expression)
        self.assertIn("is_stage_boundary", expression)
        loop_source = ast.get_source_segment(SOURCE, REFINE_LOOP)
        self.assertIn("pending_transition = transition", loop_source)
        self.assertIn('next_state = "terminal" if is_final_transition', loop_source)
        stage_source = ast.get_source_segment(SOURCE, STAGE_LOOP)
        self.assertIn("if pending_transition is not None:", stage_source)
        self.assertIn("get_valid_actions()", stage_source)

    def test_uiso_failure_decays_and_can_leave_the_valid_action_mask(self):
        loop_source = ast.get_source_segment(SOURCE, REFINE_LOOP)
        self.assertIn("elif action_selected == 5:", loop_source)
        self.assertIn("uiso_step *= 0.8", loop_source)
        self.assertIn("stage_exhausted = not remaining_actions", loop_source)

    def test_stepb_restores_pure_fit_baseline_before_rl_actions(self):
        source_segment = ast.get_source_segment(SOURCE, STAGE_LOOP)
        self.assertIn("fit_baseline = (best_score, yfit, fr, sf, r_best)", source_segment)
        self.assertIn("best_score, yfit, fr, sf, r_best = fit_baseline", source_segment)
        restore_line = next(
            line for line in source_segment.splitlines()
            if "best_score, yfit, fr, sf, r_best = fit_baseline" in line
        )
        self.assertTrue(restore_line.strip().endswith("fit_baseline"))

    def test_attempt_log_keeps_rwp_and_score_separate(self):
        push = next(
            node for node in ast.walk(OUTER)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "push_log"
        )
        self.assertEqual(ast.unparse(push.args[1]), "Rwp")
        self.assertEqual(ast.unparse(push.keywords[0].value), "score")


class QLearningAgentTests(unittest.TestCase):
    def test_exploitation_masks_invalid_high_q_and_randomizes_ties(self):
        agent = make_agent(epsilon=0.0)
        agent.q_table["state"][:] = [-10.0] * 7
        agent.q_table["state"][0] = 100.0
        agent.q_table["state"][2] = 5.0
        agent.q_table["state"][4] = 5.0
        selected = {
            agent.choose_action("state", valid_actions=[2, 4])
            for _ in range(128)
        }
        self.assertEqual(selected, {2, 4})

    def test_exploration_respects_mask(self):
        agent = make_agent(epsilon=1.0)
        selected = {
            agent.choose_action("state", valid_actions=[1, 5])
            for _ in range(128)
        }
        self.assertEqual(selected, {1, 5})

    def test_empty_mask_is_rejected(self):
        with self.assertRaises(ValueError):
            make_agent().choose_action("state", valid_actions=[])

    def test_terminal_update(self):
        agent = make_agent()
        agent.q_table["s"][2] = 1.0
        agent.learn("s", 2, 5.0, "terminal")
        self.assertAlmostEqual(agent.q_table["s"][2], 3.0)

    def test_nonterminal_update_masks_next_state(self):
        agent = make_agent()
        agent.q_table["s"][1] = 1.0
        agent.q_table["next"][:] = [100.0, 4.0, 3.0, 2.0, 0.0, 0.0, 0.0]
        agent.learn("s", 1, 2.0, "next", next_valid_actions=[1, 2])
        self.assertAlmostEqual(agent.q_table["s"][1], 3.3)

    def test_loss_composition_is_shared_and_mode_aware(self):
        self.assertEqual(compose_refinement_loss(10.0, 2.0, "fit", 0.5), 10.0)
        self.assertEqual(compose_refinement_loss(10.0, 2.0, "stoich", 0.5), 1.1)
        self.assertEqual(compose_refinement_loss(10.0, 2.0, "combined", 0.5), 11.0)


class BatchCompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        batch_path = ROOT / "src" / "yfs_xrd_refinement" / "batch.py"
        cls.batch_tree = ast.parse(batch_path.read_text(encoding="utf-8"))
        node = next(
            item for item in cls.batch_tree.body
            if isinstance(item, ast.FunctionDef) and item.name == "has_successful_output"
        )
        namespace = {"Path": Path, "re": __import__("re")}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(batch_path), "exec"), namespace)
        cls.has_successful_output = staticmethod(namespace["has_successful_output"])

    def test_failed_or_partial_run_is_not_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "refine_log.txt").write_text("ReturnCode: 1\n", encoding="utf-8")
            (folder / "yfsf_Refined.txt").write_text("partial", encoding="utf-8")
            self.assertFalse(self.has_successful_output(str(folder)))

    def test_last_return_code_controls_resume_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "refine_log.txt").write_text(
                "ReturnCode: 0\nReturnCode: 1\n", encoding="utf-8"
            )
            (folder / "yfsf_Refined.txt").write_text("old", encoding="utf-8")
            (folder / "yfsf_Refined.xy").write_text("old", encoding="utf-8")
            self.assertFalse(self.has_successful_output(str(folder)))

    def test_only_success_with_final_products_is_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            (folder / "refine_log.txt").write_text("ReturnCode: 0\n", encoding="utf-8")
            (folder / "yfsf_Refined.txt").write_text("done", encoding="utf-8")
            (folder / "yfsf_Refined.xy").write_text("done", encoding="utf-8")
            self.assertTrue(self.has_successful_output(str(folder)))


if __name__ == "__main__":
    unittest.main()
