import logging
import math
import statistics
import textwrap
from typing import Callable, NamedTuple, Optional

import numpy as np
import pydot
from IPython.display import SVG, display

from .agent import OutputV2, Task

logger = logging.getLogger("agent")


class MCTTreeNode(NamedTuple):
    id: str
    input: Optional[OutputV2]
    output_node: Optional[OutputV2]
    current_depth: int
    max_depth: int
    visit_count: int
    total_reward: float
    error: Optional[Exception]
    previous_node_id: Optional[str]

    def explore(self, output: OutputV2) -> "MCTTreeNode":
        return self._replace(output_node=output)

    def failure(self, error: Exception) -> "MCTTreeNode":
        return self._replace(error=error)

    def is_explored(self) -> bool:
        return self.error is not None or self.output_node is not None

    def potential_children_inputs(self) -> dict[str, OutputV2]:
        assert self.output_node is not None

        num_suggestions = len(self.output_node.get_best_eval_output()[0].suggestions)

        if self.is_terminal():
            return dict()
        else:
            return {
                f"{self.id}-{i}": self.output_node._replace(suggestion_index=i)
                for i in range(num_suggestions)
            }

    def is_terminal(self) -> bool:
        return self.error is not None or self.current_depth == self.max_depth

    def reward(self) -> float:
        if self.error is not None:
            return 0

        assert self.output_node is not None

        return self.output_node.get_best_eval_output()[0].model_eval_result.roc_auc

    def node_test_performance(self) -> tuple[float, float, float]:
        if self.error is not None:
            return 0, 0, 0

        assert self.output_node is not None

        result = self.output_node.get_best_eval_output()[1]
        return result.roc_auc, result.f1, result.pr_auc


class MCTS:
    def __init__(
        self,
        task: Task,
        runner: Callable[[str, Task, Optional[OutputV2]], OutputV2],
        nodes_dict: dict[str, MCTTreeNode] = dict(),
        exploration_weight=1,
        max_depth=7,
    ):
        self.task = task
        self.exploration_weight = exploration_weight
        self.rng = np.random.default_rng(42)
        self.max_depth = max_depth
        self.nodes = nodes_dict
        if len(nodes_dict) == 0:
            self.nodes["0"] = self._get_or_create_node("0", None, None)

        self.runner = runner
        self.num_rollouts = 0
        self.test_results = []

    def nodes_sorted_by_val_perf(self):
        all_explored_nodes = filter(
            lambda k: k[1].output_node is not None, list(self.nodes.items())
        )
        sorted_explored_nodes = list(
            sorted(all_explored_nodes, key=lambda x: x[1].reward(), reverse=True)
        )
        return sorted_explored_nodes

    def nodes_sorted_by_test_perf(self):
        all_explored_nodes = filter(
            lambda k: k[1].output_node is not None, list(self.nodes.items())
        )
        sorted_explored_nodes = list(
            sorted(
                all_explored_nodes,
                key=lambda x: x[1].node_test_performance()[0],
                reverse=True,
            )
        )
        return sorted_explored_nodes

    def do_rollout(self):
        logger.info("Running Rollout")
        path = self._select(self.nodes["0"])
        logger.info("Current path %s", list(map(lambda x: x.id, path)))
        leaf = path[-1]
        leaf = self._expand(leaf)
        logger.info("Done expanding")
        path[-1] = leaf
        reward = self._simulate(leaf)
        self._backpropagate(path, reward)
        self.num_rollouts = self.num_rollouts + 1
        logger.info("Done rollout - Completed %s", self.num_rollouts)

        sorted_explored_nodes = self.nodes_sorted_by_val_perf()
        sorted_explored_nodes_test = self.nodes_sorted_by_test_perf()

        top1_id_test, top1_node_test = sorted_explored_nodes_test[0]
        top1_id, top1_node = sorted_explored_nodes[0]
        top1_test_roc_auc, top1_test_f1, top1_test_pr_auc = (
            top1_node_test.node_test_performance()
        )

        top1_roc_auc, top1_f1, top1_pr_auc = top1_node.node_test_performance()

        top3 = sorted_explored_nodes[0:3]
        top3_ids = ",".join(map(lambda x: x[0], top3))
        top3_scores = list(map(lambda x: x[1].node_test_performance(), top3))
        top3_roc_auc = statistics.mean(list(map(lambda x: x[0], top3_scores)))
        top3_f1 = statistics.mean(map(lambda x: x[1], top3_scores))
        top3_pr_auc = statistics.mean(map(lambda x: x[2], top3_scores))

        top3_test = sorted_explored_nodes_test[0:3]
        top3_ids_test = ",".join(map(lambda x: x[0], top3_test))
        top3_scores_test = list(map(lambda x: x[1].node_test_performance(), top3_test))
        top3_roc_auc_test = statistics.mean(list(map(lambda x: x[0], top3_scores_test)))
        top3_f1_test = statistics.mean(map(lambda x: x[1], top3_scores_test))
        top3_pr_auc_test = statistics.mean(map(lambda x: x[2], top3_scores_test))

        top5_test = sorted_explored_nodes_test[0:5]
        top5_ids_test = ",".join(map(lambda x: x[0], top5_test))
        top5_scores_test = list(map(lambda x: x[1].node_test_performance(), top5_test))
        top5_roc_auc_test = statistics.mean(list(map(lambda x: x[0], top5_scores_test)))
        top5_f1_test = statistics.mean(map(lambda x: x[1], top5_scores_test))
        top5_pr_auc_test = statistics.mean(map(lambda x: x[2], top5_scores_test))

        result = {
            "top1_id": top1_id,
            "top1_roc_auc": top1_roc_auc,
            "top1_f1": top1_f1,
            "top1_pr_auc": top1_pr_auc,
            "top3_ids": top3_ids,
            "top3_roc_auc": top3_roc_auc,
            "top3_f1": top3_f1,
            "top3_pr_auc": top3_pr_auc,
            "top1_id_test": top1_id_test,
            "top1_test_roc_auc": top1_test_roc_auc,
            "top1_test_f1": top1_test_f1,
            "top1_test_pr_auc": top1_test_pr_auc,
            "top3_ids_test": top3_ids_test,
            "top3_roc_auc_test": top3_roc_auc_test,
            "top3_f1_test": top3_f1_test,
            "top3_pr_auc_test": top3_pr_auc_test,
            "top5_ids_test": top5_ids_test,
            "top5_roc_auc_test": top5_roc_auc_test,
            "top5_f1_test": top5_f1_test,
            "top5_pr_auc_test": top5_pr_auc_test,
        }
        logger.info("Test Result %s", result)
        self.test_results.append(result)

    def _append_graph_node(self, graph: pydot.Graph, node: Optional[MCTTreeNode]):
        if node is None:
            return

        input_suggestion = (
            "no_input" if not node.input else node.input.get_next_suggestion()
        )

        output_str = ""
        if node.output_node is not None:
            operation = node.output_node.operation
            eval_output, test_output = node.output_node.get_best_eval_output()

            if operation is None:
                output_str = f"""
            Initializer
            Features: {",".join(node.output_node.feature_plans.keys())}                

            ROC AUC: {eval_output.model_eval_result.roc_auc}
            TEST ROC AUC: {test_output.roc_auc}
            TEST F1: {test_output.f1}
            TEST PR AUC: {test_output.pr_auc}
                """
            else:
                output_str = f"""
                Proposed Operation: {operation.feature_operation}
                Proposed Feature: {operation.feature_name}
                Proposed Explantion: {textwrap.fill(operation.feature_explanation)}

                ROC AUC: {eval_output.model_eval_result.roc_auc}
                TEST ROC AUC: {test_output.roc_auc}
                TEST F1: {test_output.f1}
                TEST PR AUC: {test_output.pr_auc}
                """

        label = f"""
        {node.id} | TR: {node.total_reward} | VC: {node.visit_count}
        Explored: {node.is_explored()} | Terminal: {node.is_terminal()} | Error: {node.error}
        Input Suggestion: {textwrap.fill(input_suggestion)}
        {output_str}
        """

        graph.add_node(
            pydot.Node(node.id, label=label, shape="box", margin="0", fixedsize="false")
        )

        if node.output_node is not None:
            for child_id in node.potential_children_inputs().keys():
                graph.add_edge(pydot.Edge(node.id, child_id))
                self._append_graph_node(graph, self.nodes.get(child_id))

    def show_graph(self, path="./output.svg"):
        graph = pydot.Dot("my_graph", graph_type="graph")

        root = self.nodes["0"]
        self._append_graph_node(graph, root)
        svg_out = graph.create_svg()  # type: ignore
        with open(path, "wb") as f:
            f.write(svg_out)
        plt = SVG(svg_out)
        display(plt)

    def _is_fully_expanded(self, node: MCTTreeNode) -> bool:
        if not node.is_explored():
            return False

        for child_keys in node.potential_children_inputs().keys():
            child_node = self.nodes.get(child_keys)
            if child_node is None:
                return False

            if not child_node.is_explored():
                return False
        return True

    def _get_expanded_children(self, node: MCTTreeNode) -> list[MCTTreeNode]:
        assert node.is_explored()
        keys = node.potential_children_inputs().keys()
        return [self.nodes[k] for k in keys]

    def _expand(self, node: MCTTreeNode) -> MCTTreeNode:
        if not node.is_explored():
            node = self._explore_node(node)

        if node.is_terminal():
            return node

        children = node.potential_children_inputs()
        for id, child_input in children.items():
            self._get_or_create_node(id, node, child_input)

        return node

    def _get_or_create_node(
        self, id: str, parent: MCTTreeNode | None, input: OutputV2 | None
    ) -> MCTTreeNode:
        existing = self.nodes.get(id)
        if existing is not None:
            return existing

        current_depth = 0
        if parent is not None:
            current_depth = parent.current_depth + 1

        new_node = MCTTreeNode(
            id=id,
            input=input,
            output_node=None,
            current_depth=current_depth,
            max_depth=self.max_depth,
            visit_count=1,
            total_reward=0.0,
            error=None,
            previous_node_id=parent.id if parent is not None else None,
        )
        self.nodes[id] = new_node
        return new_node

    def _explore_node(self, node: MCTTreeNode) -> MCTTreeNode:
        logger.info("Exploring %s", node.id)
        assert not node.is_explored()
        try:
            output = self.runner(node.id, self.task, node.input)
            updated_node = node.explore(output)
        except Exception as e:
            logger.info("Failed to run %s", node.id, exc_info=e)
            updated_node = node.failure(e)

        self.nodes[node.id] = updated_node
        assert updated_node.is_explored()
        logger.info(
            "Finished exploring %s, score %s, test score %s",
            node.id,
            updated_node.reward(),
            updated_node.node_test_performance(),
        )
        return updated_node

    def _simulate(self, node: MCTTreeNode) -> float:
        while True:
            if node.is_terminal():
                current_max_reward = node.reward()
                max_node = node
                curr_node = node
                while curr_node.previous_node_id is not None:
                    curr_node = self.nodes[curr_node.previous_node_id]
                    if curr_node.reward() > current_max_reward:
                        current_max_reward = curr_node.reward()
                        max_node = curr_node

                logger.info(
                    "Simulate terminating at %s, tail node reward is %s, tail node test reward is %s,  max reward of path is %s, test reward of best node is %s",
                    node.id,
                    node.reward(),
                    node.node_test_performance(),
                    current_max_reward,
                    max_node.node_test_performance(),
                )
                return current_max_reward
            potential_children = node.potential_children_inputs()

            choice_id = self.rng.choice(list(potential_children.keys()))
            logger.info("Simulating %s", choice_id)

            next_node = self._get_or_create_node(
                choice_id, node, potential_children[choice_id]
            )
            node = self._explore_node(next_node)

    def _select(self, node: MCTTreeNode) -> list[MCTTreeNode]:
        logger.info("Running select")
        path = []
        while True:
            path.append(node)
            if node.is_terminal() or not node.is_explored():
                return path
            unexplored = []
            for child_id, child_input in node.potential_children_inputs().items():
                child_node = self._get_or_create_node(child_id, node, child_input)
                if not child_node.is_explored():
                    unexplored.append(child_node)

            if len(unexplored) > 0:
                n = unexplored.pop()
                path.append(n)
                return path
            node = self._uct_select(node)  # descend a layer deeper

    def _backpropagate(self, path: list[MCTTreeNode], reward: float):
        for node in reversed(path):
            current = self.nodes[node.id]
            self.nodes[node.id] = current._replace(
                visit_count=current.visit_count + 1,
                total_reward=current.total_reward + reward,
            )
            logger.info(
                "Backpropagating %s, new VC: %s, new TR: %s",
                node.id,
                current.visit_count + 1,
                current.total_reward + reward,
            )

    def _uct_select(self, node: MCTTreeNode):
        assert self._is_fully_expanded(node)

        log_N_vertex = math.log(node.visit_count)

        def uct(n: MCTTreeNode):
            "Upper confidence bound for trees"
            return n.total_reward / n.visit_count + self.exploration_weight * math.sqrt(
                log_N_vertex / n.visit_count
            )

        return max(self._get_expanded_children(node), key=uct)
