import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import argparse

import dill
from rich.pretty import pprint

from src.lfe.impl.agent import Task, WrappedFeatureBuilderV3, features_to_df_v2
from src.lfe.impl.treesearch import MCTS, MCTTreeNode


def predict_from_treenode(node: MCTTreeNode, task: Task, nctid: str) -> float:
    fb = WrappedFeatureBuilderV3(task=task)
    assert node.output_node is not None
    pprint("Building Features")
    pprint(node.output_node.feature_plans)
    _, values, meta = fb([nctid, node.output_node.feature_plans])
    pprint("Got Feature Values")
    pprint(values)
    pprint("Got Meta")
    pprint(meta)
    as_dict = {nctid: values}
    feat_df = features_to_df_v2(as_dict)
    pipeline = node.output_node.xgb_eval_output.model_eval_result.pipeline
    pred = pipeline.predict_proba(feat_df)
    return pred[0][1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pickled_output")
    parser.add_argument("--nctid")
    args = parser.parse_args()

    result: MCTS = dill.load(
        open(args.pickled_output, "rb")
    )

    best_node = result.nodes_sorted_by_val_perf()[0][1]
    pprint(f"Using Best Node: {best_node.id}")
    prediction = predict_from_treenode(best_node, result.task, args.nctid)
    pprint(f"Prediction for {args.nctid}: {prediction}")
