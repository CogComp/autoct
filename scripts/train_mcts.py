import json
import os
import sys

import dill

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import argparse
import logging

import mlflow
import tqdm

from src.lfe.impl.agent import Task, run_agent_as_script_v2
from src.lfe.impl.treesearch import MCTS

logging.basicConfig(level=logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("agent").setLevel(logging.DEBUG)

os.environ["DSP_NOTEBOOK_CACHEDIR"] = os.path.join(os.getcwd(), "./.cache/dspnotebook")
os.environ["DSPY_CACHEDIR"] = os.path.join(os.getcwd(), "./.cache/dspy")
os.environ["LITELLM_LOG"] = "ERROR"

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("DSPy")
mlflow.dspy.autolog()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    parser.add_argument("--rollouts")
    parser.add_argument("--depth")
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    task = Task[args.task]
    depth = int(args.depth)
    rollouts = int(args.rollouts)

    if args.resume is not None:
        with open(args.resume, "rb") as f:
            mcts: MCTS = dill.load(f)
    else:
        mcts = MCTS(task, run_agent_as_script_v2, max_depth=depth)

    for i in tqdm.tqdm(range(mcts.num_rollouts, rollouts)):
        mcts.do_rollout()

    fname = f"{task.name}-r{rollouts}-d{depth}"

    mcts.show_graph(path=f"./.output/{fname}.svg")
    with open(f".output/{fname}.pkl", "wb") as f:
        dill.dump(mcts, f)
    with open(f".output/{fname}_results.json", "w") as f:
        json.dump(mcts.test_results, f)
