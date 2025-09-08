import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
os.environ["DSP_NOTEBOOK_CACHEDIR"] = os.path.join(os.getcwd(), "./.cache/dspnotebook")
os.environ["DSPY_CACHEDIR"] = os.path.join(os.getcwd(), "./.cache/dspy")
os.environ["LITELLM_LOG"] = "ERROR"

import argparse
import logging
import threading

import dill
import mlflow

from src.lfe.impl.agent import AgentV2, Task
from src.lfe.impl.globals import shutdown_pool

logging.basicConfig(level=logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("agent").setLevel(logging.DEBUG)


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("DSPy")
mlflow.dspy.autolog()


def check_pid(pid):
    def loop():
        while True:
            try:
                os.kill(pid, 0)
            except OSError:
                print(f"Parent {pid} is dead, exiting", flush=True)
                shutdown_pool()
                os._exit(0)
            time.sleep(1)

    thread = threading.Thread(target=loop)
    thread.daemon = True
    thread.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--task")
    parser.add_argument("--pid")
    parser.add_argument("output")
    args = parser.parse_args()

    print(
        f"Starting Agent as Script Parent: {args.pid} Self {os.getpid()} Input {args.input}",
        flush=True,
    )
    check_pid(int(args.pid))
    task = Task[args.task]

    train, val, test = task.get_train_val_test()

    X_train = train["nctid"]
    y_train = train["label"]

    X_val = val["nctid"]
    y_val = val["label"]

    X_test = test["nctid"]
    y_test = test["label"]

    if args.input is not None:
        with open(args.input, "rb") as f:
            input = dill.load(f)
    else:
        input = None

    agent = AgentV2(
        task=task,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
    )
    output = agent(input)

    with open(args.output, "wb") as f:
        dill.dump(output, f)
