import atexit
import logging
import multiprocessing as mp
import os
import threading
from concurrent.futures import ProcessPoolExecutor

import duckdb
import litellm
import mlflow
from diskcache import FanoutCache

GLOBAL_FB_CACHE_DIR = "./.cache/feature-builder/"

cache = FanoutCache(directory="./.cache/dc", tag_index=True)
_db_ro_con = None
_glock = threading.Lock()
_tlocal = threading.local()


def shutdown_pool():
    global global_pool
    if _global_pool is not None:
        print("Shutting down pool", flush=True)
        _global_pool.shutdown(wait=True)


def init():
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("agent").setLevel(logging.DEBUG)

    litellm._logging._disable_debugging()

    os.environ["DSP_NOTEBOOK_CACHEDIR"] = os.path.join(
        os.getcwd(), "./.cache/dspnotebook"
    )
    os.environ["DSPY_CACHEDIR"] = os.path.join(os.getcwd(), "./.cache/dspy")
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("DSPy")
    mlflow.dspy.autolog()

    atexit.register(shutdown_pool)


_global_pool = None


def get_global_pool() -> ProcessPoolExecutor:
    global _global_pool
    if _global_pool is not None:
        return _global_pool
    with _glock:
        if _global_pool is None:
            print("Initializing global pool", flush=True)
            _global_pool = ProcessPoolExecutor(
                initializer=init, mp_context=mp.get_context("spawn")
            )
            atexit.register(shutdown_pool)
    return _global_pool


def get_duckdb_con(path="./.tmp/scratch.db") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(path)
    return con


def get_duckdb_ro_con(path="./.tmp/scratch.db") -> duckdb.DuckDBPyConnection:
    global _db_ro_con, _glock
    if _db_ro_con is not None:
        if "con" not in _tlocal.__dict__:
            print(f"[{threading.get_ident()}] Connecting to duckdb")
            _tlocal.con = _db_ro_con.cursor()
        return _tlocal.con
    with _glock:
        if _db_ro_con is None:
            print("Initializing duckdb")
            con = duckdb.connect(
                path, read_only=True, config={"access_mode": "READ_ONLY"}
            )
            # con.execute(
            #     """
            #     SET autoinstall_known_extensions=false;
            #     SET autoload_known_extensions=false;
            #     INSTALL fts;
            #     LOAD fts;
            #     """
            # )
            _db_ro_con = con
        print(f"[{threading.get_ident()}] Connecting to duckdb")
        _tlocal.con = _db_ro_con.cursor()
    return _tlocal.con
