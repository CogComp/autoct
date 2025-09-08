from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def not_na(value):
    if value is None:
        return False

    if isinstance(value, float):
        return not np.isnan(value)

    return True


def extend_file_name(original: str, suffix: str) -> str:
    path = Path(original)
    return str(path.with_stem(path.stem + suffix))


def iter_by_line_parquet(path: str, batch_size: int) -> Iterator[dict[str, Any]]:
    """Iterate over a parquet file line by line.

    Each line is represented by a dict.

    Args:
        path: path to the .parquet file.
        batch_size: number of rows to load in memory.

    Yields:
        line as dict.
    """
    parquet_file = pq.ParquetFile(path)

    for batch in parquet_file.iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()
