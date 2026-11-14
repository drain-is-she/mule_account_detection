"""Serialize account identifiers without replacing them with row positions."""
import numpy as np


def account_id_array(values):
    """Return unique integer/Unicode IDs suitable for pickle-free NPY storage."""
    # Inspect original scalar types before NumPy can coerce mixed lists.
    raw = np.asarray(values, dtype=object)
    if raw.ndim != 1 or not len(raw):
        raise ValueError("Account IDs must be a nonempty one-dimensional sequence.")
    if all(isinstance(value, str) for value in raw):
        ids = raw.astype(str)
    elif all(isinstance(value, (int, np.integer)) and
             not isinstance(value, (bool, np.bool_)) for value in raw):
        integers = [int(value) for value in raw]
        low, high = min(integers), max(integers)
        if -(2**63) <= low and high < 2**63:
            ids = np.array(integers, dtype=np.int64)
        elif 0 <= low and high < 2**64:
            ids = np.array(integers, dtype=np.uint64)
        else:
            raise ValueError("Account IDs do not fit a single 64-bit integer array.")
    else:
        raise ValueError("Account IDs must be all integers or all Unicode strings, without booleans.")
    if any(not str(value).strip() for value in ids):
        raise ValueError("Account IDs must not be blank.")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Account IDs must be unique.")
    return ids
