"""Serialize account identifiers without replacing them with row positions."""
import numpy as np


def account_id_array(values):
    """Return unique integer/Unicode IDs suitable for pickle-free NPY storage."""
    ids = np.asarray(values)
    if ids.ndim != 1 or not len(ids):
        raise ValueError("Account IDs must be a nonempty one-dimensional sequence.")
    if ids.dtype.kind == "O":
        if all(isinstance(value, str) for value in ids):
            ids = ids.astype(str)
        else:
            raise ValueError("Object account IDs must contain only strings.")
    if ids.dtype.kind not in "iuU":
        raise ValueError("Account IDs must be integers or Unicode strings.")
    if any(not str(value).strip() for value in ids):
        raise ValueError("Account IDs must not be blank.")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Account IDs must be unique.")
    return ids
