"""Input/output helpers: loading objects and converting arrays to image formats."""
import pickle
import os
import sys
import numpy as np

import torch
from torch import Tensor


def load_pkl(filename):
    """Reads in a geometry pickled object.

    Args:
        filename (str): path/to/file.pkl

    Returns:
        geometry: geometry object
    """
    
    if os.path.isfile(filename):
        print('Loading geometry object...')
        with open(filename, 'rb') as inp:
            geom = pickle.load(inp)
        
        return geom
    else:
        print('File does not exist.')
        

def normalization(field, totype='uint16'):
    """Scale a real 2D array linearly to the full range of an unsigned integer type.

    Parameters
    ----------
    field : array_like
        Real values (any units).
    totype : {'uint16', 'uint8'}
        Output type: the minimum maps to 0, the maximum to ``2**bits - 1``.

    Returns
    -------
    numpy.ndarray
        ``field`` rescaled and rounded to ``totype``. A constant field gives zeros.

    Raises
    ------
    ValueError
        For any other ``totype`` (the old names ``'int16'`` and ``'int8'`` were misleading:
        they returned unsigned types).
    """
    dtypes = {'uint16': np.uint16, 'uint8': np.uint8}
    if totype not in dtypes:
        raise ValueError(f"totype must be 'uint16' or 'uint8', got {totype!r}")
    field = np.asarray(field, dtype=np.float64)
    span = field.max() - field.min()
    unit = (field - field.min()) / span if span > 0 else np.zeros_like(field)
    top = np.iinfo(dtypes[totype]).max                       # 2**bits - 1
    return np.rint(unit * top).astype(dtypes[totype])


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
