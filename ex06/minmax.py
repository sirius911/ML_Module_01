import numpy as np


def minmax(x):
    """Computes the normalized version of a non-empty numpy.ndarray using the min-max standardization.
    Args:
        x: has to be an numpy.ndarray, a vector.
    Returns:
        x’ as a numpy.ndarray.
        None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
    Raises:
        This function shouldn’t raise any Exception.
    """
    if not isinstance(x, np.ndarray):
        return None
    if len(x.shape) == 0 or len(x.shape) > 2:
        return None
    if len(x.shape) == 2 and x.shape[1] != 1:
        return None
    ret = np.array(x - np.min(x)) / (np.max(x) - np.min(x))
    return (ret)