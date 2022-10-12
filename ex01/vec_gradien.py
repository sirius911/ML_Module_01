import numpy as np
from prediction import predict_


def simple_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.array, with a for-loop.
    The three arrays must have compatible shapes.
    Args:
        x: has to be an numpy.array, a vector of shape m * 1.
        y: has to be an numpy.array, a vector of shape m * 1.
        theta: has to be an numpy.array, a 2 * 1 vector.
    Return:
        The gradient as a numpy.array, a vector of shape 2 * 1.
        None if x, y, or theta are empty numpy.array.
        None if x, y and theta do not have compatible shapes.
        None if x, y or theta is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x,np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if len(x) == 0 or len(y) == 0 or len(theta) == 0:
        return None
    if (len(x.shape) > 1 and x.shape[1] != 1) or (len(y.shape) > 1 and y.shape[1] != 1):
        return None
    if theta.shape != (2, 1):
        return None
    if x.shape != y.shape:
        return None

    m = len(x)
    h = predict_(x, theta=theta)
    diff = h - y
    return np.array([[diff.sum() / m], [(diff * x).sum() / m]])