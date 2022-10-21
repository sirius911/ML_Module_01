from math import floor
import numpy as np
from re import I
import time

def ft_progress(lst):
    print("\x1b[?25l") # hide cursor
    i = 1
    start = time.time()
    while i <= len(lst):
        yield i
        pourc = int(i * 100 / len (lst))
        nb = int(i * 20 / len(lst))
        arrow = ">".rjust(nb, "=")
        top = time.time() - start
        eta = (len(lst) * top / i) - top
        if eta <= 100:
            etah = 0
            etam = 0
            etas = eta
        elif eta > 100 and eta < 3600:
            etah = 0
            etam = floor(eta / 60)
            etas = eta - (etam * 60)
        else:
            etah = floor(eta / (60 * 60))
            etam = floor((eta - (etah * 60 * 60)) / 60)
            etas = eta - (etah * 60 *60) - (etam * 60)
        if top <= 100:
            toph = 0
            topm = 0
            tops = top
        elif top > 100 and top < 3600:
            toph = 0
            topm = floor(top / 60)
            tops = top - (topm * 60)
        else:
            toph = floor(top / (60 * 60))
            topm = floor((top - (toph * 60 * 60)) / 60)
            tops = top - (toph * 60 * 60) - (topm * 60)
        label = f"ETA:"
        if etah > 0:
            label = f"{label} {etah}h"
        if etam > 0 or etah > 0:
            label = f"{label} {etam:02}mn"
        label = f"{label} {etas:05.2f}s [{pourc:3}%] [{arrow:<20}] {i}/{len(lst)} | elapsed time"
        if toph > 0:
            label = f"{label} {toph}h"
        if topm > 0 or toph > 0:
            label = f"{label} {topm}mn"
        label = f"{label} {tops:05.2f}s    "
        print(f"{label}", end='\r', flush=True)
        i += 1
    print("\x1b[?25h") #show cursor

def predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a vector of dimension m * 1.
        theta: has to be an numpy.array, a vector of dimension 2 * 1.
    Returns:
        y_hat as a numpy.array, a vector of dimension m * 1.
        None if x and/or theta are not numpy.array.
        None if x or theta are empty numpy.array.
        None if x or theta dimensions are not appropriate.
    Raises:
        This function should not raise any Exceptions.
    """
    # control args
    if not isinstance(x,np.ndarray) or not isinstance(theta, np.ndarray):
        return None
    if len(x.shape) > 1 and x.shape[1] != 1:
        return None
    if (len(theta.shape) > 1 and theta.shape != (2, 1)) or (len(theta.shape) == 1 and theta.shape[0] != 2):
        return None
    if x.shape[-1] == 0 or theta.shape[-1] == 0:
        return None

    x_1 = np.c_[np.ones(x.shape[0]), x]
    return x_1.dot(theta)


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
    x_1 = np.c_[np.ones(x.shape[0]), x]
    h =  x_1.dot(theta)
    diff = h - y
    return np.array([[diff.sum() / m], [(diff * x).sum() / m]])

def fit_(x, y, theta, alpha, max_iter):
    """
    Description:
        Fits the model to the training dataset contained in x and y.
    Args:
        x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
    Returns:
        new_theta: numpy.ndarray, a vector of dimension 2 * 1.
        None if there is a matching dimension problem.
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
    
    new_theta = theta.copy()
    for i in ft_progress(range(max_iter)):
        gradien = simple_gradient(x,y,new_theta)
        t0 = new_theta[0][0]
        t1 = new_theta[1][0]
        t0 -= (alpha * gradien[0][0])
        t1 -= (alpha * gradien[1][0])
        new_theta= np.array([t0, t1]).reshape((-1, 1))
    return(new_theta)
