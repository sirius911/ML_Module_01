import numpy as np
from fit import fit_, predict_
x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
theta= np.array([1, 1]).reshape((-1, 1))
# Example 0:
theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1500000)
print(theta1)
# Output:
# array([[1.40709365],
# [1.1150909 ]])
# Example 1:
print(predict_(x, theta1))
# Output:
# array([[15.3408728 ],
# [25.38243697],
# [36.59126492],
# [55.95130097],
# [65.53471499]])

print("Random x array and linear expression f(x) with theta([3.56, 2.7888])")
x = np.random.randn(5).reshape((-1, 1))
theta= np.array([3.56, 2.7888]).reshape((-1, 1))
x_ = np.c_[np.ones(x.shape[0]), x]
y = x_.dot(theta)
print("x = ")
print(x)
print("y = ")
print(y)
print("fit")
theta1 = fit_(x, y, np.array([3., 2.0]).reshape((-1, 1)), alpha=0.000001, max_iter=1500000)
print(theta1)
print(predict_(x, theta1))
