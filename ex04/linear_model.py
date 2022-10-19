# from cProfile import label
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mylinearregression import MyLinearRegression as MyLR


data = pd.read_csv("are_blue_pills_magics.csv")
Xpill = np.array(data['Micrograms']).reshape(-1,1)
Yscore = np.array(data['Score']).reshape(-1,1)
theta_hypot = np.array([[89.0], [-8.0]])

linear_model1 = MyLR(theta_hypot, 5e-8, 1500000, progress_bar=True)
mse_before_fit = MyLR.mse_(Yscore, linear_model1.predict_(Xpill))
predictions = linear_model1.predict_(Xpill)
# plt.plot(Xpill, predictions, 'r--', label="before fit_()")
label = f"hypothesis with f(x) = {theta_hypot[1][0]}x"
if theta_hypot[0][0] > 0:
    label += ' + '
elif theta_hypot[0][0] < 0:
    label +=  ' - '
label += f"{theta_hypot[0][0]} \tMSE = {mse_before_fit}"
print(label)

# fit
linear_model1.fit_(Xpill, Yscore)


predictions = linear_model1.predict_(Xpill)
mse_after_fit = MyLR.mse_(Yscore, predictions)

# draw f(x)
plt.plot(Xpill, predictions, 'g--', label="${S_{predict}}^{(pills)}$")

# draw predicted values
plt.scatter(Xpill, predictions, marker='X', color='g')

# draw real values
plt.scatter(Xpill, Yscore, c='blue', label="${S_{true}}^{(pills)}$")

t1 = round(linear_model1.thetas[1][0], 2)
t0 = round(linear_model1.thetas[0][0], 2)
mse_after_fit = round(MyLR.mse_(Yscore, predictions), 6)
label = "$\\theta_0 = "+str(t0)+"$  $\\theta_1 = "+str(t1)+"$ MSE = "+str(mse_after_fit)
plt.text(1,1,label)

plt.xlabel("Quantity of blue pill (in micrograms)")
plt.ylabel("Space driving score")
plt.legend(bbox_to_anchor=(0.0, 1.0), loc="lower left", frameon=False, ncol=2)
plt.tight_layout()
plt.grid(True)
plt.show()

min = linear_model1.thetas[0][0] - 10.0
max = linear_model1.thetas[0][0] + 10.0
for theta0 in np.linspace(min,max,5, endpoint=False):
    x = np.arange(-14, -3, 0.1)
    y = []
    for x_ in x:
        t1 = x_
        t0 = theta0
        linear_model1.thetas= np.array([t0, t1]).reshape((-1, 1))
        y_hat = linear_model1.predict_(Xpill)
        y.append(linear_model1.loss_(Yscore, y_hat))
    label="$J(\\theta_0="
    label += str(round(theta0,2))
    label += ",\\theta_1)$"
    plt.plot(x, y, label=label)

plt.xlabel("$\\theta_1$")
plt.ylabel("cost function $J(\\theta_0,\\theta_1)$")
plt.ylim([10, 140])
plt.grid(True)
plt.legend(frameon=False)
plt.show()