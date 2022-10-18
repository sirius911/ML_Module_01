import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from mylinearregression import MyLinearRegression as MyLR

green = '\033[92m'
reset = '\033[0m'
red = '\033[91m'
yellow = '\033[33m'

def okko(resultA, resultB):
    """
    return Ok or KO if resultA equal or not equal with resultB
    """
    if resultA == resultB:
        return (green + "OK" + reset)
    else:
        return (red + "KO" + reset)

data = pd.read_csv("are_blue_pills_magics.csv")
Xpill = np.array(data['Micrograms']).reshape(-1,1)
Yscore = np.array(data['Score']).reshape(-1,1)
linear_model1 = MyLR(np.array([[89.0], [-8]]), progress_bar=True)
linear_model2 = MyLR(np.array([[89.0], [-6]]), progress_bar=True)
Y_model1 = linear_model1.predict_(Xpill)
Y_model2 = linear_model2.predict_(Xpill)
mse1 = MyLR.mse_(Yscore, Y_model1)
skerl1 = mean_squared_error(Yscore, Y_model1)
print(f"{yellow}MyLR.mse_({reset}Yscore, Y_model1) = {mse1}  =>  {okko(mse1, skerl1)}")
# print(MyLR.mse_(Yscore, Y_model1))
# 57.60304285714282
# print(mean_squared_error(Yscore, Y_model1))
# 57.603042857142825
mse2 = MyLR.mse_(Yscore, Y_model2)
skerl2 = mean_squared_error(Yscore, Y_model2)
print(f"{yellow}MyLR.mse_({reset}Yscore, Y_model2) = {mse2}  =>  {okko(mse2, skerl2)}")
# print(MyLR.mse_(Yscore, Y_model2))
# 232.16344285714285
# print(mean_squared_error(Yscore, Y_model2))
# 232.16344285714285
