from numpy import *
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

#%% Q1(b)
scale = linspace(2, 7, 100000)
y_true = power(scale, 2)


### K = 1
def K1(x):
    if x < 3.5:
        return 4
    elif x < 6:
        return 25
    else:
        return 49

y1 = []

for i in scale:
    y_1 = K1(i)
    y1.append(y_1)

RMSE1 = mean_squared_error(y_true, y1, squared=False)
R2_K1 = r2_score(y_true, y1)

K = range(0, len(scale))
SUM1 = 0
for k in K:
    error1 = y_true[k] - y1[k]
    square1 = power(error1, 2)
    SUM1 += square1
    E1 = sqrt(SUM1 / len(scale))

# print(SUM1)
#print("The RMSE of K = 1 is ", E1)
print("The RMSE of K = 1 is ", RMSE1)
print("The R^2 of K = 1 is ", R2_K1)



### K = 2
def K2(x):
    if x < 2:
        return (70 - 29 * x)/(7-2 * x)
    elif x == 2:
        return 4
    elif x < 4.5:
        return 7 * x - 10
    elif x < 5:
        return (210 - 37 * x)/(6 - x)
    elif x == 5:
        return 25
    elif x < 7:
        return 12 * x - 35
    else:
        return (37 * x - 210)/(x - 6)
        
y2 = []

for i in scale:
    y_2 = K2(i)
    y2.append(y_2)

RMSE2 = mean_squared_error(y_true, y2, squared=False)
R2_K2 = r2_score(y_true, y2)

K = range(0, len(scale))
SUM2 = 0
for k in K:
    error2 = y_true[k] - y2[k]
    square2 = power(error2, 2)
    SUM2 += square2
    E2 = sqrt(SUM2 / len(scale))

# print(SUM2)
#print("The RMSE of K = 2 is ", E2)
print("The RMSE of K = 2 is ", RMSE2)
print("The R^2 of K = 2 is ", R2_K2)

### K = 3
def K3(x):
    if x < 2:
        return (78 * power(x, 2) - 616 * x + 980)/(3 * power(x, 2) - 28 * x + 59)
    elif x == 2:
        return 4
    elif x < 5:
        return - (70 * power(x, 2) - 520 * x + 700)/( -power(x, 2) + 4 * x + 11)
    elif x == 5:
        return 25
    elif x < 7:
        return - (20 * power(x, 2) - 70 * x)/(power(x, 2) - 14 * x + 39)
    else:
        return (78 * power(x, 2) - 616 * x + 980)/(3 * power(x, 2) - 28 * x + 59)
        
y3 = []

for i in scale:
    y_3 = K3(i)
    y3.append(y_3)

RMSE3 = mean_squared_error(y_true, y3, squared=False)
R2_K3 = r2_score(y_true, y3)

K = range(0, len(scale))
SUM3 = 0
for k in K:
    error3 = y_true[k] - y3[k]
    square3 = power(error3, 2)
    SUM3 += square3
    E3 = sqrt(SUM3 / len(scale))

# print(SUM3)
#print("The RMSE of K = 3 is ", E3)
print("The RMSE of K = 3 is ", RMSE3)
print("The R^2 of K = 3 is ", R2_K3)

#%%
plt.figure()
plt.plot(scale, y1, 'r') 
plt.plot(scale, y_true, "b")
plt.title("k=1")

plt.figure()
plt.plot(scale, y2, 'r') 
plt.plot(scale, y_true, "b")
plt.title("k=2")

plt.figure()
plt.plot(scale, y3, 'r') 
plt.plot(scale, y_true, "b")
plt.title("k=3")
# plt.show()
