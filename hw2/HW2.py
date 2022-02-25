# Name: 
# COMP 347 - Machine Learning
# HW No. 2

# Libraries
import pandas as pd
import scipy.linalg as LA
import numpy as np
import matplotlib.pyplot as plt
import random

# Problem 1 - Linear Regression with Athens Temperature Data
#------------------------------------------------------------------------------

# In the following problem, implement the solution to the least squares problem.

# 1a. Complete the following functions:

def A_mat(x, deg):
    """Create the matrix A part of the least squares problem.
       x: vector of input data.
       deg: degree of the polynomial fit."""
    x = np.vstack(x)
    A = np.empty(shape=(0, deg + 1))
    for point in x:
        max_deg = deg
        currRow = np.array([])
        while max_deg >= 0:
            currRow = np.append(currRow, point ** max_deg)
            max_deg = max_deg - 1
        A = np.vstack([A, currRow])
    return A

def LLS_Solve(x,y, deg):
    """Find the vector w that solves the least squares regression.
       x: vector of input data.
       y: vector of output data.
       deg: degree of the polynomial fit."""
    A = A_mat(x, deg)
    ATA = np.matmul(np.transpose(A), A)
    invATA = LA.inv(ATA)
    y = np.vstack(y)
    ATy = np.matmul(np.transpose(A), y)
    w = np.matmul(invATA, ATy)
    return w

def LLS_ridge(x,y,deg,lam):
    """Find the vector w that solves the ridge regresssion problem.
       x: vector of input data.
       y: vector of output data.
       deg: degree of the polynomial fit.
       lam: parameter for the ridge regression."""
    y = np.vstack(y)
    A = A_mat(x, deg)
    ATA = np.matmul(np.transpose(A), A)
    lamI = np.multiply(np.identity(ATA.shape[0]), lam)
    invCom = LA.inv(np.add(ATA, lamI))
    ATy = np.matmul(np.transpose(A), y)
    w = np.matmul(invCom, ATy)
    return w

def poly_func(data, coeffs):
    """Produce the vector of output data for a polynomial.
       data: x-values of the polynomial.
       coeffs: vector of coefficients for the polynomial."""
    data = np.vstack(data)
    coeffs = np.vstack(coeffs)
    A = A_mat(data, np.size(coeffs) - 1)
    output = np.matmul(A, coeffs)
    return output

def LLS_func(x,y,w,deg):
    """The linear least squares objective function.
       x: vector of input data.
       y: vector of output data.
       w: vector of weights.
       deg: degree of the polynomial."""
    A = A_mat(x, deg)
    objFunct = LA.norm(np.multiply(A, w) - y) ** 2
    return objFunct

def RMSE(x,y,w):
    """Compute the root mean square error.
       x: vector of input data.
       y: vector of output data.
       w: vector of weights."""
    x = np.vstack(x)
    A = A_mat(x, np.size(w) - 1)
    y = np.vstack(y)
    normComp = np.square(LA.norm(np.subtract(y, np.matmul(A, w))))
    rmse = (normComp/np.size(x)) ** (1/2)
    return rmse


# 1b. Solve the least squares linear regression problem for the Athens 
#     temperature data.  Make sure to annotate the plot with the RMSE.
athensDF = pd.read_csv("athens_ww2_weather.csv", usecols=['MaxTemp', 'MinTemp'])
athensx = athensDF['MinTemp'].astype(float).to_list()
athensy = athensDF['MaxTemp'].astype(float).to_list()
athenssol = LLS_Solve(athensx, athensy, 1)
athenspoints = poly_func(athensx, athenssol)
athensrmse = RMSE(athensx, athensy, athenssol)
plt.scatter(athensx, athensy)
plt.plot(athensx, athenspoints, label="RMSE: " + str(athensrmse), color='red')
plt.title("Athens Temp. Data Linear Fit")
plt.xlabel("Min. Temperature")
plt.ylabel("Max. Temperature")
plt.legend()
plt.show()


# Problem 2 -- Polynomial Regression with the Yosemite Visitor Data
#------------------------------------------------------------------------------

# 2a. Create degree-n polynomial fits for 5 years of the Yosemite data, with
#     n ranging from 1 to 20.  Additionally, create plots comparing the 
#     training error and RMSE for 3 years of data selected at random (distinct
#     from the years used for training).
yosemiteDF = pd.read_csv('Yosemite_Visits.csv')
# print(yosemiteDF)
yose2018 = [int(i.replace(',', '')) for i in yosemiteDF.iloc[0].to_list()[1:]]
yose2008 = [int(i.replace(',', '')) for i in yosemiteDF.iloc[10].to_list()[1:]]
yose1998 = [int(i.replace(',', '')) for i in yosemiteDF.iloc[20].to_list()[1:]]
yose1988 = [int(i.replace(',', '')) for i in yosemiteDF.iloc[30].to_list()[1:]]
yose1979 = [int(i.replace(',', '')) for i in yosemiteDF.iloc[39].to_list()[1:]]
# print(yose2018)
monthValues = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

def trainingVisuals():
    plt.scatter(monthValues, yose2018, c="red")
    plt.plot(monthValues, yose2018, c="red", label="2018")
    plt.scatter(monthValues, yose2008, c="orange")
    plt.plot(monthValues, yose2008, c="orange", label="2008")
    plt.scatter(monthValues, yose1998, c="yellow")
    plt.plot(monthValues, yose1998, c="yellow", label="1998")
    plt.scatter(monthValues, yose1988, c="green")
    plt.plot(monthValues, yose1988, c="green", label="1988")
    plt.scatter(monthValues, yose1979, c="blue")
    plt.plot(monthValues, yose1979, c="blue", label="1979")
    plt.xlabel("Month")
    plt.ylabel("Visitors")

yosex = []
yosex += monthValues * 5
yosex = sorted(yosex)
yosey = []
for i in range(12):
    yosey.append(yose2018[i])
    yosey.append(yose2008[i])
    yosey.append(yose1998[i])
    yosey.append(yose1988[i])
    yosey.append(yose1979[i])

for i in range(20):
    yosesol = LLS_Solve(yosex, yosey, i + 1)
    yosepoints = poly_func(np.linspace(1, 12, 100), yosesol)
    yosermse = RMSE(yosex, yosey, yosesol)
    trainingVisuals()
    plt.plot(np.linspace(1, 12, 100), yosepoints, c="purple", label="RMSE: " + str(yosermse), linewidth=3)
    plt.title("" + str(i + 1) + "-degree fit")
    plt.legend()
    plt.show()

# select 3 random non-training years
testYears = []
while 1:
    num = random.randint(1, 38)
    if len(testYears) == 3:
        break
    else:
        alreadyHas = False
        for y in testYears:
            if y == num:
                alreadyHas = True
        if not alreadyHas:
            testYears.append(num)

for year in testYears:
    values = [int(i.replace(',', '')) for i in yosemiteDF.iloc[year].to_list()[1:]]
    colors = []
    for i in range(3):
        colors.append(random.random())
    plt.scatter(monthValues, values, c=colors)
    plt.plot(monthValues, values, c=colors, label=2018-year)

yoseeval = poly_func(np.linspace(1, 12, 100), LLS_Solve(yosex, yosey, 7))

plt.plot(np.linspace(1, 12, 100), yoseeval, c="purple", linewidth=3)
plt.title("7-degree fit compared against 3 random years")
plt.xlabel("Month")
plt.ylabel("Visitors")
plt.legend()
plt.show()

# 2b. Solve the ridge regression regularization fitting for 5 years of data for
#     a fixed degree n >= 10.  Vary the parameter lam over 20 equally-spaced
#     values from 0 to 1.  Annotate the plots with this value.  
for i in range(20):
    yoseridge = LLS_ridge(yosex, yosey, 12, 0.05 * (i + 1))
    ridgepoints = poly_func(np.linspace(1, 12, 100), yoseridge)
    trainingVisuals()
    plt.plot(np.linspace(1, 12, 100), ridgepoints, c="purple", label="λ = " + str(np.around(0.05 * (i + 1), 2)), linewidth=3)
    plt.title("Ridge 12-degree fit w/ λ = " + str(np.around(0.05 * (i + 1), 2)))
    plt.legend()
    plt.show()