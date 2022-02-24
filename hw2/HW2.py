# Name: 
# COMP 347 - Machine Learning
# HW No. 2

# Libraries
import pandas as pd
import scipy.linalg as LA
import numpy as np
import matplotlib.pyplot as plt

# Problem 1 - Linear Regression with Athens Temperature Data
#------------------------------------------------------------------------------

# In the following problem, implement the solution to the least squares problem.

# 1a. Complete the following functions:

def A_mat(x, deg):
    """Create the matrix A part of the least squares problem.
       x: vector of input data.
       deg: degree of the polynomial fit."""
    A = []
    for point in x:
        max_deg = deg
        currRow = []
        while max_deg >= 0:
            currRow.append(point ** max_deg)
            max_deg = max_deg - 1
        A.append(currRow)
    A = np.asmatrix(A)
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
    print(invATA)
    print(ATy)
    # print(np.shape(w))
    return w

def LLS_ridge(x,y,deg,lam):
    """Find the vector w that solves the ridge regresssion problem.
       x: vector of input data.
       y: vector of output data.
       deg: degree of the polynomial fit.
       lam: parameter for the ridge regression."""
    A = A_mat(x, deg)
    ATA = np.matmul(np.transpose(A), A)
    lamI = np.multiply(lam, np.identity(ATA.shape[0]))
    invCom = LA.inv(np.add(ATA, lamI))
    ATy = np.matmul(np.transpose(A), y)
    w = np.multiply(invCom, ATy)
    return w

def poly_func(data, coeffs):
    """Produce the vector of output data for a polynomial.
       data: x-values of the polynomial.
       coeffs: vector of coefficients for the polynomial."""
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
    A = A_mat(x, np.size(w) - 1)
    normComp = LA.norm(y - np.matmul(A, w)) ** 2
    rmse = (1/np.size(x) * normComp) ** (1/2)
    return rmse


# 1b. Solve the least squares linear regression problem for the Athens 
#     temperature data.  Make sure to annotate the plot with the RMSE.
DF = pd.read_csv("athens_ww2_weather.csv", usecols=['MaxTemp', 'MinTemp'])
athensx = DF['MinTemp'].astype(float).to_list()
athensy = DF['MaxTemp'].astype(float).to_list()
athenssol = LLS_Solve(athensx, athensy, 1)
points = poly_func(athensx, athenssol)
athensrmse = RMSE(athensx, athensy, athenssol)
plt.scatter(athensx, athensy)
plt.plot(athensx, points, label="RMSE: " + str(athensrmse))
plt.legend()
plt.show()



# Problem 2 -- Polynomial Regression with the Yosemite Visitor Data
#------------------------------------------------------------------------------

# 2a. Create degree-n polynomial fits for 5 years of the Yosemite data, with
#     n ranging from 1 to 20.  Additionally, create plots comparing the 
#     training error and RMSE for 3 years of data selected at random (distinct
#     from the years used for training).
    
    
# 2b. Solve the ridge regression regularization fitting for 5 years of data for
#     a fixed degree n >= 10.  Vary the parameter lam over 20 equally-spaced
#     values from 0 to 1.  Annotate the plots with this value.  
   