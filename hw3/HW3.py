# Name: Wilton Cappel
# COMP 347 - Machine Learning
# HW No. 3

# Libraries
import pandas as pd
import scipy.linalg as LA
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.interpolate import make_interp_spline, BSpline

# Problem 1 - Gradient Descent Using Athens Temperature Data
#------------------------------------------------------------------------------
# For this problem you will be implementing various forms of gradient descent  
# using the Athens temperature data.  Feel free to copy over any functions you 
# wrote in HW #2 for this.  WARNING: In order to get gradient descent to work
# well, I highly recommend rewriting your cost function so that you are dividing
# by N (i.e. the number of data points there are).  Carry this over into all 
# corresponding expression where this would appear (the derivative being one of them).

# Functions from HW2
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

def poly_func(data, coeffs):
    """Produce the vector of output data for a polynomial.
       data: x-values of the polynomial.
       coeffs: vector of coefficients for the polynomial."""
    data = np.vstack(data)
    coeffs = np.vstack(coeffs)
    A = A_mat(data, np.size(coeffs) - 1)
    output = np.matmul(A, coeffs)
    return output

#Modified LLS cost funct. to divide by N
def LLS_func(x,y,w,deg):
    """The linear least squares objective function.
       x: vector of input data.
       y: vector of output data.
       w: vector of weights.
       deg: degree of the polynomial."""
    A = A_mat(x, deg)
    objFunct = LA.norm(np.matmul(A, w) - y) ** 2
    return objFunct / len(x)


# 1a. Fill out the function for the derivative of the least-squares cost function:
def LLS_deriv(x,y,w,deg):
    """Computes the derivative of the least squares cost function with input
    data x, output data y, coefficient vector w, and deg (the degree of the
    least squares model)."""
    A = A_mat(x, deg)
    Aw = np.matmul(A, w)
    AT = np.transpose(A)
    deriv = np.matmul(np.multiply(2, AT), Aw - y)
    return deriv / len(x)

# 1b. Implement gradient descent as a means of optimizing the least squares cost
#     function.  Your method should include the following:
#       a. initial vector w that you are optimizing,
#       b. a tolerance K signifying the acceptable derivative norm for stopping
#           the descent method,
#       c. initial derivative vector D (initialization at least for the sake of 
#           starting the loop),
#       d. an empty list called d_hist which captures the size (i.e. norm) of your
#           derivative vector at each iteration, 
#       e. an empty list called c_hist which captures the cost (i.e. value of
#           the cost function) at each iteration,
#       f. implement backtracking line search as part of your steepest descent
#           algorithm.  You can implement this on your own if you're feeling 
#           cavalier, or if you'd like here's a snippet of what I used in mine:
#
               # eps = 1
               # m = LA.norm(D)**2
               # t = 0.5*m
               # while LLS_func(a_min, a_max, w - eps*D, 1) > LLS_func(a_min, a_max, w, 1) - eps*t:
               #     eps *= 0.9
#
#       Plot curves showing the derivative size (i.e. d_hist) and cost (i.e. c_hist)
#       with respect to the number of iterations.
athensDF = pd.read_csv("athens_ww2_weather.csv", usecols=['MaxTemp', 'MinTemp'])
athensx = athensDF['MinTemp'].astype(float).to_list()
athensy = athensDF['MaxTemp'].astype(float).to_list()

# w = [100, -100]
# K = 0.01
# D = LLS_deriv(athensx, athensy, w, 1)
# iterations = []
# d_hist = []
# c_hist = []
# iter = 0
# while LA.norm(D) >= K:
#     iter += 1
#     iterations.append(iter)
#     d_hist.append(LA.norm(D))
#     c_hist.append(LLS_func(athensx, athensy, w, 1))
#     eps = 1
#     m = LA.norm(D) ** 2
#     t = 0.5 * m
#     while LLS_func(athensx, athensy, w - eps * D, 1) > LLS_func(athensx, athensy, w, 1) - eps * t:
#         eps *= 0.9
#     wnext = w - (eps * D)
#     w = wnext
#     D = LLS_deriv(athensx, athensy, w, 1)
#     print(w)
#
# athenspoints = poly_func(athensx, w)
# plt.scatter(athensx, athensy)
# plt.plot(athensx, athenspoints)
# plt.show()
#
# xnew = np.linspace(np.array(iterations).min(), np.array(iterations).max(), 200)
# cspl = make_interp_spline(iterations, c_hist, k=3)
# dspl = make_interp_spline(iterations, d_hist, k=3)
# c_smooth = cspl(xnew)
# d_smooth = dspl(xnew)
# plt.plot(xnew, c_smooth, label="Cost")
# plt.plot(xnew, d_smooth, label="Size of deriv.")
# plt.legend()
# plt.show()

# 1c. Repeat part 1b, but now implement mini-batch gradient descent by randomizing
#     the data points used in each iteration.  Perform mini-batch descent for batches
#     of size 5, 10, 25, and 50 data points.  For each one, plot the curves
#     for d_hist and c_hist.  Plot these all on one graph showing the relationship
#     between batch size and convergence speed (at least for the least squares 
#     problem).  Feel free to adjust the transparency of your curves so that 
#     they are easily distinguishable.

def athensDesc(batchsize):
    athensxRem = athensx
    athensyRem = athensy
    shuffleTogether = list(zip(athensxRem, athensyRem))
    random.shuffle(shuffleTogether)
    athensxRem, athensyRem = zip(*shuffleTogether)
    xbatch = athensxRem[0:batchsize]
    ybatch = athensyRem[0:batchsize]
    w = [100, -100]
    K = 0.01
    D = LLS_deriv(xbatch, ybatch, w, 1)
    d_hist = []
    c_hist = []
    iter = 0
    iterations = []
    while LA.norm(D) >= K:
        iter += 1
        iterations.append(iter)
        shuffleTogether = list(zip(athensxRem, athensyRem))
        random.shuffle(shuffleTogether)
        athensxRem, athensyRem = zip(*shuffleTogether)
        xbatch = athensxRem[0:batchsize]
        ybatch = athensyRem[0:batchsize]
        eps = 1
        m = LA.norm(D) ** 2
        t = 0.5 * m
        while LLS_func(xbatch, ybatch, w - eps * D, 1) > LLS_func(xbatch, ybatch, w, 1) - eps * t:
            eps *= 0.9
        wnext = w - (eps * D)
        w = wnext
        D = LLS_deriv(xbatch, ybatch, w, 1)
        d_hist.append(LA.norm(D))
        c_hist.append(LLS_func(xbatch, ybatch, w, 1))
        print(w)
    xnew = np.linspace(np.array(iterations).min(), np.array(iterations).max(), 200)
    cspl = make_interp_spline(iterations, c_hist, k=3)
    dspl = make_interp_spline(iterations, d_hist, k=3)
    c_smooth = cspl(xnew)
    d_smooth = dspl(xnew)
    plt.plot(xnew, c_smooth, label="Cost")
    plt.plot(xnew, d_smooth, label="Size of deriv.")
    plt.title("Batch size: " + str(batchsize))
    plt.xlabel("Iterations")
    plt.legend()
    plt.show()
    print(c_hist)
    print(d_hist)
    return (c_hist, d_hist, iterations)

athens5c, athens5d, athens5i = athensDesc(5)
athens10c, athens10d, athens10i = athensDesc(10)
athens25c, athens25d, athens25i = athensDesc(25)
athens50c, athens50d, athens50i = athensDesc(50)

# 1d. Repeat 1b, but now implement stochastic gradient descent.  Plot the curves 
#     for d_hist and c_hist.  WARNING: There is a strong possibility that your
#     cost and derivative definitions may not compute the values correctly for
#     for the 1-dimensional case.  If needed, make sure that you adjust these functions
#     to accommodate a single data point.

athens1c, athens1d, athens1i = athensDesc(1)

# 1e. Aggregate your curves for batch, mini-batch, and stochastic descent methods
#     into one final graph so that a full comparison between all methods can be
#     observed.  Make sure your legend clearly indicates the results of each 
#     method.  Adjust the transparency of the curves as needed.


# Problem 2 - LASSO Regularization
#------------------------------------------------------------------------------
# For this problem you will be implementing LASSO regression on the Yosemite data.

# 2a. Fill out the function for the soft-thresholding operator S_lambda as discussed
#     in lecture:

def soft_thresh(v, lam):
    """Perform the soft-thresholding operation of the vector v using parameter lam."""
    res = np.array([])
    for xi in v:
        if xi > lam:
            res = np.append(res, xi - lam)
        elif abs(xi) <= lam:
            res = np.append(res, 0)
        else:
            res = np.append(res, xi + lam)
    return res
    
# 2b. Using 5 years of the Yosemite data, perform LASSO regression with the values 
#     of lam ranging from 0.25 up to 5, spacing them in increments of 0.25.
#     Specifically do this for a cubic model of the Yosemite data.  In doing this
#     save each of your optimal parameter vectors w to a list as well as solving
#     for the exact solution for the least squares problem.  Make the following
#     graphs:
#
#       a. Make a graph of the l^2 norms (i.e. Euclidean) and l^1 norms of the 
#          optimal parameter vectors w as a function of the coefficient lam.  
#          Interpret lam = 0 as the exact solution.  One can find the 1-norm of
#          a vector using LA.norm(w, ord = 1)
#       b. For each coefficient in the cubic model (i.e. there are 4 of these),
#           make a separate plot of the absolute value of the coefficient as a 
#           function of the parameter lam (again, lam = 0 should be the exact
#           solution to the original least squares problem).  Is there a 
#           discernible trend of the sizes of our entries for increasing values
#           of lam?

# Friendly Reminder: for LASSO regression you don't need backtracking line search.
# In essence the parameter lam serves as our stepsize.

def athensLasso(lam):
    athensxLass = athensx
    athensyLass = athensy
    w = [100, -100]
    K = 0.01
    D = np.vstack([-1, 1])
    iter = 0
    while LA.norm(D) >= K and iter < 20000:
        threshInput = w - lam * (LLS_deriv(athensxLass, athensyLass, w, 1))
        wnext = soft_thresh(threshInput, lam)
        w = wnext
        D = LLS_deriv(athensxLass, athensyLass, w, 1)
        iter += 1
        print(w)
    return w

lams = []
l1norms = []
l2norms = []
for i in range(15):
    currW = athensLasso(0.1 * (i + 1) / len(athensx))
    l1norms.append(LA.norm(currW, ord=1))
    l2norms.append(LA.norm(currW))
    lams.append(0.1 * (i + 1) / len(athensx))
lamsx = np.linspace(np.array(lams).min(), np.array(lams).max(), 200)
l1spl = make_interp_spline(lams, l1norms, k=3)
l2spl = make_interp_spline(lams, l2norms, k=3)
l1smooth = l1spl(lamsx)
l2smooth = l2spl(lamsx)
plt.plot(lamsx, l1smooth, label="L1 norm of optimal w")
plt.plot(lamsx, l2smooth, label="L2 norm of optimal w")
plt.legend()
plt.show()
