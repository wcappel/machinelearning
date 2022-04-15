# Name: 
# COMP 347 - Machine Learning
# HW No. 3

#*********************
# PROBLEMS ARE POSTED BELOW THE FUNCTIONS SECTION!!!
#*********************

# Libraries
#------------------------------------------------------------------------------
from collections import Counter
from itertools import combinations
import numpy as np
import pandas as pd
import scipy.linalg as LA
import matplotlib.pyplot as plt

# Functions
#------------------------------------------------------------------------------
def fact(n):
    """Factorial of an integer n>=0."""
    if n in [0,1]:
        return 1
    else:
        return n*fact(n-1)

def partition(number:int, max_vals:tuple):
    S = set(combinations((k for i,val in enumerate(max_vals) for k in [i]*val), number))
    for s in S:
        c = Counter(s)
        yield tuple([c[n] for n in range(len(max_vals))])

def RBF_Approx(X,gamma,deg):
    """Transforms data in X to its RBF representation, but as an approximation
    in deg degrees.  gamma = 1/2."""
    new_X = []; N = X.shape[0]; n = X.shape[1]; count = 0
    for i in range(N):
        vec = []
        for k in range(deg+1):
            if k == 0:
                vec += [1]
            else:
                tup = (k,)*n
                parts = list(partition(k, tup))
                for part in parts:
                    vec += [np.prod([np.sqrt(gamma**deg)*(X[i,s]**part[s])/np.sqrt(fact(part[s])) for s in range(n)])]
        new_X += [np.exp(-gamma*LA.norm(X[i,:])**2)*np.asarray(vec)]
        print(str(count) + " of " + str(N))
        count += 1
    
    return np.asarray(new_X)

def smo_algorithm(X,y,C, max_iter, thresh):
    """Optimizes Lagrange multipliers in the dual formulation of SVM.
        X: The data set of size Nxn where N is the number of observations and
           n is the length of each feature vector.
        y: The class labels with values +/-1 corresponding to the feature vectors.
        C: A threshold positive value for the size of each lagrange multiplier.  
           In other words 0<= a_i <= C for each i.
        max_iter: The maximum number of successive iterations to attempt when
                  updating the multipliers.  The multipliers are randomly selected
                  as pairs a_i and a_j at each iteration and updates these according
                  to a systematic procedure of thresholding and various checks.
                  A counter is incremented if an update is less than the value
                  thresh from its previous iteration.  max_iter is the maximum
                  value this counter attains before the algorithm terminates.
        thresh: The minimum threshold difference between an update to a multiplier
                and its previous iteration.
    """
    alph = np.zeros(len(y)); b = 0
    count = 0
    while count < max_iter:
        
        num_changes = 0
        
        for i in range(len(y)):
            w = np.dot(alph*y, X)
            E_i = np.dot(w, X[i,:]) + b - y[i]
        
            if (y[i]*E_i < -thresh and alph[i] < C) or (y[i]*E_i > thresh and alph[i] > 0):
                j = np.random.choice([m for m in range(len(y)) if m != i])
                E_j = np.dot(w, X[j,:]) + b - y[j]
                
                a_1old = alph[i]; a_2old = alph[j]
                y_1 = y[i]; y_2 = y[j]
                
                # Compute L and H
                if y_1 != y_2:
                    L = np.max([0, a_2old - a_1old])
                    H = np.min([C, C + a_2old - a_1old])
                elif y_1 == y_2:
                    L = np.max([0, a_1old + a_2old - C])
                    H = np.min([C, a_1old + a_2old])
                
                if L == H:
                    continue
                eta = 2*np.dot(X[i,:], X[j,:]) - LA.norm(X[i,:])**2 - LA.norm(X[j,:])**2
                if eta >= 0:
                    continue
                #Clip value of a_2
                a_2new = a_2old - y_2*(E_i - E_j)/eta
                if a_2new >= H:
                    a_2new = H
                elif a_2new < L:
                    a_2new = L
                
                if abs(a_2new - a_2old) < thresh:
                    continue
                
                a_1new = a_1old + y_1*y_2*(a_2old - a_2new)
                
                # Compute b
                b_1 = b - E_i - y_1*(a_1new - a_1old)*LA.norm(X[i,:]) - y_2*(a_2new - a_2old)*np.dot(X[i,:], X[j,:])
                b_2 = b - E_j - y_1*(a_1new - a_1old)*np.dot(X[i,:], X[j,:]) - y_2*(a_2new - a_2old)*LA.norm(X[j,:])
                
                if 0 < a_1new < C:
                    b = b_1
                elif 0 < a_2new < C:
                    b = b_2
                else:
                    b = (b_1 + b_2)/2
                
                num_changes += 1
                alph[i] = a_1new
                alph[j] = a_2new
                
        if num_changes == 0:
            count += 1
        else:
            count = 0
        print(count)
    return alph, b


def hinge_loss(X,y,w,b):
    """Here X is assumed to be Nxn where each row is a data point of length n and
    N is the number of data points.  y is the vector of class labels with values
    either +1 or -1.  w is the support vector and b the corresponding bias."""
    result = 0
    halfNormWSquared = (LA.norm(w) ** 2) / 2
    for i, xi in enumerate(X):
        val = max(0, 1 - (y[i] * (np.dot(w, xi) + b)))
        result += halfNormWSquared + val
    return result / np.size(X)

def hinge_deriv(X,y,w,b):
    """Here X is assumed to be Nxn where each row is a data point of length n and
    N is the number of data points.  y is the vector of class labels with values
    either +1 or -1.  w is the support vector and b the corresponding bias."""
    wUpdate = 0
    bUpdate = 0
    for i, xi in enumerate(X):
        if y[i] * ((np.matmul(np.transpose(w), xi)) + b) >= 1:
            wUpdate += w
            bUpdate += 0
        else:
            wUpdate += w - (y[i] * xi)
            bUpdate += y[i] * -1
    return wUpdate / np.size(X), bUpdate / np.size(X)

# Problem #1 - Hinge Loss Optimization for SVM on Randomized Test Data
#------------------------------------------------------------------------------
#In this problem you will be performing SVM using the hinge loss formalism as 
#presented in lecture.
    
# 1a. Complete the function hinge_loss and hinge_deriv in the previous section.
    
# 1b. Perform SVM Using the hinge loss formalism on the data in svm_test_2.csv.
#     Use an appropriate initialization to your gradient descent algorithm.
svm2DF = pd.read_csv("svm_test_2.csv", usecols=['0', '1', '2'])
y = svm2DF['2'].astype(float).to_list()
X = svm2DF[['0', '1']].to_numpy()
xblue = (svm2DF.loc[svm2DF['2'] == 1])[['0', '1']]
xred = (svm2DF.loc[svm2DF['2'] == -1])[['0', '1']]
w = [-10, 10]
b = 0
eps = 0.1
K = 0.01
currDeriv = hinge_deriv(X, y, w, b)
D = [currDeriv[0][0], currDeriv[0][1], currDeriv[1]]
iter = 0
while LA.norm(D) >= K and iter < 10000:
    iter += 1
    currDeriv = hinge_deriv(X, y, w, b)
    w = w - eps * currDeriv[0]
    b = b - eps * currDeriv[1]
    D = [currDeriv[0][0], currDeriv[0][1], currDeriv[1]]
print(w)
print(b)

# 1c. Perform SVM on the data in svm_test_2.csv now using the Lagrange multiplier
#     formalism by calling the function smo_algorithm presented above.  Optimize 
#     this for values of C = 0.25, 0.5, 0.75, and 1.  I recommend taking 
#     max_iter = 2500 and thresh = 1e-5 when calling the smo_algorithm.
plt.scatter(x=xblue['0'].to_list(), y=xblue['1'].to_list(), c="blue")
plt.scatter(x=xred['0'].to_list(), y=xred['1'].to_list(), c="red")
LM25 = smo_algorithm(X, y, 0.25, 2500, 1e-5)
a25 = LM25[0]
b25 = LM25[1]
w25 = 0
LM50 = smo_algorithm(X, y, 0.5, 2500, 1e-5)
a50 = LM50[0]
b50 = LM50[1]
w50 = 0
LM75 = smo_algorithm(X, y, 0.75, 2500, 1e-5)
a75 = LM75[0]
b75 = LM75[1]
w75 = 0
LM100 = smo_algorithm(X, y, 1, 2500, 1e-5)
a100 = LM100[0]
b100 = LM100[1]
w100 = 0
for i, xi in enumerate(X):
    w100 += a100[i] * y[i] * xi
    w25 += a25[i] * y[i] * xi
    w50 += a50[i] * y[i] * xi
    w75 += a75[i] * y[i] * xi
print(w100)
print(b100)

# 1d. Make a scatter plot of the data with decision boundary lines indicating
#     the hinge model, and the various Lagrange models found from part c.  Make
#     sure you have a legend displaying each one clearly and adjust the transparency
#     as needed.
plt.plot(np.linspace(min(svm2DF['0'].to_numpy()), max(svm2DF['0'].to_numpy()), 100),
         - (1 / (w[0])) * (w[1] * (np.linspace(min(svm2DF['0'].to_numpy()), max(svm2DF['0'].to_numpy()), 100)) + b), label="Hinge boundary", alpha=0.8)
plt.plot(np.linspace(min(svm2DF['0'].to_numpy()), max(svm2DF['0'].to_numpy()), 100),
         - (1 / (w100[0])) * (w100[1] * (np.linspace(min(svm2DF['0'].to_numpy()), max(svm2DF['0'].to_numpy()), 100)) + b100), label="Lagrange C = 1 boundary", alpha=0.8)
plt.plot(np.linspace(min(svm2DF['0'].to_numpy()), max(svm2DF['0'].to_numpy()), 100),
         - (1 / (w25[0])) * (w25[1] * (np.linspace(min(svm2DF['0'].to_numpy()), max(svm2DF['0'].to_numpy()), 100)) + b25), label="Lagrange C = 0.25 boundary", alpha=0.8)
plt.plot(np.linspace(min(svm2DF['0'].to_numpy()), max(svm2DF['0'].to_numpy()), 100),
         - (1 / (w50[0])) * (w50[1] * (np.linspace(min(svm2DF['0'].to_numpy()), max(svm2DF['0'].to_numpy()), 100)) + b50), label="Lagrange C = 0.50 boundary", alpha=0.8)
plt.plot(np.linspace(min(svm2DF['0'].to_numpy()), max(svm2DF['0'].to_numpy()), 100),
         - (1 / (w75[0])) * (w75[1] * (np.linspace(min(svm2DF['0'].to_numpy()), max(svm2DF['0'].to_numpy()), 100)) + b75), label="Lagrange C = 0.75 boundary", alpha=0.8)
plt.legend()
plt.show()

# 1e. Perform SVM on the radial data, but preprocess the data by using a kernel
#     embedding of the data into 3 dimensions.  This can be accomplished by 
#     taking z = sqrt(x**2 + y**2) for each data point.  Learn an optimal model
#     for separating the data using the Lagrange multiplier formalism.  Experiment
#     with choices for C, max_iter, and thresh as desired.
radialDF = pd.read_csv("radial_data.csv")
radial2d = radialDF[['0', '1']].to_numpy()
radY = radialDF['2'].to_list()
radialX = []
for xi in radial2d:
    radialX.append([xi[0], xi[1], np.sqrt(xi[0]**2 + xi[1]**2)])
radialX = np.asarray(radialX)
LMRad = smo_algorithm(radialX, radY, 1, 2500, 1e-5)
aRad = LMRad[0]
bRad = LMRad[1]
wRad = 0
for i, xi in enumerate(radialX):
    wRad += aRad[i] * radY[i] * xi
    
# Problem #2 - Cross Validation and Testing for Breast Cancer Data
#------------------------------------------------------------------------------
# In this problem you will use the breast cancer data in an attempt to use SVM
# for a real-world classification problem.

# 2a. Pre-process the data so that you separate the main variables.  Your data
#     X should consist all but the first two and very last columns in the dataframe.
#     Create a variable Y to reinterpret the binary classifiers 'B' and 'M' as
#     -1 and +1, respectively.
bcDF = pd.read_csv("breast_cancer.csv")
bcDiags = bcDF['diagnosis'].to_list()
bcDF = bcDF.iloc[:, 2:-3]
bcY = []
bcX = []
for diag in bcDiags:
    if diag == 'B':
        bcY.append(-1)
    else:
        bcY.append(1)
bcX = np.asarray(bcDF.values.tolist())
bcY = np.asarray(bcY)

# 2b. Perform cross-validation using a linear SVM model on the data by dividing 
#     the indexes into 10 separate randomized classes (I recommend looking up 
#     np.random.shuffle and np.array_split).  Make sure you do the following:
#       1. Make two empty lists, Trained_models and Success_rates.  In Trained_models
#          save ordered pairs of the learned models [w,b] for each set of training
#          training data.  In Success_rates, save the percentage of successfully 
#          classified test points from the remaining partition of the data.  Remember
#          that the test for correct classification is that y(<w,x> + b) >= 1.
#       2. Make a histogram of your success rates.  Don't expect these to be stellar
#          numbers.  They will most likely be abysmal success rates.  Unfortunately
#          SVM is a difficult task to optimize by hand, which is why we are fortunate
#          to have kernel methods at our disposal.  Speaking of which.....

def bcSVM(X, y, method):
    if method == "rbf":
        X = RBF_Approx(X, 1e-6, 2)
    else:
        X = X
    Y = y
    sampleFolds = np.array_split(X, 10)
    labelFolds = np.array_split(Y, 10)
    Trained_models = []
    Success_rates = []
    for i, fold in enumerate(sampleFolds):
        LMBc = smo_algorithm(fold, labelFolds[i], 0.5, 500, 1e-300)
        aBc = LMBc[0]
        bBc = LMBc[1]
        wBc = 0
        for j, xi in enumerate(fold):
            wBc += aBc[j] * labelFolds[i][j] * xi
        successCount = 0
        for j, xi in enumerate(fold):
            if labelFolds[i][j] * (np.dot(np.transpose(wBc), xi) + bBc) >= 1:
                successCount += 1
        Success_rates.append(successCount / np.size(fold))
        Trained_models.append([wBc, bBc])
        print(Success_rates)
        print(Trained_models)

# bcSVM(bcX, bcY, "linear")

    
# 2c. Repeat cross-validation on the breast cancer data, but instead of a linear
#     SVM model, employ an approximation to the RBF kernel as discussed in class.
#     Note that what this does is that it transforms the original data x into a
#     variable X where the data is embedded in a higher dimension.  Generally when
#     data gets embedded in higher dimensions, there's more room for it to be spaced
#     out in and therefore increases the chances that your data will be linearly 
#     separable.  Do this for deg = 2,3.  I recommend taking gamma = 1e-6.
#     Don't be surprised if this all takes well over an hour to terminate.
bcSVM(bcX, bcY, "rbf")

# Notes for Problem #2:
# 1. To save yourself from writing the same code twice, I recommend making this
#    type of if/else statement before performing SVM on the breast cancer data:
    
#        METHOD = ''
#        if METHOD == 'Lin':
#            X = x
#        elif METHOD == 'RBF':
#            deg = 2; gamma = 1e-6
#            X = RBF_Approx(x,gamma,deg)

# 2. For implementing smo_algorithm for the breast cancer data, I recommend
#    taking max_iter = 500 and thresh = 1e-300
    
    
