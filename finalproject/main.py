import numpy as np
import scipy.linalg as LA
import pandas as pd
import matplotlib.pyplot as plt
from math import exp

def sigmoid(z):
    """
    Returns sigmoid funct. evaluation for value z
    """
    return 1 / (1 + exp(-z))

def softmax(z):
    """
    Returns softmax vector output for vector z
    """
    result = []
    for zi in z:
        currSum = 0
        for zj in z:
            currSum += exp(zj)
        result.append(exp(zi)/currSum)
    return np.asarray(result)


def decisionBoundary(val):
    """
    Returns decision boundary evaluation for value 'val'
    """
    if 1 >= val >= 0:
        if val > 0.5:
            return 1
        else:
            return 0

def crossEntropyCost(yActual, w, x, b):
    """
    Returns cross-entropy loss value for yPred and yActual
    """
    return -(yActual * np.log2(sigmoid(np.dot(w, x) + b)) + ((1 - yActual) * np.log2(1 - sigmoid(np.dot(w, x) + b))))

def crossEntropyDeriv(w, x, b, yActual):
    """
    Returns gradient of cross-entropy for an observation vector x, each xj is a a feature count value
    """
    grad = []
    for xj in x:
        grad.append((sigmoid((np.dot(w, x)) + b) - yActual) * xj)
    grad.append(sigmoid((np.dot(w, x)) + b) - yActual)
    return np.asarray(grad)

def mushroomFeatures(X):
    A = []
    for xi in X:
        currFeatures = [0] * 8
        if xi[1] == 's':
            # smooth cap, likely to be edible
            currFeatures[0] = 1
        if xi[0] == 'x':
            # convex cap, likely to be edible
            currFeatures[1] = 1
        if xi[2] == 'w':
            # white cap, likely to be edible
            currFeatures[2] = 1
        if xi[8] != 'w':
            # non-white gills, likely to be edible
            currFeatures[3] = 1
        if xi[19] == 'w':
            # white spore print, likely to be edible
            currFeatures[4] = 1
        if xi[7] == 'b':
            # broad gill size, likely to be edible
            currFeatures[5] = 1
        if xi[3] == 't':
            # has bruises, likely to be edible
            currFeatures[6] = 1
        if xi[4] != 'n':
            # not odorless, likely to be edible
            currFeatures[7] = 1
        A.append(currFeatures)
    return np.asarray(A)

mushroomsDF = pd.read_csv('mushrooms.csv')
mushroomsDF = mushroomsDF.drop(columns=['class'])
mushroomsY = pd.read_csv('mushrooms.csv', usecols=[0])['class'].to_list()
mushroomsY = np.asarray([1 if y == 'e' else 0 for y in mushroomsY])
mushroomsMatrix = mushroomsDF.to_numpy()
mushroomFeatureMatrix = mushroomFeatures(mushroomsMatrix)
mushroomsTest = mushroomFeatureMatrix[round((len(mushroomFeatureMatrix) - 1) * 0.8):]
mushroomsFeatureMatrix = mushroomFeatureMatrix[round((len(mushroomFeatureMatrix) - 1) * 0.8):]
mushroomsYTest = mushroomsY[round((len(mushroomsY) - 1) * 0.8):]
mushroomsY = mushroomsY[:round((len(mushroomsY) - 1) * 0.8)]
w = [0] * 8
b = 0
K = 0.000001
D = crossEntropyDeriv(w, mushroomFeatureMatrix[0], b, mushroomsY[0])
iter = 0
while LA.norm(D) >= K and iter < 10000:
    eps = 1
    m = LA.norm(D) ** 2
    t = 0.5 * m
    currObservation = mushroomFeatureMatrix[iter % (len(mushroomFeatureMatrix))]
    z = np.dot(w, currObservation) + b
    sigVal = sigmoid(z)
    decisionVal = decisionBoundary(sigVal)
    while crossEntropyCost(mushroomsY[iter % (len(mushroomFeatureMatrix))], w - eps * D[0:-1], currObservation, b) > crossEntropyCost(mushroomsY[iter % (len(mushroomFeatureMatrix))], w, currObservation, b):
        eps *= 0.9
    w = w - (eps * D[0:-1])
    b = D[-1]
    D = crossEntropyDeriv(w, currObservation, b, mushroomsY[iter % (len(mushroomFeatureMatrix) - 1)])
    iter += 1
    print(w)
    print(b)
    print(iter)

predicted = []
for i in range(len(mushroomsTest)):
    predictedProb = sigmoid(np.dot(w, mushroomsTest[i]) + b)
    predicted.append(decisionBoundary(predictedProb))

print("Pred.:")
print(predicted)
print("Actual:")
print(list(mushroomsYTest))



