import numpy as np
import scipy.linalg as LA
import pandas as pd
import random
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
    s = np.max(z, axis=1)[:, np.newaxis]
    return np.exp(z - s) / np.sum(np.exp(z - s), axis=1)[:, np.newaxis]

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

def multinomialCost(x, y, w):
    """
    Returns cost value for multinomial model
    """
    z = -np.dot(x, w)
    return 1/x.shape[0] * (np.trace(np.dot(np.dot(x, w), np.transpose(y))) + np.sum(np.log(np.sum(np.exp(z), axis=1))))

def crossEntropyDeriv(w, x, b, yActual):
    """
    Returns gradient of cross-entropy for an observation vector x, each xj is a a feature count value
    """
    grad = []
    for xj in x:
        grad.append((sigmoid((np.dot(w, x)) + b) - yActual) * xj)
    grad.append(sigmoid((np.dot(w, x)) + b) - yActual)
    return np.asarray(grad)

def oneHotEncoded(list, numClasses):
    """
    Returns one-hot encoded vector for class labels
    """
    encodedY = np.zeros((len(list), numClasses))
    for i in range(0, len(list)):
        encodedY[i][list[i]] = 1
    return encodedY

def mushroomFeatures(X):
    """
    Converts data matrix X to a feature representation matrix
    """
    A = []
    for xi in X:
        currFeatures = [0] * 44
        # Cap surface
        if xi[1] == 's':
            currFeatures[0] = 1
        elif xi[1] == 'f':
            currFeatures[1] = 1
        elif xi[1] == 'g':
            currFeatures[2] = 1
        else:
            currFeatures[3] = 1

        # Cap shape
        if xi[0] == 'x':
            currFeatures[4] = 1
        elif xi[0] == 'b':
            currFeatures[5] = 1
        elif xi[0] == 'c':
            currFeatures[6] = 1
        elif xi[0] == 'f':
            currFeatures[7] = 1
        elif xi[0] == 'k':
            currFeatures[8] = 1
        else:
            currFeatures[9] = 1

        # Cap color
        if xi[2] == 'w':
            currFeatures[10] = 1
        elif xi[2] == 'n':
            currFeatures[11] = 1
        elif xi[2] == 'b':
            currFeatures[12] = 1
        elif xi[2] == 'c':
            currFeatures[13] = 1
        elif xi[2] == 'g':
            currFeatures[14] = 1
        elif xi[2] == 'r':
            currFeatures[15] = 1
        elif xi[2] == 'p':
            currFeatures[16] = 1
        elif xi[2] == 'u':
            currFeatures[17] = 1
        elif xi[2] == 'e':
            currFeatures[18] = 1
        else:
            currFeatures[19] = 1

        # Gill color
        if xi[8] == 'w':
            currFeatures[20] = 1
        elif xi[8] == 'k':
            currFeatures[21] = 1
        elif xi[8] == 'n':
            currFeatures[22] = 1
        elif xi[8] == 'b':
            currFeatures[23] = 1
        elif xi[8] == 'h':
            currFeatures[24] = 1
        elif xi[8] == 'g':
            currFeatures[25] = 1
        elif xi[8] == 'r':
            currFeatures[26] = 1
        elif xi[8] == 'o':
            currFeatures[27] = 1
        elif xi[8] == 'p':
            currFeatures[28] = 1
        elif xi[8] == 'u':
            currFeatures[29] = 1
        elif xi[8] == 'e':
            currFeatures[30] = 1
        else:
            currFeatures[31] = 1

        # Odor
        if xi[4] == 'n':
            currFeatures[32] = 1
        elif xi[4] == 'a':
            currFeatures[33] = 1
        elif xi[4] == 'l':
            currFeatures[34] = 1
        elif xi[4] == 'c':
            currFeatures[35] = 1
        elif xi[4] == 'y':
            currFeatures[36] = 1
        elif xi[4] == 'f':
            currFeatures[37] = 1
        elif xi[4] == 'm':
            currFeatures[38] = 1
        elif xi[4] == 'p':
            currFeatures[39] = 1
        else:
            currFeatures[40] = 1

        # Spore print
        if xi[19] == 'w':
            currFeatures[41] = 1

        # Gill size
        if xi[7] == 'b':
            currFeatures[42] = 1

        # Bruises
        if xi[3] == 't':
            currFeatures[43] = 1
        A.append(currFeatures)
    return np.asarray(A)

mushroomsDF = pd.read_csv('mushrooms.csv')
mushroomsDF = mushroomsDF.drop(columns=['class'])
mushroomsY = pd.read_csv('mushrooms.csv', usecols=[0])['class'].to_list()
mushroomsY = np.asarray([1 if y == 'e' else 0 for y in mushroomsY])
mushroomsMatrix = mushroomsDF.to_numpy()
mushroomFeatureMatrix = mushroomFeatures(mushroomsMatrix)
mushroomsTest = mushroomFeatureMatrix[round((len(mushroomFeatureMatrix) - 1) * 0.8):]
mushroomsFeatureMatrix = mushroomFeatureMatrix[:round((len(mushroomFeatureMatrix) - 1) * 0.8)]
mushroomsYTest = mushroomsY[round((len(mushroomsY) - 1) * 0.8):]
mushroomsY = mushroomsY[:round((len(mushroomsY) - 1) * 0.8)]
w = [0] * 44
b = 0
D = crossEntropyDeriv(w, mushroomFeatureMatrix[0], b, mushroomsY[0])
iter = 0
while iter < 5000:
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
truePos = 0
falsePos = 0
trueNeg = 0
falseNeg = 0
for i in range(len(predicted)):
    if predicted[i] == 1 and mushroomsYTest[i] == 1:
        truePos += 1
    elif predicted[i] == 1 and mushroomsYTest[i] == 0:
        falsePos += 1
    elif predicted[i] == 0 and mushroomsYTest[i] == 0:
        trueNeg += 1
    elif predicted[i] == 0 and mushroomsYTest[i] == 1:
        falseNeg += 1

posPrec = truePos / (truePos + falsePos)
negPrec = trueNeg / (trueNeg + falseNeg)
posRecall = truePos / (truePos + falseNeg)
negRecall = trueNeg / (trueNeg + falsePos)
print("Pos. precision: " + str(posPrec))
print("Neg. precision: " + str(negPrec))
print("Pos. recall: " + str(posRecall))
print("Neg. recall: " + str(negRecall))

irisDF = pd.read_csv('iris.csv').drop(columns=['class'])
irisX = np.asarray(irisDF.to_numpy())
irisY = pd.read_csv('iris.csv', usecols=[4])['class'].to_list()
irisY1 = np.asarray([0 if y == 'Iris-setosa' else 1 if y == 'Iris-versicolor' else 2 for y in irisY])
# Shuffle since dataset is ordered
zipped = list(zip(irisX, irisY1))
random.shuffle(zipped)
irisX, irisY1 = zip(*zipped)
irisX = np.asarray(irisX)
irisY1 = np.asarray(irisY1)
irisXTest = irisX[round((len(irisX) - 1) * 0.7):]
irisXTrain = irisX[:round((len(irisX) - 1) * 0.7)]
irisYTest1 = irisY1[round((len(irisY1) - 1) * 0.7):]
irisYTrain1 = irisY1[:round((len(irisY1) - 1) * 0.7)]
irisYTest = oneHotEncoded(irisYTest1, 3)
irisYTrain = oneHotEncoded(irisYTrain1, 3)

w = np.zeros((irisXTrain.shape[1], irisYTrain.shape[1]))
iter = 0
iterList = []
lossList = []
wList = []
while iter < 2500:
    iter += 1
    n = irisXTrain.shape[0]
    z = -np.dot(irisXTrain, w)
    p = softmax(z)
    w = w - 0.005 * (1/n * (np.dot(irisXTrain.T, (irisYTrain - p))))
    iterList.append(iter)
    wList.append(w)
    lossList.append(multinomialCost(irisXTrain, irisYTrain, w))

z = -np.dot(irisXTest, w)
p = softmax(z)
predicted = np.argmax(p, axis = 1)
print(predicted)
print(irisYTest1)
trueSetosa = 0
falseSetosa = 0
trueVersicolor = 0
falseVersicolor = 0
trueVirginica = 0
falseVirginica = 0

for i in range(len(predicted)):
    if predicted[i] == 0 and irisYTest1[i] == 0:
        trueSetosa += 1
    elif predicted[i] == 0 and irisYTest1[i] != 0:
        falseSetosa += 1
    elif predicted[i] == 1 and irisYTest1[i] == 1:
        trueVersicolor += 1
    elif predicted[i] == 1 and irisYTest1[i] != 1:
        falseVersicolor += 1
    elif predicted[i] == 2 and irisYTest1[i] == 2:
        trueVirginica += 1
    elif predicted[i] == 2 and irisYTest1[i] != 2:
        falseVirginica += 1

setosaPrec = trueSetosa / list(irisYTest1).count(0)
versicolorPrec = trueVersicolor / list(irisYTest1).count(1)
virginicaPrec = trueVirginica / list(irisYTest1).count(2)

print("Setosa precision:")
print(str(setosaPrec))
print("Versicolor precision:")
print(str(versicolorPrec))
print("Virginica precision:")
print(str(virginicaPrec))

