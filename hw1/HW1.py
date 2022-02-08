# Name: Wilton Cappel
# COMP 347 - Machine Learning
# HW No. 1

import numpy             as np
import scipy.linalg      as LA
import time
import pandas            as pd
import matplotlib.pyplot as plt


# Problem 1 - Some Linear Algebraic Observations
#------------------------------------------------------------------------------
"""
 1a. -- Randomly initialize a matrix A that has fewer columns than rows
 (e.g. 20x5) and compute the eigenvalues of the matrices A^TA and AA^T.  What
 do you observe about the eigenvalues of the two matrices?  Compute the determinant
 of the smaller matrix and compare this to the product of all the eigenvalues.
 Write a comment explaining your observations.

 1b. -- For the smaller matrix above, find the eigenvalues of the inverse and
 compare these to the eigenvalues of the original matrix.  What is their
 relationship?  Demonstrate their relationship by writing some code below.  
 Write a comment explaining the relationship and how your code deomstrates this.

 1c. -- Initialize a random, square, non-symmetric matrix C.  Find the 
 eigenvalues of both C and its transpose.  What do you notice about the 
 eigenvalues for both matrices?  Show that they are the same by showing that
 the sum of their square differences amounts to a floating point error.  
 NOTE: you will likely need to sort both arrays.

 1d. -- Finding the eigenvalues and eigenvectors of a matrix is one example of
 a MATRIX DECOMPOSITION.  There are in fact many ways to decompose a matrix
 into a product of other matrices (the factors usually possessing some
 desirable property).  Explore the following matrix decompositions.  Write a 
 an explanation for what each one is doing and demonstrate the properties
 of the factors by writing code that demonstrates their structure.  Use the 
 internet to find formal and informal articles explaining the various 
 decompositions and cite your sources.  Write your ideas in a comment for 
 each one below:"""

# Initialize matrix D:  
    
# LU Factorization
# LA.lu(D)


# QR Factorization
# LA.qr(D)


# Singular Value Decomposition (SVD)
# LA.svd(D)

# 1a.
A = np.random.randint(-5, 5, size=(5, 4))
AT = np.transpose(A)
ATA = np.matmul(AT, A)
AAT = np.matmul(A, AT)
evATA = LA.eig(ATA)[0].real
print("ev of A^TA: ")
print(evATA)
evAAT = LA.eig(AAT)[0].real
print("ev of AA^T: ")
print(evAAT)
print("Dimensions of ATA: ")
print(ATA.shape)
print("Dimensions of AAT: ")
print(AAT.shape)
print("ATA is smaller.")
evProd = np.prod(evATA)
print("product of eigenvalues: ")
print(evProd)
ATAdet = LA.det(ATA)
print("det. of ATA: ")
print(ATAdet)
'''AA^T has one more eigenvalue than A^TA, but the other eigenvalues of each
are equal if you discount the complex numbers in A^TA.
The determinant of A^TA is equal to the product of all its eigenvalues.'''

#1b
ATAinv = LA.inv(ATA)
# print(ATAinv)
evATAinv = LA.eig(ATAinv)[0].real
print("original ev: ")
print(evATA)
print("inverse ev: ")
print(evATAinv)
print("inverse of inverse ev:")
check = []
for ev in evATAinv:
    check.append(float(1/ev.item()))
print(check)
'''My code takes the eigenvalues of the inverted matrix and inverts them.
The result is equal to the eigenvalues of the original matrix.
This displays that the eigenvalues of an inverse matrix are the inverse
of the original matrix's eigenvalues'''

#1c
C = 0
while 1:
    temp = np.random.randint(-5, 5, size=(3, 3))
    if not np.array_equal(temp, np.transpose(temp)):
        C = temp
        break

CT = np.transpose(C)
evC = LA.eig(C)[0].real
print("eigenvalues of C:")
print(evC)
evCT = LA.eig(CT)[0].real
print("eigenvalues of CT:")
print(evCT)
np.sort(evC)
np.sort(evCT)

def sumofsd(v, u):
    sum = 0
    for i, value in enumerate(v):
        sum += (v[i] - u[i]) ** 2
    return sum

print(sumofsd(evC, evCT))
'''
The eigenvalues for C and its transpose are equal. The sum of square 
differences be 0, since they are the same values, yet it is an
astronomically low number, approximate to 0. This clearly displays
that this was a floating point error due to the computer.
'''

#1d


# Problem 2 - Run Times and Efficiency
#------------------------------------------------------------------------------
"""It turns out that inverting a matrix using LA.solve 
 is actually quite a bit slower than using LA.inv.  How bad is it?  Find out 
 for yourself by creating histograms of the run times.  Compare the two matrix
 inversion methods for matrices of sizes 5, 10, 20, 50, and 200 with 1000 
 samples for each size.  Record the amount of time each trial takes in two 
 separate arrays and plot their histograms (with annotations and title).  You 
 can randomly initialize a matrix using any method you wish, but using np.random.rand
 is especially convenient.    

 Note: In plotting your histograms use the 'alpha' parameter to adjust the 
 color transparency so one can easily see the overlapping parts of your graphs.

 Some references: 
 https://scicomp.stackexchange.com/questions/22105/complexity-of-matrix-inversion-in-numpy
 https://docs.scipy.org/doc/scipy/reference/tutorial/linalg.html"""

