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
"""

# 1a.
A = np.random.rand(5, 4)
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

"""
 1b. -- For the smaller matrix above, find the eigenvalues of the inverse and
 compare these to the eigenvalues of the original matrix.  What is their
 relationship?  Demonstrate their relationship by writing some code below.  
 Write a comment explaining the relationship and how your code deomstrates this.
"""

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

"""
 1c. -- Initialize a random, square, non-symmetric matrix C.  Find the 
 eigenvalues of both C and its transpose.  What do you notice about the 
 eigenvalues for both matrices?  Show that they are the same by showing that
 the sum of their square differences amounts to a floating point error.  
 NOTE: you will likely need to sort both arrays.
"""

#1c
C = 0
while 1:
    temp = np.random.rand(3, 3)
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
differences must be 0, since they are the same values, yet it is an
astronomically low number, approximate to 0. This clearly displays
that this was a floating point error due to the computer.
'''

"""
 1d. -- Finding the eigenvalues and eigenvectors of a matrix is one example of
 a MATRIX DECOMPOSITION.  There are in fact many ways to decompose a matrix
 into a product of other matrices (the factors usually possessing some
 desirable property).  Explore the following matrix decompositions.  Write a 
 an explanation for what each one is doing and demonstrate the properties
 of the factors by writing code that demonstrates their structure.  Use the 
 internet to find formal and informal articles explaining the various 
 decompositions and cite your sources.  Write your ideas in a comment for 
 each one below:
 """

#1d
# Initialize matrix D:  
D = np.random.rand(3, 3)

# LU Factorization
print("### LU ###")
LU = LA.lu(D)
lower = LU[1]
upper = LU[2]
print("Is U equal to its upper triangle?")
print(np.array_equal(np.triu(upper), upper))
print("Is L equal to its lower triangle?")
print(np.array_equal(np.tril(lower), lower))

''' The LU decomposition decomposes a matrix into two factors, one being a lower 
triangular matrix L (where all entries above the diagonal are 0), and an upper
triangular matrix U (where all entries below the diagonal are 0). U is found through 
Gaussian elimination, where the matrix is put into row echelon form. L is then either
found through algebra or by using the remaining elements as multiplier coefficients.
My code demonstrates that U from the decomposition is equal to its upper triangle, 
and that L is equal to its lower triangle, hence proving that the factors are
upper and lower triangular matrices.
Reference: https://www.geeksforgeeks.org/l-u-decomposition-system-linear-equations/
'''

# QR Factorization
print("### QR ###")
QR = LA.qr(D)
Q = QR[0]
R = QR[1]
QT = np.transpose(Q)
# rounding to decimal place 5 here due to FP errors
QTQ = np.around(np.matmul(QT, Q), 5)
print("Is QTQ equal to its identity matrix?")
print(np.array_equal(QTQ, np.identity(QTQ.shape[0])))
print("Is R equal to its upper triangle?")
print(np.array_equal(R, np.triu(R)))

'''The QR decomposition decomposes a matrix into two factors, one being an orthogonal
matrix Q, and the other being an upper triangular matrix R. Both of these factors can
be computed through the Gram-Schmidt process, which takes a set of linearly independent 
vectors and finds an orthonormal basis for them. My code demonstrates that Q from the
decomposition is an orthogonal matrix by multiplying its transpose with the matrix on
the right to show that it is equal to its identity matrix. My code also displays that
R is equal to its upper triangle.
Reference: https://www.math.ucla.edu/~yanovsky/Teaching/Math151B/handouts/GramSchmidt.pdf'''

# Singular Value Decomposition (SVD)
print("### SVD ###")
SVD = LA.svd(D)
U = SVD[0]
V = np.transpose(SVD[2])
sigma = LA.diagsvd(SVD[1], D.shape[0], D.shape[1])
DDT = np.matmul(D, np.transpose(D))
DTD = np.matmul(np.transpose(D), D)
print("Is U an orthogonal matrix?")
UTU = np.around(np.matmul(np.transpose(U), U), 5)
print(np.array_equal(UTU, np.identity(UTU.shape[0])))
print("Is V an orthogonal matrix?")
VTV = np.around(np.matmul(np.transpose(V), V), 5)
print(np.array_equal(VTV, np.identity(VTV.shape[0])))
print("Printing Σ displays it is a diagonal matrix")
print(sigma)

'''The SVD decomposes a matrix into three factors, U being an orthogonal matrix, V^T being 
the transpose of an orthogonal matrix, and Σ being a diagonal matrix w/ positive real
entries along its diagonal. This is done by computing the singular values, which can be
found through the eigenvalues of the matrix multiplied by its transpose on the right. Then
the orthonormal set of vectors of the matrix's transpose multiplied by the matrix are found.
My code demonstrates that U and V are orthogonal matrices, and that printing Σ displays it 
is a diagonal matrix.
Reference: https://www.cs.princeton.edu/courses/archive/spring12/cos598C/svdchapter.pdf
'''


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

# So make 5 diff. histograms based on each matrix size
# Every histogram will be made from the 2 arrays containing inv and solve runtimes
# Set bins to auto in .hist()

def runtimes(samplenum, shape):
    invResults = []
    solResults = []
    for i in range(samplenum):
        matrix = np.random.rand(shape, shape)
        b = np.identity(shape)
        invStart = time.time()
        LA.inv(matrix)
        invEnd = time.time()
        solStart = time.time()
        LA.solve(matrix, b)
        solEnd = time.time()
        invResults.append(invEnd - invStart)
        solResults.append(solEnd - solStart)
    return [invResults, solResults]

print("running samples...")
shape5 = runtimes(1000, 5)
inv5 = np.array(shape5[0])
sol5 = np.array(shape5[1])

shape10 = runtimes(1000, 10)
inv10 = np.array(shape10[0])
sol10 = np.array(shape10[1])

shape20 = runtimes(1000, 20)
inv20 = np.array(shape20[0])
sol20 = np.array(shape20[1])

shape50 = runtimes(1000, 50)
inv50 = np.array(shape50[0])
sol50 = np.array(shape50[1])

shape200 = runtimes(1000, 200)
inv200 = np.array(shape200[0])
sol200 = np.array(shape200[1])

# Runtimes for 5x5 matrix
plt.hist(inv5, label=".inv", alpha=0.7)
plt.hist(sol5, label=".solve", alpha=0.7)
plt.title("Runtimes for 5x5 Matrix")
plt.xlabel("Time (seconds)")
plt.ylabel("Samples")
plt.legend()
plt.show()

# Runtimes for 10x10 matrix
plt.hist(inv10, label=".inv", alpha=0.7)
plt.hist(sol10, label=".solve", alpha=0.7)
plt.title("Runtimes for 10x10 Matrix")
plt.xlabel("Time (seconds)")
plt.ylabel("Samples")
plt.legend()
plt.show()

# Runtimes for 20x20 matrix
plt.hist(inv20, label=".inv", alpha=0.7)
plt.hist(sol20, label=".solve", alpha=0.7)
plt.title("Runtimes for 20x20 Matrix")
plt.xlabel("Time (seconds)")
plt.ylabel("Samples")
plt.legend()
plt.show()

# Runtimes for 50x50 matrix
plt.hist(inv50, label=".inv", alpha=0.7)
plt.hist(sol50, label=".solve", alpha=0.7)
plt.title("Runtimes for 50x50 Matrix")
plt.xlabel("Time (seconds)")
plt.ylabel("Samples")
plt.legend()
plt.show()

# Runtimes for 200x200 matrix
plt.hist(inv200, label=".inv", alpha=0.7)
plt.hist(sol200, label=".solve", alpha=0.7)
plt.title("Runtimes for 200x200 Matrix")
plt.xlabel("Time (seconds)")
plt.ylabel("Samples")
plt.legend()
plt.show()
