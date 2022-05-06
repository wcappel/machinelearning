# Name:
# COMP 347 - Machine Learning
# HW No. 5

#*********************
# PROBLEMS ARE POSTED BELOW THE FUNCTIONS SECTION!!!
#*********************

# Libraries
#------------------------------------------------------------------------------
import numpy as np
import scipy.linalg as LA
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import matplotlib.image  as mpimg
import pandas as pd
import matplotlib.pyplot as plt


# Functions
#------------------------------------------------------------------------------
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    Returns
    -------
    matplotlib.patches.Ellipse

    Other parameters
    ----------------
    kwargs : `~matplotlib.patches.Patch` properties
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs)

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# Euclidean edge distances
def dist_matrix(Data):
    """Returns the Euclidean distance matrix dists."""
    D = []
    for i in range(len(Data)):
        currRow = []
        for j in range(len(Data)):
            currRow.append(LA.norm(Data[i] - Data[j]) ** 2)
        D.append(currRow)
    return np.asarray(D)

# Multidimensional scaling alg
def classical_mds(Dists, dim):
    """Takes the distance matrix Dists and dimensionality dim of the desired
        output and returns the classical MDS compression of the data."""
    H = np.identity(len(Dists)) - (1/np.size(Dists) * (np.matmul(np.ones(len(Dists)), np.transpose(np.ones(len(Dists))))))
    XXT = -0.5 * np.matmul(H, np.matmul(Dists, H))
    w, P = LA.eig(XXT)
    D = LA.diagsvd(w, XXT.shape[0], XXT.shape[1])
    Ystar = np.matmul(P, np.sqrt(D))
    return Ystar[:, 0: dim]

# Construct edge matrix based on KNN or epsilon ball
def edge_matrix(Dists,eps):
    """Returns the epsilon-ball edge matrix for a data set.  In particular, the
       edge matrix should be NxN with binary entries: 0 if the two points are not
       neighbors, and 1 if they are neighbors."""
    edgeMatrix = []
    for i, dist in enumerate(Dists):
        currRow = []
        for j, otherDist in enumerate(Dists[i]):
            if i == j:
                currRow.append(0)
            elif otherDist <= eps:
                currRow.append(1)
            else:
                currRow.append(0)
        edgeMatrix.append(currRow)
    return edgeMatrix

# Construct ISOMAP graph by replacing Euclidean distances with graph distances
def isomap(dists, edges, dim):
    """Returns the Isomap compression of a data set based on the Euclidean distance
       matrix dists, the edge matrix edges, and the desired dimensionality dim of
       the output.  This should specifically output two variables: it should output
       the data compression, as well as the indices of points removed from the
       Floyd-Warshall algorithm."""
    edgeMatrix = edge_matrix(dists, edges)
    for k in range(len(dim)):
        for i, edge, in enumerate(edgeMatrix):
            for j, otherEdge, in enumerate(edgeMatrix[i]):
                edgeMatrix[i][j] = min(edgeMatrix[i][j], edgeMatrix[i][k] + edgeMatrix[k][j])
                print(min(edgeMatrix[i][j], edgeMatrix[i][k] + edgeMatrix[k][j]))
    return classical_mds(np.asarray(edgeMatrix), dim)


# Problem #1 - PCA of Data and SVD Compression
#------------------------------------------------------------------------------

#   1a. Perform principle component analysis on both the random data and sinusoidal
#       data sets.  For each data set plot the corresponding covariance ellipses
#       along with the principle vectors which should be the principle axes of
#       of the ellipses.  Use the confidence_ellipse function given above using values
#       of n_std = 1,2 as the standard deviations.  Some notes: the length of the
#       of the principle axes are given by the standard deviations of the univariate
#       data, hence the square-root of the top-left and bottom-right entries in
#       the covariance matrix.  Additionally, these vectors are orthogonal but will
#       only appear perpendicular if the image is plotted with equal axes; they may
#       appear skew otherwise.  Annotate the images appropriately.
# randDF = pd.read_csv("rand_data.csv", header=None)
# randX = randDF[0].to_numpy()
# randY = randDF[1].to_numpy()
# fig, ax = plt.subplots()
# ax.scatter(randX, randY)
# confidence_ellipse(randX, randY, ax, edgecolor='red', n_std=1)
# confidence_ellipse(randX, randY, ax, edgecolor='red', n_std=2)
# plt.show()
#
# sinDF = pd.read_csv("sin_data.csv", header=None)
# sinX = sinDF[0].to_numpy()
# sinY = sinDF[1].to_numpy()
# fig, ax = plt.subplots()
# ax.scatter(sinX, sinY)
# confidence_ellipse(sinX, sinY, ax, edgecolor='red', n_std=1)
# confidence_ellipse(sinX, sinY, ax, edgecolor='red', n_std=2)
# plt.show()

#   1b. Pick your favorite image from your photo library (something school-appropriate,
#       nothing graphic please!) and convert it to gray-scale.  Use the singular value
#       decomposition to produce a sequence of reconstructions by adding back each s.v.
#       back one at a time.  In other words, if the original decomposition is given by
#       USV.T, then for each reconstruction simply replace S with a matrix S_i where
#       i represents the number of sv's added back to the matrix.  In doing this, construct
#       a curve that displays the reconstruction accuracy as a function of the number
#       of singular values included in the reconstruction.  The reconstruction
#       accuracy can be computed as 1 - (LA.norm(Recon - Orig) / LA.norm(Orig)).
#       Annotate your plot with a legend that displays the number of singular values
#       needed to obtain 80%, 85%, 90%, 95%, and 99% accuracy.  Create a graphic that
#       that shows these five reconstructions along with the original.
img = mpimg.imread("cookies.jpeg")
imgU, imgSig, imgV = LA.svd(img)
acc = []
recons = []
accFound = 0
for i in range(len(imgSig)):
    currS = np.diag(imgSig[:i])
    decomp = np.matmul(np.matmul(np.matrix(imgU[:, :i]), np.diag(imgSig[:i])), np.matrix(imgV[:i, :]))
    currAcc = 1 - (LA.norm(decomp - img) / LA.norm(img))
    acc.append(currAcc)
    if currAcc >= 0.8 and accFound < 80:
        recons.append(decomp)
        accFound = 80
    elif currAcc >= 0.85 and accFound < 85:
        recons.append(decomp)
        accFound = 85
    elif currAcc >= 0.9 and accFound < 90:
        recons.append(decomp)
        accFound = 90
    elif currAcc >= 0.95 and accFound < 95:
        recons.append(decomp)
        accFound = 95
    elif currAcc >= 0.99 and accFound < 99:
        recons.append(decomp)
        accFound = 99
recons.append(img)
plt.figure()
f, axarr = plt.subplots(3,2)
axarr[0][0].imshow(recons[0])
axarr[0][1].imshow(recons[1])
axarr[1][0].imshow(recons[2])
axarr[1][1].imshow(recons[3])
axarr[2][0].imshow(recons[4])
axarr[2][1].imshow(recons[5])
plt.show()

count = [*range(0, len(imgSig))]
plt.plot(count, acc, label="Reconstruction accuracy per SV added")
plt.legend()
plt.show()
# Problem #2 - MDS of Breast Cancer Data and SVM Modeling with the Hinge Formalism
#------------------------------------------------------------------------------

#   2a. Complete the function dist_matrix above where the output should be the
#       NxN Euclidean distance matrix of a data set Data.

#   2b. Complete the function classical_mds that produces the multidimensional
#       scaling compression of a data set.  An important note here is that you
#       are NOT allowed to simply plug in the data set; this has to take the
#       Euclidean distance matrix as input.  The reason for this is that the
#       original algorithm was conceived of as a blind reconstruction: if only
#       given the mutual distances between points could one recover the original
#       data set?  Make sure to follow the derivation as given in the slides to
#       obtain this.

#   2c. Perform the MDS compression of the breast cancer data from UCI repository
#       (i.e. the same data set we've seen in class).  Reproduce the image from
#       the slides and show that there is a natural separation between the benign
#       malignant cases.  Make sure to color/label your data to reflect this.
bcDF = pd.read_csv("breast_cancer.csv")
bcDiags = bcDF['diagnosis'].to_list()
bcDF = bcDF.iloc[:, 2:-3]
bcX = np.asarray(bcDF.values.tolist())
bcY = []
for diag in bcDiags:
    if diag == 'B':
        bcY.append(-1)
    else:
        bcY.append(1)
bcD = dist_matrix(bcX)
bcMDS = classical_mds(bcD, 2)
bcBlue = []
bcRed = []
for i in range(len(bcY)):
    if bcY[i] == -1:
        bcBlue.append(bcMDS[i])
    else:
        bcRed.append(bcMDS[i])
bcBlue = np.asarray(bcBlue)
bcRed = np.asarray(bcRed)
plt.scatter(bcBlue[:, 0], bcBlue[:, 1])
plt.scatter(bcRed[:, 0], bcRed[:, 1])
plt.show()

#   2d. Using your code from HW #4, perform SVM on the new MDS compression of
#       the breast cancer data.  Specifically perform a 10-fold cross validation
#       using the hinge formulation of SVM and plot a bar plot showing the success
#       percentages of the learned models as well as that of the averaged model.
#       Use backtracking line search to aid in finding optimal stepsizes in the
#       gradient descent.  Use a stopping criterion of comparing the relative sizes
#       of successive derivatives of the form
#
#       while 0.99 < LA.norm(D_0)/LA.norm(D_1) < 1.01:
def hinge_loss(X,y,w,b):
    result = 0
    halfNormWSquared = (LA.norm(w) ** 2) / 2
    for i, xi in enumerate(X):
        val = max(0, 1 - (y[i] * (np.dot(w, xi) + b)))
        result += halfNormWSquared + val
    return result / np.size(X)

def hinge_deriv(X,y,w,b):
    wUpdate = [0, 0]
    bUpdate = 0
    for i, xi in enumerate(X):
        if y[i] * ((np.matmul(np.transpose(w), xi)) + b) >= 1:
            wUpdate += w
            bUpdate += 0
        else:
            wUpdate += w - (y[i] * xi)
            bUpdate += y[i] * -1
    return wUpdate / np.size(X), bUpdate / np.size(X)

bcY = np.asarray(bcY)
w = [-10, 10]
b = 0
eps = 0.1
K = 0.01
currDeriv = hinge_deriv(bcMDS, bcY, w, b)
D_0 = [currDeriv[0][0], currDeriv[0][1], currDeriv[1]]
D_1 = [currDeriv[0][0], currDeriv[0][1], currDeriv[1]]
iter = 0
while iter == 0 or (0.99 < LA.norm(D_0) / LA.norm(D_1) < 1.01 and iter < 10000):
    iter += 1
    currDeriv = hinge_deriv(bcMDS, bcY, w, b)
    eps = 1
    m = LA.norm(D_1)**2
    t = 0.5*m
    while hinge_loss(bcMDS, bcY, w - eps * D_1, 1) > hinge_loss(bcMDS, bcY, w, 1) - eps * t:
        eps *= 0.9
    w = w - eps * currDeriv[0]
    b = b - eps * currDeriv[1]
    D_0 = D_1
    D_1 = [currDeriv[0][0], currDeriv[0][1], currDeriv[1]]

# Problem #3 - Isomap of Breast Cancer Data
#------------------------------------------------------------------------------

#   3a. Complete the function edge_matrix which for a given data set produces an
#       NxN matrix of zeros and ones, 0 meaning two points are not nearest neighbors
#       and 1 indicating they are.  Have this matrix constructed using the epsilon-ball
#       method of nearest neighbors where eps is an input variable to this function.

#   3b. Complete the function isomap which produces the isomap compression of a
#       data set given the original Euclidean distance matrix, the desired edge
#       matrix, and the desired number of output dimensions.  The function should
#       output two variables: the compressed data, and a list of indices representing
#       the data points removed as a result of the Floyd-Warshall algorithm.  I
#       highly recommend putting in a print command somewhere within the Floyd-Warshall
#       part of the isomap algorithm since this has cubic complexity and often is the
#       longest part of the procedure.

#   3c. Produce a sequence of isomap embeddings of the breast cancer data using
#       a value of epsilon ranging from 100 to 500 and incrementing in steps of
#       25.  As in the slides, make sure to annotate your images to show the
#       separation between benign and malignant cases.  This part of your homework
#       will take some time to execute because of Floyd-Warshall.  Go grab a coffee
#       while you're waiting for it to terminate and submit your receipt along
#       with your code as proof.  lol, jk on this last part.
bcD = dist_matrix(bcX)
for i in range(16):
    bcIso = isomap(bcD, 100 + ((i + 1) * 25), 2)
    bcBlue = []
    bcRed = []
    for i in range(len(bcY)):
        if bcY[i] == -1:
            bcBlue.append(bcIso[i])
        else:
            bcRed.append(bcIso[i])
    bcBlue = np.asarray(bcBlue)
    bcRed = np.asarray(bcRed)
    plt.scatter(bcBlue[:, 0], bcBlue[:, 1])
    plt.scatter(bcRed[:, 0], bcRed[:, 1])
    plt.show()
