# Paul David
# Machine Learning Course
# Neural Network Class Definition with Tutorials

import numpy as np
# keras is just used to retrieve dataset
from tensorflow import keras
import time


class neural_network:
    # To incorporate other cost functs., change w_deriv and b_deriv
    """Constructs a neural network with the following:
        dim_list: a list of the sizes of each successive layer.
        X_train: input training data
        Y_train: output training data
        X_test: input testing data
        Y_test: output testing data
        sig: chosen sigmoid function to be used ubiquitously throughout the network.
            expected values are strings, either 'tanh' or 'logit'.

        Optimization will be encoded in via a least-squares cost."""

    def __init__(self, dim_list, X_train, Y_train, X_test, Y_test, sig, cost):
        # dim list is list of size of layers
        # sig will indicate which sigmoid funct. to use
        # either hyperbolic tangent or logistic funct. (can fill out on own)
        # if input data is of dimension n, and dim list is [4, 5, 6, 10]
        # then w0 will be 4 by n, w1 will be 5 by 4, w2 will be 6 by 5
        # and w3 will be 10 by 6
        self.dims = dim_list
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.sigmoid = sig
        self.weights = [np.random.rand(self.dims[0], X_train.shape[1])] + [
            np.random.rand(self.dims[i + 1], self.dims[i]) for i in
            range(len(self.dims) - 1)]  # will be a list of randomly initialized weight matrices.
        self.biases = [np.random.rand(self.dims[i]) for i in
                       range(len(self.dims))]  # will be a list of randomly initialized bias vectors.
        self.activations = []  # activations will be produced via a method
        self.w_derivs = []  # list of w-derivatives to be produced via a method
        self.b_derivs = []  # list of b-derivatives to be produced via a method
        self.cost = cost

    def trans_func(self, w, x, b):
        z = np.matmul(w, x) + b
        if self.sigmoid == 'tanh':
            return np.tanh(z)
        if self.sigmoid == 'logit':
            return 1 / (1 + np.exp(-z))

    def trans_deriv(self, w, x, b):
        z = np.matmul(w, x) + b
        if self.sigmoid == 'tanh':
            return 1 - (np.tanh(z)) ** 2
        if self.sigmoid == 'logit':
            return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

    def find_activations(self, inds):
        """Will be a list of lists.  The overarching list will be indexed by the points in X,
            and each sublist will give the activations for each data point in each layer of
            the network."""
        self.activations = []
        L = len(inds)
        for i in range(L):
            # print("Finding activations for data point " + str(i+1) + " of " + str(L))
            sub_active = []
            a = self.X_train[inds[i]]
            for j in range(len(self.dims)):
                sub_active += [self.trans_func(self.weights[j], a, self.biases[j])]
                a = sub_active[-1]
            self.activations += [sub_active]

    ### BACKPROPOGATION ####
    def b_subderiv(self, j, n, i):
        """Recursively finds the derivative of the bias vector of the nth layer.
            n: the layer on which we wish to find the derivative of the biases
            j: a dummy index used for the recursive procedure of updatig the biases
            i: the index of the data point for which we are finding the activation."""
        if j == n:
            if n == 0:
                a = self.X_train[i]
            else:
                a = self.activations[i][n - 1]
            subderiv = np.diag(self.trans_deriv(self.weights[n], a, self.biases[n]))
        else:
            sig_d = self.trans_deriv(self.weights[j], self.activations[i][j - 1], self.biases[j])
            subderiv = np.multiply(np.matmul(self.weights[j], self.b_subderiv(j - 1, n, i)), sig_d[:, np.newaxis])
        return subderiv

    def b_deriv(self, n, inds):
        """Finds the derivative of the bias at the nth layer."""
        L = len(inds)
        if self.cost == 'LS':
            diff_vecs = [self.Y_train[inds[i]] - self.activations[i][-1] for i in range(L)]
            mults = [np.matmul(self.b_subderiv(len(self.dims) - 1, n, i).T, diff_vecs[i]) for i in range(L)]
            return sum(mults) / L
        elif self.cost == 'CE':
            diffs = [self.Y_train[inds[i]] * (1 / self.activations[i][-1]) + (self.Y_train[inds[i]] - 1) * (
                        1 / (1 - self.activations[i][-1])) for i in range(L)]
            mults = [np.matmul(np.multiply(self.b_subderiv(len(self.dims) - 1, n, i), diffs[i][:, np.newaxis]).T,
                               np.ones(self.dims[-1])) for i in range(L)]
            return -sum(mults) / L

    def w_subderiv(self, j, n, i):
        """Recursively finds the derivative of the weight matrix of the nth layer.
            n: the layer on which we wish to find the derivative of the biases
            j: a dummy index used for the recursive procedure of updatig the biases
            i: the index of the data point for which we are finding the activation."""
        if j == n:
            if n == 0:
                a = self.X_train[i]
                k = self.X_train.shape[1]
            else:
                a = self.activations[i][n - 1]
                k = self.dims[n - 1]
            sig_d = self.trans_deriv(self.weights[n], a, self.biases[n])
            sig_block = np.diag(np.tile(sig_d, k))
            subderiv = np.matmul(np.kron(a, np.eye(len(sig_d))), sig_block)
        else:
            sig_d = self.trans_deriv(self.weights[j], self.activations[i][j - 1], self.biases[j])
            subderiv = np.multiply(np.matmul(self.weights[j], self.w_subderiv(j - 1, n, i)), sig_d[:, np.newaxis])
        return subderiv

    def w_deriv(self, n, inds):
        """Finds the derivative of the weight matrix at the nth layer."""
        L = len(inds)
        if self.cost == 'LS':
            diff_vecs = [self.Y_train[inds[i]] - self.activations[i][-1] for i in range(L)]
            mults = [np.matmul(self.w_subderiv(len(self.dims) - 1, n, i).T, diff_vecs[i]) for i in range(L)]
            pre = sum(mults) / L
            if n == 0:
                deriv = np.reshape(pre, [self.dims[n], self.X_train.shape[1]])
            else:
                deriv = np.reshape(pre, [self.dims[n], self.dims[n - 1]])
            return deriv
        elif self.cost == 'CE':
            diffs = [self.Y_train[inds[i]] * (1 / self.activations[i][-1]) + (self.Y_train[i] - 1) * (
                        1 / (1 - self.activations[i][-1])) for i in range(L)]
            mults = [np.matmul(np.multiply(self.w_subderiv(len(self.dims) - 1, n, i), diffs[i][:, np.newaxis]).T,
                               np.ones(self.dims[-1])) for i in range(L)]
            pre = -sum(mults) / L
            if n == 0:
                deriv = np.reshape(pre, [self.dims[n], self.X_train.shape[1]])
            else:
                deriv = np.reshape(pre, [self.dims[n], self.dims[n - 1]])
            return deriv

    # Train the Network
    def network_train(self, eps, max_iter, batch_size):
        """Performs a fixed-rate gradient descent of the network.
        eps: fixed stepsize
        max_iter: max number of iterations to perform updates on the weights and biases."""
        start = time.time()
        for i in range(max_iter):
            N = self.X_train.shape[0]
            inds = np.random.choice([i for i in range(N)], batch_size, replace=False)

            # print("Finding activations...")
            self.find_activations(inds)

            print("Epoch " + str(i + 1) + " of " + str(max_iter) + ", time elapsed: " + str(
                np.around((time.time() - start) / 60, 2)) + " minutes")
            for j in range(len(self.dims)):
                self.weights[j] -= eps * self.w_deriv(j, inds)
                self.biases[j] -= eps * self.b_deriv(j, inds)

    # Testing and Performance Summary
    def network_testing(self):
        """Should only be performed after training."""

        L = self.X_test.shape[0]
        count = 0
        labels = []
        for i in range(L):
            # print("Finding activations for data point " + str(i+1) + " of " + str(L))
            activations = []
            a = self.X_test[i]
            for j in range(len(self.dims)):
                # print(j)
                activations = self.trans_func(self.weights[j], a, self.biases[j])
                a = activations

            # activations should now just be the last layer.  Now classify
            labels += [np.where(activations == np.max(activations))[0][0]]
            true_label = np.where(self.Y_test[i] == 1)[0][0]
            if labels[-1] == true_label:
                count += 1

        success = count / L
        print("Training yielded " + str(np.around(100 * success, 3)) + "% on classification.")
        return labels


# Example Training with MNIST Digits
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess data to interpret as matrices:
N = X_train.shape[0];
X_new = np.zeros([N, 28 ** 2]);
Y_new = np.zeros([N, 10])
L = X_test.shape[0];
X_newTest = np.zeros([L, 28 ** 2]);
Y_newTest = np.zeros([L, 10])
for i in range(N):
    X_new[i, :] = np.reshape(X_train[i, :, :], [1, 28 ** 2])
    Y_new[i, y_train[i]] = 1
for j in range(L):
    X_newTest[j, :] = np.reshape(X_test[j, :, :], [1, 28 ** 2])
    Y_newTest[j, y_test[j]] = 1

mnist_NN = neural_network([20, 15, 10], X_new, Y_new, X_newTest, Y_newTest, 'tanh', 'CE')
mnist_NN.network_train(0.1, 200, 25)
y_labels = mnist_NN.network_testing()

