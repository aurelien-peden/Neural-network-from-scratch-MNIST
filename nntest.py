from sklearn.datasets import fetch_openml
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from neuralnetwork import NeuralNetwork

mnist = fetch_openml('mnist_784')
X, y = mnist["data"], mnist["target"]

X = X / 255

digits = 10
examples = y.shape[0]
y = y.reshape(1, examples)
Y_new = np.eye(digits)[y.astype('int32')]
Y_new = Y_new.T.reshape(digits, examples)

m = 60000
m_test = X.shape[0] - m
X_train, X_test = X[:m].T, X[m:].T
Y_train, Y_test = Y_new[:, :m], Y_new[:, m:]
shuffle_index = np.random.permutation(m)
X_train, Y_train = X_train[:, shuffle_index], Y_train[:, shuffle_index]

# Model
neural_network = NeuralNetwork(784, 64, digits, 4, m)
neural_network.train(X_train, Y_train, X_test, Y_test, 9)

