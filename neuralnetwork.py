import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


class NeuralNetwork(object):
    def __init__(self, n_x, n_h, n_y, learning_rate, m):
        super().__init__()
        self.n_x = n_x
        self.n_h = n_h
        self.n_y = n_y

        self.m = m

        self.learning_rate = learning_rate
        self.beta = .9
        self.batch_size = 128
        self.batches = -(-self.m // self.batch_size)

        self.parameters = initialize_parameters(n_x, n_h, n_y)

    def train(self, X_train, Y_train, X_test, Y_test, epochs):
        V_dW1 = np.zeros(self.parameters["W1"].shape)
        V_db1 = np.zeros(self.parameters["b1"].shape)
        V_dW2 = np.zeros(self.parameters["W2"].shape)
        V_db2 = np.zeros(self.parameters["b2"].shape)

        for i in range(epochs):
            permutation = np.random.permutation(X_train.shape[1])
            X_train_shuffled = X_train[:, permutation]
            Y_train_shuffled = Y_train[:, permutation]

            for j in range(self.batches):
                begin = j * self.batch_size
                end = min(begin + self.batch_size, X_train.shape[1] - 1)
                X = X_train_shuffled[:, begin:end]
                Y = Y_train_shuffled[:, begin:end]
                m_batch = end - begin

                # forward and backward
                cache = feed_forward(X, self.parameters)
                grads = back_propagate(X, Y, self.parameters, cache, m_batch)

                # momentum
                V_dW1 = (self.beta * V_dW1 + (1. - self.beta) * grads['dW1'])
                V_db1 = (self.beta * V_db1 + (1. - self.beta) * grads["db1"])
                V_dW2 = (self.beta * V_dW2 + (1. - self.beta) * grads["dW2"])
                V_db2 = (self.beta * V_db2 + (1. - self.beta) * grads["db2"])

                # gradient descent update parameters
                self.parameters["W1"] = self.parameters["W1"] - \
                    self.learning_rate * V_dW1
                self.parameters["b1"] = self.parameters["b1"] - \
                    self.learning_rate * V_db1
                self.parameters["W2"] = self.parameters["W2"] - \
                    self.learning_rate * V_dW2
                self.parameters["b2"] = self.parameters["b2"] - \
                    self.learning_rate * V_db2

            cache = feed_forward(X_train, self.parameters)
            train_cost = compute_multiclass_loss(Y_train, cache["A2"])
            cache = feed_forward(X_test, self.parameters)
            test_cost = compute_multiclass_loss(Y_test, cache["A2"])
            print("Epoch {}: training cost = {}, test cost = {}".format(
                i+1, train_cost, test_cost))

        cache = feed_forward(X_test, self.parameters)
        predictions = np.argmax(cache["A2"], axis=0)
        labels = np.argmax(Y_test, axis=0)

        print(classification_report(predictions, labels))


def initialize_parameters(n_x, n_h, n_y):
    parameters = {}

    parameters['W1'] = np.random.randn(
        n_h, n_x) * np.sqrt(1. / n_x)  # or 2 / n_x
    parameters['b1'] = np.zeros((n_h, 1)) * np.sqrt(1. / n_x)

    parameters['W2'] = np.random.randn(n_y, n_h) * np.sqrt(1. / n_h)
    parameters['b2'] = np.zeros((n_y, 1)) * np.sqrt(1. / n_h)

    return parameters


def feed_forward(X, params):
    cache = {}

    cache['Z1'] = np.matmul(params['W1'], X) + params['b1']
    cache['A1'] = sigmoid(cache['Z1'])

    cache['Z2'] = np.matmul(params['W2'], cache['A1']) + params['b2']
    # A2 = softmax(Z2)
    cache['A2'] = np.exp(cache['Z2']) / np.sum(np.exp(cache['Z2']), axis=0)

    return cache


def back_propagate(X, Y, params, cache, m_batch):
    dZ2 = cache['A2'] - Y

    dW2 = (1. / m_batch) * np.matmul(dZ2, cache['A1'].T)
    db2 = (1. / m_batch) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.matmul(params['W2'].T, dZ2)
    dZ1 = dA1 * sigmoid(cache['Z1']) * (1 - sigmoid(cache['Z1']))

    dW1 = (1. / m_batch) * np.matmul(dZ1, X.T)
    db1 = (1. / m_batch) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

    return grads


def sigmoid(z):
    s = 1. / (1. + np.exp(-z))
    return s


def compute_multiclass_loss(Y, Y_hat):
    L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
    m = Y.shape[1]
    L = -(1./m) * L_sum

    return L
