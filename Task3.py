import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


class PrepareData:
    def __init__(self, x1, x2, x3, x4, Y):
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4
        self.output = Y

    @staticmethod
    def addTrainTestData(data):
        firstArray = np.array(data[:50])
        secondArray = np.array(data[50:100])
        thirdArray = np.array(data[100:150])

        np.random.shuffle(firstArray)
        np.random.shuffle(secondArray)
        np.random.shuffle(thirdArray)

        train = np.array(firstArray[:30])
        train = np.append(train, values=secondArray[:30], axis=0)
        train = np.append(train, values=thirdArray[:30], axis=0)

        test = np.array(firstArray[30:])
        test = np.append(test, values=secondArray[30:], axis=0)
        test = np.append(test, values=thirdArray[30:], axis=0)
        return train, test


class Layer:

    def __init__(self, n_neurons, activation=None, weights=None, isbias=False):
        self.n_neurons = n_neurons
        self.weights = weights
        self.activation = activation
        self.net_values = None
        self.isBias = isbias
        self.bias = np.random.uniform(0.01, 0.1)
        self.error = None
        self.delta = None

    def activate(self, x, w):
        NetValue = np.dot(x, w)
        if self.isBias:
            NetValue = NetValue + self.bias
        net_value = self.activation_function(NetValue)
        return net_value

    def activation_function(self, netvalue):
        net = 0
        if self.activation == 'Sigmoid':
            net = 1 / (1 + np.exp(-netvalue))

        if self.activation == 'Hyperbolic Tangent sigmoid':
            net = np.tanh(netvalue)

        return net

    def activation_derivative(self, netvalue):
        delta = 0
        if self.activation == 'Sigmoid':
            delta = netvalue * (1 - netvalue)

        if self.activation == 'Hyperbolic Tangent sigmoid':
            delta = 1.0 - netvalue ** 2

        return delta


class BackPropagation:

    def __init__(self, num_of_hidden_layers=0, neurons=None, activation=None, learning_rate=0.01, numberOfIteration=100,
                 isbias=False):
        self.learning_rate = learning_rate
        self.numberOfIteration = numberOfIteration
        self.isBias = isbias
        self.num_inputs = 4
        self.num_outputs = 3
        self.num_of_hidden_layers = num_of_hidden_layers
        self.neurons = neurons
        self.activation = activation
        self.hidden_layers = []
        self.output_layer = Layer(n_neurons=self.num_outputs)

    def initialize_network(self):
        inputs = self.num_inputs
        neurons_list = self.neurons.split(',')
        for i in range(len(neurons_list)):
            neurons_list[i] = int(neurons_list[i])

        for i in range(self.num_of_hidden_layers):
            weight = np.random.uniform(0.01, 0.1, size=(inputs, neurons_list[i]))
            inputs = neurons_list[i]
            self.hidden_layers.append(
                Layer(n_neurons=neurons_list[i], activation=self.activation, weights=weight, isbias=self.isBias))

        weight = np.random.uniform(0.01, 0.1, size=(inputs, self.num_outputs))
        self.output_layer = Layer(n_neurons=self.num_outputs, activation=self.activation,
                                  weights=weight, isbias=self.isBias)

    def forward_propagate(self, inputs):
        for layer in self.hidden_layers:
            net_value = layer.activate(inputs, layer.weights)
            layer.net_values = net_value
            inputs = net_value
        net_output_value = self.output_layer.activate(inputs, self.output_layer.weights)
        self.output_layer.net_values = net_output_value
        return self.output_layer.net_values

    def predict(self, x):
        outputs = self.forward_propagate(x)
        for i in range(len(outputs)):
            for j in range(len(outputs[i])):
                maximum = np.max(outputs[i])
                if outputs[i][j] == maximum:
                    outputs[i][j] = 1
                else:
                    outputs[i][j] = 0
        return outputs

    def backpropagation(self, x, y):
        output = self.forward_propagate(x)
        for i in reversed(range(len(self.hidden_layers) + 1)):
            if i == len(self.hidden_layers):
                layer = self.output_layer
                layer.error = y - output
                layer.delta = layer.error * layer.activation_derivative(output)
            else:
                layer = self.hidden_layers[i]
                if i == len(self.hidden_layers) - 1:
                    next_layer = self.output_layer
                else:
                    next_layer = self.hidden_layers[i + 1]
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                layer.delta = layer.error * layer.activation_derivative(layer.net_values)

    def update_weights(self, x):
        for i in range(len(self.hidden_layers) + 1):
            if i == len(self.hidden_layers):
                layer = self.output_layer
                inputs = self.hidden_layers[i - 1].net_values
            else:
                if i == 0:
                    inputs = x
                else:
                    inputs = self.hidden_layers[i - 1].net_values
                layer = self.hidden_layers[i]
            layer.weights += self.learning_rate * inputs[:, np.newaxis] * layer.delta

    def train(self, x, y):
        for epoch in range(self.numberOfIteration):
            for i, x_i in enumerate(x):
                self.backpropagation(x_i, y[i])
                self.update_weights(x_i)

    @staticmethod
    def convert_from_binary(y):
        outputs = []
        for i in range(len(y)):
            if y[i][0] == 1:
                outputs.append(1)
            elif y[i][1] == 1:
                outputs.append(2)
            else:
                outputs.append(3)
        return outputs

    @staticmethod
    def accuracy_score(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy


dataframe = pd.read_csv("IrisData.txt")

classes = {'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3}
dataframe['Class'] = [classes[item] for item in dataframe['Class']]

x1 = dataframe["X1"]
x2 = dataframe["X2"]
x3 = dataframe["X3"]
x4 = dataframe["X4"]
Y = dataframe['Class']


def Call_back(layers, neurons, l_rate, epochs, activation, isbias):
    prepare = PrepareData(x1=x1, x2=x2, x3=x3, x4=x4,
                          Y=Y)
    Data = pd.DataFrame({'X1': prepare.x1, 'X2': prepare.x2, 'X3': prepare.x3,
                         'X4': prepare.x4, 'Class': prepare.output})
    trainData, testData = prepare.addTrainTestData(data=Data)

    # convert float to integers
    trainData[:, 4].astype(int)
    testData[:, 4].astype(int)

    X_train = trainData[:, :4]
    y_train = trainData[:, 4]
    X_test = testData[:, :4]
    y_test = testData[:, 4]

    backpropagation = BackPropagation(num_of_hidden_layers=layers, neurons=neurons, learning_rate=l_rate,
                                      numberOfIteration=epochs, activation=activation, isbias=isbias)
    backpropagation.initialize_network()
    backpropagation.train(X_train, y_train)

    predictions = backpropagation.predict(X_test)
    predictions1 = backpropagation.predict(X_train)

    p = backpropagation.convert_from_binary(predictions)
    p1 = backpropagation.convert_from_binary(predictions1)

    print(confusion_matrix(y_test, p))

    return backpropagation.accuracy_score(y_test, p) * 100, backpropagation.accuracy_score(y_train, p1) * 100
