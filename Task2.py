import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


class ShowPlots:
    def __init__(self, feature1, feature2, feature3, feature4):
        self.x1 = feature1
        self.x2 = feature2
        self.x3 = feature3
        self.x4 = feature4

    def X1X2Compination(self):
        plt.figure("X1fig1")
        plt.scatter(x1[:50], x2[:50])
        plt.scatter(x1[50:100], x2[50:100])
        plt.scatter(x1[100:150], x2[100:150])
        plt.ylabel("X2")
        plt.xlabel("X1")
        # plt.show()

    def X1X3Compination(self):
        plt.figure("X2fig2")
        plt.scatter(x1[:50], x3[:50])
        plt.scatter(x1[50:100], x3[50:100])
        plt.scatter(x1[100:150], x3[100:150])
        plt.ylabel("X3")
        plt.xlabel("X1")

    def X1X4Compination(self):
        plt.figure("X1fig3")
        plt.scatter(x1[:50], x4[:50])
        plt.scatter(x1[50:100], x4[50:100])
        plt.scatter(x1[100:150], x4[100:150])
        plt.ylabel("X4")
        plt.xlabel("X1")
        # plt.show()

    def X2X3Compination(self):
        plt.figure("X2fig1")
        plt.scatter(x2[:50], x3[:50])
        plt.scatter(x2[50:100], x3[50:100])
        plt.scatter(x2[100:150], x3[100:150])
        plt.ylabel("X3")
        plt.xlabel("X2")

    def X2X4Compination(self):
        plt.figure("X2fig2")
        plt.scatter(x2[:50], x4[:50])
        plt.scatter(x2[50:100], x4[50:100])
        plt.scatter(x2[100:150], x4[100:150])
        plt.ylabel("X4")
        plt.xlabel("X2")

    def X3X4Compination(self):
        plt.figure("X3fig1")
        plt.scatter(x3[:50], x4[:50])
        plt.scatter(x3[50:100], x4[50:100])
        plt.scatter(x3[100:150], x4[100:150])
        plt.ylabel("X4")
        plt.xlabel("X3")

    def showAllPlots(self):
        self.X1X2Compination()
        self.X1X3Compination()
        self.X1X4Compination()
        self.X2X3Compination()
        self.X2X4Compination()
        self.X3X4Compination()
        plt.show()


class PrepareData:
    def __init__(self, x1, x2, x3, x4, Y):
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4
        self.output = Y

    def addTrainTestData(self, firstArray, secondArray):
        train = np.array(firstArray[:30])
        train = np.append(train, values=secondArray[:30], axis=0)
        test = np.array(firstArray[30:])
        test = np.append(test, values=secondArray[30:], axis=0)
        return train, test

    def checkClassChoice(self, choice1, choice2, data):
        first = np.array(data[:50])
        second = np.array(data[50:100])
        third = np.array(data[100:150])

        np.random.shuffle(first)
        np.random.shuffle(second)
        np.random.shuffle(third)

        if (choice1 == "C1" and choice2 == "C2") or (choice1 == "C2" and choice2 == "C1"):
            return self.addTrainTestData(firstArray=first, secondArray=second)
        elif (choice1 == "C1" and choice2 == "C3") or (choice1 == "C3" and choice2 == "C1"):
            return self.addTrainTestData(firstArray=first, secondArray=third)
        elif (choice1 == "C2" and choice2 == "C3") or (choice1 == "C3" and choice2 == "C2"):
            return self.addTrainTestData(firstArray=second, secondArray=third)

    def featureSelection(self, choice1, choice2):
        if choice1 == "x1" and choice2 == "x2":
            data = pd.DataFrame({'X1': x1, 'X2': x2, 'Class': self.output})
        elif choice1 == "x1" and choice2 == "x3":
            data = pd.DataFrame({'X1': x1, 'X3': x3, 'Class': self.output})
        elif choice1 == "x1" and choice2 == "x4":
            data = pd.DataFrame({'X1': x1, 'X4': x4, 'Class': self.output})
        elif choice1 == "x2" and choice2 == "x3":
            data = pd.DataFrame({'X2': x2, 'X3': x3, 'Class': self.output})
        elif choice1 == "x2" and choice2 == "x4":
            data = pd.DataFrame({'X2': x2, 'X4': x4, 'Class': self.output})
        elif choice1 == "x3" and choice2 == "x4":
            data = pd.DataFrame({'X3': x3, 'X4': x4, 'Class': self.output})
        return data


class AdaLine:
    def __init__(self, learning_rate=0.01, numberOfIteration=100, isBios=False, threshold=0.0):
        self.learning_rate = learning_rate
        self.numberOfIteration = numberOfIteration
        self.weights = 0
        self.threshold = threshold
        self.bias = np.random.uniform(0.01, 0.1)
        self.isBios = isBios

    def fit(self, X, y):
        self.weights = np.random.uniform(0.01, 0.1, size=X.shape[1])
        self.bias = np.random.uniform(0.01, 0.1)
        for index in range(self.numberOfIteration):
            errors = 0
            for i, x_i in enumerate(X):
                netValue = np.dot(x_i, self.weights)
                if self.isBios:
                    y_predicted = netValue + self.bias
                    error = (y[i] - y_predicted)
                    self.weights += self.learning_rate * error * x_i
                    self.bias += self.learning_rate * error
                else:
                    y_predicted = netValue
                    error = (y[i] - y_predicted)
                    self.weights += self.learning_rate * error * x_i
            for i, x_i in enumerate(X):
                if self.isBios:
                    y_predicted = np.dot(x_i, self.weights) + self.bias
                    error = (y[i] - y_predicted)
                else:
                    y_predicted = np.dot(x_i, self.weights)
                    error = (y[i] - y_predicted)
                errors += error ** 2
            mse = (errors * 0.5) / len(y)
            if mse < self.threshold:
                break
            else:
                continue

        self.plot_LinearPerception(inputs=X, weights=self.weights, bias=self.bias)

    def predict(self, X):
        netValue = np.dot(X, self.weights)
        if self.isBios:
            realOutput = netValue + self.bias
        else:
            realOutput = netValue

        y_predicted = self.activationFunction(realOutput)
        return y_predicted

    def activationFunction(self, x):
        return np.where(x >= 0, 1, -1)

    def plot_LinearPerception(self, inputs, weights, bias):
        plt.figure(figsize=(7, 7))
        plt.grid(True)

        for i in np.linspace(np.amin(inputs[:, :1]), np.amax(inputs[:, :1])):
            slope = -(weights[0] / bias) / (weights[0] / weights[1])
            intercept = -weights[0] / bias
            y = (slope * i) + intercept
            plt.plot(i, y, 'ro')
        plt.show()

    def accuracy_score(self, y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy


dataframe = pd.read_csv("IrisData.txt")
classes = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': -1}
dataframe['Class'] = [classes[item] for item in dataframe['Class']]

x1 = dataframe["X1"]
x2 = dataframe["X2"]
x3 = dataframe["X3"]
x4 = dataframe["X4"]


def Call_back(features, classes, learning_rate, epochs, mse, isBias):
    prepare = PrepareData(x1=x1, x2=x2, x3=x3, x4=x4,
                          Y=dataframe["Class"])
    Feature_choices = features.split('&')
    feature1 = str(Feature_choices[0].replace(" ", ""))
    feature2 = str(Feature_choices[1].replace(" ", ""))
    d = prepare.featureSelection(choice1=feature1, choice2=feature2)

    Class_choices = classes.split('&')
    class1 = str(Class_choices[0].replace(" ", ""))
    class2 = str(Class_choices[1].replace(" ", ""))

    trainData, testData = prepare.checkClassChoice(choice1=class1, choice2=class2, data=d)

    # convert float to integers
    trainData[:, 2].astype(int)
    testData[:, 2].astype(int)

    X_train = trainData[:, :2]
    y_train = trainData[:, 2]
    X_test = testData[:, :2]
    y_test = testData[:, 2]

    showPlots = ShowPlots(feature1=x1, feature2=x2, feature3=x3, feature4=x4)
    showPlots.showAllPlots()

    p = AdaLine(learning_rate=float(learning_rate), numberOfIteration=int(epochs), threshold=float(mse), isBios=isBias)
    p.fit(X_train, y_train)

    Test_predictions = p.predict(X_test)
    Train_predictions1 = p.predict(X_train)

    print(confusion_matrix(y_test, Test_predictions))

    return p.accuracy_score(y_test, Test_predictions), p.accuracy_score(y_train, Train_predictions1)
