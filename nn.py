"""
Written by Zachary Pulliam

Contains Neural Network class
"""

import math
import numpy as np



""" Neural Network class containing all required functions for training an testing """
class NN:
    def __init__(self, path):
        self.weights = []  # list of weight arryas between each layer
        self.biases = []  # list of bias arryas between each layer
        self.alpha = 0.5  # learning rate
        self.create_architecture(path)


    """ Creates the weights and biases arrays for each layer based on architecture.txt"""
    def create_architecture(self, path):
        layers = []
        with open(path, "r") as f:
                lines = f.readlines()

                for l in lines:
                    det = l.split(" ")
                    for i, s in enumerate(det):
                        det[i] = s.rstrip()
                    layers.append(int(det[1]))

        for i in range(len(layers)):
            if i < len(layers)-1:
                self.weights.append(np.random.uniform(-1,1,[layers[i],layers[i+1]]))
                self.biases.append(np.random.uniform(-1,1,[layers[i+1]]))


    """ Sigmoid activation function; used during forward pass """
    def sigmoid(self, x):
        return 1/(1+math.pow(math.e, -x))


    """ Derivative of sigmoid activation function; used to calculate deltas from activations """
    def d_sigmoid(self, x):
        return x*(1-x)


    """ Forward pass; calculates outputs and activations of each layer """
    def forward(self, input):
        activations = []  # stores activations of nodes 
        H = np.asarray(list(map(self.sigmoid, (np.dot(self.weights[0].T, input) + self.biases[0]))))  # first hidden layer's activations
        activations.append(H)  # append hidden layer 1's activations
        for i in range(len(self.weights)-1):
            H = np.asarray(list(map(self.sigmoid, (np.dot(self.weights[i+1].T, H) + self.biases[i+1]))))  # remaining hidden layer's activations, loop to output
            activations.append(H)  # append layer n's activations
        return H, activations  # output and activations


    """ Backpropigation; calculates deltas at each layer starting from output"""
    def backprop(self, error, output, activations):
        ### delta_output = Error * g'(in)
        deltas = [error * np.asarray(list(map(self.d_sigmoid, output)))]  # delta at output layer
        for i in range(len(self.weights)-2, -1, -1):
            ### delta_hidden = g'(in) * sum( W * delta )
            deltas.append((np.asarray(list(map(self.d_sigmoid, activations[i]))).reshape(-1,1) * np.dot(self.weights[i+1], deltas[-1].reshape(-1,1))))  # backprop deltas for each layer

        deltas.reverse()  # flip the deltas list to correspond with direction of weights
        return deltas


    """ Updates weights and biases based on deltas"""
    def update(self, deltas):
        for i in range(len(self.weights)):
            ### w = w + a*delta
            self.weights[i] = self.weights[i] + 0.001*deltas[i].T
            ### b = b + a*delta
            self.biases[i] = self.biases[i] + 0.001*deltas[i].flatten()


    """ Model prediction from input """
    def predict(self, input):
        H = np.asarray(list(map(self.sigmoid, (np.dot(self.weights[0].T, input) + self.biases[0]))))  # first hidden layer's activations
        for i in range(len(self.weights)-1):
            H = np.asarray(list(map(self.sigmoid, (np.dot(self.weights[i+1].T, H) + self.biases[i+1]))))  # remaining hidden layer's activations, loop to output
        return np.argmax(H)


    """ Calculates model error"""
    def calc_error(self, x, y):
        correct, total = 0, 0
        for i, input in enumerate(x):
            prediction = self.predict(input)
            if prediction == np.argmax(y[i]):
                correct+=1
            total+=1
        print('     Training Accuracy:', round(correct/total, 3))


    """ Training of NN; prediction -> error -> deltas -> update """
    def train(self, x, y, epochs):
        print('\nBefore training...')
        self.calc_error(x, y)
        print('------------------------------')

        for e in range(epochs):
            print('Epoch:', e+1, 'of', epochs)
            for i, input in enumerate(x):
                output, activations = self.forward(input)  # calculate prediction and activations, forward pass
                error = y[i] - output  # calculate error

                deltas = self.backprop(error, output, activations)  # calculate deltas
                self.update(deltas)  # update weights
            
            self.calc_error(x, y)
        print('------------------------------')


    """ Testing of NN """
    def test(self, x, y):
        correct, total = 0, 0
        for i, input in enumerate(x):
            prediction = self.predict(input)
            if prediction == np.argmax(y[i]):
                correct+=1
            total+=1
        print('Testing Accuracy:', round(correct/total, 3))