"""
Written by Zachary Pulliam

Used to create, train, and test NN's
"""

import os

from nn import NN
from datasets import MNISTDataset_0_1_train, MNISTDataset_0_1_test
from datasets import MNISTDataset_0_4_train, MNISTDataset_0_4_test


""" Variables for the user to change"""
ROOT = ''  # path to root folder
arch_0_1 = os.path.join(ROOT, 'architecture_0_1.txt')  # path to architecture file 0 to 1 classification
arch_0_4 = os.path.join(ROOT, 'architecture_0_1.txt')  # path to architecture file 0 to 4 classification 
epochs = 10  # number of epochs to train for

""" Choose between 0 to 1 or 0 to 4 classification for MNIST """
k = 1  # set k equal to 1 or 4


def run_0_1():
    train = MNISTDataset_0_1_train(os.path.join(ROOT, 'data'))  # training data
    test = MNISTDataset_0_1_test(os.path.join(ROOT, 'data'))  # testing data

    nn = NN(arch_0_1)  # create NN
    nn.train(train.x, train.y, epochs)  # train NN
    nn.test(test.x, test.y)  # test NN
    print('')


def run_0_4():
    train = MNISTDataset_0_4_train(os.path.join(ROOT, 'data'))  # training data
    test = MNISTDataset_0_4_test(os.path.join(ROOT, 'data'))  # testing data

    nn = NN(arch_0_4)  # create NN
    nn.train(train.x, train.y, epochs)  # train NN
    nn.test(test.x, test.y)  # test NN
    print('')

    
if __name__ == '__main__':
    if k == 1:
        run_0_1()
    elif k == 4:
        run_0_1()
    else:
        print('Set k equal to 1 or 4')
