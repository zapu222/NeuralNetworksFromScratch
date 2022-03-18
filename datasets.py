"""
Written by Zachary Pulliam

Contains 4 dataset classes training and tetsing for both MNIST classification 0-1 and 0-4...
Other dataset classes can be added here.
"""

import os
import numpy as np
import pandas as pd



""" MNIST training subset 0 to 1 from .csv file """
class MNISTDataset_0_1_train:
    def __init__(self, path):
        filename = os.path.join(path, 'mnist_train_0_1.csv')
        df = pd.read_csv(filename, header=None)

        x_df = df.drop(df.columns[0], axis=1)/255
        y_df = df.drop(df.columns[1:785], axis=1)
        
        self.x = []  # list to contain array of pixel values for each image
        self.y = []  # list to contain array of one-hot encoded outputs

        for _, row in x_df.iterrows():
            self.x.append(np.array(row, dtype=float))

        for _, row in y_df.iterrows():
            if row[0] == 0:
                self.y.append(np.array([1,0]))
            else:
                self.y.append(np.array([0,1]))



""" MNIST testing subset 0 to 1 from .csv file """
class MNISTDataset_0_1_test:
    def __init__(self, path):
        filename = os.path.join(path, 'mnist_test_0_1.csv')
        df = pd.read_csv(filename, header=None)
        
        x_df = df.drop(df.columns[0], axis=1)/255
        y_df = df.drop(df.columns[1:785], axis=1)
        
        self.x = []  # list to contain array of pixel values for each image
        self.y = []  # list to contain array of one-hot encoded outputs

        for _, row in x_df.iterrows():
            self.x.append(np.array(row, dtype=float))

        for _, row in y_df.iterrows():
            if row[0] == 0:
                self.y.append(np.array([1,0]))
            else:
                self.y.append(np.array([0,1]))



""" MNIST training subset 0 to 4 from .csv file """
class MNISTDataset_0_4_train:
    def __init__(self, path):
        filename = os.path.join(path, 'mnist_train_0_4.csv')
        df = pd.read_csv(filename, header=None)
        
        x_df = df.drop(df.columns[0], axis=1)/255
        y_df = df.drop(df.columns[1:785], axis=1)
        
        self.x = []  # list to contain array of pixel values for each image
        self.y = []  # list to contain array of one-hot encoded outputs

        for _, row in x_df.iterrows():
            self.x.append(np.array(row, dtype=float))

        for _, row in y_df.iterrows():
            if row[0] == 0:
                self.y.append(np.array([1,0,0,0,0]))
            elif row[0] == 1:
                self.y.append(np.array([0,1,0,0,0]))
            elif row[0] == 2:
                self.y.append(np.array([0,0,1,0,0]))
            elif row[0] == 3:
                self.y.append(np.array([0,0,0,1,0]))
            else:
                self.y.append(np.array([0,0,0,0,1]))



""" MNIST testing subset 0 to 4 from .csv file """
class MNISTDataset_0_4_test:
    def __init__(self, path):
        filename = os.path.join(path, 'mnist_test_0_4.csv')
        df = pd.read_csv(filename, header=None)
        
        x_df = df.drop(df.columns[0], axis=1)/255
        y_df = df.drop(df.columns[1:785], axis=1)
        
        self.x = []  # list to contain array of pixel values for each image
        self.y = []  # list to contain array of one-hot encoded outputs

        for _, row in x_df.iterrows():
            self.x.append(np.array(row, dtype=float))

        for _, row in y_df.iterrows():
            if row[0] == 0:
                self.y.append(np.array([1,0,0,0,0]))
            elif row[0] == 1:
                self.y.append(np.array([0,1,0,0,0]))
            elif row[0] == 2:
                self.y.append(np.array([0,0,1,0,0]))
            elif row[0] == 3:
                self.y.append(np.array([0,0,0,1,0]))
            else:
                self.y.append(np.array([0,0,0,0,1]))