# Python, Neural Networks from Scratch - Zachary Pulliam

This code was written in order to implement flexible neural networks from scratch that will be used to predict output 
for two subsets of the MNIST dataset, the first containing only 0's and 1's and the second containing 0's to 4's.

The NN class can be used on any other dataset as long as the inputs and outputs are formatted correctly. Inputs are formatted as a list of arrays;
an array for each images containing the flattened pixel values for that image. Outputs are formatted as a list of arrays, each array a one-hot encoded array
of length *n* depending upon the number of classes (2 classes for 0-1 and 5 classes for 0-4). A NN model can be trained and tested in four steps...

1. Simply create a dataset class similar to those in datasets.py
2. Create the NN architecture similar to the examples provided in the txt files; keep in mind shape of inputs and outputs must correspond to the data
    - Hidden layers can be added by simply adding a new line to the txt file in the format: 'Hidden_n: num'
    - The input layer size should be equal to the number of pixel values in the image and the output layer size should be equal to the number of classes
4. Initialize the NN class
5. Train the NN using NN.train()
6. Test the NN using NN.test()

A full example as well as metrics and visualized predictions can be found in ex.ipynb.
