# Handwritten Digit Recognition using Deep Learning, Keras and Python
# Source code adapted from: https://gogul09.github.io/software/digits-recognition-mlp

# Imports
# Import numpy
import numpy as np
# Import matplotlib to show plots
import matplotlib.pyplot as plt
# Import Sequential model which is a linear stack of layers
from keras.models import Sequential
#Keras Imports to create the neural network model with neurons, layers and other utilities.
from keras.layers.core import Dense, Activation, Dropout
# Import MNIST datase
from keras.datasets import mnist
from keras.utils import np_utils

# Fixes a random seed for reproducibility
np.random.seed(9)

# User inputs
nb_epoch = 25 # Number of iterations needed for the network to minimize the loss function, so that it learns the weights.
num_classes = 10 # Total number of class labels or classes involved in the classification problem.
batch_size = 128 # Number of images given to the model at a particular instance.
train_size = 60000 # Number of training images to train the model.
test_size = 10000 # Number of testing images to test the model.
v_length = 784 # imension of flattened input image size i.e. if input image size is [28x28], then v_length = 784.

# Splits the MNIST data into train and test
(trainData, trainLabels), (testData, testLabels) = mnist.load_data()
print ("[INFO] train data shape: {}".format(trainData.shape))
print ("[INFO] test data shape: {}".format(testData.shape))
print( "[INFO] train samples: {}".format(trainData.shape[0]))
print ("[INFO] test samples: {}".format(testData.shape[0]))

# Reshapes the dataset
trainData = trainData.reshape(train_size, v_length)
testData = testData.reshape(test_size, v_length)
trainData = trainData.astype("float32")
testData = testData.astype("float32")
trainData /= 255
testData /= 255

print ("[INFO] train data shape: {}".format(trainData.shape))
print ("[INFO] test data shape: {}".format(testData.shape))
print ("[INFO] train samples: {}".format(trainData.shape[0]))
print ("[INFO] test samples: {}".format(testData.shape[0]))

# convert class vectors to binary class matrices --> one-hot encoding
mTrainLabels = np_utils.to_categorical(trainLabels, num_classes)
mTestLabels = np_utils.to_categorical(testLabels, num_classes)

# Creates the model
model = Sequential()
# Two hidden layers are used with 512 neurons in hidden layer 1 a
model.add(Dense(512, input_shape=(784,)))
model.add(Activation("relu"))# Activation function for hidden layers
model.add(Dropout(0.2))
model.add(Dense(256))#  256 neurons in hidden layer 2
model.add(Activation("relu"))
model.add(Dropout(0.2))
model.add(Dense(num_classes))
model.add(Activation("softmax"))#  Activation function for output layer.

# Summarizes the model
model.summary()

# Compiles the model
# categorical_crossentropy as the loss function, adam as the optimizer and accuracy as our performance metric.
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

# Fits the model
# Takes two arguments : trainin data and training labels
history = model.fit(trainData, mTrainLabels,validation_data=(testData, mTestLabels),batch_size=batch_size,nb_epoch=nb_epoch,verbose=2)

# Prints the history keys
print (history.history.keys())

# Evaluate the model and makes prediction
scores = model.evaluate(testData, mTestLabels, verbose=0)

# Using matplotlib we can visualize how our model reacts at different epochs on both training and testing data.
# History plot for accuracy
plt.plot(history.history["acc"])
plt.plot(history.history["val_acc"])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["train", "test"], loc="upper left")

# History plot for accuracy
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["train", "test"], loc="upper left")

# Print the results
print ("[INFO] test score - {}".format(scores[0]))
print ("[INFO] test accuracy - {}".format(scores[1]))

plt.show()

