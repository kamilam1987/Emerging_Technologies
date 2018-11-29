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
# Import Image for operation on image(save,open, etc)
from PIL import Image

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
# Output train data shape
print ("Train data shape: {}".format(trainData.shape))
# Output test data shape
print ("Test data shape: {}".format(testData.shape))
# Output train samples
print( "Train samples: {}".format(trainData.shape[0]))
# Output train samples test samples
print ("Test samples: {}".format(testData.shape[0]))

# Reshapes the dataset
trainData = trainData.reshape(train_size, v_length) # Reshapes the train data
testData = testData.reshape(test_size, v_length) # Reshapes the test data
trainData = trainData.astype("float32") # For train data change the pixel intensities to float32
testData = testData.astype("float32") # For test data change the pixel intensities to float32
trainData /= 255 # grayscale image pixel intensities are integers in the range [0-255]
testData /= 255

# Output reshaped train data 
print ("Train data shape: {}".format(trainData.shape))
# Output reshaped test data  
print ("Test data shape: {}".format(testData.shape))
# Output reshaped train samples  
print ("Train samples: {}".format(trainData.shape[0]))
# Output reshaped test samples
print ("Test samples: {}".format(testData.shape[0]))

# convert class vectors to binary class matrices --> one-hot encoding
mTrainLabels = np_utils.to_categorical(trainLabels, num_classes)
mTestLabels = np_utils.to_categorical(testLabels, num_classes)

# Creates the model
model = Sequential()
# Two hidden layers are used with 512 neurons in hidden layer 1 a
model.add(Dense(512, input_shape=(784,)))
model.add(Activation("relu")) # Activation function for hidden layers
model.add(Dropout(0.2)) # 20% is used as is a weight constraint on those layers
model.add(Dense(256)) #  256 neurons in hidden layer 2
model.add(Activation("relu"))  # Activation function for hidden layers
model.add(Dropout(0.2))# 20% is used as is a weight constraint on those layers
model.add(Dense(num_classes))
model.add(Activation("softmax")) #  Activation function for output layer.

# Summarizes the model
model.summary()

# Compiles the model
# categorical_crossentropy as the loss function, adam as the optimizer and accuracy as our performance metric.
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

# Fits the model
# Takes two arguments : trainin data and training labels
history = model.fit(trainData, mTrainLabels,validation_data=(testData, mTestLabels),batch_size=batch_size,nb_epoch=nb_epoch,verbose=2)

# List all data in history
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
# Shows plots
plt.show()

# History plot for accuracy
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["train", "test"], loc="upper left")
# Shows plots
plt.show()

# Print the results
print ("Test score - {}".format(scores[0]))
print ("Test accuracy - {}".format(scores[1]))

# Takes some test images from the test data
test_imgs= testData[5:9]

# Reshape the test images to standard 28x28 format
test_imgs = test_imgs.reshape(test_imgs.shape[0], 28, 28)
# Output prediction for image
print( "Test images shape - {}".format(test_imgs.shape))

# Loop over each of the test images
for i, test_img in enumerate(test_imgs, start=1):
    # Takes a copy of test image for viewing
    org_image = test_img
    # For model to understand, have to reshape the test image to 1x784 format 
    test_img = test_img.reshape(1,784)
    # Make prediction on test image using our trained model
    prediction = model.predict_classes(test_img, verbose=0)
    # Output the prediction and image
    print ("My prediction for image is - {}".format(prediction[0]))
    plt.subplot(220+i)
    plt.imshow(org_image, cmap=plt.get_cmap('gray'))

# Shows plots
plt.show()

# Test my own file with digit
# Open image with PIL
img1 = Image.open("img/test_number.png") 
# Resize image to 28x28 pixels
img1 = img1.resize((28, 28), Image.ANTIALIAS)
# Saves resized image
img1.save("img/resized_number.png")

img = np.invert(Image.open("img/resized_number2.png").convert('1'))
img = img.reshape(1,784)
score_test = model.predict(img, batch_size=1, verbose=0)
# Gets prediction
prediction_new = model.predict_classes(img, verbose=0)
#display the prediction and image
print ("I think your digit is - {}".format(prediction_new[0]))


