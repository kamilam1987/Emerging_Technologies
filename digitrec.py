# Logistic Regression MNIST
# Source code adapted from: https://www.codementor.io/mgalarny/making-your-first-machine-learning-classifier-in-scikit-learn-python-db7d7iqdh

# Import MNIST from sklearn.datasets
from sklearn.datasets import fetch_mldata
# Import train_test_split for traning and spliting MNIST data
from sklearn.model_selection import train_test_split
# Import numpy
import numpy as np
# Import matplotlib to show an images
import matplotlib.pyplot as plt

# Load the MNIST data
mnist = fetch_mldata('MNIST original')

# There are 70,000 images (28 by 28 images for a dimensionality of 784)
# Shows the amount of images in MNIST dataset
print(mnist.data.shape)

# There are 70,000 labels 
# Shows the amount of labels in MNIST dataset
print(mnist.target.shape)

# Splits the data into training and test data sets the test_size=1/7.0, training set size 60,000 images and the test set size of 10,000
train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target, test_size=1/7.0, random_state=0)

# Determine figure size
plt.figure(figsize=(20,4))

# Loop through images and labels, shows the first five train images and the first five train labels
for index, (image, label) in enumerate(zip(train_img[0:5], train_lbl[0:5])):
    plt.subplot(1, 5, index + 1)
    plt.imshow(np.reshape(image, (28,28)), cmap='gray')
    plt.title('Training: %i\n' % label, fontsize = 20)

    # Shows plots
    plt.show()