import numpy as np
import os.path
from struct import unpack
import pickle


# specify the location of the MNIST data
MNIST_path = '\\'

def mnist_Deserialization(picklename, dataType):
    """This function is used to Read image and label data
       This return a list of tuples.
       The "pickle" module is used for implementing binary protocols for serializing and de-serializing
       a Python object structure.
    
    :param
        data Type: 1 for Training data and 0 for Testing data
        picklename: MNIST data path

    """

    # Open the images with binary mode
    if dataType==0:
        images = open(MNIST_path + 'train-images.idx3-ubyte', 'rb')
        labels = open(MNIST_path + 'train-labels.idx1-ubyte', 'rb')
    elif dataType==1:
        images = open(MNIST_path + 't10k-images.idx3-ubyte', 'rb')
        labels = open(MNIST_path + 't10k-labels.idx1-ubyte', 'rb')

    # Get metadata for images
    images.read(4)
    n_images = unpack('>I', images.read(4))[0]  # n_images is the total number of images
    rows = unpack('>I', images.read(4))[0]
    cols = unpack('>I', images.read(4))[0]

    # Get metadata for labels
    labels.read(4)
    n_labels = unpack('>I', labels.read(4))[0] # n_labels is the total number of labels

    # Check whether the number of the labels match the number of images or not
    if n_images != n_labels:
        raise Exception('number of labels did not match the number of images')

    # Get the data
    x = np.zeros((n_images, rows, cols), dtype=np.uint8)  # Initialize array of Images
    y = np.zeros((n_images, 1), dtype=np.uint8)  # Initialize array of Labels
    for i in range(n_images):
        if i % 1000 == 0:
            print("i: %i" % i)
        x[i] = [[unpack('>B', images.read(1))[0] for col in range(cols)] for row in range(rows)]
        y[i] = unpack('>B', labels.read(1))[0]
    data = {'x': x, 'y': y, 'rows': rows, 'cols': cols}  # Dictionary
    pickle.dump(data, open("%s.pickle" % picklename, "wb"))
    return data

if __name__ == '__main__':

     # Example for Geting Training Data
     training_Data = mnist_Deserialization (MNIST_path + 'training', 0)

     # Example for Get Testing Data
     testing_Data = mnist_Deserialization(MNIST_path + 'testing', 1)
