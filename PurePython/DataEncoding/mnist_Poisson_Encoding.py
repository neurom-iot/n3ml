import MNIST_Deserialization  
import numpy as np

# ------------------------------------------------------------------------------
# Convert Image to Binary Sequence
# ------------------------------------------------------------------------------


def binary_Sequence_Conversion(dataType, nBins):
    """
    For given a firing-rate of the real neuron, the probability of a number of spikes
    that occur in a time period is distributed looks like the Poisson distribution

    This function is used to convert the Images to the BinarySequence (0,1) with Poisson Process

    Params
    -----------
    dataType: 0 for trainng data and 1 for testing data
    nBins : time of stimulation

    :return:
    a list of Binary Sequence of Images dataset
    """
    
    #Specify the location that contains the training.pickle and testing.pickle file
    MNIST_path = '\\'
    bs_Dataset = []     # Binary Sequence of all Images in Dataset
    bs_Image = []       # Binary Sequence of each Image

    dt = 0.001

    if dataType == 0:
        dataset = MNIST_Deserialization.mnist_Deserialization(MNIST_path + 'training', 1)
    elif dataType == 1:
        dataset = MNIST_Deserialization.mnist_Deserialization(MNIST_path + 'testing', 0)
    else:
        raise Exception('data type should be 0 for training or 1 for testing data')

    for i in range(0, len(dataset['x'])):
        indImage = np.array(dataset['x'][i])
        flatten_Image = indImage.flatten()
        for pixel in flatten_Image:
            bs_Pixel = []       # # Binary Sequence of each pixel
            if pixel == 0:
                bs_Pixel = [0] * nBins
                bs_Image.append(bs_Pixel)
            else:
                for k in range(0, nBins):
                    uniformNum = np.random.uniform(0, 1)
                    if (pixel * dt) > uniformNum:
                        bs_Pixel.append(1)
                    else:
                        bs_Pixel.append(0)
                bs_Image.append(bs_Pixel)
        bs_Dataset.append(bs_Image)
        print('Finish the ' + str(i) + ' image')


if __name__ == '__main__':

     # Example for Converting the Training Data to a Binary Sequence
     bsTrainingData = binary_Sequence_Conversion(0, 30)   # 0 for Training data, 30 is the stimulation time

     # Example for Converting the Testing Data to a Binary Sequence
     bsTestingData = binary_Sequence_Conversion(1, 30)  # 1 for Testing data, 30 is the stimulation time


