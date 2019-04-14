import numpy as np
import cv2 as cv
import os
import re
from PCA import PCA
from LDA import LDA

usePCA = False
useLDA = True
useFaceRecognition = False

def display_image(image):
    """
    Displys an image and waits for a key press from the user.
    :param image:
        The image to display as a numpy array.
    :return:
        Nothing.
    """

    cv.imshow(cv.namedWindow("image", cv.WINDOW_AUTOSIZE), image)
    cv.waitKey(0)
    return

def readRandomDataset(maxImageCount = 100):
    """
    Reads the random dataset in oeder to use it to classifiy faaces vs non faces.
    :param maxImageCount:
        the maximum number of images to load.
    :return:
        - A numpy array of the shape (400, 10304) which 
          has the data of the images.
        - A numpy array that contains the labels of each image.
    """
    
    datasetDirctory = "dataset/rand_images/"
    labelIndex = 2
    imageCount = 0

    # list the images of the random dataset.
    imageNames = os.listdir(datasetDirctory)
    data = None
    labels = []

    for imageName in imageNames:
        # add correct label.
        labels.append(labelIndex)

        # read the image and flatten it.
        imagePath = datasetDirctory + imageName
        image = np.array(cv.imread(imagePath, -1))
        image = image.flatten()

        # if first image create new array.
        if data is None:
            data = image
        # else stack the image to the bottom of the current data.
        else:
            data = np.vstack((data, image))
            
        imageCount = imageCount + 1

        # if the image count reaches the maximum count break.
        if imageCount == maxImageCount:
            break
    
    data = np.array(data)
    labels = np.array(labels)

    return splitData(data, labels, 5)

def readORLDataset(maxImageCount = 400, faceRecognition = True):
    """
    Reads the orl_faces dataset and splits it into training and testing parts.
    :param maxImageCount:
        Optional, the maximum number of images to load.
    :param faceRecognition:
        Optional, a boolean that indicated the reason for reading the dataset,
        if it is True then the labels are different for each face.
        if it is False then labels are all the same for all faces.
    :return:
        - A numpy array of the shape (400, 10304) which 
          has the data of the images.
        - A numpy array that contains the labels of each image.
    """

    datasetDirectory = "dataset/orl_faces/"
    imageCount = 0
    
    if faceRecognition:
        labelIndex = 0
    else:
        labelIndex = 1
    
    # get a list of directories in the dataset.
    directories = os.listdir(datasetDirectory)
    directories.sort(key = natural_keys)

    # create an empty list for the labels and a reference to the 
    # data array which will be used to hold the image data.
    labels = []
    data = None

    # for each subject directory.
    for directory in directories:
        # 9ncrease label index to match current subject.
        if faceRecognition:
            labelIndex = labelIndex + 1

        # get a list of all image names in the directory.
        subjectPath = datasetDirectory + directory
        imagesNames = os.listdir(subjectPath)
        imagesNames.sort(key = natural_keys)
        
        # for each image in the current subject.
        for imageName in imagesNames:
            # add correct label.
            labels.append(labelIndex)

            # read the image and flatten it.
            imagePath = subjectPath + "/" +  imageName
            image = np.array(cv.imread(imagePath, -1))
            image = image.flatten()

            # if first image create new array.
            if data is None:
                data = image
            # else stack the image to the bottom of the current data.
            else:
                data = np.vstack((data, image))
            
            imageCount = imageCount + 1

            # if the image count reaches the maximum count break.
            if imageCount == maxImageCount:
                break
        
        if imageCount == maxImageCount:
                break

    data = np.array(data)
    labels = np.array(labels)

    return splitData(data, labels, 5)

def splitData(data, labels, trainingSplitCount=5):
    """
    Splits the data and labels into a training split and a testing split.
    :param data:
        A numpy array, it contains image data. 
        It should have a shape of (400, 10304).
    :param labels:
        A numpy array, it contains the labels of the dataset.
        It should have a lenght of 400.
    :param trainingSplitCount:
        the number of training images per class, the other class images
        are going to be used for testing.
    :return:
        4 numpy arrays, the first 2 arrys contain the training data and the corresponding labels, 
        the next 2 arrays contain the testing data and the corresponding labels.
    """

    trainingData = []
    trainingLabels = []
    testingData = []
    testingLabels = []

    # calculate the round count depending on the number of samples.
    trainingSplit = trainingSplitCount
    samplesCount = data.shape[0]
    roundCount = int(samplesCount / 10)

    dataSplit = data
    labelsSplit = labels

    # loop over the data and labels and split them according to the
    # training split.
    for i in range(0, roundCount):
        dataSplit = np.split(dataSplit, [trainingSplit, 10], axis=0)
        trainingData.append(dataSplit[0])
        testingData.append(dataSplit[1])
        dataSplit = dataSplit[2]
        labelsSplit = np.split(labelsSplit, [trainingSplit, 10], axis=0)
        trainingLabels.append(labelsSplit[0])
        testingLabels.append(labelsSplit[1])
        labelsSplit = labelsSplit[2]

    trainingData = np.concatenate(trainingData)
    testingData = np.concatenate(testingData)
    trainingLabels = np.concatenate(trainingLabels)
    testingLabels = np.concatenate(testingLabels)
    return trainingData, trainingLabels, testingData, testingLabels

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

# if doing face recognition read the orl dataset only..
if useFaceRecognition:
    trainingData, trainingLabels, testingData, testingLabels = readORLDataset()  
# else read 100 images only from the orl dataset and the random dataset.
else:
    # read datasets.
    orlTrainingData, orlTrainingLabels, orlTestingData, orlTestingLabels = readORLDataset(maxImageCount=100, faceRecognition=False)
    randomTrainingData, randomTrainingLabels, randomTestingData, randomTestingLabels = readRandomDataset()

    # stack the datasets togather.
    trainingData = np.vstack((orlTrainingData, randomTrainingData))
    trainingLabels = np.hstack((orlTrainingLabels, randomTrainingLabels))
    testingData = np.vstack((orlTestingData, randomTestingData))
    testingLabels = np.hstack((orlTestingLabels, randomTestingLabels))

if usePCA:
    # initialize PCA and train and test.
    pca = PCA()
    pca.train(trainingData, trainingLabels)
    pca.test(testingData, testingLabels)

if useLDA:
    # initialize LDA and train and test.
    lda = LDA()
    lda.train(trainingData, trainingLabels)
    lda.test(testingData, testingLabels)