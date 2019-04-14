import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import distance

class PCA:
    def __init__(self):
        """
        initialized the PCA Class.
        """

        self.eignValues = None
        self.eignVectors = None
        self.projectionMatrix = None
        self.projectedTrainingData = None
        self.pca = None
        self.alpha = 0.95
        self.trainingLabels = None
        self.useSklearn = False

    # Project the training data.
    def train(self, trainingData, trainingLabels):
        """
        Used to calculate the projection matrix and then
        transform the training data.
        :param trainingData:
            The training data as a (number of observations, number of features) numpy array.
        :param trainingLabels:
            The training labels as a numpy array.
        :return:
            Nothing.
        """

        # Substract mean
        mean = np.mean(trainingData, axis=0)
        trainingData = trainingData - mean
        print("the shape of the mean is: " + str(trainingData.shape))

        # Covariance matrix
        covariance = np.cov(trainingData.T)
        print("the shape of the covariance matrix is: " + str(covariance.shape))

        # calculate eignvalues and eignvectors
        print("calculating eignvector and eignvalues.")
        if self.isSymmetric(covariance):
            self.eignValues, self.eignVectors = np.linalg.eigh(covariance)
        else:
            self.eignValues, self.eignVectors = np.linalg.eig(covariance)

        indices = np.argsort(self.eignValues)
        indices = indices[::-1]
        
        self.eignVectors = self.eignVectors[:,indices]
        self.eignValues = self.eignValues[indices]

        print("the shape of the values matrix is: " + str(self.eignValues.shape))
        print("the shape of the vectors matrix is: " + str(self.eignVectors.shape))

        dimensions = self.calculateDimensionCount(self.alpha)
        print("the number of dimensions is: " + str(dimensions))

        # Extract only the top most eigenvectors
        self.projectionMatrix = self.eignVectors[:,:dimensions]
        print("the shape of the projection matrix is: " + str(trainingData.shape))

        # set the projected training data
        if self.useSklearn:
            self.pca = sk.decomposition.PCA(n_components=dimensions)
            self.projectedTrainingData = self.pca.fit_transform(trainingData)
        else:
            self.projectedTrainingData = np.dot(trainingData, self.projectionMatrix)
        
        print("projected training data shape: " + str(self.projectedTrainingData.shape))
        self.trainingLabels = trainingLabels

    def test(self, testingData, testingLabels):
        """
        Used to project the testing data using the projection matrix.
        it then reports the number of correct and incorrect pridictions
        using K-neariest Neighbors, it reports the results for five values
        of k. these values are 1, 3, 5, 7, 9.
        finally it plots the results.
        :param testingData:
            The testing data as a (number of observations, number of features) numpy array.
        :param testingLabels:
            The testing labels as a numpy array.
        :return:
            Nothing.
        """
        
        neariestNeighborsRange = [1, 3, 5, 7, 9]
        accuracies = []

        mean = np.mean(testingData, axis=0)
        testingData = testingData - mean

        if self.useSklearn:
            projectedTestingData = self.pca.transform(testingData)
        else:
            projectedTestingData = np.dot(testingData, self.projectionMatrix)
        
        print("projected testing data shape: " + str(projectedTestingData.shape))
        
        for k in neariestNeighborsRange:
            currentIndex = 0
            correctPredictionsCount = 0
            incorrectPredictionsCount = 0
            accuracy = 0

            for testingVector in projectedTestingData:
                predictedLabel = self.predictLabel(testingVector, k)
                
                if predictedLabel == testingLabels[currentIndex]:
                    correctPredictionsCount = correctPredictionsCount + 1
                else:
                    incorrectPredictionsCount = incorrectPredictionsCount + 1
                
                currentIndex = currentIndex + 1
        
            accuracy = (correctPredictionsCount/(correctPredictionsCount + incorrectPredictionsCount)) * 100.0
            accuracies.append(accuracy)
            print("for K = " + str(k))
            print("correct predictions = " + str(correctPredictionsCount))
            print("incorrect predictions = " + str(incorrectPredictionsCount))
            print("accuracy = " + str(accuracy) + "%")

        self.plotResult(neariestNeighborsRange, accuracies)

    def plotResult(self, neariestNeighborsRange, accuracies):
        """
        Plots the testing result, it plots the K values on the x-axis
        and the accuracy on the y-axis.
        :param neariestNeighborsRange:
            The K values used for testing.
        :param accuracies:
            The accuracies obtained.
        :return:
            Nothing.
        """
        
        plt.xlabel('K Values')
        plt.ylabel('Accuracy')
        plt.title('Result')
        plt.plot(neariestNeighborsRange, accuracies)
        plt.show()

    def findNearestNeighbors(self, testingVector, neighborsCount = 1):
        """
        Used to find the neariest neighbors of a vector.
        :param testingVector:
            The testing vector.
        :param neighborsCount:
            The number of neighbors to return.
        :return:
            The neariest neighbors as a numpy array.
        """

        eculidenDistances = []
        testingVector = np.reshape(testingVector, (1, testingVector.shape[0]))
        
        for trainingVector in self.projectedTrainingData:
            trainingVector = np.reshape(trainingVector, (1, trainingVector.shape[0]))
            distance = np.linalg.norm(trainingVector - testingVector, axis=1)
            eculidenDistances.append(distance)
        
        eculidenDistances = np.array(eculidenDistances)
        eculidenDistances = np.reshape(eculidenDistances, len(eculidenDistances))
        sortedIndices = np.argsort(eculidenDistances)
        sortedIndices = sortedIndices[:neighborsCount,]
        nearestNeighborsLabels = self.trainingLabels[sortedIndices]
        return nearestNeighborsLabels

    def predictLabel(self, testingVector, neighborsCount = 1):
        """
        Predicts the label of the given testing vector.
        :param testingVector:
            The testing vector.
        :param neighborsCount:
            The number of neighbors used to predict the label.
        :return:
            The predicted label.
        """

        nearestNeighborsLabels = self.findNearestNeighbors(testingVector, neighborsCount)
        counts = np.bincount(nearestNeighborsLabels)
        predictedLabel = np.argmax(counts)
        return predictedLabel

    def calculateDimensionCount(self, alpha):
        """
        Calculates the needed dimension to get a particular alpha.
        :param alpha:
            The alpha value.
        :return:
            The number of dimension.
        """

        totalSum = np.sum(self.eignValues)
        comulativeSums = np.cumsum(self.eignValues)
        print(comulativeSums.shape)
        dimensionCount = 0

        for sum in comulativeSums:
            currentValue = float(sum/totalSum)
            dimensionCount = dimensionCount + 1

            if currentValue >= alpha:
                return dimensionCount
        
        return self.eignValues.size
    
    def isSymmetric(self, matrix, tol=1e-8):
        """
        Checks if a matrix is symmetric.
        :param matrix:
            The matrix to check.
        :return:
            True if the matrix is symmetric, False otherwise.
        """

        return np.allclose(matrix, matrix.T, atol=tol)