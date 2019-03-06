import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

class LDA:
    def __init__(self):
        self.classCount = None
        self.overallMean = None
        self.projectionMatrix = None
        self.projectedTrainingData = None
        self.trainingLabels = None
        self.lda = None
        self.featureCount = 39
        self.useSklearn = True
    
    def train(self, trainingData, trainingLabels):
        sampleCount = np.bincount(trainingLabels)
        self.classCount = sampleCount.size - 1
        self.overallMean = np.mean(trainingData, axis=0)
        print("the number of classes is: " + str(self.classCount))

        classes = np.split(trainingData, self.classCount)
        print("the shape of a single class matrix is: " + str(classes[0].shape))

        if self.useSklearn:
            self.lda = sk.discriminant_analysis.LinearDiscriminantAnalysis(n_components = self.featureCount)
            self.projectedTrainingData = self.lda.fit_transform(trainingData, trainingLabels)
            self.trainingLabels = trainingLabels
            return
        
        betweenClassScatterMatrix = np.ones((10304, 10304))
        withinClassScatterMatrix = np.ones((10304, 10304))

        for i in range(0, self.classCount):
            classMean = np.mean(classes[i], axis=0)

            meanDifference = classMean - self.overallMean
            meanDifference = np.reshape(meanDifference, (1, meanDifference.shape[0]))
            
            classes[i] = classes[i] - classMean

            if i == 0:
                np.multiply(sampleCount[i + 1], np.dot(meanDifference.T, meanDifference), out=betweenClassScatterMatrix)
                np.dot(classes[i].T, classes[i], out=withinClassScatterMatrix)
            else:
                np.add(betweenClassScatterMatrix, np.multiply(sampleCount[i + 1], np.dot(meanDifference.T, meanDifference)), out=betweenClassScatterMatrix)
                np.add(withinClassScatterMatrix, np.dot(classes[i].T, classes[i]), out=withinClassScatterMatrix)

        print("the shape of the between class scatter matrix is: " + str(betweenClassScatterMatrix.shape))
        print("the shape of the within class scatter matrix is: " + str(withinClassScatterMatrix.shape))

        del sampleCount
        del classes

        withinClassScatterMatrix = np.linalg.inv(withinClassScatterMatrix)
        finalMatrix = np.dot(withinClassScatterMatrix, betweenClassScatterMatrix)

        del betweenClassScatterMatrix
        del withinClassScatterMatrix

        print("calculating eignvectors and eignvalues.")
        eignValues, eignVectors = np.linalg.eigh(finalMatrix)
        
        del finalMatrix

        eignValues = np.real(eignValues)
        eignVectors = np.real(eignVectors)

        indices = np.argsort(eignValues)
        indices = indices[::-1]

        eignValues = eignValues[indices]
        eignVectors = eignVectors[:,indices]

        print("the shape of the eign values matrix is: " + str(eignValues.shape))
        print("the shape of the eign vectors matrix is: " + str(eignVectors.shape))

        self.projectionMatrix = eignVectors[:,:self.featureCount]
        print("the shape of the projection matrix is: " + str(self.projectionMatrix.shape))

        del eignValues
        del eignVectors
 
        self.projectedTrainingData = np.dot(trainingData, self.projectionMatrix)
        self.trainingLabels = trainingLabels
        print("the shape of the projected training data matrix is: " + str(self.projectedTrainingData.shape))

    def test(self, testingData, testingLabels, neighborsCount = 1):
        neariestNeighborsRange = [1, 3, 5, 7, 9]
        accuracies = []

        if self.useSklearn:
            projectedTestingData = self.lda.transform(testingData)
        else:
            projectedTestingData = np.dot(testingData, self.projectionMatrix)

        print("the shape of the projected testing data matrix is: " + str(projectedTestingData.shape))

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
        plt.xlabel('K Values')
        plt.ylabel('Accuracy')
        plt.title('Result')
        plt.plot(neariestNeighborsRange, accuracies)
        plt.show()

    def findNearestNeighbors(self, testingVector, neighborsCount = 1):
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
        nearestNeighborsLabels = self.findNearestNeighbors(testingVector, neighborsCount)
        counts = np.bincount(nearestNeighborsLabels)
        predictedLabel = np.argmax(counts)
        return predictedLabel

