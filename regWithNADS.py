import math
import os

# the learning rate
rate = 0.3
# the weight to learn 
weights = []
# the number of iterations
ITERATIONS = 1


def convertToFloat(tempList):
	# this function recive a list of text and return list of float 
	lineAsFloat = list()
	for item in tempList:
		tempList = float(item)
		lineAsFloat.append(tempList)
	return lineAsFloat



def readDataSet(filePath, noOfSamplesToTake):
	# this function read path of dataset file and import it as a list of float values
	dataSet= list()
	line = ""
	tempInterrupt = 0
	fileHandeler = open(filePath)
	for line in fileHandeler :
		tempInterrupt+=1
		if tempInterrupt > noOfSamplesToTake :
			break
		line.strip(line)
		if line[0] == "#" :
			continue
		lineAsList = line.split(",")
		for feature in lineAsList:
			feature.strip()
		lineAsFloat = convertToFloat(lineAsList)
		
		dataSet.append(lineAsFloat)
	fileHandeler.close()
	return dataSet

def readNormalAndAnomalyFile(folderPath,noOfSamples):
	dataSetNormal= list()
	dataSetAnomaly = list()
	normalFile = folderPath + "normal.csv"
	anomalyFile = folderPath + "anomaly.csv"
	dataSetNormal = readDataSet(normalFile,noOfSamples)
	dataSetAnomaly = readDataSet(anomalyFile,noOfSamples)
	return dataSetNormal,dataSetAnomaly

def prepareTrainingAndTestDS(tempDSN,tempDSA):
	no_Of_Samples = len(tempDSN)
	temp_trainingDS = tempDSN[0:no_Of_Samples/2] + tempDSA[0:no_Of_Samples/2]
	temp_testDS = tempDSN[no_Of_Samples/2:no_Of_Samples] + tempDSA[no_Of_Samples/2:no_Of_Samples]
	return temp_trainingDS,temp_testDS


def dataSetTune(tempDataSet,columsToRemove) :
	# this function take dataset list and delete unimportant column and also put tables in a sperated list
	tempLables = list()
	widthOfDS = len(tempDataSet[0])
	for line in tempDataSet:
		#print line,widthOfDS,tempDataSet[0]
		tempLables.append(line[widthOfDS-1])
		del(line[widthOfDS-1])
		for columNo in columsToRemove:
			del(line[columNo])
	return tempDataSet,tempLables

def sigmoid(outputValue):
	return 1.0/(1.0+math.exp(-1*outputValue))

def classify(featuresValue):
	# this function calculate preducted value = 1/1+e^(B0+B1*X1+...)
	logit = weights[0]
	for x in range(1,len(weights)) :
		#print weights[x] , float(featuresValue[x-1])
		logit = logit + weights[x]* featuresValue[x-1]
	return sigmoid(logit)

def initilizeWieght(no) :
	# initilizw wieghts first time
	for i in range(0,no) :
		weights.append(0.0)
	
def updateWieghts(ts,p,l):
	# update wieghts value after each training sample
	noOfWieghts = len(weights)
	#print weights
	for x in range(0,noOfWieghts):
		if x==0 :
			weights[x] = weights[x] + rate * (l - p) * 1
		else :
			weights[x] = weights[x] + rate * (l - p) * ts[x-1]



def train(tempTrainingFeatures,tempTrainingLables):
	noOfTrainingFeatures = len(tempTrainingFeatures[0])
	noOfTrainingSamples = len(tempTrainingFeatures) 
	initilizeWieght(noOfTrainingFeatures + 1)
	for eopic in range(0,ITERATIONS):
		for trainingSample in tempTrainingFeatures :
			indexOfLable = tempTrainingFeatures.index(trainingSample)
			lable = tempTrainingLables[indexOfLable]
			predicted = classify(trainingSample)
			#print trainingSample, predicted

			updateWieghts(trainingSample,predicted,lable)
			

def findAccuracy(list1,list2):
	trueSample = 0
	noOfSamples = len(list1);
	for sampleNo in range(0,noOfSamples) :
		#print list1[sampleNo],list2[sampleNo]
		if list1[sampleNo] == list2[sampleNo]:
			trueSample+=1
	#print float(trueSample) , float(noOfSamples)
	acc = float(trueSample)/float(noOfSamples)
	#print acc
	return acc*100.0


def test(tempTestFeatures,tempTestLable):
	calculatedLable = list()
	i=0
	for testSample in tempTestFeatures:
		classifiedOut = classify(testSample)
		#print classifiedOut
		if classifiedOut < 0.05 :
			crisp = 0.0
			calculatedLable.append(crisp)
		else :
			crisp = 0.1
			calculatedLable.append(crisp)
		#print  classifiedOut, crisp, tempTestLable[i]
		i+=1
	accuracy = findAccuracy(calculatedLable,tempTestLable)


	print "Accuracy = ",accuracy, "%"




	
	

	

def myTest():
	folder = os.getcwd() + "\\"
	print folder
	#folder = "F:\\python\\code\\reg\\"
	howManySamples = 5000
	DSN,DSA = readNormalAndAnomalyFile(folder,howManySamples)
	training_DS,test_DS = prepareTrainingAndTestDS(DSN,DSA)
	col = []
	features , lables =  dataSetTune(training_DS,col)
	train(features,lables)
	test(features, lables)
	features , lables =  dataSetTune(test_DS,col)
	test(features, lables)

	


myTest()