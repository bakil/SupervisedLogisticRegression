import math
import os

class KnnModel() :
	Accuracy = 0.0
	def __init__(self,howManySamples,keys=7,typeOfFunction="Euclidean",howToSelectSamples=1):
		self.howManySamples = howManySamples
		self.keys = keys # no of keys
		self.typeOfFunction = typeOfFunction
		self.howToSelectSamples = howToSelectSamples
		
	def convertToFloat(self,tempList):
		# this function recive a list of text and return list of float 
		lineAsFloat = list()
		for item in tempList:
			tempValue = float(item)
			lineAsFloat.append(tempValue)
		return lineAsFloat



	def readDataSet(self,filePath):
		# this function read path of dataset file and import it as a list of float values
		dataSet= list()
		line = ""
		tempInterrupt = 0
		fileHandeler = open(filePath)
		for line in fileHandeler :
			tempInterrupt+=1
			if tempInterrupt > self.howManySamples :
				break
			line.strip(line)
			if line[0] == "#" :
				continue
			lineAsList = line.split(",")
			for feature in lineAsList:
				feature.strip()
			lineAsFloat = self.convertToFloat(lineAsList)
			
			dataSet.append(lineAsFloat)
		fileHandeler.close()
		return dataSet


	def addLabelToDS(self,orignalNormalDS,orignalAnomalyDS):
		for simple in orignalNormalDS :
			simple.append(0.0)
		for simple in orignalAnomalyDS :
			simple.append(1.0)
		return orignalNormalDS,orignalAnomalyDS

	def readNormalAndAnomalyFile(self,folderPath):
		dataSetNormal= list()
		dataSetAnomaly = list()
		normalFile = folderPath + "normal.csv"
		anomalyFile = folderPath + "anomaly.csv"
		dataSetNormal = self.readDataSet(normalFile)
		dataSetAnomaly = self.readDataSet(anomalyFile)
		dataSetNormal,dataSetAnomaly = self.addLabelToDS(dataSetNormal,dataSetAnomaly)
		return dataSetNormal,dataSetAnomaly


	def prepareTrainingAndTestDS(self,tempDSN,tempDSA):
		if self.howToSelectSamples == 1 :
			tr_DS,te_DS = self.prepareTrainingAndTestDS1(tempDSN,tempDSA)
		elif self.howToSelectSamples == 2 :
			tr_DS,te_DS = self.prepareTrainingAndTestDS2(tempDSN,tempDSA)
		elif self.howToSelectSamples == 3 :
			tr_DS,te_DS = self.prepareTrainingAndTestDS3(tempDSN,tempDSA)
		elif self.howToSelectSamples == 4 :
			tr_DS,te_DS = self.prepareTrainingAndTestDS4(tempDSN,tempDSA)
		else :
			tr_DS,te_DS = self.prepareTrainingAndTestDS1(tempDSN,tempDSA)
		return tr_DS,te_DS




	def prepareTrainingAndTestDS1(self,tempDSN,tempDSA):
		no_Of_Samples = len(tempDSN)
		temp_trainingDS = tempDSN[0:no_Of_Samples/2] + tempDSA[0:no_Of_Samples/2]
		temp_testDS = tempDSN[no_Of_Samples/2:no_Of_Samples] + tempDSA[no_Of_Samples/2:no_Of_Samples]
		return temp_trainingDS,temp_testDS

	def prepareTrainingAndTestDS2(self,tempDSN,tempDSA):
		no_Of_Samples = len(tempDSN)
		temp_testDS = tempDSN[0:no_Of_Samples/2] + tempDSA[0:no_Of_Samples/2]
		temp_trainingDS = tempDSN[no_Of_Samples/2:no_Of_Samples] + tempDSA[no_Of_Samples/2:no_Of_Samples]
		return temp_trainingDS,temp_testDS

	def prepareTrainingAndTestDS3(self,tempDSN,tempDSA):
		no_Of_Samples = len(tempDSN)
		temp_trainingDS = list()
		temp_testDS = list()
		for n in range(no_Of_Samples):
			if n%2 == 0:
				temp_testDS.append(tempDSN[n])
				temp_testDS.append(tempDSA[n])
			else :
				temp_trainingDS.append(tempDSN[n])
				temp_trainingDS.append(tempDSA[n])
		return temp_trainingDS,temp_testDS

	def prepareTrainingAndTestDS4(self,tempDSN,tempDSA):
		no_Of_Samples = len(tempDSN)
		temp_trainingDS = list()
		temp_testDS = list()
		for n in range(no_Of_Samples):
			if n%2 == 1:
				temp_testDS.append(tempDSN[n])
				temp_testDS.append(tempDSA[n])
			else :
				temp_trainingDS.append(tempDSN[n])
				temp_trainingDS.append(tempDSA[n])
		return temp_trainingDS,temp_testDS


	def dataSetTune(self,tempDataSet,columsToRemove) :
		# this function take dataset list and delete unimportant column and also put tables in a sperated list
		tempLables = list()
		widthOfDS = len(tempDataSet[0])
		for line in tempDataSet:
			tempLables.append(line[widthOfDS-1])
			del(line[widthOfDS-1])
			for columNo in columsToRemove:
				del(line[columNo])
		return tempDataSet,tempLables

	
	def distanceEuclidean(self,tempSample,testSample):
		noOfFeatures = len(tempSample)
		sum =0
		for no in range(0,noOfFeatures) :
			sum = sum + math.pow(tempSample[no]-testSample[no],2)
			sum = math.sqrt(sum)
		return sum

	def distanceManhattan(self,tempSample,testSample):
		noOfFeatures = len(tempSample)
		sum =0
		for no in range(0,noOfFeatures) :
			sum = sum + abs(tempSample[no]-testSample[no])
			sum = math.sqrt(sum)
		return sum


	def calculateDistance(self,tempSample,testSample):
		if self.typeOfFunction == "Euclidean" :
			return self.distanceEuclidean(tempSample,testSample)
		if self.typeOfFunction == "Manhattan" :
			return self.distanceManhattan(tempSample,testSample)



	def KNN(self,test_sample,Tr_DS_features,Tr_DS_lables) :
		distances = list()
		for sample in Tr_DS_features :
			distances.append(self.calculateDistance(sample,test_sample))
		
		temp_min_lable = self.findMinDistancesAndRetuenThereLables(distances,Tr_DS_lables)
		return self.findOutGroup(temp_min_lable)

	def testKNN(self,Te_DS_features,Te_DS_lables,Tr_DS_features,Tr_DS_lables):
		no_Of_Samples = len(Te_DS_lables)
		no_of_true_preduction = 0
		for no in range(no_Of_Samples) :
			preducted = self.KNN(Te_DS_features[no],Tr_DS_features,Tr_DS_lables)
			if preducted == Te_DS_lables[no] :
				no_of_true_preduction = no_of_true_preduction + 1
		acc = no_of_true_preduction * 1.0 / no_Of_Samples
		accuracy = acc * 100
		#print "Accuracy : ", accuracy
		self.Accuracy = accuracy


	def findMinDistancesAndRetuenThereLables(self,dis,la):
		minDistanceLable = list()
		temp_dis = dis
		temp_la = la
		for i in range(self.keys) :
			x = temp_dis.index(min(temp_dis))
			minLa = temp_la[x]
			minDistanceLable.append(minLa)
			temp_dis[x] = temp_dis[x] * 500.0
		return minDistanceLable

	def findOutGroup(self,temp_list) :
		l0 = 0
		l1 = 0
		for item in temp_list :
			if item == 0.0:
				l0 = l0 + 1
			else :
				l1 = l1 + 1
		if l0>l1 :
			return 0.0
		else :
			return 1.0


		
		

		

	def myTest(self):
		folder = os.getcwd() + "\\"
		DSN,DSA = self.readNormalAndAnomalyFile(folder)
		training_DS,test_DS = self.prepareTrainingAndTestDS(DSN,DSA)
		#print training_DS
		col = []
		Tr_features , Tr_lables =  self.dataSetTune(training_DS,col)
		Te_features , Te_lables =  self.dataSetTune(test_DS,col)
		self.testKNN(Te_features,Te_lables,Tr_features,Tr_lables)

		





def testAccuracyWithKeyChange():
	print "Key" , "," , "Accuracy"
	for tempKey in range(1,15,2):
		obj = KnnModel(500,tempKey,"Manhattan",1)
		obj.myTest()
		print tempKey , "," , obj.Accuracy
		del obj

def testAccuracyWithSampleSelectChange():
	print "sampleSelect" , "," , "Accuracy"
	for sampleSelect in range(1,5):
		obj = KnnModel(500,5,"Manhattan",sampleSelect)
		obj.myTest()
		print sampleSelect , "," , obj.Accuracy
		del obj


# Euclidean or Manhattan


#testAccuracyWithSampleSelectChange()
testAccuracyWithKeyChange()

"""
knnObj = KnnModel(500,7,"Manhattan",2)
knnObj.myTest()
print "Accuracy",knnObj.Accuracy
del knnObj
knnObj = KnnModel(500,7,"Euclidean",2)
knnObj.myTest()
print "Accuracy",knnObj.Accuracy

"""