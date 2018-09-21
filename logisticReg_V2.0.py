import math
import os

class logReg():
	weights = list() # the weight to learn 
	Accuracy = 0.0


	def __init__(self,rate=0.1,iterations=10,howManySamples=5000,howToSelectSamples=1):
		self.rate = rate # the learning rate
		self.iterations = iterations # the number of iterations
		self.howManySamples = howManySamples
		self.howToSelectSamples = howToSelectSamples
		del self.weights[:]

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

	def sigmoid(self,outputValue):
		#return outputValue/(1.0+math.exp(-1*outputValue))
		return 1 / (1.0+math.exp(-1*outputValue))


	def classify(self,featuresValue):
		# this function calculate preducted value = 1/1+e^(B0+B1*X1+...)
		logit = self.weights[0]
		for x in range(1,len(self.weights)) :
			#print weights[x] , float(featuresValue[x-1])
			logit = logit + self.weights[x]* featuresValue[x-1]
		#print "logit", logit
		#print sigmoid(logit)
		return self.sigmoid(logit)

	def initilizeWieght(self,no) :
		# initilizw wieghts first time
		#print "no of w", no
		for i in range(0,no) :
			self.weights.append(0.0)
		
	def updateWieghts(self,ts,p,la):
		# update wieghts value after each training sample
		noOfWieghts = len(self.weights)
		#print weights
		for x in range(0,noOfWieghts):
			if x==0 :
				self.weights[x] = self.weights[x] + self.rate * (la - p)
				#weights[x] = weights[x] + rate * (la - p) * (1 - p) * p * 1
			else :
				self.weights[x] = self.weights[x] + self.rate * (la - p) * ts[x-1]
				#weights[x] = weights[x] + rate * (la - p) * (1 - p) * p * ts[x-1] 



	def train(self,tempTrainingFeatures,tempTrainingLables):
		noOfTrainingFeatures = len(tempTrainingFeatures[0])
		noOfTrainingSamples = len(tempTrainingFeatures) 
		self.initilizeWieght(noOfTrainingFeatures + 1)
		for eopic in range(0,self.iterations):
			for trainingSample in tempTrainingFeatures :
				indexOfLable = tempTrainingFeatures.index(trainingSample)
				lable = tempTrainingLables[indexOfLable]
				predicted = self.classify(trainingSample)
				#print lable, predicted

				self.updateWieghts(trainingSample,predicted,lable)
			#print "eopic ",eopic+1, "out of ", self.iterations
				

	def findAccuracy(self,list1,list2):
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


	def test(self,tempTestFeatures,tempTestLable):
		calculatedLable = list()
		for testSample in tempTestFeatures:
			classifiedOut = self.classify(testSample)
			#print classifiedOut
			if classifiedOut < 0.5 :
				crisp = 0.0
				calculatedLable.append(crisp)
			else :
				crisp = 1.0
				calculatedLable.append(crisp)
			#print  classifiedOut, crisp, tempTestLable[i]
		accuracy = self.findAccuracy(calculatedLable,tempTestLable)
		#print "iterations = ",self.iterations ,"    accuracy = ",accuracy, "%"
		return accuracy

		

	def myTest(self):
		folder = os.getcwd() + "\\"
		DSN,DSA = self.readNormalAndAnomalyFile(folder)
		training_DS,test_DS = self.prepareTrainingAndTestDS(DSN,DSA)
		col = []
		features , lables =  self.dataSetTune(training_DS,col)
		self.train(features,lables)
		features , lables =  self.dataSetTune(test_DS,col)
		self.Accuracy = self.test(features, lables)

	

def testAccuracyWithItrationChange():
	print "iterations" , "," , "Accuracy"
	for tempItration in range(1,3):
		obj = logReg(0.3,tempItration,5000)
		obj.myTest()
		print tempItration , "," , obj.Accuracy
		del obj

def testAccuracyWithSampleSelectChange():
	print "sampleSelect" , "," , "Accuracy"
	for sampleSelect in range(1,5):
		obj = logReg(0.3,5,5000,sampleSelect)
		obj.myTest()
		print sampleSelect , "," , obj.Accuracy
		del obj

def testAccuracyWithRateChange():
	print "rate" , "," , "Accuracy"
	rates = list()
	rateValue = 0.1
	while(rateValue <= 0.3) :
		rates.append(rateValue)
		rateValue+=0.01

	
	for tempRate in rates:
		obj = logReg(tempRate,8,500,1)
		obj.myTest()
		print tempRate , "," , obj.Accuracy
		del obj



#testAccuracyWithItrationChange()
#testAccuracyWithRateChange()
testAccuracyWithSampleSelectChange()