import math
import os

class logReg():
	weights = list() # the weight to learn 
	Accuracy = 0.0
	FN = 0
	FP = 0
	TN = 0
	TP = 0


	def __init__(self,rate=0.1,iterations=1,howManySamples=50,howToSelectSamples=1):
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

	def readNormalAndAnomalyFile(self):
		dataSetNormal= list()
		dataSetAnomaly = list()
		dataSetNormal = self.readDataSet("normal.csv")
		dataSetAnomaly = self.readDataSet("anomaly.csv")
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
		no_Of_Samples_N = len(tempDSN)
		no_Of_Samples_A = len(tempDSA)
		temp_trainingDS = tempDSN[0:no_Of_Samples_N/2] + tempDSA[0:no_Of_Samples_A/2]
		temp_testDS = tempDSN[no_Of_Samples_N/2:no_Of_Samples_N] + tempDSA[no_Of_Samples_A/2:no_Of_Samples_A]
		return temp_trainingDS,temp_testDS

	def prepareTrainingAndTestDS2(self,tempDSN,tempDSA):
		no_Of_Samples_N = len(tempDSN)
		no_Of_Samples_A = len(tempDSA)
		no_Of_Samples = len(tempDSN)
		temp_testDS = tempDSN[0:no_Of_Samples_N/2] + tempDSA[0:no_Of_Samples_A/2]
		temp_trainingDS = tempDSN[no_Of_Samples_N/2:no_Of_Samples_N] + tempDSA[no_Of_Samples_A/2:no_Of_Samples_A]
		return temp_trainingDS,temp_testDS

	def prepareTrainingAndTestDS3(self,tempDSN,tempDSA):
		no_Of_Samples_N = len(tempDSN)
		no_Of_Samples_A = len(tempDSA)
		no_Of_Samples = len(tempDSN)
		temp_trainingDS = list()
		temp_testDS = list()
		for n in range(no_Of_Samples_N):
			if n%2 == 0:
				temp_testDS.append(tempDSN[n])
			else :
				temp_trainingDS.append(tempDSN[n])
		for n in range(no_Of_Samples_A):
			if n%2 == 0:
				temp_testDS.append(tempDSA[n])
			else :
				temp_trainingDS.append(tempDSA[n])
		return temp_trainingDS,temp_testDS

	def prepareTrainingAndTestDS4(self,tempDSN,tempDSA):
		no_Of_Samples_N = len(tempDSN)
		no_Of_Samples_A = len(tempDSA)
		no_Of_Samples = len(tempDSN)
		temp_trainingDS = list()
		temp_testDS = list()
		for n in range(no_Of_Samples_N):
			if n%2 == 1:
				temp_testDS.append(tempDSN[n])
			else :
				temp_trainingDS.append(tempDSN[n])
		for n in range(no_Of_Samples_A):
			if n%2 == 1:
				temp_testDS.append(tempDSA[n])
			else :
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
			predicted = list1[sampleNo]
			Actual = list2[sampleNo]

			if predicted == Actual:
				trueSample+=1
				if predicted == 0.0:
					self.TN = self.TN + 1
				else :
					self.TP = self.TP + 1
			else:
				if predicted == 0.0:
					self.FN = self.FN + 1
				else :
					self.FP = self.FP + 1

		#print float(trueSample) , float(noOfSamples)
		acc = (float(trueSample)/float(noOfSamples)) * 100
		#print acc
		self.Accuracy = acc
		


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
		self.findAccuracy(calculatedLable,tempTestLable)
		#print "iterations = ",self.iterations ,"    accuracy = ",accuracy, "%"
		#return accuracy

	def save_TrDS_TeDS(self,t_training_DS,t_test_DS):
		current_directory = os.getcwd()
		final_directory = os.path.join(current_directory, r'out')
		if not os.path.exists(final_directory):
		   os.makedirs(final_directory)
		traininggDataSetFile = os.path.join(final_directory, r'trainingDataSet.csv')
		f = open(traininggDataSetFile, "w")
		for sample in t_training_DS :
			line = ",".join(str(e) for e in sample) + "\n"
			f.write(line)
		f.close()

		testingDataSetFile = os.path.join(final_directory, r'testDataSet.csv')
		f = open(testingDataSetFile, "w")
		for sample in t_test_DS :
			line = ",".join(str(e) for e in sample) + "\n"
			f.write(line)
		f.close()

		

	def myTest(self):
		DSN,DSA = self.readNormalAndAnomalyFile()
		training_DS,test_DS = self.prepareTrainingAndTestDS(DSN,DSA)
		self.save_TrDS_TeDS(training_DS,test_DS)
		col = []
		features , lables =  self.dataSetTune(training_DS,col)
		self.train(features,lables)
		features , lables =  self.dataSetTune(test_DS,col)
		self.test(features, lables)

	

def testAccuracyWithItrationChange():
	fixed_no_of_samples = 50
	fixed_value_of_rate = 0.2
	fixed_sampleSelectionMethod = 1
	#fixed_itration_value = 1
	current_directory = os.getcwd()
	final_directory = os.path.join(current_directory, r'out')
	if not os.path.exists(final_directory):
		  os.makedirs(final_directory)
	outFile = os.path.join(final_directory, r'differentItratsion.txt')

	f = open(outFile, "w")
	f.write("checking how accuracy change with changing way of iterations  \n \n")
	f.write("iterations" + "," + "Accuracy"+ "," + "FP"+ "," + "FN"+ "," + "TP"+"," + "TN \n")

	f.close()
	f = open(outFile, "a")

	print " checking how accuracy change with changing way of iterations \n"
	print "iterations" , "," , "Accuracy", "," , "FP", "," , "FN", "," , "TP","," , "TN"
	for tempItration in range(1,30):
		obj = logReg(fixed_value_of_rate,tempItration,fixed_no_of_samples,fixed_sampleSelectionMethod)
		obj.myTest()
		print tempItration , "," , obj.Accuracy
		f.write( str(tempItration) + "," + str(obj.Accuracy) + "," + str(obj.FP) + "," + str(obj.FN)+ "," + str(obj.TP) + "," + str(obj.TN) + "\n")
		del obj
	f.close()

def testAccuracyWithSampleSelectChange():
	fixed_no_of_samples = 50
	fixed_value_of_rate = 0.2
	#fixed_sampleSelectionMethod = 1
	fixed_itration_value = 1
	current_directory = os.getcwd()
	final_directory = os.path.join(current_directory, r'out')
	if not os.path.exists(final_directory):
		  os.makedirs(final_directory)
	outFile = os.path.join(final_directory, r'differentSample.txt')
	f = open(outFile, "w")
	f.write("checking how accuracy change with changing way of selecting samples \n \n")
	f.write("sampleSelect" + "," + "Accuracy"+ "," + "FP"+ "," + "FN"+ "," + "TP"+"," + "TN \n")
	f.close()
	f = open(outFile, "a")

	print " checking how accuracy change with changing way of selecting samples \n"
	print "sampleSelect" , "," , "Accuracy", "," , "FP", "," , "FN", "," , "TP","," , "TN"
	for sampleSelect in range(1,5):
		obj = logReg(fixed_value_of_rate,fixed_itration_value,fixed_no_of_samples,sampleSelect)
		obj.myTest()
		print sampleSelect , "," , obj.Accuracy, "," , obj.FP, "," , obj.FN, "," , obj.TP, "," , obj.TN
		f.write( str(sampleSelect) + "," + str(obj.Accuracy) + "," + str(obj.FP) + "," + str(obj.FN)+ "," + str(obj.TP) + "," + str(obj.TN) + "\n")
		del obj
	f.close()


def testAccuracyWithRateChange():
	current_directory = os.getcwd()
	final_directory = os.path.join(current_directory, r'out')
	if not os.path.exists(final_directory):
		  os.makedirs(final_directory)
	outFile = os.path.join(final_directory, r'differentRate.txt')

	f = open(outFile, "w")
	f.write("checking how accuracy change with changing way of Rate \n \n")
	f.write("Rate" + "," + "Accuracy"+ "," + "FP"+ "," + "FN"+ "," + "TP"+"," + "TN \n")

	f.close()
	f = open(outFile, "a")

	print " checking how accuracy change with changing Rate \n"
	print "Rate" , "," , "Accuracy", "," , "FP", "," , "FN", "," , "TP","," , "TN"
	rates = list()
	rateValue = 0.1
	while(rateValue <= 0.3) :
		rates.append(rateValue)
		rateValue+=0.01
	for tempRate in rates:
		obj = logReg(tempRate,1,50,1)
		obj.myTest()
		print tempRate , "," , obj.Accuracy
		f.write( str(tempRate) + "," + str(obj.Accuracy) + "," + str(obj.FP) + "," + str(obj.FN)+ "," + str(obj.TP) + "," + str(obj.TN) + "\n")
		del obj
	f.close()


def evaluation() :
	testAccuracyWithRateChange()
	testAccuracyWithSampleSelectChange()
	testAccuracyWithItrationChange()
	current_directory = os.getcwd()
	final_directory = os.path.join(current_directory, r'out')
	if os.path.exists(final_directory):
		os.startfile(final_directory)

evaluation()



#obj = logReg()
#obj.myTest()
#print obj.Accuracy