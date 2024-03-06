import numpy as np

class neuronInt3checkCorrection:
	def __init__(self, sizeVector):
		self.targetСategory = 0
		self.oppositeСategory = 0
		self.leftBorder = 0
		self.rightBorder = 0
		self.vectorWeights = np.zeros(sizeVector, dtype=int)
	
	def setCategories(self, targetС, oppositeС):
		self.targetСategory = targetС
		self.oppositeСategory = oppositeС

	def getTargetСategory(self):
		return self.targetСategory

	def getOppositeСategory(self):
		return self.oppositeСategory

	def setBorders(self, leftB, rightB):
		self.rightBorder = rightB
		self.leftBorder = leftB
		
	def getRightBorder(self):
		return self.rightBorder

	def getLeftBorder(self):
		return self.leftBorder

	def setVectorWeights(self, vecWeights):
		self.vectorWeights = vecWeights

	def getVectorWeights(self):
		return self.vectorWeigts

	def checkCorrectionInstance(self, vecInputs, currentCategory):
		sMul = np.dot(self.vectorWeights, vecInputs)
		if ((currentCategory != self.targetСategory)  and (sMul > self.rightBorder)) or ((currentCategory != self.oppositeСategory) and (sMul < self.leftBorder)):
			return False
		else:
			return True

inputs = np.array([2, 1, 0])
weights = np.array([0, 8, 2])		

nInt3cC = neuronInt3checkCorrection(3)
nInt3cC.setVectorWeights(weights)
nInt3cC.setBorders(-4, 7)
nInt3cC.setCategories(4, 1)
ee = nInt3cC.checkCorrectionInstance(inputs, 4)
print(nInt3cC.getRightBorder())