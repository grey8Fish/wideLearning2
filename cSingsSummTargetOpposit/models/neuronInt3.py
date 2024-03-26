import numpy as np

class neuronInt3:
	def __init__(self, sVector):
		self.targetСategory = None
		self.oppositeСategory = None
		self.leftBorder = 0
		self.rightBorder = 0
		self.sizeVector = sVector
		self.vectorWeights = np.zeros(sVector, dtype=int)
	
	def setCategories(self, oppositeС, targetС):
		self.oppositeСategory = oppositeС
		self.targetСategory = targetС

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

	def scalarMultiplication(self, vecInputs):
		sMul = np.dot(self.vectorWeights, vecInputs)
		return sMul

	def digit3activationFunction(self, vecInputs):
		sMul = np.dot(self.vectorWeights, vecInputs[:self.sizeVector])
		if sMul < self.leftBorder:
			return(-1)
		elif sMul > self.rightBorder:
			return(1)
		else:
			return(0)

n01 = neuronInt3(7)
n02 = neuronInt3(7)
n03 = neuronInt3(7)
n04 = neuronInt3(7)
n05 = neuronInt3(7)
n06 = neuronInt3(7)
n01.setBorders(-1240, 1072)
n01.setCategories(1, 0)
n01.setVectorWeights([0, 32, -4, 40, 40, 40, 40])
n02.setBorders(-650, 86)
n02.setCategories(1, 0)
n02.setVectorWeights([0, 20, 18,  0,  0,  0,  0])
n03.setBorders(-670, 977)
n03.setCategories(1, 0)
n03.setVectorWeights([0, -9,  4, 40,  0,  0,  0])
n04.setBorders(-762, 824)
n04.setCategories(1, 0)
n04.setVectorWeights([40, -4,  0,  0,  0,  0,  0])
n05.setBorders(-1838, 1574)
n05.setCategories(1, 0)
n05.setVectorWeights([40,  16,  -6,   0,   0, -40,   0])
n06.setBorders(-755, -755)
n06.setCategories(1, 0)
n06.setVectorWeights([-40,  -5,   6, -40,  40,   0,   0])

inputs = np.array([20,	-2,	-14,	-20,	20,	20,	-20,	3899])
#inputs = np.array([20,	-19,	2,	20,	20,	-20,	-20,	633])
qq = 0
while qq < 1:
	ee = n01.digit3activationFunction(inputs)
	if ee == 1:
		qq += 1
		continue
	elif ee == -1:
		qq += 1
		continue
	ee = n02.digit3activationFunction(inputs)
	if ee == 1:
		qq += 1
		continue
	elif ee == -1:
		qq += 1
		continue
	ee = n03.digit3activationFunction(inputs)
	if ee == 1:
		qq += 1
		continue
	elif ee == -1:
		qq += 1
		continue
	ee = n04.digit3activationFunction(inputs)
	if ee == 1:
		qq += 1
		continue
	elif ee == -1:
		qq += 1
		continue
	ee = n05.digit3activationFunction(inputs)
	if ee == 1:
		qq += 1
		continue
	elif ee == -1:
		qq += 1
		continue
	ee = n06.digit3activationFunction(inputs)
	if ee == 1:
		qq += 1
		continue
	elif ee == -1:
		qq += 1
		continue

	qq += 1
qq = 9
