import numpy as np

class neuronInt3checkCorrection:
	def __init__(self, sVector):
		self.targetСategory = None
		self.oppositeСategory = None
		self.leftBorder = None
		self.rightBorder = None
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

	def checkCorrectionInstance(self, vecInputs, currentCategory):
		sMul = np.dot(self.vectorWeights, vecInputs[:self.sizeVector])
		if ((currentCategory != self.targetСategory)  and (sMul > self.rightBorder)) or ((currentCategory != self.oppositeСategory) and (sMul < self.leftBorder)):
			return 1#False
		else:
			return -1#True

n01 = neuronInt3checkCorrection(7)
n02 = neuronInt3checkCorrection(7)
n03 = neuronInt3checkCorrection(7)
n04 = neuronInt3checkCorrection(7)
n05 = neuronInt3checkCorrection(7)
n06 = neuronInt3checkCorrection(7)
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

#weights = np.array([0, 8, 2])		
inputs = np.array([20, -9, 16,	20,	20,	20,	20,	2080])
qq = 0
while qq < 1:
	if n01.checkCorrectionInstance(inputs, 0) >=0:
		print('n1 error', inputs[7])
	elif n02.checkCorrectionInstance(inputs, 0) >=0:
		print('n2 error', inputs[7])
	elif n03.checkCorrectionInstance(inputs, 0) >=0:
		print('n3 error', inputs[7])
	elif n04.checkCorrectionInstance(inputs, 0) >=0:
		print('n4 error', inputs[7])
	elif n05.checkCorrectionInstance(inputs, 0) >=0:
		print('n5 error', inputs[7])
	elif n06.checkCorrectionInstance(inputs, 0) >=0:
		print('n6 error', inputs[7])
	qq += 1


'''nInt3cC = neuronInt3checkCorrection(3)
nInt3cC.setVectorWeights(weights)
nInt3cC.setBorders(-4, 7)
nInt3cC.setCategories(4, 1)
ee = nInt3cC.checkCorrectionInstance(inputs, 4)
print(nInt3cC.getRightBorder())'''
qq = 9