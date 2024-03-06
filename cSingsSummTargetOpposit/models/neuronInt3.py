import numpy as np

class neuronInt3:
	def __init__(self, sizeVector):
		self.leftBorder = 0
		self.rightBorder = 0
		self.vectorWeights = np.zeros(sizeVector, dtype=int)
	
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
		sMul = np.dot(self.vectorWeights, vecInputs)
		if sMul < self.leftBorder:
			return(-1)
		elif sMul > self.rightBorder:
			return(1)
		else:
			return(0)

inputs = np.array([2, 1, 0])
weights = np.array([0, -5, 2])		

nInt3 = neuronInt3(3)
nInt3.setVectorWeights(weights)
qq = nInt3.scalarMultiplication(inputs)
nInt3.setBorders(-4, 7)
ww = nInt3.digit3activationFunction(inputs)
print(nInt3.getRightBorder())
