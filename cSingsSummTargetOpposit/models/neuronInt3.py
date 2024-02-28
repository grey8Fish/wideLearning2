import numpy as np

class neuronInt3:
	def __init__(self, sizeVector):
		self.leftBorder = -7
		self.rightBorder = 5
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

inputs = np.array([2, 1, 0])
weights = np.array([0, 1, 2])		
nInt3 = neuronInt3(3)
nInt3.setVectorWeights(weights)
qq = nInt3.scalarMultiplication(inputs)
nInt3.setBorders(-4, 7)

print(nInt3.getRightBorder())		
