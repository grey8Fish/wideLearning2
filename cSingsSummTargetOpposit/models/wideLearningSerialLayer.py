import numpy as np

class wideLearningSerialLayer:
	def __init__(self, countClasses, maxInstance, sizeVector, nameFile):
	    #получает количество столбцов=размер вектора весов, количество классов,
        #количество экземпляров, имя файла
		
		self.currentWeights = np.zeros(sizeVector, dtype=int)
		#текущий вектор весов

		self.previousWeights = np.zeros(sizeVector, dtype=int)
		#предыдущий вектор весов

		#self.bestWeights = np.zeros(10, sizeVector+1, dtype=int)
		#10 лучших весов, в последнем столбце количество отсеченных
		
		self.countInstancesEachClassCorrection = np.zeros(countClasses, dtype=int)
		#количество экземпляров в каждом классе корректирующей выборки

		self.countInstancesEachClassTraining = np.zeros(countClasses, dtype=int)
		#количество экземпляров в каждом классе обучающей выборки
		
		#self.nameClasses
		#имена классов

		self.inputsClassCorrection = np.zeros(countClasses, maxInstance, sizeVector, dtype=int)
		#входные экземпляры корректирующей выборки
		
		self.inputsClassTraining = np.zeros(countClasses, maxInstance, sizeVector, dtype=int)
		#входные экземпляры обучающей выборки

