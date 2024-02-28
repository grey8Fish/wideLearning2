import numpy as np
import csv
from models.DataLoader import DataLoader

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
		
		#self.classesName
		#имена классов

		self.inputsClassCorrection = np.zeros((countClasses, maxInstance, sizeVector), dtype=int)
		#входные экземпляры корректирующей выборки
		
		self.inputsClassTraining = np.zeros((countClasses, maxInstance, sizeVector), dtype=int)
		#входные экземпляры обучающей выборки

	def setColumnName(self, columName):
		self.columnName = columName #имена столбцов

file_names = ['seed0_23_11_26.csv', 'seed1_23_11_26.csv', 'seed2_23_11_26.csv']
data_loader = DataLoader(file_names)
data_loader.load_data()
#argClasses = data_loader.arg_classes
#nameColumn = data_loader.get_column_names()
qq = data_loader.instances_max

wlsl = wideLearningSerialLayer(3, 67, 123, 'fileNameTmp')
wlsl.setColumnName(data_loader.get_column_names())
qq = 9.5