import numpy as np
import csv
from models.DataLoader import DataLoader

class wideLearningSerialLayer:
	def __init__(self, coClasses, maxInstance, siVector, nameFile):	#количество классов, максимальное количество экземпляров, количество столбцов=размер вектора весов, имя файла
		self.countClasses = coClasses	#количество классов
		self.sizeVector = siVector		#длина вектора весов / количество столбцов выборки
		self.currentWeights = np.zeros(siVector, dtype=int)		#текущий вектор весов
		self.previousWeights = np.zeros(siVector, dtype=int)		#предыдущий вектор весов
		#self.bestWeights = np.zeros(10, siVector+1, dtype=int)	#10 лучших весов, в последнем столбце количество отсеченных
		self.countInstancesEachClassCorrection = np.zeros(coClasses, dtype=int)	#количество экземпляров в каждом классе корректирующей выборки
		self.countInstancesEachClassTraining = np.zeros(coClasses, dtype=int)	#количество экземпляров в каждом классе обучающей выборки
		self.columnName = None		#имена столбцов
		self.classesName = None		#имена классов
		self.inputsClassCorrection = np.zeros((coClasses, maxInstance, siVector+2), dtype=int)	#входные экземпляры корректирующей выборки
		self.inputsClassTraining = None #np.zeros((coClasses, maxInstance, siVector+2), dtype=int)		#входные экземпляры обучающей выборки

	def getInputsClassTraining(self):
		return self.inputsClassTraining
	
	def getCountInstancesEachClassTraining(self):
		return self.countInstancesEachClassTraining

	def setColumnName(self, columName):
		self.columnName = columName #имена столбцов

	def getColumnName(self):
		return self.columnName		#имена столбцов

	def setClassesName(self, classesName):
		self.classesName = classesName	#имена классов

	def getClassesName(self):
		return self.classesName			#имена классов

	def calcScalarMul(self, curWeights):
		yy = 0
		while yy < self.countClasses:
			uu = 0 
			while uu < self.countInstancesEachClassTraining[uu]:
				self.inputsClassTraining[yy][uu][self.sizeVector+1] = np.dot(self.inputsClassTraining[yy, uu, :self.sizeVector], curWeights)
			uu += 1
		yy += 1

	def getMinMaxScalarMul(self):
		iiMin = self.inputsClassTraining[0][0][self.sizeVector+1]
		opMin = -1
		iiMax = self.inputsClassTraining[0][0][self.sizeVector+1]
		taMax = -1
		yy = 0
		while yy < self.countClasses:
			uu = 0 
			while uu < self.countInstancesEachClassTraining[uu]:
				if iiMin > self.inputsClassTraining[yy][uu][self.sizeVector+1]:
					iiMin = self.inputsClassTraining[yy][uu][self.sizeVector+1]
					opmin = yy
				if iiMax < self.inputsClassTraining[yy][uu][self.sizeVector+1]:
					iiMax = self.inputsClassTraining[yy][uu][self.sizeVector+1]
					taMax = yy
			uu += 1
		yy += 1
		return opMin, taMax

file_names = ['seed0_23_11_26.csv', 'seed1_23_11_26.csv', 'seed2_23_11_26.csv']#, 'cirrhosis_4.0_part0_20240301100740.csv']
data_loader = DataLoader(file_names)
data_loader.load_data()

wlsl = wideLearningSerialLayer(data_loader.classes_count, data_loader.instances_max, data_loader.ordinate_count-1, 'fileNameTmp')
wlsl.setColumnName(data_loader.get_column_names())
wlsl.setClassesName(data_loader.get_class_names())
wlsl.inputsClassTraining = data_loader.get_data().copy()
wlsl.countInstancesEachClassTraining = data_loader.get_max_instances_nparray().copy()
#wlsl.countInstancesEachClassCorrection = data_loader.get_max_instances_nparray().copy()
#ff = 0
qq = 0
while qq < wlsl.countClasses-1:
	ww = 0
	while ww < wlsl.countInstancesEachClassTraining[qq]:
		ee = qq + 1
		while ee < wlsl.countClasses:
			rr = 0
			while rr < wlsl.countInstancesEachClassTraining[ee]:
				tt = 0 
				while tt < wlsl.sizeVector:
					#первоначальное приближение вектора весов 
					wlsl.currentWeights[tt] = wlsl.inputsClassTraining[qq][ww][tt] - wlsl.inputsClassTraining[ee][rr][tt]
					tt += 1
				#
				#aa = np.dot(wlsl.inputsClassTraining[qq, ww, :wlsl.sizeVector], wlsl.currentWeights)
				wlsl.calcScalarMul(wlsl.currentWeights)
				rr += 1
				#ff += 1
			ee += 1
		ww += 1
	qq += 1
				#tt = 0

#print(wlsl.getInputsClassTraining())
#print(wlsl.getCountInstancesEachClassTraining())
qq = 9.5