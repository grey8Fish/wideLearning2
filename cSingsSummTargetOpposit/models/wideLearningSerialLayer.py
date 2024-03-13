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
	#Вернуть 3-х мерную матрицу экземпляров обучающей выборки
	def getInputsClassTraining(self):
		return self.inputsClassTraining
	#Вернуть вектор с текущим количеством экземпляров в каждом классе	
	def getCountInstancesEachClassTraining(self):
		return self.countInstancesEachClassTraining
	#Инициализировать список с именами столбцов
	def setColumnName(self, columName):
		self.columnName = columName 
	#Вернуть список с именами столбцов
	def getColumnName(self):
		return self.columnName		
	#Инициализировать список с именами классов
	def setClassesName(self, classesName):
		self.classesName = classesName	
	#Вернуть список с именами классов
	def getClassesName(self):
		return self.classesName		
	#Инициализировать столбец «значение скалярного произведения»
	def initColScalarMul(self, curWeights):
		yy = 0
		while yy < self.countClasses:
			uu = 0 
			while uu < self.countInstancesEachClassTraining[yy]:
				self.inputsClassTraining[yy][uu][self.sizeVector+1] = np.dot(self.inputsClassTraining[yy, uu, :self.sizeVector], curWeights)
				uu += 1
			yy += 1
	#Обнулить столбец «значение скалярного произведения»
	def zerosColScalarMul(self):
		yy = 0
		while yy < self.countClasses:
			uu = 0 
			while uu < self.countInstancesEachClassTraining[yy]:
				self.inputsClassTraining[yy][uu][self.sizeVector+1] = 0
				uu += 1
			yy += 1
	#Определить целевую и противоположную категории
	def getMinMaxScalarMul(self):
		iiMin = self.inputsClassTraining[0][0][self.sizeVector+1]
		opMin = 0
		iiMax = self.inputsClassTraining[0][0][self.sizeVector+1]
		taMax = 0
		yy = 0
		while yy < self.countClasses:
			uu = 0 
			while uu < self.countInstancesEachClassTraining[yy]:
				if iiMin > self.inputsClassTraining[yy][uu][self.sizeVector+1]:
					iiMin = self.inputsClassTraining[yy][uu][self.sizeVector+1]
					opMin = yy
				if iiMax < self.inputsClassTraining[yy][uu][self.sizeVector+1]:
					iiMax = self.inputsClassTraining[yy][uu][self.sizeVector+1]
					taMax = yy
				uu += 1
			yy += 1
		return opMin, taMax
	#Определить максимальное значение скалярного произведения в не целевых категориях
	def calcNoTarMax(self, tarCate):
		noTarMax = 0
		yy = 0
		while yy < self.countClasses:
			if yy == tarCate:
				yy += 1
				continue
			uu = 0 
			while uu < self.countInstancesEachClassTraining[yy]:
				if noTarMax < self.inputsClassTraining[yy][uu][self.sizeVector+1]:
					noTarMax = self.inputsClassTraining[yy][uu][self.sizeVector+1]
				uu += 1
			yy += 1
		return noTarMax
	#Определить минимальное значение скалярного произведения в не противоположных категориях
	def calcNoOppMin(self, oppCate):
		noOppMin = 0
		yy = 0
		while yy < self.countClasses:
			if yy == oppCate:
				yy += 1
				continue
			uu = 0 
			while uu < self.countInstancesEachClassTraining[yy]:
				if noOppMin > self.inputsClassTraining[yy][uu][self.sizeVector+1]:
					noOppMin = self.inputsClassTraining[yy][uu][self.sizeVector+1]
				uu += 1
			yy += 1
		return noOppMin
	#Обнулить столбец «признак отсеченности»
	def zerosColCutOffSign(self):
		yy = 0
		while yy < self.countClasses:
			uu = 0 
			while uu < self.countInstancesEachClassTraining[yy]:
				self.inputsClassTraining[yy][uu][self.sizeVector+2] = 0
				uu += 1
			yy += 1
	#Установить в 1 столбец «признак отсеченности» в целевой категории
	def setColCutOffSignTarget(self, tCat, noTaMax):
		uu = 0
		while uu < self.countInstancesEachClassTraining[tCat]:
			if noTaMax < self.inputsClassTraining[tCat][uu][self.sizeVector+1]:
				self.inputsClassTraining[tCat][uu][self.sizeVector+2] = 1
	#Определить количество отсечённых экземпляров в целевой категории
	def calcCutOffSignTarget(self, tCat, noTaMax):
		yy = 0
		uu = 0
		while uu < self.countInstancesEachClassTraining[tCat]:
			if noTaMax < self.inputsClassTraining[tCat][uu][self.sizeVector+1]:
				yy += 1
			uu += 1
		return yy
	#Определить количество отсечённых экземпляров в противоположной категории
	def calcCutOffSignOpposit(self, tOpp, noOpMin):
		yy = 0
		uu = 0
		while uu < self.countInstancesEachClassTraining[tOpp]:
			if noOpMin > self.inputsClassTraining[tOpp][uu][self.sizeVector+1]:
				yy += 1
			uu += 1
		return yy
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
countCutOffPrev = 0
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
				#Инициализировать столбец «значение скалярного произведения»
				wlsl.initColScalarMul(wlsl.currentWeights)
				#Определить целевую и противоположную категории
				mm = wlsl.getMinMaxScalarMul()
				opCat = int(mm[0])
				taCat = int(mm[1])
				#Определить максимальное значение скалярного произведения в не целевых категориях
				noTargMax = wlsl.calcNoTarMax(taCat)
				#Определить количество отсечённых экземпляров в целевой категории
				countCutOffTarget = wlsl.calcCutOffSignTarget(taCat, noTargMax)
				#Определить минимальное значение скалярного произведения в не противоположных категориях
				noOppoMin = wlsl.calcNoOppMin(opCat)
				#Определить количество отсечённых экземпляров в противоположной категории
				countCufOffOpposit = wlsl.calcCutOffSignOpposit(opCat, noOppoMin)
				countCutOffCurrent = countCufOffOpposit#countCutOffTarget# +  + 
				if countCutOffPrev < countCutOffCurrent:
					countCutOffPrev = countCutOffCurrent
					wlsl.previousWeights = wlsl.currentWeights.copy()
				rr += 1
				#ff += 1
			ee += 1
		ww += 1
	qq += 1
				#tt = 0

#print(wlsl.getInputsClassTraining())
#print(wlsl.getCountInstancesEachClassTraining())
qq = 9.5