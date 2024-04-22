import numpy as np
import csv
from DataLoader import DataLoader

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

inputs = np.array([20,	-19,	-18,	20,	20,	20,	-20,	3611])
#inputs = np.array([20,	-19,	2,	20,	20,	-20,	-20,	633])
np.set_printoptions(threshold=np.inf, linewidth=np.inf)
file_names = ['outputApple100\\apple_quality_class_0_edu_20240418154808.csv','outputApple100\\apple_quality_class_1_edu_20240418154808.csv']
data_loader = DataLoader(file_names)
data_loader.load_data()


# Как читать JSON данные через DataLoader 
data_loader.load_json_data('output\\weights_loan_sanction_test_20240422124652.json')
neurons_data = data_loader.neurons  # Получаем данные о нейронах
for neuron in neurons_data:
    print(f"Нейрон #{neuron['neuron_number']}:")
    print(f"  Время выполнения: {neuron['time_elapsed_seconds']} секунд")
    print(f"  Порог слева: {neuron['threshold_left']}")
    print(f"  Порог справа: {neuron['threshold_right']}")
    print(f"  Категория слева: {neuron['category_left']}")
    print(f"  Категория справа: {neuron['category_right']}")
    print(f"  Веса: {neuron['previous_weights']}")  # предполагается, что веса уже в формате строки
    print(f"  Отсечения слева: {neuron['cut_off_left']}, экземпляров слева: {neuron['instances_left']}")
    print(f"  Отсечения справа: {neuron['cut_off_right']}, экземпляров справа: {neuron['instances_right']}")


qq = 0
while qq < 200:
	data_loader.arg_classes[1][qq][-1] = 1
	qq += 1
ww = 0
while ww < 2:
	qq = 0
	while qq < 200:
		ee = n01.digit3activationFunction(data_loader.arg_classes[ww][qq][:7])
		if ee == -1: #Проверка на принадлежность к классу
			if data_loader.arg_classes[ww][qq][-1] != 1: 
				print('error in 1 neurons, instance ', data_loader.arg_classes[ww][qq][-3])
			qq += 1
			continue
		elif ee == 1:
			if data_loader.arg_classes[ww][qq][-1] != 0:
				print('error in 1 neurons, instance ', data_loader.arg_classes[ww][qq][-3])
			qq += 1
			continue
		ee = n02.digit3activationFunction(data_loader.arg_classes[ww][qq][:7])
		if ee == -1:
			if data_loader.arg_classes[ww][qq][-1] != 1: 
				print('error in 2 neurons, instance ', data_loader.arg_classes[ww][qq][-3])
			qq += 1
			continue
		elif ee == 1:
			if data_loader.arg_classes[ww][qq][-1] != 0: 
				print('error in 2 neurons, instance ', data_loader.arg_classes[ww][qq][-3])
			qq += 1
			continue
		ee = n03.digit3activationFunction(data_loader.arg_classes[ww][qq][:7])
		if ee == -1:
			if data_loader.arg_classes[ww][qq][-1] != 1: 
				print('error in 3 neurons, instance ', data_loader.arg_classes[ww][qq][-3])
			qq += 1
			continue
		elif ee == 1:
			if data_loader.arg_classes[ww][qq][-1] != 0: 
				print('error in 3 neurons, instance ', data_loader.arg_classes[ww][qq][-3])
			qq += 1
			continue
		ee = n04.digit3activationFunction(data_loader.arg_classes[ww][qq][:7])
		if ee == -1:
			if data_loader.arg_classes[ww][qq][-1] != 1: 
				print('error in 4 neurons, instance ', data_loader.arg_classes[ww][qq][-3])
			qq += 1
			continue
		elif ee == 1:
			if data_loader.arg_classes[ww][qq][-1] != 0: 
				print('error in 4 neurons, instance ', data_loader.arg_classes[ww][qq][-3])
			qq += 1
			continue
		ee = n05.digit3activationFunction(data_loader.arg_classes[ww][qq][:7])
		if ee == -1:
			if data_loader.arg_classes[ww][qq][-1] != 1: 
				print('error in 5 neurons, instance ', data_loader.arg_classes[ww][qq][-3])
			qq += 1
			continue
		elif ee == 1:
			if data_loader.arg_classes[ww][qq][-1] != 0: 
				print('error in 5 neurons, instance ', data_loader.arg_classes[ww][qq][-3])
			qq += 1
			continue
		ee = n06.digit3activationFunction(data_loader.arg_classes[ww][qq][:7])
		if ee == -1:
			if data_loader.arg_classes[ww][qq][-1] != 1: 
				print('error in 6 neurons, instance ', data_loader.arg_classes[ww][qq][-3])
			qq += 1
			continue
		elif ee == 1:
			if data_loader.arg_classes[ww][qq][-1] != 0: 
				print('error in 6 neurons, instance ', data_loader.arg_classes[ww][qq][-3])
			qq += 1
			continue
		qq += 1
	print(' ')
	ww += 1
qq = 9


