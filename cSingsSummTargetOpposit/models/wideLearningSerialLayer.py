#import time #отказываемся в пользу datetime
from datetime import datetime
from xml.etree.ElementTree import tostring
#import sys
import numpy as np
import csv
from pandas import concat
from DataLoader import DataLoader
import json
import os
import time
import cProfile, pstats
import pandas as pd
import openpyxl  
from pandas import ExcelWriter
import traceback


def convert_prof(profile_path, format='xlsx', delete_original=False):
    """
    Конвертирует .prof файл в формат JSON, CSV или XLSX.

    :param profile_path: Путь к .prof файлу
    :param format: Формат вывода ('json', 'csv', 'xlsx')
    """
    # Загрузка статистики из файла профилирования
    stats = pstats.Stats(profile_path)
    stats.strip_dirs().sort_stats('tottime')  # Сортировка по убыванию общего времени

    # Подготовка данных для вывода
    data = []
    for func, (cc, nc, tt, ct, callers) in stats.stats.items():
        data.append({
            'Call Count': cc,
            'Native Calls': nc,
            'Total Time': tt,
            'Cumulative Time': ct,
            'Function': func,
            'Callers': {str(k): v for k, v in callers.items()}
        })

    # Конвертация данных в DataFrame для удобства экспорта
    df = pd.DataFrame(data)
    df.sort_values(by='Total Time', ascending=False, inplace=True)  # Сортировка данных

    # Выбор формата вывода
    if format == 'json':
        output_path = profile_path.replace('.prof', '.json')
        df.to_json(output_path, orient='records', indent=4)
    elif format == 'csv':
        output_path = profile_path.replace('.prof', '.csv')
        df.to_csv(output_path, index=False)
    elif format == 'xlsx':
        output_path = profile_path.replace('.prof', '.xlsx')
        with ExcelWriter(output_path, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Sheet1')
            workbook = writer.book
            worksheet = writer.sheets['Sheet1']
            # Настройка ширины столбцов
            for column in df:
                column_width = max(df[column].astype(str).map(len).max(), len(column))
                col_idx = df.columns.get_loc(column)
                worksheet.set_column(col_idx, col_idx, column_width)
            # Добавление фильтра
            worksheet.autofilter(0, 0, 0, len(df.columns) - 1)
			
    if delete_original:
        os.remove(profile_path)
        #print(f"Оригинальный файл профайлинга '{profile_path}' был удалён.")

    print(f"Профайлинг '{output_path}' успешно сохранён.")


class wideLearningSerialLayer:
	def __init__(self, coClasses, maxInst, siVector, nameFile, timestamp, common_path):	#количество классов, максимальное количество экземпляров, количество столбцов=размер вектора весов, имя файла
		self.maxInstance = maxInst		#Удвоенное максимальное количество экземпляров выборки
		self.countClasses = coClasses	#количество классов
		self.sizeVector = siVector		#длина вектора весов / количество столбцов выборки
		self.currentWeightsInit = np.zeros(siVector, dtype=int)		#текущий вектор весов первоначальный
		self.previousWeightsInit = np.zeros(siVector, dtype=int)	#предыдущий вектор весов первоначальный
		self.currentWeightsRefined = np.zeros(siVector, dtype=int)		#текущий вектор весов уточненный
		self.previousWeightsRefined = np.zeros(siVector, dtype=int)	#предыдущий вектор весов уточненный
		self.numberBest =			10
		self.bestWeights = np.zeros((10, siVector+9), dtype=int)	#10 лучших весов, в последнем столбце количество отсеченных
		self.countInstancesEachClassCorrection = np.zeros(coClasses, dtype=int)	#количество экземпляров в каждом классе корректирующей выборки
		self.countInstancesEachClassTraining = np.zeros(coClasses, dtype=int)	#количество экземпляров в каждом классе обучающей выборки
		self.columnName = None		#имена столбцов
		self.classesName = None		#имена классов
		self.inputsClassCorrection = np.zeros((coClasses, maxInst, siVector+2), dtype=int)	#входные экземпляры корректирующей выборки
		self.inputsClassTraining = np.zeros((coClasses, maxInst, siVector+2), dtype=int)	#входные экземпляры обучающей выборки
		self.vectorDeltasCurr = np.zeros(siVector, dtype=int)#вектора поправок весов 
		self.vectorDeltasPrev = np.zeros(siVector, dtype=int)#в процедурах уточнения
		self.json_best_weights_path = os.path.join(common_path, f'weights_best_{timestamp}.json')
		self.best_weights_data = []  
	

	#Обнуление массива лучших весов
	def zeroingBestWeights(self):
		#сохранение перед обнулением, если не нужно - комментируем блок
		weights_as_lines = [" , ".join(map(str, line)) for line in self.bestWeights]# Сериализация весов в одну строку
		output_data = {
            "timestamp": datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
            "neuron_number": len(self.best_weights_data),  
            "bestWeights": weights_as_lines
        }
		self.best_weights_data.append(output_data)
		with open(self.json_best_weights_path, 'w') as f:
			json.dump(self.best_weights_data, f, indent=4)
		
		#обнуление
		self.bestWeights.fill(0)
		
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
		for yy in range(self.countClasses):
			self.inputsClassTraining[yy, :, self.sizeVector+1] = np.dot(self.inputsClassTraining[yy, :, :self.sizeVector], curWeights)
	#Обнуляет любой заданный столбец для всех классов
	def zero_column(self, column_index):
		self.inputsClassTraining[:, :, column_index] = 0
	#Обнулить столбец «значение скалярного произведения»
	def zerosColScalarMul(self):
		#self.zero_column(self.sizeVector + 1)
		yy = 0
		while yy < self.countClasses:
			uu = 0 
			while uu < self.countInstancesEachClassTraining[yy]:
				self.inputsClassTraining[yy][uu][self.sizeVector+1] = 0
				uu += 1
			yy += 1
			
	#Определить целевую и противоположную категории
	#def getMinMaxScalarMul_numpy(self):
	#	min_value = np.inf
	#	max_value = -np.inf
	#	op_min_class = None
	#	ta_max_class = None
	#
	#	for class_index in range(self.countClasses):
	#		class_data = self.inputsClassTraining[class_index, :self.countInstancesEachClassTraining[class_index], self.sizeVector+1]
	#		if class_data.size == 0:
	#			continue
	#
	#		local_min = np.min(class_data)
	#		local_max = np.max(class_data)
	#
	#		if local_min < min_value:
	#			min_value = local_min
	#			op_min_class = class_index
	#
	#		if local_max > max_value:
	#			max_value = local_max
	#			ta_max_class = class_index
	#
	#	return op_min_class, ta_max_class
		
	#Определить целевую и противоположную категории - старая версия
	def getMinMaxScalarMul(self):
		yy = 0
		while yy < self.countClasses:
			if self.countInstancesEachClassTraining[yy] == 0:
				yy += 1
				continue
			iiMin = self.inputsClassTraining[yy][0][self.sizeVector+1]
			opMin = yy
			iiMax = self.inputsClassTraining[yy][0][self.sizeVector+1]
			taMax = yy
			break
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
		noTarMax = 0#Если все экземпляры уже отсечены
		yy = 0
		while yy < self.countClasses:
			if yy == tarCate:
				yy += 1
				continue
			if self.countInstancesEachClassTraining[yy] != 0:
				noTarMax = self.inputsClassTraining[yy][0][self.sizeVector+1]
			yy += 1
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
		noOppMin = 0#Если все экземпляры уже отсечены
		yy = 0
		while yy < self.countClasses:
			if yy == oppCate:
				yy += 1
				continue
			if self.countInstancesEachClassTraining[yy] != 0:
				noOppMin = self.inputsClassTraining[yy][0][self.sizeVector+1]
			yy += 1
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
		#self.zero_column(self.sizeVector + 2)
		yy = 0
		while yy < self.countClasses:
			uu = 0 
			while uu < self.countInstancesEachClassTraining[yy]:
				self.inputsClassTraining[yy][uu][self.sizeVector+2] = 0
				uu += 1
			yy += 1
		
	#Установить в 1 столбец «признак отсеченности» в целевой категории и вернуть значение порога справа.
	def setColCutOffSignTarget(self, tCat, noTaMax):
		behindWall = 0
		uu = 0
		while uu < self.countInstancesEachClassTraining[tCat]:
			if noTaMax < self.inputsClassTraining[tCat][uu][self.sizeVector+1]:
				self.inputsClassTraining[tCat][uu][self.sizeVector+2] = 1
			else:
				self.inputsClassTraining[tCat][uu][self.sizeVector+2] = 0
			uu += 1
		while uu < self.maxInstance:
			self.inputsClassTraining[tCat][uu][self.sizeVector+2] = 3
			uu += 1
		uu = 0
		while uu < self.countInstancesEachClassTraining[tCat]:
			if self.inputsClassTraining[tCat][uu][self.sizeVector+2] == 1:# and (behindWall > self.inputsClassTraining[tCat][uu][self.sizeVector+1]):
				behindWall = self.inputsClassTraining[tCat][uu][self.sizeVector+1]
				break
			uu += 1
		while uu < self.countInstancesEachClassTraining[tCat]:
			if (self.inputsClassTraining[tCat][uu][self.sizeVector+2] == 1) and (behindWall > self.inputsClassTraining[tCat][uu][self.sizeVector+1]):
				behindWall = self.inputsClassTraining[tCat][uu][self.sizeVector+1]
			uu += 1
		return behindWall
	#Определить количество отсечённых экземпляров в целевой категории
	def calcCutOffSignTarget(self, tCat, noTaMax):
		yy = 0
		uu = 0
		while uu < self.countInstancesEachClassTraining[tCat]:
			if noTaMax < self.inputsClassTraining[tCat][uu][self.sizeVector+1]:
				yy += 1
			uu += 1
	    #scalar_products = self.inputsClassTraining[tCat, :self.countInstancesEachClassTraining[tCat],
        #                  self.sizeVector + 1]
        #cut_off_count = np.sum(scalar_products > noTaMax)
		return yy
	#Установить в 2 столбец «признак отсеченности» в противоположной категории и вернуть значение порога слева.
	def setColCutOffSignOpposit(self, oCat, noOpMin):
		behindWall = 0
		uu = 0
		while uu < self.countInstancesEachClassTraining[oCat]:
			if noOpMin > self.inputsClassTraining[oCat][uu][self.sizeVector+1]:
				self.inputsClassTraining[oCat][uu][self.sizeVector+2] = 2
			else:
				self.inputsClassTraining[oCat][uu][self.sizeVector+2] = 0
			uu += 1
		while uu < self.maxInstance:
			self.inputsClassTraining[oCat][uu][self.sizeVector+2] = 3
			uu +=1
		uu = 0
		while uu < self.countInstancesEachClassTraining[oCat]:
			if self.inputsClassTraining[oCat][uu][self.sizeVector+2] == 2:# and (behindWall < self.inputsClassTraining[oCat][uu][self.sizeVector+1]):
				behindWall = self.inputsClassTraining[oCat][uu][self.sizeVector+1]
				break
			uu += 1
		while uu < self.countInstancesEachClassTraining[oCat]:
			if (self.inputsClassTraining[oCat][uu][self.sizeVector+2] == 2) and (behindWall < self.inputsClassTraining[oCat][uu][self.sizeVector+1]):
				behindWall = self.inputsClassTraining[oCat][uu][self.sizeVector+1]
			uu += 1
		return behindWall
	#Определить количество отсечённых экземпляров в противоположной категории
	def calcCutOffSignOpposit(self, tOpp, noOpMin):
		yy = 0
		uu = 0
		while uu < self.countInstancesEachClassTraining[tOpp]:
			if noOpMin > self.inputsClassTraining[tOpp][uu][self.sizeVector+1]:
				yy += 1
			uu += 1
		return yy
	#Сортировать указанную категорию по возрастанию «признака отсечённости» 
	def sortCategoryCutOff(self, curCat):
		'''a = self.inputsClassTraining[curCat] 
		a = a[a[:,-1].argsort()]
		print(a)'''
		self.inputsClassTraining[curCat] = self.inputsClassTraining[curCat][self.inputsClassTraining[curCat][:,-1].argsort()]
	#Контрастирование весов. Делит веса на 2 без изменения количества отсеченных.
	#здесь надо избавиться от wlsl в коде
	#def contrastingWeights(self, oppoCat, targCat, countCutOffOpposit, countCutOffTarget):
	#	countCutOffPrev = countCutOffOpposit + countCutOffTarget
	#	qq = 0
	#	while qq < 1:
	#		self.currentWeightsInit //= 2
	#		nTargMax = self.calcNoTarMax(targCat)                    #расчет количества отсеченных
	#		countCutOffTarg = self.calcCutOffSignTarget(targCat, nTargMax)
	#		nOppoMin = self.calcNoOppMin(oppoCat)
	#		countCutOffOppo = self.calcCutOffSignOpposit(oppoCat, nOppoMin)
	#		countCutOffCurr = countCutOffTarg + countCutOffOppo#расчет количества отсеченных
	#		if countCutOffCurr == countCutOffPrev:
	#			wlsl.previousWeightsInit = wlsl.currentWeightsInit.copy()
	#		else:
	#			wlsl.currentWeightsInit = wlsl.previousWeightsInit.copy()
	#			break
	#		qq += 1
	#	countCutOffPrev = 9
	#Градиентный спуск сканированием, наивный вариант
	def  gradientDescentScanning(self, oppoCat, noOppoMin, targCat, noTargMax, cutOffOppo, cutOffTarg):
		oppoMax = self.setColCutOffSignOpposit(oppoCat, noOppoMin)
		targMin = self.setColCutOffSignTarget(targCat, noTargMax)
		if((noOppoMin - oppoMax) < (targMin - noTargMax)):
			distMinPrev = noOppoMin - oppoMax
		else:
			distMinPrev = targMin - noTargMax
		distMinCurr = distMinPrev
		thresholdOppo = (noOppoMin // 2) + (oppoMax // 2)
		thresholdTarg = (targMin //2 ) + (noTargMax // 2)
		self.previousWeightsRefined = self.previousWeightsInit.copy()
		stepSize = 1
		cuOffOppo = cutOffOppo
		cuOffTarg = cutOffTarg
		countCutOffCurr = countCutOffPrev = cutOffOppo + cutOffTarg
		while True:
			qq = 0
			while qq < self.sizeVector:#шаг в плюс
				self.vectorDeltasCurr -= self.vectorDeltasCurr
				self.vectorDeltasCurr[qq] += stepSize
				self.currentWeightsRefined = self.previousWeightsRefined.copy()
				self.currentWeightsRefined += self.vectorDeltasCurr
				self.initColScalarMul(self.currentWeightsRefined)			
				nTargMax = self.calcNoTarMax(targCat)					#расчет количества отсеченных
				countCutOffTarget = self.calcCutOffSignTarget(targCat, nTargMax)
				nOppoMin = self.calcNoOppMin(oppoCat)
				countCufOffOpposit = self.calcCutOffSignOpposit(oppoCat, nOppoMin)
				countCutOffCurr = countCutOffTarget + countCufOffOpposit#расчет количества отсеченных
				oppoMax = self.setColCutOffSignOpposit(oppoCat, nOppoMin)	#расчет расстояния
				targMin = self.setColCutOffSignTarget(targCat, nTargMax)
				if((nOppoMin - oppoMax) < (targMin - nTargMax)):
					distMinCurr = nOppoMin - oppoMax
				else:
					distMinCurr = targMin - nTargMax						#расчет расстояния
				if (countCutOffPrev < countCutOffCurr): #количествo увеличилось
					countCutOffPrev = countCutOffCurr
					cuOffOppo = countCufOffOpposit
					cuOffTarg = countCutOffTarget
					distMinPrev = distMinCurr
					thresholdOppo = (nOppoMin //2) + (oppoMax // 2)
					thresholdTarg = (targMin // 2) + (nTargMax // 2)
				elif countCutOffPrev > countCutOffCurr:	#количествo уменьшилось
					self.vectorDeltasCurr[qq] -= stepSize
				elif distMinPrev > distMinCurr:		#количествo не изменилось, расстояние уменьшилось
					distMinPrev = distMinCurr
					thresholdOppo = (nOppoMin //2) + (oppoMax // 2)
					thresholdTarg = (targMin // 2) + (nTargMax // 2)
				else:								#расстояние увеличилось
					self.vectorDeltasCurr[qq] -= stepSize 
				self.vectorDeltasPrev += self.vectorDeltasCurr
				qq += 1
			qq = 0
			while qq < self.sizeVector:#шаг в минус
				self.vectorDeltasCurr -= self.vectorDeltasCurr
				self.vectorDeltasCurr[qq] -= stepSize
				self.currentWeightsRefined = self.previousWeightsRefined.copy()
				self.currentWeightsRefined += self.vectorDeltasCurr
				self.initColScalarMul(self.currentWeightsRefined)			
				nTargMax = self.calcNoTarMax(targCat)					#расчет количества отсеченных
				countCutOffTarget = self.calcCutOffSignTarget(targCat, nTargMax)
				nOppoMin = self.calcNoOppMin(oppoCat)
				countCufOffOpposit = self.calcCutOffSignOpposit(oppoCat, nOppoMin)
				countCutOffCurr = countCutOffTarget + countCufOffOpposit#расчет количества отсеченных
				oppoMax = self.setColCutOffSignOpposit(oppoCat, nOppoMin)	#расчет расстояния
				targMin = self.setColCutOffSignTarget(targCat, nTargMax)
				if((nOppoMin - oppoMax) < (targMin - nTargMax)):
					distMinCurr = nOppoMin - oppoMax
				else:
					distMinCurr = targMin - nTargMax						#расчет расстояния
				if (countCutOffPrev < countCutOffCurr): #количествo увеличилось
					countCutOffPrev = countCutOffCurr
					cuOffOppo = countCufOffOpposit
					cuOffTarg = countCutOffTarget
					distMinPrev = distMinCurr
					thresholdOppo = (nOppoMin // 2) + (oppoMax // 2)
					thresholdTarg = (targMin // 2) + (nTargMax // 2)
				elif countCutOffPrev > countCutOffCurr:	#количествo уменьшилось
					self.vectorDeltasCurr[qq] += stepSize
				elif distMinPrev > distMinCurr:		#количествo не изменилось, расстояние уменьшилось
					distMinPrev = distMinCurr
					thresholdOppo = (nOppoMin // 2) + (oppoMax // 2)
					thresholdTarg = (targMin //2 ) + (nTargMax // 2)
				else:								#расстояние увеличилось
					self.vectorDeltasCurr[qq] += stepSize 
				self.vectorDeltasPrev += self.vectorDeltasCurr
				qq += 1
			qq = 0
			ww = 0
			while qq < self.sizeVector:
				ww += abs(self.vectorDeltasPrev[qq])
				qq += 1
			if ww == 0:
				break
			self.previousWeightsRefined += self.vectorDeltasPrev
			self.vectorDeltasPrev -= self.vectorDeltasPrev
			stepSize *= 2
		qq = 1
		ww = 0
		minEE = self.bestWeights[ww][-1]	#
		while qq < self.numberBest:
			if minEE > self.bestWeights[qq][-1]: 
				minEE = self.bestWeights[qq][-1]
				ww = qq
			qq += 1
		self.initColScalarMul(self.previousWeightsRefined)		
		bb = self.getMinMaxScalarMul()
		oppoCat = int(bb[0])
		targCat = int(bb[1])
		nTargMax = self.calcNoTarMax(targCat)					#расчет количества отсеченных
		countCutOffTarget = self.calcCutOffSignTarget(targCat, nTargMax)
		nOppoMin = self.calcNoOppMin(oppoCat)
		countCufOffOpposit = self.calcCutOffSignOpposit(oppoCat, nOppoMin)
		countCutOffCurr = countCutOffTarget + countCufOffOpposit#расчет количества отсеченных
		oppoMax = self.setColCutOffSignOpposit(oppoCat, nOppoMin)	#расчет расстояния
		targMin = self.setColCutOffSignTarget(targCat, nTargMax)
		thresholdOppo = (nOppoMin // 2) + (oppoMax // 2)
		thresholdTarg = (targMin // 2) + (nTargMax // 2)

		self.bestWeights[ww][:self.sizeVector] = self.previousWeightsRefined.copy()
		self.bestWeights[ww][-1] = countCutOffCurr	#Сумма отсеченных -1
		self.bestWeights[ww][-2] = self.countInstancesEachClassTraining[targCat]	#Всего справа -2
		self.bestWeights[ww][-3] = countCutOffTarget 	#Отсеченных справа -3
		self.bestWeights[ww][-4] = thresholdTarg	#Правый порог -4
		self.bestWeights[ww][-5] = targCat	#Правый класс -5
		self.bestWeights[ww][-6] = self.countInstancesEachClassTraining[oppoCat]	#Всего слева -6
		self.bestWeights[ww][-7] = countCufOffOpposit 	#Отсеченных слева -7
		self.bestWeights[ww][-8] = thresholdOppo	#Левый порог -8
		self.bestWeights[ww][-9] = oppoCat	#Левый класс -9
		
def main(file_names):
	try:
		start_time = datetime.now()
		np.set_printoptions(threshold=np.inf, linewidth=np.inf)
		data_loader = DataLoader(file_names)
		data_loader.load_data() 

		#Для JSON
		output = []
		timestamp = data_loader.get_timestamp()
		# Убедимся, что каталог для сохранения файлов существует
		output_directory = 'output'
		if not os.path.exists(output_directory):
			os.makedirs(output_directory)


		#wlsl = wideLearningSerialLayer(data_loader.classes_count, data_loader.instances_max, data_loader.ordinate_count-1, 'fileNameTmp')
		timestamp = data_loader.get_timestamp()	
		common_path = os.path.dirname(file_names[0])
		wlsl = wideLearningSerialLayer(data_loader.classes_count, data_loader.instances_max, data_loader.ordinate_count, 'fileNameTmp', timestamp, common_path)
		wlsl.setColumnName(data_loader.get_column_names())
		wlsl.setClassesName(data_loader.get_class_names())
		wlsl.inputsClassTraining = data_loader.get_data().copy()
		wlsl.countInstancesEachClassTraining = data_loader.get_max_instances_nparray().copy()
		#wlsl.countInstancesEachClassCorrection = data_loader.get_max_instances_nparray().copy()
		base_name = os.path.basename(file_names[0]).split('_class')[0]
		# Извлечение имени первого файла без расширения
		#base_file_name = base_name #os.path.splitext(os.path.basename(file_names[0]))[0]
		# Путь к файлу JSON
		temp_output_file_path = f'{common_path}/weights_{base_name}_{timestamp}.temp.json'
		final_output_file_path = f'{common_path}/weights_{base_name}_{timestamp}.json'

		nn = 2
		neuron_number = 0
		while nn >= 2:
			seconds = datetime.now().timestamp()
			local_time = datetime.fromtimestamp(seconds).strftime('%a %b %d %H:%M:%S %Y')
			formatted_time = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
			print("[", neuron_number, "]", formatted_time) #Время
			neuron_start_time = time.time()  # Начало отсчета времени для нейрона
			countCutOffPrev = 0
			qq = 0
			while qq < wlsl.countClasses:
				ww = 0
				while ww < wlsl.countInstancesEachClassTraining[qq]:
					ee = 0
					while ee < wlsl.sizeVector:#первоначальное приближение вектора весов 
						wlsl.currentWeightsInit[ee] = wlsl.inputsClassTraining[qq][ww][ee]  // 2
						ee += 1
						#Инициализировать столбец «значение скалярного произведения»
					wlsl.initColScalarMul(wlsl.currentWeightsInit)
						#Определить целевую и противоположную категории
					mm = wlsl.getMinMaxScalarMul()
					opCat = int(mm[0])
					taCat = int(mm[1])
						#Определить максимальное значение скалярного произведения в НЕ целевых категориях
					noTargMax = wlsl.calcNoTarMax(taCat)
						#print(noTargMax)
						#Определить количество отсечённых экземпляров в целевой категории
					countCutOffTarget = wlsl.calcCutOffSignTarget(taCat, noTargMax)
						#Определить минимальное значение скалярного произведения в НЕ противоположных категориях
					noOppoMin = wlsl.calcNoOppMin(opCat)
						#print(noOppoMin)
						#Определить количество отсечённых экземпляров в противоположной категории
					countCufOffOpposit = wlsl.calcCutOffSignOpposit(opCat, noOppoMin)
					countCutOffCurrent = countCutOffTarget + countCufOffOpposit
					if countCutOffPrev < countCutOffCurrent:
						countCutOffPrev = countCutOffCurrent
						wlsl.previousWeightsInit = wlsl.currentWeightsInit.copy()
							#countCutOffRight = countCutOffTarget
							#categoryRight = taCat
							#print(noTargMax)
							#maxNoRight = noTargMax
							#countCufOffLeft = countCufOffOpposit
							#categoryLeft = opCat
							#minNoLeft = noOppoMin
						wlsl.gradientDescentScanning(opCat, noOppoMin, taCat, noTargMax, countCufOffOpposit, countCutOffTarget)
		#					rr += 1
							#ff += 1
						#ee += 1
					#print(ww)
					ww += 1
				qq += 1
		
			neuron_end_time = time.time()  # Конец отсчета времени
			time_elapsed = round(neuron_end_time - neuron_start_time, 3)  # Вычисление времени выполнения
	
			qq = 1										#Сумма отсеченных	-1 Всего справа -2
			ww = 0										#Отсеченных справа	-3
			maxEE = wlsl.bestWeights[ww][-1]			#Правый порог		-4
			while qq < wlsl.numberBest:					#Правый класс		-5
				if maxEE < wlsl.bestWeights[qq][-1]:	#Всего слева		-6
					maxEE = wlsl.bestWeights[qq][-1]	#Отсеченных слева	-7
					ww = qq								#Левый порог		-8
				qq += 1									#Левый класс		-9
			print(wlsl.bestWeights[ww][-8], wlsl.bestWeights[ww][-4],sep=', ')
			print(wlsl.classesName[wlsl.bestWeights[ww][-9]], wlsl.classesName[wlsl.bestWeights[ww][-5]], sep=', ')
			print(wlsl.bestWeights[ww][:wlsl.sizeVector], sep=', ')
			print(wlsl.bestWeights[ww][-7],' out of ',wlsl.countInstancesEachClassTraining[wlsl.bestWeights[ww][-9]],'|',wlsl.bestWeights[ww][-3],' out of ',wlsl.countInstancesEachClassTraining[wlsl.bestWeights[ww][-5]])
			print(f"{int(time_elapsed // 3600):02}:{int((time_elapsed % 3600) // 60):02}:{int(time_elapsed % 60):02}")

			weights = wlsl.bestWeights[ww][:wlsl.sizeVector]
			weights_str = ", ".join(map(str, weights))

			#Блок сохранения JSON
			output_data = {
			"neuron_number": neuron_number,
			"time_elapsed_seconds": time_elapsed,
			"timestamp": formatted_time,
			"threshold_left": wlsl.bestWeights[ww][-8],
			"threshold_right": wlsl.bestWeights[ww][-4],
			"category_left": wlsl.classesName[wlsl.bestWeights[ww][-9]],
			"category_right": wlsl.classesName[wlsl.bestWeights[ww][-5]],
			"previous_weights": weights_str, #wlsl.bestWeights[ww][:wlsl.sizeVector].tolist(), # Здесь указывать размер вектора
			"cut_off_left": wlsl.bestWeights[ww][-7],
			"instances_left": wlsl.countInstancesEachClassTraining[wlsl.bestWeights[ww][-9]],
			"cut_off_right": wlsl.bestWeights[ww][-3],
			"instances_right": wlsl.countInstancesEachClassTraining[wlsl.bestWeights[ww][-5]]
			}
			#print(f"Cut off left: {output_data['cut_off_left']} out of {output_data['instances_left']}")
			#print(f"Cut off right: {output_data['cut_off_right']} out of {output_data['instances_right']}\n")
			output.append(output_data)
	
			# Запись данных во временный JSON файл на каждом шаге
			#with open(temp_output_file_path, 'w') as temp_json_file:
			#	json.dump(output, temp_json_file, indent=4, default=lambda x: x.tolist())
			temp_output_file_path = f'{common_path}/weights_{base_name}_{timestamp}_{neuron_number}.temp.json' # Изменено для сохранения всех временных файлов
			with open(temp_output_file_path, 'w') as temp_json_file:
				json.dump(output, temp_json_file, indent=4, default=lambda x: x.tolist())

			# Копирование данных из временного файла в постоянный в конце каждой итерации, закомментировать если нужно сохранять все нейроны 
			os.replace(temp_output_file_path, final_output_file_path)

			#Инициализировать столбец «значение скалярного произведения»
			wlsl.initColScalarMul(wlsl.bestWeights[ww][:wlsl.sizeVector])

			#Определить минимальное значение скалярного произведения в НЕ противоположных категориях
			noOppoMin = wlsl.calcNoOppMin(wlsl.bestWeights[ww][-9])
			#Установить в 2 столбец «признак отсеченности» в противоположной категории и вернуть значение порога слева.
			wlsl.setColCutOffSignOpposit(wlsl.bestWeights[ww][-9], noOppoMin)
			#Сортировать указанную категорию по возрастанию «признака отсечённости»
			wlsl.sortCategoryCutOff(wlsl.bestWeights[ww][-9])
			#Усечение указанных классов
			wlsl.countInstancesEachClassTraining[wlsl.bestWeights[ww][-9]] -= wlsl.bestWeights[ww][-7]

			#Определить максимальное значение скалярного произведения в НЕ целевых категориях
			noTargMax = wlsl.calcNoTarMax(wlsl.bestWeights[ww][-5])
			#Установить в 1 столбец «признак отсеченности» в целевой категории и вернуть значение порога справа.
			wlsl.setColCutOffSignTarget(wlsl.bestWeights[ww][-5], noTargMax)
			#Сортировать указанную категорию по возрастанию «признака отсечённости»
			wlsl.sortCategoryCutOff(wlsl.bestWeights[ww][-5])
			#Усечение указанных классов
			wlsl.countInstancesEachClassTraining[wlsl.bestWeights[ww][-5]] -= wlsl.bestWeights[ww][-3]

			#Обнуление массива лучших весов
			wlsl.zeroingBestWeights()

			nn = 0
			rr = 0
			while rr < wlsl.countClasses:
				if wlsl.countInstancesEachClassTraining[rr] > 0:
					nn += 1
				rr += 1
			neuron_number += 1
			print()

		qq = 9.5
		
	except Exception as e:
		print("Произошла ошибка:", str(e))
		traceback.print_exc()
		
	finally:
		end_time = datetime.now()
		time_delta = end_time - start_time
		total_time = str(time_delta) 
		final_output = {
			"script_name": "wideLearningSerialLayer.py",
			"file_names": file_names,  
			"start_time": start_time.strftime("%d.%m.%Y %H:%M:%S"),  # Время начала работы программы
			"end_time": end_time.strftime("%d.%m.%Y %H:%M:%S"),  # Время окончания работы программы
			"total_time_seconds": total_time,
			"total_neuron_count": neuron_number,
			"neurons": output          # вся информация по нейронам
			}
		with open(final_output_file_path, 'w') as final_json_file:
			json.dump(final_output, final_json_file, indent=4, default=lambda x: x.tolist())
	
	#Вывод итогов выполнения
	print(f"Файл сохранён: {final_output_file_path}")
	print("Время выполнения:", total_time)
	qq = 9

if __name__ == "__main__":
	file_names = [
#		'output\\dataset_kahraman_20240502192205\\kahraman_class_High_edu_20240502192205.csv',
#		'output\\dataset_kahraman_20240502192205\\kahraman_class_Low_edu_20240502192205.csv',
#		'output\\dataset_kahraman_20240502192205\\kahraman_class_Middle_edu_20240502192205.csv',
#		'output\\dataset_kahraman_20240502192205\\kahraman_class_very_low_edu_20240502192205.csv'    
		'output\\dataset_forest_20240503171228\\forest_class_d _edu_20240503171229.csv',
		'output\\dataset_forest_20240503171228\\forest_class_h _edu_20240503171229.csv',
		'output\\dataset_forest_20240503171228\\forest_class_o _edu_20240503171229.csv',
		'output\\dataset_forest_20240503171228\\forest_class_s _edu_20240503171229.csv'
		]

	
	profiler = cProfile.Profile()
	try:
		profiler.enable()  # Начинаем профилирование
		main(file_names)  # Запуск основной функции с передачей имён файлов
	except Exception as e:
		print("Произошла ошибка:", str(e))
		traceback.print_exc()
	finally:
		profiler.disable()  # Завершаем профилирование

		# Сохраняем результаты профилирования
		common_path = os.path.dirname(file_names[0])
		timestamp = datetime.now().strftime("%Y%m%d%H%M%S")  # Сохраняем текущую метку времени
		profile_output_file_path = os.path.join(common_path, f'profiling_wlsl_{timestamp}.prof')
		profiler.dump_stats(profile_output_file_path)
		convert_prof(profile_output_file_path, format='xlsx', delete_original=True)

		print()
		stats = pstats.Stats(profiler)
		#stats.print_stats()
		stats.sort_stats('time').print_stats(10)
"""
	file_names = [
        'outputApple400\\apple_quality_class_0_edu_20240418154718.csv',
        'outputApple400\\apple_quality_class_1_edu_20240418154718.csv'
    ]
	
		file_names = [
		'output\\dataset_Customer_20240425025244\\Customer_class_A_edu_20240425025244.csv',
		'output\\dataset_Customer_20240425025244\\Customer_class_B_edu_20240425025244.csv',
		'output\\dataset_Customer_20240425025244\\Customer_class_C_edu_20240425025244.csv',
		'output\\dataset_Customer_20240425025244\\Customer_class_D_edu_20240425025244.csv'
		]
	
#file_names = ['output\\dataset_Customer_20240425025244\\Customer_class_A_edu_20240425025244.csv','output\\dataset_Customer_20240425025244\\Customer_class_B_edu_20240425025244.csv','output\\dataset_Customer_20240425025244\\Customer_class_C_edu_20240425025244.csv','output\\dataset_Customer_20240425025244\\Customer_class_D_edu_20240425025244.csv']
#file_names = ['output\\dataset_Customer_20240425025744\\Customer_class_A_edu_20240425025744.csv','output\\dataset_Customer_20240425025744\\Customer_class_B_edu_20240425025744.csv','output\\dataset_Customer_20240425025744\\Customer_class_C_edu_20240425025744.csv','output\\dataset_Customer_20240425025744\\Customer_class_D_edu_20240425025744.csv']
#file_names = ['Output\\dataset_bodyPerformance_20240424201824\\bodyPerformance_class_A_edu_20240424201826.csv','Output\\dataset_bodyPerformance_20240424201824\\bodyPerformance_class_B_edu_20240424201826.csv','Output\\dataset_bodyPerformance_20240424201824\\bodyPerformance_class_C_edu_20240424201826.csv','Output\\dataset_bodyPerformance_20240424201824\\bodyPerformance_class_D_edu_20240424201826.csv']
#file_names = ['outputApple400\\apple_quality_class_0_edu_20240418154718.csv','outputApple400\\apple_quality_class_1_edu_20240418154718.csv']
#file_names = ['cirrhosis_1.0_part2_20240301192500.csv','cirrhosis_2.0_part2_20240301192500.csv','cirrhosis_3.0_part2_20240301192500.csv','cirrhosis_4.0_part2_20240301192500.csv']
#file_names = ['milknew_0_part0_20240320122332.csv','milknew_1_part0_20240320122332.csv','milknew_2_part0_20240320122332.csv']
#file_names = ['HotelReservations_0_part0_20240319174159.csv','HotelReservations_1_part0_20240319174159.csv']
#file_names = ['outputApple\\apple_quality_0_part0_20240328104334.csv','outputApple\\apple_quality_1_part0_20240328104334.csv']
#file_names = ['outputGenderv7\\gender_classification_v7_0_part0_20240325124654.csv','outputGenderv7\\gender_classification_v7_1_part0_20240325124654.csv']
			#'outputCancer\\cancer_prediction_dataset_0_part0_20240325154122.csv','outputCancer\\cancer_prediction_dataset_1_part0_20240325154122.csv'
			#'outputGender\\gender_classification_v7_0_part0_20240325124654.csv','outputGender\\gender_classification_v7_1_part0_20240325124654.csv'	
			  #'outputWineQT\\WineQT_5_part0_20240322162654.csv','outputWineQT\\WineQT_6_part0_20240322162654.csv','outputWineQT\\WineQT_7_part0_20240322162654.csv','outputWineQT\\WineQT_4_part0_20240322162654.csv','outputWineQT\\WineQT_8_part0_20240322162654.csv','outputWineQT\\WineQT_3_part0_20240322162654.csv'
#file_names = ['outputIonosphere4\\ionosphere3_class_0_edu_20240406105133.csv','outputIonosphere4\\ionosphere3_class_1_edu_20240406105133.csv']
#file_names = ['outputApple4\\apple_quality_class_0_edu_20240406125611.csv','outputApple4\\apple_quality_class_1_edu_20240406125611.csv']
#file_names = ['outputGenderV9\\gender_class_v7_class_0_edu_20240408170208.csv','outputGenderV9\\gender_class_v7_class_1_edu_20240408170208.csv']

"""
	