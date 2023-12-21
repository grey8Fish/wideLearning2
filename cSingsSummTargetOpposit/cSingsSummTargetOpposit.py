
#import pandas as pan
import numpy as np
import csv
def calcDistanceMinus(miNoTarget, categorySpecified, instanMax, ordCount, aClasses):
    '''получает минимум скалярного произведения, номер целевой категории, количество экземпляров
     в самом объемном классе, количество ординат и матрицу аргументов по классам
    возвращает минимальное расстояние до отрицательной стенки коридора'''
    distanceCutOffPrev = -miNoTarget
    distanceCutOffCurr = -miNoTarget
    iRow = 0
    while iRow < instanMax:
        if aClasses[categorySpecified][iRow][ordCount + 3] == 0:
            distanceCutOffCurr = aClasses[categorySpecified][iRow][ordCount + 2] - miNoTarget
            if distanceCutOffCurr < distanceCutOffPrev:
                distanceCutOffPrev = distanceCutOffCurr
        iRow += 1
    return distanceCutOffPrev
def calcDistancePlus(maNoTarget, categorySpecified, instanMax, ordCount, aClasses):
    '''получает максимум скалярного произведения, номер целевой категории, количество экземпляров
     в самом объемном классе, количество ординат и матрицу аргументов по классам
    возвращает минимальное расстояние до положительной стенки коридора'''
    distanceCutOffPrev = maNoTarget
    distanceCutOffCurr = maNoTarget
    iRow = 0
    while iRow < instanMax:
        if aClasses[categorySpecified][iRow][ordCount + 3] == 0:
            distanceCutOffCurr = maNoTarget - aClasses[categorySpecified][iRow][ordCount + 2]
            if distanceCutOffCurr < distanceCutOffPrev:
                distanceCutOffPrev = distanceCutOffCurr
        iRow += 1
    return distanceCutOffPrev
def calculationCutOffMinusNumber(categorySpecified, instanMax, ordCount, aClasses):
    '''получает номер противоположной категории, количество экземпляров в самом объемном классе,
     количество ординат и матрицу аргументов по классам
    возвращает количество отсеченных минусом'''
    countCutOffMinu = 0
    iRow = 0
    while iRow < instanMax:
        if aClasses[categorySpecified][iRow][ordCount + 3] == 1:
            countCutOffMinu += 1
        iRow += 1
    return countCutOffMinu
def calculationCutOffPlusNumber(categorySpecified, instanMax, ordCount, aClasses):
    '''получает номер целевой категории, количество экземпляров в самом объемном классе,
     количество ординат и матрицу аргументов по классам
    возвращает количество отсеченных плюсом'''
    countCutOffPlu = 0
    iRow = 0
    while iRow < instanMax:
        if aClasses[categorySpecified][iRow][ordCount + 3] == 1:
            countCutOffPlu += 1
        iRow += 1
    return countCutOffPlu
def setFlagCuOffCategoryOpposite(categorySpecified, clCount, instanMax, ordCount, aClasses):
    '''получает номер противоположной категории, количество классов, количество экземпляров
     в самом объемном классе, количество ординат и матрицу аргументов по классам
    устанавливает признак отсеченности противоположной категории
    возвращает минимум скалярного произведения из НЕ противоположной категорий'''
    miNoTarget = 0
    iMatrix = 0
    while iMatrix < clCount:
        if iMatrix == categorySpecified:
            break
        iRow = 0
        while iRow < instanMax:
            if aClasses[iMatrix][iRow][ordCount+2] < miNoTarget:
                miNoTarget = aClasses[iMatrix][iRow][ordCount+2]
            iRow += 1
        iMatrix += 1
    iRow = 0
    while iRow < instanMax:
        if aClasses[categorySpecified][iRow][ordCount+2] < miNoTarget:
            aClasses[categorySpecified][iRow][ordCount + 3] = 1
        else:
            aClasses[categorySpecified][iRow][ordCount + 3] = 0
        iRow += 1
    return miNoTarget
def setFlagCuOffCategoryTarget(categorySpecified, clCount, instanMax, ordCount, aClasses):
    '''получает номер целевой категории, количество классов, количество экземпляров
     в самом объемном классе, количество ординат и матрицу аргументов по классам
    устанавливает признак отсеченности целевой категории
    возвращает максимум скалярного произведения из НЕ целевых категорий'''
    maNoTarget = 0
    iMatrix = 0
    while iMatrix < clCount:
        if iMatrix == categorySpecified:
            break
        iRow = 0
        while iRow < instanMax:
            if aClasses[iMatrix][iRow][ordCount+2] > maNoTarget:
                maNoTarget = aClasses[iMatrix][iRow][ordCount+2]
            iRow += 1
        iMatrix += 1
    iRow = 0
    while iRow < instanMax:
        if aClasses[categorySpecified][iRow][ordCount+2] > maNoTarget:
            aClasses[categorySpecified][iRow][ordCount + 3] = 1
        else:
            aClasses[categorySpecified][iRow][ordCount + 3] = 0
        iRow += 1
    return maNoTarget
def getNameColumn(nameFile):
    'получение названий столбцов'
    with open(nameFile, encoding='utf-8') as r_file:
        # Создаем объект DictReader, указываем символ-разделитель ','
        file_reader = csv.DictReader(r_file, delimiter = ',')
        nameCol  = list(file_reader.fieldnames)  #названия столбцов
    r_file.close()
    del nameCol[-1]
    #del nameColumn[:2]
    return nameCol

def calcCutoffDistance(classesCoun, instancesMa, ordinateCoun, categoryTarge, categoryOpposit, vectorWeightsCur, argClasse):
    '''получает количество классов, количество экземпляров в самом объемном классе,
     количество ординат, номера целевой и противоположной категории и матрицу аргументов по классам
          возвращает количество отсеченных и расстояние до стенки коридора'''
    # Расчет скалярного произведения
    iMatrix = 0
    while iMatrix < classesCoun:
        iRow = 0
        while iRow < instancesMa:
            if (argClasse[iMatrix][iRow][0] * argClasse[iMatrix][iRow][0] + argClasse[iMatrix][iRow][1] *
                argClasse[iMatrix][iRow][1]) == 0:
                break
            argClasse[iMatrix][iRow][ordinateCoun + 2] = np.dot(argClasse[iMatrix, iRow, :ordinateCoun + 1],
                                                                vectorWeightsCur)
            iRow += 1
        iMatrix += 1
    # установка признака отсеченности целевой категории
    # получение максимума скалярного произведения из НЕ целевых категорий
    maxNoTarget = setFlagCuOffCategoryTarget(categoryTarge, classesCoun, instancesMa, ordinateCoun, argClasse)
    # установка признака отсеченности противоположной категории
    # получение минимума скалярного произведения из НЕ противоположных категорий
    minNoOpposite = setFlagCuOffCategoryOpposite(categoryOpposit, classesCoun, instancesMa, ordinateCoun, argClasse)
    # расчет количества отсеченых плюсом, минусом и их суммы
    countCutOffPlus = calculationCutOffPlusNumber(categoryTarge, instancesMa, ordinateCoun, argClasse)
    countCutOffMinus = calculationCutOffMinusNumber(categoryOpposit, instancesMa, ordinateCoun, argClasse)
    countCutOffSum = countCutOffPlus + countCutOffMinus
    # расчет минимального расстояния до одной из стенок коридора
    distanceCutOffPlus = calcDistancePlus(maxNoTarget, categoryTarge, instancesMa, ordinateCoun, argClasse)
    distanceCutOffMinus = calcDistanceMinus(minNoOpposite, categoryOpposit, instancesMa, ordinateCoun, argClasse)
    if distanceCutOffPlus < distanceCutOffMinus:
        distanceCutOffMi = distanceCutOffPlus
    else:
        distanceCutOffMi = distanceCutOffMinus
    return countCutOffMinus, distanceCutOffMinus#countCutOffPlus, distanceCutOffPlus  #countCutOffSum, distanceCutOffMi

def calcDescentDirection(classesCoun, instancesMa, ordinateCoun, categoryTarge, categoryOpposit, vectorWeightsCur, argClasse):
    '''получает количество классов, количество экземпляров в самом объемном классе,
     количество ординат, номера целевой и противоположной категории и матрицу аргументов по классам
          возвращает направление спуска +-1'''
    deltaMultiplierCur = 1
    weightsOl = vectorWeightsCurr[iDelta]
    vectorWeightsCurr[iDelta] = weightsOl + deltaMultiplierCur
    ww = calcCutoffDistance(classesCoun, instancesMa, ordinateCoun, categoryTarge, categoryOpposit, vectorWeightsCur, argClasse)
    countCutOffCurr = ww[0]
    distanceCutOffCurr = ww[1]
    if (countCutOffCurr < countCutOffPrev) or (distanceCutOffCurr >= distanceCutOffPrev):
        deltaMultiplierCur = -1
    return deltaMultiplierCur
def calcBiasDoorstep(classesCoun, instancesMa, ordinateCoun, categoryTarge, categoryOpposit, vectorWeightsCurr, argClasse):
    '''получает количество классов, количество экземпляров в самом объемном классе,
    количество ординат, номера целевой и противоположной категории, вектор весов и матрицу аргументов по классам
    записывает смещение в вектор весов
    возвращает +-значение порога'''

    iRow = 0
    minTarget = 2147483647#максимальный int32 argClasse[categoryTarge][0][ordinateCoun+2]
    while iRow < instancesMa:
        if argClasse[categoryTarge][iRow][ordinateCoun+3] == 1:
            if minTarget > argClasse[categoryTarge][iRow][ordinateCoun+2]:
                minTarget = argClasse[categoryTarge][iRow][ordinateCoun+2]
        iRow += 1
    iClasse = 0
    maxNoTarget = argClasse[iClasse][0][ordinateCoun+2]
    while iClasse < classesCoun:
        if iClasse == categoryTarge:
            iClasse += 1
            continue
        iRow = 1
        while iRow < instancesMa:
            if maxNoTarget < argClasse[iClasse][iRow][ordinateCoun+2]:
                maxNoTarget = argClasse[iClasse][iRow][ordinateCoun+2]
            iRow += 1
        iClasse += 1
    borderTarget = abs(minTarget + maxNoTarget) // 2
    iRow = 0
    maxOpposite = -2147483647#минимальный int32 argClasse[categoryOpposit][0][ordinateCoun+2]
    while iRow < instancesMa:
        if argClasse[categoryOpposit][iRow][ordinateCoun+3] == 1:
            if maxOpposite < argClasse[categoryOpposit][iRow][ordinateCoun+2]:
                maxOpposite = argClasse[categoryOpposit][iRow][ordinateCoun+2]
        iRow += 1
    iClasse = 0
    minNoOpposite = argClasse[iClasse][0][ordinateCoun+2]
    while iClasse < classesCoun:
        if iClasse == categoryOpposit:
            iClasse += 1
            continue
        iRow = 1
        while iRow < instancesMa:
            if minNoOpposite > argClasse[iClasse][iRow][ordinateCoun+2]:
                minNoOpposite = argClasse[iClasse][iRow][ordinateCoun+2]
            iRow += 1
        iClasse += 1
    borderOpposite = abs(maxOpposite + minNoOpposite) // 2
    if borderTarget > borderOpposite:
        borderPM = (borderTarget - borderOpposite) // 2
    else:
        borderPM = (borderOpposite - borderTarget) // 2
    vectorWeightsCurr[ordinateCoun] = borderPM
    borderPM += borderTarget
    return borderPM




#НАЧАЛО ПРОГРАММЫ
nameFileTrain0 = 'seed0_23_11_26.csv'
nameFileTrain1 = 'seed1_23_11_26.csv'
nameFileTrain2 = 'seed2_23_11_26.csv'
classesCount = 3    #количество классов
instancesMax = 67   #количество экземпляров в самом объемном классе
ordinateCount = 7   #количество ординат
vectorWeightsCurr = np.zeros((ordinateCount+1), dtype=np.int32)
vectorWeightsCurr[0] = 179
vectorWeightsCurr[1] = 183
vectorWeightsCurr[2] = 9
vectorWeightsCurr[3] = 185
vectorWeightsCurr[4] = 139
vectorWeightsCurr[5] = -83# -77
vectorWeightsCurr[6] = 147#  -23
vectorWeightsCurr[7] = 0 #376022
#vectorWeightsPrev = np.zeros((ordinateCount+1), dtype=np.int32)
#0-й индекс поправка весов, 1-й количество отсеченных, 2-й расстояние до стенки
deltaCutoffDistance = np.zeros((3, ordinateCount+1), dtype=np.int32)
#сначала Х-ы затем смещение=1, номер экземпляра, скалярное произведение
#и признак отсеченности
argClasses = np.zeros((classesCount, instancesMax, ordinateCount+4), dtype=np.int32)
categoryTarget = 1
categoryOpposite = 2
nameColumn = getNameColumn(nameFileTrain0)
#Чтение аргументов из файла одного из классов
with open(nameFileTrain0, encoding='utf-8') as r_file:
    # Создаем объект DictReader, указываем символ-разделитель ','
    file_reader = csv.DictReader(r_file, delimiter=',')
    iRow = 0
    for row in file_reader:
        iVector = 0
        for iColumn in nameColumn:
            qq = row[iColumn]
            argClasses[0][iRow][iVector] = qq
            iVector += 1
        iRow += 1
#Чтение аргументов из файла одного из классов
with open(nameFileTrain1, encoding='utf-8') as r_file:
    # Создаем объект DictReader, указываем символ-разделитель ','
    file_reader = csv.DictReader(r_file, delimiter=',')
    iRow = 0
    for row in file_reader:
        iVector = 0
        for iColumn in nameColumn:
            qq = row[iColumn]
            argClasses[1][iRow][iVector] = qq
            iVector += 1
        iRow += 1
#Чтение аргументов из файла одного из классов
with open(nameFileTrain2, encoding='utf-8') as r_file:
    # Создаем объект DictReader, указываем символ-разделитель ','
    file_reader = csv.DictReader(r_file, delimiter=',')
    iRow = 0
    for row in file_reader:
        iVector = 0
        for iColumn in nameColumn:
            qq = row[iColumn]
            argClasses[2][iRow][iVector] = qq
            iVector += 1
        iRow += 1
#Сдвинуть номер экземпляра вправо на один индекс
#Записать единицу для скалярного умножения
iMatrix = 0
while iMatrix < classesCount:
    iRow = 0
    while iRow < instancesMax:
        if (argClasses[iMatrix][iRow][0] * argClasses[iMatrix][iRow][0] + argClasses[iMatrix][iRow][1]*argClasses[iMatrix][iRow][1]) == 0:
            break
        argClasses[iMatrix][iRow][ordinateCount+1] = argClasses[iMatrix][iRow][ordinateCount]
        argClasses[iMatrix][iRow][ordinateCount] = 1
        iRow += 1
    iMatrix += 1
#Покоординатный спуск
while True:
    iDelta = 0
    while iDelta < ordinateCount:
        ww = calcCutoffDistance(classesCount, instancesMax, ordinateCount, categoryTarget, categoryOpposite,
                                vectorWeightsCurr, argClasses)
        countCutOffPrev = ww[0]
        distanceCutOffPrev = ww[1]
        # deltaSign = 1
        deltaMultiplierPrev = 0
        # Проверка направления спуска
        weightsOld = vectorWeightsCurr[iDelta]
        deltaMultiplierCurr = calcDescentDirection(classesCount, instancesMax, ordinateCount, categoryTarget,
                                                   categoryOpposite, vectorWeightsCurr, argClasses)
        while True:
            deltaMultiplierCurr = deltaMultiplierCurr * 2
            vectorWeightsCurr[iDelta] = weightsOld + deltaMultiplierCurr + deltaMultiplierPrev
            ww = calcCutoffDistance(classesCount, instancesMax, ordinateCount, categoryTarget, categoryOpposite,
                                    vectorWeightsCurr, argClasses)
            countCutOffCurr = ww[0]
            distanceCutOffCurr = ww[1]
            if countCutOffCurr > countCutOffPrev:
                countCutOffPrev = countCutOffCurr
                distanceCutOffPrev = distanceCutOffCurr
                deltaMultiplierPrev += deltaMultiplierCurr
                deltaMultiplierCurr = calcDescentDirection(classesCount, instancesMax, ordinateCount, categoryTarget,
                                                           categoryOpposite, vectorWeightsCurr, argClasses)
            elif countCutOffCurr < countCutOffPrev:
                deltaCutoffDistance[0][iDelta] = vectorWeightsCurr[iDelta] - deltaMultiplierCurr - weightsOld
                deltaCutoffDistance[1][iDelta] = countCutOffPrev
                vectorWeightsCurr[iDelta] = weightsOld
                break
            elif distanceCutOffCurr > distanceCutOffPrev:
                deltaCutoffDistance[0][iDelta] = vectorWeightsCurr[iDelta] - deltaMultiplierCurr - weightsOld
                deltaCutoffDistance[1][iDelta] = countCutOffPrev
                vectorWeightsCurr[iDelta] = weightsOld
                break
            else:
                countCutOffPrev = countCutOffCurr
                distanceCutOffPrev = distanceCutOffCurr
        iDelta += 1
    iDelta = 1
    maxCutoffDistance = deltaCutoffDistance[1][0]
    maxCutoffIndex = ordinateCount
    while iDelta < ordinateCount:#
        if maxCutoffDistance < deltaCutoffDistance[1][iDelta]:
            maxCutoffDistance = deltaCutoffDistance[1][iDelta]
            maxCutoffIndex = iDelta
        iDelta += 1
    vectorWeightsCurr[maxCutoffIndex] += deltaCutoffDistance[0][maxCutoffIndex]
    #пересчитать скалярные произведения
    if maxCutoffIndex == ordinateCount:
        break
calcCutoffDistance(classesCount, instancesMax, ordinateCount, categoryTarget, categoryOpposite, vectorWeightsCurr, argClasses)
valueDoorstep = calcBiasDoorstep(classesCount, instancesMax, ordinateCount, categoryTarget, categoryOpposite, vectorWeightsCurr, argClasses)

iVector = 9
#КОНЕЦ ПРОГРАММЫ