#1. Пробный шаг надо сделать первым шагом
#2. Вставить проверку противоположной категории

import csv
import numpy as np

class DataLoader:
    """
    Класс DataLoader предназначен для загрузки и обработки данных из файлов CSV.

    Аттрибуты:
        file_names (list): Список имен файлов для загрузки данных.
        classes_count (int): Количество классов данных.
        instances_max (int): Максимальное количество экземпляров данных в каждом классе.
        ordinate_count (int): Количество ординат .
        arg_classes (numpy.ndarray): Массив для хранения загруженных данных.
        column_names (list): Список названий столбцов данных.
    """

    def __init__(self, file_names, classes_count, instances_max, ordinate_count):
        """
        Конструктор класса DataLoader.

        Аргументы:
            file_names (list): Список имен файлов для загрузки данных.
            classes_count (int): Количество классов данных.
            instances_max (int): Максимальное количество экземпляров данных в каждом классе.
            ordinate_count (int): Количество ординат (столбцов) в данных.
        """
        self.file_names = file_names
        self.classes_count = classes_count
        self.instances_max = instances_max
        self.ordinate_count = ordinate_count
        # Инициализация массива для хранения данных
        self.arg_classes = np.zeros((classes_count, instances_max, ordinate_count + 4), dtype=np.int32)
        self.column_names = None

    def load_data(self):
        """
        Загрузка и обработка данных из файлов.
        """
        for i, file_name in enumerate(self.file_names):
            with open(file_name, encoding='utf-8') as file:
                csv_reader = csv.reader(file, delimiter=',')
                for j, row in enumerate(csv_reader):
                    if j < self.instances_max:
                        processed_row = []
                        for item in row:
                            try:
                                # Преобразование в целое число, если возможно
                                processed_row.append(int(item))
                            except ValueError:
                                # Обработка нечисловых значений (например, пропуск или замена на 0)
                                processed_row.append(0)
                        # Сохранение обработанного ряда в массив
                        self.arg_classes[i, j, :len(processed_row)] = processed_row

    def get_data(self):
        """
        Получение загруженных данных.

        Возвращает:
            numpy.ndarray: Массив загруженных данных.
        """
        return self.arg_classes

    def get_column_names(self):
        """
        Получение названий столбцов данных.

        Возвращает:
            list: Список названий столбцов.
        """
        if not self.column_names and self.file_names:
            with open(self.file_names[0], encoding='utf-8') as file:
                csv_reader = csv.DictReader(file)
                self.column_names = csv_reader.fieldnames
        return self.column_names
