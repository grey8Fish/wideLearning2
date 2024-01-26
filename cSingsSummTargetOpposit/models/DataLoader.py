#1. Пробный шаг надо сделать первым шагом
#2. Вставить проверку противоположной категории

import csv
import numpy as np

class DataLoader:
    """
    Класс DataLoader предназначен для загрузки и обработки данных из файлов CSV.

    Аттрибуты:
        file_names (list): Список имен файлов для загрузки данных.
        arg_classes (numpy.ndarray): Массив для хранения загруженных данных.
        column_names (list): Список названий столбцов данных.
    """

    def __init__(self, file_names):
        """
        Конструктор класса DataLoader.

        Аргументы:
            file_names (list): Список имен файлов для загрузки данных.
            classes_count (int): Количество классов данных.
            instances_max (int): Максимальное количество экземпляров данных в каждом классе.
            ordinate_count (int): Количество ординат (столбцов) в данных.
        """
        self.file_names = file_names  # Список имен файлов CSV для обработки
        self.classes_count = 0        # Количество уникальных классов в данных
        self.instances_max = 0        # Максимальное количество экземпляров в каждом классе
        self.ordinate_count = 0       # Количество столбцов в данных за исключением 'number' и 'target'
        self.arg_classes = None       # Массив для хранения обработанных данных
        self.column_names = None      # Список названий столбцов
        self._prepare_data()          # Вызов метода для подготовки и анализа данных

    def _prepare_data(self):
        """
        Подготовка данных, расчет classes_count, instances_max и ordinate_count.
        Этот метод анализирует файлы данных для определения необходимых параметров.
        """
        class_instances = {}  # Словарь для подсчета экземпляров каждого класса
        for file_name in self.file_names:
            with open(file_name, encoding='utf-8') as file:
                csv_reader = csv.DictReader(file)
                if not self.column_names:
                    # Определение списка столбцов, исключая 'number' и 'target'
                    self.column_names = [col for col in csv_reader.fieldnames if col not in ['number', 'target']]
                    self.ordinate_count = len(self.column_names)

                for row in csv_reader:
                    # Подсчет экземпляров для каждого класса
                    class_label = row['target']
                    if class_label not in class_instances:
                        class_instances[class_label] = 0
                    class_instances[class_label] += 1

        # Расчет общего количества классов и максимального количества экземпляров
        self.classes_count = len(class_instances)
        self.instances_max = max(class_instances.values())
        # Инициализация массива для хранения данных
        self.arg_classes = np.zeros((self.classes_count, self.instances_max, self.ordinate_count + 4), dtype=np.int32)




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
