import csv
from datetime import datetime
import numpy as np
import pandas as pd
import re


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
        self.ordinate_count = 0       # Количество столбцов в данных за исключением 'RowNum' и target_column
        self.arg_classes = None       # Массив для хранения обработанных данных
        self.column_names = None      # Список названий столбцов
        self.target_column = None     # Имя самой правой колонки
        self._prepare_data()          # Вызов метода для подготовки и анализа данных
        #self.class_names = []         # Инициализация атрибута для хранения названий классов
        #self.class_instances = {}     # Добавление для хранения количества экземпляров каждого класса

    def get_timestamp(self):
        """
        Пытается извлечь timestamp из имени файла. Если не удается, возвращает текущий timestamp.
        Предполагается, что timestamp — это последовательность в формате YYYYMMDDHHMMSS в конце имени файла перед расширением.
        """
        # Попытка извлечь timestamp из первого файла в списке
        try:
            # Пример имени файла: 'outputGenderv7\\gender_classification_v7_0_part0_20240325124654.csv'
            # Регулярное выражение для поиска последовательности цифр, соответствующих timestamp
            match = re.search(r"(\d{14})(?=\.\w+)$", self.file_names[0])
            if match:
                timestamp = match.group(1)
                # Проверяем, что найденная строка действительно является валидной датой
                datetime.strptime(timestamp, "%Y%m%d%H%M%S")
                return timestamp
            else:
                raise ValueError("No valid timestamp found")
        except ValueError:
            # В случае ошибки или отсутствия timestamp возвращаем текущий timestamp
            return datetime.now().strftime("%Y%m%d%H%M%S")


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
                    # Задание имени самой правой колонки
                    self.target_column = csv_reader.fieldnames[-1]
                    # Исключаем 'RowNum' и самую правую колонку, имя которой хранится в self.target_column
                    self.column_names = [col for col in csv_reader.fieldnames if col not in [self.target_column]]#'RowNum', self.target_column]]
                    self.ordinate_count = len(self.column_names)-1
                    #print(len(self.column_names))

                for row in csv_reader:
                    class_label = row[self.target_column]
                    if class_label not in class_instances:
                        class_instances[class_label] = 1
                    else:
                        class_instances[class_label] += 1

        # Расчет общего количества классов и максимального количества экземпляров
        self.classes_count = len(class_instances)
        self.instances_max = max(class_instances.values())
        # Инициализация массива для хранения данных
        #print('ArgClasses initialized with parameters:')
        #print([self.classes_count, self.instances_max, self.ordinate_count + 4])
        self.arg_classes = np.zeros((self.classes_count, self.instances_max, self.ordinate_count + 3), dtype=np.int32)
        #self.arg_classes = np.zeros((self.classes_count+1, self.instances_max*2, self.ordinate_count + 2), dtype=np.int32)
        self.class_names = list(class_instances.keys())
        self.class_instances = class_instances 
        

    def load_data(self):
        """
        Загружает данные из файлов CSV в массив arg_classes.
        """
        # Проверяем, что column_names уже определены
        if not self.column_names:
            raise ValueError("Column names must be defined before loading data.")
        
        # Проходим по каждому имени файла и его индексу в списке self.file_names
        for class_index, file_name in enumerate(self.file_names):
            with open(file_name, encoding='utf-8') as file:
                csv_reader = csv.DictReader(file)
                row_index = 0  # Индекс для отслеживания текущей строки в arg_classes
            
                for row in csv_reader:
                    if row_index >= self.instances_max:
                        break  # Прекращаем чтение, если достигли максимального количества экземпляров
                
                    # Заполняем соответствующую строку в arg_classes для текущего класса
                    for iVector, column_name in enumerate(self.column_names):
                        # Преобразуем значение в int, предполагая, что все значения числовые
                        try:
                            value = int(row[column_name])
                        except ValueError:
                            # Если преобразование не удалось, используем 0 как запасное значение
                            value = 0
                        
                        self.arg_classes[class_index][row_index][iVector] = value
                
                    row_index += 1
        #print(self.arg_classes)


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
    

    def print_column_names(self):
        """
        Print список названий столбцов данных.
        """
        column_names = self.get_column_names()
        print("Column names:", column_names)


    def print_arg_classes(self):
        """
        Вывод данных arg_classes в виде таблицы.
        """
        # Предполагаем, что есть 4 дополнительных столбца (например, 'Extra')
        total_columns = self.ordinate_count + 4
    
        # Проходим по каждому классу
        for class_index in range(self.classes_count):
            class_name = self.class_names[class_index]
            print(f"\nClass {class_name}:")
        
            # Проходим по каждому экземпляру в классе
            for instance_index in range(self.instances_max):
                instance_data = []
                # Проходим по каждому атрибуту 
                for attribute_index in range(self.ordinate_count):
                    # Получаем значение атрибута
                    value = self.arg_classes[class_index, instance_index, attribute_index]
                    # Формируем строку атрибут-значение
                    attribute_value_str = f"{self.column_names[attribute_index]}: {value}"
                    instance_data.append(attribute_value_str)
            
                # Добавляем дополнительные столбцы, если они есть
                for extra_column_index in range(self.ordinate_count, total_columns):
                    extra_value = self.arg_classes[class_index, instance_index, extra_column_index]
                    if extra_value != 0:  # Предположим, что выводим дополнительные столбцы, только если они не нулевые
                        instance_data.append(f"Extra{extra_column_index-self.ordinate_count+1}: {extra_value}")
            
                # Выводим данные экземпляра
                print(f"Instance {instance_index+1}: " + ", ".join(instance_data))


    def get_class_names(self):
        """
        Возвращает список уникальных названий классов.
        """
        return self.class_names


    def print_class_instances_table(self):
        """
        Выводит таблицу соответствия классов и количества их экземпляров.
        """
        df = pd.DataFrame(list(self.class_instances.items()), columns=['Class', 'Instances Count'])
        print(df.to_string(index=False))


    def get_max_instances_nparray(self):
        """
        Возвращает массив numpy, содержащий максимальное количество экземпляров в каждом классе.
        """
        # Инициализация массива для хранения максимального количества экземпляров каждого класса
        max_instances_per_class = np.zeros(self.classes_count, dtype=int)

        # Заполнение массива данными
        for class_index in range(self.classes_count):
            class_name = self.class_names[class_index]
            max_instances_per_class[class_index] = self.class_instances[class_name]

        return max_instances_per_class
