import csv
import numpy as np

class DataLoader:
    """
    Класс DataLoader для загрузки и анализа данных из файлов CSV.

    Атрибуты:
        file_names (list of str): Список имен файлов для загрузки.
        name_columns (list of str): Список названий столбцов, извлеченных из файлов CSV.
        arg_classes (list of list of list): Вложенный список для хранения данных каждого класса.
    """

    def __init__(self, file_names):
        """
        Инициализирует DataLoader списком имен файлов.

        Аргументы:
            file_names (list of str): Список имен файлов для загрузки.
        """
        self.file_names = file_names
        self.name_columns = []
        self.arg_classes = []

    def load_data(self):
        """
        Загружает данные из указанных файлов CSV в атрибут arg_classes.
        """
        for index, file_name in enumerate(self.file_names):
            self._read_file(file_name, index)

    def _read_file(self, file_name, class_index):
        """
        Читает один файл CSV и добавляет его данные в список arg_classes.

        Аргументы:
            file_name (str): Имя файла для чтения.
            class_index (int): Индекс, представляющий класс считываемых данных.
        """
        with open(file_name, encoding='utf-8') as file:
            file_reader = csv.DictReader(file, delimiter=',')
            
            # Извлечь названия столбцов только из первого файла
            if not self.name_columns:
                self.name_columns = file_reader.fieldnames[:-1]  # Предполагается, что последний столбец не нужен

            # Инициализировать arg_classes для этого индекса, если это еще не сделано
            if len(self.arg_classes) <= class_index:
                self.arg_classes.append([])

            # Конвертировать строки в целые числа и добавить в arg_classes
            for row in file_reader:
                data_row = [int(row[col_name]) for col_name in self.name_columns]
                self.arg_classes[class_index].append(data_row)

    def get_data(self):
        """
        Возвращает загруженные данные.

        Возвращает:
            list of list of list: Загруженные данные из всех файлов CSV.
        """
        return self.arg_classes

    def get_column_names(self):
        """
        Возвращает названия столбцов, извлеченных из файлов CSV.

        Возвращает:
            list of str: Названия столбцов.
        """
        return self.name_columns
