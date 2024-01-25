import numpy as np
import pandas as pd
import os
from datetime import datetime
from decimal import Decimal, InvalidOperation

class DataProcessor:
    def __init__(self, file_name, class_column, instance_column=None):
        self.file_name = file_name
        self.class_column = class_column
        self.instance_column = instance_column
        self.source_folder = 'Sources'
        self.output_folder = 'Output'

    def read_file(self):
        # Определение формата файла по расширению
        file_extension = os.path.splitext(self.file_name)[1].lower()
        file_path = os.path.join(self.source_folder, self.file_name)
        
        # Чтение файла в зависимости от его формата
        if file_extension == '.txt':
            return pd.read_csv(file_path, sep='\t')
        elif file_extension == '.csv':
            return pd.read_csv(file_path)
        elif file_extension == '.tsv':
            return pd.read_csv(file_path, sep='\t')
        elif file_extension in ['.xls', '.xlsx']:
            return pd.read_excel(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
    def process(self):
        # Чтение файла
        df = self.read_file()
    
        # Шаг 1: Замена текстовых классов числовыми. Создание словаря для сопоставления текстовых классов с числовыми индексами.
        if df[self.class_column].dtype == object:
            class_mapping = {label: idx for idx, label in enumerate(df[self.class_column].unique())}
            df[self.class_column] = df[self.class_column].map(class_mapping)
            # Сохранение словаря классов в отдельный файл
            pd.DataFrame.from_dict(class_mapping, orient='index').to_csv(os.path.join(self.output_folder, f'class_mapping_{self.file_name}'))
        
        # Замена строк "NA" на NaN
        df = df.replace('NA', np.nan)
        # Исключение строк с NaN из обработки
        df = df.dropna()
            
        for column in df.columns:
            if df[column].dtype == object and column != self.class_column:
                # Создание словаря для сопоставления текстовых значений с числовыми индексами
                column_mapping = {label: idx for idx, label in enumerate(df[column].unique())}
                df[column] = df[column].map(column_mapping)
        
                # Сохранение словаря в отдельный файл
                mapping_file_name = f"{column}_mapping_{self.file_name}"
                pd.DataFrame.from_dict(column_mapping, orient='index').to_csv(os.path.join(self.output_folder, mapping_file_name))

    
        # Определение колонок, которые не будут обрабатываться (колонка класса и, если указано, колонка экземпляра)
        columns_to_exclude = [self.class_column]
        if self.instance_column is not None:
            columns_to_exclude.append(self.instance_column)

        # Шаги 2-4: Обработка каждой колонки данных, исключая колонки класса и экземпляра
        for column in df.columns:
            if column not in columns_to_exclude:
                # Шаг 2: Умножение на 10^n, где n - количество знаков после запятой
                n_decimal = self.get_decimal_places(df[column])
                df[column] *= 10**n_decimal

                # Шаг 3: Вычитание минимального значения из каждого элемента столбца
                df[column] -= df[column].min()

                # Шаг 4: Вычитание половины максимального значения из каждого элемента столбца
                half_max = round(df[column].max() / 2)
                df[column] -= half_max

        # Поиск глобального максимума после шагов 2-4
        global_max = df.drop(columns_to_exclude, axis=1).max().max()

        # Шаг 5: Масштабирование данных относительно глобального максимума
        for column in df.columns:
            if column not in columns_to_exclude:
                # Определение коэффициента масштабирования для текущей колонки
                scale_factor = global_max / df[column].max()
                scale_factor = round(scale_factor, 0)  # Округление коэффициента масштабирования

                # Применение масштабирования к значениям в колонке
                df[column] *= scale_factor

                # Конвертация значений в целые числа
                df[column] = df[column].astype(int)

                print(f"Column: {column}, Scale Factor: {scale_factor}")
            
        # Проверка на наличие колонки с номером экземпляра
        if self.instance_column is None or self.instance_column not in df.columns:
            # Задаём имя для новой колонки, если оно не было задано
            self.instance_column = self.instance_column or "RowNum"
            # Добавление колонки с порядковыми номерами
            df[self.instance_column] = range(len(df))

        # Сохранение результата в новый файл с меткой времени
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_file_name = f"{os.path.splitext(self.file_name)[0]}_{timestamp}.csv"
        df.to_csv(os.path.join(self.output_folder, output_file_name), index=False)

    @staticmethod
    def get_decimal_places(series):
        def decimal_places(x):
            try:
                # Конвертируем в строку и затем в Decimal, если значение не NaN и не 'NA'
                return abs(Decimal(str(x)).as_tuple().exponent) if pd.notna(x) and x != 'NA' else 0
            except InvalidOperation:
                return 0  # Возвращаем 0 для некорректных значений

        # Применяем функцию к каждому элементу серии и возвращаем максимальное значение
        return series.apply(decimal_places).max()


# Использование класса
processor = DataProcessor("cirrhosis.csv", "Status")
processor.process()
