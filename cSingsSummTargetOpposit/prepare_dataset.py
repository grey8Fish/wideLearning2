﻿import numpy as np
import pandas as pd
import os
from datetime import datetime
from decimal import Decimal


def read_file(file_name, source_folder):
    # Определение формата файла по расширению
    file_extension = os.path.splitext(file_name)[1].lower()
    file_path = os.path.join(source_folder, file_name)

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


def get_decimal_places(series):
    def decimal_places(x):
        try: # Конвертируем в строку и затем в Decimal, если значение не NaN и не 'NA'
            return abs(Decimal(str(x)).as_tuple().exponent) if pd.notna(x) and x != 'NA' else 0
        except InvalidOperation:
            return 0    # Возвращаем 0 для некорректных значений
    # Применяем функцию к каждому элементу серии и возвращаем максимальное значение
    return series.apply(decimal_places).max()


def initialize_output_directory(output_folder='output'):
    """
    Инициализация и очистка выходной директории.
    Очистка директории output если она существует, создать папку если она не существует
    """
    if os.path.exists(output_folder):
        for file in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, file))
    else:
        os.makedirs(output_folder)


def prepare_and_map_df(df, file_name, output_folder, class_column, excluded_columns=None, instance_column=None, ignored_columns=None):
    """
    Обработка DataFrame: создание маппингов для текстовых данных и удаление указанных колонок.
    """
    # Исключение колонок, указанных в input:excluded_columns
    if excluded_columns is not None:
        df.drop(columns=excluded_columns, errors='ignore', inplace=True)

    # Удаление instance_column из датафрейма
    if instance_column is not None:
        df = df.drop(columns=[instance_column], errors='ignore')  
        

    # Шаг 1: Замена текстовых классов числовыми. Создание словаря для сопоставления текстовых классов с числовыми индексами.
    # Проверка, содержит ли колонка значения Yes/No или Y/N
    for column in df.columns:
        if df[column].dtype == object and column != class_column:
            unique_values = set(df[column].dropna().unique())
            yes_no_values = {'Y', 'N'}
    
            # Проверка на наличие значений Y/N и максимум одного дополнительного значения
            if yes_no_values.issubset(unique_values) and len(unique_values - yes_no_values) <= 1:
                # Определение маппинга для Yes/No значений
                yes_no_mapping = {val: (1 if val == 'Y' else -1 if val == 'N' else 0) for val in unique_values}
                yes_no_mapping['NA'] = 0
                df[column] = df[column].map(yes_no_mapping)
    
                # Сохранение словаря в отдельный файл
                mapping_file_name = f"mapping_{os.path.splitext(file_name)[0]}_{column}.csv"
                mapping_df = pd.DataFrame(list(yes_no_mapping.items()), columns=['Value', 'Mapped'])
                mapping_df.to_csv(os.path.join(output_folder, mapping_file_name), index=False)
                continue  # Пропускаем оставшуюся часть цикла для этой колонки
    
        else:
            if df[class_column].dtype == object:
                class_mapping = {label: idx for idx, label in enumerate(df[class_column].unique())}
                df[class_column] = df[class_column].map(class_mapping)
                
                # Создание DataFrame из маппинга
                class_mapping_df = pd.DataFrame(list(class_mapping.items()), columns=['Value', 'Mapped'])
        
                # Сохранение словаря классов в отдельный файл
                mapping_file_name = f"mapping_{os.path.splitext(file_name)[0]}_{class_column}.csv"
                class_mapping_df.to_csv(os.path.join(output_folder, mapping_file_name), index=False)


    # Замена строк "NA" на NaN и исключение строк с NaN
    df = df.replace('NA', np.nan).dropna()

    # Обработка колонок для создания маппингов
    for column in df.columns:
        if df[column].dtype == object and column != class_column:
            # Создание словаря для сопоставления текстовых значений с числовыми индексами
            column_mapping = {label: idx for idx, label in enumerate(df[column].unique())}
            df[column] = df[column].map(column_mapping)
        
            # Создание DataFrame из маппинга
            column_mapping_df = pd.DataFrame(list(column_mapping.items()), columns=['Value', 'Mapped'])
            
            # Проверка, являются ли все значения в class_column целыми числами - необходимо из-за особенности вывода в pandas
            if df[class_column].dropna().apply(lambda x: float(x).is_integer()).all():  # Проверка, являются ли все значения в class_column целыми числами
                df[class_column] = df[class_column].astype(int)  # Преобразование к int, если условие выполняется
    
            # Сохранение словаря в отдельный файл с измененным форматированием названия
            mapping_file_name = f"mapping_{os.path.splitext(file_name)[0]}_{column}.csv"
            column_mapping_df.to_csv(os.path.join(output_folder, mapping_file_name), index=False)

    return df


        
def process(file_name, class_column, instance_column=None, excluded_columns=None, ignored_columns=None):
    source_folder = 'sources'
    output_folder = 'output'
    
    initialize_output_directory(output_folder)
    
    # Определение timestamp для именования файлов
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    # Инициализация списка для сбора информации о колонках
    columns_data = []

    df = read_file(file_name, source_folder)
    
    # Вызов новой функции для подготовки и маппинга DataFrame
    df = prepare_and_map_df(df, file_name, output_folder, class_column, excluded_columns, instance_column, ignored_columns)

    # Определение колонок, которые не будут обрабатываться (колонка класса)
    columns_to_exclude = [class_column]
    # Добавление ignored_columns к списку исключаемых из обработки, если таковые имеются
    if ignored_columns is not None:
        columns_to_exclude.extend(ignored_columns)
    
    # Шаги 2-4: Обработка каждой колонки данных, исключая колонки класса и экземпляра
    for column in df.columns:
        if column not in columns_to_exclude:
            # Шаг 2: Умножение на 10^n, где n - количество знаков после запятой
            n_decimal = get_decimal_places(df[column])
            df[column] *= 10**n_decimal
    
            # Шаг 3: Вычитание минимального значения из каждого элемента столбца
            df[column] -= df[column].min()
    
            # Шаг 4: Вычитание половины максимального значения из каждого элемента столбца
            half_max = df[column].max() / 2
            df[column] -= half_max
    
    # Поиск глобального максимума после шагов 2-4
    global_max = df.drop(columns_to_exclude, axis=1).max().max()
    
    # Шаг 5: Масштабирование данных относительно глобального максимума
    for column in df.columns:
        if column not in columns_to_exclude:
            # Определение коэффициента масштабирования для текущей колонки
            scale_factor = global_max / df[column].max()
            scale_factor = scale_factor  # Округление коэффициента масштабирования
    
            # Применение масштабирования к значениям в колонке
            df[column] *= scale_factor
    
            # Конвертация значений в целые числа
            df[column] = df[column].astype(int)
    
            # Сбор информации о колонке (для вывода, в расчётах не участвуеи)
            unique_values = len(df[column].unique())
            min_value = df[column].min()
            max_value = df[column].max()
            columns_data.append({'Column Name': column, 
                                 'ScaleFactor': round(scale_factor), 
                                 'UniqueCount': unique_values, 
                                 'Min': min_value, 
                                 'Max': max_value})
        
    # Добавление RowNum
    df['RowNum'] = np.arange(len(df))

    # Перемещение class_column в самый правый столбец - сохранение и удаление class_column из DataFrame
    class_column_data = df.pop(class_column)
    df[class_column] = class_column_data
    
    # Сохранение результата в новый файл с меткой времени
    output_file_name = f"{os.path.splitext(file_name)[0]}_{timestamp}.csv"
    df.to_csv(os.path.join(output_folder, output_file_name), index=False)
    
    # Сохранение отдельных файлов по классу с разбиением на части
    grouped_df = df.groupby(class_column)
    for class_val, group in grouped_df:
        n_rows = len(group)
        rows_per_file = max(n_rows // 3, 1)  # Деление на 3 части, но не меньше одной строки на файл

        for part in range(3):
            start_row = part * rows_per_file
            end_row = start_row + rows_per_file
            subset_df = group.iloc[start_row:end_row]

            # Генерация названия файла с учетом класса и части
            subset_file_name = f"{os.path.splitext(file_name)[0]}_{class_val}_part{part}_{timestamp}.csv"
            subset_df.to_csv(os.path.join(output_folder, subset_file_name), index=False)
    
    # Создание DataFrame из собранных данных по колонкам
    columns_info = pd.DataFrame(columns_data)

    # Вывод информации о колонках
    print(columns_info.to_string(index=False))


if __name__ == "__main__":
    file_name = "Hotel Reservations.csv"
    class_column = "booking_status"  # Целевая колонка
    instance_column = "Booking_ID"  # ID колонка (если есть)
    #excluded_columns = []  # Список колонок, которые будут ИСКЛЮЧЕНЫ из выборки (если необходимо) - данных колонок НЕ будет в выходном файле
    #ignored_columns = []  # Список колонок, которые будут ИГНОРИРОВАТЬСЯ обработчиком (если необходимо) - данные колонки будут в выходном файле, но не будут преобразованы

    # Создание словаря с аргументами для функции и проверками на существование аргумента
    process_args = {
        "file_name": file_name,
        "class_column": class_column,
        **({"instance_column": locals().get('instance_column')} if 'instance_column' in locals() else {}),  # Условное добавление instance_column с проверкой на существование
        **({"excluded_columns": locals().get('excluded_columns', [])}),  # Условное добавление excluded_columns с проверкой на существование и использованием пустого списка как значения по умолчанию
        **({"ignored_columns": locals().get('ignored_columns', [])}),  # Условное добавление ignored_columns с проверкой на существование и использованием пустого списка как значения по умолчанию
    }

    # Вызов функции с использованием распаковки словаря аргументов
    process(**process_args)
