#Подготовка датасета для widelearning
#Блок настройки в конце программы (if __name__ == "__main__":)

from asyncio import get_event_loop
import math
import numpy as np
import pandas as pd
import os
from datetime import datetime
from decimal import Decimal, InvalidOperation
import json


def read_file(file_name, source_folder):
    """
    Чтение данных из файла различных форматов и конвертация их в DataFrame.

    :param file_name: Имя файла для чтения.
    :param source_folder: Путь к директории, содержащей файл.
    :return: DataFrame, загруженный из указанного файла.
    """
    # Определение формата файла по расширению
    file_extension = os.path.splitext(file_name)[1].lower()
    file_path = os.path.join(source_folder, file_name)

    # Чтение файла в зависимости от его формата
    if file_extension == '.txt':
        df = pd.read_csv(file_path, sep='\t')
    elif file_extension == '.csv':
        df = pd.read_csv(file_path)
    elif file_extension == '.tsv':
        df = pd.read_csv(file_path, sep='\t')
    elif file_extension in ['.xls', '.xlsx']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    initial_row_count = len(df)  # Сохраняем исходное количество строк
    return df, initial_row_count


def save_json(data, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)


def get_decimal_places(series):
    """
    Определение максимального количества знаков после запятой для числовых значений в колонке.

    :param series: Колонка, для которой необходимо определить количество знаков после запятой.
    :return: Максимальное количество знаков после запятой среди всех числовых значений.
    """
    def decimal_places(x):
        try: # Конвертируем в строку и затем в Decimal, если значение не NaN и не 'NA'
            return abs(Decimal(str(x)).as_tuple().exponent) if pd.notna(x) and x != 'NA' else 0
        except InvalidOperation:
            return 0    # Возвращаем 0 для некорректных значений
    # Применяем функцию к каждому элементу серии и возвращаем максимальное значение
    return series.apply(decimal_places).max()


def initialize_output_directory(output_folder='output'):
    """
    Инициализация выходной директории: очистка, если она существует, или создание новой.

    :param output_folder: Название выходной директории. По умолчанию 'output'.
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    output_folder = f"output//dataset_{os.path.splitext(file_name)[0]}_{timestamp}"

    if os.path.exists(output_folder):
        for file in os.listdir(output_folder):
            os.remove(os.path.join(output_folder, file))
    else:
        os.makedirs(output_folder)

    return output_folder


def prepare_df(df, excluded_columns=None, instance_column=None):
    """
    Подготовка DataFrame к обработке: удаление указанных колонок.
    :param df: Исходный DataFrame для обработки.
    :param excluded_columns: Список колонок для исключения из DataFrame. По умолчанию None.
    :param instance_column: Название колонки с идентификаторами экземпляров. По умолчанию None.
    :return: DataFrame после удаления указанных колонок.
    """
    if excluded_columns is not None:
        df.drop(columns=excluded_columns, errors='ignore', inplace=True)

    if instance_column is not None:
        df = df.drop(columns=[instance_column], errors='ignore')

    # Замена строк "NA" на NaN и исключение строк с NaN
    df = df.replace('NA', np.nan).dropna()

    return df


def remove_duplicates(df, class_column, excluded_columns=None, ignored_columns=None, instance_column=None):
    initial_row_count = len(df)
    columns_to_consider = df.columns.difference([class_column] + (excluded_columns or []) + (ignored_columns or []) + ([instance_column] if instance_column else []))
    df.drop_duplicates(subset=columns_to_consider, inplace=True)
    duplicates_count = initial_row_count - len(df)
    percent_duplicates = 0
    if duplicates_count > 0:
        percent_duplicates = (duplicates_count / initial_row_count) * 100
        print(f"В исходном файле обнаружены дубликаты: {duplicates_count} строк удалено ({percent_duplicates:.2f}%)")
    return df, duplicates_count, percent_duplicates


def map_df(df, file_name, output_folder, class_column):
    """
    Маппинг текстовых данных в DataFrame и сохранение маппингов в один JSON файл.
    """
    mappings = {}

    # Обработка каждой колонки в DataFrame
    for column in df.columns:
        if df[column].dtype == object and column != class_column:
            unique_values = set(df[column].dropna().unique())
            yes_no_values = {'Y', 'N'}
            
            # Проверка на наличие значений Y/N и максимум одного дополнительного значения
            if yes_no_values.issubset(unique_values) and len(unique_values - yes_no_values) <= 1:
                # Определение маппинга для Yes/No значений
                yes_no_mapping = {val: (1 if val == 'Y' else -1 if val == 'N' else 0) for val in unique_values}
                yes_no_mapping['NA'] = 0  # Добавляем значение для NA
                df[column] = df[column].map(yes_no_mapping)
                
                # Сохранение маппинга в общий словарь
                mappings[column] = yes_no_mapping
                continue  # Пропускаем оставшуюся часть цикла для этой колонки

            # Создание маппинга для обычных текстовых значений
            else:
                general_mapping = {value: i for i, value in enumerate(unique_values)}
                df[column] = df[column].map(general_mapping)
                
                # Сохранение маппинга в общий словарь
                mappings[column] = general_mapping

    # Сохранение всех маппингов в один JSON файл
    save_json(mappings, os.path.join(output_folder, f"{os.path.splitext(file_name)[0]}_mappings.json"))

    return df


def map_df_csv(df, file_name, output_folder, class_column):
    """
    Маппинг текстовых данных в DataFrame.
    :param df: DataFrame для маппинга.
    :param file_name: Имя файла, используемое для генерации имен файлов маппинга.
    :param output_folder: Папка для сохранения файлов маппинга.
    :param class_column: Название колонки, содержащей классы (целевая переменная).
    :return: DataFrame с маппингом текстовых данных.
    Сохранение в CSV. Не используется.
    """

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


def calculate_columns(df, class_column, ignored_columns, columns_data, significant_digits=None):
    """
    Выполнение математических операций над колонками DataFrame: масштабирование и центрирование данных.

    :param df: DataFrame для обработки.
    :param class_column: Название колонки класса, которая исключается из обработки.
    :param ignored_columns: Список колонок, которые не подлежат обработке.
    :param columns_data: Список для сбора информации о колонках после обработки.
    :return: Обработанный DataFrame с масштабированными и центрированными данными.
    """

    # Определение колонок, которые не будут обрабатываться
    columns_to_exclude = [class_column]
    if ignored_columns is not None:
        columns_to_exclude.extend(ignored_columns)
    
    # Шаги 2-4: Обработка каждой колонки данных, исключая колонки класса и экземпляра
    for column in df.columns:
        if column not in columns_to_exclude:
            # Замена inf и -inf на NaN и удаление строк с NaN
            df[column].replace([np.inf, -np.inf], np.nan, inplace=True)
            df.dropna(subset=[column], inplace=True)

            # Если задано макс. количество значащих цифр, округляем
            if significant_digits is not None and get_decimal_places(df[column]) > 0:
                df[column] = df[column].apply(lambda x: round(x, significant_digits - int(math.floor(math.log10(abs(x)))) - 1) if x != 0 else 0)
                # Шаг 2: Умножение на 10^n, где n - количество знаков после запятой
                df[column] *= 10**significant_digits
            else:
                # Умножение на 10^n, где n - количество знаков после запятой
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
            columns_data.append({
                'Column Name': column,
                'ScaleFactor': float(scale_factor),  
                'UniqueCount': int(unique_values),  
                'Min': int(min_value), 
                'Max': int(max_value)})
            
    return df


def save_and_rearrange_df(df, output_folder, file_name, class_column, max_rows_per_class, percent_edu=34, percent_test=33, percent_correct=33):
    """
    Сохранение и перестановка колонок в DataFrame перед сохранением в файл.
    
    :param df: DataFrame для сохранения.
    :param output_folder: Папка для сохранения файла.
    :param file_name: Исходное имя файла для создания имени выходного файла.
    :param class_column: Название целевой колонки класса, которая будет перемещена в конец DataFrame.
    :param max_rows_per_class: Максимальное количество строк на класс.
    :param percent_edu: Процент данных для обучения.
    :param percent_test: Процент данных для тестирования.
    :param percent_correct: Процент данных для коррекции.
    """
    # Определение timestamp для именования файлов
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")    
    
    # Списки файлов
    edu_files = []
    test_files = []
    cor_files = []
 
    # Добавление RowNum
    df['RowNum'] = np.arange(len(df))

    # Перемещение class_column в самый правый столбец - сохранение и удаление class_column из DataFrame
    class_column_data = df.pop(class_column)
    df[class_column] = class_column_data
    
    # Удаление дубликатов, исключая RowNum и class_column
    initial_row_count = len(df)
    duplicates_mask = df.drop(columns=[class_column, 'RowNum']).duplicated(keep='first')
    num_duplicates = duplicates_mask.sum()  # Подсчет количества дубликатов (без первых уникальных вхождений)
    percent_duplicates = 0
    if num_duplicates > 0:
        percent_duplicates = (num_duplicates / initial_row_count) * 100
        print(f'В обработанной выборке обнаружены и удалены {num_duplicates} дубликатов. ({percent_duplicates:.2f}%)')
    df = df[~duplicates_mask]

    # Сохранение результата в новый файл с меткой времени
    output_file_name = f"{os.path.splitext(file_name)[0]}_{timestamp}.csv"
    df.to_csv(os.path.join(output_folder, output_file_name), index=False)
    
    # Сохранение отдельных файлов по классу с разбиением на части
    grouped_df = df.groupby(class_column)
    for class_val, group in grouped_df:
        if max_rows_per_class is not None: # Ограничение кол-ва экземепляров в одном файле
            if len(group) > max_rows_per_class:
                group = group.sample(n=max_rows_per_class, random_state=1)  # random_state для воспроизводимости
            #else:
                #group = group

        n_rows = len(group)
        rows_for_edu = round(n_rows * percent_edu / 100)
        rows_for_test = round(n_rows * percent_test / 100)
        rows_for_correct = n_rows - rows_for_edu - rows_for_test
        
        #rows_per_file = max(n_rows // 3, 1)  # Деление на 3 части, но не меньше одной строки на файл

        # Создание и сохранение файлов по частям
        for part, rows_count, name_part in zip([0, 1, 2], [rows_for_edu, rows_for_test, rows_for_correct], ['edu', 'test', 'cor']):
            start_row = sum([rows_for_edu, rows_for_test, rows_for_correct][:part])
            end_row = start_row + rows_count
            subset_df = group.iloc[start_row:end_row]

            # Генерация названия файла с учетом класса и части
            subset_file_name = f"{os.path.splitext(file_name)[0]}_class_{class_val}_{name_part}_{timestamp}.csv"
            subset_df.to_csv(os.path.join(output_folder, subset_file_name), index=False)
            
            # Заполняем списки файлов для вывода
            file_info = {"file_name": subset_file_name, "num_instances": rows_count}
            if name_part == 'edu':
                edu_files.append(file_info)
            elif name_part == 'test':
                test_files.append(file_info)
            elif name_part == 'cor':
                cor_files.append(file_info)
                
    final_row_count = len(df)
    return edu_files, test_files, cor_files, final_row_count, num_duplicates, percent_duplicates

def process(file_name, class_column, instance_column=None, excluded_columns=None, ignored_columns=None, significant_digits=None, max_rows_per_class=None, percent_edu=None, percent_test=None, percent_correct=None):
    """
    Главная функция обработки файла: чтение, подготовка, обработка и сохранение данных. Функция выполняет следующие шаги:
    1. Инициализирует выходную директорию, очищая её или создавая новую, если необходимо.
    2. Читает данные из файла, опираясь на его формат.
    3. Подготавливает данные: удаляет ненужные колонки.
    4. Маппинг текстовых данных в числовые значения и сохранение словарей маппинга.
    5. Выполняет математические операции над колонками: масштабирование и центрирование данных.
    6. Сохраняет и переставляет колонки в DataFrame перед его сохранением в файл.
    7. Выводит информацию о колонках после обработки.

    :param file_name: Имя файла для обработки.
    :param class_column: Название целевой колонки.
    :param instance_column: Название колонки с идентификаторами экземпляров. Опционально.
    :param excluded_columns: Список колонок для исключения из обработки. Опционально.
    :param ignored_columns: Список колонок, которые будут проигнорированы при обработке. Опционально.
    """
    # Шаг 1: Инициализация выходной директории
    source_folder = 'sources'
    output_folder = initialize_output_directory(file_name)

    # Шаг 2: Чтение файла
    df, initial_row_count = read_file(file_name, source_folder)
    #Удаление дубликатов
    df, duplicates_count, percent_duplicates = remove_duplicates(df, class_column, excluded_columns, ignored_columns, instance_column)
    
    # Шаг 3: Подготовка DataFrame - удаляет ненужные колонки
    df = prepare_df(df, excluded_columns, instance_column)
    
    # Шаг 4: Маппинг текстовых колонок в DataFrame
    df = map_df(df, file_name, output_folder, class_column)
    
    # Шаг 5: Выполнение математических операций над колонками
    columns_data = []  # Инициализация списка для сбора информации о колонках
    df = calculate_columns(df, class_column, ignored_columns, columns_data, significant_digits)

    # Шаг 6: Сохранение и перестановка колонок перед сохранением
    edu_files, test_files, cor_files, final_row_count, post_process_duplicates, post_process_percent_duplicates = save_and_rearrange_df(
        df, output_folder, file_name, class_column, max_rows_per_class, percent_edu, percent_test, percent_correct)
  
    # Шаг 7: Вывод информации о колонках после обработки
    columns_info = pd.DataFrame(columns_data)
    print(columns_info.to_string(index=False))

    #Сохранение JSON
    process_info = {
        "ProcessStartTime": datetime.now().strftime("%d.%m.%Y %H:%M"),
        "RunParams": {
            "FileName": file_name,
            "ClassColumn": class_column,
            "InstanceColumn": instance_column,
            "ExcludedColumns": excluded_columns or [],
            "IgnoredColumns": ignored_columns or [],
            "SignificantDigits": significant_digits,
            "MaxRowsPerClass": max_rows_per_class,
            "TrainingSetPercentage": f"{percent_edu:.2f}%",
            "TestingSetPercentage": f"{percent_test:.2f}%",
            "ValidationSetPercentage": f"{percent_correct:.2f}%"
        },
        "InitialStatistics": {
            "RowCount": int(initial_row_count),
            "DuplicateInfo": {
                "InitialDuplicatesCount": int(duplicates_count),
                "InitialDuplicatesPercentage": f"{float(percent_duplicates):.2f}%",
                "PostProcessingDuplicatesCount": int(post_process_duplicates),
                "PostProcessingDuplicatesPercentage": f"{float(post_process_percent_duplicates):.2f}%"
            }
        },
        "FinalStatistics": {
            "RowCount": int(final_row_count)
        },
        "FilePaths": {
            "SourceFolder": source_folder,
            "OutputFolder": output_folder
        },
        "FileGroups": {
            "TrainingFiles": edu_files,
            "TestingFiles": test_files,
            "ValidationFiles": cor_files
        },
        "ColumnsData": columns_data  
    }
    save_json(process_info, os.path.join(output_folder, "_process_info.json"))


#######################################################################
# Настройка здесь
# В случае если получили ошибку на какой-либо колонке, добавляем её в excluded_columns    
if __name__ == "__main__":
    file_name = "cirrhosis.csv"     # Имя файла (с расширением)
    class_column = "Stage"            # Целевая колонка
    instance_column = "ID"            # ID колонка, любой итератор (если есть). Если нет - комментируем всю строчку или оставляем пустой.
    #significant_digits = 5             # Максимальное количество значащих цифр перед округлением. Можно закомментировать, будет использоваться максимальное по датасету.
    #max_rows_per_class = 1000           # Устанавливаем ограничение количества строк в одном классе. Можно закомментировать, опционально.
    
    # Разделение выборки, в процентах
    percent_edu = 33       
    percent_test = 33
    percent_correct = 33

   
#   file_name = "milknew.csv" # Имя файла (с расширением)
#   class_column = "Grade"  # Целевая колонка
#   instance_column = "Id"  # ID колонка, любой итератор (если есть). Если нет - комментируем всю строчку или оставляем пустой.
#   file_name = "milknew.csv" # Имя файла (с расширением)
#   class_column = "Grade"  # Целевая колонка
#   instance_column = "Id"  # ID колонка, любой итератор (если есть). Если нет - комментируем всю строчку или оставляем пустой.
#   file_name = "HotelReservations.csv" # Имя файла (с расширением)
#   class_column = "booking_status"  # Целевая колонка
#   instance_column = "Booking_ID"  # ID колонка, любой итератор (если есть). Если нет - комментируем всю строчку или оставляем пустой.
#   file_name = "WineQT.csv" # Имя файла (с расширением)
#   class_column = "quality"  # Целевая колонка


    #excluded_columns = []  # Список колонок, которые будут ИСКЛЮЧЕНЫ из выборки (если необходимо) - данных колонок НЕ будет в выходном файле Если нет - комментируем всю строчку или оставляем пусстой список.
    #ignored_columns = []  # Список колонок, которые будут ИГНОРИРОВАТЬСЯ обработчиком (если необходимо) - данные колонки будут в выходном файле, но не будут преобразованы. Если нет - комментируем всю строчку или оставляем пусстой список.

    # Конец настройки
    # Создание словаря с аргументами для функции и проверками на существование аргумента
    process_args = {
        "file_name": file_name,
        "class_column": class_column,
        **({"instance_column": locals().get('instance_column')} if 'instance_column' in locals() else {}),  # Условное добавление instance_column с проверкой на существование
        **({"excluded_columns": locals().get('excluded_columns', [])}),  # Условное добавление excluded_columns с проверкой на существование и использованием пустого списка как значения по умолчанию
        **({"ignored_columns": locals().get('ignored_columns', [])}),  # Условное добавление ignored_columns с проверкой на существование и использованием пустого списка как значения по умолчанию
        **({"significant_digits": locals().get('significant_digits')} if 'significant_digits' in locals() else {}),  # Условное добавление significant_digits с проверкой на существование
        **({"max_rows_per_class": locals().get('max_rows_per_class')} if 'max_rows_per_class' in locals() else {}),  # Условное добавление max_rows_per_class с проверкой на существование
        "percent_edu": percent_edu,
        "percent_test": percent_test,
        "percent_correct": percent_correct,
    }

    # Вызов функции с использованием распаковки словаря аргументов
    process(**process_args)
