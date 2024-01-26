import numpy as np
import pandas as pd
import os
from datetime import datetime
from decimal import Decimal, InvalidOperation


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

        
def process(file_name, class_column, instance_column=None):
    source_folder = 'sources'
    output_folder = 'output'
    
    # Определение timestamp для именования файлов
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

    df = read_file(file_name, source_folder)
        
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
                mapping_file_name = f"{column}_yesno_mapping_{file_name}"
                pd.DataFrame.from_dict(yes_no_mapping, orient='index').to_csv(os.path.join(output_folder, mapping_file_name))
                continue  # Пропускаем оставшуюся часть цикла для этой колонки
    
            else:   
                if df[class_column].dtype == object:
                    class_mapping = {label: idx for idx, label in enumerate(df[class_column].unique())}
                    df[class_column] = df[class_column].map(class_mapping)
                    # Сохранение словаря классов в отдельный файл
                    pd.DataFrame.from_dict(class_mapping, orient='index').to_csv(os.path.join(output_folder, f'class_mapping_{file_name}'))
    
    # Замена строк "NA" на NaN
    df = df.replace('NA', np.nan)
    # Исключение строк с NaN из обработки
    df = df.dropna()
        
    for column in df.columns:
        if df[column].dtype == object and column != class_column:
            # Создание словаря для сопоставления текстовых значений с числовыми индексами
            column_mapping = {label: idx for idx, label in enumerate(df[column].unique())}
            df[column] = df[column].map(column_mapping)
    
            # Сохранение словаря в отдельный файл
            mapping_file_name = f"{column}_mapping_{file_name}"
            pd.DataFrame.from_dict(column_mapping, orient='index').to_csv(os.path.join(output_folder, mapping_file_name))
    
    
    # Определение колонок, которые не будут обрабатываться (колонка класса и, если указано, колонка экземпляра)
    columns_to_exclude = [class_column]
    if instance_column is not None:
        columns_to_exclude.append(instance_column)
    
    # Шаги 2-4: Обработка каждой колонки данных, исключая колонки класса и экземпляра
    for column in df.columns:
        if column not in columns_to_exclude:
            # Шаг 2: Умножение на 10^n, где n - количество знаков после запятой
            n_decimal = get_decimal_places(df[column])
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
    if instance_column is None or instance_column not in df.columns:
        # Задаём имя для новой колонки, если оно не было задано
        instance_column = instance_column or "RowNum"
        # Добавление колонки с порядковыми номерами
        df[instance_column] = range(len(df))

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



if __name__ == "__main__":
    file_name = "cirrhosis.csv"
    class_column = "Status"
    instance_column = "ID"
    process(file_name, class_column, instance_column)
