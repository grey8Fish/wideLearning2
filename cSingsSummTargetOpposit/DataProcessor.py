import pandas as pd

class DataProcessor:
    """
    Класс для обработки данных из CSV файла.

    Атрибуты:
    file_name (str): Путь к файлу CSV.
    class_column (str): Имя колонки, по которой будет происходить разделение данных.
    data (DataFrame): Данные, загруженные из CSV файла.

    Методы:
    load_and_parse_csv(): Загружает и обрабатывает CSV файл.
    split_by_class(): Разделяет данные на подмножества в соответствии с указанной колонкой.
    """

    def __init__(self, file_name, class_column):
        self.file_name = file_name
        self.class_column = class_column
        self.data = self.load_and_parse_csv()

    def load_and_parse_csv(self):
        data = pd.read_csv(self.file_name)

        for column in data.columns:
            if pd.api.types.is_numeric_dtype(data[column]):
                data[column] = data[column].apply(self.convert_to_int)

        return data

    def convert_to_int(self, x):
        """
        Преобразует числовое значение в целое число, умножая на 10 в степени количества знаков после запятой.

        Параметры:
        x (float): Числовое значение.

        Возвращает:
        int: Преобразованное в целое число значение.
        """
        if '.' in str(x):
            decimal_places = len(str(x).split('.')[1])
            return int(x * (10 ** decimal_places))
        return int(x)

    def split_by_class(self):
        class_datasets = {}
        for class_value in self.data[self.class_column].unique():
            class_datasets[class_value] = self.data[self.data[self.class_column] == class_value]

        return class_datasets
