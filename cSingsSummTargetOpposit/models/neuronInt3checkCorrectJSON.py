import numpy as np
import json
import pandas as pd

class NeuronInt3CheckCorrectionJSON:
    def __init__(self, vector_size):
        self.target_category = None
        self.opposite_category = None
        self.left_border = None
        self.right_border = None
        self.weight_vector = np.zeros(vector_size, dtype=int)

    def set_categories(self, opposite_category, target_category):
        self.opposite_category = opposite_category
        self.target_category = target_category

    def set_borders(self, left_border, right_border):
        self.left_border = left_border
        self.right_border = right_border

    def set_weights(self, weight_vector):
        self.weight_vector = np.array(weight_vector)

    def check_instance(self, input_vector, current_category):
        if len(input_vector) < len(self.weight_vector):
            raise ValueError("Длина input_vector меньше длины weight_vector.")
        scalar_product = np.dot(self.weight_vector, input_vector[:len(self.weight_vector)])
        if ((current_category != self.target_category) and (scalar_product > self.right_border)) or \
           ((current_category != self.opposite_category) and (scalar_product < self.left_border)):
            return False
        return True

# Загрузка данных из JSON файла
with open("output\\weights_apple_quality_20240424135749.json", "r") as file:
    data = json.load(file)

# Чтение данных из CSV файла
csv_file = "outputApple400\\apple_quality_class_0_test_20240418154718.csv"
df = pd.read_csv(csv_file)


# Инициализация списка для результатов
summary_results = []

# Проход по каждой строке в CSV файле
for index, row in df.iterrows():
    input_vector = row[:-1].to_numpy()
    current_category = row.iloc[-1]
    source_rownum = row.iloc[-2]
    correct_count = 0
    total_neurons = len(data['neurons'])

    # Проход по каждому нейрону в JSON файле
    for neuron_data in data['neurons']:
        checker = NeuronInt3CheckCorrectionJSON(len(neuron_data['previous_weights'].split(', ')))
        checker.set_categories(int(neuron_data['category_left']), int(neuron_data['category_right']))
        checker.set_borders(neuron_data['threshold_left'], neuron_data['threshold_right'])
        checker.set_weights(list(map(int, neuron_data['previous_weights'].split(', '))))

        # Проверка входного вектора
        if checker.check_instance(input_vector, current_category):
            correct_count += 1

    # Расчёт процента корректных нейронов
    correct_percentage = (correct_count / total_neurons) * 100

    # Добавление результатов в список
    summary_results.append({
        #'Row': index, 
        'Source RowNum': source_rownum,
        'Correct Neurons': correct_count, 
        'Incorrect Neurons': total_neurons - correct_count, 
        'Category': current_category, 
        'Correct Percentage': f"{correct_percentage:.2f}%"
    })

# Создание DataFrame из списка результатов
summary_df = pd.DataFrame(summary_results)

# Настройки для отображения всех строк и столбцов DataFrame
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# Вывод таблицы с результатами
print(summary_df)
