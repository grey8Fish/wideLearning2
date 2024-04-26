import numpy as np
import json
import pandas as pd
import cProfile, pstats

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

    @staticmethod
    def collect_weights(data):
        weights_array = []
        for neuron in data['neurons']:
            weights = list(map(int, neuron['previous_weights'].split(', ')))
            weights_array.append(weights)
        return weights_array

def main():
    # Загрузка данных из JSON файла
    paths = []    
    with open("outputApple400\\weights_apple_quality_20240426103915.json", "r") as file:
        data = json.load(file)
        paths = data['file_names']
    paths = [path.replace("_edu_", "_test_") for path in paths]
    
    # Сбор массива весов
    weights_array = NeuronInt3CheckCorrectionJSON.collect_weights(data)

    # Чтение данных из CSV файла
    dfs = [pd.read_csv(file) for file in paths]
    df = pd.concat(dfs, ignore_index=True)

    # Инициализация списка для результатов
    results = []

    # Проход по каждой строке в CSV файле
    for index, row in df.iterrows():
        input_vector = row[:-1].to_numpy()
        current_category = row.iloc[-1]
        source_rownum = row.iloc[-2]

        # Проход по каждому нейрону в JSON файле
        for neuron_index, neuron_data in enumerate(data['neurons']):
            checker = NeuronInt3CheckCorrectionJSON(len(neuron_data['previous_weights'].split(', ')))
            checker.set_categories(neuron_data['category_left'], neuron_data['category_right'])
            checker.set_borders(neuron_data['threshold_left'], neuron_data['threshold_right'])
            checker.set_weights(list(map(int, neuron_data['previous_weights'].split(', '))))

            # Проверка входного вектора
            if checker.check_instance(input_vector, current_category):
                results.append({
                    "Строка CSV": source_rownum,
                    "Номер нейрона": neuron_index,
                    "Категория из файла": current_category,
                    "Категория по нейрону": checker.target_category if current_category != checker.opposite_category else checker.opposite_category
                })
                break

    # Создание DataFrame для результатов
    results_df = pd.DataFrame(results)

    # Расчёт точности
    accuracy = np.mean(results_df['Категория из файла'] == results_df['Категория по нейрону']) * 100

    # Вывод результатов
    print(f"Точность: {accuracy:.2f}%")
    pd.set_option('display.max_rows', None)  # Вывод всех строк DataFrame
    print(results_df.to_string())

    return results_df

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()  # Начинаем профилирование
    result_df = main() 
