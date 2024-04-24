import numpy as np
import json

class neuronInt3checkCorrectionJSON:
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
        scalar_product = np.dot(self.weight_vector, input_vector[:-1])
        if ((current_category != self.target_category) and (scalar_product > self.right_border)) or \
           ((current_category != self.opposite_category) and (scalar_product < self.left_border)):
            return False  # Ошибка в классификации
        return True  # Корректная классификация

# Загрузка данных из JSON файла
with open("output\\weights_apple_quality_20240424135749.json", "r") as file:
    data = json.load(file)

# Пример входного вектора для проверки
input_vector = np.array([-304,-623,1412,1432,2401,1,-2515,0])  # Последний элемент - текущая категория

# Проход по каждому нейрону в JSON файле
for neuron_data in data['neurons']:
    checker = neuronInt3checkCorrectionJSON(len(neuron_data['previous_weights'].split(', ')))
    checker.set_categories(int(neuron_data['category_left']), int(neuron_data['category_right']))
    checker.set_borders(neuron_data['threshold_left'], neuron_data['threshold_right'])
    checker.set_weights(list(map(int, neuron_data['previous_weights'].split(', '))))
    
    # Проверка входного вектора
    if not checker.check_instance(input_vector, input_vector[-1]):
        print(f"Ошибка в нейроне {neuron_data['neuron_number']} при входном векторе {input_vector[-1]}")
    else:
        print(f"Нейрон {neuron_data['neuron_number']} корректно классифицировал входные данные.")

