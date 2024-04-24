import json
import numpy as np
import csv
import pandas as pd

class PredictionModel:
    def __init__(self, weights_file, test_file):
        self.weights, self.thresholds, self.categories = self.load_weights(weights_file)
        self.test_data, self.column_names = self.load_test_data(test_file)

    # Загрузка весов и порогов из JSON файла
    def load_weights(self, file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
        weights = []
        thresholds = []
        categories = []
        # Извлечение весов для всех нейронов
        for neuron in data['neurons']:
            weights_str = neuron['previous_weights']
            weights_array = np.array(list(map(int, weights_str.split(", "))))
            weights.append(weights_array)
            thresholds.append((neuron['threshold_left'], neuron['threshold_right']))
            categories.append((neuron['category_left'], neuron['category_right']))
        return weights, thresholds, categories

    # Загрузка тестовых данных
    def load_test_data(self, file_path):
        test_data = []
        with open(file_path, newline='') as file:
            reader = csv.reader(file)
            column_names = next(reader)[:-2]  # Считывание заголовков, исключая RowNum и класс
            for row in reader:
                test_data.append(list(map(int, row[:-2])))  # Все данные за исключением RowNum и класс
        return test_data, column_names

    # Предсказание класса
    def predict(self):
        predictions = []
        for data in self.test_data:
            prediction = []
            for weights, (left, right), (cat_left, cat_right) in zip(self.weights, self.thresholds, self.categories):
                scalar_product = np.dot(np.array(data), weights)
                # Присвоение класса на основе порогов
                if scalar_product < left:
                    prediction.append(cat_left) ##
                    
            predictions.append(prediction)
        return predictions

# Путь к файлу с весами и тестовому файлу
weights_file = 'output\\weights_apple_quality_20240424135749.json'
test_file = 'outputApple400\\apple_quality_class_0_test_20240418154718.csv'

# Создание экземпляра модели 
model = PredictionModel(weights_file, test_file)
predictions = model.predict()

# Сохранение результатов в файл
output_data = [data_row + prediction for data_row, prediction in zip(model.test_data, predictions)]
output_file = 'output\\predictions.csv'
with open(output_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(model.column_names + ['Predicted Class by Neuron'] * len(model.weights))
    writer.writerows(output_data)


result_data = pd.read_csv(output_file)
print(result_data)
