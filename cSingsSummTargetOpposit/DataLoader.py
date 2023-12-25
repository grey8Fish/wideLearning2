import csv
import numpy as np

class DataLoader:
    """
    ����� DataLoader ��� �������� � ������� ������ �� ������ CSV.

    ��������:
        file_names (list of str): ������ ���� ������ ��� ��������.
        name_columns (list of str): ������ �������� ��������, ����������� �� ������ CSV.
        arg_classes (list of list of list): ��������� ������ ��� �������� ������ ������� ������.
    """

    def __init__(self, file_names):
        """
        �������������� DataLoader ������� ���� ������.

        ���������:
            file_names (list of str): ������ ���� ������ ��� ��������.
        """
        self.file_names = file_names
        self.name_columns = []
        self.arg_classes = []

    def load_data(self):
        """
        ��������� ������ �� ��������� ������ CSV � ������� arg_classes.
        """
        for index, file_name in enumerate(self.file_names):
            self._read_file(file_name, index)

    def _read_file(self, file_name, class_index):
        """
        ������ ���� ���� CSV � ��������� ��� ������ � ������ arg_classes.

        ���������:
            file_name (str): ��� ����� ��� ������.
            class_index (int): ������, �������������� ����� ����������� ������.
        """
        with open(file_name, encoding='utf-8') as file:
            file_reader = csv.DictReader(file, delimiter=',')
            
            # ������� �������� �������� ������ �� ������� �����
            if not self.name_columns:
                self.name_columns = file_reader.fieldnames[:-1]  # ��������������, ��� ��������� ������� �� �����

            # ���������������� arg_classes ��� ����� �������, ���� ��� ��� �� �������
            if len(self.arg_classes) <= class_index:
                self.arg_classes.append([])

            # �������������� ������ � ����� ����� � �������� � arg_classes
            for row in file_reader:
                data_row = [int(row[col_name]) for col_name in self.name_columns]
                self.arg_classes[class_index].append(data_row)

    def get_data(self):
        """
        ���������� ����������� ������.

        ����������:
            list of list of list: ����������� ������ �� ���� ������ CSV.
        """
        return self.arg_classes

    def get_column_names(self):
        """
        ���������� �������� ��������, ����������� �� ������ CSV.

        ����������:
            list of str: �������� ��������.
        """
        return self.name_columns
