import csv
import numpy as np


def parse_data(file_name):
    attributes_arr = []
    classes_arr = []
    with open(file_name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            if len(row) == 0:
                break
            attributes_arr += [[float(row[0]), float(row[1]), float(row[2]), float(row[3])]]
            classes_arr += [iris_name_to_number(row[4])]
    return np.array(attributes_arr), np.array(classes_arr)


def iris_name_to_number(flower_name):
    if flower_name == "Iris-setosa":
        return 0
    if flower_name == "Iris-versicolor":
        return 1
    if flower_name == "Iris-virginica":
        return 2
