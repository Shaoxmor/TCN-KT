import numpy as np


def embedding(fileName):
    dict = {}
    with open(fileName, 'r') as f:
        for line in f:
            fields = line.strip().split("	")
            label, exercise_id, concept_name = int(fields[0]), int(fields[1]), (fields[2])
            dict[exercise_id] = label
    same = np.zeros((110, 110))
    differ = np.zeros((110, 110))
    for i in range(110):
        for j in range(110):
            if i != j:
                if (dict[i + 1] == dict[j + 1]):
                    same[i][j] = 1
                    differ[i][j]= 0
                else:
                    same[i][j] = 0
                    differ[i][j] = 1

    return same,differ






