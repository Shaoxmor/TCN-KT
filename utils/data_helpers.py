import os
import random
import csv
import logging


def logger_fn(name, input_file, level=logging.INFO):
    tf_logger = logging.getLogger(name)
    tf_logger.setLevel(level)
    log_dir = os.path.dirname(input_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    fh = logging.FileHandler(input_file, mode='w')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    tf_logger.addHandler(fh)
    return tf_logger


def read_data_from_csv_file(fileName):
    rows = []
    max_skill_num = 0
    max_num_problems = 380
    with open(fileName, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            rows.append(row)
    index = 0
    tuple_rows = []
    while(index < len(rows)):
        problems_num = int(rows[index][0])
        tmp_max_skill = max(map(int, rows[index+1]))
        if(tmp_max_skill > max_skill_num):
            max_skill_num = tmp_max_skill
        if(problems_num <= 2):
            index += 3
        else:
            if problems_num > max_num_problems:
                count = problems_num // max_num_problems
                iii = 0
                while(iii <= count):
                    if iii != count:
                        tup = (max_num_problems, rows[index+1][iii * max_num_problems : (iii+1)*max_num_problems], rows[index+2][iii * max_num_problems : (iii+1)*max_num_problems])
                    elif problems_num - iii*max_num_problems > 2:
                        tup = (problems_num - iii*max_num_problems, rows[index+1][iii * max_num_problems : (iii+1)*max_num_problems], rows[index+2][iii * max_num_problems : (iii+1)*max_num_problems])
                    else:
                        break
                    tuple_rows.append(tup)
                    iii += 1
                index += 3
            else:
                tup = (problems_num, rows[index+1], rows[index+2])
                tuple_rows.append(tup)
                index += 3

    random.shuffle(tuple_rows)
    return tuple_rows, max_num_problems, max_skill_num+1


def read_test_data_from_csv_file(fileName):
    rows = []
    max_skill_num = 0
    max_num_problems = 380
    with open(fileName, "r") as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            rows.append(row)
    index = 0
    tuple_rows = []
    while(index < len(rows)):
        problems_num = int(rows[index][0])
        tmp_max_skill = max(map(int, rows[index+1]))
        if(tmp_max_skill > max_skill_num):
            max_skill_num = tmp_max_skill
        if(problems_num <= 2):
            index += 3
        else:
            if problems_num > max_num_problems:
                count = problems_num // max_num_problems
                iii = 0
                while(iii <= count):
                    if iii != count:
                        tup = (max_num_problems, rows[index+1][iii * max_num_problems : (iii+1)*max_num_problems], rows[index+2][iii * max_num_problems : (iii+1)*max_num_problems])
                    elif problems_num - iii*max_num_problems > 2:
                        tup = (problems_num - iii*max_num_problems, rows[index+1][iii * max_num_problems : (iii+1)*max_num_problems], rows[index+2][iii * max_num_problems : (iii+1)*max_num_problems])
                    else:
                        break                    
                    tuple_rows.append(tup)
                    iii += 1
                index += 3
            else:
                tup = (problems_num, rows[index+1], rows[index+2])
                tuple_rows.append(tup)
                index += 3

    return tuple_rows, max_num_problems, max_skill_num+1
