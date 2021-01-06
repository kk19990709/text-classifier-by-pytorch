'''
Author: your name
Date: 2021-01-06 17:09:32
LastEditTime: 2021-01-06 17:36:53
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /digital/utils/utils.py
'''


def load_data(file):
    X = []
    y = []
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            data_label = line.split(sep='\t', maxsplit=1)
            y.append(data_label[0])
            X.append(data_label[1])
    label2num = {'earn': 0, 'acq': 1, 'trade': 2, 'ship': 3, 'grain': 4, 'crude': 5, 'interest': 6, 'money-fx': 7}
    y = [label2num[item] for item in y]
    return X, y
