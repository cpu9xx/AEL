import pandas as pd
import numpy as np

def read_mod(train_path, test_path, sep, header):
    train_set_dict, test_set_dict = {}, {}
    df_train = pd.read_csv(train_path, sep=sep, header=header)
    df_test = pd.read_csv(test_path, sep=sep, header=header)
    # 处理训练集
    for item in df_train.itertuples():
        uid, i_id = item[1], item[2]
        train_set_dict.setdefault(uid, {}).setdefault(i_id, 1)
    for item in df_test.itertuples():
        uid, i_id = item[1], item[2]
        test_set_dict.setdefault(uid, {}).setdefault(i_id, 1)
    return train_set_dict, test_set_dict

def read_adressa(train_path, test_path, sep, header):
    train_set_dict, test_set_dict = {}, {}
    df_train = pd.read_csv(train_path, sep=sep, header=header)
    df_test = pd.read_csv(test_path, sep=sep, header=header)
    # 处理训练集
    for item in df_train.itertuples():
        uid, i_id = item[1], item[2]
        train_set_dict.setdefault(uid, {}).setdefault(i_id, 1)
    for item in df_test.itertuples():
        uid, i_id = item[1], item[2]
        test_set_dict.setdefault(uid, {}).setdefault(i_id, 1)
    return train_set_dict, test_set_dict

def read_ml100k(train_path, test_path, sep, header):
    train_set_dict, test_set_dict = {}, {}
    df_train = pd.read_csv(train_path, sep=sep, header=header)-1
    df_test = pd.read_csv(test_path, sep=sep, header=header)-1
    # 处理训练集
    for item in df_train.itertuples():
        uid, i_id = item[1], item[2]
        train_set_dict.setdefault(uid, {}).setdefault(i_id, 1)
    for item in df_test.itertuples():
        uid, i_id = item[1], item[2]
        test_set_dict.setdefault(uid, {}).setdefault(i_id, 1)
    return train_set_dict, test_set_dict

def clean_read_ml100k(train_path, test_path, sep, header):
    train_set_dict, test_set_dict = {}, {}
    df_train = pd.read_csv(train_path, sep=sep, header=header)-1
    df_test = pd.read_csv(test_path, sep=sep, header=header)-1
    # 处理训练集
    for item in df_train.itertuples():
        uid, i_id, rating = item[1], item[2], item[3]
        if rating >= 4:
            train_set_dict.setdefault(uid, {}).setdefault(i_id, 1)
        if rating >= 5:
            print("^")
    for item in df_test.itertuples():
        uid, i_id = item[1], item[2]
        test_set_dict.setdefault(uid, {}).setdefault(i_id, 1)
    return train_set_dict, test_set_dict

def get_matrix(train_set_dict, test_set_dict, nb_user, nb_item):
    train_set, test_set = np.zeros(shape=(nb_user, nb_item)), np.zeros(shape=(nb_user, nb_item))

    for u in train_set_dict.keys():
        for i in train_set_dict[u].keys():
            train_set[u][i] = 1
    for u in test_set_dict.keys():
        for i in test_set_dict[u]:
            test_set[u][i] = 1
    print("Data prepared")
    return train_set, test_set


