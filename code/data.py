import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import dok_matrix

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

def read_baby(train_path, sep, clean=None):
    train_set_dict, test_set_dict = {}, {}
    df = pd.read_csv(train_path, sep=sep)
    df = df[['reviewerID', 'asin', 'overall']]
    
    print(f"df: {len(df)}")

    value_counts = df['reviewerID'].value_counts()
    if clean is not None:
        value_counts = df['reviewerID'].value_counts()
        values_to_remove = value_counts[value_counts < clean].index
        df = df[~df['reviewerID'].isin(values_to_remove)]

    print(f"df: {len(df)}")
    print(df['reviewerID'].nunique())
    print(df['asin'].nunique())
    total_rows = len(df)
    split_rows = int(total_rows * 0.2)
    np.random.seed(88)
    split_indices = np.random.choice(df.index, size=split_rows, replace=True)
    df_test = df.loc[split_indices]
    df_train = df.drop(split_indices)
    # df_test.to_csv('baby_test.csv', index=False)
    # df_train.to_csv('baby_train.csv', index=False)
    print(f"df_train: {len(df_train)}")
    print(f"df_test: {len(df_test)}")
    
    for item in df_train.itertuples():
        uid, i_id, r = item[1], item[2], item[3]
        train_set_dict.setdefault(uid, {}).setdefault(i_id, 1)

    for item in df_test.itertuples():
        uid, i_id, r = item[1], item[2], item[3]
        test_set_dict.setdefault(uid, {}).setdefault(i_id, 1)
    return train_set_dict, test_set_dict


# ('dataset/ml-100k/u1.base', 'dataset/ml-100k/u1.test', sep='\t', header=None)
def read_BX(train_path, sep):
    train_set_dict, test_set_dict = {}, {}
    df = pd.read_csv(train_path, sep=sep, encoding='latin1')
    total_rows = len(df)
    split_rows = int(total_rows * 0.2)
    np.random.seed(88)
    split_indices = np.random.choice(df.index, size=split_rows, replace=True)
    df_train = df.loc[split_indices]
    df_test = df.drop(split_indices)
    # print(len(df_train))
    # print(len(df_test))
    for item in df_train.itertuples():
        uid, i_id, r = item[1], item[2], item[3]
        if r != 0:
            train_set_dict.setdefault(uid, {}).setdefault(i_id, 1)

    for item in df_test.itertuples():
        uid, i_id, r = item[1], item[2], item[3]
        if r != 0:
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


def read_ml1m(filepath, sep='::', header='infer'):
    train_set_dict, test_set_dict = {}, {}
    df = pd.read_csv(filepath, sep=sep, header=header).iloc[:, :3]-1
    df = df.values.tolist()
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=1231)
    for uid, iid, score in train_set:
        train_set_dict.setdefault(uid, {}).setdefault(iid, 1)
    for uid, iid, score in test_set:
        test_set_dict.setdefault(uid, {}).setdefault(iid, 1)
    return train_set_dict, test_set_dict


def get_matrix(train_set_dict, test_set_dict, nb_user, nb_item):
    train_set, test_set = np.zeros(shape=(nb_user, nb_item)), np.zeros(shape=(nb_user, nb_item))
    try:
        print(len(train_set_dict[0].keys()))
        print(len(test_set_dict[0].keys()))
    except:
        pass
    for u in train_set_dict.keys():
        for i in train_set_dict[u].keys():
            train_set[u][i] = 1
    for u in test_set_dict.keys():
        for i in test_set_dict[u]:
            test_set[u][i] = 1
    print("Data prepared")
    return train_set, test_set

def get_baby_matrix(train_set_dict, test_set_dict, nb_user, nb_item):
    train_set, test_set = np.zeros(shape=(nb_user, nb_item)), np.zeros(shape=(nb_user, nb_item))
    user_index_map, item_index_map = {}, {}
    new_test_set_dict = {}
    user_index_counter, item_index_counter = 0, 0
    for u in train_set_dict.keys():
        if u not in user_index_map:
            user_index_map[u] = user_index_counter
            user_index_counter += 1
        for i in train_set_dict[u].keys():
            if i not in item_index_map:
                item_index_map[i] = item_index_counter
                item_index_counter += 1
            train_set[user_index_map[u], item_index_map[i]] = 1

    for u in test_set_dict.keys():
        if u not in user_index_map:
            user_index_map[u] = user_index_counter
            user_index_counter += 1
        for i in test_set_dict[u]:
            if i not in item_index_map:
                item_index_map[i] = item_index_counter
                item_index_counter += 1
            test_set[user_index_map[u], item_index_map[i]] = 1
            new_test_set_dict.setdefault(user_index_map[u], {}).setdefault(item_index_map[i], 1)
    return train_set, test_set, new_test_set_dict, user_index_map, item_index_map

def get_sparse_matrix(train_set_dict, test_set_dict, nb_user, nb_item):
    train_set = dok_matrix((nb_user, nb_item), dtype=np.float32)
    test_set = dok_matrix((nb_user, nb_item), dtype=np.float32)
    
    item_index_map = {}
    item_index_counter = 0
    
    for u in train_set_dict.keys():
        for i in train_set_dict[u].keys():
            if i not in item_index_map:
                item_index_map[i] = item_index_counter
                item_index_counter += 1
            train_set[u, item_index_map[i]] = 1.0

    for u in test_set_dict.keys():
        for i in test_set_dict[u]:
            if i not in item_index_map:
                item_index_map[i] = item_index_counter
                item_index_counter += 1
            test_set[u, item_index_map[i]] = 1.0
            
    return train_set, test_set

