import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import config as cfg
from torch.utils.data import Dataset, DataLoader
from data import read_ml1m, get_matrix, read_ml100k, clean_read_ml100k, read_BX, get_sparse_matrix, read_baby, get_baby_matrix, read_mod, read_adressa
from layers import AE, SAE
from matplotlib import pyplot as plt

import sys
import os
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
        self.start_log = True
    def write(self, message):
        self.terminal.write(message)
        if not self.start_log:
            if '倒数第 1 天' in message:#'Inited strategy' in message:
                self.start_log = True
            elif 'time' in message or 'Warning' in message:
                self.log.write(message)
        else:
            self.log.write(message)
    def flush(self):
        pass

current_folder_path = os.path.dirname(__file__) + '/'
sys.stdout = Logger(current_folder_path + "log.txt")


def plot_precision(epoche_list, precision_list, color='blue', if_o=True):
    plt.title('')
    if if_o:
        plt.plot(epoche_list, precision_list, marker='o', color=color)
    else:        
        plt.plot(epoche_list, precision_list, color=color)

    plt.grid(True)
    plt.savefig('precision_ml100k888.png')
    


class M_Dataset(Dataset):
    def __init__(self, train_set):
        self.train_set = train_set

    def __getitem__(self, idx):
        try:
            purchase_vec = torch.tensor(self.train_set[idx], dtype=torch.float)
            uid = torch.tensor([idx,], dtype=torch.long)
        except TypeError:
            u_indices, i_indices = self.train_set[idx].nonzero()
            purchase_vec = torch.zeros(self.train_set.shape[1], dtype=torch.float)
            purchase_vec[i_indices] = 1.0
            uid = torch.tensor([idx,], dtype=torch.long)
            raise
        return purchase_vec, uid

    def __len__(self):
        try:
            return len(self.train_set)
        except:
            return self.train_set.shape[0]
            raise

def select_negative_items(batch_history_data, nb):# (purchase_vec.cpu()[batch_size, nb_item], nb_mask)
    data = np.array(batch_history_data)
    idx = np.zeros_like(data)
    #print(f"data: {data.shape[0]}")
    for i in range(data.shape[0]):
        #print(f"i: {i} ##{data} \n ##{np.where(data[i] == 0)} \n ##{np.where(data[i] == 0)[0]} \n ##{np.where(data[i] == 0)[0].tolist()}")
        items = np.where(data[i] == 0)[0].tolist()
        try:
            tmp_zr = random.sample(items, nb)
        except:
            tmp_zr = random.sample(items, len(items))
        idx[i][tmp_zr] = 1
    return idx

def loss_function(y, t, drop_rate=0.2):
    # 计算均方误差损失
    print(f"y: {y.shape}, t: {t.shape}")
    loss = F.mse_loss(y, t, reduction='none')

    loss_mul = loss * t
    print(f"loss_mul: {loss_mul.data.shape}")

    ind_sorted = torch.argsort(loss_mul).to(cfg.device)
    loss_sorted = loss[ind_sorted]
    print(f"loss_sorted: {loss_sorted.shape}")
    remember_rate = 1 - drop_rate
    num_remember = int(remember_rate * len(loss_sorted))
    print(f"num_remember: {num_remember}")
    print(f"ind_sorted: {ind_sorted.shape}")
    ind_update = ind_sorted[:num_remember]
    print(f"ind_update: {ind_update.shape}")
    print(f"y.shape: {y.shape}, t.shape: {t.shape}")
    print(f"y[ind_update].shape: {y[ind_update].shape}, t[ind_update].shape: {t[ind_update].shape}")
    # 从预测值和目标值中选择相应的样本，然后计算均方误差损失
    loss_update = F.mse_loss(y[ind_update], t[ind_update], reduction='mean')
    print(f"loss_update: {loss_update.shape}")
    return loss_update


def SAE_train(nb_user, nb_item, nb_hidden_ls, epoches_ls, dataloader, lr, nb_mask, train_set, test_set_dict, top_k, user_index_map=None):
    # 收集数据
    epoche_dict, precision_dict, avg_precision_dict = {}, {}, {}
    model = SAE(nb_item, nb_user, nb_hidden_ls).to(cfg.device)  # Move model to device
    
    
    model.load_state_dict(torch.load(cfg.model))
    print("Use pretrained model")


    # criterion = nn.MSELoss()
    criterion = nn.BCELoss()
    # criterion = nn.CrossEntropyLoss() #修改layers,sigmoid
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoches in epoches_ls:
        epoche_list, precision_list, avg_precision_list = [], [], []
        for e in range(epoches):
            model.train()
            for purchase_vec, uid in dataloader:
                # purchase_vec: [batch_size, nb_item]
                purchase_vec = purchase_vec.to(cfg.device)  # Move data to device
                uid = uid.to(cfg.device)
                
                if model.stage == 0:
                    mask_vec = torch.tensor(select_negative_items(purchase_vec.cpu(), nb_mask)).to(cfg.device) + purchase_vec
                    out = model(uid, purchase_vec) * mask_vec   # 1G
                    
                    # loss = torch.sum((out - purchase_vec).square()) / mask_vec.sum()  # 1G 2G 3G
                    loss = criterion(out, purchase_vec) / mask_vec.sum()
                else:
                    for i in range(model.stage):
                        purchase_vec = model.AEs[i](uid, purchase_vec, decode=False)
                        # print(f"AE: {i}, purchase_vec: {purchase_vec.shape}")
                    purchase_vec = purchase_vec.detach()
                    # print(uid.shape)
                    out = model(uid, purchase_vec)
                    loss = criterion(out, purchase_vec)

                    # out = model(uid, purchase_vec) * purchase_vec
                    # loss = criterion(out, purchase_vec)
                    # loss = loss_function(out, purchase_vec)

                opt.zero_grad()
                loss.backward()
                opt.step()
            if (e + 1) % 5 == 0:
                print(f'{"==" * 5} {e + 1} {"==" * 5}')
            if (e + 1) % 10 == 0:
                print(f'{"==" * 5} {e + 1} {"==" * 5}')
                avg_precision, _, _, _ = SAE_test(model, test_set_dict, train_set, top_k_ls=top_k, if_avg=True)
                precision, _, _, _ = SAE_test(model, test_set_dict, train_set, top_k_ls=top_k, if_avg=False)
                epoche_list.append(e + 1)
                precision_list.append(precision)
                avg_precision_list.append(avg_precision)
        epoche_dict[model.stage] = epoche_list
        precision_dict[model.stage] = precision_list
        avg_precision_dict[model.stage] = avg_precision_list
        model.trained()
        SAE_test_all(model, test_set_dict, train_set, top_k=top_k, user_index_map=user_index_map)

    torch.save(model.state_dict(), 'model.pth')
    for stage in range(model.stage+1):
        plt.subplot(model.stage+1, 1, stage+1)
        plot_precision(epoche_dict[stage], precision_dict[stage], color='blue')
        plot_precision(epoche_dict[stage], avg_precision_dict[stage], color='red')
    plt.show()
    


def SAE_test(model, test_set_dict, train_set, top_k_ls, if_avg=False):
    clean_recall, clean_precision, clean_NDCG, clean_MRR = [], [], [], []
    for top_k in top_k_ls:
        model.eval()
        users = list(test_set_dict.keys())
        input_data = torch.tensor(train_set[users], dtype=torch.float).to(cfg.device)
        ori_input_data = input_data
        uids = torch.tensor(users, dtype=torch.long).view(-1, 1).to(cfg.device)

        if if_avg:
            ori_stage = model.stage
            avg_out = torch.zeros_like(input_data)
            for stage in range(len(model.AEs)):
                model.stage = stage
                input_data = ori_input_data
                for i in range(model.stage):
                    input_data = model.AEs[i](uids, input_data, decode=False)
                    # print(f"AE: {i}, input_data: {input_data.shape}")

                out = model(uids, input_data)

                for i in range(model.stage - 1, -1, -1):
                    out = model.AEs[i](None, None, code=out, decode=True)
                    # print(f"AE: {i}, out: {out.shape}")
                avg_out += out
            model.stage = ori_stage
            out = avg_out/len(model.AEs)
        else:
            for i in range(model.stage):
                input_data = model.AEs[i](uids, input_data, decode=False)
                # print(f"AE: {i}, input_data: {input_data.shape}")

            out = model(uids, input_data)

            for i in range(model.stage - 1, -1, -1):
                out = model.AEs[i](None, None, code=out, decode=True)
                # print(f"AE: {i}, out: {out.shape}")

        # print(f"out: {out}")
        # print(f"ori_input_data: {ori_input_data}")

        out = (out - 999*ori_input_data).detach().cpu().numpy()
        # print(f"out: {out}")
        precisions = 0
        recalls = 0
        ndcgs = 0
        mrrs = 0
        hits = 0
        total_purchase_nb = 0
        for i, u in enumerate(users):
            hit = 0
            tmp_list = [(idx, value) for idx, value in enumerate(out[i])]
            tmp_list = sorted(tmp_list, key=lambda x:x[1], reverse=True)[:top_k] # 对value 排序
            # print(f"tmp_list: {tmp_list}")
            for k, v in tmp_list:
                if k in test_set_dict[u]:
                    # print(f"test_set_dict[u]: {test_set_dict[u]}")
                    hit += 1
            recalls += hit/len(test_set_dict[u])
            precisions += hit/top_k
            hits += hit
            total_purchase_nb += len(test_set_dict[u])

            # 计算 NDCG
            dcg = 0
            idcg = 0
            for j in range(min(top_k, len(test_set_dict[u]))):
                if tmp_list[j][0] in test_set_dict[u]:
                    dcg += 1 / np.log2(j + 2)
                idcg += 1 / np.log2(j + 2)
            ndcgs += dcg / idcg

            # 计算 MRR
            mrr = 0
            for j in range(len(tmp_list)):
                if tmp_list[j][0] in test_set_dict[u]:
                    mrr = 1 / (j + 1)
                    break
            mrrs += mrr

        recall = recalls/len(users)
        precision = precisions/len(users)
        ndcg = ndcgs/len(users)
        mrr = mrrs/len(users)


        clean_recall.append(recall)
        clean_precision.append(precision)
        clean_NDCG.append(ndcg)
        clean_MRR.append(mrr)

    # print(f'recall:{recall:.4f}, precision:{precision:.4f}, NDCG:{ndcg:.4f}, MRR:{mrr:.4f}')
    print(f"################ {model.stage} AVG: {if_avg} ######################")
    print("Recall {:.4f}-{:.4f}-{:.4f}-{:.4f}".format(clean_recall[0], clean_recall[1],clean_recall[2],clean_recall[3]))
    print("Precision {:.4f}-{:.4f}-{:.4f}-{:.4f}".format(clean_precision[0], clean_precision[1],clean_precision[2],clean_precision[3]))
    print("NDCG {:.4f}-{:.4f}-{:.4f}-{:.4f}".format(clean_NDCG[0], clean_NDCG[1],clean_NDCG[2],clean_NDCG[3]))
    print("MRR {:.4f}-{:.4f}-{:.4f}-{:.4f}".format(clean_MRR[0], clean_MRR[1],clean_MRR[2],clean_MRR[3]))
    return precision, recall, ndcg, mrr

def SAE_test_all(model, test_set_dict, train_set, top_k=5, user_index_map=None):
    print("##"*10)
    stage = model.stage
    model.stage = 0
    SAE_test(model, test_set_dict, train_set, top_k_ls=top_k)
    model.stage = 1
    SAE_test(model, test_set_dict, train_set, top_k_ls=top_k)
    model.stage = 2
    SAE_test(model, test_set_dict, train_set, top_k_ls=top_k)
    model.stage = stage
    print("##"*10)

def avg_test(model, test_set_dict, train_set, top_k=5):
    model.eval()
    users = list(test_set_dict.keys())
    input_data = torch.tensor(train_set[users], dtype=torch.float).to(cfg.device)
    ori_input_data = input_data
    uids = torch.tensor(users, dtype=torch.long).view(-1, 1).to(cfg.device)

    for i in range(model.stage):
        input_data = model.AEs[i](uids, input_data, decode=False)
        # print(f"AE: {i}, input_data: {input_data.shape}")

    out = model(uids, input_data)

    for i in range(model.stage - 1, -1, -1):
        out = model.AEs[i](None, None, code=out, decode=True)
        # print(f"AE: {i}, out: {out.shape}")
    print(f"out: {out}")
    print(f"ori_input_data: {ori_input_data}")
    out = (out - 999*ori_input_data).detach().cpu().numpy()
    print(f"out: {out}")
    precisions = 0
    recalls = 0
    hits = 0
    total_purchase_nb = 0
    for i, u in enumerate(users):
        hit = 0
        tmp_list = [(idx, value) for idx, value in enumerate(out[i])]
        tmp_list = sorted(tmp_list, key=lambda x:x[1], reverse=True)[:top_k]
        for k, v in tmp_list:
            if k in test_set_dict[u]:
                hit += 1
        recalls += hit/len(test_set_dict[u])
        precisions += hit/top_k
        hits += hit
        total_purchase_nb += len(test_set_dict[u])
    recall = recalls/len(users)
    precision = precisions/len(users)
    print('stage:{}, recall:{}, precision:{}'.format(model.stage, recall, precision))
    return precision, recall

"""
todo:


"""






if __name__ == '__main__':

    # nb_user = cfg.ml_1m.nb_user
    # nb_item = cfg.ml_1m.nb_item
    # nb_hidden = cfg.ml_1m.nb_hidden
    # nb_hidden_ls = cfg.ml_1m.nb_hidden_ls
    # train_set_dict, test_set_dict = read_ml1m('dataset/ml-1m/ratings.dat')
    # train_set, test_set = get_matrix(train_set_dict, test_set_dict, nb_user=nb_user, nb_item=nb_item)

    nb_user = cfg.ml_100k.nb_user
    nb_item = cfg.ml_100k.nb_item
    nb_hidden = cfg.ml_100k.nb_hidden
    nb_hidden_ls = cfg.ml_100k.nb_hidden_ls
    train_set_dict, test_set_dict = read_ml100k('dataset/ml-100k/u1.base', 'dataset/ml-100k/u1.test', sep='\t', header=None)
    #train_set_dict, test_set_dict = clean_read_ml100k('dataset/ml-100k/u1.base', 'dataset/ml-100k/u1.test', sep='\t', header=None)
    print(train_set_dict)
    print(test_set_dict)
    train_set, test_set = get_matrix(train_set_dict, test_set_dict, nb_user=nb_user, nb_item=nb_item)




    # nb_user = cfg.modcloth.nb_user
    # nb_item = cfg.modcloth.nb_item
    # nb_hidden = cfg.modcloth.nb_hidden
    # nb_hidden_ls = cfg.modcloth.nb_hidden_ls
    # train_set_dict, test_set_dict = read_mod('dataset/mod/modcloth.train.rating', 'dataset/mod/modcloth.test.negative', sep='\t', header=None)
    # train_set, test_set = get_matrix(train_set_dict, test_set_dict, nb_user=nb_user, nb_item=nb_item)




    # nb_user = cfg.adressa.nb_user
    # nb_item = cfg.adressa.nb_item
    # nb_hidden = cfg.adressa.nb_hidden
    # nb_hidden_ls = cfg.adressa.nb_hidden_ls
    # train_set_dict, test_set_dict = read_adressa('dataset/adressa/adressa.train.rating', 'dataset/adressa/adressa.test.negative', sep='\t', header=None) 
    # train_set, test_set = get_matrix(train_set_dict, test_set_dict, nb_user=nb_user, nb_item=nb_item)





    # nb_user = cfg.baby.nb_user
    # nb_item = cfg.baby.nb_item
    # nb_hidden = cfg.baby.nb_hidden
    # nb_hidden_ls = cfg.baby.nb_hidden_ls
    # raw_train_set_dict, raw_test_set_dict = read_baby('dataset/reviews_Baby_5_final_dataset.csv', sep=',', clean=5)
    
    # train_set, test_set, test_set_dict, user_index_map, item_index_map = get_baby_matrix(raw_train_set_dict, raw_test_set_dict, nb_user=nb_user, nb_item=nb_item)



    # nb_user = cfg.bx.nb_user
    # nb_item = cfg.bx.nb_item
    # nb_hidden = cfg.bx.nb_hidden
    # nb_hidden_ls = cfg.bx.nb_hidden_ls
    # train_set_dict, test_set_dict = read_BX('dataset/BX/BX-Book-Ratings.csv', sep=';')

    # train_set, test_set = get_sparse_matrix(train_set_dict, test_set_dict, nb_user, nb_item)


    import time
    start_time = time.time()
    dataset = M_Dataset(train_set)
    dataloader = DataLoader(dataset=dataset, batch_size=cfg.batch_size, shuffle=True, pin_memory=True)
    SAE_train(nb_user, nb_item, nb_hidden_ls, epoches_ls=cfg.epoches_ls, dataloader=dataloader, lr=cfg.lr, nb_mask=cfg.nb_mask, train_set=train_set, test_set_dict=test_set_dict, top_k=cfg.top_k, user_index_map=None)
    end_time = time.time()
    print(f'time: {(end_time-start_time)/60} min')