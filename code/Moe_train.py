import torch
import torch.nn as nn
from layers import AE, parentAEs, parentAEs_pro
import config as cfg
from MoE import SparseMOE, myMoE
from data import get_matrix, read_ml100k, read_mod, read_adressa
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
import numpy as np
from train import M_Dataset
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
            pass
        else:
            self.log.write(message)
    def flush(self):
        pass

current_folder_path = os.path.dirname(__file__) + '/'
sys.stdout = Logger(current_folder_path + "log.txt")

def plot_precision(epoche_list, precision_list, label=None, color='blue', if_o=True):
    plt.title('')
    plt.plot(epoche_list, precision_list, label=label, marker='o' if if_o else None, color=color)
    plt.legend()
    plt.grid(True)

def train(model, epoches, dataloader, lr, train_set, test_set_dict, top_k):
    # 收集数据
    epoche_list, precision3_list, precision2_list, precision1_list = [], [], [], []
    epoche_list_loss, loss1_list, bias_list= [], [], []
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    # criterion = nn.BCELoss()
    for e in range(epoches+1):
        model.train()
        mean_loss1, mean_bias, count = 0, 0, 0
        for purchase_vec, uid in dataloader:
            purchase_vec = purchase_vec.to(cfg.device)  # Move data to device
            uid = uid.to(cfg.device)

            # _, loss = model(uid, purchase_vec, loss_coef=1e-2)
            y, bias = model(uid, purchase_vec, loss_coef=1)
            loss1 = criterion(y, purchase_vec)

            mean_loss1 += loss1.detach()
            mean_bias += bias.detach()
            count += 1
            loss = loss1 + bias
            opt.zero_grad()
            loss.backward()
            opt.step()


        mean_loss1 = mean_loss1/count
        mean_bias = mean_bias/count
        epoche_list_loss.append(e)
        loss1_list.append(mean_loss1.cpu())
        bias_list.append(mean_bias.cpu())

        if e % 5 == 0:
            print(f'{"==" * 5} {e} {"==" * 5}')
            precision3, _, ndcg3, _ = test(model, test_set_dict, train_set, top_k_ls=top_k, select_E=3)
            precision2, _, ndcg2, _ = test(model, test_set_dict, train_set, top_k_ls=top_k, select_E=2)
            precision1, _, ndcg1, _ = test(model, test_set_dict, train_set, top_k_ls=top_k, select_E=1)
            epoche_list.append(e)
            # precision3_list.append(precision3)
            # precision2_list.append(precision2)
            # precision1_list.append(precision1)
            precision3_list.append(ndcg3)
            precision2_list.append(ndcg2)
            precision1_list.append(ndcg1)
    plt.subplot(2, 1, 1)
    plot_precision(epoche_list, precision3_list, label="3 experts", color='green')
    plot_precision(epoche_list, precision2_list, label="2 experts", color='blue')
    plot_precision(epoche_list, precision1_list, label="1 experts", color='red')
    # plot_precision(epoche_list, ndcg_list, color='green')
    plt.subplot(2, 1, 2)
    plot_precision(epoche_list_loss, loss1_list, label="y", color='green', if_o=False)
    plot_precision(epoche_list_loss, bias_list, label="balance", color='blue', if_o=False)
    plt.show()

def test(model, test_set_dict, train_set, top_k_ls, select_E=None):
    clean_recall, clean_precision, clean_NDCG, clean_MRR = [], [], [], []
    for top_k in top_k_ls:
        model.eval(select_E)
        users = list(test_set_dict.keys())
        input_data = torch.tensor(train_set[users], dtype=torch.float).to(cfg.device)
        uids = torch.tensor(users, dtype=torch.long).view(-1, 1).to(cfg.device)
        out, _ = model(uids, input_data)
        out = (out - 999*input_data).detach().cpu().numpy()
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
            for k, v in tmp_list:
                if k in test_set_dict[u]:
                    hit += 1
            recalls += hit/len(test_set_dict[u])
            precisions += hit/top_k
            hits += hit
            total_purchase_nb += len(test_set_dict[u])

            dcg = 0
            idcg = 0
            for j in range(min(top_k, len(test_set_dict[u]))):
                if tmp_list[j][0] in test_set_dict[u]:
                    dcg += 1 / np.log2(j + 2)
                idcg += 1 / np.log2(j + 2)
            ndcgs += dcg / idcg

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

    print("################### CLEAN TEST ######################")
    print("Recall {:.4f}-{:.4f}-{:.4f}-{:.4f}".format(clean_recall[0], clean_recall[1],clean_recall[2],clean_recall[3]))
    print("Precision {:.4f}-{:.4f}-{:.4f}-{:.4f}".format(clean_precision[0], clean_precision[1],clean_precision[2],clean_precision[3]))
    print("NDCG {:.4f}-{:.4f}-{:.4f}-{:.4f}".format(clean_NDCG[0], clean_NDCG[1],clean_NDCG[2],clean_NDCG[3]))
    print("MRR {:.4f}-{:.4f}-{:.4f}-{:.4f}".format(clean_MRR[0], clean_MRR[1],clean_MRR[2],clean_MRR[3]))
    return precision, recall, ndcg, mrr

if __name__ == '__main__':
    nb_user = cfg.ml_100k.nb_user
    nb_item = cfg.ml_100k.nb_item
    nb_hidden_ls = cfg.ml_100k.nb_hidden_ls
    train_set_dict, test_set_dict = read_ml100k('dataset/ml-100k/u1.base', 'dataset/ml-100k/u1.test', sep='\t', header=None)
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


    model = parentAEs_pro(nb_item, nb_user, nb_hidden_ls).to(cfg.device)
    model.load_state_dict(torch.load(cfg.moe.model))

    dataset = M_Dataset(train_set)
    dataloader = DataLoader(dataset=dataset, batch_size=cfg.batch_size, shuffle=True, pin_memory=True)

    moe = myMoE(model, cfg.moe.train_experts, nb_item, nb_user).to(cfg.device)
    train(moe, epoches=cfg.moe.epoches, dataloader=dataloader, lr=cfg.moe.lr, train_set=train_set, test_set_dict=test_set_dict, top_k=cfg.top_k)

