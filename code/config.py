import torch
from easydict import EasyDict as edict

ml_100k = edict()
ml_100k.nb_user = 943
ml_100k.nb_item = 1682
ml_100k.nb_hidden = 48
ml_100k.nb_hidden_ls = [128, 48, 12]

modcloth = edict()
modcloth.nb_user = 44784
modcloth.nb_item = 1020
modcloth.nb_hidden = 48
modcloth.nb_hidden_ls = [128, 48, 12]

adressa = edict()
adressa.nb_user = 212231
adressa.nb_item = 6596
adressa.nb_hidden = 48
adressa.nb_hidden_ls = [128, 48, 12]

moe = edict()
moe.lr = 1e-4
moe.epoches = 200
moe.train_experts = 2
moe.model = 'model.pth'

nb_mask = 256
lr=0.0005   
# lr = 0.005   

model = 'model.pth'
epoches = 1000
epoches_ls = [2500, 500, 300]
batch_size = 32
# batch_size = 8192
top_k = [3, 5, 10, 20]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
