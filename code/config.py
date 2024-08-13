import torch
from easydict import EasyDict as edict

ml_1m = edict()
ml_1m.nb_user = 6040
ml_1m.nb_item = 3952
ml_1m.nb_hidden = 48
ml_1m.nb_hidden_ls = [128, 48, 12]

ml_100k = edict()
ml_100k.nb_user = 943
ml_100k.nb_item = 1682
ml_100k.nb_hidden = 48
ml_100k.nb_hidden_ls = [128, 48, 12]
# ml_100k.nb_hidden_ls = [128, 8]



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




bx = edict()
bx.nb_user = 278858
bx.nb_item = 271379
bx.nb_hidden = 12
bx.nb_hidden_ls = [128, 48, 12]

# baby = edict()
# baby.nb_user = 17168
# baby.nb_item = 6782
# baby.nb_hidden = 12
# baby.nb_hidden_ls = [128, 48, 12]

# clean 15
# baby = edict()
# baby.nb_user = 179
# baby.nb_item = 2153
# baby.nb_hidden = 12
# baby.nb_hidden_ls = [128, 48, 12]

# clean 5
baby = edict()
baby.nb_user = 3557
baby.nb_item = 5788
baby.nb_hidden = 48
baby.nb_hidden_ls = [128, 48, 12]
# baby.nb_hidden_ls = [1024, 256, 64]

moe = edict()
moe.lr = 1e-4
# moe.lr = 0.0001
moe.epoches = 200
moe.train_experts = 2
# moe.model = 'model1024.pth'
moe.model = 'model.pth'

nb_mask = 256
# nb_mask = 128
lr=0.0001   
# lr = 0.005   

model = 'model.pth'
epoches = 1000
# epoches_ls = [200, 200, 200]
epoches_ls = [21, 21, 21]
# epoches_ls = [30, 30]
# ml100k:32  ml1m:128  baby:1024  
batch_size = 1024
# batch_size = 8192
# top_k = 5
top_k = [3, 5, 10, 20]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")