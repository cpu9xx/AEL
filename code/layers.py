import torch
import torch.nn as nn

class AE(nn.Module):
    def __init__(self, nb_item, nb_user, nb_hidden):
        super(AE, self).__init__()
        self.input_size = nb_item
        self.item2hidden = nn.Linear(nb_item, nb_hidden)
        self.id2hidden = nn.Embedding(nb_user, nb_hidden)
        self.decoder = nn.Linear(nb_hidden, nb_item)
        self.sigmoid = nn.Sigmoid()


    def forward(self, uid, purchase_vec, code=None, decode=True):
        if code is None:
            try:
                encoded = self.sigmoid(self.id2hidden(uid).squeeze(dim=1)+self.item2hidden(purchase_vec))
            except:
                print(f"uid: {uid.shape}, purchase_vec: {purchase_vec.shape}")
                raise
            if not decode:
                return encoded
        else:
            encoded = code
        decoded = self.sigmoid(self.decoder(encoded))
        # decoded = self.decoder(encoded)
        return decoded



class parentAEs(nn.Module):
    def __init__(self, nb_item, nb_user, nb_hidden_ls):
        super(parentAEs, self).__init__()

        self.stage = 0
        self.AEs = nn.ModuleList([AE(nb_item, nb_user, nb_hidden_ls[0])])
        for i in range(1, len(nb_hidden_ls)):
            self.AEs.append(AE(nb_hidden_ls[i-1], nb_user, nb_hidden_ls[i]))

    def trained(self):
        self.stage += 1
        if self.stage >= len(self.AEs):
            self.stage = len(self.AEs) - 1
            print('All sub-AEs have been trained!')
        print('current stage:', self.stage)

    def forward(self, uid, purchase_vec):
        out = self.AEs[self.stage](uid, purchase_vec)
        return out

class parentAEs_pro(nn.Module):
    def __init__(self, nb_item, nb_user, nb_hidden_ls):
        super(parentAEs_pro, self).__init__()
        self.stage = 0
        self.AEs = nn.ModuleList([AE(nb_item, nb_user, nb_hidden_ls[0])])
        for i in range(1, len(nb_hidden_ls)):
            self.AEs.append(AE(nb_hidden_ls[i-1], nb_user, nb_hidden_ls[i]))

    def trained(self):
        self.stage += 1
        if self.stage >= len(self.AEs):
            self.stage = len(self.AEs) - 1
            print('All stages have been trained!')
        print('current stage:', self.stage)

    def forward(self, uid, purchase_vec, stage):
        for i in range(stage):
            purchase_vec = self.AEs[i](uid, purchase_vec, decode=False)

        out = self.AEs[stage](uid, purchase_vec)

        for i in range(stage - 1, -1, -1):
            out = self.AEs[i](None, None, code=out, decode=True)
        return out
