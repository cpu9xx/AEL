import torch
import torch.nn as nn
from torch.distributions.normal import Normal
class SparseMOE(nn.Module):
    # expert_list: [Large，Medium，Small]
    def __init__(self, expert_list:nn.ModuleList, select_experts:int):
        super(SparseMOE, self).__init__()
        self.num_expert = len(expert_list)
        self.noisy_gating = True
        # self.training = True
        self.k = select_experts
        self.experts = expert_list
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))

    # x->(MOE, MLP)-> A·x'

    def forward(self, data=None, edge_weight=None, ood_algorithm=None, loss_coef=1e-2, mode=None):
        gates, load = self.noisy_top_k_gating(data, self.training)
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef
        outputs = torch.stack([expert(data=data, edge_weight=edge_weight, ood_algorithm=ood_algorithm) for expert in self.experts], dim=2)
        gates = gates.unsqueeze(1).expand_as(outputs)
        y = torch.sum(outputs * gates, dim=2)
        return y, loss

    
    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean()**2 + eps)

    def gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
        threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)

        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in)/noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out)/noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((self.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_expert), dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = self.softmax(top_k_logits)
        
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)
        
            
        if self.noisy_gating and self.k < self.num_expert and train:
            load = (self.prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)).sum(0)
        else:
            load = self.gates_to_load(gates)

        if not self.training:
            print(f"{self.training}  {top_k_gates[0:3]}")
            print(f"{load}")
        return gates, load

    def train(self, mode=True):
        super(SparseMOE, self).train(mode)
        self.training = mode

    def eval(self):
        print(f"set eval()")
        super(SparseMOE, self).eval()
        self.training = False

class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        self.gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self.expert_index = sorted_experts.split(1, dim=1)
        self.batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self.part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self.batch_index.flatten()]
        self.nonzero_gates = torch.gather(gates_exp, 1, self.expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self.batch_index].squeeze(1)
        return torch.split(inp_exp, self.part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0)

        if multiply_by_gates:
            stitched = stitched.mul(self.nonzero_gates)
        zeros = torch.zeros(self.gates.size(0), expert_out[-1].size(1), requires_grad=True, device=stitched.device)
        combined = zeros.index_add(0, self.batch_index, stitched.float())
        return combined
    
    def expert_to_gates(self):
        return torch.split(self.nonzero_gates, self.part_sizes, dim=0)
    
class myMoE(SparseMOE):
    def __init__(self, model, select_experts:int, nb_item, nb_user):
        super(myMoE, self).__init__(model.AEs, select_experts)
        self.expert_selector = model
        self.train_experts = select_experts
        self.w_gate = nn.Parameter(torch.zeros(self.experts[0].input_size, self.num_expert), requires_grad=True)
        self.w_noise = nn.Parameter(torch.zeros(self.experts[0].input_size, self.num_expert), requires_grad=True)
        self.id2item = nn.Embedding(nb_user, nb_item)

    def forward(self, uid, purchase_vec, loss_coef=1e-2, mode=None):
        x = purchase_vec
        gates, load = self.noisy_top_k_gating(x, self.training)
        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= loss_coef
        
        outputs = torch.stack([self.expert_selector(uid, purchase_vec, stage=select) for select in range(self.num_expert)], dim=2).detach()
        gates = gates.unsqueeze(1).expand_as(outputs)
        y = torch.sum(outputs * gates, dim=2)
        return y, loss
    
    def train(self, mode=True):
        if mode:
            self.k = self.train_experts
        super(SparseMOE, self).train(mode)
        self.training = mode

    def eval(self, k=None):
        if k:
            self.k = k
            print(f"k: {self.k}")
        super(SparseMOE, self).eval()
        self.training = False
        
