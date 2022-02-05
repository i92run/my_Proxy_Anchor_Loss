import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from pytorch_metric_learning import miners, losses

def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

class Proxy_Anchor(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=32, beta=1):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        self.beta = beta
        
    def forward(self, X, T):
        P = self.proxies

        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(dim=1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies

        cos = F.linear(l2_norm(X), l2_norm(P))  # Calculate cosine similarity

        proxy = F.linear(l2_norm(P), l2_norm(P))
        # proxy_data = (proxy * 125) + 125
        proxy_cos = proxy[T]
        pos_cos = torch.where(P_one_hot == 1, cos, torch.zeros_like(cos)).sum(dim=1, keepdim=True).repeat(1, self.nb_classes)

        p_bc1 = P_one_hot.view(P_one_hot.shape[0], P_one_hot.shape[1], 1)
        p_bcb = p_bc1.repeat(1, 1, P_one_hot.shape[0])
        p_bcb_t = torch.transpose(p_bcb, 0, 2)
        P_one_hot_bcb = p_bcb + p_bcb_t
        P_one_hot_sum = P_one_hot.sum(dim=0)
        P_one_hot_sum = torch.where(P_one_hot_sum == 0, torch.ones_like(P_one_hot_sum), P_one_hot_sum)
        cos_exp = torch.exp(self.beta * cos)
        exp_bc1 = cos_exp.view(cos_exp.shape[0], cos_exp.shape[1], 1)
        exp_bcb = exp_bc1.repeat(1, 1, exp_bc1.shape[0])
        exp_bcb_t = torch.transpose(exp_bcb, 0, 2)
        exp_P = torch.where(P_one_hot == 1, cos_exp, torch.zeros_like(cos_exp))
        exp_N = torch.where(P_one_hot_bcb == 2, torch.zeros_like(exp_bcb_t), exp_bcb_t)
        exp_N_sum = exp_N.sum(dim=2, keepdim=True).squeeze()
        exp_div = torch.clamp(exp_P / (exp_P + exp_N_sum), 1e-4, 1.)
        CE = -torch.log(exp_div)

        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        proxy_pos_exp = torch.exp(-self.alpha * ((pos_cos - proxy_cos) + self.mrg))
        cos_pos_exp = torch.exp(-self.alpha * ((pos_cos - cos) + self.mrg))
        proxy_exp = torch.exp(self.alpha * (proxy_cos + self.mrg))

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        exp_sim_sum = torch.where(P_one_hot == 1, CE, torch.zeros_like(CE)).sum(dim=0) / P_one_hot_sum
        PP_sim_sum = torch.where(N_one_hot == 1, proxy_pos_exp, torch.zeros_like(proxy_pos_exp)).sum(dim=0)
        CP_sim_sum = torch.where(N_one_hot == 1, cos_pos_exp, torch.zeros_like(cos_pos_exp)).sum(dim=0)
        proxy_sum = torch.where(N_one_hot == 1, proxy_exp, torch.zeros_like(proxy_exp)).sum(dim=0)

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        exp_term = exp_sim_sum.sum() / num_valid_proxies
        PP_term = torch.log(1 + PP_sim_sum).sum() / self.nb_classes
        CP_term = torch.log(1 + CP_sim_sum).sum() / self.nb_classes
        proxy_term = torch.log(1 + proxy_sum).sum() / self.nb_classes
        loss = exp_term

        return loss, pos_term, neg_term, proxy_term, PP_term, CP_term

# We use PyTorch Metric Learning library for the following codes.
# Please refer to "https://github.com/KevinMusgrave/pytorch-metric-learning" for details.
class Proxy_NCA(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, scale=32):
        super(Proxy_NCA, self).__init__()
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.scale = scale
        self.loss_func = losses.ProxyNCALoss(num_classes = self.nb_classes, embedding_size = self.sz_embed, softmax_scale = self.scale).cuda()

    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss
    
class MultiSimilarityLoss(torch.nn.Module):
    def __init__(self, ):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.epsilon = 0.1
        self.scale_pos = 2
        self.scale_neg = 50
        
        self.miner = miners.MultiSimilarityMiner(epsilon=self.epsilon)
        self.loss_func = losses.MultiSimilarityLoss(self.scale_pos, self.scale_neg, self.thresh)
        
    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss
    
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.loss_func = losses.ContrastiveLoss(neg_margin=self.margin) 
        
    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss
    
class TripletLoss(nn.Module):
    def __init__(self, margin=0.1, **kwargs):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.miner = miners.TripletMarginMiner(margin, type_of_triplets = 'semihard')
        self.loss_func = losses.TripletMarginLoss(margin = self.margin)
        
    def forward(self, embeddings, labels):
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss
    
class NPairLoss(nn.Module):
    def __init__(self, l2_reg=0):
        super(NPairLoss, self).__init__()
        self.l2_reg = l2_reg
        self.loss_func = losses.NPairsLoss(l2_reg_weight=self.l2_reg, normalize_embeddings = False)
        
    def forward(self, embeddings, labels):
        loss = self.loss_func(embeddings, labels)
        return loss