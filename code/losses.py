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
    def __init__(self, nb_classes, sz_embed, mrg=0.1, alpha=32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        
    def forward(self, X, T):
        P = self.proxies

        P_one_hot = binarize(T=T, nb_classes=self.nb_classes)
        N_one_hot = 1 - P_one_hot

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim=0) != 0).squeeze(dim=1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies

        cos = F.linear(l2_norm(X), l2_norm(P))  # Calculate cosine similarity

        proxy = F.linear(l2_norm(P), l2_norm(P))
        proxy_data = (proxy * 125) + 125
        proxy_cos = proxy[T]
        pos_cos = torch.where(P_one_hot == 1, cos, torch.zeros_like(cos)).sum(dim=1, keepdim=True).repeat(1, self.nb_classes)

        # proxy_cos_bc1 = proxy_cos.view(proxy_cos.shape[0], proxy_cos.shape[1], 1)
        # proxy_cos_bcb = proxy_cos_bc1.repeat(1, 1, proxy_cos_bc1.shape[0])
        # proxy_cos_bcb_t = torch.transpose(proxy_cos_bcb, 0, 2)

        # p_bc1 = P_one_hot.view(P_one_hot.shape[0], P_one_hot.shape[1], 1)
        # p_bcb = p_bc1.repeat(1, 1, P_one_hot.shape[0])
        # p_bcb_t = torch.transpose(p_bcb, 0, 2)
        # P_one_hot_bcb = p_bcb + p_bcb_t

        # cos_bc1 = cos.view(cos.shape[0], cos.shape[1], 1)
        # cos_bcb = cos_bc1.repeat(1, 1, cos_bc1.shape[0])
        # cos_bcb_t = torch.transpose(cos_bcb, 0, 2)
        #
        # cos_proxy = torch.exp(32. * (cos_bcb_t - cos_bcb + 0.2))
        # cos_proxy = torch.where(P_one_hot_bcb == 2, torch.zeros_like(cos_proxy), cos_proxy)
        # cos_proxy_sum = torch.log(1. + cos_proxy.sum(dim=2, keepdim=True).squeeze())
        # cos_proxy_sum = torch.where(P_one_hot == 1, cos_proxy_sum, torch.zeros_like(cos_proxy_sum)).sum(dim=0)
        # P_one_hot_div = P_one_hot.sum(dim=0)
        # P_one_hot_div = torch.where(P_one_hot_div == 0, torch.ones_like(P_one_hot_div), P_one_hot_div)
        # cos_proxy_sum = cos_proxy_sum / P_one_hot_div
        # cos_proxy_sum = (cos_proxy_sum / cos_proxy_sum.sum()) * num_valid_proxies
        # cos_proxy_sum = torch.where(cos_proxy_sum == 0, torch.ones_like(cos_proxy_sum), cos_proxy_sum)

        # neg_cos = torch.where((P_one_hot == 0)*(cos < 0), torch.zeros_like(cos), cos)
        # cos_NCA = torch.sigmoid(self.beta * (pos_cos - cos + 1e-4) / (pos_cos + 1e-4))
        # cos_sum = cos_proxy_sum + cos_NCA
        # theta = torch.where(P_one_hot == 1, cos_sum, torch.zeros_like(cos_sum)).sum(dim=0)
        # my_cos = F.linear(l2_norm(X), l2_norm(X))
        # my_one_hot = F.linear(l2_norm(P_one_hot), l2_norm(P_one_hot))
        # my_n_one_hot = 1 - my_one_hot
        # my_p_one_hot = my_one_hot - torch.eye(len(T)).cuda()
        # my_p_one_hot2 = my_p_one_hot.sum(dim=0)
        # my_num_valid_proxies = torch.where(my_p_one_hot2 != 0, torch.ones_like(my_p_one_hot2), my_p_one_hot2).sum()
        # my_pos_exp = torch.exp(-self.alpha * (my_cos - self.mrg))
        # my_neg_exp = torch.exp(self.alpha * (my_cos + self.mrg))

        # all_length = torch.clamp(2. - 2. * cos, 1e-12, 4.)
        # all_root = torch.sqrt(all_length)
        # proxy_length = torch.clamp(2. - 2. * proxy_cos, 1e-12, 4.)
        # proxy_root = torch.sqrt(proxy_length)
        # pos_length = torch.clamp(2. - 2. * pos_cos, 1e-12, 4.)
        # pos_root = torch.sqrt(pos_length)

        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        # my_P_sim_sum = torch.where(my_p_one_hot == 1, my_pos_exp, torch.zeros_like(my_pos_exp))
        # my_P_sim_sum2 = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=1, keepdim=True).transpose(0, 1)
        # my_P_sim_sum3 = torch.cat([my_P_sim_sum, my_P_sim_sum2], dim=0).sum(dim=0)
        # my_N_sim_sum = torch.where(my_n_one_hot == 1, my_neg_exp, torch.zeros_like(my_neg_exp))
        # my_N_sim_sum2 = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=1, keepdim=True).transpose(0, 1)
        # my_N_sim_sum3 = torch.cat([my_N_sim_sum, my_N_sim_sum2], dim=0).sum(dim=0)
        #
        # my_pos_term = torch.log(1 + my_P_sim_sum).sum() / len(T)
        # my_neg_term = torch.log(1 + my_N_sim_sum3).sum() / len(T)

        proxy_pos_exp = torch.exp(-self.alpha * ((pos_cos - proxy_cos) + self.mrg))
        cos_pos_exp = torch.exp(-self.alpha * ((pos_cos - cos) + self.mrg))
        proxy_exp = torch.exp(self.alpha * (proxy_cos + self.mrg))

        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0)
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        PP_sim_sum = torch.where(N_one_hot == 1, proxy_pos_exp, torch.zeros_like(proxy_pos_exp)).sum(dim=0)
        CP_sim_sum = torch.where(N_one_hot == 1, cos_pos_exp, torch.zeros_like(cos_pos_exp)).sum(dim=0)
        proxy_sum = torch.where(N_one_hot == 1, proxy_exp, torch.zeros_like(proxy_exp)).sum(dim=0)

        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        PP_term = torch.log(1 + PP_sim_sum).sum() / self.nb_classes
        CP_term = torch.log(1 + CP_sim_sum).sum() / self.nb_classes
        proxy_term = torch.log(1 + proxy_sum).sum() / self.nb_classes
        loss = pos_term + neg_term

        return loss, pos_term, neg_term, proxy_term, PP_term, CP_term, proxy_data

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