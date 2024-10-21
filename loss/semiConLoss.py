"""
Author: ZiHao Zhou
Date: May 07, 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def unlabelProbability(label_feat, unlabel_feat, label, device, la=0):
    alpha = 0.2
    simuu = torch.matmul(unlabel_feat, unlabel_feat.t())
    simuu = F.softmax(simuu, dim=1)
    mat = torch.inverse(torch.eye(simuu.shape[0]).to(device) - alpha * simuu)
    simul = torch.exp(torch.matmul(unlabel_feat, label_feat.t()) / 0.1)

    py_sum = torch.sum(label, dim=0)
    py_sum = py_sum.repeat(simul.size(0), 1)
    logits = torch.log(torch.matmul(simul, label) / py_sum)
    logits = logits + la
    pl = (1 - alpha) * F.softmax(logits, dim=1)
    return torch.matmul(mat, pl)


def crossLoss(a, b):
    return -torch.mean(torch.sum(a * torch.log(b), dim=1), dim=0)


def sharp(a, T):
    a = a ** T
    a_sum = torch.sum(a, dim=1, keepdim=True)
    a = a / a_sum
    return a.detach()


class SemiConLoss(nn.Module):
    def __init__(self, labeled_bs, unlabeled_bs, num_class, args, temperature=0.1):
        super(SemiConLoss, self).__init__()
        self.temperature = temperature
        self.labeled_bs = labeled_bs
        self.unlabeled_bs = unlabeled_bs
        self.num_class = num_class
        self.args = args
        self.py_unlabeled = args.py_uni.to(args.device)
        self.py_Num = torch.zeros(num_class).to(args.device)

    def forward(self, feat, label, la=0):
        label = label.float()
        # la = torch.log(self.py_unlabeled ** 1.0 + 1e-12).to(self.args.device)
        label_feat = feat[:self.num_class + 2 * self.labeled_bs, :]
        unlabel_feat = feat[self.num_class + 2 * self.labeled_bs:, :]
        anchor_feat, unlabel_feat1, unlabel_feat2 = unlabel_feat.chunk(3)
        with torch.no_grad():
            up_aim = unlabelProbability(label_feat, anchor_feat, label, self.args.device, la)
            self.py_Num += torch.sum(up_aim, dim=0)
            up_aim = sharp(up_aim, 4.0)

        up1 = unlabelProbability(label_feat, unlabel_feat1, label, self.args.device, la)
        up2 = unlabelProbability(label_feat, unlabel_feat2, label, self.args.device, la)
        up_u = (up1 + up2) / 2

        loss = (crossLoss(up_aim, up1) + crossLoss(up_aim, up2) + crossLoss(up_u, up_u)) / 3
        self.py_Num += torch.sum(up_u.detach(), dim=0)
        return loss


class softConLoss(nn.Module):
    def __init__(self, labeled_bs, unlabeled_bs, num_class, args, temperature=0.1):
        super(softConLoss, self).__init__()
        self.temperature = temperature
        self.labeled_bs = labeled_bs
        self.unlabeled_bs = unlabeled_bs
        self.num_class = num_class
        self.args = args
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, label, feat, device, la=0):
        label = label.float()
        sim = torch.exp(torch.matmul(feat, feat.t()) / self.temperature)
        py_sum = torch.sum(label, dim=0)
        py_sum = py_sum.repeat(sim.size(0), 1) - label
        matrix = torch.ones((label.shape[0], label.shape[0]))
        matrix = matrix.fill_diagonal_(0).to(device)
        sim = sim * matrix
        logits = torch.log(torch.matmul(sim, label) / py_sum)

        logits = logits[self.num_class:, :]
        logits = logits + la
        label = label[self.num_class:, :]

        loss = self.criterion(logits, label)
        return loss
