import torch
from torch import nn
import numpy as np


class InteractionLoss(nn.Module):
    def __init__(self):
        super(InteractionLoss, self).__init__()

    def forward(self, predict, ground_true):
        gt_acted = ground_true[0]
        gt_action = ground_true[1]
        pred_acted = predict[0]
        pred_action = predict[1]
        change_acted = predict[2]
        change_action = predict[3]

        crossentropy = self.cross_entropy(change_acted, pred_acted, gt_action) + self.cross_entropy(change_action,
                                                                                                    pred_action,
                                                                                                    gt_acted)

        return torch.norm(change_acted - pred_acted + gt_action, 2) + torch.norm(change_action - pred_action +
                                                                                 gt_acted, 2) + crossentropy

    @staticmethod
    def cross_entropy(change, res, gt):
        return torch.sum(change * torch.log(torch.abs(res / gt)))
