'''
Author: your name
Date: 2021-01-06 14:46:49
LastEditTime: 2021-01-06 17:18:05
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /digital/model/FocalLoss.py
'''
import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, size_average=True):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, result, classes):
        loss = 0.0
        for i in range(len(result)):
            tmp_loss = 0.0
            for j in range(len(result[0])):
                if torch.isnan(result[i][j]):
                    result[i][j] = 1e-6
                if j == classes[i]:
                    tmp_loss += -self.alpha * torch.pow((1 - result[i][j]), self.gamma) * torch.log(result[i][j])
                else:
                    tmp_loss += -(1 - self.alpha) * torch.pow(result[i][j], self.gamma) * torch.log(1 - result[i][j])
            loss += tmp_loss
        if self.size_average:
            return loss / len(result)
        else:
            return loss
