import audmetric
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# classification loss
class CELoss(nn.Module):

    def __init__(self):
        super(CELoss, self).__init__()
        self.loss = nn.NLLLoss(reduction='sum')

    def forward(self, pred, target):
        pred = F.log_softmax(pred, 1) # [n_samples, n_classes]
        target = target.long()        # [n_samples]
        loss = self.loss(pred, target) / len(pred)
        return loss

# regression loss
class MSELoss(nn.Module):

    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='sum')

    def forward(self, pred, target):
        #torch.Size([32, 1000, 2])
        pred = pred.view(-1,1)
        target = target.view(-1,1)
        loss = self.loss(pred, target) / len(pred)
        return loss

class CCCLoss(nn.Module):

    def __init__(self):
        super(CCCLoss, self).__init__()

    def forward(self, pred, target):
        arousal_preds = pred[:,:,1]
        valence_preds = pred[:,:, 0]
        arousal_labels = target[:,:, 1]
        valence_labels = target[:,:, 0]
        batch_size, seq_len = arousal_preds.shape

        # 使用列表保存每个视频段的CCC损失
        arousal_losses = [self.ccc_loss(arousal_preds[i], arousal_labels[i]) for i in range(batch_size)]
        valence_losses = [self.ccc_loss(valence_preds[i], valence_labels[i]) for i in range(batch_size)]

        # 将列表转换为张量
        arousal_losses = torch.stack(arousal_losses)
        valence_losses = torch.stack(valence_losses)

        # 计算平均损失
        loss = 2 - (torch.mean(arousal_losses) + torch.mean(valence_losses))

        return loss

    def ccc_loss(self, preds, labels):
        # 计算皮尔逊相关系数
        r = torch.mean((preds - torch.mean(preds)) * (labels - torch.mean(labels))) / (
                torch.std(preds) * torch.std(labels) + 1e-10  # 防止除零错误
        )

        # 计算 CCC
        x_mean = torch.mean(preds)
        y_mean = torch.mean(labels)
        x_std = torch.std(preds)
        y_std = torch.std(labels)
        denominator = (x_std * x_std + y_std * y_std + (x_mean - y_mean) * (x_mean - y_mean) + 1e-10)  # 防止除零错误
        ccc = 2 * r * x_std * y_std / denominator

        return ccc

# class CCCLoss(nn.Module):
#
#     def __init__(self):
#         super(CCCLoss, self).__init__()
#         self.loss = self.ccc_loss
#
#     def forward(self, pred, target):
#         arousal_preds = pred[:,:,1]
#         valence_preds = pred[:,:, 0]
#         arousal_labels = target[:,:, 1]
#         valence_labels = target[:,:, 0]
#         batch_size, seq_len= arousal_preds.shape
#         # 使用列表推导式计算每个序列的 CCC 损失
#         arousal_losses = [self.ccc_loss(arousal_preds[i], arousal_labels[i]) for i in range(batch_size)]
#         arousal_losses=torch.stack(arousal_losses)
#         valence_losses = [self.ccc_loss(valence_preds[i], valence_labels[i]) for i in range(batch_size)]
#         valence_losses=torch.stack(valence_losses)
#         average_arousal_loss = torch.mean(arousal_losses)
#         average_valence_loss = torch.mean(valence_losses)
#         loss = 2-(average_arousal_loss+average_valence_loss)
#         return loss
#
#     def ccc_loss(self, preds, labels):
#         # return audmetric.concordance_cc(preds,labels)
#
#         if len(preds) < 2:
#             return 0
#
#         r = self.pearson_cc(preds, labels)
#
#         x_mean = torch.mean(preds)
#         y_mean = torch.mean(labels)
#         x_std = torch.std(preds)
#         y_std = torch.std(labels)
#
#         denominator = (
#                 x_std * x_std
#                 + y_std * y_std
#                 + (x_mean - y_mean) * (x_mean - y_mean)
#         )
#
#         if denominator == 0:
#             ccc = 0
#         else:
#             ccc = 2 * r * x_std * y_std / denominator
#
#         return ccc
#
#     def pearson_cc(self,x,y):
#         x_mean = torch.mean(x)
#         y_mean = torch.mean(y)
#         numerator = torch.sum((x - x_mean) * (y - y_mean))
#         denominator = torch.sqrt(torch.sum((x - x_mean) ** 2) * torch.sum((y - y_mean) ** 2))
#         if denominator == 0:
#             return 0.0
#         else:
#             return numerator / denominator
