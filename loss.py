import torch
import torch.nn as nn
import torch.nn.functional as F


def log_nnl(inpt, target, weight=None, size_average=None, ignore_index=-100,
             reduce=None, reduction='elementwise_mean'):
    return F.nll_loss(torch.log(inpt), target, weight, size_average, ignore_index,
                      reduce, reduction)


def smooth_label_cross_entropy(inpt, target, num_classes, eps=0.1, reduction='mean'):
    smoothed_one_hot = F.one_hot(target, num_classes).type_as(inpt)
    smoothed_one_hot = smoothed_one_hot * (1 - eps) + (1 - smoothed_one_hot) * eps / (num_classes - 1)

    log_prb = F.log_softmax(inpt.reshape(-1, num_classes), dim=1)
    loss = -(smoothed_one_hot * log_prb).sum(dim=1)
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'none':
        return loss


class SLCELoss(nn.Module):
    def __init__(self, num_classes, eps=0.1):
        super(SLCELoss, self).__init__()
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, inpt, target, reduction='mean'):
        return smooth_label_cross_entropy(
            inpt, target, self.num_classes, eps=self.eps, reduction=reduction)

    def extra_repr(self):
        return 'num_classes={}, eps={}'.format(
            self.num_classes, self.eps
        )


