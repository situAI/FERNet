import torch
import torch.nn.functional as F
import torch.nn as nn

from utils.registery import LOSS_REGISTRY

@LOSS_REGISTRY.register()
class CCCLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x, y):
        y = y.contiguous().view(-1)
        x = x.contiguous().view(-1)
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        rho = torch.sum(vx * vy) / (
                    torch.sqrt(torch.sum(torch.pow(vx, 2))) * torch.sqrt(torch.sum(torch.pow(vy, 2))) + self.eps)
        x_m = torch.mean(x)
        y_m = torch.mean(y)
        x_s = torch.std(x)
        y_s = torch.std(y)
        ccc = 2 * rho * x_s * y_s / (torch.pow(x_s, 2) + torch.pow(y_s, 2) + torch.pow(x_m - y_m, 2))
        return 1 - ccc


@LOSS_REGISTRY.register()
class ERILoss(nn.Module):
    def __init__(self, alpha=1, beta=1, eps=1e-8):
        super().__init__()
        # self.ccc = CCCLoss(eps=eps)
        # self.weights = [0.42560/0.34758, 0.42560/0.42560, 0.42560/0.37593, 0.42560/0.35176, 0.42560/0.33881, 0.42560/0.37492, 0.42560/0.35858]
        # self.criterion = nn.L1Loss().cuda()
        self.criterion = nn.MSELoss().cuda()
    def forward(self, x, y):
        loss = 0.
        for i in range(y.shape[1]):
            # loss += self.ccc(x[:, i], y[:, i]) * self.weights[i]
            loss += self.criterion(x[:, i].float(), y[:, i].float())

        return loss

@LOSS_REGISTRY.register()
class PLCCLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(PLCCLoss, self).__init__()

    def forward(self, input, target):
        input0 = input - torch.mean(input)
        target0 = target - torch.mean(target)

        out = 1 - torch.sum(input0 * target0) / (torch.sqrt(torch.sum(input0 ** 2)) * torch.sqrt(torch.sum(target0 ** 2)))
        return out

@LOSS_REGISTRY.register()
class VALoss(nn.Module):
    def __init__(self, alpha=1, beta=1, eps=1e-8):
        super().__init__()
        self.ccc = CCCLoss(eps=eps)
        self.alpha = alpha
        self.beta = beta

    def forward(self, x, y):
        loss = self.alpha * self.ccc(x[:, 0], y[:, 0]) + self.beta * self.ccc(x[:, 1], y[:, 1])

        return loss

@LOSS_REGISTRY.register()
class AULoss(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        pw = torch.FloatTensor(pos_weight)
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pw)

    def forward(self, x, y):
        return self.bce(x, y)

@LOSS_REGISTRY.register()
class ExprLoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        w = torch.FloatTensor(weight)
        self.ce = nn.CrossEntropyLoss(weight=w)

    def forward(self, x, y):
        return self.ce(x, y)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, gt):
        ce_loss = F.cross_entropy(pred, gt, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1-pt)** self.gamma * ce_loss).mean()

        return focal_loss


class SoftTarget(nn.Module):
	def __init__(self, T):
		super(SoftTarget, self).__init__()
		self.T = T


	def forward(self, out_s, out_t):
		loss = F.kl_div(F.log_softmax(out_s/self.T, dim=1),
						F.softmax(out_t/self.T, dim=1),
						reduction='batchmean') * self.T * self.T

		return loss


@LOSS_REGISTRY.register()
class ExprKDLoss(nn.Module):
    def __init__(self, weight, T=4.0, alpha=0.5):
        super().__init__()
        self.soft_loss = SoftTarget(T)
        self.ce_loss = ExprLoss(weight)
        self.alpha = alpha

    def forward(self, student_out, teacher_out, label):
        softloss = self.soft_loss(student_out, teacher_out)
        hardloss = self.ce_loss(student_out, label)
        return self.alpha * hardloss + (1 - self.alpha) * softloss



class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
 
    def forward(self, inputs, target):
        target = F.softmax(target, dim=-1)
        logprobs = F.log_softmax(inputs.view(inputs.shape[0], -1), dim=1)
        batchloss = - torch.sum(target.view(target.shape[0], -1) * logprobs, dim=1)
        return torch.mean(batchloss)

class WeightedSoftCELoss(torch.nn.Module):
    def __init__(self, weight=None):
        super(WeightedSoftCELoss, self).__init__()
        self.weight = weight
    
    def forward(self, inputs, targets):
        loss = targets * F.log_softmax(inputs, dim=-1)
        # 加权
        if self.weight is not None:
            self.weight = self.weight.to(loss.device)
            loss = loss * self.weight
        # 计算交叉熵损失
        loss = -torch.sum(loss, dim=-1)
        
        # 平均损失
        loss = torch.mean(loss)
        
        return loss


@LOSS_REGISTRY.register()
class RDropLoss(nn.Module):
    def __init__(self, weight, alpha=5):
        super().__init__()
        w = torch.FloatTensor(weight)
        # self.ce = torch.nn.CrossEntropyLoss(weight=w, reduction='mean')
        self.ce = WeightedSoftCELoss(weight=w)
        self.kld = nn.KLDivLoss(reduction='none')
        self.alpha = alpha

    def forward(self, logits1, logits2, gt):
        # gt = torch.argmax(gt, dim=-1)
        ce_loss = (self.ce(logits1, gt) + self.ce(logits2, gt)) / 2
        kl_loss1 = self.kld(F.log_softmax(logits1, dim=-1), F.softmax(logits2, dim=-1)).sum(-1)
        kl_loss2 = self.kld(F.log_softmax(logits2, dim=-1), F.softmax(logits1, dim=-1)).sum(-1)
        kl_loss = (kl_loss1 + kl_loss2) / 2
        loss = ce_loss + self.alpha * kl_loss

        loss = loss.mean(-1)

        return loss

