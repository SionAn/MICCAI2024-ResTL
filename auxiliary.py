import torch
import torch.nn as nn
import torch.functional as F
import torchvision
import pdb
import random


class Disentanglement(nn.Module):
    def __init__(self, out_ch):
        super(Disentanglement, self).__init__()
        self.conv1 = nn.Conv2d(out_ch, out_ch, [3, 1], [1, 1], [1, 0])
        torch.nn.init.xavier_normal_(self.conv1.weight)
        self.conv2 = nn.Conv2d(out_ch, out_ch * 2, [3, 1], [1, 1], [1, 0])
        torch.nn.init.xavier_normal_(self.conv2.weight)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class Self_Conv(nn.Module):
    def __init__(self, out_ch):
        super(Self_Conv, self).__init__()
        self.conv1 = nn.Conv2d(out_ch, 32, [3, 1], [1, 1], padding=[1, 0])
        self.conv2 = nn.Conv2d(32, 1, [3, 1], [1, 1], padding=[1, 0])
        self.cnn = nn.Sequential(
            self.conv1,
            nn.Sigmoid(),
            self.conv2,
            nn.Sigmoid()
        )
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)  # , mean=0.0, std=0.0001)

    def forward(self, x):
        layer = self.cnn(x)

        return x * layer


class Self_Attention(nn.Module):
    def __init__(self):
        super(Self_Attention, self).__init__()

    def forward(self, query, key, value):
        '''
        :param query: [N, C, H, W]
        :param key: [N, K, C, H, W]
        :param value: [N, K, C, H, W]
        :return:
        '''
        N, K, C, H, W = key.shape
        query = torch.permute(query, [0, 2, 3, 1]).reshape(N, H * W, C)
        query = query.repeat_interleave(repeats=K, dim=0)  # [NK, HW, C]
        key = torch.permute(key, [0, 1, 3, 4, 2]).reshape(N * K, H * W, C)
        value = torch.permute(value, [0, 1, 3, 4, 2]).reshape(N * K, H * W, C)

        corr = torch.bmm(query, key.transpose(1, 2))  # [NK, HW, HW]
        norm = torch.nn.functional.softmax(corr, dim=2)
        out = torch.bmm(norm, value)  # [NK, HW, C]
        out = torch.permute(out, [0, 2, 1])

        return out.reshape(N, K, C, H, W)


class Attention(nn.Module):
    def __init__(self, out_ch):
        super(Attention, self).__init__()
        self.conv1 = nn.Conv2d(out_ch * 2, 128, [30, 1], [1, 1], padding=[14, 0])
        self.avgpool = nn.AvgPool2d([3, 1], [3, 1], [1, 0])
        self.conv2 = nn.Conv2d(128, 256, [15, 1], [1, 1], padding=[7, 0])
        self.cnn = nn.Sequential(
            self.conv1,
            nn.ELU(),
            self.avgpool,
            self.conv2,
            nn.ELU()
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 256, bias=True),
            nn.ELU(),
            nn.Linear(256, 100, bias=True),
            nn.ELU(),
            nn.Linear(100, 1, bias=True),
            nn.Sigmoid()
        )
        self._init_weight()

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)  # , mean=0.0, std=0.0001)

    def forward(self, sup, que):
        '''
        :param sup: [B, K, ch, H, W]
        :param que: [B, ch, H, W]
        :return: prototype
        '''
        que = torch.unsqueeze(que, dim=1).repeat(1, sup.shape[1], 1, 1, 1)
        x = torch.cat((sup, que), dim=2)
        x_size = x.shape
        x = torch.reshape(x, (x_size[0] * x_size[1], x_size[2], x_size[3], x_size[4]))
        layer = self.cnn(x)
        layer = torch.mean(layer, dim=(-2, -1))  # Global average pooling
        layer = self.fc(layer)
        attention_score = torch.reshape(layer, (x_size[0], x_size[1]))
        attention_sum = torch.sum(attention_score, dim=1)
        attention_norm = torch.div(attention_score, torch.unsqueeze(attention_sum, -1))
        attention_norm = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(attention_norm, -1), -1), -1)
        proto = sup * attention_norm
        proto = torch.sum(proto, dim=1)

        return proto


class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """

    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.reshape_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)


class MINE(nn.Module):
    def __init__(self, nx1, nx2):
        super(MINE, self).__init__()
        self.nf = [nx1, nx2]
        self.fc1 = nn.ModuleList([nn.Linear(self.nf[i], int(self.nf[i] / 16)) for i in range(2)])
        self.bn1 = nn.ModuleList([nn.BatchNorm1d(int(self.nf[i] / 16)) for i in range(2)])
        self.fc2 = nn.Linear(int(nx1 / 16) + int(nx2 / 16), int(nx1 / 16) + int(nx2 / 16))
        self.bn2 = nn.BatchNorm1d(int(nx1 / 16) + int(nx2 / 16))
        self.fc3 = nn.Linear(int(nx1 / 16) + int(nx2 / 16), 1)

    def forward(self, x, y, lambd=1):
        # x = GradReverse.grad_reverse(x, lambd)
        # y = GradReverse.grad_reverse(y, lambd)

        x = self.fc1[0](x)
        x = self.bn1[0](x)
        x = torch.nn.functional.dropout(x)

        y = self.fc1[1](y)
        y = self.bn1[1](y)
        y = torch.nn.functional.dropout(y)

        h = torch.nn.functional.elu(torch.cat((x, y), dim=-1))
        h = torch.nn.functional.elu(self.bn2(self.fc2(h)))
        h = self.fc3(h)

        return h

class MixStyle(nn.Module):
    """MixStyle.
    Reference:
      Zhou et al. Domain Generalization with MixStyle. ICLR 2021.
    """

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix='random'):
        """
        Args:
          p (float): probability of using MixStyle.
          alpha (float): parameter of the Beta distribution.
          eps (float): scaling parameter to avoid numerical issues.
          mix (str): how to mix.
        """
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        if self.mix == 'random':
            # random shuffle
            perm = torch.randperm(B)

        elif self.mix == 'crossdomain':
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1) # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(B // 2)]
            perm_a = perm_a[torch.randperm(B // 2)]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu*lmda + mu2 * (1-lmda)
        sig_mix = sig*lmda + sig2 * (1-lmda)

        return x_normed*sig_mix + mu_mix


def estimate_JSD_MI(joint, marginal, mean=False):
    joint = (torch.log(torch.tensor(2.0)) - torch.nn.functional.softplus(-joint))
    marginal = (torch.nn.functional.softplus(-marginal) + marginal - torch.log(torch.tensor(2.0)))

    out = joint - marginal
    if mean:
        out = out.mean()
    return out

def Euclidean_distance(x, y):
    '''
        Compute euclidean distance between two tensors
        '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def compute_irm_penalty(logits, y):
    scale = torch.tensor(1.).to(logits.device).requires_grad_()
    loss = torch.nn.functional.cross_entropy(torch.nn.functional.softmax(logits, 1) * scale, y)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad ** 2)