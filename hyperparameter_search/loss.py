import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial
from itertools import combinations
from sklearn.metrics import pairwise_distances, pairwise


def EntropyLoss(input_):
    mask = input_.ge(0.000001)
    mask_out = torch.masked_select(input_, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(input_.size(0))


def PADA(features, ad_net, grl_layer, weight_ad, use_gpu=True):
    '''domain adversarial loss
    Args: 
        features: torch.FloatTensor, concatenated source domain and target domain features
        ad_net: nn.Module, domain classification network
        grl_layer: gradient reversal layer
        weight_ad: torch.FloatTensor, weight of each sample, default all 1's
    '''
    ad_out, _ = ad_net(grl_layer.apply(features))
    batch_size = int(ad_out.size(0) / 2)
    dc_target = Variable(torch.from_numpy(
        np.array([[1]] * batch_size + [[0]] * batch_size)).float())
    if use_gpu:
        dc_target = dc_target.cuda()
        weight_ad = weight_ad.cuda()
    return nn.BCELoss(weight=weight_ad.view(-1))(ad_out.view(-1), dc_target.view(-1))


def CORAL(source, target):
    '''CORAL loss
       https://github.com/SSARCandy/DeepCORAL/blob/master/models.py
    '''
    batch_size, d = source.size()  # assume that source, target are 2d tensors

    # source covariance
    xm = torch.mean(source, 0, keepdim=True) - source
    xc = (1. / (batch_size - 1)) * torch.matmul(xm.t(), xm)

    # target covariance
    xmt = torch.mean(target, 0, keepdim=True) - target
    xct = (1. / (batch_size - 1)) * torch.matmul(xmt.t(), xmt)

    # frobenius norm between source and target
    loss = torch.mean(torch.mul((xc - xct), (xc - xct)))
    # loss = loss / (4*d*d)

    return loss


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(
        int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                  for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)  # /len(kernel_val)


def compute_pairwise_distances(x, y):
    if not x.dim() == y.dim() == 2:
        raise ValueError('Both inputs should be matrices.')
    if x.size(1) != y.size(1):
        raise ValueError('The number of features should be the same.')

    norm = lambda x: torch.sum(torch.pow(x, 2), 1)
    return torch.transpose(norm(torch.unsqueeze(x, 2) - torch.transpose(y, 0, 1)), 0, 1)


def gaussian_kernel_matrix(x, y, sigmas):
    beta = 1. / (2. * (torch.unsqueeze(sigmas, 1))) #what is beta
    dist = compute_pairwise_distances(x, y)
    # print('dist shape={}'.format(dist.size()))
    s = torch.matmul(beta, dist.contiguous().view(1, -1))
    return torch.sum(torch.exp(-s), 0).view(*dist.size())


#this is their equivalent to D(X,Y, Fancy F)
def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
    cost = torch.mean(kernel(x, x))
    cost += torch.mean(kernel(y, y))
    cost -= 2 * torch.mean(kernel(x, y))
    # We do not allow the loss to become negative. #should we absolute value instead?
    cost = torch.clamp(cost, min=0.0)
    return cost


def mmd_distance(hs, ht, use_gpu=False):
    '''maximum mean discrepancy, a combination of multiple kernels
    '''
    sigmas = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5,
              10, 15, 20, 25, 30, 35, 100, 1e3, 1e4, 1e5, 1e6]
    if use_gpu:
        gaussian_kernel = partial(gaussian_kernel_matrix,
                                sigmas=torch.Tensor(sigmas).float().cuda())
    else:
        gaussian_kernel = partial(gaussian_kernel_matrix,
                                sigmas=torch.Tensor(sigmas).float())

    loss_value = maximum_mean_discrepancy(hs, ht, kernel=gaussian_kernel)
    return torch.clamp(loss_value, min=1e-4)


class FisherTD(nn.Module):
    """MMC loss by auto-grad
    adapted from: https://github.com/KaiyangZhou/pytorch-center-loss/blob/master/center_loss.py

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(FisherTD, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(
                self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(
                torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels, inter_class="global", intra_loss_weight=1.0, inter_loss_weight=0.0):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
            inter_class: str, one of ["global", "sample"]. 
                         if inter_class=="global", calculate intra class distance by distances of centers and global center. 
                         if inter_class=="sample", calculate intra class distance by distances of samples and centers of different classes. 
            intra_loss_weight: float, default=1.0
        """
        batch_size = x.size(0)

        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
            torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(
                self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        mask = labels.unsqueeze(1).expand(batch_size, self.num_classes).eq(
            classes.expand(batch_size, self.num_classes))  # mask is ohe of labels

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            # for numerical stability
            value = value.clamp(min=1e-12, max=1e+12)
            dist.append(value)
        dist = torch.cat(dist)
        intra_loss = dist.sum()

        # between class distance
        if inter_class == "global":
            global_center = torch.mean(self.centers, 0)
            inter_loss = torch.pow(torch.norm(
                self.centers - global_center, p=2, dim=1), 2).sum()
        else:
            raise ValueError(
                "invalid value for inter_class argument, must be one of [global, sample]. ")

        loss = intra_loss_weight * intra_loss - inter_loss_weight * inter_loss
        return loss, intra_loss, inter_loss, None


class FisherTR(nn.Module):
    """ Fisher loss in Trace Ratio form
    adapted from: https://github.com/KaiyangZhou/pytorch-center-loss/blob/master/center_loss.py

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=False):
        super(FisherTR, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(
                self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(
                torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels, inter_class="global", intra_loss_weight=1.0, inter_loss_weight=1.0):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
            inter_class: str, one of ["global", "sample"]. 
                         if inter_class=="global", calculate intra class distance by distances of centers and global center. 
                         if inter_class=="sample", calculate intra class distance by distances of samples and centers of different classes. 
            intra_loss_weight: float, default=1.0
        """
        batch_size = x.size(0)

        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
            torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(
                self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        mask = labels.unsqueeze(1).expand(batch_size, self.num_classes).eq(
            classes.expand(batch_size, self.num_classes))  # mask is ohe of labels

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            # for numerical stability
            value = value.clamp(min=1e-12, max=1e+12)
            dist.append(value)
        dist = torch.cat(dist)
        intra_loss = dist.sum()

        # between class distance
        if inter_class == "global":
            global_center = torch.mean(self.centers, 0)
            inter_loss = torch.pow(torch.norm(
                self.centers - global_center, p=2, dim=1), 2).sum()
        else:
            raise ValueError(
                "invalid value for inter_class argument, must be one of [global, sample]. ")

        loss = intra_loss_weight * intra_loss / \
            (inter_loss_weight * inter_loss)
        return loss, intra_loss, inter_loss, None
