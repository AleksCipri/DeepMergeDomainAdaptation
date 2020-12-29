# Importing needed packages
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
from torch.autograd import Variable

class AdversarialLayer(torch.autograd.Function):
    '''
    Gradient reversal layer that we need to put befor domain classifier
    '''
  @staticmethod  
  #def forward(self, input):
  def forward(ctx, input, iter_num=0, alpha=10, low=0.0, high=1.0, max_iter=10000.0):
    iter_num += 1
    ctx.save_for_backward(input) 
    ctx.intermediate_results = (iter_num, alpha, low, high, max_iter)
    output = input * 1.0
    return output

  @staticmethod 
  def backward(ctx, gradOutput):
    input = ctx.saved_tensors
    iter_num, alpha, low, high, max_iter = ctx.intermediate_results
    return -1.0 * gradOutput

class SilenceLayer(torch.autograd.Function):
  def __init__(self):
    pass
  def forward(self, input):
    return input * 1.0

  def backward(self, gradOutput):
    return 0 * gradOutput


#We can load any of these ResNets. We decide to try only ResNet18.
resnet_dict = {"ResNet18":models.resnet18, "ResNet34":models.resnet34, "ResNet50":models.resnet50, "ResNet101":models.resnet101, "ResNet152":models.resnet152}

class ResNetFc(nn.Module):
  def __init__(self, resnet_name, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=2):
    super(ResNetFc, self).__init__()
    model_resnet = resnet_dict[resnet_name](pretrained=False)
    self.conv1 = model_resnet.conv1
    self.bn1 = model_resnet.bn1
    self.relu = model_resnet.relu
    self.maxpool = model_resnet.maxpool
    self.layer1 = model_resnet.layer1
    self.layer2 = model_resnet.layer2
    self.layer3 = model_resnet.layer3
    self.layer4 = model_resnet.layer4
    self.avgpool = model_resnet.avgpool
    self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                         self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

    self.use_bottleneck = use_bottleneck
    self.new_cls = new_cls
    if new_cls:
        if self.use_bottleneck:
            self.bottleneck = nn.Linear(model_resnet.fc.in_features, bottleneck_dim)
            self.bottleneck.weight.data.normal_(0, 0.005)
            self.bottleneck.bias.data.fill_(0.0)
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.weight.data.normal_(0, 0.01)
            self.fc.bias.data.fill_(0.0)
            self.__in_features = bottleneck_dim
        else:
            self.fc = nn.Linear(model_resnet.fc.in_features, class_num)
            self.fc.weight.data.normal_(0, 0.01)
            self.fc.bias.data.fill_(0.0)
            self.__in_features = model_resnet.fc.in_features
    else:
        self.fc = model_resnet.fc
        self.__in_features = model_resnet.fc.in_features
      
  def forward(self, x):
    x = self.feature_layers(x)
    x = x.view(x.size(0), -1)
    if self.use_bottleneck and self.new_cls:
        x = self.bottleneck(x)
    y = self.fc(x)
    return x, y

  def output_num(self):
    return self.__in_features


class DeepMerge(nn.Module):
    '''
    CNN from Ciprijanovic et al. (2020) Astronomy and Comupting, 32, 100390
    '''
    def __init__(self, use_bottleneck=False, bottleneck_dim=32 * 9 * 9, new_cls=False, class_num=2):
        super(DeepMerge, self).__init__()

        self.class_num = class_num
        self.use_bottleneck = use_bottleneck
        self.new_cls = new_cls
        self.in_features = 32 * 9 * 9

        self.conv1 = nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.batchn1 = nn.BatchNorm2d(8)
        self.batchn2 = nn.BatchNorm2d(16)
        self.batchn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * 9 * 9, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, class_num)
        self.relu =  nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.maxpool(self.relu(self.batchn1(self.conv1(x))))
        x = self.maxpool(self.relu(self.batchn2(self.conv2(x))))
        x = self.maxpool(self.relu(self.batchn3(self.conv3(x))))
        x = x.view(-1, 32 * 9 * 9)
        y = F.relu(self.fc1(x))
        y = F.relu(self.fc2(y))
        y = self.fc3(y)
        return x, y

    def output_num(self):
    	return self.in_features



class AdversarialNetwork(nn.Module):
    '''
    Neural network for domain classification
    '''
  def __init__(self, in_feature):
    super(AdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, 1024)
    self.ad_layer2 = nn.Linear(1024,1024)
    self.ad_layer3 = nn.Linear(1024, 1)
    self.ad_layer1.weight.data.normal_(0, 0.01)
    self.ad_layer2.weight.data.normal_(0, 0.01)
    self.ad_layer3.weight.data.normal_(0, 0.3)
    self.ad_layer1.bias.data.fill_(0.0)
    self.ad_layer2.bias.data.fill_(0.0)
    self.ad_layer3.bias.data.fill_(0.0)
    self.relu1 = nn.ReLU()
    self.relu2 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.dropout2 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.relu2(x)
    ad_features = self.dropout2(x)
    x = self.ad_layer3(ad_features)
    y = self.sigmoid(x)
    return y, ad_features

  def ad_feature_dim(self):
      return 1024

  def output_num(self):
    return 1