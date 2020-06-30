import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
from torch.autograd import Variable

class AdversarialLayer(torch.autograd.Function):
  # def __init__(self, high_value=1.0, max_iter_value=10000.0):
  #   self.iter_num = 0
  #   self.alpha = 10
  #   self.low = 0.0
  #   self.high = high_value
  #   self.max_iter = max_iter_value
  
  @staticmethod  
  #def forward(self, input):
  def forward(ctx, input, iter_num=0, alpha=10, low=0.0, high=1.0, max_iter=10000.0):
    iter_num += 1
    ctx.save_for_backward(input) 
    ctx.intermediate_results = (iter_num, alpha, low, high, max_iter)
    #self.save_for_backward(self.iter_num)
    output = input * 1.0
    return output

  @staticmethod 
  def backward(ctx, gradOutput):
    input = ctx.saved_tensors
    iter_num, alpha, low, high, max_iter = ctx.intermediate_results
    #self.iter_,num += 1
    # self.coeff = np.float(2.0 * (self.high - self.low) / (1.0 + np.exp(-self.alpha*self.iter_num / self.max_iter)) - (self.high - self.low) + self.low)
    #coeff = np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)
    #return -coeff * gradOutput
    return -1.0 * gradOutput

class SilenceLayer(torch.autograd.Function):
  def __init__(self):
    pass
  def forward(self, input):
    return input * 1.0

  def backward(self, gradOutput):
    return 0 * gradOutput



resnet_dict = {"ResNet18":models.resnet18, "ResNet34":models.resnet34, "ResNet50":models.resnet50, "ResNet101":models.resnet101, "ResNet152":models.resnet152}

#no more pretraining
#might want to compress to 0-1 before normalizing to mu and sigma
class ResNetFc(nn.Module):
  def __init__(self, resnet_name, use_bottleneck=True, bottleneck_dim=256, new_cls=False, class_num=1000):
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
    def __init__(self, use_bottleneck=True, bottleneck_dim=32 * 9 * 9, new_cls=True, class_num=2):
        super(DeepMerge, self).__init__()

        self.class_num = class_num
        self.conv1 = nn.Conv2d(3, 8, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.batchn1 = nn.BatchNorm2d(8)
        self.batchn2 = nn.BatchNorm2d(16)
        self.batchn3 = nn.BatchNorm2d(32)
        self.relu =  nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        #self.drop = nn.Dropout(0.2)

        self.feature_layers = nn.Sequential(self.conv1, self.batchn1, self.relu, self.maxpool, \
                         self.conv2, self.batchn2, self.relu, self.maxpool, \
                         self.conv3, self.batchn3, self.relu, self.maxpool)

        # self.feature_layers = nn.Sequential(self.conv1, self.batchn1, self.relu, self.maxpool, self.drop, \
        #                  self.conv2, self.batchn2, self.relu, self.maxpool, self.drop, \
        #                  self.conv3, self.batchn3, self.relu, self.maxpool, self.drop)

        self.use_bottleneck = use_bottleneck
        self.__in_features = bottleneck_dim
        self.new_cls = new_cls

        self.lin1 =  nn.Linear(32 * 9 * 9, 64)
        self.lin2 =  nn.Linear(64, 32)
        self.lin3 =  nn.Linear(32, class_num)

        self.lin_layers = nn.Sequential(self.lin1, self.relu, self.lin2, self.relu, self.lin3)


    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        if self.use_bottleneck and self.new_cls:
            x = self.lin_layers(x)
        y = self.lin_layers(x)
        return x, y

    def output_num(self):
        return self.__in_features




class AdversarialNetwork(nn.Module):
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


class SmallAdversarialNetwork(nn.Module):
  def __init__(self, in_feature):
    super(SmallAdversarialNetwork, self).__init__()
    self.ad_layer1 = nn.Linear(in_feature, 256)
    self.ad_layer2 = nn.Linear(256, 1)
    self.ad_layer1.weight.data.normal_(0, 0.01)
    self.ad_layer2.weight.data.normal_(0, 0.01)
    self.ad_layer1.bias.data.fill_(0.0)
    self.ad_layer2.bias.data.fill_(0.0)
    self.relu1 = nn.ReLU()
    self.dropout1 = nn.Dropout(0.5)
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.ad_layer1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.ad_layer2(x)
    x = self.sigmoid(x)
    return x

  def output_num(self):
    return 1

class LittleAdversarialNetwork(nn.Module):
  def __init__(self, in_feature):
    super(LittleAdversarialNetwork, self).__init__()
    self.in_feature = in_feature
    self.ad_layer1 = nn.Linear(in_feature, 2)
    self.ad_layer1.weight.data.normal_(0, 0.01)
    self.ad_layer1.bias.data.fill_(0.0)
    self.softmax = nn.Softmax(dim=1)

  def forward(self, x):
    ad_features = self.ad_layer1(x)
    y = self.softmax(ad_features)
    return y, ad_features

  def ad_feature_dim():
    return self.in_feature

  def output_num(self):
    return 2


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


################################
# Adding ResNet2 and any ResNet
################################

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Blocks_net(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(Blocks_net, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


#def ResNet18():
#    return ResNet_Any(BasicBlock, [2, 2, 2, 2])

# def ResNet34():
#     return ResNet_Any(BasicBlock, [3, 4, 6, 3])


# def ResNet50():
#     return ResNet_Any(Bottleneck, [3, 4, 6, 3])


# def ResNet101():
#     return ResNet_Any(Bottleneck, [3, 4, 23, 3])


# def ResNet152():
#     return ResNet_Any(Bottleneck, [3, 8, 36, 3])

 # def ResNet2():
 #     return ResNet_Any(Bottleneck, [3, 8, 36, 3], num_classes=2)

class Custom_net(nn.Module):   
    def __init__(self, use_bottleneck=False, bottleneck_dim=64, new_cls=True, class_num=2):
        super(Custom_net, self).__init__()

        self.use_bottleneck = use_bottleneck
        self.__in_features = bottleneck_dim
        self.new_cls = new_cls

        # if self.use_bottleneck and self.new_cls:
        #     x = Blocks_net(Bottleneck, [2, 2, 2, 2], num_classes=2)
        # y = Blocks_net(Bottleneck, [2, 2, 2, 2], num_classes=2)
        # return x, y
        return Blocks_net(Bottleneck, [2, 2, 2, 2], num_classes=2), Blocks_net(Bottleneck, [2, 2, 2, 2], num_classes=2)

    def output_num(self):
        return self.__in_features

