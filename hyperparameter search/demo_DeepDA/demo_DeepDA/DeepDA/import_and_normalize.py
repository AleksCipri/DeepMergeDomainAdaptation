# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 13:30:47 2020

@author: dk
"""

import torch
import numpy as np
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torchvision.transforms as transform
import matplotlib.pyplot as plt

def array_to_tensor(name):
  data_list = list(np.load(str(name)))
  return(torch.Tensor(data_list))

def update(t1, t2):

  #trying to compress the domain
  # def zero_one(t):
    
  #   max1 = t[:,0].max().item()
  #   max2 = t[:,1].max().item()
  #   max3 = t[:,2].max().item()

  #   min1 = t[:,0].min().item()
  #   min2 = t[:,1].min().item()
  #   min3 = t[:,2].min().item()

  #   t[:,0] = (t[:,0] - min1)/(max1 - min1)
  #   t[:,1] = (t[:,1] - min2)/(max2 - min2)
  #   t[:,2] = (t[:,2] - min3)/(max3 - min3)
    
  #   return t

  # t1 = zero_one(t1)
  # t2 = zero_one(t2)

  #method online: https://discuss.pytorch.org/t/normalize-a-vector-to-0-1/14594/5
  # t1[:,0] = t1[:,0] / t1[:,0].sum(0).expand_as(t1[:,0])
  # t1[:,1] = t1[:,1] / t1[:,1].sum(0).expand_as(t1[:,1])
  # t1[:,2] = t1[:,2] / t1[:,2].sum(0).expand_as(t1[:,2])

  # t2[:,0] = t2[:,0] / t2[:,0].sum(0).expand_as(t2[:,0])
  # t2[:,1] = t2[:,1] / t2[:,1].sum(0).expand_as(t2[:,1])
  # t2[:,2] = t2[:,2] / t2[:,2].sum(0).expand_as(t2[:,2])

  #using just the max value
  # t1[:,0] = t1[:,0]/t1[:,0].max().item()
  # t1[:,1] = t1[:,1]/t1[:,1].max().item()
  # t1[:,2] = t1[:,2]/t1[:,2].max().item()

  # t2[:,0] = t2[:,0]/t2[:,0].max().item()
  # t2[:,1] = t2[:,1]/t2[:,1].max().item()
  # t2[:,2] = t2[:,2]/t2[:,2].max().item()

  def normalization(t):

    mean1 = t[:,0].mean().item()
    mean2 = t[:,1].mean().item()
    mean3 = t[:,2].mean().item()

    std1 = t[:,0].std().item()
    std2 = t[:,1].std().item()
    std3 = t[:,2].std().item()

    return np.array([[mean1, mean2, mean3], [std1, std2, std3]])

  pristine = normalization(t1)
  noisy = normalization(t2)

  pr_trf = transform.Normalize(mean = pristine[0], std = pristine[1], inplace=True)
  no_trf = transform.Normalize(mean = noisy[0], std = noisy[1], inplace=True)

  for i in range(0, len(t1)-1):
      pr_trf(t1[i])
  
  for i in range(0, len(t2)-1):
      no_trf(t2[i])