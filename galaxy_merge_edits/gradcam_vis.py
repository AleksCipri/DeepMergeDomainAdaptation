import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import network
import loss
import random
import cv2
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LogNorm
import matplotlib.cm as cmx
import sys
import copy

from PIL import Image
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.autograd import Variable
from import_and_normalize import array_to_tensor, update
from grad_cam import grad_cam

def cam(config):
    ## prepare data

    classes = {}
    classes[0] = 'non-merger'
    classes[1] = 'merger'

    dsets = {}
    
    #Loading final 20% of the data (test set)
    pristine_x_test = pristine_x[int(np.floor(.98*len(pristine_x))):]
    pristine_y_test = pristine_y[int(np.floor(.98*len(pristine_x))):]


    #noisy_indices = torch.randperm(len(noisy_x))
    noisy_x_test = noisy_x[int(np.floor(.98*len(noisy_x))):]
    noisy_y_test = noisy_y[int(np.floor(.98*len(noisy_x))):]

    dsets["source"] = pristine_x_test
    dsets["target"] = noisy_x_test

    class_num = config["network"]["params"]["class_num"]

    # load checkpoint
    print('load model from {}'.format(config['ckpt_path']))
    ckpt = torch.load(config['ckpt_path']+'/best_model.pth.tar')

    ## set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network.load_state_dict(ckpt['base_network'])

    use_gpu = torch.cuda.is_available()

    if use_gpu:
        base_network = base_network.cuda()

        #might not get to tstack this?
        source_images = torch.stack(list(dsets["source"])).cuda()
        target_images = torch.stack(list(dsets["target"])).cuda()

    base_network.train(False)

    model_name = config["network"]
    
    if config["class"] == 'non-merger':
        target_class = 0 #non-merger
    elif config["class"] == 'merger':
        target_class = 1 #non-merger
    else:
        print("incorrect class choice")
        sys.exit()

    save_where = osp.join(config["ckpt_path"], str(config["which"])+'-'+str(config["class"]))
    output_dir = save_where

    if not osp.exists(save_where):
        os.makedirs(save_where)
    else:
        os.chdir(save_where)

    #if config["network"]["name"] == 'DeepMerge':
    heatmap_layer = base_network.conv3
    #else:
    #    heatmap_layer = base_network.layer4[1].conv2

    if config["which"] == 'source':
        print("start source test: ")
        
        for j in range(0, len(source_images)-1):
            input_tensor = source_images[j]
            label = target_class

            ## Saving LogNorm galaxy images
            # my_cmap = copy.copy(plt.cm.get_cmap('inferno'))
            # my_cmap.set_bad(my_cmap.colors[0])
            # fig1=plt.figure(figsize=(8,8))
            # plt.imshow(input_tensor[0].cpu(), aspect='auto', cmap=my_cmap, norm=LogNorm())
            # plt.imshow(input_tensor[1].cpu(), aspect='auto', cmap=my_cmap, norm=LogNorm())
            # plt.imshow(input_tensor[2].cpu(), aspect='auto', cmap=my_cmap, norm=LogNorm())
            # plt.savefig(osp.join(
            #     output_dir,
            #     "image{}-{}.png".format(j, classes[target_class])))
            # with open(osp.join(output_dir, "image{}-{}.npy".format(j, classes[target_class])), 'wb') as f:
            #     np.save(f, np.asarray(image))

            image = grad_cam(base_network, input_tensor, heatmap_layer, label)
            #Saving Grad-CAMs without overplotted galaxy image
            ## in case we want to save it as image
            # cv2.imwrite(osp.join(
            #     output_dir,
            #     "{}-{}.png".format(j, classes[target_class])), image)

            #saving Grad-CAMs as numpy arrays
            with open(osp.join(output_dir, "{}-{}.npy".format(j, classes[target_class])), 'wb') as f:
                np.save(f, np.asarray(image))

    elif config["which"] == 'target':
        print("start target test: ")
        
        for j in range(0, len(target_images)-1):

            input_tensor = target_images[j]
            label = target_class

            #my_cmap = copy.copy(plt.cm.get_cmap('inferno'))
            #my_cmap.set_bad(my_cmap.colors[0])
            #fig1=plt.figure(figsize=(8,8))
            #plt.imshow(input_tensor[0].cpu(), aspect='auto', cmap=my_cmap, norm=LogNorm())
            #plt.imshow(input_tensor[1].cpu(), aspect='auto', cmap=my_cmap, norm=LogNorm())
            #plt.imshow(input_tensor[2].cpu(), aspect='auto', cmap=my_cmap, norm=LogNorm())
            # plt.savefig(osp.join(
            #     output_dir,
            #     "image{}-{}.png".format(j, classes[target_class])))

            with open(osp.join(output_dir, "{}-{}-IMAGE.npy".format(j, classes[target_class])), 'wb') as g:
                np.save(g, np.asarray(input_tensor[0].cpu()))

            image = grad_cam(base_network, input_tensor, heatmap_layer, label)
            # cv2.imwrite(osp.join(
            #     output_dir,
            #     "{}-{}.png".format(j, classes[target_class])), image)

            with open(osp.join(output_dir, "{}-{}.npy".format(j, classes[target_class])), 'wb') as f:
                np.save(f, np.asarray(image))
            # plt.close('all')

    else:
        print("incorrect domain choice")
        sys.exit()

    return ()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate DA models.')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet50', help="Options: ResNet18,34,50,101,152; AlexNet")
    parser.add_argument('--dset', type=str, default='galaxy', help="The dataset or source dataset used")
    parser.add_argument('--ckpt_path', type=str, required=True, help="path to load ckpt")
    parser.add_argument('--dset_path', type=str, default=None, help="The dataset directory path")
    parser.add_argument('--source_x_file', type=str, default='SB_version_00_numpy_3_filters_pristine_SB00_augmented_3FILT.npy',
                         help="Source domain x-values filename")
    parser.add_argument('--source_y_file', type=str, default='SB_version_00_numpy_3_filters_pristine_SB00_augmented_y_3FILT.npy',
                         help="Source domain y-values filename")
    parser.add_argument('--target_x_file', type=str, default='SB_version_00_numpy_3_filters_noisy_SB25_augmented_3FILT.npy',
                         help="Target domain x-values filename")
    parser.add_argument('--target_y_file', type=str, default='SB_version_00_numpy_3_filters_noisy_SB25_augmented_y_3FILT.npy',
                         help="Target domain y-values filename")
    parser.add_argument('--which', type=str, default='source', help= 'Source or target?')
    parser.add_argument('--classy', type=str, default='non-merger', help='Merger or nonmerger?')

    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # train config
    config = {}
    config["high"] = 1.0
    config["ckpt_path"] = args.ckpt_path
    config["which"] = args.which
    config["class"] = args.classy

    if "DeepMerge" in args.net:
        config["network"] = {"name":network.DeepMerge, \
            "params":{"class_num":2, "new_cls":True, "use_bottleneck":False, "bottleneck_dim":32*9*9} }
    elif "ResNet" in args.net:
        network_class = network.ResNetFc
        config["network"] = {"name":network_class, \
            "params":{"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }

    #how do we handle adnet?

    config["dataset"] = args.dset
    config["path"] = args.dset_path

    if config["dataset"] == 'galaxy':
        pristine_x = array_to_tensor(osp.join(config['path'], args.source_x_file))
        pristine_y = array_to_tensor(osp.join(config['path'], args.source_y_file))

        noisy_x = array_to_tensor(osp.join(config['path'], args.target_x_file))
        noisy_y = array_to_tensor(osp.join(config['path'], args.target_y_file))

        update(pristine_x, noisy_x)

        config["network"]["params"]["class_num"] = 2
    else:
        raise ValueError('{} not supported. '.format(config["dataset"]))
    
    cam(config)
