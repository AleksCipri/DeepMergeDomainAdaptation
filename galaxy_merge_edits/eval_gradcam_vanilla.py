import argparse
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss
import random
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import sys

from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.autograd import Variable
from import_and_normalize import array_to_tensor, update
from grad_cam import (GradCAM) #maybe you need parenthesis for objects?
from main import save_gradcam

def cam(config):
    ## prepare data

    classes = {}
    classes[0] = 'non-merger'
    classes[1] = 'merger'

    dsets = {}

    #sampling WOR, i guess we leave the 10 in the middle to validate?
    pristine_indices = torch.randperm(len(pristine_x))
    pristine_x_test = pristine_x[pristine_indices[int(np.floor(.8*len(pristine_x))):]]
    pristine_y_test = pristine_y[pristine_indices[int(np.floor(.8*len(pristine_x))):]]

    noisy_indices = torch.randperm(len(noisy_x))
    noisy_x_test = noisy_x[noisy_indices[int(np.floor(.8*len(noisy_x))):]]
    noisy_y_test = noisy_y[noisy_indices[int(np.floor(.8*len(noisy_x))):]]

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
    
    if net_config == "Resnet18":
        target_layer = base_network.layer4[1].conv2
    else:
        target_layer = config["target_layer"] #no idea... maybe i can print this out? layer4 for resnet, relu for deepemerge

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
   
    gcam = GradCAM(model=base_network)

    if config["which"] == 'source':
        print("start source test: ")
        
        probs, ids = gcam.forward(source_images) #see how they load in images
        #not sure what to do about this
        if use_gpu:
            ids_ = torch.LongTensor([[target_class]] * len(source_images)).cuda()
        else:
            ids_ = torch.LongTensor([[target_class]] * len(source_images))

        gcam.backward(ids=ids_)

        regions = gcam.generate(target_layer=target_layer)

        for j in range(0, len(source_images)-1):

            print(
                "\t#{}: {} ({:.5f})".format(
                    j, classes[target_class], float(probs[:, target_class][j])
                )
            )

            save_gradcam(
                filename=osp.join(
                    output_dir,
                    "{}-{}-gradcam-{}-{}.png".format(
                        j, model_name, target_layer, classes[target_class]
                    ),
                ),
                gcam=regions[j, 0],
                #raw_image=raw_images[j],
                raw_image=source_images[j],
            )

    elif config["which"] == 'target':
        print("start target test: ")
        
        probs, ids = gcam.forward(target_images) #see how they load in images
        #not sure what to do about this
        if use_gpu:
            ids_ = torch.LongTensor([[target_class]] * len(target_images)).cuda()
        else:
            ids_ = torch.LongTensor([[target_class]] * len(target_images))

        gcam.backward(ids=ids_)

        regions = gcam.generate(target_layer=target_layer)

        for j in range(0, len(target_images)-1):

            print(
                "\t#{}: {} ({:.5f})".format(
                    j, classes[target_class], float(probs[:, target_class][j])
                )
            )

            save_gradcam(
                filename=osp.join(
                    output_dir,
                    "{}-{}-gradcam-{}-{}.png".format(
                        j, model_name, target_layer, classes[target_class]
                    ),
                ),
                gcam=regions[j, 0],
                #raw_image=raw_images[j],
                raw_image= target_images[j],
            )

    else:
        print("incorrect domain choice")
        sys.exit()

    return ()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate DA models.')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--net', type=str, default='ResNet18', help="Options: ResNet18,34,50,101,152; AlexNet")
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
        config["target_layer"] = "conv3"
    elif "ResNet" in args.net:
        network_class = network.ResNetFc
        config["network"] = {"name":network_class, \
            "params":{"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
        config["target_layer"] = "layer4"

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