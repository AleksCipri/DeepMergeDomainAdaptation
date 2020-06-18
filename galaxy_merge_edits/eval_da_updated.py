'''
script to launch training: 
nohup python test_pada_py2.py --gpu_id 1 --net ResNet50 --dset office --s_dset_path ../data/office/webcam_31_list.txt --t_dset_path ../data/office/amazon_10_list.txt --test_interval 500 --snapshot_interval 10000 --output_dir san/w2a
'''

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
import pickle as pkl
import galaxy_utils
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

from galaxy_utils import image_classification_test, distance_classification_test 
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.autograd import Variable
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from import_and_normalize import array_to_tensor, update

def test(config):
    ## prepare data
    dsets = {}
    dset_loaders = {}

    #sampling WOR, i guess we leave the 10 in the middle to validate?
    pristine_indices = torch.randperm(len(pristine_x))
    #train
    pristine_x_train = pristine_x[pristine_indices[:int(np.floor(.7*len(pristine_x)))]]
    pristine_y_train = pristine_y[pristine_indices[:int(np.floor(.7*len(pristine_x)))]]
    #validate --- gets passed into test functions in train file
    pristine_x_valid = pristine_x[pristine_indices[int(np.floor(.7*len(pristine_x))) : int(np.floor(.8*len(pristine_x)))]]
    pristine_y_valid = pristine_y[pristine_indices[int(np.floor(.7*len(pristine_x))) : int(np.floor(.8*len(pristine_x)))]]
    #test for evaluation file
    pristine_x_test = pristine_x[pristine_indices[int(np.floor(.8*len(pristine_x))):]]
    pristine_y_test = pristine_y[pristine_indices[int(np.floor(.8*len(pristine_x))):]]

    noisy_indices = torch.randperm(len(noisy_x))
    #train
    noisy_x_train = noisy_x[noisy_indices[:int(np.floor(.7*len(noisy_x)))]]
    noisy_y_train = noisy_y[noisy_indices[:int(np.floor(.7*len(noisy_x)))]]
    #validate --- gets passed into test functions in train file
    noisy_x_valid = noisy_x[noisy_indices[int(np.floor(.7*len(noisy_x))) : int(np.floor(.8*len(noisy_x)))]]
    noisy_y_valid = noisy_y[noisy_indices[int(np.floor(.7*len(noisy_x))) : int(np.floor(.8*len(noisy_x)))]]
    #test for evaluation file
    noisy_x_test = noisy_x[noisy_indices[int(np.floor(.8*len(noisy_x))):]]
    noisy_y_test = noisy_y[noisy_indices[int(np.floor(.8*len(noisy_x))):]]


    dsets["source"] = TensorDataset(pristine_x_train, pristine_y_train)
    dsets["target"] = TensorDataset(noisy_x_train, noisy_y_train)

    dsets["source_valid"] = TensorDataset(pristine_x_valid, pristine_y_valid)
    dsets["target_valid"] = TensorDataset(noisy_x_valid, noisy_y_valid)

    dsets["source_test"] = TensorDataset(pristine_x_test, pristine_y_test)
    dsets["target_test"] = TensorDataset(noisy_x_test, noisy_y_test)

    #put your dataloaders here
    #i stole batch size numbers from below
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size = 36, shuffle = True, num_workers = 1)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size = 36, shuffle = True, num_workers = 1)

    #guessing batch size based on what was done for testing in the original file
    dset_loaders["source_valid"] = DataLoader(dsets["source_valid"], batch_size = 4, shuffle = True, num_workers = 1)
    dset_loaders["target_valid"] = DataLoader(dsets["target_valid"], batch_size = 4, shuffle = True, num_workers = 1)

    dset_loaders["source_test"] = DataLoader(dsets["source_test"], batch_size = 4, shuffle = True, num_workers = 1)
    dset_loaders["target_test"] = DataLoader(dsets["target_test"], batch_size = 4, shuffle = True, num_workers = 1)


    class_num = config["network"]["params"]["class_num"]

    # load checkpoint
    print('load model from {}'.format(config['ckpt_path']))

    ckpt = torch.load(config['ckpt_path'])
    print('recorded best training accuracy: {:0.4f} at step {}'.format(ckpt["train accuracy"], ckpt["step"]))
    print('recorded best validation accuracy: {:04f} at step {}'.format(ckpt["valid accuracy"], ckpt["step"]))

    train_accuracy = ckpt["train accuracy"]
    valid_accuracy = ckpt["valid accuracy"]
    ## set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network.load_state_dict(ckpt['base_network'])

    centroids = None
    if 'center_criterion' in ckpt.keys():
        centroids = ckpt['center_criterion']['centers'].cpu()
    target_centroids = None
    if 'target_center_criterion' in ckpt.keys():
        target_centroids = ckpt['target_center_criterion']['centers'].cpu()

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        base_network = base_network.cuda()

    ## test
    print("start test: ")
    base_network.train(False)
    if config["ly_type"] == 'cosine':
        source_test_acc, source_test_confusion_matrix = image_classification_test(dset_loaders, "source_test", \
            base_network, \
            gpu=use_gpu)
        target_test_acc, target_test_confusion_matrix = image_classification_test(dset_loaders, "target_test", \
            base_network, \
            gpu=use_gpu)

    elif config["ly_type"] == "euclidean":
        eval_centroids = None
        if centroids is not None:
            eval_centroids = centroids
        if target_centroids is not None:
            eval_centroids = target_centroids
        source_test_acc, source_test_confusion_matrix = distance_classification_test(dset_loaders, "source_test", \
            base_network, eval_centroids, \
            gpu=use_gpu)
        target_test_acc, target_test_confusion_matrix = distance_classification_test(dset_loaders, "target_test", \
            base_network, eval_centroids, \
            gpu=use_gpu)

    # save train/test accuracy as pkl file
    #why do we want this in a pkl file?
    with open(os.path.join(config["output_path"], 'accuracy.pkl'), 'wb') as pkl_file:
        pkl.dump({'train': train_accuracy, 'valid': valid_accuracy, 'source test': source_test_acc, 'target test': target_test_acc}, pkl_file)
    
    np.set_printoptions(precision=2)
    log_str = "train accuracy: {:.5f}\tvalid accuracy: {:5f}\nsource test accuracy: {:.5f}\nsource confusion matrix:\n{}\ntarget test accuracy: {:.5f}\ntarget confusion matrix:\n{}\n".format(
        train_accuracy, valid_accuracy, source_test_acc, source_test_confusion_matrix, target_test_acc, target_test_confusion_matrix)
    config["out_file"].write(log_str)
    config["out_file"].flush()
    print(log_str)

    return (source_test_acc, target_test_acc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate DA models.')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--ly_type', type=str, default="cosine", choices=["cosine", "euclidean"], help="type of classification loss.")
    parser.add_argument('--net', type=str, default='ResNet50', help="Options: ResNet18,34,50,101,152; AlexNet")
    parser.add_argument('--dset', type=str, default='galaxy', help="The dataset or source dataset used")
    parser.add_argument('--dset_path', type=str, default='/arrays', help="The source dataset path list")
    #parser.add_argument('--domain', type=str, default='source_test', help="Either source_test or target_test")
    parser.add_argument('--ckpt_path', type=str, required=True, help="path to load ckpt")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # train config
    config = {}
    config["high"] = 1.0
    config["output_for_test"] = True
    config["ckpt_path"] = args.ckpt_path
    config["output_path"] = os.path.split(args.ckpt_path)[0]
    config['ly_type'] = args.ly_type

    if not osp.exists(config["output_path"]):
        os.makedirs(config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "test_log.txt"), "w")
    if not osp.exists(config["output_path"]):
        os.makedirs(config["output_path"])

    config["loss"] = {"trade_off":1.0, "update_iter":500}

    if "DeepMerge" in args.net:
        config["network"] = {"name":network.DeepMerge, \
            "params":{"class_num":2, "new_cls":True, "use_bottleneck":False, "bottleneck_dim":32*9*9} }
    elif "ResNet" in args.net:
        network_class = network.ResNetFc
        config["network"] = {"name":network_class, \
            "params":{"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }

    config["dataset"] = args.dset
    config["path"] = args.dset_path
    #config["domain"] = args.domain

    if config["dataset"] == 'galaxy': 
        pristine_x = array_to_tensor(osp.join(os.getcwd(), config['path'], 'SB_version_00_numpy_3_filters_pristine_SB00_augmented_3FILT.npy'))
        pristine_y = array_to_tensor(osp.join(os.getcwd(), config['path'], 'SB_version_00_numpy_3_filters_pristine_SB00_augmented_y_3FILT.npy'))

        noisy_x = array_to_tensor(osp.join(os.getcwd(), config['path'], 'SB_version_00_numpy_3_filters_noisy_SB25_augmented_3FILT.npy'))
        noisy_y = array_to_tensor(osp.join(os.getcwd(), config['path'], 'SB_version_00_numpy_3_filters_noisy_SB25_augmented_y_3FILT.npy'))

        update(pristine_x, noisy_x)

        config["network"]["params"]["class_num"] = 2
    else:
        raise ValueError('{} not supported. '.format(config["dataset"]))
    
    print('config: {}\n'.format(config))
    test(config)
    config['out_file'].close()
