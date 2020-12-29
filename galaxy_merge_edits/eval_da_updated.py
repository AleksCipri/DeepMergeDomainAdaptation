'''
Example script to launch evaluation of the trained model:
!python eval_da_updated.py --gpu_id 0 \
                --net DeepMerge \
                --dset 'galaxy' \
                --dset_path 'arrays/' \
                --ly_type cosine \
                --ckpt_path 'output_DeepMerge_SDSS/MMD+F' \
                --source_x_file Illustris_Xdata_05_augmented_combined_rotzoom_SMALL_3000_3000.npy \
                --source_y_file Illustris_ydata_05_augmented_combined_rotzoom_SMALL_3000_3000.npy \
                --target_x_file SDSS_x_data_mergers_and_nonmergers.npy \
                --target_y_file SDSS_y_data_mergers_and_nonmergers.npy \
                --seed 1 
'''

# Importing needed packages
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
    
    # Fix random seed and unable deterministic calcualtions
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True


    # Prepare image data. Image shuffling is fixed with the random seed choice.
    # Train:validation:test = 70:10:20
    dsets = {}
    dset_loaders = {}


    pristine_indices = torch.randperm(len(pristine_x))
    # Test sample for evaluation file
    pristine_x_test = pristine_x[pristine_indices[int(np.floor(.8*len(pristine_x))):]]
    pristine_y_test = pristine_y[pristine_indices[int(np.floor(.8*len(pristine_x))):]]

    noisy_indices = torch.randperm(len(noisy_x))
    # Test sample for evaluation file
    noisy_x_test = noisy_x[noisy_indices[int(np.floor(.8*len(noisy_x))):]]
    noisy_y_test = noisy_y[noisy_indices[int(np.floor(.8*len(noisy_x))):]]

    dsets["source_test"] = TensorDataset(pristine_x_test, pristine_y_test)
    dsets["target_test"] = TensorDataset(noisy_x_test, noisy_y_test)

    dset_loaders["source_test"] = DataLoader(dsets["source_test"], batch_size = 64, shuffle = True, num_workers = 1)
    dset_loaders["target_test"] = DataLoader(dsets["target_test"], batch_size = 64, shuffle = True, num_workers = 1)

    class_num = config["network"]["params"]["class_num"]

    # Load checkpoint
    print('load model from {}'.format(config['ckpt_path']))
    ckpt = torch.load(config['ckpt_path']+'/best_model.pth.tar')
    print('recorded best training accuracy: {:0.4f} at epoch {}'.format(ckpt["train accuracy"], ckpt["epoch"]))
    print('recorded best validation accuracy: {:04f} at epoch {}'.format(ckpt["valid accuracy"], ckpt["epoch"]))

    train_accuracy = ckpt["train accuracy"]
    valid_accuracy = ckpt["valid accuracy"]
    
    # Set base network
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

    ###################
    ###### TEST ######
    ###################
    print("start test: ")
    base_network.train(False)
    if config["ly_type"] == 'cosine':
        source_test_acc, source_test_confusion_matrix = image_classification_test(dset_loaders, "source_test", \
            base_network, gpu=use_gpu, verbose = True, save_where = config['ckpt_path'])
        target_test_acc, target_test_confusion_matrix = image_classification_test(dset_loaders, "target_test", \
            base_network, gpu=use_gpu, verbose = True, save_where = config['ckpt_path'])

    elif config["ly_type"] == "euclidean":
        eval_centroids = None
        if centroids is not None:
            eval_centroids = centroids
        if target_centroids is not None:
            eval_centroids = target_centroids
        source_test_acc, source_test_confusion_matrix = distance_classification_test(dset_loaders, "source_test", \
            base_network, eval_centroids, gpu=use_gpu, verbose = True, save_where = config['ckpt_path'])
        target_test_acc, target_test_confusion_matrix = distance_classification_test(dset_loaders, "target_test", \
            base_network, eval_centroids, gpu=use_gpu, verbose = True, save_where = config['ckpt_path'])

    # Save train/test accuracy as pkl file
    with open(os.path.join(config["output_path"], 'accuracy.pkl'), 'wb') as pkl_file:
        pkl.dump({'train': train_accuracy, 'valid': valid_accuracy, 'source test': source_test_acc, 'target test': target_test_acc}, pkl_file)
    
    # Logging
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
    parser.add_argument('--net', type=str, default='ResNet50', help="Options: ResNet18, DeepMerge")
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
    parser.add_argument('--seed', type=int, default=3, help='Set random seed.')


    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # Train config
    config = {}
    config["high"] = 1.0
    config["output_for_test"] = True
    config["ckpt_path"] = args.ckpt_path
    config["output_path"] = args.ckpt_path
    config['ly_type'] = args.ly_type
    config["seed"] = args.seed

    # Set log file
    if not osp.exists(config["output_path"]):
        os.makedirs(config["output_path"])
        config["out_file"] = open(osp.join(config["output_path"], "test_log.txt"), "w")
    if osp.exists(config["output_path"]):
        config["out_file"] = open(osp.join(config["output_path"], "test_log.txt"), "w") 

    config["loss"] = {"trade_off":1.0, "update_iter":200}

    # Set parameters that depend on the choice of the network
    if "DeepMerge" in args.net:
        config["network"] = {"name":network.DeepMerge, \
            "params":{"class_num":2, "new_cls":True, "use_bottleneck":False, "bottleneck_dim":32*9*9} }
    elif "ResNet" in args.net:
        network_class = network.ResNetFc
        config["network"] = {"name":network_class, \
            "params":{"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }

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
    
    print('config: {}\n'.format(config))
    test(config)
    config['out_file'].close()
