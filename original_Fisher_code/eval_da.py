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
import pre_process as prep
import torch.utils.data as util_data
import data_list
from data_list import ImageList
from torch.autograd import Variable
import random
import json
from sklearn.metrics import confusion_matrix
import pickle as pkl

import utils

# visualization packages
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx


optim_dict = {"SGD": optim.SGD}


def image_classification_test(loader, model, test_10crop=True, gpu=True, iter_num=-1):
    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(loader['test'+str(i)]) for i in range(10)]
            for i in range(len(loader['test0'])):
                data = [iter_test[j].next() for j in range(10)]
                inputs = [data[j][0] for j in range(10)]
                labels = data[0][1]
                if gpu:
                    for j in range(10):
                        inputs[j] = Variable(inputs[j].cuda(), requires_grad=False)
                    labels = Variable(labels.cuda(), requires_grad=False)
                outputs = []
                for j in range(10):
                    _, predict_out = model(inputs[j])
                    outputs.append(nn.Softmax(dim=1)(predict_out))
                outputs = sum(outputs)
                if start_test:
                    all_output = outputs.data.float()
                    all_label = labels.data.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.data.float()), 0)
                    all_label = torch.cat((all_label, labels.data.float()), 0)
        else:
            iter_test = iter(loader["test"])
            for i in range(len(loader['test'])):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                if gpu:
                    inputs = Variable(inputs.cuda(), requires_grad=False)
                    labels = Variable(labels.cuda(), requires_grad=False)
                _, outputs = model(inputs)
                if start_test:
                    all_output = outputs.data.float()
                    all_label = labels.data.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.data.float()), 0)
                    all_label = torch.cat((all_label, labels.data.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).float() / float(all_label.size()[0])
    conf_matrix = confusion_matrix(all_label.cpu().numpy(), predict.cpu().numpy())
    return accuracy.item(), conf_matrix


def test(config):
    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train( \
                            resize_size=prep_config["resize_size"], \
                            crop_size=prep_config["crop_size"])
    prep_dict["target"] = prep.image_train( \
                            resize_size=prep_config["resize_size"], \
                            crop_size=prep_config["crop_size"])
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop( \
                            resize_size=prep_config["resize_size"], \
                            crop_size=prep_config["crop_size"])
    else:
        prep_dict["test"] = prep.image_test( \
                            resize_size=prep_config["resize_size"], \
                            crop_size=prep_config["crop_size"])

    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]

    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), \
                                transform=prep_dict["source"])
    dset_loaders["source"] = util_data.DataLoader(dsets["source"], \
            batch_size=data_config["source"]["batch_size"], \
            shuffle=True, num_workers=2)
    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["target"])
    dset_loaders["target"] = util_data.DataLoader(dsets["target"], \
            batch_size=data_config["target"]["batch_size"], \
            shuffle=True, num_workers=2)

    if prep_config["test_10crop"]:
        for i in range(10):
            dsets["test"+str(i)] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                transform=prep_dict["test"]["val"+str(i)])
            dset_loaders["test"+str(i)] = util_data.DataLoader(dsets["test"+str(i)], \
                                batch_size=data_config["test"]["batch_size"], \
                                shuffle=False, num_workers=2)

            dsets["target"+str(i)] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["test"]["val"+str(i)])
            dset_loaders["target"+str(i)] = util_data.DataLoader(dsets["target"+str(i)], \
                                batch_size=data_config["test"]["batch_size"], \
                                shuffle=False, num_workers=2)
    else:
        dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                transform=prep_dict["test"])
        dset_loaders["test"] = util_data.DataLoader(dsets["test"], \
                                batch_size=data_config["test"]["batch_size"], \
                                shuffle=False, num_workers=2)

        dsets["target_test"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["test"])
        dset_loaders["target_test"] = util_data.DataLoader(dsets["target_test"], \
                                batch_size=data_config["test"]["batch_size"], \
                                shuffle=False, num_workers=2)

    class_num = config["network"]["params"]["class_num"]

    # load checkpoint
    print('load model from {}'.format(config['ckpt_path']))
    # load in an old way
    # base_network = torch.load(config["ckpt_path"])[0]
    # recommended practice
    ckpt = torch.load(config['ckpt_path'])
    print('recorded best precision: {:0.4f} at step {}'.format(ckpt["precision"], ckpt["step"]))
    train_accuracy = ckpt["precision"]
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
        test_acc, test_confusion_matrix = image_classification_test(dset_loaders, \
            base_network, test_10crop=prep_config["test_10crop"], \
            gpu=use_gpu)
    elif config["ly_type"] == "euclidean":
        eval_centroids = None
        if centroids is not None:
            eval_centroids = centroids
        if target_centroids is not None:
            eval_centroids = target_centroids
        test_acc, test_confusion_matrix = utils.distance_classification_test(dset_loaders, \
            base_network, eval_centroids, test_10crop=prep_config["test_10crop"], \
            gpu=use_gpu)

    # save train/test accuracy as pkl file
    with open(os.path.join(config["output_path"], 'accuracy.pkl'), 'wb') as pkl_file:
        pkl.dump({'train': train_accuracy, 'test': test_acc}, pkl_file)
    
    np.set_printoptions(precision=2)
    log_str = "train precision: {:.5f}\ttest precision: {:.5f}\nconfusion matrix:\n{}\n".format(
        train_accuracy, test_acc, test_confusion_matrix)
    config["out_file"].write(log_str)
    config["out_file"].flush()
    print(log_str)

    return test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate DA models.')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--ly_type', type=str, default="cosine", choices=["cosine", "euclidean"], help="type of classification loss.")
    parser.add_argument('--net', type=str, default='ResNet50', help="Options: ResNet18,34,50,101,152; AlexNet")
    parser.add_argument('--dset', type=str, default='office', help="The dataset or source dataset used")
    parser.add_argument('--s_dset_path', type=str, default='../data/office/amazon_31_list.txt', help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='../data/office/webcam_10_list.txt', help="The target dataset path list")
    parser.add_argument('--ckpt_path', type=str, required=True, help="path to load ckpt")
    args = parser.parse_args()
    # print('args: {}'.format(args))

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

    config["prep"] = {"test_10crop":False, "resize_size":256, "crop_size":224}
    config["loss"] = {"trade_off":1.0, "update_iter":500}
    if "AlexNet" in args.net:
        config["network"] = {"name":network.AlexNetFc, \
            "params":{"use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    elif "ResNet" in args.net:
        network_class = network.ResNetFc
        config["network"] = {"name":network_class, \
            "params":{"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    elif "VGG" in args.net:
        config["network"] = {"name":network.VGGFc, \
            "params":{"vgg_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    config["optimizer"] = {"type":"SGD", "optim_params":{"lr":1.0, "momentum":0.9, \
                           "weight_decay":0.0005, "nesterov":True}, "lr_type":"inv", \
                           "lr_param":{"init_lr":0.001, "gamma":0.001, "power":0.75} }

    config["dataset"] = args.dset
    if config["dataset"] == "office":
        config["data"] = {"source":{"list_path":args.s_dset_path, "batch_size":36}, \
                          "target":{"list_path":args.t_dset_path, "batch_size":36}, \
                          "test":{"list_path":args.t_dset_path, "batch_size":4}}
        if "amazon" in config["data"]["test"]["list_path"]:
            config["optimizer"]["lr_param"]["init_lr"] = 0.0003
        else:
            config["optimizer"]["lr_param"]["init_lr"] = 0.001
        config["network"]["params"]["class_num"] = 31
    elif config["dataset"] == "office-home":
        config["num_iterations"] = 20000
        config["data"] = {"source":{"list_path":args.s_dset_path, "batch_size":36}, \
                          "target":{"list_path":args.t_dset_path, "batch_size":36}, \
                          "test":{"list_path":args.t_dset_path, "batch_size":4}}
        config["network"]["params"]["class_num"] = 65
    else:
        raise ValueError('{} not supported. '.format(config["dataset"]))
    
    print('config: {}\n'.format(config))
    test(config)
    config['out_file'].close()
