'''
script to launch training: 
nohup python2 train_pada.py --gpu_id 1 --net ResNet50 --dset office --s_dset_path ../data/office/webcam_31_list.txt --t_dset_path ../data/office/amazon_10_list.txt --test_interval 500 --snapshot_interval 10000 --output_dir san/w2a
'''

import argparse
import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tensorboardX
from tensorboardX import SummaryWriter
import network
import loss
import pre_process as prep
import torch.utils.data as util_data
import lr_schedule
import data_list
from data_list import ImageList, stratify_sampling
from torch.autograd import Variable
import random

from utils import EarlyStopping, distance_classification_test, domain_cls_accuracy


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
                        inputs[j] = Variable(inputs[j].cuda())
                    labels = Variable(labels.cuda())
                else:
                    for j in range(10):
                        inputs[j] = Variable(inputs[j])
                    labels = Variable(labels)
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
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs = Variable(inputs)
                    labels = Variable(labels)
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
    return accuracy.item()


def train(config):
    ## set up summary writer
    writer = SummaryWriter(config['output_path'])

    # set up early stop
    early_stop_engine = EarlyStopping(config["early_stop_patience"])

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
               
    ## set loss
    class_num = config["network"]["params"]["class_num"]

    class_criterion = nn.CrossEntropyLoss()

    transfer_criterion = config["loss"]["name"]
    center_criterion = config["loss"]["discriminant_loss"](num_classes=class_num, 
                                       feat_dim=config["network"]["params"]["bottleneck_dim"])
    loss_params = config["loss"]

    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    dsets["source"] = ImageList(stratify_sampling(open(data_config["source"]["list_path"]).readlines(), ratio=prep_config["source_size"]), \
                                transform=prep_dict["source"])
    dset_loaders["source"] = util_data.DataLoader(dsets["source"], \
            batch_size=data_config["source"]["batch_size"], \
            shuffle=True, num_workers=1)
    dsets["target"] = ImageList(stratify_sampling(open(data_config["target"]["list_path"]).readlines(), ratio=prep_config['target_size']), \
                                transform=prep_dict["target"])
    dset_loaders["target"] = util_data.DataLoader(dsets["target"], \
            batch_size=data_config["target"]["batch_size"], \
            shuffle=True, num_workers=1)

    if prep_config["test_10crop"]:
        for i in range(10):
            dsets["test"+str(i)] = ImageList(stratify_sampling(open(data_config["test"]["list_path"]).readlines(), ratio=prep_config['target_size']), \
                                transform=prep_dict["test"]["val"+str(i)])
            dset_loaders["test"+str(i)] = util_data.DataLoader(dsets["test"+str(i)], \
                                batch_size=data_config["test"]["batch_size"], \
                                shuffle=False, num_workers=1)

            dsets["target"+str(i)] = ImageList(stratify_sampling(open(data_config["target"]["list_path"]).readlines(), ratio=prep_config['target_size']), \
                                transform=prep_dict["test"]["val"+str(i)])
            dset_loaders["target"+str(i)] = util_data.DataLoader(dsets["target"+str(i)], \
                                batch_size=data_config["test"]["batch_size"], \
                                shuffle=False, num_workers=1)
    else:
        dsets["test"] = ImageList(stratify_sampling(open(data_config["test"]["list_path"]).readlines(), ratio=prep_config['target_size']), \
                                transform=prep_dict["test"])
        dset_loaders["test"] = util_data.DataLoader(dsets["test"], \
                                batch_size=data_config["test"]["batch_size"], \
                                shuffle=False, num_workers=1)

        dsets["target_test"] = ImageList(stratify_sampling(open(data_config["target"]["list_path"]).readlines(), ratio=prep_config['target_size']), \
                                transform=prep_dict["test"])
        dset_loaders["target_test"] = MyDataLoader(dsets["target_test"], \
                                batch_size=data_config["test"]["batch_size"], \
                                shuffle=False, num_workers=1)
    config['out_file'].write("dataset sizes: source={}, target={}\n".format(
        len(dsets["source"]), len(dsets["target"])))

    ## set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])


    use_gpu = torch.cuda.is_available()
    if use_gpu:
        base_network = base_network.cuda()

    ## collect parameters
    if net_config["params"]["new_cls"]:
        if net_config["params"]["use_bottleneck"]:
            parameter_list = [{"params":base_network.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":base_network.bottleneck.parameters(), "lr_mult":10, 'decay_mult':2}, \
                            {"params":base_network.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
        else:
            parameter_list = [{"params":base_network.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
                            {"params":base_network.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
    else:
        parameter_list = [{"params":base_network.parameters(), "lr_mult":1, 'decay_mult':2}]

    ## add additional network for some methods
    class_weight = torch.from_numpy(np.array([1.0] * class_num))
    if use_gpu:
        class_weight = class_weight.cuda()
    parameter_list.append({"params":center_criterion.parameters(), "lr_mult": 10, 'decay_mult':1})
 
    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optim_dict[optimizer_config["type"]](parameter_list, \
                    **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]


    ## train   
    len_train_source = len(dset_loaders["source"]) - 1
    len_train_target = len(dset_loaders["target"]) - 1
    transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
    best_acc = 0.0
    for i in range(config["num_iterations"]):
        if i % config["test_interval"] == 0:
            base_network.train(False)
            if config['loss']['ly_type'] == "cosine":
                temp_acc = image_classification_test(dset_loaders, \
                    base_network, test_10crop=prep_config["test_10crop"], \
                    gpu=use_gpu)
            elif config['loss']['ly_type'] == "euclidean":
                temp_acc, _ = distance_classification_test(dset_loaders, \
                    base_network, center_criterion.centers.detach(), test_10crop=prep_config["test_10crop"], \
                    gpu=use_gpu)
            else:
                raise ValueError("no test method for cls loss: {}".format(config['loss']['ly_type']))
            
            snapshot_obj = {'step': i, 
                            "base_network": base_network.state_dict(), 
                            'precision': temp_acc, 
                            }
            snapshot_obj['center_criterion'] = center_criterion.state_dict()
            if temp_acc > best_acc:
                best_acc = temp_acc
                # save best model
                torch.save(snapshot_obj, 
                           osp.join(config["output_path"], "best_model.pth.tar"))
            log_str = "iter: {:05d}, {} precision: {:.5f}\n".format(i, config['loss']['ly_type'], temp_acc)
            config["out_file"].write(log_str)
            config["out_file"].flush()
            writer.add_scalar("precision", temp_acc, i)

            if early_stop_engine.is_stop_training(temp_acc):
                config["out_file"].write("no improvement after {}, stop training at step {}\n".format(
                    config["early_stop_patience"], i))
                # config["out_file"].write("finish training! \n")
                break

        if (i+1) % config["snapshot_interval"] == 0:
            torch.save(snapshot_obj, 
                        osp.join(config["output_path"], "iter_{:05d}_model.pth.tar".format(i)))
                    

        ## train one iter
        base_network.train(True)
        optimizer = lr_scheduler(param_lr, optimizer, i, **schedule_param)
        optimizer.zero_grad()
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()
        if use_gpu:
            inputs_source, inputs_target, labels_source = \
                Variable(inputs_source).cuda(), Variable(inputs_target).cuda(), \
                Variable(labels_source).cuda()
        else:
            inputs_source, inputs_target, labels_source = Variable(inputs_source), \
                Variable(inputs_target), Variable(labels_source)
           
        inputs = torch.cat((inputs_source, inputs_target), dim=0)
        source_batch_size = inputs_source.size(0)

        if config['loss']['ly_type'] == 'cosine':
            features, logits = base_network(inputs)
            source_logits = logits.narrow(0, 0, source_batch_size)
        elif config['loss']['ly_type'] == 'euclidean':
            features, _ = base_network(inputs)
            logits = -1.0 * loss.distance_to_centroids(features, center_criterion.centers.detach())
            source_logits = logits.narrow(0, 0, source_batch_size)

        transfer_loss = transfer_criterion(features[:source_batch_size], features[source_batch_size:])

        # source domain classification task loss
        classifier_loss = class_criterion(source_logits, labels_source)
        # fisher loss on labeled source domain
        fisher_loss, fisher_intra_loss, fisher_inter_loss, center_grad = center_criterion(features.narrow(0, 0, int(inputs.size(0)/2)), labels_source, inter_class=loss_params["inter_type"], 
                                                                               intra_loss_weight=loss_params["intra_loss_coef"], inter_loss_weight=loss_params["inter_loss_coef"])
        # entropy minimization loss
        em_loss = loss.EntropyLoss(nn.Softmax(dim=1)(logits))
        
        # final loss
        total_loss = loss_params["trade_off"] * transfer_loss \
                     + fisher_loss \
                     + loss_params["em_loss_coef"] * em_loss \
                     + classifier_loss
        total_loss.backward()
        
        if center_grad is not None:
            # clear mmc_loss
            center_criterion.centers.grad.zero_()
            # Manually assign centers gradients other than using autograd
            center_criterion.centers.backward(center_grad)

        optimizer.step()

        if i % config["log_iter"] == 0:
            config['out_file'].write('iter {} transfer loss={:0.4f}, cls loss={:0.4f}, '
                'em loss={:0.4f}, '
                'mmc loss={:0.4f}, intra loss={:0.4f}, inter loss={:0.4f}\n'.format(
                i, transfer_loss.data.cpu().float().item(), classifier_loss.data.cpu().float().item(), 
                em_loss.data.cpu().float().item(), 
                fisher_loss.cpu().float().item(), fisher_intra_loss.cpu().float().item(), fisher_inter_loss.cpu().float().item(),
                ))
            config['out_file'].flush()
            writer.add_scalar("total_loss", total_loss.data.cpu().float().item(), i)
            writer.add_scalar("cls_loss", classifier_loss.data.cpu().float().item(), i)
            writer.add_scalar("transfer_loss", transfer_loss.data.cpu().float().item(), i)
            writer.add_scalar("d_loss/total", fisher_loss.data.cpu().float().item(), i)
            writer.add_scalar("d_loss/intra", fisher_intra_loss.data.cpu().float().item(), i)
            writer.add_scalar("d_loss/inter", fisher_inter_loss.data.cpu().float().item(), i)
            
    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Feature-based Transfer Learning')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--lr', type=float, help="learning rate")
    parser.add_argument('--loss_type', type=str, default='mmd', choices=["coral", "mmd"], help="type of transfer loss.")
    parser.add_argument('--ly_type', type=str, default="cosine", choices=["cosine", "euclidean"], help="type of classification loss.")
    parser.add_argument('--trade_off', type=float, default=1.0, help="coef of transfer_loss")
    parser.add_argument('--intra_loss_coef', type=float, default=0.0, help="coef of intra_loss.")
    parser.add_argument('--inter_loss_coef', type=float, default=0.0, help="coef of inter_loss.")
    parser.add_argument('--em_loss_coef', type=float, default=0.0, help="coef of entropy minimization loss.")
    parser.add_argument('--fisher_loss_type', type=str, default="tr", 
                        choices=["tr", "td"], 
                        help="type of Fisher loss.")
    parser.add_argument('--inter_type', type=str, default="global", choices=["sample", "global"], help="type of inter_class loss.")
    parser.add_argument('--source_size', type=float, default=1.0, help="source domain sampling size")
    parser.add_argument('--target_size', type=float, default=1.0, help="target domain sampling size")
    parser.add_argument('--net', type=str, default='ResNet50', help="Options: ResNet18,34,50,101,152; AlexNet")
    parser.add_argument('--dset', type=str, default='office', help="The dataset or source dataset used")
    parser.add_argument('--s_dset_path', type=str, default='../data/office/amazon_31_list.txt', help="The source dataset path list")
    parser.add_argument('--t_dset_path', type=str, default='../data/office/webcam_10_list.txt', help="The target dataset path list")
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=5000, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='san', help="output directory of our model (in ../snapshot directory)")
    # parser.add_argument('--note', type=str, help="description of the experiment. ")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # train config
    config = {}
    config["high"] = 1.0
    config["num_iterations"] = 12004
    # config["num_iterations"] = 1 # debug
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["output_for_test"] = True
    config["output_path"] = args.output_dir
    config["log_iter"] = 100
    config["early_stop_patience"] = 10

    if not osp.exists(config["output_path"]):
        os.makedirs(config["output_path"])
    config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")
    if not osp.exists(config["output_path"]):
        os.makedirs(config["output_path"])

    config["prep"] = {"test_10crop":True, "resize_size":256, "crop_size":224, 
                      "source_size": args.source_size, "target_size": args.target_size}

    # set loss
    loss_dict = {"coral":loss.CORAL, "mmd":loss.mmd_distance}
    fisher_loss_dict = {"tr": loss.FisherTR, 
                         "td": loss.FisherTD, 
                         }
    config["loss"] = {"name": loss_dict[args.loss_type], 
                      "ly_type": args.ly_type, 
                      "fisher_loss_type": args.fisher_loss_type,
                      "discriminant_loss": fisher_loss_dict[args.fisher_loss_type],
                      "trade_off":args.trade_off, "update_iter":500, 
                      "intra_loss_coef": args.intra_loss_coef, "inter_loss_coef": args.inter_loss_coef, "inter_type": args.inter_type, 
                      "em_loss_coef": args.em_loss_coef, }
    
    if "AlexNet" in args.net:
        config["network"] = {"name":network.AlexNetFc, \
            "params":{"use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    elif "ResNet" in args.net:
        config["network"] = {"name":network.ResNetFc, \
            "params":{"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    elif "VGG" in args.net:
        config["network"] = {"name":network.VGGFc, \
            "params":{"vgg_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    config["optimizer"] = {"type":"SGD", "optim_params":{"lr":1.0, "momentum":0.9, \
                           "weight_decay":0.0005, "nesterov":True}, "lr_type":"inv", \
                           "lr_param":{"init_lr":0.001, "gamma":0.001, "power":0.75} }

    if args.lr is not None:
        config["optimizer"]["lr_param"]["init_lr"] = args.lr
    config["dataset"] = args.dset
    if config["dataset"] == "office":
        config["early_stop_patience"] = 5
        config["data"] = {"source":{"list_path":args.s_dset_path, "batch_size":36}, \
                          "target":{"list_path":args.t_dset_path, "batch_size":36}, \
                          "test":{"list_path":args.t_dset_path, "batch_size":4}}
        if "amazon" in config["data"]["test"]["list_path"]:
            config["test_interval"] = 200
            config["early_stop_patience"] = 10
        if args.lr is None:
            if "amazon" in config["data"]["test"]["list_path"]:
                config["optimizer"]["lr_param"]["init_lr"] = 0.0003
            else:
                config["optimizer"]["lr_param"]["init_lr"] = 0.001
        config["network"]["params"]["class_num"] = 31
    elif config["dataset"] == "office-home":
        config["num_iterations"] = 25004
        config["early_stop_patience"] = 5
        config["data"] = {"source":{"list_path":args.s_dset_path, "batch_size":36}, \
                          "target":{"list_path":args.t_dset_path, "batch_size":36}, \
                          "test":{"list_path":args.t_dset_path, "batch_size":4}}
        config["network"]["params"]["class_num"] = 65
        if "Real_World" in args.s_dset_path and "Art" in args.t_dset_path:
            if args.lr is None:
                config["optimizer"]["lr_param"]["init_lr"] = 0.0003
        elif "Clipart" in args.s_dset_path and "Art" in args.t_dset_path:
            config["loss"]["em_loss_coef"] = 0.
        elif "Real_World" in args.s_dset_path:
            if args.lr is None:
                config["optimizer"]["lr_param"]["init_lr"] = 0.001
        elif "Art" in args.s_dset_path:
            if args.lr is None:
                config["optimizer"]["lr_param"]["init_lr"] = 0.0003
            config["high"] = 0.5
            if "Real_World" in args.t_dset_path:
                config["high"] = 0.25
        elif "Product" in args.s_dset_path:
            if args.lr is None:
                config["optimizer"]["lr_param"]["init_lr"] = 0.0003
            config["high"] = 0.5
            if "Real_World" in args.t_dset_path:
                config["high"] = 0.3
        else:
            if args.lr is None:
                config["optimizer"]["lr_param"]["init_lr"] = 0.0003
            if "Real_World" in args.t_dset_path:
                config["high"] = 0.5
    else:
        raise ValueError('{} cannot be found. '.format(config["dataset"]))
    config["out_file"].write("config: {}\n".format(config))
    config["out_file"].flush()
    train(config)
    config["out_file"].write("finish training! \n")
    config["out_file"].close()
