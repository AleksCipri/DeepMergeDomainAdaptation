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
import network
import loss
import lr_schedule
import torchvision.transforms as transform

from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.autograd import Variable
from galaxy_utils import EarlyStopping, distance_classification_test, image_classification_test, domain_cls_accuracy
from import_and_normalize import array_to_tensor, update

optim_dict = {"SGD": optim.SGD, "Adam": optim.Adam}

def train(config):
    ## set up summary writer
    writer = SummaryWriter(config['output_path'])

    # set up early stop
    early_stop_engine = EarlyStopping(config["early_stop_patience"])
               
    ## set loss
    class_num = config["network"]["params"]["class_num"]
    loss_params = config["loss"]

    class_criterion = nn.CrossEntropyLoss()
    transfer_criterion = loss.PADA
    center_criterion = loss_params["loss_type"](num_classes=class_num, 
                                       feat_dim=config["network"]["params"]["bottleneck_dim"])

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

    config['out_file'].write("dataset sizes: source={}, target={}\n".format(
        len(dsets["source"]), len(dsets["target"])))

    ## set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        base_network = base_network.cuda()

    ## collect parameters
    if "DeepMerge" in args.net:
            parameter_list = [{"params":base_network.parameters(), "lr_mult":1, 'decay_mult':2}]
    elif net_config["params"]["new_cls"]:
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
    ad_net = network.AdversarialNetwork(base_network.output_num())
    gradient_reverse_layer = network.AdversarialLayer(high_value = config["high"]) #, 
                                                      #max_iter_value=config["num_iterations"])
    if use_gpu:
        ad_net = ad_net.cuda()
    parameter_list.append({"params":ad_net.parameters(), "lr_mult":10, 'decay_mult':2})
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
    len_valid_source = len(dset_loaders["source_valid"]) - 1
    len_valid_target = len(dset_loaders["target_valid"]) - 1

    transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
    best_acc = 0.0
    for i in range(config["num_iterations"]):
        if i % config["test_interval"] == 0:
            base_network.train(False)
            if config['loss']['ly_type'] == "cosine":
                temp_acc, _ = image_classification_test(dset_loaders, 'source_valid', \
                    base_network, \
                    gpu=use_gpu)
                train_acc, _ = image_classification_test(dset_loaders, 'source', \
                    base_network, \
                    gpu=use_gpu)
            elif config['loss']['ly_type'] == "euclidean":
                temp_acc, _ = distance_classification_test(dset_loaders, 'source_valid', \
                    base_network, center_criterion.centers.detach(), \
                    gpu=use_gpu)
                train_acc, _ = distance_classification_test(dset_loaders, 'source', \
                    base_network, \
                    gpu=use_gpu)
            else:
                raise ValueError("no test method for cls loss: {}".format(config['loss']['ly_type']))
            
            snapshot_obj = {'step': i, 
                            "base_network": base_network.state_dict(), 
                            'valid accuracy': temp_acc,
                            'train accuracy' : train_acc,                            
                            }
            if config["loss"]["loss_name"] != "laplacian" and config["loss"]["ly_type"] == "euclidean":
                snapshot_obj['center_criterion'] = center_criterion.state_dict()
            if temp_acc > best_acc:
                best_acc = temp_acc
                # save best model
                torch.save(snapshot_obj, 
                           osp.join(config["output_path"], "best_model.pth.tar"))
            log_str = "iter: {:05d}, {} validation accuracy: {:.5f}, {} training accuracy: {:.5f}\n".format(i, config['loss']['ly_type'], temp_acc, config['loss']['ly_type'], train_acc)
            config["out_file"].write(log_str)
            config["out_file"].flush()
            writer.add_scalar("validation accuracy", temp_acc, i)
            writer.add_scalar("training accuracy", train_acc, i)

            if early_stop_engine.is_stop_training(temp_acc):
                config["out_file"].write("no improvement after {}, stop training at step {}\n".format(
                    config["early_stop_patience"], i))
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

        try:
            inputs_source, labels_source = iter_source.next()
            inputs_target, labels_target = iter_target.next()
        except StopIteration:
            iter_source = iter(dset_loaders["source"])
            iter_target = iter(dset_loaders["target"])

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

        ad_net.train(True)
        weight_ad = torch.ones(inputs.size(0))
        # transfer_loss = transfer_criterion(features, ad_net, gradient_reverse_layer, \
        #                                    weight_ad, use_gpu)
        ad_out, _ = ad_net(features.detach())
        ad_acc, source_acc_ad, target_acc_ad = domain_cls_accuracy(ad_out)

        # source domain classification task loss
        classifier_loss = class_criterion(source_logits, labels_source.long())
        # fisher loss on labeled source domain

        if config["fisher_or_no"] == 'no':
            total_loss = classifier_loss

            if i % config["log_iter"] == 0:
                config['out_file'].write('iter {}: train total loss={:0.4f}, train classifier loss={:0.4f},'
                    'train source+target domain accuracy={:0.4f}, train source domain accuracy={:0.4f}, train target domain accuracy={:0.4f}\n'.format(
                    i, total_loss.data.cpu().float().item(), classifier_loss.data.cpu().float().item(),
                    ad_acc, source_acc_ad, target_acc_ad,
                    ))
                config['out_file'].flush()
                writer.add_scalar("training total loss", total_loss.data.cpu().float().item(), i)
                writer.add_scalar("training classifier loss", classifier_loss.data.cpu().float().item(), i)
                writer.add_scalar("training source+target domain accuracy", ad_acc, i)
                writer.add_scalar("training source domain accuracy", source_acc_ad, i)
                writer.add_scalar("training target domain accuracy", target_acc_ad, i)

        else:  
            transfer_loss = transfer_criterion(features, ad_net, gradient_reverse_layer, \
                                           weight_ad, use_gpu)      
            fisher_loss, fisher_intra_loss, fisher_inter_loss, center_grad = center_criterion(features.narrow(0, 0, int(inputs.size(0)/2)), labels_source, inter_class=config["loss"]["inter_type"], 
                                                                                   intra_loss_weight=loss_params["intra_loss_coef"], inter_loss_weight=loss_params["inter_loss_coef"])
            # entropy minimization loss
            em_loss = loss.EntropyLoss(nn.Softmax(dim=1)(logits))

            # final loss
            total_loss = loss_params["trade_off"] * transfer_loss \
                         + fisher_loss \
                         + loss_params["em_loss_coef"] * em_loss \
                         + classifier_loss

            if center_grad is not None:
                # clear mmc_loss
                center_criterion.centers.grad.zero_()
                # Manually assign centers gradients other than using autograd
                center_criterion.centers.backward(center_grad)

            if i % config["log_iter"] == 0:
                config['out_file'].write('iter {}: train total loss={:0.4f}, train transfer loss={:0.4f}, train classifier loss={:0.4f}, '
                    'train entropy min loss={:0.4f}, '
                    'train fisher loss={:0.4f}, train intra-group fisher loss={:0.4f}, train inter-group fisher loss={:0.4f}, '
                    'train source+target domain accuracy={:0.4f}, train source domain accuracy={:0.4f}, train target domain accuracy={:0.4f}\n'.format(
                    i, total_loss.data.cpu().float().item(), transfer_loss.data.cpu().float().item(), classifier_loss.data.cpu().float().item(), 
                    em_loss.data.cpu().float().item(), 
                    fisher_loss.cpu().float().item(), fisher_intra_loss.cpu().float().item(), fisher_inter_loss.cpu().float().item(),
                    ad_acc, source_acc_ad, target_acc_ad, 
                    ))

                config['out_file'].flush()
                writer.add_scalar("training total loss", total_loss.data.cpu().float().item(), i)
                writer.add_scalar("training classifier loss", classifier_loss.data.cpu().float().item(), i)
                writer.add_scalar("training transfer_loss", transfer_loss.data.cpu().float().item(), i)
                writer.add_scalar("training total fisher loss", fisher_loss.data.cpu().float().item(), i)
                writer.add_scalar("training intra-group fisher", fisher_intra_loss.data.cpu().float().item(), i)
                writer.add_scalar("training inter-group fisher", fisher_inter_loss.data.cpu().float().item(), i)
                writer.add_scalar("training source+target domain accuracy", ad_acc, i)
                writer.add_scalar("training source domain accuracy", source_acc_ad, i)
                writer.add_scalar("training target domain accuracy", target_acc_ad, i)

        total_loss.backward()

        optimizer.step()

        #attempted validation step
        #if i < len_valid_source:
        base_network.eval()
        with torch.no_grad():
            if i % len_valid_source == 0:
                iter_source = iter(dset_loaders["source_valid"])
            if i % len_valid_target == 0:
                iter_target = iter(dset_loaders["target_valid"])

            try:
                inputs_source, labels_source = iter_source.next()
                inputs_target, labels_target = iter_target.next()
            except StopIteration:
                iter_source = iter(dset_loaders["source_valid"])
                iter_target = iter(dset_loaders["target_valid"])

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

            ad_net.train(False)
            weight_ad = torch.ones(inputs.size(0))
            # transfer_loss = transfer_criterion(features, ad_net, gradient_reverse_layer, \
            #                                    weight_ad, use_gpu)
            ad_out, _ = ad_net(features.detach())
            ad_acc, source_acc_ad, target_acc_ad = domain_cls_accuracy(ad_out)

            # source domain classification task loss
            classifier_loss = class_criterion(source_logits, labels_source.long())


            if config["fisher_or_no"] == 'no':
                total_loss = classifier_loss

                if i % config["log_iter"] == 0:
                    config['out_file'].write('iter {}: valid total loss={:0.4f}, valid classifier loss={:0.4f},'
                        'valid source+target domain accuracy={:0.4f}, valid source domain accuracy={:0.4f}, valid target domain accuracy={:0.4f}\n'.format(
                        i, total_loss.data.cpu().float().item(), classifier_loss.data.cpu().float().item(),
                        ad_acc, source_acc_ad, target_acc_ad,
                        ))
                    config['out_file'].flush()
                    writer.add_scalar("validation total loss", total_loss.data.cpu().float().item(), i)
                    writer.add_scalar("validation classifier loss", classifier_loss.data.cpu().float().item(), i)
                    writer.add_scalar("validation source+target domain accuracy", ad_acc, i)
                    writer.add_scalar("validation source domain accuracy", source_acc_ad, i)
                    writer.add_scalar("validation target domain accuracy", target_acc_ad, i)
            else:
                transfer_loss = transfer_criterion(features, ad_net, gradient_reverse_layer, \
                                                   weight_ad, use_gpu)
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
            #total_loss.backward() no backprop on the eval mode

                if i % config["log_iter"] == 0:
                    config['out_file'].write('iter {}: valid total loss={:0.4f}, valid transfer loss={:0.4f}, valid classifier loss={:0.4f}, '
                        'valid entropy min loss={:0.4f}, '
                        'valid fisher loss={:0.4f}, valid intra-group fisher loss={:0.4f}, valid inter-group fisher loss={:0.4f}, '
                        'valid source+target domain accuracy={:0.4f}, valid source domain accuracy={:0.4f}, valid target domain accuracy={:0.4f}\n'.format(
                        i, total_loss.data.cpu().float().item(), transfer_loss.data.cpu().float().item(), classifier_loss.data.cpu().float().item(), 
                        em_loss.data.cpu().float().item(), 
                        fisher_loss.cpu().float().item(), fisher_intra_loss.cpu().float().item(), fisher_inter_loss.cpu().float().item(),
                        ad_acc, source_acc_ad, target_acc_ad, 
                        ))

                    config['out_file'].flush()
                    writer.add_scalar("validation total loss", total_loss.data.cpu().float().item(), i)
                    writer.add_scalar("validation classifier loss", classifier_loss.data.cpu().float().item(), i)
                    writer.add_scalar("validation transfer_loss", transfer_loss.data.cpu().float().item(), i)
                    writer.add_scalar("validation total fisher loss", fisher_loss.data.cpu().float().item(), i)
                    writer.add_scalar("validation intra-group fisher", fisher_intra_loss.data.cpu().float().item(), i)
                    writer.add_scalar("validation inter-group fisher", fisher_inter_loss.data.cpu().float().item(), i)
                    writer.add_scalar("validation source+target domain accuracy", ad_acc, i)
                    writer.add_scalar("validation source domain accuracy", source_acc_ad, i)
                    writer.add_scalar("validation target domain accuracy", target_acc_ad, i)

    return best_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TL with disciminative loss')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--lr', type=float, help="learning rate")
    parser.add_argument('--ly_type', type=str, default="cosine", choices=["cosine", "euclidean"], help="type of classification loss.")
    parser.add_argument('--trade_off', type=float, default=1.0, help="coef of transfer_loss")
    parser.add_argument('--intra_loss_coef', type=float, default=0.01, help="coef of intra_loss.")
    parser.add_argument('--inter_loss_coef', type=float, default=0.01, help="coef of inter_loss.")
    parser.add_argument('--em_loss_coef', type=float, default=0.0, help="coef of entropy minimization loss.")
    parser.add_argument('--fisher_loss_type', type=str, default="tr", 
                        choices=["tr", "td"], 
                        help="type of Fisher loss.")
    parser.add_argument('--inter_type', type=str, default="global", choices=["none", "sample", "global"], help="type of inter_class loss.")
    parser.add_argument('--source_size', type=float, default=1.0, help="source domain sampling size")
    parser.add_argument('--target_size', type=float, default=1.0, help="target domain sampling size")
    parser.add_argument('--net', type=str, default='ResNet50', help="Options: ResNet18,34,50,101,152; AlexNet")
    parser.add_argument('--dset', type=str, default='office', help="The dataset or source dataset used")
    parser.add_argument('--dset_path', type=str, default='/arrays', help="The source dataset path")
    parser.add_argument('--test_interval', type=int, default=500, help="interval of two continuous test phase")
    parser.add_argument('--snapshot_interval', type=int, default=5000, help="interval of two continuous output model")
    parser.add_argument('--output_dir', type=str, default='san', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--optim_choice', type=str, default='SGD', help='Adam or SGD')
    parser.add_argument('--fisher_or_no', type=str, default='Fisher', help='run the code without fisher loss')
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
    config["early_stop_patience"] = 100
    config["optim_choice"] = args.optim_choice
    config["fisher_or_no"] = args.fisher_or_no

    if not osp.exists(config["output_path"]):
        # os.makedirs(config["output_path"])
        os.makedirs(osp.join(config["output_path"]))
        config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")
    else:
        config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")

    loss_dict = {"tr": loss.FisherTR, 
                 "td": loss.FisherTD, 
                 }
    config["loss"] = {"loss_name": args.fisher_loss_type,
                      "loss_type": loss_dict[args.fisher_loss_type],
                      "ly_type": args.ly_type, 
                      "trade_off":args.trade_off, 
                      "intra_loss_coef": args.intra_loss_coef, "inter_loss_coef": args.inter_loss_coef, "inter_type": args.inter_type, 
                      "em_loss_coef": args.em_loss_coef}
    
    if "DeepMerge" in args.net:
        config["network"] = {"name":network.DeepMerge, \
            "params":{"class_num":2, "new_cls":True, "use_bottleneck":False, "bottleneck_dim":32*9*9} }
    elif "ResNet" in args.net:
        config["network"] = {"name":network.ResNetFc, \
            "params":{"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    

    if config["optim_choice"] == 'Adam':
        config["optimizer"] = {"type":"Adam", "optim_params":{"lr":1.0, "betas":(0.7,0.8), "weight_decay":0.0001, "amsgrad":True, "eps":1e-8}, \
                        "lr_type":"inv", "lr_param":{"init_lr":0.0001, "gamma":0.001, "power":0.75} }
    else:
        config["optimizer"] = {"type":"SGD", "optim_params":{"lr":1.0, "momentum":0.9, \
                               "weight_decay":0.0005, "nesterov":True}, "lr_type":"inv", \
                               "lr_param":{"init_lr":0.001, "gamma":0.001, "power":0.75} }

    if args.lr is not None:
        config["optimizer"]["optim_params"]["lr"] = args.lr
    if args.lr is None:
        config["optimizer"]["optim_params"]["lr"] = 0.0001
    else:
         raise ValueError('{} cannot be found. ')

    config["dataset"] = args.dset
    config["path"] = args.dset_path

    if config["dataset"] == 'galaxy': 
        pristine_x = array_to_tensor(osp.join(os.getcwd(), config['path'], 'SB_version_00_numpy_3_filters_pristine_SB00_augmented_3FILT.npy'))
        pristine_y = array_to_tensor(osp.join(os.getcwd(), config['path'], 'SB_version_00_numpy_3_filters_pristine_SB00_augmented_y_3FILT.npy'))

        noisy_x = array_to_tensor(osp.join(os.getcwd(), config['path'], 'SB_version_00_numpy_3_filters_noisy_SB25_augmented_3FILT.npy'))
        noisy_y = array_to_tensor(osp.join(os.getcwd(), config['path'], 'SB_version_00_numpy_3_filters_noisy_SB25_augmented_y_3FILT.npy'))

        update(pristine_x, noisy_x)

        config["network"]["params"]["class_num"] = 2

    else:
        raise ValueError("invalid argument {} for dataset".format(config["dataset"]))
    
    config["out_file"].write("config: {}\n".format(config))
    config["out_file"].flush()
    train(config)
    config["out_file"].write("finish training! \n")
    config["out_file"].close()
