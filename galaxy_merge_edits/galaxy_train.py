'''
script to launch training: 
nohup python2 train_pada.py --gpu_id 1 --net ResNet50 --dset office --s_dset_path ../data/office/webcam_31_list.txt --t_dset_path ../data/office/amazon_10_list.txt --test_interval 500 --snapshot_interval 10000 --output_dir san/w2a
'''

import argparse
import os
import os.path as osp
import sys
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
from galaxy_utils import EarlyStopping, image_classification_test, distance_classification_test, domain_cls_accuracy, visualizePerformance
from import_and_normalize import array_to_tensor, update
from visualize import plot_grad_flow, plot_learning_rate_scan

optim_dict = {"SGD": optim.SGD, "Adam": optim.Adam}

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.enabled=False
torch.backends.cudnn.deterministic=True

def train(config):
    ## set up summary writer
    writer = SummaryWriter(config['output_path'])
    class_num = config["network"]["params"]["class_num"]
    class_criterion = nn.CrossEntropyLoss()
    transfer_criterion = config["loss"]["name"]
    center_criterion = config["loss"]["discriminant_loss"](num_classes=class_num, feat_dim=config["network"]["params"]["bottleneck_dim"])
    loss_params = config["loss"]

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
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size = 128, shuffle = True, num_workers = 1)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size = 128, shuffle = True, num_workers = 1)

    #guessing batch size based on what was done for testing in the original file
    dset_loaders["source_valid"] = DataLoader(dsets["source_valid"], batch_size = 64, shuffle = True, num_workers = 1)
    dset_loaders["target_valid"] = DataLoader(dsets["target_valid"], batch_size = 64, shuffle = True, num_workers = 1)

    dset_loaders["source_test"] = DataLoader(dsets["source_test"], batch_size = 64, shuffle = True, num_workers = 1)
    dset_loaders["target_test"] = DataLoader(dsets["target_test"], batch_size = 64, shuffle = True, num_workers = 1)

    config['out_file'].write("dataset sizes: source={}, target={}\n".format(
        len(dsets["source"]), len(dsets["target"])))

    config["num_iterations"] = len(dset_loaders["source"])*config["epochs"]+1
    config["test_interval"] = len(dset_loaders["source"])
    config["snapshot_interval"] = len(dset_loaders["source"])*config["epochs"]*.25
    config["log_iter"] = len(dset_loaders["source"])

    #print the configuration you are using
    config["out_file"].write("config: {}\n".format(config))
    config["out_file"].flush()

    # set up early stop
    early_stop_engine = EarlyStopping(config["early_stop_patience"])

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
        parameter_list = [{"params":base_network.parameters(), "lr_mult":10, 'decay_mult':2}]

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

    scan_lr = []
    scan_loss = []

    ## train   
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    len_valid_source = len(dset_loaders["source_valid"])
    len_valid_target = len(dset_loaders["target_valid"])

    transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
    best_acc = 0.0

    for i in range(config["num_iterations"]):

        if i % config["test_interval"] == 0:
            base_network.train(False)
            if config['loss']['ly_type'] == "cosine":
                temp_acc, _ = image_classification_test(dset_loaders, 'source_valid', \
                    base_network, gpu=use_gpu, verbose = False, save_where = None)
                train_acc, _ = image_classification_test(dset_loaders, 'source', \
                    base_network, gpu=use_gpu, verbose = False, save_where = None)
            elif config['loss']['ly_type'] == "euclidean":
                temp_acc, _ = distance_classification_test(dset_loaders, 'source_valid', \
                    base_network, center_criterion.centers.detach(), gpu=use_gpu, verbose = False, save_where = None)
                train_acc, _ = distance_classification_test(dset_loaders, 'source', \
                    base_network, center_criterion.centers.detach(), gpu=use_gpu, verbose = False, save_where = None)
            else:
                raise ValueError("no test method for cls loss: {}".format(config['loss']['ly_type']))
            
            snapshot_obj = {'epoch': i/len(dset_loaders["source"]), 
                            "base_network": base_network.state_dict(), 
                            'valid accuracy': temp_acc,
                            'train accuracy' : train_acc,
                            }

            snapshot_obj['center_criterion'] = center_criterion.state_dict()

            if (i+1) % config["snapshot_interval"] == 0:
                torch.save(snapshot_obj, 
                        osp.join(config["output_path"], "epoch_{}_model.pth.tar".format(i/len(dset_loaders["source"]))))

            if temp_acc > best_acc:
                best_acc = temp_acc
                # save best model
                torch.save(snapshot_obj, 
                           osp.join(config["output_path"], "best_model.pth.tar"))
            
            log_str = "epoch: {}, {} validation accuracy: {:.5f}, {} training accuracy: {:.5f}\n".format(i/len(dset_loaders["source"]), config['loss']['ly_type'], temp_acc, config['loss']['ly_type'], train_acc)
            config["out_file"].write(log_str)
            config["out_file"].flush()
            writer.add_scalar("validation accuracy", temp_acc, i/len(dset_loaders["source"]))
            writer.add_scalar("training accuracy", train_acc, i/len(dset_loaders["source"]))

        ## train one iter
        base_network.train(True)

        if i % config["log_iter"] == 0:
            optimizer = lr_scheduler(param_lr, optimizer, i, config["log_iter"], config["frozen lr"], config["cycle_length"], **schedule_param)

        if config["optimizer"]["lr_type"] == "one-cycle":
            optimizer = lr_scheduler(param_lr, optimizer, i, config["log_iter"], config["frozen lr"], config["cycle_length"], **schedule_param)

        if config["optimizer"]["lr_type"] == "linear":
            optimizer = lr_scheduler(param_lr, optimizer, i, config["log_iter"], config["frozen lr"], config["cycle_length"], **schedule_param)


        optim = optimizer.state_dict()
        scan_lr.append(optim['param_groups'][0]['lr'])

        optimizer.zero_grad()

        try:
            inputs_source, labels_source = iter(dset_loaders["source"]).next()
            inputs_target, labels_target = iter(dset_loaders["target"]).next()

        except StopIteration:
            iter(dset_loaders["source"])
            iter(dset_loaders["target"])

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
        classifier_loss = class_criterion(source_logits, labels_source.long())
        
        if config["fisher_or_no"] == 'no':
            total_loss = loss_params["trade_off"] * transfer_loss \
            + classifier_loss

            scan_loss.append(total_loss.cpu().float().item())

            total_loss.backward()

            ######################################
            # Plot embeddings periodically.
            if args.blobs is not None and i/len(dset_loaders["source"]) % 20 == 0:
                visualizePerformance(base_network, dset_loaders["source"], dset_loaders["target"], batch_size=128,  num_of_samples=2000, imgName='embedding_' + str(i/len(dset_loaders["source"])), save_dir=osp.join(config["output_path"], "blobs"))

            # if center_grad is not None:
            #     # clear mmc_loss
            #     center_criterion.centers.grad.zero_()
            #     # Manually assign centers gradients other than using autograd
            #     center_criterion.centers.backward(center_grad)

            optimizer.step()

            if i % config["log_iter"] == 0:

                if config['lr_scan'] != 'no':
                    if not osp.exists(osp.join(config["output_path"], "learning_rate_scan")):
                        os.makedirs(osp.join(config["output_path"], "learning_rate_scan"))

                    plot_learning_rate_scan(scan_lr, scan_loss, i/len(dset_loaders["source"]), osp.join(config["output_path"], "learning_rate_scan"))

                if config['grad_vis'] != 'no':
                    if not osp.exists(osp.join(config["output_path"], "gradients")):
                        os.makedirs(osp.join(config["output_path"], "gradients"))

                    plot_grad_flow(osp.join(config["output_path"], "gradients"), i/len(dset_loaders["source"]), base_network.named_parameters())

                config['out_file'].write('epoch {}: train total loss={:0.4f}, train transfer loss={:0.4f}, train classifier loss={:0.4f}\n'.format(
                    i/len(dset_loaders["source"]), total_loss.data.cpu().float().item(), transfer_loss.data.cpu().float().item(), classifier_loss.data.cpu().float().item(),))

                config['out_file'].flush()

                writer.add_scalar("training total loss", total_loss.data.cpu().float().item(), i/len(dset_loaders["source"]))
                writer.add_scalar("training classifier loss", classifier_loss.data.cpu().float().item(), i/len(dset_loaders["source"]))
                writer.add_scalar("training transfer loss", transfer_loss.data.cpu().float().item(), i/len(dset_loaders["source"]))

                #attempted validation step
                for j in range(0, len(dset_loaders["source_valid"])):
                    base_network.train(False)
                    with torch.no_grad():

                        try:
                            inputs_valid_source, labels_valid_source = iter(dset_loaders["source_valid"]).next()
                            inputs_valid_target, labels_valid_target = iter(dset_loaders["target_valid"]).next()
                        except StopIteration:
                            iter(dset_loaders["source_valid"])
                            iter(dset_loaders["target_valid"])

                        if use_gpu:
                            inputs_valid_source, inputs_valid_target, labels_valid_source = \
                                Variable(inputs_valid_source).cuda(), Variable(inputs_valid_target).cuda(), \
                                Variable(labels_valid_source).cuda()
                        else:
                            inputs_valid_source, inputs_valid_target, labels_valid_source = Variable(inputs_valid_source), \
                                Variable(inputs_valid_target), Variable(labels_valid_source)
                           
                        valid_inputs = torch.cat((inputs_valid_source, inputs_valid_target), dim=0)
                        valid_source_batch_size = inputs_valid_source.size(0)

                        if config['loss']['ly_type'] == 'cosine':
                            features, logits = base_network(valid_inputs)
                            source_logits = logits.narrow(0, 0, valid_source_batch_size)
                        elif config['loss']['ly_type'] == 'euclidean':
                            features, _ = base_network(valid_inputs)
                            logits = -1.0 * loss.distance_to_centroids(features, center_criterion.centers.detach())
                            source_logits = logits.narrow(0, 0, valid_source_batch_size)

                        # source domain classification task loss
                        classifier_loss = class_criterion(source_logits, labels_valid_source.long())

                        total_loss = loss_params["trade_off"] * transfer_loss \
                                    + classifier_loss

                    if j % len(dset_loaders["source_valid"]) == 0:
                        config['out_file'].write('epoch {}: valid total loss={:0.4f}, valid transfer loss={:0.4f}, valid classifier loss={:0.4f}\n'.format(
                            i/len(dset_loaders["source"]), total_loss.data.cpu().float().item(), transfer_loss.data.cpu().float().item(), classifier_loss.data.cpu().float().item(),))
                        config['out_file'].flush()
                        writer.add_scalar("validation total loss", total_loss.data.cpu().float().item(), i/len(dset_loaders["source"]))
                        writer.add_scalar("validation classifier loss", classifier_loss.data.cpu().float().item(), i/len(dset_loaders["source"]))
                        writer.add_scalar("validation transfer loss", transfer_loss.data.cpu().float().item(), i/len(dset_loaders["source"]))

                        if early_stop_engine.is_stop_training(classifier_loss.cpu().float().item()):
                            config["out_file"].write("no improvement after {}, stop training at step {}\n".format(
                            config["early_stop_patience"], i/len(dset_loaders["source"])))
                            
                            sys.exit()   


        else: # fisher loss on labeled source domain
            fisher_loss, fisher_intra_loss, fisher_inter_loss, center_grad = center_criterion(features.narrow(0, 0, int(inputs.size(0)/2)), labels_source, inter_class=loss_params["inter_type"], intra_loss_weight=loss_params["intra_loss_coef"], inter_loss_weight=loss_params["inter_loss_coef"])
                                                                                   
            # entropy minimization loss
            em_loss = loss.EntropyLoss(nn.Softmax(dim=1)(logits))

            total_loss = loss_params["trade_off"] * transfer_loss \
                 + fisher_loss \
                 + loss_params["em_loss_coef"] * em_loss \
                 + classifier_loss

            scan_loss.append(total_loss.cpu().float().item())
            
            total_loss.backward()

            ######################################
            # Plot embeddings periodically.
            if args.blobs is not None and i/len(dset_loaders["source"]) % 20 == 0:
                visualizePerformance(base_network, dset_loaders["source"], dset_loaders["target"], batch_size=128, num_of_samples=2000, imgName='embedding_' + str(i/len(dset_loaders["source"])), save_dir=osp.join(config["output_path"], "blobs"))
            ##########################################

            if center_grad is not None:
                # clear mmc_loss
                center_criterion.centers.grad.zero_()
                # Manually assign centers gradients other than using autograd
                center_criterion.centers.backward(center_grad)

            optimizer.step()

            if i % config["log_iter"] == 0:

                if config['lr_scan'] != 'no':
                    if not osp.exists(osp.join(config["output_path"], "learning_rate_scan")):
                        os.makedirs(osp.join(config["output_path"], "learning_rate_scan"))

                    plot_learning_rate_scan(scan_lr, scan_loss, i/len(dset_loaders["source"]), osp.join(config["output_path"], "learning_rate_scan"))

                if config['grad_vis'] != 'no':
                    if not osp.exists(osp.join(config["output_path"], "gradients")):
                        os.makedirs(osp.join(config["output_path"], "gradients"))

                    plot_grad_flow(osp.join(config["output_path"], "gradients"), i/len(dset_loaders["source"]), base_network.named_parameters())

                config['out_file'].write('epoch {}: train total loss={:0.4f}, train transfer loss={:0.4f}, train classifier loss={:0.4f}, '
                    'train entropy min loss={:0.4f}, '
                    'train fisher loss={:0.4f}, train intra-group fisher loss={:0.4f}, train inter-group fisher loss={:0.4f}\n'.format(
                    i/len(dset_loaders["source"]), total_loss.data.cpu(), transfer_loss.data.cpu().float().item(), classifier_loss.data.cpu().float().item(), 
                    em_loss.data.cpu().float().item(), 
                    fisher_loss.cpu().float().item(), fisher_intra_loss.cpu().float().item(), fisher_inter_loss.cpu().float().item(),
                    ))
                config['out_file'].flush()
                writer.add_scalar("training total loss", total_loss.data.cpu().float().item(), i/len(dset_loaders["source"]))
                writer.add_scalar("training transfer loss", transfer_loss.data.cpu().float().item(), i/len(dset_loaders["source"]))
                writer.add_scalar("training classifier loss", classifier_loss.data.cpu().float().item(), i/len(dset_loaders["source"]))
                writer.add_scalar("training entropy minimization loss", em_loss.data.cpu().float().item(), i/len(dset_loaders["source"]))
                writer.add_scalar("training total fisher loss", fisher_loss.data.cpu().float().item(), i/len(dset_loaders["source"]))
                writer.add_scalar("training intra-group fisher", fisher_intra_loss.data.cpu().float().item(), i/len(dset_loaders["source"]))
                writer.add_scalar("training inter-group fisher", fisher_inter_loss.data.cpu().float().item(), i/len(dset_loaders["source"]))

                #attempted validation step
                for j in range(0, len(dset_loaders["source_valid"])):
                    base_network.train(False)
                    with torch.no_grad():

                        try:
                            inputs_valid_source, labels_valid_source = iter(dset_loaders["source_valid"]).next()
                            inputs_valid_target, labels_valid_target = iter(dset_loaders["target_valid"]).next()

                        except StopIteration:
                            iter(dset_loaders["source_valid"])
                            iter(dset_loaders["target_valid"])

                        if use_gpu:
                            inputs_valid_source, inputs_valid_target, labels_valid_source = \
                                Variable(inputs_valid_source).cuda(), Variable(inputs_valid_target).cuda(), \
                                Variable(labels_valid_source).cuda()
                        else:
                            inputs_valid_source, inputs_valid_target, labels_valid_source = Variable(inputs_valid_source), \
                                Variable(inputs_valid_target), Variable(labels_valid_source)
                           
                        valid_inputs = torch.cat((inputs_valid_source, inputs_valid_target), dim=0)
                        valid_source_batch_size = inputs_valid_source.size(0)

                        if config['loss']['ly_type'] == 'cosine':
                            features, logits = base_network(valid_inputs)
                            source_logits = logits.narrow(0, 0, valid_source_batch_size)
                        elif config['loss']['ly_type'] == 'euclidean':
                            features, _ = base_network(valid_inputs)
                            logits = -1.0 * loss.distance_to_centroids(features, center_criterion.centers.detach())
                            source_logits = logits.narrow(0, 0, valid_source_batch_size)

                        transfer_loss = transfer_criterion(features[:valid_source_batch_size], features[valid_source_batch_size:])

                        # source domain classification task loss
                        classifier_loss = class_criterion(source_logits, labels_valid_source.long())
                        # fisher loss on labeled source domain
                        fisher_loss, fisher_intra_loss, fisher_inter_loss, center_grad = center_criterion(features.narrow(0, 0, int(valid_inputs.size(0)/2)), labels_valid_source, inter_class=loss_params["inter_type"], intra_loss_weight=loss_params["intra_loss_coef"], inter_loss_weight=loss_params["inter_loss_coef"])
                                                                                               
                        # entropy minimization loss
                        em_loss = loss.EntropyLoss(nn.Softmax(dim=1)(logits))
                        
                        # final loss
                        total_loss = loss_params["trade_off"] * transfer_loss \
                                     + fisher_loss \
                                     + loss_params["em_loss_coef"] * em_loss \
                                     + classifier_loss
                        #total_loss.backward() no backprop on the eval mode

                    if j % len(dset_loaders["source_valid"]) == 0:
                        config['out_file'].write('epoch {}, valid transfer loss={:0.4f}, valid classifier loss={:0.4f}, '
                            'valid entropy min loss={:0.4f}, '
                            'valid fisher loss={:0.4f}, valid intra-group fisher loss={:0.4f}, valid inter-group fisher loss={:0.4f}\n'.format(
                            i/len(dset_loaders["source"]), transfer_loss.data.cpu().float().item(), classifier_loss.data.cpu().float().item(), 
                            em_loss.data.cpu().float().item(), 
                            fisher_loss.cpu().float().item(), fisher_intra_loss.cpu().float().item(), fisher_inter_loss.cpu().float().item(),
                            ))
                        config['out_file'].flush()
                        writer.add_scalar("validation total loss", total_loss.data.cpu().float().item(), i/len(dset_loaders["source"]))
                        writer.add_scalar("validation transfer loss", transfer_loss.data.cpu().float().item(), i/len(dset_loaders["source"]))
                        writer.add_scalar("validation classifier loss", classifier_loss.data.cpu().float().item(), i/len(dset_loaders["source"]))
                        writer.add_scalar("validation entropy minimization loss", em_loss.data.cpu().float().item(), i/len(dset_loaders["source"]))
                        writer.add_scalar("validation total fisher loss", fisher_loss.data.cpu().float().item(), i/len(dset_loaders["source"]))
                        writer.add_scalar("validation intra-group fisher", fisher_intra_loss.data.cpu().float().item(), i/len(dset_loaders["source"]))
                        writer.add_scalar("validation inter-group fisher", fisher_inter_loss.data.cpu().float().item(), i/len(dset_loaders["source"]))

                        if early_stop_engine.is_stop_training(classifier_loss.cpu().float().item()):
                            config["out_file"].write("no improvement after {}, stop training at epoch {}\n".format(
                            config["early_stop_patience"], i/len(dset_loaders["source"])))

                            sys.exit()

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
    parser.add_argument('--net', type=str, default='ResNet50', help="Options: ResNet18,34,50,101,152; AlexNet")
    parser.add_argument('--dset', type=str, default='galaxy', help="The dataset or source dataset used")
    # parser.add_argument('--dset_path', type=str, default='/arrays', help="The source dataset path")
    parser.add_argument('--output_dir', type=str, default='san', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--optim_choice', type=str, default='SGD', help='Adam or SGD')
    parser.add_argument('--epochs', type=int, default=200, help='How many epochs do you want to train?')
    parser.add_argument('--grad_vis', type=str, default='no', help='Do you want to visualize your gradients?')
    parser.add_argument('--dset_path', type=str, default=None, help="The dataset directory path")
    parser.add_argument('--source_x_file', type=str, default='SB_version_00_numpy_3_filters_pristine_SB00_augmented_3FILT.npy',
                         help="Source domain x-values filename")
    parser.add_argument('--source_y_file', type=str, default='SB_version_00_numpy_3_filters_pristine_SB00_augmented_y_3FILT.npy',
                         help="Source domain y-values filename")
    parser.add_argument('--target_x_file', type=str, default='SB_version_00_numpy_3_filters_noisy_SB25_augmented_3FILT.npy',
                         help="Target domain x-values filename")
    parser.add_argument('--target_y_file', type=str, default='SB_version_00_numpy_3_filters_noisy_SB25_augmented_y_3FILT.npy',
                         help="Target domain y-values filename")
    parser.add_argument('--one_cycle', type=str, default = None, help='Do you want to turn on one-cycle learning rate?')
    parser.add_argument('--lr_scan', type=str, default = 'no', help='Set to yes for learning rate scan')
    parser.add_argument('--cycle_length', type=int, default = 2, help = 'If using one-cycle learning, how many epochs should one learning rate cycle be?')
    parser.add_argument('--early_stop_patience', type=int, default = 10, help = 'Number of epochs for early stopping.')
    parser.add_argument('--weight_decay', type=float, default = 5e-4, help= 'How much do you want to penalize large weights?')
    parser.add_argument('--blobs', type=str, default=None, help='Plot blob figures.')
    parser.add_argument('--fisher_or_no', type=str, default='Fisher', help='run the code without fisher loss')
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # train config
    config = {}
    config["epochs"] = args.epochs
    config["output_for_test"] = True
    config["output_path"] = args.output_dir
    config["optim_choice"] = args.optim_choice
    config["grad_vis"] = args.grad_vis
    config["lr_scan"] = args.lr_scan
    config["cycle_length"] = args.cycle_length
    config["early_stop_patience"] = args.early_stop_patience
    config["weight_decay"] = args.weight_decay
    config["blobs"] = args.blobs
    config["fisher_or_no"] = args.fisher_or_no

    if not osp.exists(config["output_path"]):
        os.makedirs(config["output_path"])
        config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")
    if osp.exists(config["output_path"]):
        config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w") 

    # set loss
    loss_dict = {"coral":loss.CORAL, "mmd":loss.mmd_distance}
    fisher_loss_dict = {"tr": loss.FisherTR, 
                         "td": loss.FisherTD, 
                         }
    config["loss"] = {"name": loss_dict[args.loss_type], 
                      "ly_type": args.ly_type, 
                      "fisher_loss_type": args.fisher_loss_type,
                      "discriminant_loss": fisher_loss_dict[args.fisher_loss_type],
                      "trade_off":args.trade_off, "update_iter":200,
                      "intra_loss_coef": args.intra_loss_coef, "inter_loss_coef": args.inter_loss_coef, "inter_type": args.inter_type, 
                      "em_loss_coef": args.em_loss_coef, }
    
    if "DeepMerge" in args.net:
        config["network"] = {"name":network.DeepMerge, \
            "params":{"class_num":2, "new_cls":True, "use_bottleneck":False, "bottleneck_dim":32*9*9} }
    elif "ResNet" in args.net:
        config["network"] = {"name":network.ResNetFc, \
            "params":{"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }

    if config["optim_choice"] == "Adam":
        config["optimizer"] = {"type":"Adam", "optim_params":{"lr":1e-6, "betas":(0.7,0.8), "weight_decay": config["weight_decay"], "amsgrad":True, "eps":1e-8}, \
                        "lr_type":"inv", "lr_param":{"init_lr":0.0001, "gamma":0.001, "power":0.75} }
    else:
        config["optimizer"] = {"type":"SGD", "optim_params":{"lr":1.0, "momentum":0.9, \
                               "weight_decay": config["weight_decay"], "nesterov":True}, "lr_type":"inv", \
                               "lr_param":{"init_lr":0.001, "gamma":0.001, "power":0.75} }

    if args.lr is not None:
        config["optimizer"]["optim_params"]["lr"] = args.lr
        config["optimizer"]["lr_param"]["init_lr"] = args.lr
        config["frozen lr"] = args.lr

    if args.one_cycle is not None:
        config["optimizer"]["lr_type"] = "one-cycle"
 
    if args.lr_scan == 'yes':
        config["optimizer"]["lr_type"] = "linear"
        config["optimizer"]["optim_params"]["lr"] = 1e-6
        config["optimizer"]["lr_param"]["init_lr"] = 1e-6
        config["frozen lr"] = 1e-6
        config["epochs"] = 5
        
    config["dataset"] = args.dset
    config["path"] = args.dset_path

    if config["dataset"] == 'galaxy':
        pristine_x = array_to_tensor(osp.join(config['path'], args.source_x_file))
        pristine_y = array_to_tensor(osp.join(config['path'], args.source_y_file))

        noisy_x = array_to_tensor(osp.join(config['path'], args.target_x_file))
        noisy_y = array_to_tensor(osp.join(config['path'], args.target_y_file))

        update(pristine_x, noisy_x)

        config["network"]["params"]["class_num"] = 2

    train(config)

    config["out_file"].write("finish training! \n")
    config["out_file"].close()
