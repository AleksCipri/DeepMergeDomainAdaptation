import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import network
import loss
import lr_schedule
import torchvision.transforms as transform

from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.autograd import Variable
from galaxy_utils import domain_cls_accuracy
from import_and_normalize import array_to_tensor, update

optim_dict = {"SGD": optim.SGD, "Adam": optim.Adam}

def train(config):
    class_num = config["network"]["params"]["class_num"]
    loss_params = config["loss"]
    class_criterion = nn.CrossEntropyLoss()
    transfer_criterion = loss.PADA
    center_criterion = loss_params["loss_type"](num_classes=class_num, 
                                       feat_dim=config["network"]["params"]["bottleneck_dim"])

    ## prepare data
    dsets = {}
    dset_loaders = {}

    #sampling WOR
    pristine_indices = torch.randperm(len(pristine_x))
    
    pristine_x_train = pristine_x[pristine_indices]
    pristine_y_train = pristine_y[pristine_indices]

    noisy_indices = torch.randperm(len(noisy_x))
    noisy_x_train = noisy_x[noisy_indices]
    noisy_y_train = noisy_y[noisy_indices]

    dsets["source"] = TensorDataset(pristine_x_train, pristine_y_train)
    dsets["target"] = TensorDataset(noisy_x_train, noisy_y_train)

    dset_loaders["source"] = DataLoader(dsets["source"], batch_size =128, shuffle = True, num_workers = 1)
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size = 128, shuffle = True, num_workers = 1)

    config["num_iterations"] = len(dset_loaders["source"])*config["epochs"]+1
    config["test_interval"] = len(dset_loaders["source"])
    config["snapshot_interval"] = len(dset_loaders["source"])*config["epochs"]*.25
    config["log_iter"] = len(dset_loaders["source"])

    ## set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])

    use_gpu = torch.cuda.is_available()

    if use_gpu:
        base_network = base_network.cuda()

    ## add additional network for some methods
    ad_net = network.AdversarialNetwork(base_network.output_num())
    gradient_reverse_layer = network.AdversarialLayer(high_value = config["high"])

    if use_gpu:
        ad_net = ad_net.cuda()

        ## collect parameters
    if "DeepMerge" in args.net:
        parameter_list = [{"params":base_network.parameters(), "lr_mult":1, 'decay_mult':2}]
        parameter_list.append({"params":ad_net.parameters(), "lr_mult":.1, 'decay_mult':2})
        parameter_list.append({"params":center_criterion.parameters(), "lr_mult": 10, 'decay_mult':1})
    elif "ResNet18" in args.net:
        parameter_list = [{"params":base_network.parameters(), "lr_mult":1, 'decay_mult':2}]
        parameter_list.append({"params":ad_net.parameters(), "lr_mult":.1, 'decay_mult':2})
        parameter_list.append({"params":center_criterion.parameters(), "lr_mult": 10, 'decay_mult':1})

	    if net_config["params"]["new_cls"]:
	        if net_config["params"]["use_bottleneck"]:
	            parameter_list = [{"params":base_network.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
	                            {"params":base_network.bottleneck.parameters(), "lr_mult":10, 'decay_mult':2}, \
	                            {"params":base_network.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
	            parameter_list.append({"params":ad_net.parameters(), "lr_mult": config["ad_net_mult_lr"], 'decay_mult':2})
	            parameter_list.append({"params":center_criterion.parameters(), "lr_mult": 10, 'decay_mult':1})
	        else:
	            parameter_list = [{"params":base_network.feature_layers.parameters(), "lr_mult":1, 'decay_mult':2}, \
	                            {"params":base_network.fc.parameters(), "lr_mult":10, 'decay_mult':2}]
	            parameter_list.append({"params":ad_net.parameters(), "lr_mult": config["ad_net_mult_lr"], 'decay_mult':2})
	            parameter_list.append({"params":center_criterion.parameters(), "lr_mult": 10, 'decay_mult':1})
    else:
        parameter_list = [{"params":base_network.parameters(), "lr_mult":1, 'decay_mult':2}]
        parameter_list.append({"params":ad_net.parameters(), "lr_mult": config["ad_net_mult_lr"], 'decay_mult':2})
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
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])

    transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
    best_acc = 0.0

    for i in range(config["num_iterations"]):
        
        ## train one iter
        base_network.train(True)

        if i % config["log_iter"] == 0:
            optimizer = lr_scheduler(param_lr, optimizer, i, config["log_iter"], config["frozen lr"], config["cycle_length"], **schedule_param)

        if config["optimizer"]["lr_type"] == "one-cycle":
            optimizer = lr_scheduler(param_lr, optimizer, i, config["log_iter"], config["frozen lr"], config["cycle_length"], **schedule_param)
        elif config["optimizer"]["lr_type"] == "linear":
            optimizer = lr_scheduler(param_lr, optimizer, i, config["log_iter"], config["frozen lr"], config["cycle_length"], **schedule_param)

        optim = optimizer.state_dict()
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

        ad_net.train(True)
        weight_ad = torch.ones(inputs.size(0))
        transfer_loss = transfer_criterion(features, ad_net, gradient_reverse_layer, \
                                            weight_ad, use_gpu)
        ad_out, _ = ad_net(features.detach())
        ad_acc, source_acc_ad, target_acc_ad = domain_cls_accuracy(ad_out)

        # source domain classification task loss
        classifier_loss = class_criterion(source_logits, labels_source.long())
        # fisher loss on labeled source domain

        if config["fisher_or_no"] == 'no':
            total_loss = loss_params["trade_off"] * transfer_loss \
            + classifier_loss

            total_loss.backward()

            optimizer.step()

        else:       
            fisher_loss, fisher_intra_loss, fisher_inter_loss, center_grad = center_criterion(features.narrow(0, 0, int(inputs.size(0)/2)), labels_source, inter_class=config["loss"]["inter_type"], 
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

    return total_loss.cpu().float().item()
