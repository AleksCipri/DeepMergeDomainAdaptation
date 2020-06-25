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
from galaxy_utils import EarlyStopping, image_classification_test, distance_classification_test, domain_cls_accuracy
from import_and_normalize import array_to_tensor

optim_dict = {"SGD": optim.SGD, "Adam": optim.Adam}

def train(config):
    ## set up summary writer
    writer = SummaryWriter(config['output_path'])
    class_num = config["network"]["params"]["class_num"]
    class_criterion = nn.CrossEntropyLoss()
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

    dsets["source"] = TensorDataset(pristine_x_train, pristine_y_train)
    dsets["source_valid"] = TensorDataset(pristine_x_valid, pristine_y_valid)
    dsets["source_test"] = TensorDataset(pristine_x_test, pristine_y_test)

    #put your dataloaders here
    #i stole batch size numbers from below
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size = 128, shuffle = True, num_workers = 1)
    dset_loaders["source_valid"] = DataLoader(dsets["source_valid"], batch_size = 64, shuffle = True, num_workers = 1)
    dset_loaders["source_test"] = DataLoader(dsets["source_test"], batch_size = 64, shuffle = True, num_workers = 1)

    config['out_file'].write("dataset sizes: source={}\n".format(
        len(dsets["source"])))

    config["num_iterations"] = len(dset_loaders["source"])*config["epochs"]
    config["early_stop_patience"] = len(dset_loaders["source"])*20
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
        parameter_list = [{"params":base_network.parameters(), "lr_mult":1, 'decay_mult':2}]

    ## add additional network for some methods
    class_weight = torch.from_numpy(np.array([1.0] * class_num))
    if use_gpu:
        class_weight = class_weight.cuda()
 
    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optim_dict[optimizer_config["type"]](parameter_list, \
                    **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]] #just make this Adam

    ## train   
    len_train_source = len(dset_loaders["source"]) - 1
    len_valid_source = len(dset_loaders["source_valid"]) - 1

    classifier_loss_value = 0.0
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
            # you can't use the euclidean distance_loss because it involves the target domain

            else:
                raise ValueError("no test method for cls loss: {}".format(config['loss']['ly_type']))
            
            snapshot_obj = {'epoch': i/len(dset_loaders["source"]), 
                            "base_network": base_network.state_dict(), 
                            'valid accuracy': temp_acc,
                            'train accuracy' : train_acc,
                            }
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

            if early_stop_engine.is_stop_training(temp_acc):
                config["out_file"].write("no improvement after {}, stop training at epoch {}\n".format(
                    config["early_stop_patience"], i/len(dset_loaders["source"])))
                # config["out_file"].write("finish training! \n")
                break

        if (i+1) % config["snapshot_interval"] == 0:
            torch.save(snapshot_obj, 
                        osp.join(config["output_path"], "epoch_{}_model.pth.tar".format(i/len(dset_loaders["source"]))))
                    

        ## train one iter
        base_network.train(True)

        if i % config["log_iter"] == 0:
            optimizer = lr_scheduler(param_lr, optimizer, i, **schedule_param)

        optimizer.zero_grad()

        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])

        try:
            inputs_source, labels_source = iter_source.next()
        except StopIteration:
            iter_source = iter(dset_loaders["source"])

        if use_gpu:
            inputs_source, labels_source = Variable(inputs_source).cuda(), Variable(labels_source).cuda()
        else:
            inputs_source, labels_source = Variable(inputs_source), Variable(labels_source)
           
        inputs = inputs_source
        source_batch_size = inputs_source.size(0)

        features, logits = base_network(inputs)
        source_logits = logits.narrow(0, 0, source_batch_size)

        # source domain classification task loss
        classifier_loss = class_criterion(source_logits, labels_source.long())

        total_loss = classifier_loss
   
        total_loss.backward()

        optimizer.step()

        if i % config["log_iter"] == 0:
            config['out_file'].write('epoch {}: train total loss={:0.4f}, train classifier loss={:0.4f}\n'.format(i/len(dset_loaders["source"]), \
                total_loss.data.cpu(), classifier_loss.data.cpu().float().item(),))
            config['out_file'].flush()
            writer.add_scalar("training total loss", total_loss.data.cpu().float().item(), i/len(dset_loaders["source"]))
            writer.add_scalar("training classifier loss", classifier_loss.data.cpu().float().item(), i/len(dset_loaders["source"]))

            #attempted validation step
            for j in range(0, len(dset_loaders["source_valid"])):
                base_network.train(False)
                #base_network.eval() should be the same
                with torch.no_grad():
                    if i % len_valid_source == 0:
                        iter_valid = iter(dset_loaders["source_valid"])

                    try:
                        inputs_source, labels_source = iter_valid.next() #is this why it's overfitting
                    except StopIteration:
                        iter_valid = iter(dset_loaders["source_valid"])

                    if use_gpu:
                        inputs_source, labels_source = Variable(inputs_source).cuda(), Variable(labels_source).cuda()
                    else:
                        inputs_source,labels_source = Variable(inputs_source), Variable(labels_source)
                       
                    inputs = inputs_source
                    source_batch_size = inputs_source.size(0)

                    features, logits = base_network(inputs)
                    source_logits = logits.narrow(0, 0, source_batch_size)
                
                    # source domain classification task loss
                    classifier_loss = class_criterion(source_logits, labels_source.long())
                    
                    # final loss
                    total_loss = classifier_loss
                    #total_loss.backward() no backprop on the eval mode

                if j % len(dset_loaders["source_valid"]) == 0:
                    config['out_file'].write('epoch {}: valid total loss={:0.4f}, valid classifier loss={:0.4f}\n'.format(i/len(dset_loaders["source"]), \
                        total_loss.data.cpu(), classifier_loss.data.cpu().float().item(),))
                    config['out_file'].flush()
                    writer.add_scalar("validation total loss", total_loss.data.cpu().float().item(), i/len(dset_loaders["source"]))
                    writer.add_scalar("validation classifier loss", classifier_loss.data.cpu().float().item(), i/len(dset_loaders["source"]))
            
    return best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Feature-based Transfer Learning')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--ly_type', type=str, default="cosine", choices=["cosine", "euclidean"], help="type of classification loss.")
    parser.add_argument('--net', type=str, default='ResNet50', help="Options: ResNet18,34,50,101,152; AlexNet")
    parser.add_argument('--dset', type=str, default='galaxy', help="The dataset or source dataset used")
    parser.add_argument('--dset_path', type=str, default='/arrays', help="The source dataset path")
    parser.add_argument('--output_dir', type=str, default='san', help="output directory of our model (in ../snapshot directory)")
    parser.add_argument('--optim_choice', type=str, default='SGD', help='Adam or SGD')
    parser.add_argument('--epochs', type=int, default=200, help='How many epochs do you want to train?')
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # train config
    config = {}
    config["epochs"] = args.epochs
    config["output_for_test"] = True
    config["output_path"] = args.output_dir
    config["optim_choice"] = args.optim_choice

    if not osp.exists(config["output_path"]):
        os.makedirs(osp.join(config["output_path"]))
        config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")
    else:
        config["out_file"] = open(osp.join(config["output_path"], "log.txt"), "w")

    # set loss
    config["loss"] = { "ly_type": args.ly_type, 
                      "update_iter":200, }    
    
    #set network parameters
    if "DeepMerge" in args.net:
        config["network"] = {"name":network.DeepMerge, \
            "params":{"class_num":2, "new_cls":True, "use_bottleneck":False, "bottleneck_dim":32*9*9} }
    elif "ResNet" in args.net:
        config["network"] = {"name":network.ResNetFc, \
            "params":{"resnet_name":args.net, "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    
    #set optimizer
    if config["optim_choice"] == 'Adam':
        config["optimizer"] = {"type":"Adam", "optim_params":{"lr":0.001, "betas":(0.9,0.999), "weight_decay":0.01, \
                                 "amsgrad":False, "eps":1e-8} , \
                        "lr_type":"inv", "lr_param":{"init_lr":0.001, "gamma":0.001, "power":0.75}}
    else:
        config["optimizer"] = {"type":"SGD", "optim_params":{"lr":0.001, "momentum":0.9, \
                               "weight_decay":0.005, "nesterov":True}, "lr_type":"inv" , \
                               "lr_param":{"init_lr":0.001, "gamma":0.001, "power":0.75}}

    #override default if it is specified
    if args.lr is not None:
        config["optimizer"]["optim_params"]["lr"] = args.lr
        config["optimizer"]["lr_param"]["init_lr"] = args.lr
    
    #load dataset    
    config["dataset"] = args.dset
    config["path"] = args.dset_path

    if config["dataset"] == 'galaxy': 
        pristine_x = array_to_tensor(osp.join(os.getcwd(), config['path'], 'SB_version_00_numpy_3_filters_pristine_SB00_augmented_3FILT.npy'))
        pristine_y = array_to_tensor(osp.join(os.getcwd(), config['path'], 'SB_version_00_numpy_3_filters_pristine_SB00_augmented_y_3FILT.npy'))

        def normalization(t):
            mean1 = t[:,0].mean().item()
            mean2 = t[:,1].mean().item()
            mean3 = t[:,2].mean().item()

            std1 = t[:,0].std().item()
            std2 = t[:,1].std().item()
            std3 = t[:,2].std().item()

            return np.array([[mean1, mean2, mean3], [std1, std2, std3]])

        pristine = normalization(pristine_x)
        pr_trf = transform.Normalize(mean = pristine[0], std = pristine[1], inplace=True)

        for i in range(0, len(pristine_x)-1):
            pr_trf(pristine_x[i])

        config["network"]["params"]["class_num"] = 2

    train(config)

    config["out_file"].write("finish training! \n")
    config["out_file"].close()


