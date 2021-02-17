import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import hyper_train_mmd
from hyper_train_mmd import train
import network
import loss
import lr_schedule
import torch
import torch.optim as optim
from import_and_normalize import array_to_tensor, update
import os
import numpy as np
import os.path as osp

def run(point):
    
    #fix seed
    np.random.seed(1)
    
    config = {}
    config["net"] = "DeepMerge"
    config["epochs"] = 30
    config["optim_choice"] = 'Adam'
    config["cycle_length"] = point["cycle_length"]
    config["early_stop_patience"] = 15
    config["weight_decay"] = point["weight_decay"]
    config["lr"] = point["lr"]
    config["fisher_or_no"] = "no"
    config["transfer_type"] = "mmd"
    config["ckpt_path"] = "/lus/theta-fs0/projects/OptADDN/Alex/DeepDA_SDSS_Alex/DeepDA_SDSS_Alex/DeepDA_MMD_TL/"
    
    loss_dict = {"tr": loss.FisherTR, "td ": loss.FisherTD}
    optim_dict = {"SGD": optim.SGD, "Adam": optim.Adam}
    transfer_loss_dict = {"coral":loss.CORAL, "mmd":loss.mmd_distance}
    fisher_loss_dict = {"tr": loss.FisherTR, "td": loss.FisherTD}

    config["loss"] = {"name": transfer_loss_dict[config["transfer_type"]], 
                      "ly_type": "cosine", 
                      "fisher_loss_type": "tr",
                      "discriminant_loss": fisher_loss_dict["tr"],
                      "trade_off": point["trade_off"], "update_iter":200,
                      "intra_loss_coef": .01, "inter_loss_coef": .01, "inter_type": "global", 
                      "em_loss_coef": .01}
    
    if "DeepMerge" in config["net"]:
        config["network"] = {"name":network.DeepMerge, \
            "params":{"class_num":2, "new_cls":True, "use_bottleneck":False, "bottleneck_dim":32*9*9} }
    elif "ResNet" in config["net"]:
        config["network"] = {"name":network.ResNetFc, \
            "params":{"resnet_name":config["net"], "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }

    if config["optim_choice"] == "Adam":
        config["optimizer"] = {"type":"Adam", "optim_params":{"lr":1e-6, "betas":(0.7,0.8), "weight_decay": config["weight_decay"], "amsgrad":True, "eps":1e-8}, \
                        "lr_type":"inv", "lr_param":{"init_lr":0.0001, "gamma":0.001, "power":0.75} }
    else:
        config["optimizer"] = {"type":"SGD", "optim_params":{"lr":1.0, "momentum":0.9, \
                               "weight_decay": config["weight_decay"], "nesterov":True}, "lr_type":"inv", \
                               "lr_param":{"init_lr":0.001, "gamma":0.001, "power":0.75} }

    if point["lr"] is not None:
        config["optimizer"]["optim_params"]["lr"] = config["lr"]
        config["optimizer"]["lr_param"]["init_lr"] = config["lr"]
        config["frozen lr"] = config["lr"]

    config["optimizer"]["lr_type"] = "one-cycle"
        
    config["dataset"] = 'galaxy'
    config["path"] = '/lus/theta-fs0/projects/OptADDN/Alex/DeepHyper_Alex/data_2/'


    if config["dataset"] == 'galaxy':
        pristine_x = array_to_tensor(osp.join(config['path'], 'Xdata_Illustris.npy'))
        pristine_y = array_to_tensor(osp.join(config['path'], 'ydata_Illustris.npy'))

        noisy_x = array_to_tensor(osp.join(config['path'], 'SDSS_X_postmergers.npy'))
        noisy_y = array_to_tensor(osp.join(config['path'], 'SDSS_y_postmergers.npy'))

        update(pristine_x, noisy_x)

        config["network"]["params"]["class_num"] = 2

    result = train(config, (pristine_x,pristine_y,noisy_x,noisy_y))

    return result

if __name__ == "__main__":
    point = {"lr":1e-4, "trade_off":.01, "cycle_length":2, "weight_decay": 1e-4}
    
    objective = run(point)
    print("objective: ", objective)
