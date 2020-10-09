from hyper_train_ada import train
import os
import loss
import torch.optim as optim
import network
from import_and_normalize import array_to_tensor, update
import os.path as osp

def run(point):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    config = {}
    config["high"] = 1.0
    config["epochs"] = 100
    config["optim_choice"] = 'Adam'
    config["fisher_or_no"] = 'Fisher'
    config["cycle_length"] = point["cycle_length"]
    config["early_stop_patience"] = 20
    config["weight_decay"] = point["weight_decay"]
    config["ad_net_mult_lr"] = point["ad_net_mult_lr"]
    config["net"] = "DeepMerge"
    config["lr"] = point["lr"]
    #config["beta_1"] = point["beta_1"]
    #config["beta_2"] = point["beta_2"] 

    loss_dict = {"tr": loss.FisherTR, "td ": loss.FisherTD}
    optim_dict = {"SGD": optim.SGD, "Adam": optim.Adam}

    config["loss"] = {"loss_name": "tr",
                      "loss_type": loss_dict["tr"],
                      "ly_type": "cosine", 
                      "trade_off": point["trade_off"], "update_iter":200,
                      "intra_loss_coef": point["intra_loss_coef"], "inter_loss_coef": point["inter_loss_coef"], "inter_type": "global", 
                      "em_loss_coef": point["em_loss_coef"]}

    if "DeepMerge" in config["net"]:
        config["network"] = {"name":network.DeepMerge, \
            "params":{"class_num":2, "new_cls":True, "use_bottleneck":False, "bottleneck_dim":32*9*9} }
    elif "ResNet" in config["net"]:
        config["network"] = {"name":network.ResNetFc, \
            "params":{"resnet_name":config["net"], "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True} }
    
    #set optimizer
    if config["optim_choice"] == 'Adam':
        config["optimizer"] = {"type":"Adam", "optim_params":{"lr":0.001, "betas":(0.7,0.9), "weight_decay": config["weight_decay"], \
                                "amsgrad":False, "eps":1e-8}, \
                        "lr_type":"inv", "lr_param":{"init_lr":0.001, "gamma":0.001, "power":0.75} }
    else:
        config["optimizer"] = {"type":"SGD", "optim_params":{"lr":0.001, "momentum":0.9, \
                               "weight_decay": config["weight_decay"], "nesterov":True}, "lr_type":"inv", \
                               "lr_param":{"init_lr":0.005, "gamma":0.001, "power":0.75} }

    if config["lr"] is not None:
        config["optimizer"]["optim_params"]["lr"] = config["lr"]
        config["optimizer"]["lr_param"]["init_lr"] = config["lr"]
        config["frozen lr"] = config["lr"]

    #if all([config["beta_1"], config["beta_2"]]):
    #    config["optimizer"]["optim_params"]["betas"] = (config["beta_1"], config["beta_2"])

    config["optimizer"]["lr_type"] = "one-cycle"
    
    #or however the path will look
    config["dataset"] = 'galaxy'
    config["path"] = './small_20percent/' #'DeepMerge/small_20percent/'

    #TODO: change paths to get to dataset
    if config["dataset"] == 'galaxy':
        pristine_x = array_to_tensor(osp.join(config['path'], 'Pristine_small_20percent.npy'))
        pristine_y = array_to_tensor(osp.join(config['path'], 'Pristine_small_labels_20percent.npy'))

        noisy_x = array_to_tensor(osp.join(config['path'], 'Noisy_small_20percent.npy'))
        noisy_y = array_to_tensor(osp.join(config['path'], 'Noisy_small_labels_20percent.npy'))

        update(pristine_x, noisy_x)

        config["network"]["params"]["class_num"] = 2

    objective = train(config,(pristine_x,pristine_y,noisy_x,noisy_y))

    return objective

if __name__ == "__main__":
    point = {"lr":1e-4, "trade_off":.01, "intra_loss_coef":.01, "inter_loss_coef":.01, "em_loss_coef":.01, "cycle_length":2, \
    "weight_decay": 1e-4, "ad_net_mult_lr": .01, "beta_1": .7, "beta_2": .8}
    
    objective = run(point)
    print("objective: ", objective)