from hyper_train_mmd import train
import network
import loss
import lr_schedule
import torch.optim as optim
from import_and_normalize import array_to_tensor, update
import os
import os.path as osp


def run(point):
    config = {}
    config["net"] = "ResNet18"
    config["epochs"] = 30
    config["optim_choice"] = 'Adam'
    config["cycle_length"] = point["cycle_length"]
    config["early_stop_patience"] = 20
    config["weight_decay"] = point["weight_decay"]
    config["net"] = "ResNet18"
    config["lr"] = point["lr"]
    config["fisher_or_no"] = "no"
    config["transfer_type"] = point["transfer_type"]

    loss_dict = {"tr": loss.FisherTR, "td ": loss.FisherTD}
    optim_dict = {"SGD": optim.SGD, "Adam": optim.Adam}
    transfer_loss_dict = {"coral":loss.CORAL, "mmd":loss.mmd_distance}
    fisher_loss_dict = {"tr": loss.FisherTR, "td": loss.FisherTD}

    config["loss"] = {"name": transfer_loss_dict[config["transfer_type"]], 
                      "ly_type": "cosine", 
                      "fisher_loss_type": "tr",
                      "discriminant_loss": fisher_loss_dict["tr"],
                      "trade_off": point["trade_off"], "update_iter":200,
                      "intra_loss_coef": point["intra_loss_coef"], "inter_loss_coef": point["inter_loss_coef"], "inter_type": "global", 
                      "em_loss_coef": point["em_loss_coef"]}
    
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
    #config["path"] = './small_20percent/'
    config["path"] = 'arrays/small dataset/'


    if config["dataset"] == 'galaxy':
        pristine_x = array_to_tensor(osp.join(config['path'], 'Pristine_small_20percent.npy'))
        pristine_y = array_to_tensor(osp.join(config['path'], 'Pristine_small_labels_20percent.npy'))

        noisy_x = array_to_tensor(osp.join(config['path'], 'Noisy_small_20percent.npy'))
        noisy_y = array_to_tensor(osp.join(config['path'], 'Noisy_small_labels_20percent.npy'))

        update(pristine_x, noisy_x)

        config["network"]["params"]["class_num"] = 2

    train(config, (pristine_x,pristine_y,noisy_x,noisy_y))

if __name__ == "__main__":
    point = {"transfer_type": "mmd", "lr":1e-4, "trade_off":.01, "intra_loss_coef":.01, "inter_loss_coef":.01, "em_loss_coef":.01, "cycle_length":2, "weight_decay": 1e-4}
    
    objective = run(point)
    print("objective: ", objective)