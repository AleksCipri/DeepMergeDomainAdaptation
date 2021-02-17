'''
Run to create onnx file of your network. Open onnx file with network_graphs.ipynb. Example script to launch training: 
!python create_netgraph.py --gpu_id 0 \
                              --net DeepMerge \
                              --dset 'galaxy' \
                              --dset_path 'arrays/SDSS_Illustris_z0/' \
                              --output_dir 'output_DeepMerge_SDSS/noDA' \
                              --source_x_file Illustris_Xdata_05_augmented_combined_rotzoom_SMALL_3000_3000.npy \
                              --source_y_file Illustris_ydata_05_augmented_combined_rotzoom_SMALL_3000_3000.npy \
                              --target_x_file SDSS_x_data_postmergers_and_nonmergers.npy \
                              --target_y_file SDSS_y_data_postmergers_and_nonmergers.npy \
'''

import network
import torch
import netron
import argparse
import os
import numpy as np
import os.path as osp
import torchvision.transforms as transform
from torch.utils.data import Dataset, TensorDataset, DataLoader
from torch.autograd import Variable
from import_and_normalize import array_to_tensor, update

def show_me_the_graphs(config):

	if config["network"] == "DeepMerge":
		config["network"] = {"name":network.DeepMerge, "params":{"class_num":2, "new_cls":True, "use_bottleneck":False, "bottleneck_dim":32*9*9}}
	elif config["network"] == "Res18":
	    config["network"] = {"name":network.ResNetFc, "params":{"class_num":2, "resnet_name": "ResNet18", "use_bottleneck":True, "bottleneck_dim":256, "new_cls":True}}

	use_gpu = torch.cuda.is_available()
	net_config = config["network"]
	base_network = net_config["name"](**net_config["params"])
	ad_net = network.AdversarialNetwork(base_network.output_num())
	#gradient_reverse_layer = network.AdversarialLayer(high_value = config["high"])

	if use_gpu:
		base_network = base_network.cuda()
		ad_net = ad_net.cuda()

	## prepare data
	dsets = {}
	dset_loaders = {}

	pristine_indices = torch.randperm(len(pristine_x))
	pristine_x_train = pristine_x[pristine_indices[:int(np.floor(.7*len(pristine_x)))]]
	pristine_y_train = pristine_y[pristine_indices[:int(np.floor(.7*len(pristine_x)))]]

	noisy_indices = torch.randperm(len(noisy_x))
	noisy_x_train = noisy_x[noisy_indices[:int(np.floor(.7*len(noisy_x)))]]
	noisy_y_train = noisy_y[noisy_indices[:int(np.floor(.7*len(noisy_x)))]]

	dsets["source"] = TensorDataset(pristine_x_train, pristine_y_train)
	dsets["target"] = TensorDataset(noisy_x_train, noisy_y_train)

	dset_loaders["source"] = DataLoader(dsets["source"], batch_size =128, shuffle = True, num_workers = 1)
	dset_loaders["target"] = DataLoader(dsets["target"], batch_size = 128, shuffle = True, num_workers = 1)

	#give a dummy batch, except wait, features are important for ad_net
	inputs_source, labels_source = iter(dset_loaders["source"]).next()
	inputs_target, labels_target = iter(dset_loaders["target"]).next()

	# source_batch = torch.from_numpy(np.array(source_batch, dtype='int32'))
	# target_batch = torch.from_numpy(np.array(target_batch, dtype='int32'))

	if use_gpu:
		source_batch = Variable(inputs_source).cuda()
		target_batch = Variable(inputs_target).cuda()
	else:
		source_batch = Variable(inputs_source)
		target_batch = Variable(inputs_target)

	inputs = torch.cat((source_batch, target_batch), dim=0)
	weight_ad = torch.ones(inputs.size(0))

	features, base_logits = base_network(inputs)
	yhat_ad = ad_net(features.detach())

	input_names_base = ['Galaxy Array']
	output_names_base = ['Merger or Not']

	input_names_adnet = ['Base Network Features']
	output_names_adnet = ['Source or Target Domain']

	torch.onnx.export(base_network, inputs, 'base.onnx', input_names=input_names_base, output_names=output_names_base)
	torch.onnx.export(ad_net, features, 'ad_net.onnx', input_names=input_names_adnet, output_names=output_names_adnet)

if __name__ == '__main__':
	parser = argparse.ArgumentParser('Looking at graphs')
	parser.add_argument('--net', type=str, default='Res18', help='Res18 or DeepMerge')
	parser.add_argument('--dset', type=str, default='galaxy', help="The dataset or source dataset used")
	parser.add_argument('--dset_path', type=str, default=None, help="The dataset directory path")
	parser.add_argument('--source_x_file', type=str, default='SB_version_00_numpy_3_filters_pristine_SB00_augmented_3FILT.npy', help="Source domain x-values filename")
	parser.add_argument('--target_x_file', type=str, default='SB_version_00_numpy_3_filters_noisy_SB25_augmented_3FILT.npy', help="Target domain x-values filename")
	parser.add_argument('--source_y_file', type=str, default='SB_version_00_numpy_3_filters_pristine_SB00_augmented_y_3FILT.npy', help="Source domain y-values filename")
	parser.add_argument('--target_y_file', type=str, default='SB_version_00_numpy_3_filters_noisy_SB25_augmented_y_3FILT.npy', help="Target domain y-values filename")

	args = parser.parse_args()

	config = {}
	config["network"] = args.net
	config["dataset"] = args.dset
	config["path"] = args.dset_path

	if config["dataset"] == 'galaxy':
		pristine_x = array_to_tensor(osp.join(config['path'], args.source_x_file))
		pristine_y = array_to_tensor(osp.join(config['path'], args.source_y_file))

		noisy_x = array_to_tensor(osp.join(config['path'], args.target_x_file))
		noisy_y = array_to_tensor(osp.join(config['path'], args.target_y_file))

		update(pristine_x, noisy_x)

	show_me_the_graphs(config)
