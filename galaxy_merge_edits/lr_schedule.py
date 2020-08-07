import numpy as np

def inv_lr_scheduler(param_lr, optimizer, iter_num, epoch_length, extra_lr, extra_cycle, gamma, power, init_lr=0.001, weight_decay=5e-4):
	"""Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
	lr = init_lr * (1 + gamma * iter_num) ** (-power)
	i=0
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr * param_group['lr_mult']
		#param_group['weight_decay'] = 1 * param_group['decay_mult']
		i+=1

	return optimizer

def cycle1(param_lr, optimizer, iter_num, epoch_length, lr, cycle_length, **kwargs):
	min_lr = lr/10
	max_lr = lr
	num_steps = cycle_length*epoch_length
	annihilation_frac = .1
	reduce_factor = .01
	num_cycle_steps = int(num_steps * (1. - annihilation_frac)) 

	optim_dict = optimizer.state_dict()
	
	#print(optim_dict)

	if 'momentum' in optim_dict['param_groups'][0]:
		momentum = optim_dict['param_groups'][0]['momentum']
		min_momentum = momentum
		max_momentum = 10*momentum
	else:
		momentum = None

	on_step = (iter_num/num_cycle_steps - (iter_num//num_cycle_steps))*num_cycle_steps

	#scale up
	if on_step <= num_cycle_steps//2:
		scale = on_step / (num_cycle_steps // 2)
		lr = min_lr + (max_lr - min_lr) * scale

		if momentum is not None:
			momentum = max_momentum - (max_momentum - min_momentum) * scale

	#scale down
	elif on_step <= num_cycle_steps:
		scale = (on_step - num_cycle_steps // 2) / (num_cycle_steps - num_cycle_steps // 2)
		lr = max_lr - (max_lr - min_lr) * scale

		if momentum is not None:
			momentum = min_momentum + (max_momentum - min_momentum) * scale

	#annihilation step only changes the learning rate
	elif on_step <= num_steps:
		scale = (on_step - num_cycle_steps) / (num_steps - num_cycle_steps)
		lr = min_lr - (min_lr - final_lr) * scale
		momentum = 0

	for param_group in optimizer.param_groups:
		factor = param_group['lr_mult']
		param_group['lr'] = lr*factor

		if momentum is not None:
			param_group['momentum'] = momentum

	return optimizer


def linear_to_find_baseline(param_lr, optimizer, iter_num, epoch_length, lr, extra_cycle, **kwargs):

	# min_lr = lr/1000
	# max_lr = 1000*lr
	num_epochs = 5
	# lr_spectrum = np.arange(min_lr, num_epochs, max_lr)
	lr_spectrum = np.linspace(1e-6, 1e-1, 85*num_epochs+1)
	num_steps = epoch_length

	#on_step = iter_num//epoch_length

	lr = lr_spectrum[iter_num]

	for param_group in optimizer.param_groups:
		param_group['lr'] = lr

	return optimizer

schedule_dict = {"inv":inv_lr_scheduler, "one-cycle": cycle1, "linear": linear_to_find_baseline}
