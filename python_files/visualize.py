#optional plotting methods

def plot_grad_flow(path, epoch, named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''

    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    import os

    cwd = os.getcwd()

    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("Average Gradient")
    plt.title("Gradient Flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    
    os.chdir(path)
    plt.savefig('epoch '+str(epoch)+'.png')
    os.chdir(cwd)
    plt.clf()

def plot_learning_rate_scan(lr_rates, total_loss, epoch, path):
    '''Plot learning rate scan to find ideal learning rate for one-cycle learning.'''

    import matplotlib.pyplot as plt
    import numpy as np
    import os

    cwd = os.getcwd()

    plt.plot(np.array(lr_rates), np.array(total_loss))
    plt.semilogx()
    # plt.semilogy()
    plt.grid(True)
    plt.xlabel("Learning Rate")
    plt.ylabel("Total Loss")
    plt.title("Learning Rate Scan")

    os.chdir(path)
    plt.savefig('epoch '+str(epoch)+'.png')
    os.chdir(cwd)
    plt.clf()


