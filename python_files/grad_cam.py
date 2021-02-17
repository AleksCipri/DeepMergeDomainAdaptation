import cv2
import torch
import numpy as np

class InfoHolder():

    def __init__(self, heatmap_layer):
        self.gradient = None
        self.activation = None
        self.heatmap_layer = heatmap_layer

    def get_gradient(self, grad):
        self.gradient = grad

    def hook(self, model, input, output):
        output.register_hook(self.get_gradient)
        self.activation = output.detach()

def generate_heatmap(weighted_activation):
    '''
    Function that generates a heatmap from the weighted activations
    '''
    raw_heatmap = torch.mean(weighted_activation, 0)
    heatmap = np.maximum(raw_heatmap.detach().cpu(), 0)
    heatmap /= torch.max(heatmap) + 1e-10
    return heatmap.numpy()

def superimpose(input_img, heatmap):
    '''
    Function that superimposes Grad-CAMs on top of the image
    '''
    img = cv2.cvtColor(input_img,cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150,150))
    heatmap = cv2.resize(heatmap, (150,150))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS)
    superimposed_img = np.uint8(heatmap * 0.2 + img * 0.8)
    pil_img = cv2.applyColorMap(superimposed_img, cv2.COLORMAP_VIRIDIS)
    return pil_img
    

def to_RGB(tensor):
    '''
    Function to convert image to RGB in case we are interested in saving plots as images
    '''
    tensor = (tensor - tensor.min())
    tensor = tensor/(tensor.max() + 1e-10)
    image_binary = np.transpose(tensor.numpy(), (1, 2, 0))
    image = np.uint8(255 * image_binary)
    return image

def grad_cam(model, input_tensor, heatmap_layer, truelabel):
    '''
    Function that creates final Grad-CAM using the activations from a specified network layer.
    '''
    info = InfoHolder(heatmap_layer)
    heatmap_layer.register_forward_hook(info.hook)
    
    features, logits = model(input_tensor.unsqueeze(0))

    output = logits[0][truelabel].cpu()

    output.backward()

    weights = torch.mean(info.gradient, [0, 2, 3])
    activation = info.activation.squeeze(0)

    weighted_activation = torch.zeros(activation.shape)
    for idx, (weight, activation) in enumerate(zip(weights, activation)):
        weighted_activation[idx] = weight * activation

    heatmap = generate_heatmap(weighted_activation)
    input_image = to_RGB(input_tensor.cpu())

    #in case we want to output superimposed images and grad-cam we can return this instead
    #return superimpose(input_image, heatmap)

    #we want to return only gradcam because overplotting images makes everything less visible
    return heatmap
