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
    raw_heatmap = torch.mean(weighted_activation, 0)
    heatmap = np.maximum(raw_heatmap.detach().cpu(), 0)
    heatmap /= torch.max(heatmap) + 1e-10
    return heatmap.numpy()

def superimpose(input_img, heatmap):
    img = cv2.cvtColor(input_img,cv2.COLOR_BGR2RGB)
    #heatmap = cv2.resize(heatmap, (img.shape[0], img.shape[1]))
    img = cv2.resize(img, (150,150))
    heatmap = cv2.resize(heatmap, (150,150))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_HOT)
    #superimposed_img = np.uint8(heatmap * 0.2 + img * 0.8)
    superimposed_img = np.uint8(heatmap * 1.0 + img * 0.0) #remove overplotting of the galaxy image
    pil_img = cv2.applyColorMap(superimposed_img, cv2.COLORMAP_HOT)
    #pil_img = cv2.cvtColor(superimposed_img,cv2.COLOR_BGR2RGB)
    return pil_img
    #return heatmap
    

def to_RGB(tensor):
    tensor = (tensor - tensor.min())
    tensor = tensor/(tensor.max() + 1e-10)
    image_binary = np.transpose(tensor.numpy(), (1, 2, 0))
    #print(image_binary)
    image = np.uint8(255 * image_binary)
    return image

def grad_cam(model, input_tensor, heatmap_layer, truelabel):
    info = InfoHolder(heatmap_layer)
    heatmap_layer.register_forward_hook(info.hook)
    
    features, logits = model(input_tensor.unsqueeze(0))
    #print("logits",logits)

    #truelabel = truelabel if truelabel else torch.argmax(output)
    #print("logits0",logits[0][truelabel])

    output = logits[0][truelabel].cpu()

    output.backward()

    weights = torch.mean(info.gradient, [0, 2, 3])
    activation = info.activation.squeeze(0)

    weighted_activation = torch.zeros(activation.shape)
    for idx, (weight, activation) in enumerate(zip(weights, activation)):
        weighted_activation[idx] = weight * activation

    heatmap = generate_heatmap(weighted_activation)
    input_image = to_RGB(input_tensor.cpu())

    #print(heatmap)
    #print(input_image)
    #print(np.shape(heatmap))
    #print(np.shape(input_image))
    #im = input_tensor.cpu()
    #np.save('input_image.npy', im)
    #np.save('heatmap.npy', heatmap)

    return superimpose(input_image, heatmap)
    #return heatmap

