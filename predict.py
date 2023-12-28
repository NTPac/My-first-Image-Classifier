import argparse
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils import data
import torch.optim as optim
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json

from collections import OrderedDict 

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    match checkpoint['vgg_type']:
        case "vgg11":
            model = models.vgg11(pretrained=True)
        case "vgg13":
            model = models.vgg13(pretrained=True)
        case "vgg16":
            model = models.vgg16(pretrained=True)
        case "vgg19":
            model = models.vgg19(pretrained=True)
        case "vgg11":
            model = models.vgg11(pretrained=True)
        
    
    # freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model


def process_image(image: Image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    image.resize((256,256))
    distance = (256 - 224)/2
    image = image.crop((distance, distance, 256-distance, 256-distance))
    image = np.array(image)
    image =  image.astype(np.float64)
    np_image = image/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean)/std
    np_image = np_image.transpose((2 , 0, 1))
    tensor = torch.from_numpy(np_image)
    tensor = tensor.type(torch.FloatTensor)
    return tensor


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        _, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    plt.title(title)
    
    return ax

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    image = Image.open(image_path)
    image = process_image(image)
    image = image.unsqueeze(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device) 
    model.to(device); 
    
    ps = torch.exp(model(image))    
    top_p, top_class = ps.topk(topk, dim=1)
    
    # This post was very helpful 
    # https://discuss.pytorch.org/t/convert-to-numpy-cuda-variable/499/10
    top_p = top_p.cpu().detach().numpy()
    top_p = top_p.flatten()    
    top_p = top_p.tolist()
    
    top_class = top_class.cpu().detach().numpy()    
    top_class = top_class.flatten()
    top_class = top_class.tolist()
    
    return top_p, top_class


def get_input_args():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(description='Process predict.')
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('input')
    parser.add_argument('checkpoint')
    parser.add_argument('--top_k', default='3')
    parser.add_argument('--category_names', default='cat_to_name.json')
    parser.add_argument('---gpu', default='20')
    
    
    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return parser.parse_args()

def make_plot(classes, probs, class_to_idx, image_path, flower_key, cat_to_name ):
    title_image = cat_to_name[flower_key]
    image = Image.open(image_path)
    image = process_image(image)
    imshow(image, ax=None, title=title_image)            
    
    y_pos = np.arange(len(classes))    
    name_classes = []
    
    for i in classes:
        flower_key = None
        for k, v in class_to_idx.items():
            if v == i: 
                flower_key = k
        name_classes.append(cat_to_name[flower_key])
        
    _, axis = plt.subplots()
    axis.barh(y_pos, probs,  align='center', color='blue', ecolor='black')
    axis.set_yticks(y_pos)
    axis.set_yticklabels(name_classes)
    axis.invert_yaxis()
    plt.show()
    
def main():
    in_arg = get_input_args()
    model = load_checkpoint(in_arg.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device is {device}")   
    with open(in_arg.category_names, 'r') as f:
        cat_to_name = json.load(f)
    probs, classes = predict(in_arg.input, model)
    make_plot(classes, probs, model.class_to_idx, in_arg.input, in_arg.category_names, cat_to_name)

# Call to main function to run the program
if __name__ == "__main__":
    main()