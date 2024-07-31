import argparse
import json
import torch
from torchvision import models
from PIL import Image
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Predict flower name from an image along with the probability of that name")
    parser.add_argument('input', type=str, help='Path to the input image')
    parser.add_argument('checkpoint', type=str, default='checkpoint.pth', help='Path to the model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to a JSON file mapping categories to real names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    return parser.parse_args()

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    if pil_image.size[0] < pil_image.size[1]:
        pil_image.thumbnail((256, pil_image.size[1]))
    else:
        pil_image.thumbnail((pil_image.size[0], 256))

    width, height = pil_image.size
    new_width, new_height = 224, 224
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    pil_image = pil_image.crop((left, top, right, bottom))

    np_image = np.array(pil_image) / 255.0

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))
    tensor_image = torch.tensor(np_image, dtype=torch.float)
    return tensor_image

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    epoch = checkpoint['epoch']
    return model, optimizer, epoch

def predict(image_path, model, topk=5, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Implement the code to predict the class from an image file
    model.eval()
    with torch.no_grad():
        image = process_image(image_path)
        image = image.unsqueeze(0).to(device)

        # Get the class probabilities
        output = model(image)
        probabilities = torch.exp(output)
        
        # Get the topk probabilities and indices
        top_probs, top_indices = probabilities.topk(topk)
        top_probs = top_probs.cpu().numpy().flatten()
        top_indices = top_indices.cpu().numpy().flatten()

        # Convert indices to classes
        idx_to_class = {v: k for k, v in model.class_to_idx.items()}
        top_classes = [idx_to_class[idx] for idx in top_indices]

    return top_probs, top_classes

def main():
    args = parse_args()

    # Loading the checkpoint
    model, optimizer, epoch = load_checkpoint(args.checkpoint)

    # Use GPU if it's available
	device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
	model.to(device)

	probs, classes = predict(args.input, model, args.top_k, device)

	if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
        classes = [cat_to_name[str(c)] for c in classes]

    print(probs)
	print(classes)  

if __name__ == '__main__':
    main()