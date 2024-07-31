import argparse
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

def parse_args():
    parser = argparse.ArgumentParser(description="Train a new network on a dataset")
    parser.add_argument('data_directory', type=str, default='flowers', help='Directory with the data')
    parser.add_argument('--arch', type=str, default='vgg16', help='Architecture to use')
    parser.add_argument('--learning_rate', type=float, default=0.003, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=1024, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=6, help='Number of epochs')
    parser.add_argument('--save_dir', type=str, default='checkpoint.pth', help='Directory to save checkpoint')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    return parser.parse_args()

def main():
	args = parse_args()

	# Load the data
	data_dir = args.data_directory
	train_dir = data_dir + '/train'
	valid_dir = data_dir + '/valid'
	test_dir = data_dir + '/test'

	# Define transforms for the training, validation, and testing sets
	data_transforms = {}
	data_transforms['train'] = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])]) 

	data_transforms['test'] = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

	# Load the datasets with ImageFolder
	image_datasets = {}
	image_datasets['train'] = datasets.ImageFolder(train_dir, transform=data_transforms['train'])
	image_datasets['valid'] = datasets.ImageFolder(valid_dir, transform=data_transforms['test'])
	image_datasets['test'] = datasets.ImageFolder(test_dir, transform=data_transforms['test'])

	# Using the image datasets and the trainforms, define the dataloaders
	dataloaders = {}
	dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True)
	dataloaders['valid'] = torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64)
	dataloaders['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size=64)

	# Use GPU if it's available
	device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

	# Load model
	if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    else:
        raise ValueError("Unsupported architecture")

    model.classifier = nn.Sequential(nn.Linear(25088, args.hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(args.hidden_units, 102),
                                 nn.LogSoftmax(dim=1))

	criterion = nn.NLLLoss()
	# Only train the classifier parameters, feature parameters are frozen
	optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    model.to(device)

    # Train and validate
    epochs = args.epochs
	steps = 0
	running_loss = 0
	print_every = 5
	for epoch in range(epochs):
	    for inputs, labels in dataloaders['train']:
	        steps += 1
	        # Move input and label tensors to the default device
	        inputs, labels = inputs.to(device), labels.to(device)
	        
	        optimizer.zero_grad()
	        
	        logps = model.forward(inputs)
	        loss = criterion(logps, labels)
	        loss.backward()
	        optimizer.step()

	        running_loss += loss.item()
	        
	        if steps % print_every == 0:
	            test_loss = 0
	            accuracy = 0
	            model.eval()
	            with torch.no_grad():
	                for inputs, labels in dataloaders['valid']:
	                    inputs, labels = inputs.to(device), labels.to(device)
	                    logps = model.forward(inputs)
	                    batch_loss = criterion(logps, labels)
	                    
	                    test_loss += batch_loss.item()
	                    
	                    # Calculate accuracy
	                    ps = torch.exp(logps)
	                    top_p, top_class = ps.topk(1, dim=1)
	                    equals = top_class == labels.view(*top_class.shape)
	                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
	                    
	            print(f"Epoch {epoch+1}/{epochs}.. "
	                  f"Train loss: {running_loss/print_every:.3f}.. "
	                  f"Valid loss: {test_loss/len(dataloaders['valid']):.3f}.. "
	                  f"Valid accuracy: {accuracy/len(dataloaders['valid']):.3f}")
	            running_loss = 0
	            model.train

	# Save the checkpoint
	model.class_to_idx = image_datasets['train'].class_to_idx

	checkpoint = {
	    'model_state_dict': model.state_dict(),
	    'optimizer_state_dict': optimizer.state_dict(),
	    'epoch': epoch,
	    'class_to_idx': model.class_to_idx
	}

	torch.save(checkpoint, 'checkpoint.pth')             

if __name__ == '__main__':
    main()