import argparse
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torch.utils import data
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim


def create_tranform(image_size):
    training_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return training_transform, transform

def create_dataset(dir_path, transform):
    return ImageFolder(root=dir_path, transform=transform)

def create_data_loader(dataset, batch_size, shuffle):
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_input_args():
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser(description='Process train.')
    # Create 3 command line arguments as mentioned above using add_argument() from ArguementParser method
    parser.add_argument('data_dir')
    parser.add_argument('--save_dir', default='save_directory')
    parser.add_argument('--arch', default='vgg16')
    parser.add_argument('--learning_rate', default='0.01')
    parser.add_argument('--hidden_units', default='512')
    parser.add_argument('--epochs', default='20')
    parser.add_argument('--gpu')
    
    
    # Replace None with parser.parse_args() parsed argument collection that 
    # you created with this function 
    return parser.parse_args()

def main():
    in_arg = get_input_args()
    print(in_arg)
    training_transform, transform = create_tranform(256)
    training_data_transforms = training_transform
    validation_data_transforms = transform
    testing_data_transforms = transform
    train_dir = in_arg.data_dir + '/train'
    valid_dir = in_arg.data_dir + '/valid'
    test_dir = in_arg.data_dir + '/test'
    # TODO: Load the datasets with ImageFolder
    training_image_datasets = create_dataset(train_dir, training_data_transforms)
    validation_image_datasets = create_dataset(valid_dir, validation_data_transforms)
    testing_image_datasets = create_dataset(test_dir, testing_data_transforms)
    batch_size = 16
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    training_dataloaders = create_data_loader(training_image_datasets, batch_size,True)
    validation_dataloaders = create_data_loader(validation_image_datasets, batch_size,True)
    testing_dataloaders = create_data_loader(testing_image_datasets, batch_size,True)
    match in_arg.arch:
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
    for param in model.parameters():
        param.requires_grad = False 
    
    # Modify the last fully connected layer to match the number of classes in your dataset
    vgg_classifier_in_features = model.classifier[0].in_features
    num_classes = len(training_image_datasets.classes)
    classifier = nn.Sequential(OrderedDict([
                            ('fc1', nn.Linear(vgg_classifier_in_features, 512)),
                            ('relu', nn.ReLU()),
                            ('fc2', nn.Linear(512, num_classes)),
                            ('output', nn.LogSoftmax(dim=1))
                            ]))
    model.classifier = classifier

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = optim.Adam(model.classifier.parameters(), float(in_arg.learning_rate))
    if in_arg.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(f"device is {device}")   
    model = model.to(device)
    model.train()
    # Iterate over the training dataset
    epochs =  int(in_arg.epochs)
    steps = 0
    print_every = 5
    running_loss = 0.0

    for epoch in range(epochs):
        print(f"==========Epoch {epoch+1}==========")
        for images, labels in training_dataloaders:
            steps += 1
            images, labels = images.to(device), labels.to(device)
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                for images, labels in validation_dataloaders:
                    images, labels = images.to(device), labels.to(device)

                    logps = model.forward(images)
                    loss = criterion(logps, labels)

                    valid_loss += loss.item()

                    # Calculate our accuracy
                    ps = torch.exp(logps)
                    top_ps, top_class = ps.topk(1, dim=1)
                    equality = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equality.type(torch.FloatTensor)).item()
                    
                print(f"Train loss: {running_loss/print_every:.3f}.. "    
                    f"Valid loss: {valid_loss/len(validation_dataloaders):.3f}.. "
                    f"Valid accuracy: {accuracy/len(validation_dataloaders):.3f}")
                running_loss = 0
                model.train()
                
    # TODO: Do validation on the test set
    test_loss = 0 
    accuracy = 0 

    with torch.no_grad():
        for inputs, labels in testing_dataloaders:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
                        
            test_loss += batch_loss.item()
                        
            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            
    print(f"Test loss: {test_loss/len(testing_dataloaders):.3f}.. "
        f"Test accuracy: {accuracy/len(testing_dataloaders):.3f}")
    
    # TODO: Save the checkpoint 
    model.class_to_idx = training_image_datasets.class_to_idx
    checkpoint = {'classifier': model.classifier,
                'state_dict': model.state_dict(),
                'epochs': epochs,
                'optim_stat_dict': optimizer.state_dict(),
                'class_to_idx': training_image_datasets.class_to_idx,
                'hidden_units': in_arg.hidden_units 
                }

    torch.save(checkpoint,  os.path.join(in_arg.save_dir,'checkpoint.pth'))
    

# Call to main function to run the program
if __name__ == "__main__":
    main()