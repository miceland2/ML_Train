import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]), 
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
}

data_dir = 'oxford_pets'
sets = ['train', 'test']

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'test']}
data = datasets.ImageFolder(data_dir)

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=0) for x in ['train', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes
class_names = np.array(class_names)

def get_hidden_features(x, layer):
    activation = {}

    def get_activation(name):
        def hook(m, i, o):
            activation[name] = o.detach()

        return hook

    model.register_forward_hook(get_activation(layer))
    _ = model(x)
    return activation[layer]

def train_model(model, mdoel2, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        for phase in ['train']:
            if (phase == 'train'):
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            cnt = 1
            for inputs, labels in dataloaders[phase]:
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    norm = nn.functional.normalize(outputs)
                    outputs2 = model2(norm)
                    
                    _, preds = torch.max(outputs2, 1)
                    loss = criterion(outputs2, labels)
                
                    if (phase == 'train'):
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        

                print(cnt * 4)
                cnt += 1
            
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            #if (phase == 'train'):
            #    scheduler.step()
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            #print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if (phase == 'test' and epoch_acc > best_acc):
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()
            
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
    #print(f'Best test Acc: {best_acc:4f}')
    print()
    
    model.load_state_dict(best_model_wts)
    return (model)

model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1, progress=True)
for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 37)

#model.fc_backup = nn.Linear(512, 37)
model.fc = nn.Sequential()
model.to(device)
model2 = nn.Linear(num_ftrs, 37)
model2.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model2.parameters(), lr=0.001)

step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

model = train_model(model, model2, criterion, optimizer, step_lr_scheduler, num_epochs=2)

test_loss = 0.0
class_correct = list(0 for i in range(len(class_names)))
class_total = list(0 for i in range(len(class_names)))

model.fc = model2;
                
for data, target in dataloaders['test']:
    data, target = data.to(device), target.to(device)
    with torch.no_grad(): # turn off autograd for faster testing
        output = model(data)
        loss = criterion(output, target)
    test_loss = loss.item() * data.size(0)
    _, pred = torch.max(output,  1)
    correct_tensor = pred.eq(target.data.view_as(pred))
    correct = np.squeeze(correct_tensor.cpu().numpy())
    for i in range(len(target)):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

test_loss = test_loss / dataset_sizes['test']
#print('Test Loss: {:.4f}'.format(test_loss))
for i in range(len(class_names)):
    if class_total[i] > 0:
        print("Test Accuracy of %5s: %2d%% (%2d/%2d)" % (
            class_names[i], 100*class_correct[i]/class_total[i], np.sum(class_correct[i]), np.sum(class_total[i])
        ))
    else:
        print("Test accuracy of %5s: NA" % (class_names[i]))
print("Test Accuracy of %2d%% (%2d/%2d)" % (100*np.sum(class_correct)/np.sum(class_total), np.sum(class_correct), np.sum(class_total)))