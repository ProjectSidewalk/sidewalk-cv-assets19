import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
#import matplotlib.pyplot as plt
import time
import os
import copy
from collections import defaultdict

from TwoFileFolder import TwoFileFolder
from resnet_extended2 import extended_resnet18

##### IMPORTANT: READ BEFORE STARTING A RUN ######
# Checklists:
# Correct Model? eg right resnet_extended
# Correct Dataloader for model?
# Correct Dataset Source?
# Correct Number of Epochs?
# Correct Model Save-Path?
#
#
##################################################

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


#data_dir = '/home/gweld/sliding_window_dataset/'
data_dir  = '/home/gweld/centered_crops_subset_with_meta'

print("Building datasets...")

# image_datasets = {x:TwoFileFolder(os.path.join(data_dir, x), data_transforms[x])
#                   for x in ['train', 'test']}
image_datasets = {x:datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'test']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'test']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


print("Finished loading data. Discovered {} classes:".format(len(class_names)))
print(", ".join(class_names))
print("")




def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            if phase == 'test':
                class_corrects  = defaultdict(int)
                class_totals    = defaultdict(int)
                class_predicted = defaultdict(int)

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if phase == 'test':
                    for index, pred in enumerate(preds):
                        actual = labels.data[index]
                        class_name = class_names[actual]
                        
                        if actual == pred: class_corrects[class_name] += 1
                        class_totals[class_name] += 1
                        class_predicted[ class_names[pred] ] += 1

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'test':
                print("Class Precision and Recall on Test Data")

                for class_name in class_totals:
                    correct   = float(class_corrects[class_name])
                    predicted = float(class_predicted[class_name])
                    actual    = float(class_totals[class_name])

                    p = 100 * (correct / predicted) if predicted >0 else float('nan')
                    r = 100 * (correct / actual)

                    print("{:20}{:06.2f}% {:06.2f}%".format(class_name, p, r))
                print("\n")

            # deep copy the model
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        #print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best test Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model




model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_names))


#model_ft  = extended_resnet18(True, num_classes=len(class_names), len_ex_feats=7)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


# Train and evaluate
# ^^^^^^^^^^^^^^^^^^

print('Beginning Training on {} train and {} test images.'.format(dataset_sizes['train'], dataset_sizes['test']))


model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
        num_epochs=20)




torch.save(model_ft.state_dict(), '20ep_new_old_r18.pt')





