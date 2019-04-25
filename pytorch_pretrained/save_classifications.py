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
import csv
from collections import defaultdict

from TwoFileFolder import TwoFileFolder
from resnet_extended2 import extended_resnet18 #extended_resnet34, extended_resnet50

################ IMPORTANT: READ BEFORE STARTING A RUN ################
# Checklists:
# Correct Model? eg right resnet_extended
# Correct Extra Features? eg right TwoFileFolder meta_to_tensor_version
# Correct Dataset class for model?
# Correct Dataset Source?
# Correct Number of Epochs?
# Correct Model Save-Path?
#
#######################################################################


#data_dir = '/mnt/c/Users/gweld/sidewalk/sidewalk_ml/mini_ds/'
#data_dir = '/home/gweld/sliding_window_dataset/'
data_dir  = '/home/gweld/centered_crops_subset_with_meta'


model_basename  = '20ep_new_old_re18_2'
model_to_load ='models/{}.pt'.format(model_basename)
ouput_path = '{}.csv'.format(model_basename)

downsample = None





data_transform = transforms.Compose([
                 transforms.Resize(256),
                 transforms.CenterCrop(224),
                 transforms.ToTensor(),
                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

print("Building datasets...")

image_dataset = TwoFileFolder(os.path.join(data_dir, 'test'), meta_to_tensor_version=1, transform=data_transform, downsample=downsample)
#image_datasets = {x:datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
#                  for x in ['train', 'test']}

dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=4, shuffle=True, num_workers=4)

len_ex_feats = image_dataset.len_ex_feats

dataset_size = len(image_dataset)
class_names = image_dataset.classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Finished loading data. Discovered {} classes:".format(len(class_names)))
print(", ".join(class_names))
print("")
print("Using dataloader that supplies {} extra features.".format(len_ex_feats))
print("")

# build model
model_ft = extended_resnet18(False, num_classes=len(class_names), len_ex_feats=len_ex_feats)
try:
    model_ft.load_state_dict(torch.load(model_to_load))
except RuntimeError as e:
    model_ft.load_state_dict(torch.load(model_to_load, map_location='cpu'))

model_ft = model_ft.to(device)
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)


def get_predictions(model, optimizer):
    paths_out = []
    true_out  = []
    pred_out  = []


    model.eval()

    for inputs, labels, paths in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            paths_out += list(paths)
            true_out += list(labels.tolist())
            pred_out += list(preds.tolist())

    return paths_out, true_out, pred_out

print('Beginning computing labels on {} images...'.format(dataset_size))
paths, true, predicted = get_predictions(model_ft, optimizer_ft)
print('...finished!')

# write results to file
counter  = 0
with open(ouput_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['File', 'Actual', 'Predicted'])
    for path, true, predicted in zip(paths, true, predicted):
        pathname, filename = os.path.split(path)

        _, cont_dir = os.path.split(pathname)
        shortpath = os.path.join(cont_dir, filename)

        #print "{:<70} {:<20} {:<20}".format(shortpath, class_names[true], class_names[predicted])
        writer.writerow((shortpath, class_names[true], class_names[predicted]))
        counter += 1
print("Wrote {} rows to {}".format(counter, ouput_path))
