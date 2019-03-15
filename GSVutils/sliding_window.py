# use this to make sliding window crops


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append( os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pytorch_pretrained') )

import GSVutils.utils

from GSVutils.point import Point as Point
from GSVutils.pano_feats import Pano as Pano
from GSVutils.pano_feats import Feat as Feat

import torchvision
from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from TwoFileFolder import TwoFileFolder
import resnet_extended 

GSV_IMAGE_HEIGHT = GSVutils.utils.GSV_IMAGE_HEIGHT
GSV_IMAGE_WIDTH  = GSVutils.utils.GSV_IMAGE_WIDTH

model_path = '/mnt/c/Users/gweld/sidewalk/sidewalk_ml/pytorch_pretrained/models/20e_slid_win_w_reats_r18.pt'

############ Load the Model ################
# we're gonna do this just once

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

model_ft = resnet_extended.extended_resnet18()
#num_ftrs = model_ft.fc.in_features
#model_ft.fc = nn.Linear(num_ftrs, 5)
model_ft = model_ft.to( device )

model_ft.load_state_dict( torch.load(model_path, map_location='cpu') )
model_ft.eval()


############################################


def get_batch_predictions(dir_containing_images):
    ''' use the TwoFileFolder dataloader to load images and feed them through the model'''
    dataset    = TwoFileFolder(dir_containing_images, data_transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=4)

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)


def sliding_window(pano, stride=100, bottom_space=1600, side_space=300, cor_thresh=70):
    ''' take in a pano and produce a set of feats, ready for writing to a file
        labels assigned if the crop is within cor_thresh of a true label
        
        try cor_thresh = stride/sqrt(2)
    '''

    x, y = side_space, 0
    while(y > - (GSV_IMAGE_HEIGHT/2 - bottom_space)):
        while(x < GSV_IMAGE_WIDTH - side_space):
            # do things in one row
            
            # check if there's any features near this x,y point
            p = Point(x,y)
            
            label = 8 # for null
            for feat in pano.all_feats():
                if p.dist( feat.point() ) <= cor_thresh:
                    if label == 8:
                        label = feat.label_type
                    else:
                        if label != feat.label_type:
                            #print "Found conflicting labels, skipping."
                            continue
            row = [pano.pano_id, x, y, label, pano.photog_heading, None,None,None]
            yield Feat(row)
            
            x += stride
        y -= stride # jump down a row
        x = side_space


def write_predictions_to_file(predictions, path):
    count = 0
    with open(path, 'w') as csvfile:
        writer = csv.writer(csvfile)

        for coords, prediction in predictions.iteritems():
            x,y = coords.split(',')
            row = [x,y] + prediction
            writer.writerow(row)
            count += 1
        print "\tWrote {} predictions to {}.".format(count, path)

def write_gt_to_file(gt, path):
    count = 0
    with open(path, 'w') as csvfile:
        writer = csv.writer(csvfile)

        for coords, label in gt.iteritems():
            x,y = coords.split(',')
            row = [x,y] + [label]
            writer.writerow(row)
            count += 1
        print "\tWrote {} predictions to {}.".format(count, path)


def read_predictions_from_file(path):
    predictions = defaultdict(list)

    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)

        for row in reader:
            x, y = row[0], row[1]
            prediction = map(float, row[2:])

            # let this work for processed predictions, as well
            if len(prediction) == 1: prediction = int(prediction[0])

            predictions["{},{}".format(x,y)] = prediction
    return predictions