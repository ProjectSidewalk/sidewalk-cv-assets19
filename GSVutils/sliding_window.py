# use this to make sliding window crops

from collections import defaultdict
import sys
import os
import csv
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append( os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pytorch_pretrained') )

import GSVutils.utils

from GSVutils.point import Point as Point
from GSVutils.pano_feats import Pano as Pano
from GSVutils.pano_feats import Feat as Feat
from GSVutils.clustering import non_max_sup
from GSVutils.precision_recall import precision_recall, partition_based_on_correctness

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

path_to_gsv_scrapes  = "/mnt/f/scrapes_dump/"
pano_db_export = '/mnt/c/Users/gweld/sidewalk/minus_onboard.csv'

model_path = '/mnt/c/Users/gweld/sidewalk/sidewalk_ml/pytorch_pretrained/models/20e_slid_win_w_feats_r18.pt'

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


def predict_from_crops(dir_containing_crops, verbose=False):
    ''' use the TwoFileFolder dataloader to load images and feed them
        through the model
        returns a dict mapping pano_ids to dicts of {coord: prediction lists}
    '''
    predictions = defaultdict(dict)

    print "Building dataset..."

    dataset    = TwoFileFolder(dir_containing_crops, data_transform)

    for img_path, meta_path, _ in dataset.samples:
        _, img_name = os.path.split(img_path)
        img_name, _ = os.path.splitext(img_name)
        pano_id, coords = img_name.split('crop')

        if verbose:
            print "Getting predictions for pano {} at {}".format( pano_id, coords )

        both = dataset.load_img_and_meta(img_path, meta_path)
        both = both.view((1, both.size(0)))
        with torch.no_grad():
            prediction = model_ft( both )
        prediction = prediction.flatten().tolist()

        if verbose:
            print '\t'+str(prediction)

        predictions[pano_id][coords] = prediction

    return predictions


def get_and_write_batch_ground_truth(dir_containing_crops):
    ground_truths = {}
    for pano_id in os.listdir(dir_containing_crops):
        print "Getting ground truth for pano {}".format(pano_id)
        pano_xml_path   = os.path.join(path_to_gsv_scrapes, pano_id[:2], pano_id + ".xml")
        true_pano_yaw_deg = GSVutils.utils.extract_panoyawdeg(pano_xml_path)

        gt = get_ground_truth(pano_id, true_pano_yaw_deg)

        print "\tFound {} ground truth labels.".format(len(gt))
        ground_truths[pano_id] = gt

    write_batch_predictions_to_file(ground_truths,dir_containing_crops, 'ground_truth.csv')


def write_predictions_to_file(predictions, path):
    count = 0
    with open(path, 'w') as csvfile:
        writer = csv.writer(csvfile)

        for coords, prediction in predictions.iteritems():
            if type(prediction) != list:
                # this way we can write int labels too
                prediction = [prediction]

            x,y = coords.split(',')
            row = [x,y] + prediction
            writer.writerow(row)
            count += 1
        print "\tWrote {} predictions to {}.".format(count, path)


def write_batch_predictions_to_file(batch_predictions, root_path, pred_file_name):
    for pano_id, predictions in batch_predictions.items():
        path = os.path.join(root_path, pano_id, pred_file_name)

        write_predictions_to_file(predictions, path)


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


def convert_to_real_coords(sv_x, sv_y, pano_yaw_deg):
    x = ((float(pano_yaw_deg) / 360) * GSV_IMAGE_WIDTH + sv_x) % GSV_IMAGE_WIDTH
    y = GSV_IMAGE_HEIGHT / 2 - sv_y

    return int(x), int(y)


def convert_to_sv_coords(x, y, pano_yaw_deg):
    sv_x = x - (float(pano_yaw_deg)/360 * GSV_IMAGE_WIDTH)
    sv_y = GSV_IMAGE_HEIGHT / 2 - y

    if sv_x < 0: sv_x += GSV_IMAGE_WIDTH 

    return int(sv_x), int(sv_y)


def get_ground_truth(pano_id, true_pano_yaw_deg):
    ''' returns dict of str coords mapped to int label '''
    labels = {}
    with open(pano_db_export, 'r') as csvfile:
        reader = csv.reader(csvfile)

        for row in reader:
            if row[0] != pano_id: continue

            x, y = int(row[1]), int(row[2])
            label = int(row[3])-1 # compensate for 1-indexing
            photog_heading = float(row[4])

            pano_yaw_deg = 180 - photog_heading

            x, y = convert_to_real_coords(x, y, pano_yaw_deg)
            x, y = convert_to_sv_coords(x, y, true_pano_yaw_deg)

            # ignore other labels 
            if label not in range(4): continue

            labels["{},{}".format(x,y)] = label
    return labels


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


def annotate(img, pano_yaw_deg, coords, label, color, show_coords=True, box=None):
    """ takes in an image object and labels it at specified coords
        translates streetview coords to pixel coords
        if given a box, marks that box around the label
    """
    sv_x, sv_y = coords
    x = ((float(pano_yaw_deg) / 360) * gsv_image_width + sv_x) % gsv_image_width
    y = gsv_image_height / 2 - sv_y

    if show_coords: label = "{},{} {}".format(sv_x, sv_y, label)

    # radius for dot
    r = 20
    draw = ImageDraw.Draw(img)
    draw.ellipse((x - r, y - r, x + r, y + r), fill=color)
    if box is not None:
        half_box = box/2
        p1 = (x-half_box, y-half_box)
        p2 = (x+half_box, y+half_box)
        draw.rectangle([p1,p2], outline=color)

    font  = ImageFont.truetype("roboto.ttf", 60, encoding="unic")
    draw.text((x+r+10, y), label, fill=color, font=font)


def show_predictions_on_image(pano_root, predictions, out_img, ground_truth=True, show_coords=True, show_box=False):
    ''' annotates an image with a dict of string coordinates and labels
        if ground truth: also gets the ground truth and displays that as well '''
    pano_img_path   = pano_root + ".jpg"
    pano_xml_path   = pano_root + ".xml"
    pano_depth_path = pano_root + ".depth.txt"
    pano_yaw_deg    = extract_panoyawdeg(pano_xml_path)

    img = Image.open(pano_img_path)

    # convert from pytorch encoding to str
    for coord in predictions:
        int_label = predictions[coord]
        predictions[coord] = pytorch_label_from_int[int_label]

    def annotate_batch(predictions, color):
        count = 0
        for coords, prediction in predictions.iteritems():
            sv_x, sv_y = map(int, coords.split(','))

            if show_box:
                x = ((float(pano_yaw_deg) / 360) * gsv_image_width + sv_x) % gsv_image_width
                y = gsv_image_height / 2 - sv_y
                box = predict_crop_size(x, y, gsv_image_width, gsv_image_height, pano_depth_path)
            else: box = None

            label = str(prediction)
        
            #print "Found a {} at ({},{})".format(label, sv_x, sv_y)
            annotate(img, pano_yaw_deg, (sv_x, sv_y), label, color, show_coords, box)
            count += 1
        return count

    true_color = ImageColor.getrgb('Navy')
    pred_color = ImageColor.getrgb('red')
    cor_color  = ImageColor.getrgb('Chocolate')
    inc_color  = ImageColor.getrgb('FireBrick')

    
    if ground_truth:
        gt = get_ground_truth(os.path.split(pano_root)[1], pano_yaw_deg)
        for coord in gt: # convert to string labels
            gt[coord] = label_from_int[gt[coord]]
        true = annotate_batch(gt, true_color)

        correct, incorrect = partition_based_on_correctness(predictions, gt, R=200)
        pred = annotate_batch(correct, cor_color)
        pred += annotate_batch(incorrect, inc_color)
    else: true = 0

    if not ground_truth: pred = annotate_batch(predictions, pred_color)

    img.save(out_img)
    print "Marked {} predicted and {} true labels on {}.".format(pred, true, out_img)

    return

simple_dir = '/mnt/c/Users/gweld/sidewalk/sidewalk_ml/sliding_window/tiny_slid_win_crops/'
big_dir    = '/mnt/c/Users/gweld/sidewalk/sidewalk_ml/sliding_window/new_sliding_window_crops/'
pred_file_name = "20e_slid_win_w_feats_r18.csv"

# get and write predictions
#bps = predict_from_crops(big_dir, verbose=True)
#write_batch_predictions_to_file(bps, big_dir, pred_file_name)

# get and write ground_truth
# get_and_write_batch_ground_truth(big_dir)

# see if ground truth looks good
