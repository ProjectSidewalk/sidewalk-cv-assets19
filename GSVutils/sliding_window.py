# use this to make sliding window crops

from collections import defaultdict
import sys
import os
import csv
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append( os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'pytorch_pretrained') )

from PIL import Image, ImageDraw, ImageFont, ImageColor
import numpy as np

import GSVutils.utils

from GSVutils.point import Point as Point
from GSVutils.pano_feats import Pano as Pano
from GSVutils.pano_feats import Feat as Feat
from GSVutils.clustering import non_max_sup
from GSVutils.precision_recall import precision_recall, partition_based_on_correctness
from GSVutils.scoring import score

import torchvision
from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from TwoFileFolder import TwoFileFolder
from resnet_extended import extended_resnet18

GSV_IMAGE_HEIGHT = GSVutils.utils.GSV_IMAGE_HEIGHT
GSV_IMAGE_WIDTH  = GSVutils.utils.GSV_IMAGE_WIDTH

label_from_int = ('Curb Cut', 'Missing Cut', 'Obstruction', 'Sfc Problem')
pytorch_label_from_int = ('Missing Cut', "Null", 'Obstruction', "Curb Cut", "Sfc Problem")

path_to_gsv_scrapes  = "/mnt/f/scrapes_dump/"
#pano_db_export = '/mnt/c/Users/gweld/sidewalk/minus_onboard.csv'
pano_db_export = '/mnt/c/Users/gweld/sidewalk/sidewalk_ml/ground_truth/ground_truth_labels.csv'



model_dir = '/mnt/c/Users/gweld/sidewalk/sidewalk_ml/pytorch_pretrained/models/'
model_name = "20ep_sw_re18_2ff2"
#model_name = "25epoch_full_ds_resnet18"

model_path = os.path.join(model_dir, model_name+'.pt')



############################################


def predict_from_crops(dir_containing_crops, verbose=False):
    ''' use the TwoFileFolder dataloader to load images and feed them
        through the model
        returns a dict mapping pano_ids to dicts of {coord: prediction lists}
    '''
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    print "Building dataset and loading model..."
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image_dataset = TwoFileFolder(dir_containing_crops, meta_to_tensor_version=2, transform=data_transform)
    dataloader    = torch.utils.data.DataLoader(image_dataset, batch_size=4, shuffle=True, num_workers=4)

    len_ex_feats = image_dataset.len_ex_feats
    dataset_size = len(image_dataset)

    panos = image_dataset.classes

    print("Using dataloader that supplies {} extra features.".format(len_ex_feats))
    print("")
    print("Finished loading data. Got crops from {} panos.".format(len(panos)))


    model_ft = extended_resnet18(len_ex_feats=len_ex_feats)

    try:
        model_ft.load_state_dict(torch.load(model_path))
    except RuntimeError as e:
        model_ft.load_state_dict(torch.load(model_path, map_location='cpu'))
    model_ft = model_ft.to( device )
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    model_ft.eval()

    paths_out = []
    pred_out  = []

    print "Computing predictions...."
    for inputs, labels, paths in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimizer_ft.zero_grad()

        with torch.set_grad_enabled(False):
            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)

            paths_out += list(paths)
            pred_out  += list(outputs.tolist())

    predictions = defaultdict(dict)
    for i in range(len(paths_out)):
        path  = paths_out[i]
        preds = pred_out[i]

        _, filename = os.path.split(path)
        filebase, _ = os.path.splitext(filename)
        pano_id, coords = filebase.split('crop')

        #print pano_id, coords, preds
        predictions[pano_id][coords] = preds

    return predictions


def get_and_write_batch_ground_truth(dir_containing_crops):
    ground_truths = {}
    sizess        = {}
    for pano_id in os.listdir(dir_containing_crops):
        print "Getting ground truth for pano {}".format(pano_id)
        pano_xml_path   = os.path.join(path_to_gsv_scrapes, pano_id[:2], pano_id + ".xml")
        true_pano_yaw_deg = GSVutils.utils.extract_panoyawdeg(pano_xml_path)

        gt, sizes = get_ground_truth(pano_id, true_pano_yaw_deg)

        print "\tFound {} ground truth labels.".format(len(gt))
        ground_truths[pano_id] = gt
        sizess[pano_id]        = sizes

    write_batch_predictions_to_file(ground_truths,dir_containing_crops, 'ground_truth.csv')
    write_batch_predictions_to_file(sizess, dir_containing_crops, 'gt_sizes.csv')


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


def make_sliding_window_crops(list_of_panos, dir_to_save_to, skip_existing_dirs=False):
    ''' take a text file containing a list of panos and add to dir'''
    panos_to_crop = set()

    with open(list_of_panos) as f:
        for line in f:
            panos_to_crop.add(line[:-2])
    print "Making crops for {} panos.".format(len(panos_to_crop))

    num_panos = 0
    num_crops = 0
    num_fail  = 0

    error_panos =  set()

    for pano_id in panos_to_crop:
        pano_root = os.path.join(path_to_gsv_scrapes, pano_id[:2], pano_id)
        pano_img_path   = pano_root + ".jpg"
        pano_xml_path   = pano_root + ".xml"
        pano_depth_path = pano_root + ".depth.txt"
        try:
            pano_yaw_deg = GSVutils.utils.extract_panoyawdeg(pano_xml_path)
        except Exception as e:
            print "Error extracting Pano Yaw Deg:"
            print e
            error_panos.add(pano_id)
            continue

        destination_folder = os.path.join(dir_to_save_to, pano_id)
        if os.path.isdir(destination_folder) and skip_existing_dirs:
            print "Skipping existing crops for pano {}".format(pano_id)
            continue
        if not os.path.isdir(destination_folder):
            os.makedirs(destination_folder)

        pano = Pano()
        pano.pano_id = pano_id
        pano.photog_heading = None

        for feat in sliding_window(pano): # ignoring labels here
            sv_x, sv_y = feat.sv_image_x, feat.sv_image_y
            print "cropping around ({},{})".format(sv_x, sv_y)
            output_filebase = os.path.join(destination_folder, pano_id+'crop{},{}'.format(sv_x, sv_y))

            try:
                GSVutils.utils.make_single_crop(pano_id, sv_x, sv_y, pano_yaw_deg, output_filebase)
                num_crops += 1
            except Exception as e:
                print "\t cropping failed"
                print e
                num_fail += 1
        num_panos += 1
    print "Finished. {} crops succeeded, {} failed. {} panos.".format(num_crops, num_fail, num_panos)
    print "Failed to find XML for {} panos:".format(len(error_panos))
    for pano_id in error_panos:
        print pano_id


def convert_to_sv_coords_to_real_coords(sv_x, sv_y, pano_yaw_deg):
    x = ((float(pano_yaw_deg) / 360) * GSV_IMAGE_WIDTH + sv_x) % GSV_IMAGE_WIDTH
    y = GSV_IMAGE_HEIGHT / 2 - sv_y

    return int(x), int(y)


def convert_to_sv_coords(x, y, pano_yaw_deg):
    sv_x = x - (float(pano_yaw_deg)/360 * GSV_IMAGE_WIDTH)
    sv_y = GSV_IMAGE_HEIGHT / 2 - y

    if sv_x < 0: sv_x += GSV_IMAGE_WIDTH 

    return int(sv_x), int(sv_y)


def get_ground_truth(pano_id, true_pano_yaw_deg):
    ''' returns dict of str coords mapped to int label, and
        a dict of crop sizes for each gt label
     '''
    path_to_depth = os.path.join(path_to_gsv_scrapes, pano_id[:2], pano_id+'.depth.txt')

    labels = {}
    sizes =  {}
    with open(pano_db_export, 'r') as csvfile:
        reader = csv.reader(csvfile)

        for row in reader:
            if row[0] != pano_id: continue

            x, y = int(row[1]), int(row[2])
            label = int(row[3])-1 # compensate for 1-indexing
            photog_heading = float(row[4])

            pano_yaw_deg = 180 - photog_heading

            # I don't entirely know why this is needed, but it is
            # my guess is to convert from viewer to photog yaw
            x, y = convert_to_sv_coords_to_real_coords(x, y, pano_yaw_deg)
            x, y = convert_to_sv_coords(x, y, true_pano_yaw_deg)

            try:
                predicted_crop_size = GSVutils.utils.predict_crop_size(x, y, GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT, path_to_depth)
            except (ValueError, IndexError) as e:
                predicted_crop_size = GSVutils.utils.predict_crop_size_by_position(x, y, GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT)

            # ignore other labels 
            if label not in range(4): continue

            labels["{},{}".format(x,y)] = label
            sizes["{},{}".format(x,y)]  = predicted_crop_size
    return labels, sizes


def read_predictions_from_file(path):
    predictions = defaultdict(list)

    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile)

        for row in reader:
            x, y = row[0], row[1]
            prediction = map(float, row[2:])

            # let this work for processed predictions, as well
            if len(prediction) == 1:
                try:
                    prediction = int(prediction[0])
                except ValueError:
                    continue

            predictions["{},{}".format(x,y)] = prediction
    return predictions


def annotate(img, pano_yaw_deg, coords, label, color, show_coords=True, box=None):
    """ takes in an image object and labels it at specified coords
        translates streetview coords to pixel coords
        if given a box, marks that box around the label
    """
    sv_x, sv_y = coords
    x = ((float(pano_yaw_deg) / 360) * GSV_IMAGE_WIDTH + sv_x) % GSV_IMAGE_WIDTH
    y = GSV_IMAGE_HEIGHT / 2 - sv_y

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


def show_predictions_on_image(pano_root, correct, incorrect, out_img, predicted_gt_pts={}, missed_gt_points={}, show_coords=True, show_box=False):
    ''' annotates an image with with predictions. 
        each of the arguments in (correct, incorrect, predicted_gt_pts, missed_gt_points is
        a dict of string coordinates and labels (output from scoring.score)
        leave predicted and missed as default to skip ground truth
        show_coords will plot the coords, and show box will plot the bounding box. '''
    pano_img_path   = pano_root + ".jpg"
    pano_xml_path   = pano_root + ".xml"
    pano_depth_path = pano_root + ".depth.txt"
    pano_yaw_deg    = GSVutils.utils.extract_panoyawdeg(pano_xml_path)

    img = Image.open(pano_img_path)

    # convert from pytorch encoding to str
    for d in (correct, incorrect, predicted_gt_pts, missed_gt_points):
        for coord in d:
            int_label = d[coord]
            d[coord] = label_from_int[int_label]

    def annotate_batch(predictions, color):
        count = 0
        for coords, prediction in predictions.iteritems():
            sv_x, sv_y = map(float, coords.split(','))

            if show_box:
                x = ((float(pano_yaw_deg) / 360) * GSV_IMAGE_WIDTH + sv_x) % GSV_IMAGE_WIDTH
                y = GSV_IMAGE_HEIGHT / 2 - sv_y
                try:
                    box = GSVutils.utils.predict_crop_size(x, y, GSV_IMAGE_WIDTH, GSV_IMAGE_HEIGHT, pano_depth_path)
                except:
                    print "Couldn't get crop size... skipping box"
                    box = None
            else: box = None

            label = str(prediction)
        
            #print "Found a {} at ({},{})".format(label, sv_x, sv_y)
            annotate(img, pano_yaw_deg, (sv_x, sv_y), label, color, show_coords, box)
            count += 1
        return count

    # gt colors
    true_color = ImageColor.getrgb('lightseagreen')
    miss_color = ImageColor.getrgb('red')

    # prediction colors
    cor_color  = ImageColor.getrgb('palegreen')
    inc_color  = ImageColor.getrgb('lightsalmon')

    true = 0
    pred = 0
    for color, d in ((true_color, predicted_gt_pts), (miss_color, missed_gt_points), (cor_color, correct), (inc_color, incorrect)):
        marked = annotate_batch(d, color)

        if d in (predicted_gt_pts, missed_gt_points):
            true += 1
        if d in (correct, incorrect):
            pred += 1

    img.save(out_img)
    print "Marked {} predicted and {} true labels on {}.".format(pred, true, out_img)

    return


def batch_visualize_preds(dir_containing_panos, outdir, clust_r, cor_r, dynamic_r=True, clip_val=None, preds_filename='predictions.csv'):
    count = 0
    for pano_id in os.listdir(dir_containing_panos):
        print "Annotating {}".format(pano_id)

        predictions_file = os.path.join(dir_containing_panos, pano_id, preds_filename)
        pred = read_predictions_from_file(predictions_file)

        gt_file = os.path.join(dir_containing_panos, pano_id, 'ground_truth.csv')
        gt = read_predictions_from_file(gt_file)

        sizes_file = os.path.join(dir_containing_panos, pano_id, 'gt_sizes.csv')
        s = read_predictions_from_file(sizes_file)

        correct, incorrect, predicted, missed = score(pred, gt, s, cor_r, clust_r=clust_r, dynamic_r=dynamic_r, clip_val=clip_val)

        outfile = os.path.join(outdir, pano_id+'.jpg')

        pano_root = os.path.join(path_to_gsv_scrapes, pano_id[:2], pano_id)
        show_predictions_on_image(pano_root, correct, incorrect, outfile, predicted, missed, show_coords=False, show_box=True)
        count += 1

        #if count > 0: break
    print "Wrote predictions for {} panos to {}".format(count, outdir)
    return


def fetch_unlabeled_panos(dir_containing_panos, outdir):
    ''' simply copy panos from the scrapes folder to outdir '''
    count = 0
    for pano_id in os.listdir(dir_containing_panos):
        print "Getting {}".format(pano_id)

        panofile = os.path.join(path_to_gsv_scrapes, pano_id[:2], pano_id+'.jpg')

        outfile = os.path.join(outdir, pano_id+'.jpg')

        shutil.copy(panofile, outfile)
        count += 1

        #if count > 0: break
    print "Copied {} panos to {}".format(count, outdir)
    return


def batch_p_r(dir_containing_preds, clust_r, cor_r, dynamic_r=True, clip_val=None, preds_filename='predictions.csv'):
    """ Computes precision and recall given a directory containing subdirectories
        containg predictions and ground truth csvs

        clust_r sets the distance below which adjacent predictions will be clustered together

        cor_r is a radius for correctness. This float is a fraction of the crop size,
        eg: for a feature to be marked correct, it must be within r * crop_size (length of a single edge)
    """
    num_panos = 0

    num_correct = defaultdict(int)
    num_actual  = defaultdict(int)
    num_pred    = defaultdict(int)

    for root, dirs, files in os.walk(dir_containing_preds):

        # skip directories that don't contain 
        # predictions and ground truths
        if len(files) < 2: continue

        pano_root = os.path.basename(root)
        print "Processing predictions for {}".format(pano_root)
        p_file     = os.path.join(root, preds_filename)
        gt_file    = os.path.join(root, 'ground_truth.csv')
        sizes_file = os.path.join(root, 'gt_sizes.csv')

        try:
            predictions = read_predictions_from_file(p_file)
            gt          = read_predictions_from_file(gt_file)
            sizes       = read_predictions_from_file(sizes_file)
            print "\t Loaded {} predictions and {} true labels".format(len(predictions), len(gt))

            corrects, incorrects, _, _ = score(predictions, gt, sizes, cor_r, clust_r, dynamic_r=dynamic_r, clip_val=clip_val)
            # taking care of converting predictions to single int labels
            predictions = {}
            for d in (corrects, incorrects):
                for coords, label in d.items():
                    predictions[coords] =  label

            # now we can compute the number of corrects, predicteds, and actual for each label
            for d_to_add, d_to_iter in ((num_correct, corrects), (num_actual, gt), (num_pred, predictions)):
                for _, label_int in d_to_iter.items():
                    label_name = label_from_int[label_int]
                    d_to_add[label_name] += 1

            num_panos += 1
        except IOError as e:
            print "\t Could not read predictions for {}, skipping.".format(pano_root)
    print('Scored predictions for {} panos.'.format(num_panos))

    print("{:<20}{:^6} {:^6} {:^6}".format("Label", "cor", "pred", 'act'))
    for label in num_actual:
        correct   = int(num_correct[label])
        predicted = int(num_pred[label])
        actual    = int(num_actual[label])
        print("{:20}{:6d} {:6d} {:6d}".format(label, correct, predicted, actual))
    print("")

    output = {}
    print("{:<20}{:^6}   {:^6} ".format("Label", "p", "r"))
    for label in num_actual:
        correct   = float(num_correct[label])
        predicted = float(num_pred[label])
        actual    = float(num_actual[label])

        p = 100 * (correct / predicted) if predicted >0 else float('nan')
        r = 100 * (correct / actual)

        print("{:20}{:6.2f}%  {:06.2f}%".format(label, p, r))
        output[label] = (p,r)
    # let's do for overall
    correct   = float(sum( num_correct.values() ) )
    predicted = float(sum( num_pred.values()    ) )
    actual    = float(sum( num_actual.values()  ) )

    p = 100 * (correct / predicted) if predicted >0 else float('nan')
    r = 100 * (correct / actual)
    print("{:20}{:06.2f}%  {:06.2f}%".format("Overall", p, r))
    output["Overall"] = (p,r)

    return output








simple_dir = '/mnt/c/Users/gweld/sidewalk/sidewalk_ml/sliding_window/gt_crops_small/'
#big_dir    = '/mnt/c/Users/gweld/sidewalk/sidewalk_ml/sliding_window/new_sliding_window_crops/'
gt_dir     = '/mnt/c/Users/gweld/sidewalk/sidewalk_ml/sliding_window/ground_truth_crops/'

pred_file_name = model_name + ".csv"



# get and write predictions
#bps = predict_from_crops(gt_dir, verbose=True)
#write_batch_predictions_to_file(bps, gt_dir, pred_file_name)

# get and write ground_truth
#get_and_write_batch_ground_truth(gt_dir)

# see if ground truth looks good
#batch_visualize_preds(simple_dir, '/mnt/c/Users/gweld/sidewalk/sidewalk_ml/sliding_window/test/', pred_file_name)

# let's try this out...
#batch_p_r(simple_dir, 150, 1.0, clip_val=4.5, preds_filename=pred_file_name)


# stuff for genrerating ground truth crops here
#ground_truth_labels = '/mnt/c/Users/gweld/sidewalk/sidewalk_ml/ground_truth/ground_truth_labels.csv'
#ground_truth_panos = '/mnt/c/Users/gweld/sidewalk/sidewalk_ml/ground_truth/ground_truth_panos.txt'
#make_sliding_window_crops(ground_truth_panos, gt_dir, skip_existing_dirs=True)

# show labels
outdir = '/mnt/c/Users/gweld/sidewalk/sidewalk_ml/sliding_window/labeled_gt_panos'
batch_visualize_preds(gt_dir, outdir, 150, 1.0, clip_val=4.5, preds_filename=pred_file_name)

# fetch panos
#outdir = '/mnt/c/Users/gweld/sidewalk/sidewalk_ml/sliding_window/unlabeled_gt_panos'
#fetch_unlabeled_panos(gt_dir, outdir)

