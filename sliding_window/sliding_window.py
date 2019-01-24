# to be run in the sidewalk_pytorch environment defined in
# pytorch_pretrained/environment.yml

import base64, sys, json, os
from PIL import Image, ImageDraw, ImageFont, ImageColor
import numpy as np
import math
from collections import defaultdict
import csv
from copy import deepcopy
from clustering import non_max_sup
from precision_recall import precision_recall
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torchvision
from torchvision import datasets, models, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

try:
	from xml.etree import cElementTree as ET
except ImportError, e:
	from xml.etree import ElementTree as ET


gsv_image_width = 13312
gsv_image_height = 6656

path_to_gsv_scrapes  = "/mnt/c/Users/gweld/sidewalk/panos_drive_full/scrapes_dump/"
pano_db_export = '../../minus_onboard.csv'

label_from_int = ('Curb Cut', 'Missing Cut', 'Obstruction', 'Sfc Problem')
pytorch_label_from_int = ('Mussing Cut', "Null", 'Obstruction', "Curb Cut", "Sfc Problem")

model_path = '25epoch_full_ds_resnet18.pt'



############ Load the Model ################
# we're gonna do this just once

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

model_ft = models.resnet18()
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 5)
model_ft = model_ft.to( device )

model_ft.load_state_dict( torch.load(model_path, map_location='cpu') )
model_ft.eval()


############################################

def predict_single_image(imgfile):
	''' takes an image and returns a list of preds '''
	if not imgfile.endswith('.jpg'):
		raise IOError("{} is not a .jpg image".format(imgfile))


	loaded_img = Image.open(imgfile)
	loaded_img = data_transform(loaded_img).float()
	loaded_img = loaded_img.unsqueeze(0)

	with torch.no_grad():
		prediction = model_ft( loaded_img )

	return prediction.flatten().tolist()


def bilinear_interpolation(x, y, points):
	'''Interpolate (x,y) from values associated with four points.

	The four points are a list of four triplets:  (x, y, value).
	The four points can be in any order.  They should form a rectangle.

		>>> bilinear_interpolation(12, 5.5,
		...                        [(10, 4, 100),
		...                         (20, 4, 200),
		...                         (10, 6, 150),
		...                         (20, 6, 300)])
		165.0
	
	Code written by Raymond Hettinger. Check:
	http://stackoverflow.com/questions/8661537/how-to-perform-bilinear-interpolation-in-python
	
	Modified by Kotaro.
	In case four points have same x values or y values, perform linear interpolation
	'''
	# See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

	points = sorted(points)               # order points by x, then by y
	(x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points


	if (x1 == _x1) and (x1 == x2) and (x1 == _x2):
		if x != x1:
			raise ValueError('(x, y) not on the x-axis')
		if y == y1:
			return q11
		return (q11 * (_y2 - y) + q22 * (y - y1)) / ((_y2 - y1) + 0.0)
	if (y1 == _y1) and (y1 == y2) and (y1 == _y2):
		if y != y1 :
			raise ValueError('(x, y) not on the y-axis')
		if x == x1:
			return q11
		return (q11 * (_x2 - x) + q22 * (x - x1)) / ((_x2 - x1) + 0.0)
			

	if x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2:
		raise ValueError('points do not form a rectangle')
	if not x1 <= x <= x2 or not y1 <= y <= y2:
		print "x, y, x1, x2, y1, y2", x, y, x1, x2, y1, y2 
		raise ValueError('(x, y) not within the rectangle')

	return (q11 * (x2 - x) * (y2 - y) +
			q21 * (x - x1) * (y2 - y) +
			q12 * (x2 - x) * (y - y1) +
			q22 * (x - x1) * (y - y1)
		   ) / ((x2 - x1) * (y2 - y1) + 0.0)


def interpolated_3d_point(xi, yi, x_3d, y_3d, z_3d, scale=26):
	"""
	 This function takes a GSV image point (xi, yi) and 3d point cloud data (x_3d, y_3d, z_3d) and 
	 returns its estimated 3d point. 
	"""
	xi = float(xi) / scale
	yi = float(yi) / scale
	xi1 = int(math.floor(xi))
	xi2 = int(math.ceil(xi))
	yi1 = int(math.floor(yi))
	yi2 = int(math.ceil(yi))
	
	if xi1 == xi2 and yi1 == yi2:
		val_x = x_3d[yi1, xi1]
		val_y = y_3d[yi1, xi1]
		val_z = z_3d[yi1, xi1]
	else:
		points_x = ((xi1, yi1, x_3d[yi1, xi1]),   (xi1, yi2, x_3d[yi2, xi1]), (xi2, yi1, x_3d[yi1, xi2]), (xi2, yi2, x_3d[yi2, xi2]))         
		points_y = ((xi1, yi1, y_3d[yi1, xi1]),   (xi1, yi2, y_3d[yi2, xi1]), (xi2, yi1, y_3d[yi1, xi2]), (xi2, yi2, y_3d[yi2, xi2]))
		points_z = ((xi1, yi1, z_3d[yi1, xi1]),   (xi1, yi2, z_3d[yi2, xi1]), (xi2, yi1, z_3d[yi1, xi2]), (xi2, yi2, z_3d[yi2, xi2]))                  
		val_x = bilinear_interpolation(xi, yi, points_x)
		val_y = bilinear_interpolation(xi, yi, points_y)
		val_z = bilinear_interpolation(xi, yi, points_z)
	
	return (val_x, val_y, val_z)


def extract_panoyawdeg(path_to_metadata_xml):
	pano = {}
	pano_xml = open(path_to_metadata_xml, 'rb')
	tree = ET.parse(pano_xml)
	root = tree.getroot()
	for child in root:
		if child.tag == 'projection_properties':
			pano[child.tag] = child.attrib

	return pano['projection_properties']['pano_yaw_deg']


def get_depth_at_location(path_to_depth_txt, xi, yi):
	depth_location = path_to_depth_txt

	filename = depth_location

	with open(filename, 'rb') as f:
		depth = np.loadtxt(f)

	depth_x = depth[:, 0::3]
	depth_y = depth[:, 1::3]
	depth_z = depth[:, 2::3]

	val_x, val_y, val_z = interpolated_3d_point(xi, yi, depth_x, depth_y, depth_z)
	return val_x, val_y, val_z


def predict_crop_size(x, y, im_width, im_height, path_to_depth_file):
	"""
	# Calculate distance from point to image center
	dist_to_center = math.sqrt((x-im_width/2)**2 + (y-im_height/2)**2)
	# Calculate distance from point to center of left edge
	dist_to_left_edge = math.sqrt((x-0)**2 + (y-im_height/2)**2)
	# Calculate distance from point to center of right edge
	dist_to_right_edge = math.sqrt((x - im_width) ** 2 + (y - im_height/2) ** 2)

	min_dist = min([dist_to_center, dist_to_left_edge, dist_to_right_edge])

	crop_size = (4.0/15.0)*min_dist + 200

	print("Min dist was "+str(min_dist))
	"""
	crop_size = 0
	try:
		depth = get_depth_at_location(path_to_depth_file, x, y)
		depth_x = depth[0]
		depth_y = depth[1]
		depth_z = depth[2]

		distance = math.sqrt(depth_x ** 2 + depth_y ** 2 + depth_z ** 2)
		#print "\tDistance is {}".format(distance)
		if distance == "nan" or math.isnan(distance):
			# If no depth data is available, use position in panorama as fallback
			# Calculate distance from point to image center
			dist_to_center = math.sqrt((x - im_width / 2) ** 2 + (y - im_height / 2) ** 2)
			# Calculate distance from point to center of left edge
			dist_to_left_edge = math.sqrt((x - 0) ** 2 + (y - im_height / 2) ** 2)
			# Calculate distance from point to center of right edge
			dist_to_right_edge = math.sqrt((x - im_width) ** 2 + (y - im_height / 2) ** 2)

			min_dist = min([dist_to_center, dist_to_left_edge, dist_to_right_edge])

			crop_size = (4.0 / 15.0) * min_dist + 200

			#print "Depth data unavailable; using crop size " + str(crop_size)
		else:
			# crop_size = (30700.0/37.0)-(300.0/37.0)*distance
			# crop_size = 2600 - 220*distance
			# crop_size = (5875.0/3.0)-(275.0/3.0)*distance
			crop_size = 2050 - 110 * distance
			crop_size = 8725.6 * (distance ** -1.192)
			if crop_size < 50:
				crop_size = 50
			elif crop_size > 1500:
				crop_size = 1500

	except IOError:
		# If no depth data is available, use position in panorama as fallback
		# Calculate distance from point to image center
		dist_to_center = math.sqrt((x - im_width / 2) ** 2 + (y - im_height / 2) ** 2)
		# Calculate distance from point to center of left edge
		dist_to_left_edge = math.sqrt((x - 0) ** 2 + (y - im_height / 2) ** 2)
		# Calculate distance from point to center of right edge
		dist_to_right_edge = math.sqrt((x - im_width) ** 2 + (y - im_height / 2) ** 2)

		min_dist = min([dist_to_center, dist_to_left_edge, dist_to_right_edge])

		crop_size = (4.0 / 15.0) * min_dist + 200

		print "Depth data unavailable; using crop size " + str(crop_size)

	return crop_size


def make_single_crop_from_depth(path_to_image, sv_image_x, sv_image_y, PanoYawDeg, path_to_depth, output_filename):
	im_width = gsv_image_width
	im_height = gsv_image_height
	im = Image.open(path_to_image)
	draw = ImageDraw.Draw(im)
	# sv_image_x = sv_image_x - 100
	x = ((float(PanoYawDeg) / 360) * im_width + sv_image_x) % im_width
	y = im_height / 2 - sv_image_y

	crop_size = predict_crop_size(x, y, im_width, im_height, path_to_depth)
	try:
		crop_size = int(crop_size)
	except ValueError:
		crop_size = 300
		print "Invalid depth data, using crop_size=300 instead"

	# Crop rectangle around label
	cropped_square = None
	crop_width = int(crop_size)
	crop_height = int(crop_size)
	top_left_x = int(x - crop_width / 2)
	top_left_y = int(y - crop_height / 2)
	crop_box = (top_left_x, top_left_y, top_left_x + crop_width, top_left_y + crop_height)
	cropped_square = im.crop(crop_box)

	cropped_square.save(output_filename)

	return


def make_sliding_window_crops(pano_root, outdir, stride=100, bottom_space=1600, side_space=500):
	''' side padding is a multiple of stride '''
	pano_img_path   = pano_root + ".jpg"
	pano_xml_path   = pano_root + ".xml"
	pano_depth_path = pano_root + ".depth.txt"
	pano_yaw_deg = extract_panoyawdeg(pano_xml_path)

	x, y = side_space, 0
	while(y > - (gsv_image_height/2 - bottom_space)):
		while(x < gsv_image_width - side_space):
			# do things in one row
			output_filename  = os.path.join(outdir, "{},{}.jpg".format(x,y))
			print "cropping around ({},{})".format(x,y)
			try:
				make_single_crop_from_depth(pano_img_path, x, y, pano_yaw_deg, pano_depth_path, output_filename)
			except:
				print '\t cropping failed'
			x += stride
		y -= stride
		x = 2 * stride


def predict_from_crops(crops_dir, verbose=False):
	''' takes a directory full of crops, and returns dict
		mapping string of coords to list of predictions '''
	predictions = defaultdict(list)

	for imagename in os.listdir(crops_dir):
		if not imagename.endswith('.jpg'): continue
		coords = imagename[:-4]
		x,y = coords.split(',')			

		if verbose: print "getting predictions for {}".format(imagename)

		prediction = predict_single_image( os.path.join(crops_dir, imagename) )

		if verbose:	print prediction

		predictions[coords] = prediction

	return predictions


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


def predict(prediction):
	''' applies an algorithm to return the predicted class
		returns None if there's no prediction '''

	if type(prediction) == str: return prediction

	# currently returns the strongest prediction if either one
	# is >.5, otherwise returns None

	# Only look at ramps and missing ramps
	ramp = prediction[0]
	missing = prediction[1]
	overall = max(ramp, missing)
	best = label_from_int[0] if ramp > missing else label_from_int[1]
	if overall < .85: return None
	return best


def annotate(img, pano_yaw_deg, coords, label, color, show_coords=True):
	""" takes in an image object and labels it at specified coords
		translates streetview coords to pixel coords """
	sv_x, sv_y = coords
	x = ((float(pano_yaw_deg) / 360) * gsv_image_width + sv_x) % gsv_image_width
	y = gsv_image_height / 2 - sv_y

	if show_coords: label = "{},{} {}".format(sv_x, sv_y, label)

	# radius for dot
	r = 20
	draw = ImageDraw.Draw(img)
	draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

	font  = ImageFont.truetype("roboto.ttf", 60, encoding="unic")
	draw.text((x+r+10, y), label, fill=color, font=font)


def convert_to_real_coords(sv_x, sv_y, pano_yaw_deg):
	x = ((float(pano_yaw_deg) / 360) * gsv_image_width + sv_x) % gsv_image_width
	y = gsv_image_height / 2 - sv_y

	return int(x), int(y)

def convert_to_sv_coords(x, y, pano_yaw_deg):
	sv_x = x - (float(pano_yaw_deg)/360 * gsv_image_width)
	sv_y = gsv_image_height / 2 - y

	if sv_x < 0: sv_x += gsv_image_width 

	return int(sv_x), int(sv_y)


def get_ground_truth(pano_id, true_pano_yaw_deg):
	''' returns dict of str coords mapped to int label '''
	labels = {}
	with open(pano_db_export, 'r') as csvfile:
		reader = csv.reader(csvfile)

		for row in reader:
			if row[0] != pano_id: continue

			x, y = int(row[1]), int(row[2])
			label = int(row[3])-1
			photog_heading = float(row[4])

			pano_yaw_deg = 180 - photog_heading

			x, y = convert_to_real_coords(x, y, pano_yaw_deg)
			x, y = convert_to_sv_coords(x, y, true_pano_yaw_deg)

			# ignore other labels 
			if label not in range(4): continue

			labels["{},{}".format(x,y)] = label
	return labels


def show_predictions_on_image(pano_root, predictions, out_img, ground_truth=True):
	''' annotates an image with a dict of string coordinates and labels
		if ground truth: also gets the ground truth and displays that as well '''
	pano_img_path   = pano_root + ".jpg"
	pano_xml_path   = pano_root + ".xml"
	pano_depth_path = pano_root + ".depth.txt"
	pano_yaw_deg = extract_panoyawdeg(pano_xml_path)

	img = Image.open(pano_img_path)

	def annotate_batch(predictions, color):
		count = 0
		for coords, prediction in predictions.iteritems():
			sv_x, sv_y = map(int, coords.split(','))

			label = str(prediction)
		
			#print "Found a {} at ({},{})".format(label, sv_x, sv_y)
			annotate(img, pano_yaw_deg, (sv_x, sv_y), label, color, show_coords=True)
			count += 1
		return count

	true_color = ImageColor.getrgb('blue')
	pred_color = ImageColor.getrgb('red')

	
	pred = annotate_batch(predictions, pred_color)
	if ground_truth:
		gt = get_ground_truth(pano_root, pano_yaw_deg)
		true = annotate_batch(gt, true_color)
	else: true = 0
	img.save(out_img)
	print "Marked {} predicted and {} true labels on {}.".format(pred, true, out_img)

	#############
	return


def scale_non_null_predictions(predictions, factor):
	''' multiply all non-nullcrop terms (ie all but the last)
		of a set of	predictions by a constant factor  '''

	# short circuit if we don't need to change anything
	if factor == 1: return predictions

	def scale(l):
		new_l = []
		for i in range(len(l)-1):
			new_l.append( l[i] * factor )
		new_l.append(l[-1])
		return new_l

	new_preds = {}
	for coords in predictions:
		new_preds[coords] = scale(predictions[coords])
	
	return new_preds


def batch_sliding_window(pano_roots_path, outdir, stride=150, bottom_space=1500, side_space=500):
	''' takes a list of pano roots in a text file, one per line
		and gets pr
		note: side padding is a multiple of stride
	'''

	pano_roots = []
	with open(pano_roots_path, 'r') as pano_list_file:
		for pano_root in pano_list_file.readlines():
			pano_roots.append(pano_root.strip())

	print "Attempting to process {} panoramas.".format(len(pano_roots))

	for pano_root in pano_roots:
		try:
			print "Starting on {}".format(pano_root)
			pano_xml_path   = os.path.join(path_to_gsv_scrapes, pano_root[:2], pano_root + ".xml")
			pano_img_path   = os.path.join(path_to_gsv_scrapes, pano_root[:2], pano_root + ".jpg")
			pano_depth_path = os.path.join(path_to_gsv_scrapes, pano_root[:2], pano_root + ".depth.txt")
			pano_yaw_deg = extract_panoyawdeg(pano_xml_path)

			output_dir = os.path.join(outdir, pano_root)
			if (not os.path.exists(output_dir)) or (not os.path.isdir(output_dir)):
				print "\t Making output directory {}".format(output_dir)
				os.mkdir(output_dir)
			predictions_file  = os.path.join(output_dir, 'predictions.csv')
			ground_truth_file = os.path.join(output_dir, 'ground_truth.csv')

			#### CROPS ####

			print "\t Cropping into {}".format(output_dir)
			num_succesful_crops = 0
			x, y = side_space, 0
			while(y > - (gsv_image_height/2 - bottom_space)):
				while(x < gsv_image_width - side_space):
					# do things in one row
					output_filename  = os.path.join(output_dir, "{},{}.jpg".format(x,y))
					try:
						make_single_crop_from_depth(pano_img_path, x, y, pano_yaw_deg, pano_depth_path, output_filename)
						num_succesful_crops += 1
					except Exception as e:
						print '\t\tcropping around ({},{}) failed:'.format(x,y)
						print e
					x += stride
				y -= stride
				x = 2 * stride

			print "\t Finished cropping - {} crops made successfully".format(num_succesful_crops)

			#### PREDICTIONS ####
			print "\t Getting predictions for {} crops".format(num_succesful_crops)
			predictions = predict_from_crops(output_dir)
			write_predictions_to_file(predictions, predictions_file)

			#### GROUND TRUTH ####
			print "\t Getting ground truth for {}".format(pano_root)
			gt = get_ground_truth(pano_root, pano_yaw_deg)
			print "\t Found {} ground truth points".format(len(gt))
			write_gt_to_file(gt, ground_truth_file)

			print "Finished processing {}".format(pano_root)
		except Exception as e:
			print "Processing {} failed".format(pano_root)
			print e


def batch_predictions_only(dir_containing_crops, filename=model_path[:-3]+"_preds.csv"):
	""" gets predictions for premade crops only
		takes a directory containing directories of crops, and in
		each subdirectory containing crops, writes a file with predictions
		for later use.
	"""

	num_panos = 0
	num_preds = 0

	for root, dirs, files in os.walk(dir_containing_crops):
		# skip empty dirs
		if len(files) < 1: continue

		pano_root = os.path.basename(root)
		p_file = os.path.join(root, filename)
		print "Processing predictions for {}".format(pano_root)

		print "\tAttempting to get predictions for {} crops.".format(len(files))

		predictions = predict_from_crops(root)

		print "\tSuccessfully got {} predictions.".format(len(predictions))

		write_predictions_to_file(predictions, p_file)

		num_panos += 1
		num_preds += len(predictions)

	print "Computed and saved {} predictions for {} panos.".format(num_preds, num_panos)
	return



def batch_p_r(dir_containing_preds, scaling=1, clust_r, cor_r, clip_val=None):
	""" Computes precision and recall given a directory containing subdirectories
		containg predictions and ground truth csvs

		scaling multiplies non-null crop predictions by a constant factor
		clust_r sets the distance below which adjacent predictions will be clustered together
		cor_r sets the 'correct' distance, a prediction within this distance of a 
			ground truth  point will be considered correct
		clip_val will ignore predictions with a strength less than this value
	"""
	
	# sum_pr keeps track of the counts of [correct, predicted, actual]
	sum_pr = np.zeros((4,3))

	for root, dirs, files in os.walk(dir_containing_preds):

		# skip directories that don't contain 
		# predictions and ground truths
		if len(files) < 2: continue

		pano_root = os.path.basename(root)
		print "Processing predictions for {}".format(pano_root)
		p_file = os.path.join(root, 'predictions.csv')
		gt_file = os.path.join(root, 'ground_truth.csv') 

		try:
			predictions = read_predictions_from_file(p_file)
			gt = read_predictions_from_file(gt_file)

			print "\t Loaded {} predictions and {} true labels".format(len(predictions), len(gt))

			predictions = scale_non_null_predictions(predictions, scaling)
			predictions = non_max_sup(predictions, radius=clust_r, clip_val=clip_val, ignore_last=True)

			pr = precision_recall(predictions, gt, cor_r, N_classes=4)

			# sum_pr keeps track of the counts of [correct, predicted, actual]
			sum_pr += pr

		except IOError as e:
			print "\t Could not read predictions for {}, skipping.".format(pano_root)
			
	pr_dict = {}
	for num, name in enumerate(label_from_int):
		cor, pred, actual = sum_pr[num,:]

		if pred > 0:
			precision = float(cor)/pred
		else: precision = float('nan')

		if actual > 0:
			recall = float(cor)/actual
		else: recall = float('nan')

		pr_dict[name] = (precision,recall)

		print "{:15}\t{:0.2f}\t{:0.2f}".format(name, precision, recall)

	return pr_dict



#batch_p_r('./batch_test/', 5, 150, 500)

# predictions = read_predictions_from_file('new_test_preds.csv')
# predictions = scale_non_null_predictions(predictions, 5)
# predictions = non_max_sup(predictions, radius=100, clip_val=None, ignore_last=True)
# show_predictions_on_image('1_1OfETDixMMCUhSWn-hcA', predictions, 'non_max.jpg', ground_truth=True)

# paused here
#batch_sliding_window("inc_sample50.txt", "./batch_50/", stride=150)



#make_sliding_window_crops('1_1OfETDixMMCUhSWn-hcA', test_crops, stride=100)

#show predictions for single img of curb ramp
#print predict_label('38.jpg')

batch_predictions_only('batch_50')