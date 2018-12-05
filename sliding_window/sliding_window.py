import base64, sys, json, os
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont, ImageColor
from oauth2client.client import GoogleCredentials
from googleapiclient import discovery
import numpy as np
import math
from collections import defaultdict
import csv
from copy import deepcopy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from GSV import GSVImage
#from GSV.utilities import *
try:
	from xml.etree import cElementTree as ET
except ImportError, e:
	from xml.etree import ElementTree as ET

# will need to install Google API client
# pip install --upgrade google-api-python-client

GCLOUD_SERVICE_KEY = '/home/gweld/gcloud_acct_key.json'
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = GCLOUD_SERVICE_KEY

PROJECT = 'sidewalk-dl'
BUCKET = 'sidewalk_crops_subset'
REGION = 'us-central1'

MODEL = 'sidewalk'
VERSION = 'resnet'

gsv_image_width = 13312
gsv_image_height = 6656

label_from_int = ('Curb Cut', 'Missing Cut', 'Obstruction', 'Sfc Problem')

def predict_label(img_path):
	def predict_json(project, model, instances, version=None):
		# from:
		# https://cloud.google.com/ml-engine/docs/tensorflow/online-predict#creating_models_and_versions
		service = discovery.build('ml', 'v1')
		name = 'projects/{}/models/{}'.format(project, model)

		if version is not None:
			name += '/versions/{}'.format(version)

		response = service.projects().predict(
			name=name,
			body={'instances': instances}
		).execute()

		if 'error' in response:
			raise RuntimeError(response['error'])

		return response['predictions']

	with open(img_path, 'r') as imgfile:
		image_data = imgfile.read()
		img = base64.b64encode(image_data)
		instances = {'image_bytes' : {'b64': img}}
		predictions = predict_json(PROJECT, MODEL, instances, VERSION)
		return predictions[0][u'probabilities']


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
		print "\tDistance is {}".format(distance)
		if distance == "nan":
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

predictions = {'1600,-500':[.5, 0, 0, 0], "3200,-800":[.1, 0, 0, 0]}


def predict_from_crops(crops_dir):
	''' returns dict mapping string of coords to list of predictions '''
	predictions = defaultdict(list)

	for imagename in os.listdir(crops_dir):
		if not imagename.endswith('.jpg'): continue
		coords = imagename[:-4]
		x,y = coords.split(',')

		prediction = predict_label(os.path.join(crops_dir, imagename))

		print "getting predictions for {}".format(imagename)
		print prediction

		predictions[coords] = prediction

	return predictions


def write_predictions_to_file(predictions, path):
	count = 0
	with open(path, 'w') as csvfile:
		writer = csv.writer(csvfile)

		for coods, prediction in predictions.iteritems():
			x,y = coods.split(',')
			row = [x,y] + prediction
			writer.writerow(row)
			count += 1
		print "Wrote {} predictions to {}.".format(count, path)


def read_predictions_from_file(path):
	predictions = defaultdict(list)

	with open(path, 'r') as csvfile:
		reader = csv.reader(csvfile)

		for row in reader:
			x, y = row[0], row[1]
			prediction = map(float, row[2:])
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

	return int(sv_x), int(sv_y)


def get_ground_truth(pano_id, true_pano_yaw_deg, cropsfile='../../minus_onboard.csv'):
	labels = {}
	with open(cropsfile, 'r') as csvfile:
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

			labels["{},{}".format(x,y)] = label_from_int[label]
	return labels


# WORK IN PROGRESS FUNCTION, DON'T USE!!!!!!!!!!!
def non_max_suppression(predictions, radius, clip=None):
	''' non_max suppresion, ignoring predictions with magnitude < clip '''

	def near_any(this, cluster):
		for point in cluster:
			dif = (point[0]-this[0], point[1]-this[0])
			dist = math.sqrt(dif[0]**2 + dist[1]**2)

			if dist <= radius: return True
		return False

	predictions = deepcopy(predictions) # don't edit

	for coords in predictions:
		predcition = predictions[coords]
		if clip is not None and max(prediction) < clip:
			del predictions[coords]
		# ignore if last label is strongest (eg nullcrop)
		if prediction.index(max(prediction)) == len(prediction)-1:
			del predictions[coords]

	# load coords into list of tups
	coords = set()
	for coord in predictions:
		x,y = map(int, coord.split(','))
		coords.add( (x,y) )

	clusters = []
	# cluster coords into sets
	while len(coords) > 0:
		# get arbitrary element and remove
		this = coords.pop()

		# find if it goes in an existing cluster 
		for cluster in clusters:
			if near_any(this, cluster):
				cluster.append(this)
				break

		# create a new cluster for this point
		clusters.append([this])

	# now we have our clusters
	# need to get the max point for each cluster
	for cluster in clusters:
		pass

	# need account for different types!!!


def show_predictions_on_image(pano_root, predictions, out_img, ground_truth=True):
	pano_img_path   = pano_root + ".jpg"
	pano_xml_path   = pano_root + ".xml"
	pano_depth_path = pano_root + ".depth.txt"
	pano_yaw_deg = extract_panoyawdeg(pano_xml_path)
	print "Pano Yaw Degree={}".format(pano_yaw_deg)

	img = Image.open(pano_img_path)

	def annotate_batch(predictions, color):
		count = 0
		for coords, prediction in predictions.iteritems():
			sv_x, sv_y = map(int, coords.split(','))
			label = predict(prediction)
			if label is not None:
				print "Found a {} at ({},{})".format(label, sv_x, sv_y)
				annotate(img, pano_yaw_deg, (sv_x, sv_y), label, color, show_coords=False)
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
	return










#predictions = read_predictions_from_file('test_preds.csv')

#predictions = {"1500,-500":"Test", '8024,-541': 'Curb Cut from file',}
#show_predictions_on_image('1_1OfETDixMMCUhSWn-hcA', predictions, 'preds.jpg', ground_truth=True)

#make_sliding_window_crops('1_1OfETDixMMCUhSWn-hcA', test_crops, stride=100)

# make new predictions 
#predictions = predict_from_crops('test_crops')
#write_predictions_to_file(predictions, 'test_preds.csv')

