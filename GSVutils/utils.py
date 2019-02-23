from collections import defaultdict
import math
import csv
import os
import numpy as np
from copy import deepcopy

import base64, sys, json, os
from PIL import Image, ImageDraw, ImageFont, ImageColor
try:
	from xml.etree import cElementTree as ET
except ImportError, e:
	from xml.etree import ElementTree as ET

GSV_IMAGE_WIDTH  = 13312
GSV_IMAGE_HEIGHT = 6656

label_from_int   = ('Curb Cut', 'Missing Cut', 'Obstruction', 'Sfc Problem')

path_to_gsv_scrapes = "/mnt/f/scrapes_dump/"


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


def extract_panoyawdeg(path_to_metadata_xml):
	pano = {}
	pano_xml = open(path_to_metadata_xml, 'rb')
	tree = ET.parse(pano_xml)
	root = tree.getroot()
	for child in root:
		if child.tag == 'projection_properties':
			pano[child.tag] = child.attrib

	return pano['projection_properties']['pano_yaw_deg']


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


def predict_crop_size_by_position(x, y, im_width, im_height):
	dist_to_center = math.sqrt((x - im_width / 2) ** 2 + (y - im_height / 2) ** 2)
	# Calculate distance from point to center of left edge
	dist_to_left_edge = math.sqrt((x - 0) ** 2 + (y - im_height / 2) ** 2)
	# Calculate distance from point to center of right edge
	dist_to_right_edge = math.sqrt((x - im_width) ** 2 + (y - im_height / 2) ** 2)

	min_dist = min([dist_to_center, dist_to_left_edge, dist_to_right_edge])

	crop_size = (4.0 / 15.0) * min_dist + 200

	return crop_size


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

	return crop_size


def make_single_crop(pano_id, sv_image_x, sv_image_y, PanoYawDeg, output_filebase):
	img_filename  = output_filebase + '.jpg'
	meta_filename = output_filebase + '.json'

	path_to_depth = os.path.join(path_to_gsv_scrapes, pano_id[:2], pano_id + ".depth.txt")
	path_to_image = os.path.join(path_to_gsv_scrapes, pano_id[:2], pano_id + ".jpg")

	im_width = GSV_IMAGE_WIDTH
	im_height = GSV_IMAGE_HEIGHT
	im = Image.open(path_to_image)
	draw = ImageDraw.Draw(im)
	# sv_image_x = sv_image_x - 100
	x = ((float(PanoYawDeg) / 360) * im_width + sv_image_x) % im_width
	y = im_height / 2 - sv_image_y

	# Crop rectangle around label
	cropped_square = None
	try:
		predicted_crop_size = predict_crop_size(x, y, im_width, im_height, path_to_depth)
		crop_width = predicted_crop_size
		crop_height = predicted_crop_size
		print(x, y)
		top_left_x = x - crop_width / 2
		top_left_y = y - crop_height / 2
		cropped_square = im.crop((top_left_x, top_left_y, top_left_x + crop_width, top_left_y + crop_height))
	except (ValueError, IndexError) as e:

		predicted_crop_size = predict_crop_size_by_position(x, y, im_width, im_height)
		crop_width = predicted_crop_size
		crop_height = predicted_crop_size
		print(x, y)
		top_left_x = x - crop_width / 2
		top_left_y = y - crop_height / 2
		cropped_square = im.crop((top_left_x, top_left_y, top_left_x + crop_width, top_left_y + crop_height))
	cropped_square.save(img_filename)

	# write metadata
	meta = {'crop size' : predicted_crop_size,
			'sv_x'      : sv_image_x,
			'sv_y'      : sv_image_y,
			'crop_x'    : x,
			'crop_y'    : y,
			'pano yaw'  : PanoYawDeg,
			'pano id'   : pano_id
		   }

	with open(meta_filename, 'w') as metafile:
		json.dump(meta, metafile)

	return

def bulk_extract_crops(path_to_crop_csv, destination_dir):
	csv_file = open(path_to_crop_csv)
	csv_f = csv.reader(csv_file)
	counter = 0
	success = 0
	crop_fail = 0
	no_pano_fail = 0

	for row in csv_f:

		# skip header row
		if counter == 0:
			counter += 1
			continue

		pano_id = row[0]

		sv_image_x = float(row[1])
		sv_image_y = float(row[2])
		label_type = int(row[3])
		photographer_heading = float(row[4])

		pano_img_path = os.path.join(path_to_gsv_scrapes, pano_id[:2], pano_id + ".jpg")

		pano_yaw_deg = 180 - photographer_heading

		# Extract the crop
		if os.path.exists(pano_img_path):
			counter += 1
			destination_folder = os.path.join(destination_dir, str(label_type))
			if not os.path.isdir(destination_folder):
				os.makedirs(destination_folder)

			destination_basename = '{}crop{},{}'.format(pano_id, sv_image_x, sv_image_y)

			crop_destination = os.path.join(destination_dir, str(label_type), destination_basename)
			try:
				make_single_crop(pano_id, sv_image_x, sv_image_y, pano_yaw_deg, crop_destination)
				print "Successfully extracted crop to {}".format( destination_basename )
				success += 1
			except Exception as e:
				print "Cropping {} at {},{} failed.".format(pano_id, sv_image_x, sv_image_y)
				print e
				crop_fail += 1
		else:
			no_pano_fail += 1
			print "Panorama image not found for {} at {}".format(pano_id, pano_img_path)

	print("Finished.")
	print str(no_pano_fail) + " extractions failed because panorama image was not found."
	print str(crop_fail) + " extractions failed because metadata was not found."
	print "{} crops extracted successfully.".format(success)