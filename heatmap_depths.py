from collections import defaultdict
import math
import csv
import os
from numpy import loadtxt
try:
	from xml.etree import cElementTree as ET
except ImportError, e:
	from xml.etree import ElementTree as ET

path_to_gsv_scrapes  = "/mnt/c/Users/gweld/sidewalk/panos_drive_full/scrapes_dump/"
pano_db_export = '/mnt/c/Users/gweld/sidewalk/minus_onboard.csv'

label_from_int = ('Curb Cut', 'Missing Cut', 'Obstruction', 'Sfc Problem')

gsv_image_width = 13312
gsv_image_height = 6656

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


def depth(path_to_image, sv_image_x, sv_image_y, pano_yaw_deg, path_to_depth):
	x = ((float(pano_yaw_deg) / 360) * gsv_image_width + sv_image_x) % gsv_image_width
	y = gsv_image_height / 2 - sv_image_y

	with open(path_to_depth, 'rb') as f:
		depth = loadtxt(f)

	depth_x = depth[:, 0::3]
	depth_y = depth[:, 1::3]
	depth_z = depth[:, 2::3]

	val_x, val_y, val_z = interpolated_3d_point(x, y, depth_x, depth_y, depth_z)

	distance = math.sqrt(val_x ** 2 + val_y ** 2 + val_z ** 2)
	return distance

def depth_generator():
	with open(pano_db_export) as csvfile:
		reader = csv.reader(csvfile)

		for row in reader:
			pano_id = row[0]

			sv_image_x = float(row[1])
			sv_image_y = float(row[2])
			label_type = int(row[3])
			photographer_heading = float(row[4])
			heading = float(row[5])
			label_id = int(row[7])

			if label_type-1 >= len(label_from_int):
				print "skipping label int {}".format(label_type)
				continue
			label_name = label_from_int[label_type-1]

			# Extract Yaw from metadata xml file
			pano_xml_path = os.path.join(path_to_gsv_scrapes, pano_id[:2], pano_id + ".xml")
			pano_img_path = os.path.join(path_to_gsv_scrapes, pano_id[:2], pano_id + ".jpg")
			pano_depth_path = os.path.join(path_to_gsv_scrapes, pano_id[:2], pano_id + ".depth.txt")

			try:
				if (os.path.exists(pano_xml_path)):
					pano_yaw_deg = float(extract_panoyawdeg(pano_xml_path))
				else:
					print "Skipping {} due to missing XML data".format(pano_id)
					continue
			except KeyError as e:
				print "Invalid XML data, skipping {}".format(pano_id)
				continue

			try:
				d = depth(pano_img_path, sv_image_x, sv_image_y, pano_yaw_deg, pano_depth_path)
			except IndexError as e:
				print "Error interpolating depth data for {}, skipping.".format(pano_id)
			#print label_name, d

			yield label_name, d


