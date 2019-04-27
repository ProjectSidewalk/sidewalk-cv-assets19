from PIL import Image
import os
import random
import shutil
from collections import defaultdict

#dir_to_scan = '/mnt/g/test'
dir_to_scan = '/mnt/f/crops_for_esther_galen/all_sidewalk'

def is_all_black(path):
	''' returns true if image is all black '''
	i = Image.open(path)
	return (not i.getbbox())


def count_blacks(dir_to_scan=dir_to_scan):
	''' counts the number of total and black images in dir_to_scan and subfolders'''
	total = 0
	black = 0
	for root, dirs, files in os.walk(dir_to_scan):
		_, folder = os.path.split(root)
		print "Starting on {}".format(folder)

		for f in files:
			_, ext = os.path.splitext(f)
			if ext != '.jpg': continue

			path = os.path.join(root, f)

			total += 1
			black += is_all_black(path)

	print "Checked {} files. Found {} black.".format(total, black)
	print "{}% black.".format(100*(float(black)/total))





#print is_all_black(os.path.join(dir_to_scan, 'black.jpg'))

#print is_all_black(os.path.join(dir_to_scan, 'good.jpg'))

count_blacks()