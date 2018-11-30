import os
import csv
from GSV import GSVImage
from PIL import Image, ImageDraw
try:
    from xml.etree import cElementTree as ET
except ImportError, e:
    from xml.etree import ElementTree as ET

# This script takes a file containing null crops
# generated using create_null_crops.py
# and crops those panos into output_dir

gsv_pano_path = "/vagrant/panos_drive_full/scrapes_dump/"
input_csv = '/vagrant/random_null_crops.csv'
# format is [pano_id, x, y, type, crop_size]

output_dir = '/vagrant/null_crops/'

def extract_panoyawdeg(path_to_metadata_xml):
    pano = {}
    pano_xml = open(path_to_metadata_xml, 'rb')
    tree = ET.parse(pano_xml)
    root = tree.getroot()
    for child in root:
        if child.tag == 'projection_properties':
            pano[child.tag] = child.attrib
    print(pano['projection_properties']['pano_yaw_deg'])

    return pano['projection_properties']['pano_yaw_deg']

def round_str_to_int(str):
	return int( round( float(str) ) )


def make_single_crop(path_to_image, sv_image_x, sv_image_y, PanoYawDeg, crop_size, output_filename):
    im_width = GSVImage.GSVImage.im_width
    im_height = GSVImage.GSVImage.im_height
    im = Image.open(path_to_image)
    draw = ImageDraw.Draw(im)
    # sv_image_x = sv_image_x - 100
    x = ((float(PanoYawDeg) / 360) * im_width + sv_image_x) % im_width
    y = im_height / 2 - sv_image_y

    # Crop rectangle around label
    cropped_square = None
    crop_width = crop_size
    crop_height = crop_size
    print(x, y)
    top_left_x = x - crop_width / 2
    top_left_y = y - crop_height / 2
    cropped_square = im.crop((top_left_x, top_left_y, top_left_x + crop_width, top_left_y + crop_height))

    cropped_square.save(output_filename)

    return


##### MAIN #####


def main():
	with open(input_csv, 'r') as crop_file:
		reader = csv.reader(crop_file)

		crop_count =0
		fail_count = 0

		for row in reader:
			pano_id, x, y, f_type, crop_size = row
			x = float(x)
			y = float(y)
			f_type = int(f_type)
			crop_size = int(round(float(crop_size)))

			pano_img_path = os.path.join(gsv_pano_path, pano_id[:2], pano_id + ".jpg")
			pano_xml_path = os.path.join(gsv_pano_path, pano_id[:2], pano_id + ".xml")

			if os.path.exists(pano_xml_path):
				pano_yaw_deg = float(extract_panoyawdeg(pano_xml_path))
			else:
				print "Couldn't find XML for pano {}. Skipping.".format(pano_id)
				print "Looked in {}\n".format(pano_xml_path)
				fail_count += 1 
				continue

			output_filename = os.path.join(output_dir, "null" + str(crop_count) + ".jpg")

			print pano_id

			if os.path.exists(pano_img_path):
				print "Attempting to crop from pano {}".format(pano_id)
				make_single_crop(pano_img_path, x, y, pano_yaw_deg, crop_size, output_filename)

				crop_count += 1
			else:
				print "Pano {} not found. Skipping.".format(pano_id)
				print "Looked in {}\n".format(pano_img_path)
				fail_count += 1

			# uncomment for testing:
			#if crop_count + fail_count > 10: break

	print "Finished."
	print "{} crops succeeded\n{} crops failed.".format(crop_count, fail_count)



main()