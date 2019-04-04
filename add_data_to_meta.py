# Galen Weld, April 2019
# Use this to add metadata to existing sidecars

import sys
label_intersection_dir = '/mnt/c/Users/gweld/sidewalk/label-intersection-proximity'
sys.path.append( label_intersection_dir )
from intersection_proximity import compute_proximity

from GSVutils.utils import add_metadata, extract_pano_lat_lng


def helper(input_dict):
	#print(input_dict)
	pano_id = input_dict[u'pano id']
	lat, lng = extract_pano_lat_lng(pano_id)

	input_dict[u'latitude']  = lat
	input_dict[u'longitude'] = lng

	distance, middleness = compute_proximity(lat, lng)
	print(distance, middleness)
	input_dict[u'dist to intersection'] = distance
	input_dict[u'middleness'] = middleness
	return input_dict

add_metadata('/mnt/c/Users/gweld/sidewalk/sidewalk_ml/baby_ds', helper)