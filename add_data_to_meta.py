# Galen Weld, April 2019
# Use this to add metadata to existing sidecars

from GSVutils.utils import add_metadata


def helper(input_dict):
	#print(input_dict)
	input_dict[u'pano_id'] = pano_id
	return input_dict

add_metadata('/mnt/c/Users/gweld/sidewalk/sidewalk_ml/baby_ds', helper)