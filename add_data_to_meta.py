# Galen Weld, April 2019
# Use this to add metadata to existing sidecars

import sys
label_intersection_dir = '/mnt/c/Users/gweld/sidewalk/label-intersection-proximity'
sys.path.append( label_intersection_dir )
from intersection_proximity import compute_proximity

from GSVutils.utils import add_metadata, extract_pano_lat_lng

from math import sin, cos, atan2, radians, degrees, sqrt


# DEFINE THE CENTER OF WASHINGTON DC HERE
dc_lat, dc_lng = 38.8977, -77.0366
# this is the middle of the White House


def calc_bearing_and_distance_between_points(lat1, lng1, lat2=dc_lat, lng2=dc_lng):
	''' returns a tuple of the bearing and distance between
		two latitude and longitude points'''
	lat1 = radians(lat1)
	lng1 = radians(lng1)
	lat2 = radians(lat2)
	lng2 = radians(lng2)

	delta_lat = abs( lat2-lat1 )
	delta_lng = abs( lng2-lng1 )

	# let's calculate the bearing
	# https://www.igismap.com/formula-to-find-bearing-or-heading-angle-between-two-points-latitude-longitude/
	x = cos(lat2) * sin( delta_lng )
	y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(delta_lng)
	#print("x:{},y:{}".format(x,y))
	bearing = degrees( atan2(x,y) )

	# let's calculate the distance
	# https://andrew.hedges.name/experiments/haversine/
	R = 3961 #miles
	# this value is approximately correct for the area around Washington DC
	a = (sin(delta_lat/2))**2 + cos(lat1) * cos(lat2) * (sin(delta_lng/2))**2
	c = 2 * atan2( sqrt(a), sqrt(1-a) )
	dist = R * c #where R is the radius of the Earth

	return bearing, dist


def helper(input_dict):
	#print(input_dict)
	pano_id = input_dict[u'pano id']

	lat, lng = extract_pano_lat_lng(pano_id)
	input_dict[u'latitude']  = lat
	input_dict[u'longitude'] = lng

	distance, middleness = compute_proximity(lat, lng)
	input_dict[u'dist to intersection'] = distance
	input_dict[u'block middleness']     = middleness

	bearing, distance = calc_bearing_and_distance_between_points(lat, lng)
	input_dict[u'bearing to cbd'] = bearing
	input_dict[u'dist to cbd']    = distance
	return input_dict

add_metadata('/mnt/c/Users/gweld/sidewalk/sidewalk_ml/baby_ds', helper)

# us naval observatory
#bearing, dist = calc_bearing_and_distance_between_points(38.9220, -77.0668)
#print( "bearing:{}, dist:{}".format(bearing, dist) )