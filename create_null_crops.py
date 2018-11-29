"""
** Various helper scripts for getting Project Sidewalk nn training examples **
Note: this needs to be in same directory as GSVImage library (available in sidewalk-panorama-tools repo)
"""

# *****************************************
# Update paths below                      *
# *****************************************

# Path to crop sizes
crop_size_path = "/vagrant/resources/fake_crop_sizes.csv"
# Path to random crops
random_crops_path = "/vagrant/resources/random_crops.csv"
# Path to panos dict
panos_dict_path = "/vagrant/resources/panos_dict.json"

import csv
import GSVImage
import json
import numpy as np
#import matplotlib.pyplot as plt

#plt.style.use('seaborn')


# write pano_id, x_center, y_center, label type, crop_size to csv
def write_to_csv(vals_arr, filename):
    with open(filename, 'a') as f:
        cwriter = csv.writer(f, delimiter=',')
        cwriter.writerow(vals_arr)


def read_from_csv(filename):
    read_arr = []
    with open(filename, 'r') as f:
        creader = csv.reader(f, delimiter=',')
        for row in creader:
            read_arr.append(row)
    return read_arr


# input is dict d
def write_to_json(d, filename):
    with open(filename, 'w') as f:
        json.dump(d, f)


# input format is [pano_id, x_center, y_center, label type, crop_size]
def create_pano_dict(data): 
    d = {}
    for crop in data:
        #print "crop", crop
        if crop[0] not in d.keys():
            # pano_id: [((center), crop_size, label_type)]
            #print "new pano"
            d[crop[0]] = [((int(crop[1]), int(crop[2])), float(crop[4]), crop[3])]
        else:
            #print "same pano"
            d[crop[0]] = d[crop[0]] + [((int(crop[1]), int(crop[2])), float(crop[4]), crop[3])]
            #print d[crop[0]]
    return d


def size_dist_gauss(size_array):
    x_max = np.nanmax(size_array)
    x_min = np.nanmin(size_array)
    diff = x_max - x_min 
    #print diff
    # very crude approx- TODO fix later
    stdv = np.sqrt(diff/2)
    center = np.nanmean(size_array)
    return center, stdv


# Other crops format: ((x, y), size, label)
def random_coords(other_crops, x_range, y_range, crop_size):
    im_width = GSVImage.GSVImage.im_width
    im_height = GSVImage.GSVImage.im_height
    (x_low, x_high) = x_range
    (y_low, y_high) = y_range
    
    avoid = False
    while not avoid:
        x = np.random.uniform(x_low, x_high)
        y = np.random.uniform(y_low, y_high)
        #print "tried (x,y):", (x, y)
        if ((x + crop_size/2 < im_width) and (x - crop_size/2 > 0)):
            if ((y + crop_size/2 < im_height/2) and (y - crop_size/2 > -1 * im_height/2)):
                all_nan = True
                for box in other_crops:
                    if not np.isnan(box[1]):
                        all_nan = False
                        diff = np.array([x - box[0][0], y - box[0][1]])
                        dist = np.linalg.norm(diff) 
                        if ((dist > crop_size) and (dist > box[1])):
                            avoid = True
                            #print "success"
                        else:
                            break
                            #print "fail"
                if all_nan:
                    avoid = True
    return x, y


# Esther: we could use this on the null crops too?
def predict_crop_size_by_position(x, y, im_width, im_height):
    print("Predicting crop size by panorama position")
    dist_to_center = math.sqrt((x - im_width / 2) ** 2 + (y - im_height / 2) ** 2)
    # Calculate distance from point to center of left edge
    dist_to_left_edge = math.sqrt((x - 0) ** 2 + (y - im_height / 2) ** 2)
    # Calculate distance from point to center of right edge
    dist_to_right_edge = math.sqrt((x - im_width) ** 2 + (y - im_height / 2) ** 2)

    min_dist = min([dist_to_center, dist_to_left_edge, dist_to_right_edge])

    crop_size = (4.0 / 15.0) * min_dist + 200

    #logging.info("Depth data unavailable; using crop size " + str(crop_size))

    return crop_size


############## Main ##############

# Params
N = 10000
#N = 10
NULL_ID = 7


#data = read_from_csv(crop_size_path) 

# create crop size array
#crop_sizes = [] 
#for line in data:
#    if len(line) > 1: 
#        size = float(line[4])
#        crop_sizes.append(size)
#    else:
#        data.remove(line)

#crop_size_arr = np.array(crop_sizes)

## create dictionary format panos
#crop_dict = create_pano_dict(data)
#write_to_json(crop_dict, panos_dict_path)
#print "crop dict written to json"

with open(panos_dict_path) as j:
    crop_dict = json.load(j)

pano_keys = crop_dict.keys()

im_height = GSVImage.GSVImage.im_height
im_width = GSVImage.GSVImage.im_width

y_min = -1 * im_height/2
y_max = im_height/2

y_top = 0
y_bot = 0

crop_sizes = [] 
for e in xrange(len(pano_keys)):
    for crop in crop_dict[pano_keys[e]]: 
        crop_sizes.append(float(crop[1]))
        bot = int(crop[0][1]) - float(crop[1])/2
        top = int(crop[0][1]) + float(crop[1])/2
        if (y_top < top):
            y_top = top
        if (y_bot > bot):
            y_bot = bot
# compute size dist
crop_size_arr = np.array(crop_sizes)
size_gauss_mean, size_gauss_stdv = size_dist_gauss(crop_size_arr)
print "mean", size_gauss_mean
print "stdv", size_gauss_stdv

for i in range(N):
    # generate gaussian noise
    size_noisy = np.random.normal(size_gauss_mean, size_gauss_stdv)
    #print "size", size_noisy
    pano_id = np.random.choice(pano_keys)
    #print "pano", pano_id
    existing_crops = crop_dict[pano_id]
    #print "existing crops", existing_crops

    #TODO define x_range, y_range
    x_range = (0, im_width)
    y_range = (y_bot, y_top)

    center_x, center_y = random_coords(existing_crops, x_range, y_range, size_noisy)
    #print "loc", (center_x, center_y)

    # output format is [pano_id, x_center, y_center, label type, crop_size]
    out = [pano_id, center_x, center_y, NULL_ID, size_noisy] 
    write_to_csv(out, random_crops_path) 
print "written to csv"

