import torch.utils.data as data

from PIL import Image

import os
import os.path
import sys

import numpy as np
import json
import torch

from collections import defaultdict
import random
import shutil

source_dataset = 'new_ds_exports/'
destin_dataset = 'mini_ds/'

samples_per_class = 100


def make_dataset(dir, class_to_idx):
    images = defaultdict(list)
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            sample_roots = set() # not including extensions here
            for fname in sorted(fnames):
                basename, ext  = os.path.splitext(fname)
                if ext in ('.jpg', '.json'):
                    sample_roots.add(basename)
            for basename in sample_roots:
                    img_path  = os.path.join(root, basename + '.jpg')
                    meta_path = os.path.join(root, basename + '.json')
                    item = (img_path, meta_path)
                    if os.path.exists(img_path) and os.path.exists(meta_path):
                        images[target].append(item)
                    if not os.path.exists(img_path):
                        print( "Couldn't find img {}".format(img_path) )
                    if not os.path.exists(meta_path):
                        print( "Couldn't find meta {}".format(meta_path) )

    return images

def find_classes(dir):
    """
    Finds the class folders in a dataset.
    Args:
        dir (string): Root directory path.
    Returns:
        tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
    Ensures:
        No class is a subdirectory of another.
    """
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def downsample_dataset(source_dataset, destin_dataset, samples_per_class):
	''' for each class in source_dataset, copy samples_per_class random samples
		to destin_dataset
	'''
	print "Loading source dataset..."
	classes, class_to_idx = find_classes(source_dataset)
	images = make_dataset(source_dataset, class_to_idx)
	downsampled = {}
	total_samples = sum( map(lambda x: len(x), images.values()) )
	print "Loaded {} samples from source.".format(total_samples)
	print "\t{:<20} {}".format("Class", "Count")
	print "\t" + ((20+len("Count"))*'-')
	for label, items in images.iteritems():
		print "\t{:<20} {}".format(label, len(items))
		samples = random.sample(items, samples_per_class)
		downsampled[label] = samples

	print "Writing new dataset to {}".format(destin_dataset)
	if not os.path.isdir(destin_dataset):
		os.makedirs(destin_dataset)
	for c in classes:
		class_dir = os.path.join(destin_dataset, c)
		if not os.path.isdir( class_dir ):
			os.makedirs(class_dir)

	for label, items in downsampled.iteritems():
		print "Copying {} samples for {} class.".format( len(items), label )
		for item in items:
			for file in item:
				base = os.path.basename(file)
				dest = os.path.join(destin_dataset, label, base)
				#print "Copying {} to {}".format(file, dest)
				shutil.copyfile(file, dest)

for phase in ('train', 'test'):
	src  = os.path.join(source_dataset, phase)
	dest = os.path.join(destin_dataset, phase)
	downsample_dataset(src, dest, samples_per_class)