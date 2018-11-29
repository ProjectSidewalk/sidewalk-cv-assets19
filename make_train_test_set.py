import os
import csv
import random
from collections import defaultdict

crops_dir = os.path.join('..', 'crops')
# directory containing crops from CropRunner.py

gs_path = "gs://sidewalk_crops_subset/imgs/"
# path on google storage where imgs will be located

train_ratio = .9
# the fraction of images that will be put in the train set
# 1-train_ratio is the fraction put in the validation set
# eg train_ratio = 0.9 gives a 90%/10% split

output_path = '.'
# will create three files in this directory:
# train_set.csv
# eval_set.csv
# labels.txt

labels = ['curb_ramp','missing_ramp','obstruction','surface_problem','no_sidewalk','occlusion','other']
train_counts = defaultdict(int)
test_counts = defaultdict(int)

with open(os.path.join(output_path,"train_set.csv"), 'wb') as train_file,\
	 open(os.path.join(output_path,"eval_set.csv"), 'wb') as test_file:

	train_writer = csv.writer(train_file)
	test_writer  = csv.writer(test_file)

	count_all = 0
	for root, _, files in os.walk(crops_dir):
		try:
			label = labels[int(root[-1])-1]
		except:
			label = "no label"
		print "Now processing {}".format(root)

		count_this_type = 0
		for file in files:
			#if count_this_type >= 30: break

			_, extension = os.path.splitext(file)
			if extension != '.jpg': continue

			new_path = gs_path + file

			is_test = random.random() > train_ratio
			writer = test_writer if is_test else train_writer
			counter = test_counts if is_test else train_counts

			writer.writerow((new_path, label))
			counter[label] += 1

			count_this_type += 1
			count_all += 1
		print "\n"

with open(os.path.join(output_path, 'labels.txt'), 'w') as labels_file:
	for label in labels:
		labels_file.write(label+'\n')

print "Finished processing data"
print "{} images processed succesfully\n".format(count_all)

s = "{:15} {:10d} {:10d}"
print "{:15} {:>10} {:>10}".format("LABEL", "TRAIN", "TEST")
for label in labels:
	print s.format(label, train_counts[label], test_counts[label])
print s.format("Total", sum(train_counts.values()), sum(test_counts.values()))