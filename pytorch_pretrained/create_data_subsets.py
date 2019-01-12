import os
import random
from shutil import copyfile

crops_dir = '/mnt/c/Users/gweld/sidewalk/crops_no_mark_no_onboard/'
# this is of format class/images, where class in [1,2,3,4,5,6,7,8]

output_dir = '/mnt/c/Users/gweld/sidewalk/sidewalk_ml/pytorch_pretrained/data/medium_sidewalk'


num_train_imgs = 2000
num_val_imgs = 400


use_all_imgs = True
# if this is set to true, the counts above will be ignored
# all images will be included, partitioned according to the ratio below

partition_ratio = .8
# .8 indicates 80/20 split


labels_to_name = {'1':'ramp',
                  '2':'not_ramp',
                  '3':'obstruction',
                  '4':'sfc_prob',
                  '8':'null'}

####


train_dir = os.path.join(output_dir, "train")
val_dir   = os.path.join(output_dir, "val")

for label, name in labels_to_name.items():
	in_path    = os.path.join(crops_dir, label)

	# make these dirs if need be
	train_path = os.path.join(train_dir, name)
	val_path   = os.path.join(val_dir,   name)

	if not os.path.exists(train_path): os.makedirs(train_path)
	if not os.path.exists(val_path): os.makedirs(val_path)



	files = os.listdir(in_path)

	sample = random.sample(files, num_val_imgs + num_train_imgs)

	train_set = sample[:num_train_imgs]
	val_set   = sample[num_train_imgs:]

	print "choosing {}".format(name)
	print "train_len {}, val_len {}".format(len(train_set), len(val_set))

	for img in train_set:
		img_path = os.path.join(in_path, img)
		out_path = os.path.join(train_path, img)

		if img_path.endswith('.jpg'): copyfile(img_path, out_path)

	for img in val_set:
		img_path = os.path.join(in_path, img)
		out_path = os.path.join(val_path, img)

		if img_path.endswith('.jpg'): copyfile(img_path, out_path)