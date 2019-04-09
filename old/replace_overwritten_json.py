import csv
import json
import os
from shutil import copyfile

dir_of_overwritten_files = "/mnt/c/Users/gweld/sidewalk/sidewalk_ml/full_ds/test/missing_ramp/"

files_to_get = set()

for filename in os.listdir(dir_of_overwritten_files):
	_, ext = os.path.splitext(filename)
	if ext != ".json":
		# we care only about json
		continue

	filepath = os.path.join(dir_of_overwritten_files, filename)

	try:
		with open(filepath) as jsonfile:
			meta = json.load(jsonfile)

	except ValueError as e:
		files_to_get.add(filename)

print( "Missing {} files.".format(len(files_to_get)) )

with open("list_of_missing_files.txt", 'w') as missinglist:
	for file in files_to_get:
		path = os.path.join('gs://sliding_window_dataset/test/missing_ramp/', file)
		missinglist.write(path + '\n')

##
# then use
# mkdir recovered_json
# cat list_of_missing_files.txt | gsutil -m cp -I ./recovered_json
##


# once done, continue here:
count = 1

for filename in os.listdir('./recovered_json'):
	org_path = os.path.join('./recovered_json', filename)
	dest_path = os.path.join(dir_of_overwritten_files, filename)
	print("Copying {}".format(filename))
	copyfile(org_path, dest_path)
	count += 1

print( "Copied {} recovered files.".format(count) )