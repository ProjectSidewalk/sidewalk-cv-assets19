import os, csv, random

path_to_db_export  = "/mnt/c/Users/gweld/sidewalk/minus_onboard.csv"

num_to_pick = 50

#############################
panos = []
with open(path_to_db_export, 'r') as csvfile:
	reader = csv.reader(csvfile)

	for row in reader:
		panos.append( row[0] )

picked = random.sample(panos, num_to_pick)


for p in picked:
	print p