import csv
import matplotlib.pyplot as plt
import numpy as np
import os.path

path_to_csv = os.path.join('..', 'all_points.csv')


get_label_name = ["Curb Ramp",
				  "Missing Curb Ramp",
				  "Obstruction",
				  "Surface Problem",
				  "No Sidewalk",
				  "Occlusion",
				  "Other"]


xs = []
ys = []
lb = []

with open(path_to_csv) as csvfile:
	csvreader = csv.reader(csvfile, delimiter=',')

	for row in csvreader:
		xs.append(float(row[1])) # sv_image_x
		ys.append(float(row[2])) # sc_image_y
		lb.append(int(  row[3])-1) # label_type_id

print "xs span {} - {}".format(min(xs), max(xs))
print "ys span {} - {}".format(min(ys), max(ys))
print "with {} points counted".format(len(xs))

r = ((0, 13312),(-3328, 3328))

def plot_all():
	plt.subplot(2,1,1)
	heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=(80,40), range=r)
	extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
	plt.imshow(np.flipud(heatmap.T), extent=extent, cmap='Greys', interpolation='bicubic')
	plt.title("Bins (80,40) with bicubic interpolation")
	plt.colorbar()

	plt.subplot(2,1,2)
	heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=(500,250), range=r)
	extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
	plt.imshow(np.flipud(heatmap.T), extent=extent, cmap='Greys', interpolation='none')
	plt.title("Bins (500,250) without interpolation")
	plt.colorbar()

	plt.tight_layout()
	plt.show()
	return

def plot_each(include):
	plot_idx = 1
	for label_id, label in enumerate(get_label_name):
		if label_id not in include: continue

		# gets just points of label_id
		filtered_pts = filter(lambda x: x[2]==label_id, zip(xs, ys, lb))
		print "Filtering {}({}) \t found {} pts".format(label, label_id, len(filtered_pts))
		filt_xs, filt_ys, _ = zip(*filtered_pts)

		print "\txs span {} - {}".format(min(filt_xs), max(filt_xs))
		print "\tys span {} - {}".format(min(filt_ys), max(filt_ys))

		heatmap, xedges, yedges = np.histogram2d(filt_xs, filt_ys, bins=(80,40), range=r)
		extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

		plt.subplot(len(include), 1, plot_idx)
		plt.imshow(np.flipud(heatmap.T), extent=extent, cmap='Greys', interpolation='bicubic')
		plt.title("Locations of {}".format(label))
		plt.colorbar()

		plot_idx += 1


	plt.tight_layout()
	plt.show()


plot_each((1, 2, 3, 4))