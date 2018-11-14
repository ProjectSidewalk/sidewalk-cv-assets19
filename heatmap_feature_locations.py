import csv
import matplotlib.pyplot as plt
import numpy as np
import os.path

path_to_csv = os.path.join('..', 'all_points.csv')

xs = []
ys = []


with open(path_to_csv) as csvfile:
	csvreader = csv.reader(csvfile, delimiter=',')

	for row in csvreader:
		xs.append(float(row[1])) # sv_image_x
		ys.append(float(row[2])) # sc_image_y

print "xs span {} - {}".format(min(xs), max(xs))
print "ys span {} - {}".format(min(ys), max(ys))
print "with {} points counted".format(len(xs))

r = ((0, 13312),(-3328, 3328))

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