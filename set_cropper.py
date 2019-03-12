# Galen Weld, Feb 2019
# Use this to extract sliding window datasets crops and sidecars
# from CSVs produced using make_train_test_sets.ipynb
# row in the csv should be
# Pano ID, SV_x, SV_y, Label, Photog Heading, Heading, Label ID 

from GSVutils.utils import bulk_extract_crops


bulk_extract_crops('sliding_window_crops_to_make.csv', 'sliding_window/new_sliding_window_crops/')
