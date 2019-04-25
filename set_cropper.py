# Galen Weld, Feb 2019
# Use this to extract sliding window datasets crops and sidecars
# from CSVs produced using make_train_test_sets.ipynb
# row in the csv should be
# Pano ID, SV_x, SV_y, Label, Photog Heading, Heading, Label ID 

from GSVutils.utils import bulk_extract_crops


# use this to make new sliding window crops for analysis
#bulk_extract_crops('sliding_window_crops_to_make.csv', 'sliding_window/new_sliding_window_crops/')

# this to make dataset
#bulk_extract_crops('dataset_csvs/Val.csv', 'new_ds_exports/Val/')

# this to make old dataset
#bulk_extract_crops('new_old_dataset_csvs/Val.csv', '/mnt/e/old_dataset/val/')
#bulk_extract_crops('new_old_dataset_csvs/Train.csv', '/mnt/e/old_dataset/train/')

# this to make newberg centered dataset
bulk_extract_crops('new_cities/newberg-labels-researchers.csv', '/mnt/g/newberg_center_crops_researchers/', '/mnt/g/scrapes_dump_newberg')
#bulk_extract_crops('new_cities/newberg-labels.csv', '/mnt/e/newberg_center_crops_all/', '/mnt/e/scrapes_dump_newberg')

# use this to make seattle centered dataset
bulk_extract_crops('new_cities/seattle-labels-researchers.csv', '/mnt/g/seattle_center_crops_researchers/', '/mnt/g/scrapes_dump_seattle')
#bulk_extract_crops('new_cities/seattle-labels.csv', '/mnt/e/seattle_center_crops_all/', '/mnt/e/scrapes_dump_seattle')

