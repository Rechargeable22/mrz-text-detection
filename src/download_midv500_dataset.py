import midv500

# This file is used to download the midv500 dataset. You can stop the download after a while as we only use the first
# 15 or so datasets.

# set directory for dataset to be downloaded
dataset_dir = 'data/midv500_data/'

# download and unzip the base midv500 dataset
dataset_name = "midv500"
midv500.download_dataset(dataset_dir, dataset_name)
