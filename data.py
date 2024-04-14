
# https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
# https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz

import os
import requests
import tarfile
import shutil
import random

from PIL import Image
import numpy as np
import h5py

def prepare_dataset(ratio_train):
    
    # Create a 'data' folder to store the downloaded data
    DATA_PATH = './data'
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    else:
        print("DATA_PATH already exists. Continuing to check the HDF5 files.")

    # Check if HDF5 files already exist in the 'data' folder
    existing_h5_files = [file for file in os.listdir(DATA_PATH) if file.endswith('.h5')]

    if existing_h5_files:
        print("HDF5 files already exist in the 'data' folder. Continuing to fine-tuning.")
    else:
        print("HDF5 files do not exist in the 'data' folder. Downloading and converting the data.")

        ## download dataset
        filenames = ['images.tar.gz', 'annotations.tar.gz']
        url_base = 'https://www.robots.ox.ac.uk/~vgg/data/pets/data/'

        for temp_file in filenames:
            url = url_base + temp_file
            print(url + ' ...')
            r = requests.get(url,allow_redirects=True)
            _ = open(temp_file,'wb').write(r.content)
            with tarfile.open(temp_file) as tar_obj:
                tar_obj.extractall()
                tar_obj.close()
            os.remove(temp_file)

        ## spliting and converting
        img_dir = 'images'
        seg_dir = 'annotations/trimaps'
        im_size = (224,224)
        img_h5s, seg_h5s = [], []
        for s in ["train", "val", "test"]:
            img_h5s.append(h5py.File(os.path.join(DATA_PATH,"images_{:s}.h5".format(s)), "w"))
            seg_h5s.append(h5py.File(os.path.join(DATA_PATH,"labels_{:s}.h5".format(s)), "w"))

        img_filenames = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        num_data = len(img_filenames)
        num_train = int(num_data * ratio_train)
        num_test = 0.1
        num_val = num_data - num_train - num_test
        print("Dataset Loaded: num_train: %d, num_val: %d, num_test: %d" % (num_train, num_val, num_test))

        random.seed(90)
        random.shuffle(img_filenames)

        # write all images/labels to h5 file
        for idx, im_file in enumerate(img_filenames):

            if idx < num_train:  # train
                ids = 0
            elif idx < (num_train + num_val):  # val
                ids = 1
            else:  # test
                ids = 2

            with Image.open(os.path.join(img_dir,im_file)) as img:
                img = np.array(img.convert('RGB').resize(im_size).getdata(),dtype='uint8').reshape(im_size[0],im_size[1],3)
                img_h5s[ids].create_dataset("{:06d}".format(idx), data=img)
            with Image.open(os.path.join(seg_dir,im_file.split('.')[0]+'.png')) as seg:
                seg = np.array(seg.resize(im_size).getdata(),dtype='uint8').reshape(im_size[0],im_size[1])
                seg_h5s[ids].create_dataset("{:06d}".format(idx), data=seg)

        for ids in range(len(img_h5s)):
            img_h5s[ids].flush()
            img_h5s[ids].close()
            seg_h5s[ids].flush()
            seg_h5s[ids].close()

        shutil.rmtree(img_dir)
        shutil.rmtree(seg_dir.split('/')[0]) #remove entire annatations folder

        print('Data saved in %s.' % os.path.abspath(DATA_PATH))
