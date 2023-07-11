import h5py
import numpy as np
import csv
import os
import torchvision
from tqdm import tqdm

base_path = '/mnt/user/saksham/data_distill/DatasetCondensation/data'
save_path = f'{base_path}/image_h5.hdf5'
fps = 8

frame_path = '/mnt/user/saksham/data/frames_224'
list_file = '/mnt/user/saksham/data_distill/data/labels/trainSet.csv'
list_sample = []
for row in csv.reader(open(list_file, 'r'), delimiter=','):
    if len(row) < 2:
        continue
    list_sample.append(row)

created_groups = set()
with h5py.File(save_path, "w") as f:
    for row in tqdm(list_sample):
        vid, start, end, _ = row
        start, end = int(start), int(end)
        for i in range(start, end):
            for j in range(1, fps+1):
                if f'{vid}_{i}_0{j}' in created_groups:
                    continue
                file = os.path.join(frame_path, vid, str(i), f'{vid}_{i}_0{j}.jpg')
                img = torchvision.io.read_image(file)
                img = img.numpy()
                grp = f.create_dataset(f'{vid}_{i}_0{j}', data=img)
                created_groups.add(f'{vid}_{i}_0{j}')

