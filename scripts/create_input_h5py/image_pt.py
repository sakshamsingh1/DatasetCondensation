import numpy as np
import csv
import os
import torchvision
from tqdm import tqdm
import torch

base_path = '/mnt/user/saksham/data_distill/DatasetCondensation/data'
save_path = f'{base_path}/image_pt.pt'
fps = 8

frame_path = '/mnt/user/saksham/data/frames_224'
list_file = '/mnt/user/saksham/data_distill/data/labels/trainSet.csv'
index_map = {}
img_list = []

index = 0
for row in tqdm(csv.reader(open(list_file, 'r'), delimiter=',')):
    if len(row) < 2:
        continue
    
    vid, start, end, _ = row
    start, end = int(start), int(end)
    for i in range(start, end):
        for j in range(1, fps+1):
            file = f'{vid}_{i}_0{j}'
            if file in index_map:
                continue
            index_map[file] = index
            file = os.path.join(frame_path, vid, str(i), f'{vid}_{i}_0{j}.jpg')
            img = torchvision.io.read_image(file)
            img_list.append(img)
            index += 1

img_tensor = torch.stack(img_list)
img_map = {'image': img_tensor, 'index_map': index_map} 
torch.save(img_map, save_path)