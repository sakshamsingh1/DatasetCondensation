'''
Script to get channel wise mean and std of the dataset

Note: VideoID = 'VWi2ENBuTbw' has start_time = end_time = 0. So, we skip it.

Results:
Mean = [0.42555316953448996, 0.3969898255671646, 0.3702121671694102]
Std = [0.2291990694620772, 0.22453432182841992, 0.22165034052220153]

'''

from PIL import Image
import numpy as np
import pandas as pd
import os
from glob import glob
from tqdm import tqdm

base_path = '/mnt/data/datasets/AVE_Dataset'
label_path = os.path.join(base_path, 'trainSet.txt')

frame_path = '/mnt/user/saksham/data/frames'


# read df with no header and columns as category, video_id, quality, start_time, end_time
df = pd.read_csv(label_path, delimiter='&', header=None, names=['category', 'video_id', 'quality', 'start_time', 'end_time'])

videos = list(df['video_id'].unique())

def get_video_met(v_id):
    avg_0, avg_1, avg_2 = [], [], []
    std_0, std_1, std_2 = [], [], []

    def get_mean(img, chan):
        return np.mean(img[:,:,chan])

    def get_std(img, chan):
        return np.std(img[:,:,chan])

    video_path = os.path.join(frame_path, v_id)
    frames = glob(os.path.join(video_path, '**/*.jpg'))
    frames = [np.array(Image.open(frame))/255.0 for frame in frames]
    
    for frame in frames:
        avg_0.append(get_mean(frame, 0))
        avg_1.append(get_mean(frame, 1))
        avg_2.append(get_mean(frame, 2))

        std_0.append(get_std(frame, 0))
        std_1.append(get_std(frame, 1))
        std_2.append(get_std(frame, 2))
    
    return np.mean(np.array(avg_0)), np.mean(np.array(avg_1)), np.mean(np.array(avg_2)), np.mean(np.array(std_0)), np.mean(np.array(std_1)), np.mean(np.array(std_2))


avg_0, avg_1, avg_2 = [], [], []
std_0, std_1, std_2 = [], [], []

for vid in tqdm(videos):
    a0, a1, a2, s0, s1, s2 = get_video_met(vid)
    if np.isnan(a0):
        continue
    avg_0.append(a0)
    avg_1.append(a1)
    avg_2.append(a2)
    std_0.append(s0)
    std_1.append(s1)
    std_2.append(s2)

print(np.mean(np.array(avg_0)), np.mean(np.array(avg_1)), np.mean(np.array(avg_2)), np.mean(np.array(std_0)), np.mean(np.array(std_1)), np.mean(np.array(std_2)))