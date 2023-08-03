'''
Script to create input .pt file for training.
It consists of these keys:
- 'classes'
- 'audio_train'
- 'audio_val'
- 'audio_test'
- 'labels_train'
- 'labels_val'
- 'labels_test'
'''

import os
import pandas as pd
import torch
from tqdm import tqdm
from scipy.io import wavfile

from audio_utils import waveform_to_examples

base_path = '/mnt/data/datasets/AVE_Dataset'
label_train_path = os.path.join(base_path, 'trainSet.txt')
label_val_path = os.path.join(base_path, 'valSet.txt')
label_test_path = os.path.join(base_path, 'testSet.txt')
label_idx_to_map_path = '/mnt/user/saksham/data_distill/data/labels/cat_id_map.pt'

audio_path = '/mnt/user/saksham/data_distill/data/audio'

input_data = {}
correct_label = {
'Church bell':'Church_bell',
'Male speech, man speaking': 'Male_speech',
'Bark':'Bark',
'Fixed-wing aircraft, airplane':'airplane',
'Race car, auto racing':'Race_car',
'Female speech, woman speaking':'Female_speech',
'Helicopter':'Helicopter',
'Violin, fiddle':'Violin',
'Flute':'Flute',
'Ukulele': 'Ukulele', 
'Frying (food)': 'Frying',
'Truck': 'Truck', 
'Shofar': 'Shofar',
'Motorcycle': 'Motorcycle',
'Acoustic guitar': 'Acoustic_guitar', 
'Train horn': 'Train_horn', 
'Clock': 'Clock', 
'Banjo': 'Banjo',
'Goat': 'Goat', 
'Baby cry, infant cry': 'Baby_cry', 
'Bus': 'Bus', 
'Chainsaw': 'Chainsaw', 
'Cat': 'Cat', 
'Horse': 'Horse',
'Toilet flush': 'Toilet_flush', 
'Rodents, rats, mice': 'Rodents', 
'Accordion': 'Accordion', 
'Mandolin': 'Mandolin'
}
label_to_idx = torch.load(label_idx_to_map_path)

class_map = {}

df_train = pd.read_csv(label_train_path, delimiter='&', header=None, names=['category', 'video_id', 'quality', 'start_time', 'end_time'])
df_train = df_train[df_train['video_id']!='VWi2ENBuTbw']
df_train = df_train[df_train['video_id']!='0-I1-DOC-r8']
df_train = df_train[df_train['video_id']!='TTYevyM_tUw']
df_train = df_train[df_train['video_id']!='K6F1sogt46U']
df_train = df_train[df_train['video_id']!='MWFQerde_h8']
df_train = df_train[df_train['video_id']!='GmetnCLxFHE']

df_val = pd.read_csv(label_val_path, delimiter='&', header=None, names=['category', 'video_id', 'quality', 'start_time', 'end_time'])
df_test = pd.read_csv(label_test_path, delimiter='&', header=None, names=['category', 'video_id', 'quality', 'start_time', 'end_time'])

def get_label_idx(label):
    label = correct_label[label]
    return label_to_idx[label]

def get_spectrogram(file):
    sr, wav_data = wavfile.read(file)
    wav_data = wav_data / 32768.0

    spec = waveform_to_examples(wav_data, sr)
    spec = torch.from_numpy(spec)
    return spec


train_count = 0
val_count = 0
test_count = 0

for index, row in df_train.iterrows():
    start_time = row['start_time']
    end_time = row['end_time']
    frame_count = end_time - start_time
    train_count += frame_count

for index, row in df_val.iterrows():
    start_time = row['start_time']
    end_time = row['end_time']
    frame_count = end_time - start_time
    val_count += frame_count

for index, row in df_test.iterrows():
    start_time = row['start_time']
    end_time = row['end_time']
    frame_count = end_time - start_time
    test_count += frame_count     
print(f'train:{train_count}, val:{val_count}, test:{test_count}')

audio_train = torch.Tensor(train_count, 1, 96, 64)
labels_train = torch.zeros(train_count, dtype=torch.int32)

print('Creating train data...')
curr_count = 0
# import pdb; pdb.set_trace()
for index, row in tqdm(df_train.iterrows()):
    video_id = row['video_id']
    category = row['category']
    start_time = row['start_time']
    end_time = row['end_time']

    for curr_sec in range(start_time, end_time):
        file = os.path.join(audio_path, video_id, f'{video_id}_{curr_sec}.wav')
        spec = get_spectrogram(file)
        audio_train[curr_count] = spec
        labels_train[curr_count] = get_label_idx(category)
        curr_count += 1

print(f'Final count: {curr_count}')

audio_val = torch.Tensor(val_count, 1, 96, 64)
labels_val = torch.zeros(val_count, dtype=torch.int32)

print('Creating val data...')
curr_count = 0
for index, row in tqdm(df_val.iterrows()):
    video_id = row['video_id']
    category = row['category']
    start_time = row['start_time']
    end_time = row['end_time']

    for curr_sec in range(start_time, end_time):
        file = os.path.join(audio_path, video_id, f'{video_id}_{curr_sec}.wav')
        spec = get_spectrogram(file)
        audio_val[curr_count] = spec
        labels_val[curr_count] = get_label_idx(category)
        curr_count += 1


print('Creating test data...')
audio_test = torch.Tensor(test_count, 1, 96, 64)
labels_test = torch.zeros(test_count, dtype=torch.int32)

curr_count = 0
for index, row in tqdm(df_test.iterrows()):
    video_id = row['video_id']
    category = row['category']
    start_time = row['start_time']
    end_time = row['end_time']

    for curr_sec in range(start_time, end_time):
        file = os.path.join(audio_path, video_id, f'{video_id}_{curr_sec}.wav')
        spec = get_spectrogram(file)
        audio_test[curr_count] = spec
        labels_test[curr_count] = get_label_idx(category)
        curr_count += 1

input_data['classes'] = list(label_to_idx.keys())
input_data['images_train'] = audio_train
input_data['images_val'] = audio_val
input_data['labels_train'] = labels_train
input_data['labels_val'] = labels_val
input_data['images_test'] = audio_test
input_data['labels_test'] = labels_test

torch.save(input_data, '/mnt/user/saksham/data_distill/DatasetCondensation/data/ave_audio.pt')