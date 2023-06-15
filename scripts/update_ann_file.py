import pandas as pd

label_map = {
    'Church bell': 'Church_bell', 
    'Male speech, man speaking': 'Male_speech',
    'Bark': 'Bark',
    'Fixed-wing aircraft, airplane': 'airplane',
    'Race car, auto racing': 'Race_car',
    'Female speech, woman speaking': 'Female_speech',
    'Helicopter': 'Helicopter',
    'Violin, fiddle': 'Violin',
    'Flute': 'Flute',
    'Ukulele': 'Ukulele',
    'Frying (food)': 'Frying',
    'Truck': 'Truck',
    'Shofar': 'Shofar',
    'Motorcycle': 'Motorcycle',
    'Chainsaw': 'Chainsaw',
    'Acoustic guitar': 'Acoustic_guitar',
    'Train horn': 'Train_horn',
    'Clock': 'Clock',
    'Banjo': 'Banjo',
    'Goat': 'Goat',
    'Baby cry, infant cry': 'Baby_cry',
    'Bus': 'Bus',
    'Cat': 'Cat',
    'Horse': 'Horse',
    'Toilet flush': 'Toilet_flush',
    'Rodents, rats, mice': 'Rodents',
    'Accordion': 'Accordion',
    'Mandolin': 'Mandolin'
    }

files = ['trainSet', 'testSet', 'valSet']
for file in files:
    read_path = f'/mnt/data/datasets/AVE_Dataset/{file}.txt'
    df = pd.read_csv(read_path, delimiter='&', header=None, names=['category', 'video_id', 'quality', 'start_time', 'end_time'])
    df = df[df['video_id']!='VWi2ENBuTbw']
    df['label'] = df['category'].map(label_map)

    path = f'/mnt/user/saksham/data_distill/data/labels/{file}.csv'
    df[['video_id', 'start_time', 'end_time', 'label']].to_csv(path, header=False, index=False) 