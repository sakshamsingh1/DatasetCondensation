import random
import os
import csv
import numpy as np
import torch
import torch.utils.data as torchdata
from torchvision import transforms
import librosa
from PIL import Image
from scipy.io import wavfile
from .audio_utils import waveform_to_examples

from . import video_transforms as vtransforms


class BaseDataset(torchdata.Dataset):
    def __init__(self, list_sample, opt, max_sample=-1, split='train', one_frame_per_video=True):
        # params
        # max_sample = 100
        self.frameRate = opt.frameRate
        self.imgSize = opt.imgSize
        self.audRate = opt.audRate
        self.num_frames = opt.num_frames
        # self.audLen = opt.audLen
        # self.audSec = 1. * self.audLen / self.audRate


        # STFT params
        # self.log_freq = opt.log_freq
        # self.stft_frame = opt.stft_frame
        # self.stft_hop = opt.stft_hop
        # self.HS = opt.stft_frame // 2 + 1
        # self.WS = (self.audLen + 1) // self.stft_hop

        self.split = split
        self.seed = opt.seed
        random.seed(self.seed)

        # initialize video transform
        self._init_vtransform()

        # list_sample can be a python list or a csv file of list
        if isinstance(list_sample, str):
            self.list_sample = []
            for row in csv.reader(open(list_sample, 'r'), delimiter=','):
                if len(row) < 2:
                    continue
                if one_frame_per_video:
                    self.list_sample.append(row)
                else: # one frame per second for start and end annotations
                    for i in range(int(row[1]), int(row[2])):
                        temp_row = [row[0], i, i, row[3]]
                        self.list_sample.append(temp_row)
        elif isinstance(list_sample, list):
            self.list_sample = list_sample
        else:
            raise('Error list_sample!')

        if self.split == 'train':
            random.shuffle(self.list_sample)
        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]

        num_sample = len(self.list_sample)
        assert num_sample > 0
        print('# samples: {}'.format(num_sample))

    def __len__(self):
        return len(self.list_sample)

    # video transform funcs
    def _init_vtransform(self):
        transform_list = []
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if self.split == 'train':
            transform_list.append(vtransforms.Resize(int(self.imgSize * 1.1), Image.BICUBIC))
            transform_list.append(vtransforms.RandomCrop(self.imgSize))
            transform_list.append(vtransforms.RandomHorizontalFlip())
        else:
            transform_list.append(vtransforms.Resize(self.imgSize, Image.BICUBIC))
            transform_list.append(vtransforms.CenterCrop(self.imgSize))

        transform_list.append(vtransforms.ToTensor())
        transform_list.append(vtransforms.Normalize(mean, std))
        transform_list.append(vtransforms.Stack())
        self.vid_transform = transforms.Compose(transform_list)

    # image transform funcs, deprecated
    def _init_transform(self):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if self.split == 'train':
            self.img_transform = transforms.Compose([
                transforms.Scale(int(self.imgSize * 1.2)),
                transforms.RandomCrop(self.imgSize),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])
        else:
            self.img_transform = transforms.Compose([
                transforms.Scale(self.imgSize),
                transforms.CenterCrop(self.imgSize),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)])

    def _load_frames(self, paths):
        frames = []
        for path in paths:
            frames.append(self._load_frame(path))
        frames = self.vid_transform(frames)
        return frames

    def _load_frame(self, path):
        img = Image.open(path).convert('RGB')
        return img

    def _stft(self, audio):
        spec = librosa.stft(
            audio, n_fft=self.stft_frame, hop_length=self.stft_hop)
        amp = np.abs(spec)
        phase = np.angle(spec)
        return torch.from_numpy(amp), torch.from_numpy(phase)

    def _load_audio_file(self, path):

        audio_raw, rate = librosa.load(path, sr=None, mono=True)
            # '''
            # audio_raw, rate = torchaudio.load(path)
            # if audio_raw.size(0)>1:
            #     audio_raw = torch.mean(audio_raw, dim=0)
            # audio_raw = audio_raw.view(-1)
            # audio_raw = audio_raw.numpy().astype(np.float32)
            # '''

        return audio_raw, rate

    def get_spectrogram(self, file):
        sr, wav_data = wavfile.read(file)
        wav_data = wav_data / 32768.0

        spec = waveform_to_examples(wav_data, sr)
        spec = torch.from_numpy(spec).type(torch.float32)
        return spec

    def _load_audio(self, path, center_timestamp, nearest_resample=False):
        audio = np.zeros(self.audLen, dtype=np.float32)

        # silent
        if path.endswith('silent'):
            return audio

        # load audio
        audio_raw, rate = self._load_audio_file(path)

        # repeat if audio is too short
        if audio_raw.shape[0] < rate * self.audSec:
            n = int(rate * self.audSec / audio_raw.shape[0]) + 1
            audio_raw = np.tile(audio_raw, n)

        # resample
        if rate > self.audRate:
            if nearest_resample:
                audio_raw = audio_raw[::rate//self.audRate]
            else:
                audio_raw = librosa.resample(audio_raw, rate, self.audRate)

        # crop N seconds
        len_raw = audio_raw.shape[0]
        center = int(center_timestamp * self.audRate)
        start = max(0, center - self.audLen // 2)
        end = min(len_raw, center + self.audLen // 2)

        audio[self.audLen//2-(center-start): self.audLen//2+(end-center)] = \
            audio_raw[start:end]

        # randomize volume
        if self.split == 'train':
            scale = random.random() + 0.5     # 0.5-1.5
            audio *= scale
        audio[audio > 1.] = 1.
        audio[audio < -1.] = -1.
        audio = torch.from_numpy(audio).unsqueeze(0)
        return audio

    def _n_and_stft(self, audios):
        N = len(audios)
        mags = [None for n in range(N)]
        phs = [None for n in range(N)]
        for n in range(N):
            ampN, phN = self._stft(audios[n])
            mags[n] = ampN.unsqueeze(0)
            phs[n] = phN.unsqueeze(0)
        for n in range(N):
            audios[n] = torch.from_numpy(audios[n])

        return mags, phs

    def dummy_mix_data(self, N):
        frames = [None for n in range(N)]
        audios = [None for n in range(N)]
        mags = [None for n in range(N)]
        phs = [None for n in range(N)]

        for n in range(N):
            frames[n] = torch.zeros(
                3, self.num_frames, self.imgSize, self.imgSize)
            audios[n] = torch.zeros(self.audLen).unsqueeze(0)
            mags[n] = torch.zeros(1, self.HS, self.WS)
            phs[n] = torch.zeros(1, self.HS, self.WS)
        return mags, frames, audios, phs