import os
import random
import numpy as np
import torch
from .base import BaseDataset

class MUSICDataset(BaseDataset):
    def __init__(self, list_sample, opt, **kwargs):
        super(MUSICDataset, self).__init__(
            list_sample, opt, **kwargs)
        self.fps = opt.frameRate
        self.audio_path = opt.audio_path
        self.frame_path = opt.frame_path
        self.categories = opt.categories
        self.num_mix = opt.num_mix

        categories = self.categories.split(' ')
        self.id_to_idx = {id: index for index, id in enumerate(categories)}

    def encode(self, id):
        """ label encoding

            Returns:
              1d array, multimonial representation, e.g. [1,0,1,0,0,...]
            """
        index = self.id_to_idx[id]
        label = torch.from_numpy(np.array(index)).long()
        return label

    def __getitem__(self, index):

        N = self.num_mix
        frames = [None for n in range(N)]
        audios = [None for n in range(N)]
        infos = [[] for n in range(N)]
        path_frames = [[] for n in range(N)]
        path_audios = ['' for n in range(N)]
        center_frames = [0 for n in range(N)]
        labels = [None for n in range(N)]

        # the first video
        infos[0] = self.list_sample[index]
        cls = infos[0][3]
        labels[0] = self.encode(cls)
        # sample other videos
        # block during tsne
        if not self.split == 'train':
            random.seed(index)

        for n in range(1, N):
            indexN = random.randint(0, len(self.list_sample)-1)
            infos[n] = self.list_sample[indexN]
            labels[n] = self.encode(infos[n][0].split('/')[1])

        # select frames
        for n, infoN in enumerate(infos):
            video_id, st, endt, label = infoN
            st, endt = int(st), int(endt)
            # path_audioN, path_frameN, count_framesN = infoN

            if self.split == 'train':
                # random, not to sample start and end n-frames
                curr_sec = random.randint(int(st), int(endt)-1)
                curr_frame = random.randint(1, self.fps)
            else:
                curr_sec = st + ((endt - st) // 2)
                curr_frame = 1 + ((self.fps-1) // 2)

            # absolute frame/audio paths
            for i in range(self.num_frames):
                path_frames[n].append(
                    os.path.join(self.frame_path, video_id, str(curr_sec), f'{video_id}_{curr_sec}_0{curr_frame}.jpg'))
            path_audios[n] = os.path.join(self.audio_path, video_id, f'{video_id}_{curr_sec}.wav')

        # load frames and audios, STFT
        try:
            for n, infoN in enumerate(infos):
                frames[n] = self._load_frames(path_frames[n])
                # jitter audio
                # center_timeN = (center_frames[n] - random.random()) / self.fps
                audios[n] = self.get_spectrogram(path_audios[n])
                # audios[n] = None

        except Exception as e:
            print('Failed loading frame/audio: {}'.format(e))
            # create dummy data
            mags, frames, audios, phs = \
                self.dummy_mix_data(N)

        ret_dict = {'frames': frames, 'audios': audios, 'labels':labels}
        # ret_dict = {'frames': frames, 'labels':labels}
        if self.split != 'train':
            ret_dict['infos'] = infos

        return ret_dict

