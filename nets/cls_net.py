import torch
import torch.nn as nn

class Classifier_Concat(nn.Module):
    def __init__(self, cls_num, input_modality):
        super(Classifier_Concat, self).__init__()
        # dim_size = 640
        dim_size = 1024
        if input_modality == 'v':
            dim_size = 512
            # dim_size = 128
        elif input_modality == 'a':
            # dim_size = 128
            dim_size = 512
        self.fc1 = nn.Linear(dim_size, cls_num)

    def forward(self, feat_img, feat_sound):
        if feat_img is None:
            feat = feat_sound
        elif feat_sound is None:
            feat = feat_img
        else:
            feat = torch.cat((feat_img,  feat_sound), dim =-1)
        g = self.fc1(feat)
        return g

class Classifier_joint(nn.Module):
    def __init__(self, cls_num, input_dim_size):
        super(Classifier_joint, self).__init__()
        self.fc1 = nn.Linear(input_dim_size, cls_num)

    def forward(self, feat_img, feat_sound):
        if feat_img is None:
            feat = feat_sound
        elif feat_sound is None:
            feat = feat_img
        else:
            feat = torch.cat((feat_img,  feat_sound), dim =-1)
        g = self.fc1(feat)
        return g