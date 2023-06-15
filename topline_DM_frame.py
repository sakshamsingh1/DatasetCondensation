
import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
import random
from tqdm import tqdm

from utils import get_loops, get_dataset, get_network, evaluate_synset_upd, ParamDiffAug
from dataset import MUSICDataset
from additional_utils.train_utils import evaluate, makedirs, train, create_optimizer, checkpoint, adjust_learning_rate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='AVE_frame', help='dataset')
    parser.add_argument('--model', type=str, default='ResNet18', help='model')    
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data') 
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')

    args = parser.parse_args()
    # args.method = 'DM'
    args.frameRate = 8

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # args.dsa_param = ParamDiffAug()
    # args.dsa = False if args.dsa_strategy in ['none', 'None'] else True
    args.audio_path = '/mnt/user/saksham/data_distill/data/audio'
    args.frame_path = '/mnt/user/saksham/data_distill/data/frames'
    args.categories = "Church_bell Male_speech Bark airplane Race_car Female_speech Helicopter Violin Flute Ukulele Frying Truck Shofar Motorcycle Chainsaw Acoustic_guitar Train_horn Clock Banjo Goat Baby_cry Bus Cat Horse Toilet_flush Rodents Accordion Mandolin"
    args.num_mix = 1
    args.num_frames = 1
    args.workers = 4
    args.ckpt = '/mnt/user/saksham/data_distill/DatasetCondensation/data/ckpt'
    args.seed = 1234
    args.best_err = float("inf")
    args.best_acc = 0
    args.beta1 = 0.9
    args.weight_decay = 1e-4
    args.lr_sound = 1e-3
    args.lr_frame = 1e-4
    args.lr_classifier = 1e-3
    args.eval_epoch = 1
    args.lr_steps = [10, 20]
    args.imgSize = 128
    args.audRate = 11000
    args.num_mix = 1
    args.disp_iter = 20
    args.weights_frame = ''

    args.batch_size = 64
    args.mode = 'train'
    args.num_epoch = 150
    args.list_train = '/mnt/user/saksham/data_distill/data/labels/trainSet.csv'
    args.list_val = '/mnt/user/saksham/data_distill/data/labels/valSet.csv'
    args.id = ''

    it_eval = 0

    dataset_train = MUSICDataset(args.list_train, args, split='train')
    dataset_val = MUSICDataset(args.list_val, args, split='test')

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        drop_last=True)
    loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=int(args.workers),
        drop_last=False)
    
    # import pdb; pdb.set_trace()

    channel = 3
    num_classes = 28
    
    
    args.id = '-{}-{}'.format(
        args.model, args.dataset)
    args.id += '-epoch{}'.format(args.num_epoch)

    print('Model ID: {}'.format(args.id))
    
    # paths to save/load output
    args.ckpt = os.path.join(args.ckpt, args.id)
    if args.mode == 'train':
        makedirs(args.ckpt, remove=True)
    if args.mode == 'eval':
        args.weights_frame = os.path.join(args.ckpt, 'frame_best.pth')

    random.seed(args.seed)
    torch.manual_seed(args.seed)    
    
    net = get_network(args.model, channel, num_classes, [args.imgSize,args.imgSize]).to(args.device)
    history = {
        'train': {'epoch': [], 'err': []},
        'val': {'epoch': [], 'err': [], 'acc':[], 'cos':[]}
    }

    if len(args.weights_frame) > 0:
        print('Loading weights for net')
        net.load_state_dict(torch.load(args.weights_frame))

    args.epoch_iters = len(dataset_train) // args.batch_size
    print('1 Epoch = {} iters'.format(args.epoch_iters))

    # Eval mode
    evaluate(net, loader_val, history, 0, args)
    if args.mode == 'eval':
        print('Evaluation Done!')
        return
    
    optimizer = create_optimizer(net, args)

    # Training loop
    for epoch in tqdm(range(1, args.num_epoch + 1)):
        train(net, loader_train, optimizer, history, epoch, args)

        # Evaluation and visualization
        if epoch % args.eval_epoch == 0:
            evaluate(net, loader_val, history, epoch, args)
            checkpoint(net, history, epoch, args)
        
        if epoch in args.lr_steps:
            adjust_learning_rate(optimizer, args)    

if __name__ == '__main__':
    main()