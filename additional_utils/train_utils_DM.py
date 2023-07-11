import torch
import torch.nn as nn
import os
import h5py

import time
from tqdm import tqdm
import csv
import random
import numpy as np
import torchvision
from functools import lru_cache
import multiprocessing as mp

from nets import ModelBuilder
from additional_utils.train_utils import NetWrapper, AverageMeter, create_optimizer, plot_loss_metrics, checkpoint, adjust_learning_rate, makedirs
from utils import TensorDataset, DiffAugment

def create_net(args):
    builder = ModelBuilder()

    # net_sound = builder.build_sound(
    #     arch=args.arch_sound,
    #     weights=args.weights_sound)

    net_frame = builder.build_frame(
        arch=args.arch_frame,
        pool_type=args.img_pool,
        weights=args.weights_frame, pretrained=args.pretrained)

    net_classifier = builder.build_classifier(
        arch=args.arch_classifier,
        cls_num=args.cls_num,
        weights=args.weights_classifier,
        input_modality=args.input_modality)

    nets = (None, net_frame, net_classifier)
    # if args.input_modality == 'av':
    #     nets = (net_sound, net_frame, net_classifier)
    # elif args.input_modality == 'a':
    #     nets = (net_sound, None, net_classifier)
    # elif args.input_modality == 'v':
    #     nets = (None, net_frame, net_classifier)
    
    netWrapper = NetWrapper(args, nets)
    netWrapper = torch.nn.DataParallel(netWrapper, device_ids=range(args.num_gpus))
    netWrapper.to(args.device)

    return nets, netWrapper

def reset_params(args):
    args.weights_sound = ''
    args.weights_frame = ''
    args.weights_classifier = ''
    args.best_acc = 0
    args.lr_sound = 1e-3
    args.lr_frame = 1e-4
    args.lr_classifier = 1e-3
    makedirs(args.ckpt, remove=True)

# train one epoch
def train(netWrapper, loader, optimizer, history, epoch, args):
    torch.set_grad_enabled(True)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    criterion = nn.CrossEntropyLoss()

    # switch to train mode
    netWrapper.train()

    # main loop
    torch.cuda.synchronize()
    tic = time.perf_counter()

    for i, batch_data in enumerate(loader):

        # measure data time
        torch.cuda.synchronize()
        data_time.update(time.perf_counter() - tic)

        audio = None
        frame = batch_data[0].float().to(args.device)
        gt = batch_data[1].to(args.device)

        frame = DiffAugment(frame, args.dsa_strategy, param=args.dsa_param)
        # forward pass
        netWrapper.zero_grad()
        output = netWrapper.forward(audio, frame)
        err = criterion(output, gt)

        # backward
        err.backward()
        optimizer.step()

        # measure total time
        torch.cuda.synchronize()
        batch_time.update(time.perf_counter() - tic)
        tic = time.perf_counter()

        # display
        # if i % args.disp_iter == 0:
        #     print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
        #           'lr_sound: {}, lr_frame: {}, lr_classifier: {}, '
        #           'loss: {:.4f}'
        #           .format(epoch, i, args.epoch_iters,
        #                   batch_time.average(), data_time.average(),
        #                   args.lr_sound, args.lr_frame, args.lr_classifier,
        #                   err.item()))
        #     fractional_epoch = epoch - 1 + 1. * i / args.epoch_iters
        #     history['train']['epoch'].append(fractional_epoch)
        #     history['train']['err'].append(err.item())
    return

def evaluate(netWrapper, loader, history, epoch, args, test_mode=False):
    # print('Evaluating at {} epochs...'.format(epoch))
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        # switch to eval mode
        netWrapper.eval()

        # initialize meters
        loss_meter = AverageMeter()
        correct = 0

        total = 0

        for i, batch_data in enumerate(loader):
            audio = None
            frame = batch_data[0].float().to(args.device)
            gt = batch_data[1].to(args.device)
            
            # forward pass
            preds = netWrapper(audio, frame)
            err = criterion(preds, gt)

            _, predicted = torch.max(preds.data, 1)
            total += preds.size(0)
            correct += (predicted == gt).sum().item()

            loss_meter.update(err.item())

        acc = 100 * correct / total
        # print('[Eval Summary] Epoch: {}, Loss: {:.4f}'
        #       .format(epoch, loss_meter.average()))
        history['val']['epoch'].append(epoch)
        history['val']['err'].append(loss_meter.average())
        history['val']['acc'].append(acc)
        if test_mode:
            print(f'iteration:{args.it} ', ' test synthetic data accuracy: %.2f %%' % (100 * correct / total))

        # Plot figure
        if epoch > 0:
            # print('Plotting figures...')
            plot_loss_metrics(args.ckpt, history)
    
    return (100 * correct / total)

def evaluate_test(args, testloader):
    # print('Evaluating on test set...')

    if (args.input_modality == 'a') or (args.input_modality == 'av'):
        args.weights_sound = os.path.join(args.ckpt, 'sound_best.pth')

    if (args.input_modality == 'v') or (args.input_modality == 'av'):    
        args.weights_frame = os.path.join(args.ckpt, 'frame_best.pth')
        
    args.weights_classifier = os.path.join(args.ckpt, 'classifier_best.pth')
    
    nets, netWrapper = create_net(args)

    history = {
        'train': {'epoch': [], 'err': []},
        'val': {'epoch': [], 'err': [], 'acc':[], 'cos':[]}
    }

    epoch = 0

    acc = evaluate(netWrapper, testloader, history, epoch, args, test_mode=True)
    return acc

def eval_synthetic(args, images_train, labels_train, valloader, testloader):
    reset_params(args)
    history = {
        'train': {'epoch': [], 'err': []},
        'val': {'epoch': [], 'err': [], 'acc':[], 'cos':[]}
    }

    images_train = images_train.to(args.device)
    labels_train = labels_train.to(args.device)

    dst_train = TensorDataset(images_train, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_syn, shuffle=True, num_workers=0)
    args.epoch_iters = len(dst_train) // args.batch_syn
    
    ## create network
    nets, netWrapper = create_net(args)
    optimizer = create_optimizer(nets, args)

    ## train loop
    for epoch in tqdm(range(1, args.epoch_eval_train + 1)):
        train(netWrapper, trainloader, optimizer, history, epoch, args)

        # Evaluation and visualization
        if epoch % args.eval_epoch == 0:
            _ = evaluate(netWrapper, valloader, history, epoch, args)
            checkpoint(nets, history, epoch, args)
        
        if epoch in args.lr_steps:
            adjust_learning_rate(optimizer, args)  
    
    acc = evaluate_test(args, testloader)
    return acc


class realBatchPT():
    def __init__(self, list_file, args):
        self.frame_path = '/mnt/user/saksham/data/frames_224'
        self.args = args
        categories = args.categories.split(' ')
        id_to_idx = {id: index for index, id in enumerate(categories)}

        self.class_to_Vid_sec_map = {}
        for i in range(len(categories)):
            self.class_to_Vid_sec_map[i] = []

        self.pt = '/mnt/user/saksham/data_distill/DatasetCondensation/data/image_pt.pt'
        self.index_map = torch.load(self.pt)['index_map']
        self.img_data = torch.load(self.pt)['image']

        list_sample = []
        for row in csv.reader(open(list_file, 'r'), delimiter=','):
            if len(row) < 2:
                continue
            list_sample.append(row)
        
        for row in list_sample:
            class_id = id_to_idx[row[3]]
            for i in range(int(row[1]), int(row[2])):
                self.class_to_Vid_sec_map[class_id].append([row[0],i])
    
    def get_realBatch(self, c, num):
        cand_data = self.class_to_Vid_sec_map[c]
        cand_data = np.random.permutation(cand_data)[:num]
        img_idxs = []
        for index, cand_info in enumerate(cand_data):
            curr_frame = random.randint(1, self.args.fps)
            video_id, curr_sec = cand_info
            img_file = f'{video_id}_{curr_sec}_0{curr_frame}'
            img_idx = self.index_map[img_file]
            img_idxs.append(img_idx)
        return self.img_data[img_idxs, :].float()

class realBatchHDF5():
    def __init__(self, list_file, args):
        self.frame_path = '/mnt/user/saksham/data/frames_224'
        self.args = args
        categories = args.categories.split(' ')
        h5py_file = '/mnt/user/saksham/data_distill/DatasetCondensation/data/image_h5.hdf5'
        self.hdf_file = h5py.File(h5py_file, 'r', driver='core')
        id_to_idx = {id: index for index, id in enumerate(categories)}

        self.class_to_Vid_sec_map = {}
        for i in range(len(categories)):
            self.class_to_Vid_sec_map[i] = []

        list_sample = []
        for row in csv.reader(open(list_file, 'r'), delimiter=','):
            if len(row) < 2:
                continue
            list_sample.append(row)
        
        for row in list_sample:
            class_id = id_to_idx[row[3]]
            for i in range(int(row[1]), int(row[2])):
                self.class_to_Vid_sec_map[class_id].append([row[0],i])
        
    
    def get_realBatch(self, c, num):
        cand_data = self.class_to_Vid_sec_map[c]
        cand_data = np.random.permutation(cand_data)[:num]
        ans_data = torch.randn(num, 3, 224, 224)
        for index, cand_info in enumerate(cand_data):
            curr_frame = random.randint(1, self.args.fps)
            video_id, curr_sec = cand_info
            # file = os.path.join(self.frame_path, video_id, str(curr_sec), f'{video_id}_{curr_sec}_0{curr_frame}.jpg')
            img = self.hdf_file[f'{video_id}_{curr_sec}_0{curr_frame}']
            img = torch.from_numpy(img[:])
            ans_data[index,:] = img
        return ans_data


class realBatch():
    def __init__(self, list_file, args):
        self.frame_path = '/mnt/user/saksham/data/frames_224'
        self.args = args
        categories = args.categories.split(' ')
        id_to_idx = {id: index for index, id in enumerate(categories)}

        self.class_to_Vid_sec_map = {}
        for i in range(len(categories)):
            self.class_to_Vid_sec_map[i] = []

        list_sample = []
        for row in csv.reader(open(list_file, 'r'), delimiter=','):
            if len(row) < 2:
                continue
            list_sample.append(row)
        
        for row in list_sample:
            class_id = id_to_idx[row[3]]
            for i in range(int(row[1]), int(row[2])):
                self.class_to_Vid_sec_map[class_id].append([row[0],i])
    
    def get_realBatch(self, c, num):
        cand_data = self.class_to_Vid_sec_map[c]
        cand_data = np.random.permutation(cand_data)[:num]
        ans_data = torch.randn(num, 3, 224, 224)
        for index, cand_info in enumerate(cand_data):
            curr_frame = random.randint(1, self.args.fps)
            video_id, curr_sec = cand_info
            file = os.path.join(self.frame_path, video_id, str(curr_sec), f'{video_id}_{curr_sec}_0{curr_frame}.jpg')
            img = torchvision.io.read_image(file)
            ans_data[index,:] = img
        return ans_data 

    def process_frame(self, index, cand_info, frame_path, fps):
        curr_frame = random.randint(1, fps)
        video_id, curr_sec = cand_info
        file = os.path.join(frame_path, video_id, str(curr_sec), f'{video_id}_{curr_sec}_0{curr_frame}.jpg')
        img = torchvision.io.read_image(file)
        return index, img

    def get_realBatch_par(self, c, num):
        cand_data = self.class_to_Vid_sec_map[c]
        cand_data = np.random.permutation(cand_data)[:num]
        ans_data = torch.randn(num, 3, 224, 224)
        
        pool = mp.Pool(processes=4)
        results = []
        
        for index, cand_info in enumerate(cand_data):
            l_args = (index, cand_info, self.frame_path, self.args.fps)
            results.append(pool.apply_async(self.process_frame, l_args))
        
        for result in results:
            index, img = result.get()
            ans_data[index, :] = img
        
        pool.close()
        pool.join()
        
        return ans_data


class realBatchParallel():
    def __init__(self, list_file, args):
        self.frame_path = '/mnt/user/saksham/data/frames_224'
        self.args = args
        categories = args.categories.split(' ')
        id_to_idx = {id: index for index, id in enumerate(categories)}

        self.class_to_Vid_sec_map = {}
        for i in range(len(categories)):
            self.class_to_Vid_sec_map[i] = []

        list_sample = []
        for row in csv.reader(open(list_file, 'r'), delimiter=','):
            if len(row) < 2:
                continue
            list_sample.append(row)
        
        for row in list_sample:
            class_id = id_to_idx[row[3]]
            for i in range(int(row[1]), int(row[2])):
                self.class_to_Vid_sec_map[class_id].append([row[0],i])
    
    @lru_cache(maxsize=None)
    def load_image(self, file_path):
        return torchvision.io.read_image(file_path)

    def load_images_in_parallel(self, file_paths):
        # Use multiprocessing to load images in parallel
        with torch.multiprocessing.Pool(processes=2) as pool:
            images = pool.map(self.load_image, file_paths)
        return torch.stack(images)
    
    def get_realBatch(self, c, num):
        cand_data = self.class_to_Vid_sec_map[c]
        cand_data = np.random.permutation(cand_data)[:num]
        ans_data = torch.randn(num, 3, 224, 224)
        file_paths = []
        for index, cand_info in enumerate(cand_data):
            curr_frame = random.randint(1, self.args.fps)
            video_id, curr_sec = cand_info
            file = os.path.join(self.frame_path, video_id, str(curr_sec), f'{video_id}_{curr_sec}_0{curr_frame}.jpg')
            file_paths.append(file)

        ans_data[:] = self.load_images_in_parallel(file_paths)
        return ans_data        