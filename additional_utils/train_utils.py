# System libs
import os
import random
import time
import shutil

# Numerical libs
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
import numpy as np
import scipy.io.wavfile as wavfile
# from scipy.misc import imsave
from tqdm import tqdm
import imageio

# Our libs
from nets import activate

import matplotlib.pyplot as plt

def plot_loss_metrics(path, history):
    fig = plt.figure()
    plt.plot(history['train']['epoch'], history['train']['err'],
             color='b', label='training')
    plt.plot(history['val']['epoch'], history['val']['err'],
             color='c', label='validation')
    plt.legend()
    fig.savefig(os.path.join(path, 'loss.png'), dpi=200)
    plt.close('all')

def makedirs(path, remove=False):
    if os.path.isdir(path):
        if remove:
            shutil.rmtree(path)
            print('removed existing directory...')
        else:
            return
    os.makedirs(path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val*weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        val = np.asarray(val)
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        if self.val is None:
            return 0.
        else:
            return self.val.tolist()

    def average(self):
        if self.avg is None:
            return 0.
        else:
            return self.avg.tolist()


# Network wrapper, defines forward pass
class NetWrapper(torch.nn.Module):
    def __init__(self, args, nets):
        super(NetWrapper, self).__init__() 
        self.net_sound, self.net_frame, self.net_classifier = nets
        self.args = args

    def forward(self, audio, frame):
        feat_sound, feat_frame = None, None
        if self.net_sound is not None:
            feat_sound = self.net_sound(audio)
            feat_sound = activate(feat_sound, self.args.sound_activation)
        if self.net_frame is not None:
            feat_frame = self.net_frame(frame)
            feat_frame = activate(feat_frame, self.args.img_activation)  
        pred = self.net_classifier(feat_sound, feat_frame)
        return pred


def evaluate(netWrapper, loader, history, epoch, args):
    print('Evaluating at {} epochs...'.format(epoch))
    criterion = nn.CrossEntropyLoss()
    torch.set_grad_enabled(False)

    # switch to eval mode
    netWrapper.eval()

    # initialize meters
    loss_meter = AverageMeter()
    correct = 0

    total = 0

    for i, batch_data in enumerate(loader):
        audios = batch_data['audios']
        frames = batch_data['frames']
        gts = batch_data['labels']

        audio = audios[0].to(args.device).detach()
        frame = frames[0].to(args.device).squeeze(2).detach()
        gt = gts[0].to(args.device)
        
        # forward pass
        preds = netWrapper(audio, frame)
        err = criterion(preds, gt)

        _, predicted = torch.max(preds.data, 1)
        total += preds.size(0)
        correct += (predicted == gt).sum().item()

        loss_meter.update(err.item())

    acc = 100 * correct / total
    print('[Eval Summary] Epoch: {}, Loss: {:.4f}'
          .format(epoch, loss_meter.average()))
    history['val']['epoch'].append(epoch)
    history['val']['err'].append(loss_meter.average())
    history['val']['acc'].append(acc)
    print('Accuracy of the audio-visual event recognition network: %.2f %%' % (
            100 * correct / total))

    # Plot figure
    if epoch > 0:
        print('Plotting figures...')
        plot_loss_metrics(args.ckpt, history)

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

        audios = batch_data['audios']
        frames = batch_data['frames']
        gts = batch_data['labels']
        audio = audios[0].to(args.device)
        frame = frames[0].to(args.device).squeeze(2)
        gt = gts[0].to(args.device)


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
        if i % args.disp_iter == 0:
            print('Epoch: [{}][{}/{}], Time: {:.2f}, Data: {:.2f}, '
                  'lr_sound: {}, lr_frame: {}, lr_classifier: {}, '
                  'loss: {:.4f}'
                  .format(epoch, i, args.epoch_iters,
                          batch_time.average(), data_time.average(),
                          args.lr_sound, args.lr_frame, args.lr_classifier,
                          err.item()))
            fractional_epoch = epoch - 1 + 1. * i / args.epoch_iters
            history['train']['epoch'].append(fractional_epoch)
            history['train']['err'].append(err.item())
    return

def checkpoint(nets, history, epoch, args):
    print('Saving checkpoints at {} epochs.'.format(epoch))
    (net_sound, net_frame, net_classifier) = nets
    suffix_latest = 'latest.pth'
    suffix_best = 'best.pth'

    torch.save(history,
               '{}/history_{}'.format(args.ckpt, suffix_latest))
    if net_sound is not None:
        torch.save(net_sound.state_dict(),
                '{}/sound_{}'.format(args.ckpt, suffix_latest))
    if net_frame is not None:
        torch.save(net_frame.state_dict(),
                '{}/frame_{}'.format(args.ckpt, suffix_latest))
    torch.save(net_classifier.state_dict(),
               '{}/classifier_{}'.format(args.ckpt, suffix_latest))

    cur_acc = history['val']['acc'][-1]
    if cur_acc > args.best_acc:
        args.best_acc = cur_acc
        if net_sound is not None:
            torch.save(net_sound.state_dict(),
                    '{}/sound_{}'.format(args.ckpt, suffix_best))
        if net_frame is not None:        
            torch.save(net_frame.state_dict(),
                    '{}/frame_{}'.format(args.ckpt, suffix_best))
        torch.save(net_classifier.state_dict(),
                   '{}/classifier_{}'.format(args.ckpt, suffix_best))

def create_optimizer(nets, args):
    (net_sound, net_frame, net_classifier) = nets
    param_groups = [{'params': net_classifier.parameters(), 'lr': args.lr_classifier}]
    if net_sound is not None:
        param_groups += [{'params': net_sound.parameters(), 'lr': args.lr_sound}]
    if net_frame is not None:
        param_groups += [{'params': net_frame.parameters(), 'lr': args.lr_frame}]
    return torch.optim.SGD(param_groups, momentum=args.beta1, weight_decay=args.weight_decay)

def adjust_learning_rate(optimizer, args):
    args.lr_sound *= 0.1
    args.lr_frame *= 0.1
    args.lr_classifier *= 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] *= 0.1

