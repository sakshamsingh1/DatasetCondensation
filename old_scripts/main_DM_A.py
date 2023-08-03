import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import get_dataset, get_time, DiffAugment, ParamDiffAug
from additional_utils.train_utils_DM import create_net,  eval_synthetic, realBatchAudioPT
from additional_utils.train_utils import makedirs
from nets import ModelBuilder

import warnings
warnings.filterwarnings("ignore")

def main(args):
    interval = 100
    eval_it_pool = np.arange(0, args.Iteration+1, interval).tolist()
    
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, valloader = get_dataset(args.dataset, args.data_path, args)

    data_save = []
    labels_all = []

    indices_class = [[] for c in range(num_classes)]
    final_acc = None

    labels_all = [dst_train[i][1] for i in range(len(dst_train))]
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)
    labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

    for c in range(num_classes):
        print('class c = %d: %d real images'%(c, len(indices_class[c])))

    def get_images(c, n): # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        idx_img = dst_train[idx_shuffle][0].to(args.device)
        return idx_img

    ''' initialize the synthetic data '''
    image_syn = torch.randn(size=(num_classes*args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=args.device)
    label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

    if args.init == 'real':
        print('initialize synthetic data from random real images')
        for c in range(num_classes):
            image_syn.data[c*args.ipc:(c+1)*args.ipc] = get_images(c, args.ipc).detach().data
    else:
        print('initialize synthetic data from random noise')

    # ''' training '''
    optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5) 
    # optimizer_img = torch.optim.Adam([image_syn, ], lr=args.lr_img) 
    optimizer_img.zero_grad()
    real_images = realBatchAudioPT(args.list_train, args)

    nets, _ = create_net(args) # get a random model
    net_sound, _, _ = nets     
    # builder = ModelBuilder()
    # net_frame = builder.build_frame(
    #     arch='resnet18',
    #     pool_type=args.img_pool,
    #     weights=args.weights_frame, pretrained=args.pretrained)
    # net_frame = net_frame.to(args.device)

    net_sound.train()
    for param in list(net_sound.parameters()):
        param.requires_grad = False

    print('%s training begins'%get_time())

    for it in range(args.Iteration+1):

        ''' Evaluate synthetic data '''
        if it in eval_it_pool:
            args.it = it
            image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
            final_acc = eval_synthetic(args, image_syn_eval, label_syn_eval, valloader, testloader)

            ''' visualize and save '''
            save_name = os.path.join(args.save_path, 'vis_%s_%s_%s_%dipc_iter%d.png'%(args.method, args.dataset, args.model, args.ipc, it))
            image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
            for ch in range(channel):
                image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
            image_syn_vis[image_syn_vis<0] = 0.0
            image_syn_vis[image_syn_vis>1] = 1.0
            save_image(image_syn_vis, save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.

        ''' Train synthetic data '''

        loss_avg = 0

        ''' update synthetic data '''
        loss = torch.tensor(0.0).to(args.device)
        for c in range(num_classes):
            # ts = time.time()
            img_real = real_images.get_realBatch(c, args.batch_real).to(args.device)
            # print(f"real batch for class={c}: {time.time() - ts}")
            img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel, im_size[0], im_size[1]))

            if args.dsa:
                seed = int(time.time() * 1000) % 100000
                img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

            output_real = net_sound(img_real).detach()
            output_syn = net_sound(img_syn)

            loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0))**2)

        optimizer_img.zero_grad()
        loss.backward()
        optimizer_img.step()
        loss_avg += loss.item()

        loss_avg /= (num_classes)

        if it%10 == 0:
            print('%s iter = %05d, loss = %.4f' % (get_time(), it, loss_avg))

        if it == args.Iteration: # only record the final results
            data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
            torch.save({'data': data_save, 'acc': final_acc, }, os.path.join(args.save_path, 'res_%s_%s_%dipc.pt'%(args.method, args.dataset, args.ipc)))

    print('\n==================== Final Results ====================\n')
    print('Final accuracy: %.2f%%'%(final_acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='AVE_audio', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=20, help='image(s) per class')
    parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=30, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    parser.add_argument('--Iteration', type=int, default=20000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=1.0, help='learning rate for updating synthetic images')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=32, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='real', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataloader')

    args = parser.parse_args()
    args.method = 'DM'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = False if args.dsa_strategy in ['none', 'None'] else True
    args.it = None
    args.categories = "Church_bell Male_speech Bark airplane Race_car Female_speech Helicopter Violin Flute Ukulele Frying Truck Shofar Motorcycle Chainsaw Acoustic_guitar Train_horn Clock Banjo Goat Baby_cry Bus Cat Horse Toilet_flush Rodents Accordion Mandolin"
    args.list_train = '/mnt/user/saksham/data_distill/data/labels/trainSet.csv'
    args.fps = 8
    args.seed = 1234

    ## new params
    args.num_gpus = 1
    args.disp_iter = 20
    args.eval_epoch = 1
    args.ckpt = '/mnt/user/saksham/data_distill/DatasetCondensation/data/ckpt'
    args.best_acc = 0
    args.lr_steps = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    args.arch_sound = 'convNet'
    args.weights_sound = ''
    args.lr_sound = 1e-3
    args.lr_frame = 1e-4
    args.lr_classifier = 1e-3
    args.beta1 = 0.9
    args.weight_decay = 1e-4
    args.img_activation = 'no'

    args.arch_frame = 'resnet18'
    # args.arch_frame = 'convNet'
    args.img_pool = 'avgpool'
    args.weights_frame = ''
    args.pretrained = False
    args.sound_activation = 'no'

    args.arch_classifier = 'concat'
    args.cls_num = 28
    args.weights_classifier = ''
    args.input_modality = 'a'

    args.id = f'{args.method}_{args.dataset}_mod-{args.input_modality}_ipc-{args.ipc}_pretrained-{args.pretrained}'

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    args.save_path = os.path.join(args.save_path, args.id)
    if not os.path.exists(args.save_path):
        makedirs(args.save_path, remove=True)
    
    args.ckpt = os.path.join(args.ckpt, args.id)

    if not os.path.exists(args.ckpt):
        makedirs(args.ckpt, remove=True)

    main(args)


