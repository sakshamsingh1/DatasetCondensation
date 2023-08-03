import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import get_dataset, get_time, DiffAugment, ParamDiffAug
from additional_utils.train_utils_DM import eval_synthetic_comb, realBatchCombPT, adjust_lr_DM, evaluate_test_comb
from additional_utils.train_utils import makedirs
from nets import ModelBuilder
import torch.nn.functional as F

torch.set_num_threads(4)

import warnings
warnings.filterwarnings("ignore")

def main(args):
    interval = 50
    eval_it_pool = np.arange(0, args.Iteration+1, interval).tolist()
    
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, valloader = get_dataset(args.dataset, args.data_path, args)

    data_save = []
    labels_all = []

    indices_class = [[] for c in range(num_classes)]
    final_acc = None

    labels_all = [dst_train[i]['label'] for i in range(len(dst_train))]
    for i, lab in enumerate(labels_all):
        indices_class[lab].append(i)
    labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

    for c in range(num_classes):
        print('class c = %d: %d real images'%(c, len(indices_class[c])))

    def get_aud_images(c, n): # get random n images from class c
        idx_aud, idx_img = None, None
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        if args.input_modality == 'a' or args.input_modality == 'av':
            idx_aud = dst_train[idx_shuffle]['audio'].to(args.device)
        if args.input_modality == 'v' or args.input_modality == 'av':
            idx_img = dst_train[idx_shuffle]['frame'].to(args.device)
        return idx_aud, idx_img

    ''' initialize the synthetic data '''
    image_syn, aud_syn = None, None
    if args.input_modality == 'a' or args.input_modality == 'av':
        aud_syn = torch.randn(size=(num_classes*args.ipc, channel[0], im_size[0][0], im_size[0][1]), dtype=torch.float, requires_grad=True, device=args.device)
    
    if args.input_modality == 'v' or args.input_modality == 'av':
        image_syn = torch.randn(size=(num_classes*args.ipc, channel[1], im_size[1][0], im_size[1][1]), dtype=torch.float, requires_grad=True, device=args.device)
    
    label_syn = torch.tensor([np.ones(args.ipc)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1)

    if args.init == 'real':
        print('initialize synthetic data from random real images')
        for c in range(num_classes):
            aud_real_init, img_real_init = get_aud_images(c, args.ipc)
            
            if args.input_modality == 'a' or args.input_modality == 'av':
                aud_real_init = aud_real_init.detach().data
                aud_syn.data[c*args.ipc:(c+1)*args.ipc] = aud_real_init

            if args.input_modality == 'v' or args.input_modality == 'av':
                img_real_init = img_real_init.detach().data
                image_syn.data[c*args.ipc:(c+1)*args.ipc] = img_real_init
    else:
        print('initialize synthetic data from random noise')

    def get_syn_optimizer(aud_syn, img_syn):
        param_groups = []
        if args.input_modality == 'a' or args.input_modality == 'av':
            param_groups += [{'params': aud_syn, 'lr': args.lr_syn_aud}]
        
        if args.input_modality == 'v' or args.input_modality == 'av':
            param_groups += [{'params': img_syn, 'lr': args.lr_syn_img}]
        return torch.optim.SGD(param_groups, momentum=0.5)
        # return torch.optim.Adam(param_groups)

    # ''' training '''
    optimizer_comb = get_syn_optimizer(aud_syn, image_syn)
    real_obj = realBatchCombPT(args.list_train, args)  

    #embedding networks
    builder = ModelBuilder()
    net_frame = builder.build_frame(
        arch='resnet18',
        pool_type=args.img_pool,
        weights=args.weights_frame, pretrained=False)
    net_frame = net_frame.to(args.device)

    net_audio = builder.build_sound(
        arch='resnet18',
        weights='')
    net_audio = net_audio.to(args.device)

    net_frame.train()
    net_audio.train()
    for param in list(net_frame.parameters()):
        param.requires_grad = False
    for param in list(net_audio.parameters()):
        param.requires_grad = False

    print('%s training begins'%get_time())
    for it in range(args.Iteration+1):

        ''' Train synthetic data '''
        loss_avg = 0
        loss = torch.tensor(0.0).to(args.device)
        for c in range(num_classes):
            aud_real, img_real = real_obj.get_realBatch(c, args.batch_real)
            aud_real, img_real = aud_real.to(args.device), img_real.to(args.device)
            
            if args.input_modality == 'a' or args.input_modality == 'av':
                curr_aud_syn = aud_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel[0], im_size[0][0], im_size[0][1]))
            if args.input_modality == 'v' or args.input_modality == 'av':
                curr_img_syn = image_syn[c*args.ipc:(c+1)*args.ipc].reshape((args.ipc, channel[1], im_size[1][0], im_size[1][1]))

            if args.dsa:
                seed = int(time.time() * 1000) % 100000
                if args.input_modality == 'a' or args.input_modality == 'av':
                    aud_real = DiffAugment(aud_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                    curr_aud_syn = DiffAugment(curr_aud_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

                if args.input_modality == 'v' or args.input_modality == 'av':
                    img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                    curr_img_syn = DiffAugment(curr_img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)

            if args.input_modality == 'a' or args.input_modality == 'av':
                embd_aud_real = net_audio(aud_real).detach()
                embd_aud_syn = net_audio(curr_aud_syn)
            
            if args.input_modality == 'v' or args.input_modality == 'av':
                embd_img_real = net_frame(img_real).detach()
                embd_img_syn = net_frame(curr_img_syn)

            if args.input_modality == 'av':
                if args.DM_loss_type == 'type_1':
                    loss += torch.sum((torch.mean(embd_aud_real, dim=0) - torch.mean(embd_aud_syn, dim=0))**2)
                    loss += torch.sum((torch.mean(embd_img_real, dim=0) - torch.mean(embd_img_syn, dim=0))**2)
                    loss /= 2
                
                elif args.DM_loss_type == 'type_2':
                    embd_real = torch.cat((embd_aud_real, embd_img_real), dim=1)
                    embd_syn = torch.cat((embd_aud_syn, embd_img_syn), dim=1)
                    loss += torch.sum((torch.mean(embd_real, dim=0) - torch.mean(embd_syn, dim=0))**2)
                
                elif args.DM_loss_type == 'type_3':
                    loss += torch.sum((torch.mean(embd_aud_real, dim=0) - torch.mean(embd_aud_syn, dim=0))**2)
                    loss += torch.sum((torch.mean(embd_img_real, dim=0) - torch.mean(embd_img_syn, dim=0))**2)

                    embd_real = torch.cat((embd_aud_real, embd_img_real), dim=1)
                    embd_syn = torch.cat((embd_aud_syn, embd_img_syn), dim=1)
                    loss += torch.sum((torch.mean(embd_real, dim=0) - torch.mean(embd_syn, dim=0))**2)
                    loss /= 3

                elif args.DM_loss_type == 'type_4':
                    loss += torch.sum((torch.mean(embd_aud_real, dim=0) - torch.mean(embd_aud_syn, dim=0))**2)
                    loss += torch.sum((torch.mean(embd_img_real, dim=0) - torch.mean(embd_img_syn, dim=0))**2)

                    loss += torch.sum((torch.mean(embd_aud_real, dim=0) - torch.mean(embd_img_syn, dim=0))**2)
                    loss += torch.sum((torch.mean(embd_img_real, dim=0) - torch.mean(embd_aud_syn, dim=0))**2)

                    loss /= 4
                
                if args.sampleMatch:
                    # import pdb; pdb.set_trace()
                    # cosine_similarity = F.cosine_similarity(embd_aud_syn, embd_img_syn, dim=1)
                    # mean_cosine_similarity = torch.mean(cosine_similarity)
                    # distance = 1 - mean_cosine_similarity
                    # loss += distance

                    dot_product = torch.bmm(embd_aud_syn.view(args.ipc, 1, 512), embd_img_syn.view(args.ipc, 512, 1)).squeeze()
                    mean_dot_product = torch.mean(dot_product)
                    loss -= 0.01*mean_dot_product

            elif args.input_modality == 'a':
                loss += torch.sum((torch.mean(embd_aud_real, dim=0) - torch.mean(embd_aud_syn, dim=0))**2)
            
            elif args.input_modality == 'v':
                loss += torch.sum((torch.mean(embd_img_real, dim=0) - torch.mean(embd_img_syn, dim=0))**2)

        optimizer_comb.zero_grad()
        loss.backward()
        optimizer_comb.step()
        loss_avg += loss.item()
        loss_avg /= (num_classes)

        if it%10 == 0:
            # print(f'{get_time()} iter = %05d, loss = %.4f, (, it, loss_avg)')
            print(f'{get_time()} iter = {it:05d}, loss = {loss_avg:.4f}, lr = {optimizer_comb.param_groups[0]["lr"]:.6f}')
        
        if args.lr_decay and ((it+1) % args.lr_train_drop_freq==0):
            adjust_lr_DM(optimizer_comb, args)

        if it in eval_it_pool:
            ''' Evaluate synthetic data '''
            # print(f'{get_time()} iter = {it:05d}, loss = {loss_avg:.4f}, lr = {optimizer_comb.param_groups[0]["lr"]:.6f}')
            args.it = it
            aud_syn_eval, image_syn_eval = None, None
            
            if args.input_modality == 'a' or args.input_modality == 'av':
                aud_syn_eval = copy.deepcopy(aud_syn.detach())
            if args.input_modality == 'v' or args.input_modality == 'av':
                image_syn_eval = copy.deepcopy(image_syn.detach())
            label_syn_eval = copy.deepcopy(label_syn.detach()) # avoid any unaware modification
            acc = eval_synthetic_comb(args, aud_syn_eval, image_syn_eval, label_syn_eval, valloader, testloader)
            print(f'it: {it} Val acc: {acc:.2f}%')
            
            ''' visualize and save '''
            if args.input_modality == 'a' or args.input_modality == 'av':
                #audio
                aud_save_name = os.path.join(args.save_path, f'aud_vis_{it}.png')
                aud_syn_vis = copy.deepcopy(aud_syn_eval.detach().cpu())
                aud_syn_vis[aud_syn_vis<0] = 0.0
                aud_syn_vis[aud_syn_vis>1] = 1.0
                save_image(aud_syn_vis, aud_save_name, nrow=args.ipc)

            if args.input_modality == 'v' or args.input_modality == 'av':
                #image
                img_save_name = os.path.join(args.save_path, f'img_vis_{it}.png')
                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel[1]):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                image_syn_vis[image_syn_vis<0] = 0.0
                image_syn_vis[image_syn_vis>1] = 1.0
                save_image(image_syn_vis, img_save_name, nrow=args.ipc) # Trying normalize = True/False may get better visual effects.

        # if it == args.Iteration: # only record the final results
        #     data_save.append([copy.deepcopy(aud_syn.detach().cpu()), copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
        #     torch.save({'data': data_save, 'acc': final_acc, }, os.path.join(args.save_path, 'final_data.pt'))

    print('\n==================== Final Results ====================\n')
    final_acc = evaluate_test_comb(args, testloader)
    print('Final accuracy: %.2f%%'%(final_acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='AVE_comb', help='dataset')
    parser.add_argument('--ipc', type=int, default=20, help='image(s) per class')
    parser.add_argument('--epoch_eval_train', type=int, default=30, help='epochs to train a model with synthetic data') # it can be small for speeding up with little performance drop
    parser.add_argument('--Iteration', type=int, default=10000, help='training iterations')
    parser.add_argument('--lr_syn_aud', type=float, default=1.0, help='learning rate for updating synthetic audio specs')
    parser.add_argument('--lr_syn_img', type=float, default=1.0, help='learning rate for updating synthetic audio specs')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=32, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='real', help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='result', help='path to save results')
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--DM_loss_type', type=str, default='type_1', help='loss_type during DM training')
    parser.add_argument('--input_modality', type=str, default='a', help='a/v/av')
    parser.add_argument('--lr_decay', action='store_true', help='learning rate decay or not')
    parser.add_argument('--curr_best_val_acc', type=float, default=0.0, help='current best validation accuracy')
    parser.add_argument('--sampleMatch', action='store_true', help='whether to use sample correspondence loss or not')
    # type_1: loss = DM(Ra,Sa) + DM(Ri,Si)
    # type_2: loss = DM([Ra;Ri], [Sa;Si])
    # type_3: loss = DM(Ra,Sa) + DM(Ri,Si) + DM([Ra;Ri], [Sa;Si]) : typr 1 + type 2
    # type_4: loss =  DM(Ra,Sa) + DM(Ri,Si) + DM(Ra,Si) + DM(Ri,Sa)

    #common params and distill params
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
    args.num_gpus = 1
    args.eval_epoch = 1
    args.ckpt = '/mnt/user/saksham/data_distill/DatasetCondensation/data/ckpt'
    args.best_acc = 0
    args.lr_steps = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    args.lr_train_drop_freq = 500
    args.patience_thr = 2000

    # sound params
    # args.arch_sound = 'convNet'
    args.arch_sound = 'resnet18'
    args.weights_sound = ''
    args.lr_sound = 1e-3
    args.beta1 = 0.9
    args.weight_decay = 1e-4
    args.img_activation = 'no'

    # image params
    args.arch_frame = 'resnet18'
    args.lr_frame = 1e-4
    # args.arch_frame = 'convNet'
    args.img_pool = 'avgpool'
    args.weights_frame = ''
    args.pretrained = False

    # classifier params
    args.lr_classifier = 1e-3
    args.arch_classifier = 'concat'
    args.cls_num = 28
    args.weights_classifier = ''
    args.sound_activation = 'no'

    args.id = f'{args.method}_{args.dataset}_mod-{args.input_modality}_ipc-{args.ipc}_init-{args.init}_loss:{args.DM_loss_type}_sampleMatch:{args.sampleMatch}'

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    args.save_path = os.path.join(args.save_path, args.id)
    makedirs(args.save_path, remove=True)
    
    args.ckpt = os.path.join(args.ckpt, args.id)
    makedirs(args.ckpt, remove=True)

    main(args)


