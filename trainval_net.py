# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
      adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

from utils import DictToArgs, Sampler


class Trainer(object):
    """
    Trains a model on a given dataset with set parameters
    """

    def __init__(self, args, cli=False):
        """
        If cli is False then args is a dict
        otherwise, it is an `argparse` object
        """
        self.cli = cli
        self.build_args(args)

    def build_args(self, args):
        # Build an args object if not inputs from a CLI
        if not self.cli:
            default_args = vars(build_parser().parse_known_args()[0])
            default_args.update(args)
            self.args = DictToArgs(default_args)
        else:
            self.args = args

    def run(self):
        if self.args.transfer:
            assert self.args.resume == True,\
                   "Resume must be true when transfer learning"
            assert self.args.resume_classes is not None,\
                   "resume_classes must have a value when transfer learning"

        if self.args.dataset == "pascal_voc":
            self.args.imdb_name = "voc_2007_trainval"
            self.args.imdbval_name = "voc_2007_test"
            self.args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS',
                             '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif self.args.dataset == "pigs_voc":
            self.args.imdb_name = "pigs_voc_train"
            self.args.imdbval_name = "pigs_voc_train"
            self.args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS',
                             '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif self.args.dataset == "pascal_voc_0712":
            self.args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
            self.args.imdbval_name = "voc_2007_test"
            self.args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS',
                             '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif self.args.dataset == "coco":
            self.args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
            self.args.imdbval_name = "coco_2014_minival"
            self.args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS',
                             '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
        elif self.args.dataset == "imagenet":
            self.args.imdb_name = "imagenet_train"
            self.args.imdbval_name = "imagenet_val"
            self.args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS',
                             '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
        elif self.args.dataset == "vg":
            # train sizes: train, smalltrain, minitrain
            # train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150',
            # '1750-700-450', '1600-400-20']
            self.args.imdb_name = "vg_150-50-50_minitrain"
            self.args.imdbval_name = "vg_150-50-50_minival"
            self.args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS',
                             '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

        self.args.cfg_file = "cfgs/{}_ls.yml".format(self.args.net) \
                        if self.args.large_scale \
                        else "cfgs/{}.yml".format(self.args.net)

        if self.args.cfg_file is not None:
            cfg_from_file(self.args.cfg_file)
        if self.args.set_cfgs is not None:
            cfg_from_list(self.args.set_cfgs)

        print('Using config:')
        pprint.pprint(cfg)
        np.random.seed(cfg.RNG_SEED)

        #torch.backends.cudnn.benchmark = True
        if torch.cuda.is_available() and not self.args.cuda:
            print("WARNING: You have a CUDA device, "+\
                  "so you should probably run with --cuda")

        # train set
        # Note: Use validation set and disable the flipped to enable faster loading.
        if self.args.dataset == "pigs_voc":
            cfg.TRAIN.USE_FLIPPED = False
        else:
            cfg.TRAIN.USE_FLIPPED = True

        cfg.USE_GPU_NMS = self.args.cuda
        imdb, roidb, ratio_list, ratio_index = combined_roidb(self.args.imdb_name)
        train_size = len(roidb)

        print('{:d} roidb entries'.format(len(roidb)))

        output_dir = self.args.save_dir + "/" + self.args.net + "/" + self.args.dataset
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        sampler_batch = Sampler(train_size, self.args.batch_size)

        dataset = roibatchLoader(roidb, ratio_list, ratio_index, self.args.batch_size, \
                                 imdb.num_classes, training=True)

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.args.batch_size,
                                                 sampler=sampler_batch,
                                                 num_workers=self.args.num_workers)

        # initilize the tensor holder here.
        im_data = torch.FloatTensor(1)
        im_info = torch.FloatTensor(1)
        num_boxes = torch.LongTensor(1)
        gt_boxes = torch.FloatTensor(1)

        # ship to cuda
        if self.args.cuda:
            im_data = im_data.cuda()
            im_info = im_info.cuda()
            num_boxes = num_boxes.cuda()
            gt_boxes = gt_boxes.cuda()

        # make variable
        im_data = Variable(im_data)
        im_info = Variable(im_info)
        num_boxes = Variable(num_boxes)
        gt_boxes = Variable(gt_boxes)

        if self.args.cuda:
            cfg.CUDA = True

        # If we are resuming a network and doing transfer learning
        # then we want the network needs to be initialised with the
        # resuming datasets number of classes rather than the new
        # dataset that was loaded in to imdb
        if self.args.resume and self.args.transfer:
            n_classes = list(range(self.args.resume_classes))
        else:
            n_classes = imdb.classes

        # initilize the network here.
        if self.args.net == 'vgg16':
            fasterRCNN = vgg16(n_classes, pretrained=True,
                               class_agnostic=self.args.class_agnostic)
        elif self.args.net == 'res101':
            fasterRCNN = resnet(n_classes, 101, pretrained=True,
                                class_agnostic=self.args.class_agnostic)
        elif self.args.net == 'res50':
            fasterRCNN = resnet(n_classes, 50, pretrained=True,
                                class_agnostic=self.args.class_agnostic)
        elif self.args.net == 'res152':
            fasterRCNN = resnet(n_classes, 152, pretrained=True,
                                class_agnostic=self.args.class_agnostic)
        else:
            print("network is not defined")
            pdb.set_trace()

        fasterRCNN.create_architecture()

        lr = cfg.TRAIN.LEARNING_RATE
        lr = self.args.lr
        #tr_momentum = cfg.TRAIN.MOMENTUM
        #tr_momentum = self.args.momentum

        params = []
        for key, value in dict(fasterRCNN.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params':[value],
                                'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1),
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and \
                                                cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    params += [{'params':[value],
                                'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

        if self.args.optimizer == "adam":
            lr = lr * 0.1
            optimizer = torch.optim.Adam(params)

        elif self.args.optimizer == "sgd":
            optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

        if self.args.cuda:
            fasterRCNN.cuda()

        if self.args.resume:
            load_name = os.path.join(self.args.save_dir, self.args.net, self.args.resume_dataset,
              'faster_rcnn_{}_{}_{}.pth'.format(self.args.checksession, self.args.checkepoch,
                                                self.args.checkpoint))
            print("loading checkpoint %s" % (load_name))
            checkpoint = torch.load(load_name)
            self.args.session = checkpoint['session']
            self.args.start_epoch = checkpoint['epoch']
            fasterRCNN.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr = optimizer.param_groups[0]['lr']
            if 'pooling_mode' in checkpoint.keys():
                cfg.POOLING_MODE = checkpoint['pooling_mode']
            print("loaded checkpoint %s" % (load_name))

        # Reconfigure the FC layer to the new datasets number of classes
        # 2048 is the output from the previous layer
        if self.args.transfer:
            fasterRCNN.RCNN_cls_score = nn.Linear(2048, imdb.num_classes)
            if self.args.cuda:
                fasterRCNN.cuda()

        if self.args.mGPUs:
            fasterRCNN = nn.DataParallel(fasterRCNN)

        iters_per_epoch = int(train_size / self.args.batch_size)

        if self.args.use_tfboard:
            from tensorboardX import SummaryWriter
            logger = SummaryWriter("logs")

        for epoch in range(self.args.start_epoch, self.args.max_epochs + 1):
            # setting to train mode
            fasterRCNN.train()
            loss_temp = 0
            start = time.time()

            if epoch % (self.args.lr_decay_step + 1) == 0:
                adjust_learning_rate(optimizer, self.args.lr_decay_gamma)
                lr *= self.args.lr_decay_gamma

            data_iter = iter(dataloader)
            for step in range(iters_per_epoch):
                data = next(data_iter)
                im_data.data.resize_(data[0].size()).copy_(data[0])
                im_info.data.resize_(data[1].size()).copy_(data[1])
                gt_boxes.data.resize_(data[2].size()).copy_(data[2])
                num_boxes.data.resize_(data[3].size()).copy_(data[3])

                fasterRCNN.zero_grad()
                rois, cls_prob, bbox_pred, \
                rpn_loss_cls, rpn_loss_box, \
                RCNN_loss_cls, RCNN_loss_bbox, \
                rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

                loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                     + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
                loss_temp += loss.item()

                # backward
                optimizer.zero_grad()
                loss.backward()
                if self.args.net == "vgg16":
                    clip_gradient(fasterRCNN, 10.)
                optimizer.step()

                if step % self.args.disp_interval == 0:
                    end = time.time()
                    if step > 0:
                        loss_temp /= (self.args.disp_interval + 1)

                    if self.args.mGPUs:
                        loss_rpn_cls = rpn_loss_cls.mean().item()
                        loss_rpn_box = rpn_loss_box.mean().item()
                        loss_rcnn_cls = RCNN_loss_cls.mean().item()
                        loss_rcnn_box = RCNN_loss_bbox.mean().item()
                        fg_cnt = torch.sum(rois_label.data.ne(0))
                        bg_cnt = rois_label.data.numel() - fg_cnt
                    else:
                        loss_rpn_cls = rpn_loss_cls.item()
                        loss_rpn_box = rpn_loss_box.item()
                        loss_rcnn_cls = RCNN_loss_cls.item()
                        loss_rcnn_box = RCNN_loss_bbox.item()
                        fg_cnt = torch.sum(rois_label.data.ne(0))
                        bg_cnt = rois_label.data.numel() - fg_cnt

                    print(f"[session {self.args.session}] "+\
                          f"[epoch {epoch}][iter {step}/{iters_per_epoch}] "+\
                          f"loss: {loss_temp}, lr: {lr}")
                    print(f"\t\t\t\tfg/bg=({int(fg_cnt)}/{int(bg_cnt)}), "+\
                          f"time cost: {end-start}")
                    print(f"\t\t\t\trpn_cls: {round(loss_rpn_cls, 3)}, "+\
                          f"rpn_box: {round(loss_rpn_box, 3)}, "+\
                          f"rcnn_cls: {round(loss_rcnn_cls, 3)}, "+\
                          f"rcnn_box: {round(loss_rcnn_box, 3)}")

                    if self.args.use_tfboard:
                        info = {
                          'loss': loss_temp,
                          'loss_rpn_cls': loss_rpn_cls,
                          'loss_rpn_box': loss_rpn_box,
                          'loss_rcnn_cls': loss_rcnn_cls,
                          'loss_rcnn_box': loss_rcnn_box
                        }
                        logger.add_scalars("logs_s_{}/losses".format(self.args.session),
                                           info, (epoch - 1) * iters_per_epoch +\
                                           step)

                    loss_temp = 0
                    start = time.time()


            save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'\
                                                 .format(self.args.session, epoch, step))
            save_checkpoint({
              'session': self.args.session,
              'epoch': epoch + 1,
              'model': fasterRCNN.module.state_dict() if self.args.mGPUs \
                                                      else fasterRCNN.state_dict(),
              'optimizer': optimizer.state_dict(),
              'pooling_mode': cfg.POOLING_MODE,
              'class_agnostic': self.args.class_agnostic,
            }, save_name)
            print('save model: {}'.format(save_name))

        if self.args.use_tfboard:
            logger.close()

def build_parser():
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--net', dest='net',
                      help='vgg16, res101',
                      default='vgg16', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=20, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=100, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="models",
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)

    # set training session
    parser.add_argument('--session', dest='session',
                        help='training session',
                        default=1, type=int)

    # resume trained model
    parser.add_argument('--resume', dest='resume',
                        help='resume checkpoint or not',
                        default=False, action="store_true")
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=0, type=int)
    parser.add_argument('--resume_dataset', dest="resume_dataset",
                        help="The dataset the resuming model was trained on",
                        type=str)

    # transfer learning
    parser.add_argument('--transfer', dest='transfer',
                        help="turn on the transfer learning",
                        default=False, action="store_true")
    parser.add_argument("--resume_classes", dest="resume_classes",
                        help="Resume classes", default=None,
                        type=int)

    # log and diaplay
    parser.add_argument('--use_tfb', dest='use_tfboard',
                        help='whether use tensorboard',
                        action='store_true')

    return parser

if __name__ == '__main__':
    def parse_args():
        """
        Parse input arguments
        """
        parser = build_parser()
        args = parser.parse_args()
        return args

    cli_args = parse_args()
    print('Called with args:')
    print(cli_args)

    Trainer(cli_args, cli=True)
