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
import logging

from datetime import datetime

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
from test_net import Tester


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
        os.makedirs(self.args.log_path, exist_ok=True)


    def build_args(self, args):
        # Build an args object if not inputs from a CLI
        if not self.cli:
            default_args = vars(build_parser().parse_known_args()[0])
            default_args.update(args)
            self.args = DictToArgs(default_args)
        else:
            self.args = args
    def check_transfer_args(self):
        if self.args.transfer:
            assert self.args.resume == True,\
                   "Resume must be true when transfer learning"
            assert self.args.resume_classes is not None,\
                   "resume_classes must have a value when transfer learning"
            assert self.args.resume_dataset is not None,\
                   "resume_dataset must have a value when transfer learning"

    def create_performance_file(self):
        if not self.args.terminal_logging:
            with open(os.path.join(self.args.log_path,
                                   "performance.csv"),"w") as f:
                f.write("datetime,stage,epoch,score\n")

    def set_data_names(self):
        if self.args.dataset == "pascal_voc":
            self.args.imdb_name = "voc_2007_trainval"
            self.args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                                  'ANCHOR_RATIOS', '[0.5,1,2]',
                                  'MAX_NUM_GT_BOXES', '20']
        elif self.args.dataset == "pigs_voc":
            self.args.imdb_name = "pigs_voc_train"
            self.args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                                  'ANCHOR_RATIOS', '[0.5,1,2]',
                                  'MAX_NUM_GT_BOXES', '20']
        elif self.args.dataset == "pascal_voc_0712":
            self.args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
            self.args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                                  'ANCHOR_RATIOS', '[0.5,1,2]',
                                  'MAX_NUM_GT_BOXES', '20']
        elif self.args.dataset == "coco":
            self.args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
            self.args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]',
                                  'ANCHOR_RATIOS', '[0.5,1,2]',
                                  'MAX_NUM_GT_BOXES', '50']
        elif self.args.dataset == "imagenet":
            self.args.imdb_name = "imagenet_train"
            self.args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]',
                                  'ANCHOR_RATIOS', '[0.5,1,2]',
                                  'MAX_NUM_GT_BOXES', '30']
        elif self.args.dataset == "vg":
            # train sizes: train, smalltrain, minitrain
            # train scale: ['150-50-20','150-50-50','500-150-80', '750-250-150',
            # '1750-700-450', '1600-400-20']
            self.args.imdb_name = "vg_150-50-50_minitrain"
            self.args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]',
                                  'ANCHOR_RATIOS', '[0.5,1,2]',
                                  'MAX_NUM_GT_BOXES', '50']
    def set_config(self):
        self.args.cfg_file = "cfgs/{}_ls.yml".format(self.args.net) \
                        if self.args.large_scale \
                        else "cfgs/{}.yml".format(self.args.net)

        if self.args.cfg_file is not None:
            cfg_from_file(self.args.cfg_file)
        if self.args.set_cfgs is not None:
            cfg_from_list(self.args.set_cfgs)

        logging.debug('Using config:')
        logging.debug(pprint.pformat(cfg))

        if self.args.is_optimising is True:
            seed = 1234
            logging.debug(f"Using random seed {seed}")
            import random
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        #torch.backends.cudnn.benchmark = True
        if torch.cuda.is_available() and not self.args.cuda:
            logging.warning("You have a CUDA device, so you should probably "+\
                            f"run with --cuda")

        # train set
        # Note: Use validation set and disable the flipped to enable faster
        # loading.
        if self.args.dataset == "pigs_voc":
            cfg.TRAIN.USE_FLIPPED = False
        else:
            cfg.TRAIN.USE_FLIPPED = True

        cfg.USE_GPU_NMS = self.args.cuda

        if self.args.cuda:
            cfg.CUDA = True


    def setup_data(self):
        self.train_imdb,\
        self.train_roidb,\
        self.train_ratio_list,\
        self.train_ratio_index = combined_roidb(self.args.imdb_name)

        self.train_size = len(self.train_roidb)

        logging.debug(f'{len(self.train_roidb)} training roidb entries')

    def create_output_dir(self):
        output_dir = os.path.join(self.args.save_dir,
                                  self.args.net,
                                  self.args.dataset)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.output_dir = output_dir

    def build_training_dataloader(self):
        sampler_batch = Sampler(self.train_size, self.args.batch_size)

        dataset = roibatchLoader(self.train_roidb, self.train_ratio_list,
                                 self.train_ratio_index, self.args.batch_size,
                                 self.train_imdb.num_classes, training=True)

        self.dataloader = torch.utils.data.DataLoader(dataset,
                                              batch_size=self.args.batch_size,
                                              sampler=sampler_batch,
                                              num_workers=self.args.num_workers)
    def initialise_tensor_data(self):
        # initilize the tensor holder here.
        self.im_data = torch.FloatTensor(1)
        self.im_info = torch.FloatTensor(1)
        self.num_boxes = torch.LongTensor(1)
        self.gt_boxes = torch.FloatTensor(1)

        # ship to cuda
        if self.args.cuda:
            self.im_data = self.im_data.cuda()
            self.im_info = self.im_info.cuda()
            self.num_boxes = self.num_boxes.cuda()
            self.gt_boxes = self.gt_boxes.cuda()

        # make variable
        self.im_data = Variable(self.im_data)
        self.im_info = Variable(self.im_info)
        self.num_boxes = Variable(self.num_boxes)
        self.gt_boxes = Variable(self.gt_boxes)

    def construct_network(self):
        # If we are resuming a network and doing transfer learning
        # then we want the network needs to be initialised with the
        # resuming datasets number of classes rather than the new
        # dataset that was loaded in to self.train_imdb
        if self.args.resume and self.args.transfer:
            n_classes = list(range(self.args.resume_classes))
        else:
            n_classes = self.train_imdb.classes

        # initilize the network here.
        if self.args.net == 'vgg16':
            self.fasterRCNN = vgg16(n_classes, pretrained=True,
                               class_agnostic=self.args.class_agnostic)
        elif self.args.net == 'res101':
            self.fasterRCNN = resnet(n_classes, 101, pretrained=True,
                                class_agnostic=self.args.class_agnostic)
        elif self.args.net == 'res50':
            self.fasterRCNN = resnet(n_classes, 50, pretrained=True,
                                class_agnostic=self.args.class_agnostic)
        elif self.args.net == 'res152':
            self.fasterRCNN = resnet(n_classes, 152, pretrained=True,
                                class_agnostic=self.args.class_agnostic)
        else:
            logging.error("network is not defined")
            pdb.set_trace()

        self.fasterRCNN.create_architecture()

        self.lr = self.args.lr
        #tr_momentum = cfg.TRAIN.MOMENTUM
        #tr_momentum = self.args.momentum

        params = []
        for key, value in dict(self.fasterRCNN.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params':[value],
                                'lr':self.lr*(cfg.TRAIN.DOUBLE_BIAS + 1),
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and \
                                                cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    params += [{'params':[value],
                                'lr':self.lr,
                                'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

        if self.args.optimizer == "adam":
            self.lr = self.lr * 0.1
            self.optimizer = torch.optim.Adam(params)

        elif self.args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(params,
                                              momentum=cfg.TRAIN.MOMENTUM)

        if self.args.cuda:
            self.fasterRCNN.cuda()

    def resume_network(self):
        load_name = os.path.join(self.args.save_dir, self.args.net,
                                 self.args.resume_dataset,
          'faster_rcnn_{}_{}_{}.pth'.format(self.args.checksession,
                                            self.args.checkepoch,
                                            self.args.checkpoint))
        logging.debug("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        self.args.session = checkpoint['session']
        self.args.start_epoch = checkpoint['epoch']
        self.fasterRCNN.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr = self.optimizer.param_groups[0]['lr']
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']
        logging.debug("loaded checkpoint %s" % (load_name))

    def reconfigure_fc_layer(self):
        """
        Reconfigure the FC layers to the new datasets number of classes
        2048 is the output from the previous layer

        `RCNN_cls_score` is the number of new classes
        `RCNN_bbox_pred` is the number of new classes * 4 (the number of points
                         needed to describe a Bounding Box
        """
        logging.warning(f"Modifying class labels layer from "+\
                        f"{self.args.resume_classes} classes to "+\
                        f"{self.train_imdb.num_classes}")
        self.fasterRCNN.RCNN_cls_score = nn.Linear(2048,
                                                   self.train_imdb.num_classes)

        logging.warning(f"Modifying class bboxes layer from "+\
                        f"{self.args.resume_classes*4} classes to "+\
                        f"{self.train_imdb.num_classes*4}")
        self.fasterRCNN.RCNN_bbox_pred = nn.Linear(2048,
                                                  self.train_imdb.num_classes*4)

        self.lr = self.args.lr

        params = []
        for key, value in dict(self.fasterRCNN.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params':[value],
                                'lr':self.lr*(cfg.TRAIN.DOUBLE_BIAS + 1),
                                'weight_decay': cfg.TRAIN.BIAS_DECAY and \
                                                cfg.TRAIN.WEIGHT_DECAY or 0}]
                else:
                    params += [{'params':[value],
                                'lr':self.lr,
                                'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

        if self.args.optimizer == "adam":
            self.lr = self.lr * 0.1
            self.optimizer = torch.optim.Adam(params)

        elif self.args.optimizer == "sgd":
            self.optimizer = torch.optim.SGD(params,
                                              momentum=cfg.TRAIN.MOMENTUM)

        if self.args.cuda:
            self.fasterRCNN.cuda()

    def train_epoch(self, epoch):
        # setting to train mode
        self.fasterRCNN.train()
        loss_temp = 0
        start = time.time()

        if epoch % (self.args.lr_decay_step + 1) == 0:
            adjust_learning_rate(self.optimizer, self.args.lr_decay_gamma)
            self.lr *= self.args.lr_decay_gamma

        data_iter = iter(self.dataloader)
        for step in range(self.iters_per_epoch):
            data = next(data_iter)
            self.im_data.data.resize_(data[0].size()).copy_(data[0])
            self.im_info.data.resize_(data[1].size()).copy_(data[1])
            self.gt_boxes.data.resize_(data[2].size()).copy_(data[2])
            self.num_boxes.data.resize_(data[3].size()).copy_(data[3])

            self.fasterRCNN.zero_grad()
            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = self.fasterRCNN(self.im_data,
                                         self.im_info,
                                         self.gt_boxes,
                                         self.num_boxes)

            loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
                 + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
            loss_temp += loss.item()

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            if self.args.net == "vgg16":
                clip_gradient(self.fasterRCNN, 10.)
            self.optimizer.step()

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

                logging.info(f"[session {self.args.session}] "+\
                      f"[epoch {epoch}][iter {step}/{self.iters_per_epoch}]")
                logging.info(f"Loss: {loss_temp}, lr: {self.lr}")
                logging.info(f"FG/BG=({int(fg_cnt)}/{int(bg_cnt)}), "+\
                      f"Time cost: {end-start}")
                logging.info(f"rpn_cls: {round(loss_rpn_cls, 3)}, "+\
                      f"rpn_box: {round(loss_rpn_box, 3)}, "+\
                      f"rcnn_cls: {round(loss_rcnn_cls, 3)}, "+\
                      f"rcnn_box: {round(loss_rcnn_box, 3)}")
                logging.info("--------------------------------------------")

                if self.args.use_tfboard:
                    info = {
                      'loss': loss_temp,
                      'loss_rpn_cls': loss_rpn_cls,
                      'loss_rpn_box': loss_rpn_box,
                      'loss_rcnn_cls': loss_rcnn_cls,
                      'loss_rcnn_box': loss_rcnn_box
                    }
                    logger.add_scalars(f"logs_s_{self.args.session}/losses",
                                       info,
                                       (epoch - 1) * self.iters_per_epoch +\
                                       step)

                loss_temp = 0
                start = time.time()


        save_name = os.path.join(self.output_dir,
                                 f'faster_rcnn_{self.args.session}_{epoch}_'+\
                                 f'{step}.pth')
        save_checkpoint({
          'session': self.args.session,
          'epoch': epoch + 1,
          'model': self.fasterRCNN.module.state_dict() \
                   if self.args.mGPUs \
                   else self.fasterRCNN.state_dict(),
          'optimizer': self.optimizer.state_dict(),
          'pooling_mode': cfg.POOLING_MODE,
          'class_agnostic': self.args.class_agnostic,
        }, save_name)
        logging.info('save model: {}'.format(save_name))

    def validate(self, epoch):
        validator = Tester({"dataset": self.args.dataset,
                            "net": self.args.net,
                            "load_dir": self.args.save_dir,
                            "cuda": self.args.cuda,
                            "checksession": self.args.session,
                            "checkepoch": epoch,
                            "checkpoint": self.iters_per_epoch-1,
                            "validate": True})
        val_result = validator.test(self.args.ovthresh)
        return val_result

    def test(self):
        tester = Tester({"dataset": self.args.dataset,
                         "net": self.args.net,
                         "load_dir": self.args.save_dir,
                         "cuda": self.args.cuda,
                         "checksession": self.args.session,
                         "checkepoch": self.args.max_epochs,
                         "checkpoint": self.iters_per_epoch-1,
                         "validate": False})
        test_result = tester.test(self.args.ovthresh)
        return test_result

    def write_result_to_file(self, res, epoch, stage="val"):
        if not self.args.terminal_logging:
            with open(os.path.join(self.args.log_path,
                                   "performance.csv"),"a") as f:
                f.write(f"{datetime.now()},{stage},{epoch},{np.mean(res)}\n")

    def train(self):
        self.check_transfer_args()
        self.create_performance_file()
        logging.info("Loading data and configuring network")
        self.set_data_names()
        self.set_config()
        self.setup_data()
        self.create_output_dir()

        self.build_training_dataloader()

        self.initialise_tensor_data()

        self.construct_network()

        if self.args.resume:
            self.resume_network()

        if self.args.transfer:
            self.reconfigure_fc_layer()

        if self.args.mGPUs:
            self.fasterRCNN = nn.DataParallel(self.fasterRCNN)

        self.iters_per_epoch = int(self.train_size / self.args.batch_size)

        if self.args.use_tfboard:
            from tensorboardX import SummaryWriter
            logger = SummaryWriter("logs")

        for epoch in range(self.args.start_epoch, self.args.max_epochs + 1):
            logging.info(f"Training epoch {epoch} of {self.args.max_epochs}")
            self.train_epoch(epoch)
            logging.info(f"Validating epoch {epoch}")
            val_result = self.validate(epoch)
            self.write_result_to_file(val_result, epoch, "val")

        if self.args.and_test:
            logging.info(f"Testing using the final test set")
            test_result = self.test()
            self.write_result_to_file(test_result, epoch, "test")

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
    parser.add_argument('--optimising', dest='is_optimising',
                        help='Whether to use random seeds or not',
                        action="store_true")

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

    parser.add_argument('-v', dest="verbose", help="Use debug statements",
                        action="store_true")

    parser.add_argument('-c', dest="terminal_logging",
                        help="Log to terminal", action="store_true")

    parser.add_argument('--log_path', dest="log_path",
                        help="Path to folder for log file",
                        default=os.path.join(f"logs/"+\
                                f"{str(datetime.now()).replace(' ', '_')}"))
    # Testing params
    parser.add_argument('--ovthresh', dest="ovthresh", type=float,
                        help="Threhsold of overlap for correct BB", default=0.5)

    parser.add_argument('--and_test', dest="and_test", action="store_true",
                        help="Whether to do the final test run", default=False)



    return parser

if __name__ == '__main__':
    import coloredlogs
    def parse_args():
        """
        Parse input arguments
        """
        parser = build_parser()
        args = parser.parse_args()
        return args

    cli_args = parse_args()

    if cli_args.terminal_logging is True:
        verbosity = "DEBUG" if cli_args.verbose else "INFO"
        coloredlogs.install(level=verbosity,
                            fmt="%(asctime)s %(levelname)s %(module)s" + \
                                "- %(funcName)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    else:
        os.makedirs(cli_args.log_path)
        verbosity = logging.DEBUG if cli_args.verbose else logging.INFO
        logging.basicConfig(filename=os.path.join(cli_args.log_path, "log.log"),
                            level=verbosity,
                            format="%(asctime)s %(levelname)s %(module)s" +
                            "- %(funcName)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    logging.debug('Called with args:')
    logging.debug(cli_args)

    trainer = Trainer(cli_args, cli=True)
    trainer.train()
