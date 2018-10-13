# --------------------------------------------------------
# Tensorflow Faster R-CNN
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
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
import logging
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

from utils import DictToArgs

import pdb

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


class Tester(object):
    """
    Runs the test set of the given dataset on a saved model
    """

    def __init__(self, args, cli=False):
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

    def set_data_names(self):
        if self.args.dataset == "pascal_voc":
            self.args.imdb_name = "voc_2007_trainval"
            self.args.imdbval_name = "voc_2007_test"
            self.args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                                  'ANCHOR_RATIOS', '[0.5,1,2]']
        elif self.args.dataset == "pigs_voc":
            self.args.imdb_name = "pigs_voc_train"
            self.args.imdbval_name = "pigs_voc_val"
            self.args.imdbtest_name = "pigs_voc_test"
            self.args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                                  'ANCHOR_RATIOS', '[0.5,1,2]',
                                  'MAX_NUM_GT_BOXES', '20']
        elif self.args.dataset == "pascal_voc_0712":
            self.args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
            self.args.imdbval_name = "voc_2007_test"
            self.args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                                  'ANCHOR_RATIOS', '[0.5,1,2]']
        elif self.args.dataset == "coco":
            self.args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
            self.args.imdbval_name = "coco_2014_minival"
            self.args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]',
                                  'ANCHOR_RATIOS', '[0.5,1,2]']
        elif self.args.dataset == "imagenet":
            self.args.imdb_name = "imagenet_train"
            self.args.imdbval_name = "imagenet_val"
            self.args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]',
                                  'ANCHOR_RATIOS', '[0.5,1,2]']
        elif self.args.dataset == "vg":
            self.args.imdb_name = "vg_150-50-50_minitrain"
            self.args.imdbval_name = "vg_150-50-50_minival"
            self.args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]',
                                  'ANCHOR_RATIOS', '[0.5,1,2]']

    def set_config(self):
        if torch.cuda.is_available() and not self.args.cuda:
            logging.warning("You have a CUDA device, so you should probably "+\
                            "run with --cuda")

        np.random.seed(cfg.RNG_SEED)

        self.args.cfg_file = "cfgs/{}_ls.yml".format(self.args.net)\
                             if self.args.large_scale \
                             else "cfgs/{}.yml".format(self.args.net)

        if self.args.cfg_file is not None:
            cfg_from_file(self.args.cfg_file)
        if self.args.set_cfgs is not None:
            cfg_from_list(self.args.set_cfgs)

        logging.debug('Using config:')
        pprint.pformat(cfg)

        cfg.TRAIN.USE_FLIPPED = False

    def setup_data(self):
        self.imdb, self.roidb, self.ratio_list, self.ratio_index = \
                combined_roidb(self.data_to_read, False)
        self.imdb.competition_mode(on=True)

        logging.debug('{:d} roidb entries'.format(len(self.roidb)))

    def create_input_dir(self):
        self.input_dir = os.path.join(self.args.load_dir,
                                 self.args.net,
                                 self.args.dataset)
        if not os.path.exists(self.input_dir):
            raise Exception('There is no input directory for loading '+\
                            'network from ' + input_dir)

    def construct_network(self):
        load_name = os.path.join(self.input_dir,
          'faster_rcnn_{}_{}_{}.pth'.format(self.args.checksession,
                                            self.args.checkepoch,
                                            self.args.checkpoint))
        # initilize the network here.
        if self.args.net == 'vgg16':
            self.fasterRCNN = vgg16(self.imdb.classes, pretrained=False,
                                    class_agnostic=self.args.class_agnostic)
        elif self.args.net == 'res101':
            self.fasterRCNN = resnet(self.imdb.classes, 101, pretrained=False,
                                     class_agnostic=self.args.class_agnostic)
        elif self.args.net == 'res50':
            self.fasterRCNN = resnet(self.imdb.classes, 50, pretrained=False,
                                     class_agnostic=self.args.class_agnostic)
        elif self.args.net == 'res152':
            self.fasterRCNN = resnet(self.imdb.classes, 152, pretrained=False,
                                     class_agnostic=self.args.class_agnostic)
        else:
            logging.error("network is not defined")
            pdb.set_trace()

        self.fasterRCNN.create_architecture()

        logging.debug("load checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name)
        self.fasterRCNN.load_state_dict(checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
            cfg.POOLING_MODE = checkpoint['pooling_mode']


        logging.debug('load model successfully!')

    def initialise_tensor_data(self):
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
        self.im_data = Variable(im_data)
        self.im_info = Variable(im_info)
        self.num_boxes = Variable(num_boxes)
        self.gt_boxes = Variable(gt_boxes)

        if self.args.cuda:
            cfg.CUDA = True

        if self.args.cuda:
            self.fasterRCNN.cuda()
    def pre_data_load_config(self):
        self.vis = self.args.vis
        if self.vis:
            self.thresh = 0.05
        else:
            self.thresh = 0.0

        self.save_name = 'faster_rcnn_10'
        self.num_images = len(self.imdb.image_index)
        self.all_boxes = [[[] for _ in xrange(self.num_images)]
                     for _ in xrange(self.imdb.num_classes)]

        self.output_dir = get_output_dir(self.imdb, self.save_name)
        self.max_per_image = 100

    def load_data(self):
        dataset = roibatchLoader(self.roidb,
                                 self.ratio_list,
                                 self.ratio_index,
                                 1,
                                 self.imdb.num_classes,
                                 training=False,
                                 normalize = False)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=1,
                                                 shuffle=False, num_workers=0,
                                                 pin_memory=True)

        data_iter = iter(dataloader)

        return data_iter

    def process_image(self, data_iter, i):
        data = next(data_iter)
        self.im_data.data.resize_(data[0].size()).copy_(data[0])
        self.im_info.data.resize_(data[1].size()).copy_(data[1])
        self.gt_boxes.data.resize_(data[2].size()).copy_(data[2])
        self.num_boxes.data.resize_(data[3].size()).copy_(data[3])

        det_tic = time.time()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = self.fasterRCNN(self.im_data,
                                     self.im_info,
                                     self.gt_boxes,
                                     self.num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
                if self.args.class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * \
                     torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() +\
                     torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * \
                     torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() +\
                     torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4 * \
                                 len(self.imdb.classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, self.im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= data[1][0][2].item()

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        det_toc = time.time()
        detect_time = det_toc - det_tic
        misc_tic = time.time()
        if self.vis:
            im = cv2.imread(self.imdb.image_path_at(i))
            im2show = np.copy(im)
        for j in xrange(1, self.imdb.num_classes):
            inds = torch.nonzero(scores[:,j]>self.thresh).view(-1)
            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:,j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if self.args.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]

                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
                # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_dets, cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if self.vis:
                    im2show = self.vis_detections(im2show,
                                                  self.imdb.classes[j],
                                                  cls_dets.cpu().numpy(),
                                                  0.3)
                self.all_boxes[j][i] = cls_dets.cpu().numpy()
            else:
                self.all_boxes[j][i] = empty_array

        # Limit to self.max_per_image detections *over all classes*
        if self.max_per_image > 0:
            image_scores = np.hstack([self.all_boxes[j][i][:, -1]
                                      for j in xrange(1,
                                                      self.imdb.num_classes)])
            if len(image_scores) > self.max_per_image:
                image_thresh = np.sort(image_scores)[-self.max_per_image]
                for j in xrange(1, self.imdb.num_classes):
                    keep = np.where(self.all_boxes[j][i][:, -1] >=\
                                    image_thresh)[0]
                    self.all_boxes[j][i] = self.all_boxes[j][i][keep, :]

        misc_toc = time.time()
        nms_time = misc_toc - misc_tic

        sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
            .format(i + 1, self.num_images, detect_time, nms_time))
        sys.stdout.flush()

        if self.vis:
            cv2.imwrite('result.png', im2show)
            pdb.set_trace()
            #cv2.imshow('test', im2show)
            #cv2.waitKey(0)

    def evaluate_all_images(self):
        data_iter = self.load_data()

        self.fasterRCNN.eval()
        empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
        for i in range(self.num_images):
            self.process_image(data_iter, i)

    def pickle_detections(self):
        det_file = os.path.join(self.output_dir, 'detections.pkl')
        with open(det_file, 'wb') as f:
            pickle.dump(self.all_boxes, f, pickle.HIGHEST_PROTOCOL)

    def test(self):
        self.set_data_names()
        self.data_to_read = self.args.imdbval_name \
                            if self.args.validate is True \
                            else self.args.imdbtest_name
        self.set_config()
        self.setup_data()
        self.create_input_dir()
        self.construct_network()
        self.initialise_tensor_data()
        self.pre_data_load_config()

        start = time.time()

        logging.debug(f"Processing image set {self.imdb._name}")
        self.evaluate_all_images()
        self.pickle_detections()

        logging.debug('Evaluating detections')
        self.imdb.evaluate_detections(self.all_boxes,
                                      self.output_dir,
                                      self.args.validate)

        end = time.time()
        logging.debug("test time: %0.4fs" % (end - start))

def build_parser():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='pascal_voc', type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfgs/vgg16.yml', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res50, res101, res152',
                        default='res101', type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--load_dir', dest='load_dir',
                        help='directory to load models', default="models",
                        type=str)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')
    parser.add_argument('--parallel_type', dest='parallel_type',
                        help='which part of model to parallel, 0: all, '+\
                             '1: model before roi pooling',
                        default=0, type=int)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load network',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load network',
                        default=10021, type=int)
    parser.add_argument('--vis', dest='vis',
                        help='visualization mode',
                        action='store_true')
    parser.add_argument('-v', dest="verbose", help="Use debug statements",
                        action="store_true")
    return parser

if __name__ == '__main__':
    import coloredlogs
    cli_args = build_parser().parse_args()

    verbosity = "DEBUG" if cli_args.verbose else "INFO"
    coloredlogs.install(level=verbosity,
                        format="%(asctime)s %(levelname)s %(module)s" + \
                               "- %(funcName)s: %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    logging.info('Called with args:')
    logging.info(cli_args)

    tester = Tester(cli_args, cli=True)
    tester.test()
