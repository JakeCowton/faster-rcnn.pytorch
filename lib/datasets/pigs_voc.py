from __future__ import print_function
from __future__ import absolute_import

import logging

from os import listdir, path
import xml.etree.ElementTree as ET
import pickle

import numpy as np
import scipy

from .imdb import imdb


class pigs_voc(imdb):

    def __init__(self):
        imdb.__init__(self, "pigs_voc")
        self._classes = ('__background__',
                         'pig')
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))

        self._img_root = "/home/jake/pig_voc"
        self.cache_file = path.join(self._img_root, "cache")
        self._img_jpg_folder = "jpg_images"
        self._img_annotation_folder = "annotations"
        self._annotation_ext = ".xml"
        self._img_ext = ".jpg"
        self._image_filepaths = self._load_image_filepaths()

        self._image_index = self._load_image_set_index()

        self._roidb_handler = self.gt_roidb

        self.config = {'cleanup': True,
                       'use_salt': True,
                       'use_diff': False,
                       'matlab_eval': False,
                       'rpn_file': None,
                       'min_size': 2}

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self._image_filepaths[i]

    def image_id_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return i

    def image_path_from_index(self, i):
        """
        Construct an image path from the image's "index".
        """
        return self.image_path_at(index)

    def _load_image_filepaths(self):
        """
        Only return images that have corresponding XML files
        """
        filepaths = [path.join(self._img_root,
                               self._img_jpg_folder,
                               fn.replace(self._annotation_ext,
                                          self._img_ext))\
                     for fn in sorted(listdir(path.join(self._img_root,
                                                  self._img_annotation_folder)))]
        return filepaths

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = path.join(self.cache_file, self.name + '_gt_roidb.pkl')
        if path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = pickle.load(fid)
            logging.info(f"{self.name} gt roidb loaded from {cache_file}")
            return roidb

        gt_roidb = [self._load_annotation(path.basename(image_path))
                    for image_path in self._image_filepaths]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        logging.info(f"Wrote gt roidb to {cache_file}")

        return gt_roidb

    def _load_annotation(self, img_filename):
        filename = img_filename.replace(self._img_ext, self._annotation_ext)

        tree = ET.parse(path.join(self._img_root, self._img_annotation_folder, filename))
        objs = tree.findall('object')
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)
        ishards = np.zeros((num_objs), dtype=np.int32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1

            diffc = obj.find('difficult')
            difficult = 0 if diffc == None else int(diffc.text)
            ishards[ix] = difficult

            cls = self._class_to_ind[obj.find('name').text.lower().strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_ishard': ishards,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        return range(len(listdir(path.join(self._img_root,
                                           self._img_annotation_folder))))
