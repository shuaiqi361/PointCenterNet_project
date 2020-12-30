import os
import cv2
import json
import math
import numpy as np

import torch
import torch.utils.data as data
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as cocomask
from scipy.signal import resample

from utils.image import get_border, get_affine_transform, affine_transform, color_aug
from utils.image import draw_umich_gaussian, gaussian_radius
from utils.sparse_coding import fast_ista, check_clockwise_polygon, get_connected_polygon_using_mask, turning_angle_resample

COCO_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
              'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
              'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
              'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
              'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
              'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
              'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
              'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
              'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
              'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
              'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
              'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
              'scissors', 'teddy bear', 'hair drier', 'toothbrush']

COCO_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90]

COCO_MEAN = [0.40789654, 0.44719302, 0.47026115]
COCO_STD = [0.28863828, 0.27408164, 0.27809835]
COCO_EIGEN_VALUES = [0.2141788, 0.01817699, 0.00341571]
COCO_EIGEN_VECTORS = [[-0.58752847, -0.69563484, 0.41340352],
                      [-0.5832747, 0.00994535, -0.81221408],
                      [-0.56089297, 0.71832671, 0.41158938]]


def encode_mask(mask):
    """Convert mask to coco rle"""
    rle = cocomask.encode(np.array(mask[:, :, np.newaxis], order='F'))[0]
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


class COCOSEGMCMM(data.Dataset):
    def __init__(self, data_dir, dictionary_file, split, split_ratio=1.0, img_size=512, padding=31):
        super(COCOSEGMCMM, self).__init__()
        self.num_classes = 80
        self.class_name = COCO_NAMES
        self.valid_ids = COCO_IDS
        self.cat_ids = {v: i for i, v in enumerate(self.valid_ids)}

        self.data_rng = np.random.RandomState(900)
        self.eig_val = np.array(COCO_EIGEN_VALUES, dtype=np.float32)
        self.eig_vec = np.array(COCO_EIGEN_VECTORS, dtype=np.float32)
        self.mean = np.array(COCO_MEAN, dtype=np.float32)[None, None, :]
        self.std = np.array(COCO_STD, dtype=np.float32)[None, None, :]

        self.split = split
        self.dictionary_file = dictionary_file
        self.data_dir = data_dir
        self.img_dir = os.path.join(self.data_dir, '%s2017' % split)
        if split == 'test':
            self.annot_path = os.path.join(self.data_dir, 'annotations', 'image_info_test-dev2017.json')
        else:
            self.annot_path = os.path.join(self.data_dir, 'annotations', 'instances_%s2017.json' % split)

        self.max_objs = 128
        self.padding = padding  # 31 for resnet/resdcn
        self.down_ratio = 4
        self.img_size = {'h': img_size, 'w': img_size}
        self.fmap_size = {'h': img_size // self.down_ratio, 'w': img_size // self.down_ratio}
        self.rand_scales = np.arange(0.6, 1.3, 0.1)
        self.gaussian_iou = 0.7

        self.n_vertices = 32
        self.n_codes = 64
        self.sparse_alpha = 0.01

        print('==> initializing coco 2017 %s data.' % split)
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.dictionary = np.load(self.dictionary_file)  # ndarray, shape (n_coeffs, n_vertices * 2)

        if 0 < split_ratio < 1:
            split_size = int(np.clip(split_ratio * len(self.images), 1, len(self.images)))
            self.images = self.images[:split_size]

        self.num_samples = len(self.images)

        print('Loaded %d %s samples' % (self.num_samples, split))

    def __getitem__(self, index):
        img_id = self.images[index]
        img_path = os.path.join(self.img_dir, self.coco.loadImgs(ids=[img_id])[0]['file_name'])
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        annotations = self.coco.loadAnns(ids=ann_ids)
        img = self.coco.loadImgs(ids=[img_id])[0]
        w_img = int(img['width'])
        h_img = int(img['height'])

        labels = []
        bboxes = []
        shapes = []

        for anno in annotations:
            if anno['iscrowd'] == 1:  # Excludes crowd objects
                continue

            polygons = get_connected_polygon_using_mask(anno['segmentation'], (h_img, w_img),
                                                        n_vertices=self.n_vertices, closing_max_kernel=50)

            gt_x1, gt_y1, gt_w, gt_h = anno['bbox']
            contour = np.array(polygons).reshape((-1, 2))

            # Downsample the contour to fix number of vertices
            if len(contour) > self.n_vertices:
                fixed_contour = resample(contour, num=self.n_vertices)
            else:
                fixed_contour = turning_angle_resample(contour, self.n_vertices)

            fixed_contour[:, 0] = np.clip(fixed_contour[:, 0], gt_x1, gt_x1 + gt_w)
            fixed_contour[:, 1] = np.clip(fixed_contour[:, 1], gt_y1, gt_y1 + gt_h)

            contour_std = np.sqrt(np.sum(np.std(fixed_contour, axis=0) ** 2))
            if contour_std < 1e-6 or contour_std == np.inf or contour_std == np.nan:  # invalid shapes
                continue

            updated_bbox = [np.min(fixed_contour[:, 0]), np.min(fixed_contour[:, 1]),
                            np.max(fixed_contour[:, 0]), np.max(fixed_contour[:, 1])]

            shapes.append(np.ndarray.flatten(fixed_contour).tolist())
            labels.append(self.cat_ids[anno['category_id']])
            bboxes.append(updated_bbox)

        labels = np.array(labels)
        bboxes = np.array(bboxes, dtype=np.float32)
        shapes = np.array(shapes, dtype=np.float32)

        if len(bboxes) == 0:
            bboxes = np.array([[0., 0., 0., 0.]], dtype=np.float32)
            labels = np.array([[0]])
            shapes = np.zeros((1, self.n_vertices * 2), dtype=np.float32)
        # bboxes[:, 2:] += bboxes[:, :2]  # xywh to xyxy

        img = cv2.imread(img_path)
        height, width = img.shape[0], img.shape[1]
        center = np.array([width / 2., height / 2.], dtype=np.float32)  # center of image
        scale = max(height, width) * 1.0

        flipped = False
        if self.split == 'train':
            scale = scale * np.random.choice(self.rand_scales)
            w_border = get_border(128, width)
            h_border = get_border(128, height)
            center[0] = np.random.randint(low=w_border, high=width - w_border)
            center[1] = np.random.randint(low=h_border, high=height - h_border)

            if np.random.random() < 0.5:
                flipped = True
                img = img[:, ::-1, :]
                center[0] = width - center[0] - 1

        trans_img = get_affine_transform(center, scale, 0, [self.img_size['w'], self.img_size['h']])
        img = cv2.warpAffine(img, trans_img, (self.img_size['w'], self.img_size['h']))

        # -----------------------------------debug---------------------------------
        # image_show = img.copy()
        # for bbox, label in zip(bboxes, labels):
        #   if flipped:
        #     bbox[[0, 2]] = width - bbox[[2, 0]] - 1
        #   bbox[:2] = affine_transform(bbox[:2], trans_img)
        #   bbox[2:] = affine_transform(bbox[2:], trans_img)
        #   bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.img_size['w'] - 1)
        #   bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.img_size['h'] - 1)
        #   cv2.rectangle(image_show, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
        #   cv2.putText(image_show, self.class_name[label + 1], (int(bbox[0]), int(bbox[1])),
        #               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # cv2.imshow('img', image_show)
        # cv2.waitKey()
        # -----------------------------------debug---------------------------------

        img = img.astype(np.float32) / 255.

        if self.split == 'train':
            color_aug(self.data_rng, img, self.eig_val, self.eig_vec)

        img -= self.mean
        img /= self.std
        img = img.transpose(2, 0, 1)  # from [H, W, C] to [C, H, W]

        trans_fmap = get_affine_transform(center, scale, 0, [self.fmap_size['w'], self.fmap_size['h']])

        hmap = np.zeros((self.num_classes, self.fmap_size['h'], self.fmap_size['w']), dtype=np.float32)  # heatmap
        w_h_ = np.zeros((self.max_objs, 2), dtype=np.float32)  # width and height of inmodal bboxes
        shapes_ = np.zeros((self.max_objs, self.n_vertices * 2), dtype=np.float32)  # gt amodal segmentation polygons
        center_offsets = np.zeros((self.max_objs, 2), dtype=np.float32)  # gt amodal mass centers to inmodal bbox center
        codes_ = np.zeros((self.max_objs, self.n_codes), dtype=np.float32)
        regs = np.zeros((self.max_objs, 2), dtype=np.float32)  # regression for offsets of shape center
        inds = np.zeros((self.max_objs,), dtype=np.int64)
        ind_masks = np.zeros((self.max_objs,), dtype=np.uint8)

        # detections = []
        for k, (bbox, label, shape) in enumerate(zip(bboxes, labels, shapes)):
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
                # Flip the contour x-axis
                for m in range(self.n_vertices):
                    shape[2 * m] = width - shape[2 * m] - 1

            bbox[:2] = affine_transform(bbox[:2], trans_fmap)
            bbox[2:] = affine_transform(bbox[2:], trans_fmap)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.fmap_size['w'] - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.fmap_size['h'] - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]

            # generate gt shape mean and std from contours
            for m in range(self.n_vertices):  # apply scale and crop transform to shapes
                shape[2 * m:2 * m + 2] = affine_transform(shape[2 * m:2 * m + 2], trans_fmap)

            shape_clipped = np.reshape(shape, (self.n_vertices, 2))

            shape_clipped[:, 0] = np.clip(shape_clipped[:, 0], 0, self.fmap_size['w'] - 1)
            shape_clipped[:, 1] = np.clip(shape_clipped[:, 1], 0, self.fmap_size['h'] - 1)

            clockwise_flag = check_clockwise_polygon(shape_clipped)
            if not clockwise_flag:
                fixed_contour = np.flip(shape_clipped, axis=0)
            else:
                fixed_contour = shape_clipped.copy()
            # Indexing from the left-most vertex, argmin x-axis
            idx = np.argmin(fixed_contour[:, 0])
            indexed_shape = np.concatenate((fixed_contour[idx:, :], fixed_contour[:idx, :]), axis=0)

            mass_center = np.mean(indexed_shape, axis=0)
            if h < 1e-6 or w < 1e-6:  # remove small bboxes
                continue

            centered_shape = indexed_shape - mass_center

            if h > 0 and w > 0:
                obj_c = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                obj_c_int = obj_c.astype(np.int32)

                radius = max(0, int(gaussian_radius((math.ceil(h), math.ceil(w)), self.gaussian_iou)))
                draw_umich_gaussian(hmap[label], obj_c_int, radius)
                shapes_[k] = centered_shape.reshape((1, -1))
                center_offsets[k] = mass_center - obj_c
                codes_[k], _ = fast_ista(centered_shape.reshape((1, -1)), self.dictionary,
                                         lmbda=self.sparse_alpha, max_iter=60)
                w_h_[k] = 1. * w, 1. * h
                regs[k] = obj_c - obj_c_int  # discretization error
                inds[k] = obj_c_int[1] * self.fmap_size['w'] + obj_c_int[0]
                ind_masks[k] = 1

        # detections = np.array(detections, dtype=np.float32) \
        #   if len(detections) > 0 else np.zeros((1, 6), dtype=np.float32)

        # -----------------------------------debug---------------------------------
        # canvas = np.zeros((self.fmap_size['h'] * 2, self.fmap_size['w'] * 2, 3), dtype=np.float32)
        # canvas[0:self.fmap_size['h'], 0:self.fmap_size['w'], :] = np.tile(np.expand_dims(hmap[0], 2), (1, 1, 3))
        # canvas[0:self.fmap_size['h'], self.fmap_size['w']:, :] = np.tile(np.expand_dims(hmap[1], 2), (1, 1, 3))
        # canvas[self.fmap_size['h']:, 0:self.fmap_size['w'], :] = np.tile(np.expand_dims(hmap[2], 2), (1, 1, 3))
        # canvas[self.fmap_size['h']:, self.fmap_size['w']:, :] = np.tile(np.expand_dims(hmap[3], 2), (1, 1, 3))
        # print(w_h_[0], regs[0])
        # cv2.imshow('hmap', canvas)
        # cv2.waitKey()
        # -----------------------------------debug---------------------------------

        return {'image': img, 'shapes': shapes_, 'codes': codes_, 'offsets': center_offsets,
                'hmap': hmap, 'w_h_': w_h_, 'regs': regs, 'inds': inds, 'ind_masks': ind_masks,
                'c': center, 's': scale, 'img_id': img_id}

    def __len__(self):
        return self.num_samples


class COCO_eval_segm_cmm(COCOSEGMCMM):
    def __init__(self, data_dir, dictionary_file, split, test_scales=(1,), test_flip=False, fix_size=False, padding=31):
        super(COCO_eval_segm_cmm, self).__init__(data_dir, dictionary_file, split, padding)
        self.test_flip = test_flip
        self.test_scales = test_scales
        self.fix_size = fix_size

    def __getitem__(self, index):
        img_id = self.images[index]
        img_path = os.path.join(self.img_dir, self.coco.loadImgs(ids=[img_id])[0]['file_name'])
        image = cv2.imread(img_path)
        height, width = image.shape[0:2]

        out = {}
        for scale in self.test_scales:
            new_height = int(height * scale)
            new_width = int(width * scale)

            if self.fix_size:
                img_height, img_width = self.img_size['h'], self.img_size['w']
                center = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
                scaled_size = max(height, width) * 1.0
                scaled_size = np.array([scaled_size, scaled_size], dtype=np.float32)
            else:
                img_height = (new_height | self.padding) + 1
                img_width = (new_width | self.padding) + 1
                center = np.array([new_width // 2, new_height // 2], dtype=np.float32)
                scaled_size = np.array([img_width, img_height], dtype=np.float32)

            img = cv2.resize(image, (new_width, new_height))
            trans_img = get_affine_transform(center, scaled_size, 0, [img_width, img_height])
            img = cv2.warpAffine(img, trans_img, (img_width, img_height))

            img = img.astype(np.float32) / 255.
            img -= self.mean
            img /= self.std
            img = img.transpose(2, 0, 1)[None, :, :, :]  # from [H, W, C] to [1, C, H, W]

            if self.test_flip:
                img = np.concatenate((img, img[:, :, :, ::-1].copy()), axis=0)

            out[scale] = {'image': img,
                          'center': center,
                          'scale': scaled_size,
                          'fmap_h': img_height // self.down_ratio,
                          'fmap_w': img_width // self.down_ratio}

        return img_id, out

    def convert_eval_format(self, all_segments):
        # all_bboxes: num_samples x num_classes x 5
        segments = []
        for image_id in all_segments:
            img = self.coco.loadImgs(image_id)[0]
            w_img = int(img['width'])
            h_img = int(img['height'])
            for cls_ind in all_segments[image_id]:
                category_id = self.valid_ids[cls_ind - 1]
                for segm in all_segments[image_id][cls_ind]:  # decode the segments to RLE
                    poly = segm[:-5].reshape((-1, 2))

                    x1, y1, x2, y2 = segm[-5:-1]
                    bbox = [x1, y1, x2 - x1, y2 - y1]
                    bbox_out = list(map(lambda x: float("{:.2f}".format(x)), bbox))

                    poly = np.ndarray.flatten(poly).tolist()

                    rles = cocomask.frPyObjects([poly], h_img, w_img)
                    rle = cocomask.merge(rles)
                    m = cocomask.decode(rle)
                    rle_new = encode_mask(m.astype(np.uint8))
                    score = segm[-1]

                    detection = {"image_id": int(image_id),
                                 "category_id": int(category_id),
                                 'segmentation': rle_new,
                                 'bbox': bbox_out,
                                 "score": float("{:.2f}".format(score))}
                    segments.append(detection)
        return segments

    def run_eval(self, results, save_dir=None):
        segments = self.convert_eval_format(results)

        if save_dir is not None:
            result_json = os.path.join(save_dir, "segm_iterative_cmm_results.json")
            json.dump(segments, open(result_json, "w"))

        coco_segms = self.coco.loadRes(segments)
        coco_eval_seg = COCOeval(self.coco, coco_segms, "segm")
        coco_eval_seg.evaluate()
        coco_eval_seg.accumulate()
        coco_eval_seg.summarize()

        coco_eval = COCOeval(self.coco, coco_segms, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return coco_eval_seg.stats

    @staticmethod
    def collate_fn(batch):
        out = []
        for img_id, sample in batch:
            out.append((img_id, {s: {k: torch.from_numpy(sample[s][k]).float()
            if k == 'image' else np.array(sample[s][k]) for k in sample[s]} for s in sample}))
        return out
