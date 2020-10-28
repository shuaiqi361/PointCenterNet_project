import os
import sys

from pathlib import Path

current_path = Path(Path(__file__).parent.absolute())
sys.path.append(str(current_path.parent))

import cv2
import argparse
import numpy as np
import time
from scipy.signal import resample
import json
from pycocotools.coco import COCO
from pycocotools import mask as cocomask
from pycocotools.cocoeval import COCOeval
from utils.sparse_coding import fast_ista, check_clockwise_polygon

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

import torch
import torch.utils.data

from datasets.coco import COCO_MEAN, COCO_STD, COCO_NAMES
from datasets.detrac import DETRAC_MEAN, DETRAC_STD, DETRAC_NAMES

from nets.hourglass_segm import exkp

from utils.utils import load_demo_model
from utils.image import transform_preds, get_affine_transform
from utils.post_process import ctsegm_decode

from lib.nms.nms import soft_nms

COCO_COLORS = sns.color_palette('hls', len(COCO_NAMES))
DETRAC_COLORS = sns.color_palette('hls', len(DETRAC_NAMES))

# Training settings
parser = argparse.ArgumentParser(description='centernet_traffic')

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='./')
parser.add_argument('--data_type', type=str, default='val2017')
parser.add_argument('--ckpt_dir', type=str, default='./ckpt/checkpoint.t7')
parser.add_argument('--dictionary_file', type=str)

parser.add_argument('--arch', type=str, default='large_hourglass')
parser.add_argument('--dataset', type=str, default='coco', choices=['coco', 'DETRAC', 'pascal'])
parser.add_argument('--img_size', type=int, default=512)

parser.add_argument('--test_flip', action='store_true')
parser.add_argument('--test_scales', type=str, default='1')  # 0.5,0.75,1,1.25,1.5
parser.add_argument('--test_topk', type=int, default=100)
parser.add_argument('--detect_thres', type=float, default=0.3)
parser.add_argument('--num_vertices', type=int, default=32)
parser.add_argument('--n_codes', type=int, default=64)
parser.add_argument('--detector_name', type=str, default='CenterNetSegm')

cfg = parser.parse_args()

os.chdir(cfg.root_dir)

cfg.test_scales = [float(s) for s in cfg.test_scales.split(',')]


def main():
    cfg.device = torch.device('cuda')
    torch.backends.cudnn.benchmark = False

    max_per_image = 100
    num_classes = 80 if cfg.dataset == 'coco' else 4
    dictionary = np.load(cfg.dictionary_file)

    colors = COCO_COLORS if cfg.dataset == 'coco' else DETRAC_COLORS
    names = COCO_NAMES if cfg.dataset == 'coco' else DETRAC_NAMES
    for j in range(len(names)):
        col_ = [c * 255 for c in colors[j]]
        colors[j] = tuple(col_)

    print('Creating model and recover from checkpoint ...')
    if 'hourglass' in cfg.arch:
        model = exkp(n=5, nstack=2, dims=[256, 256, 384, 384, 384, 512],
                     modules=[2, 2, 2, 2, 2, 4], num_classes=num_classes)
    else:
        raise NotImplementedError

    model = load_demo_model(model, cfg.ckpt_dir)
    model = model.to(cfg.device)
    model.eval()

    # Loading COCO validation images
    annotation_file = '{}/annotations/instances_{}.json'.format(cfg.data_dir, cfg.data_type)
    coco = COCO(annotation_file)

    # Load all annotations
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    catIds = coco.getCatIds(catNms=nms)
    imgIds = coco.getImgIds(catIds=catIds)
    annIds = coco.getAnnIds(catIds=catIds)
    all_anns = coco.loadAnns(ids=annIds)

    for annotation in all_anns:
        if annotation['iscrowd'] == 1 or type(annotation['segmentation']) != list:
            continue

        img = coco.loadImgs(annotation['image_id'])[0]
        image_path = '%s/images/%s/%s' % (cfg.data_dir, cfg.data_type, img['file_name'])
        w_img = int(img['width'])
        h_img = int(img['height'])
        if w_img < 1 or h_img < 1:
            continue

        polygons = annotation['segmentation'][0]
        gt_bbox = annotation['bbox']
        gt_x1, gt_y1, gt_w, gt_h = gt_bbox
        contour = np.array(polygons).reshape((-1, 2))

        # Downsample the contour to fix number of vertices
        fixed_contour = resample(contour, num=cfg.num_vertices)

        # Indexing from the left-most vertex, argmin x-axis
        idx = np.argmin(fixed_contour[:, 0])
        indexed_shape = np.concatenate((fixed_contour[idx:, :], fixed_contour[:idx, :]), axis=0)

        clockwise_flag = check_clockwise_polygon(indexed_shape)
        if not clockwise_flag:
            fixed_contour = np.flip(indexed_shape, axis=0)
        else:
            fixed_contour = indexed_shape.copy()

        fixed_contour[:, 0] = np.clip(fixed_contour[:, 0], gt_x1, gt_x1 + gt_w)
        fixed_contour[:, 1] = np.clip(fixed_contour[:, 1], gt_y1, gt_y1 + gt_h)

        contour_mean = np.mean(fixed_contour, axis=0)
        contour_std = np.std(fixed_contour, axis=0)
        # norm_shape = (fixed_contour - contour_mean) / np.sqrt(np.sum(contour_std ** 2.))

        # plot gt mean and std
        # image = cv2.imread(image_path)
        # # cv2.ellipse(image, center=(int(contour_mean[0]), int(contour_mean[1])),
        # #             axes=(int(contour_std[0]), int(contour_std[1])),
        # #             angle=0, startAngle=0, endAngle=360, color=(0, 255, 0),
        # #             thickness=2)
        # cv2.rectangle(image, pt1=(int(contour_mean[0] - contour_std[0] / 2.), int(contour_mean[1] - contour_std[1] / 2.)),
        #               pt2=(int(contour_mean[0] + contour_std[0] / 2.), int(contour_mean[1] + contour_std[1] / 2.)),
        #               color=(0, 255, 0), thickness=2)
        # cv2.polylines(image, [fixed_contour.astype(np.int32)], True, (0, 0, 255))
        # cv2.rectangle(image, pt1=(int(min(fixed_contour[:, 0])), int(min(fixed_contour[:, 1]))),
        #               pt2=(int(max(fixed_contour[:, 0])), int(max(fixed_contour[:, 1]))),
        #               color=(255, 0, 0), thickness=2)
        # cv2.imshow('GT segments', image)
        # if cv2.waitKey() & 0xFF == ord('q'):
        #     break

        image = cv2.imread(image_path)
        original_image = image.copy()
        height, width = image.shape[0:2]
        padding = 127 if 'hourglass' in cfg.arch else 31
        imgs = {}
        for scale in cfg.test_scales:
            new_height = int(height * scale)
            new_width = int(width * scale)

            if cfg.img_size > 0:
                img_height, img_width = cfg.img_size, cfg.img_size
                center = np.array([new_width / 2., new_height / 2.], dtype=np.float32)
                scaled_size = max(height, width) * 1.0
                scaled_size = np.array([scaled_size, scaled_size], dtype=np.float32)
            else:
                img_height = (new_height | padding) + 1
                img_width = (new_width | padding) + 1
                center = np.array([new_width // 2, new_height // 2], dtype=np.float32)
                scaled_size = np.array([img_width, img_height], dtype=np.float32)

            img = cv2.resize(image, (new_width, new_height))
            trans_img = get_affine_transform(center, scaled_size, 0, [img_width, img_height])
            img = cv2.warpAffine(img, trans_img, (img_width, img_height))

            img = img.astype(np.float32) / 255.
            img -= np.array(COCO_MEAN if cfg.dataset == 'coco' else DETRAC_MEAN, dtype=np.float32)[None, None, :]
            img /= np.array(COCO_STD if cfg.dataset == 'coco' else DETRAC_STD, dtype=np.float32)[None, None, :]
            img = img.transpose(2, 0, 1)[None, :, :, :]  # from [H, W, C] to [1, C, H, W]

            # if cfg.test_flip:
            #     img = np.concatenate((img, img[:, :, :, ::-1].copy()), axis=0)

            imgs[scale] = {'image': torch.from_numpy(img).float(),
                           'center': np.array(center),
                           'scale': np.array(scaled_size),
                           'fmap_h': np.array(img_height // 4),
                           'fmap_w': np.array(img_width // 4)}

        with torch.no_grad():
            segmentations = []
            start_time = time.time()
            for scale in imgs:
                imgs[scale]['image'] = imgs[scale]['image'].to(cfg.device)

                output = model(imgs[scale]['image'])[-1]
                segms = ctsegm_decode(*output, torch.from_numpy(dictionary.astype(np.float32)).to(cfg.device),
                                      K=cfg.test_topk)
                segms = segms.detach().cpu().numpy().reshape(1, -1, segms.shape[2])[0]

                top_preds = {}
                for j in range(cfg.num_vertices):
                    segms[:, 2 * j:2 * j + 2] = transform_preds(segms[:, 2 * j:2 * j + 2],
                                                                imgs[scale]['center'],
                                                                imgs[scale]['scale'],
                                                                (imgs[scale]['fmap_w'], imgs[scale]['fmap_h']))

                clses = segms[:, -1]
                for j in range(num_classes):
                    inds = (clses == j)
                    top_preds[j + 1] = segms[inds, :cfg.num_vertices * 2 + 1].astype(np.float32)
                    top_preds[j + 1][:, :cfg.num_vertices * 2] /= scale

                segmentations.append(top_preds)

            segms_and_scores = {j: np.concatenate([d[j] for d in segmentations], axis=0)
                                for j in range(1, num_classes + 1)}  # a Dict label: segments
            scores = np.hstack(
                [segms_and_scores[j][:, cfg.num_vertices * 2] for j in range(1, num_classes + 1)])

            if len(scores) > max_per_image:
                kth = len(scores) - max_per_image
                thresh = np.partition(scores, kth)[kth]
                for j in range(1, num_classes + 1):
                    keep_inds = (segms_and_scores[j][:, cfg.num_vertices * 2] >= thresh)
                    segms_and_scores[j] = segms_and_scores[j][keep_inds]

            # Use opencv functions to output a video
            output_image = original_image

            for lab in segms_and_scores:
                for res in segms_and_scores[lab]:
                    contour, score = res[:-1], res[-1]
                    if score > cfg.detect_thres:
                        text = names[lab] + ' %.2f' % score
                        label_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)
                        polygon = contour.reshape((-1, 2))
                        contour_mean = np.mean(polygon, axis=0)
                        contour_std = np.std(polygon, axis=0)
                        center_x, center_y = np.mean(polygon, axis=0).astype(np.int32)
                        text_location = [center_x, center_y,
                                         center_x + label_size[0][0],
                                         center_y + label_size[0][1]]
                        # cv2.rectangle(output_image, pt1=(int(contour_mean[0] - contour_std[0] / 2.), int(contour_mean[1] - contour_std[1] / 2.)),
                        #               pt2=(int(contour_mean[0] + contour_std[0] / 2.), int(contour_mean[1] + contour_std[1] / 2.)),
                        #               color=(0, 255, 0), thickness=1)
                        cv2.polylines(output_image, [polygon.astype(np.int32)], True, (255, 0, 0), thickness=2)
                        # cv2.putText(output_image, text, org=(int(text_location[0]), int(text_location[3])),
                        #             fontFace=cv2.FONT_HERSHEY_COMPLEX, thickness=1, fontScale=0.5,
                        #             color=(0, 0, 255))

            cv2.imshow('Results', output_image)
            if cv2.waitKey() & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    main()
