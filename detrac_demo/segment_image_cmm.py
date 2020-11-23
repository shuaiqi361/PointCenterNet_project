import os
import sys

from pathlib import Path

current_path = Path(Path(__file__).parent.absolute())
sys.path.append(str(current_path.parent))
import random

import cv2
import argparse
import numpy as np
import time
from scipy.signal import resample
import json
from pycocotools.coco import COCO
from pycocotools import mask as cocomask
from pycocotools.cocoeval import COCOeval
from utils.sparse_coding import fast_ista, check_clockwise_polygon, \
    get_connected_polygon_using_mask, turning_angle_resample

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
from bounding_box import bounding_box as bb

import torch
import torch.utils.data

from datasets.coco import COCO_MEAN, COCO_STD, COCO_NAMES
from datasets.detrac import DETRAC_MEAN, DETRAC_STD, DETRAC_NAMES

from nets.hourglass_segm_cmm import exkp

from utils.utils import load_demo_model
from utils.image import transform_preds, get_affine_transform
from utils.post_process import ctsegm_shift_decode, ctsegm_cmm_decode

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
COLOR_WORLD = ['navy', 'blue', 'aqua', 'teal', 'olive', 'green', 'lime', 'yellow', 'orange', 'red', 'maroon',
               'fuchsia', 'purple', 'black', 'gray', 'silver']
RGB_DICT = {'navy': (0, 38, 63), 'blue': (0, 120, 210), 'aqua': (115, 221, 252), 'teal': (15, 205, 202),
            'olive': (52, 153, 114), 'green': (0, 204, 84), 'lime': (1, 255, 127), 'yellow': (255, 216, 70),
            'orange': (255, 125, 57), 'red': (255, 47, 65), 'maroon': (135, 13, 75), 'fuchsia': (246, 0, 184),
            'purple': (179, 17, 193), 'black': (24, 24, 24), 'gray': (168, 168, 168), 'silver': (220, 220, 220)}

for k, v in RGB_DICT.items():
    RGB_DICT[k] = (v[2], v[1], v[0])  # RGB to BGR


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
                     modules=[2, 2, 2, 2, 2, 4],
                     dictionary=torch.from_numpy(dictionary.astype(np.float32)).to(cfg.device))
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
    # imgIds = coco.getImgIds(catIds=catIds)
    imgIds = coco.getImgIds()
    # annIds = coco.getAnnIds(catIds=catIds)
    # all_anns = coco.loadAnns(ids=annIds)
    # print(len(imgIds), imgIds)

    for id in imgIds:
        annt_ids = coco.getAnnIds(imgIds=[id])
        annotations_per_img = coco.loadAnns(ids=annt_ids)
        # print('All annots: ', len(annotations_per_img), annotations_per_img)
        img = coco.loadImgs(id)[0]
        image_path = '%s/images/%s/%s' % (cfg.data_dir, cfg.data_type, img['file_name'])
        w_img = int(img['width'])
        h_img = int(img['height'])
        if w_img < 1 or h_img < 1:
            continue

        img_original = cv2.imread(image_path)
        img_connect = cv2.imread(image_path)
        img_recon = cv2.imread(image_path)
        print('Image id: ', id)

        for annt in annotations_per_img:
            if annt['iscrowd'] == 1 or type(annt['segmentation']) != list:
                continue

            polygons = get_connected_polygon_using_mask(annt['segmentation'], (h_img, w_img),
                                                        n_vertices=cfg.num_vertices, closing_max_kernel=60)
            gt_bbox = annt['bbox']
            gt_x1, gt_y1, gt_w, gt_h = gt_bbox
            contour = np.array(polygons).reshape((-1, 2))

            # Downsample the contour to fix number of vertices
            if len(contour) > cfg.num_vertices:
                resampled_contour = resample(contour, num=cfg.num_vertices)
            else:
                resampled_contour = turning_angle_resample(contour, cfg.num_vertices)

            resampled_contour[:, 0] = np.clip(resampled_contour[:, 0], gt_x1, gt_x1 + gt_w)
            resampled_contour[:, 1] = np.clip(resampled_contour[:, 1], gt_y1, gt_y1 + gt_h)

            clockwise_flag = check_clockwise_polygon(resampled_contour)
            if not clockwise_flag:
                fixed_contour = np.flip(resampled_contour, axis=0)
            else:
                fixed_contour = resampled_contour.copy()

            # Indexing from the left-most vertex, argmin x-axis
            idx = np.argmin(fixed_contour[:, 0])
            indexed_shape = np.concatenate((fixed_contour[idx:, :], fixed_contour[:idx, :]), axis=0)

            x1, y1, x2, y2 = gt_x1, gt_y1, gt_x1 + gt_w, gt_y1 + gt_h

            # bbox_width, bbox_height = x2 - x1, y2 - y1
            # bbox = [x1, y1, bbox_width, bbox_height]
            bbox_center = np.array([(x1 + x2) / 2., (y1 + y2) / 2.])
            # bbox_center = np.mean(indexed_shape, axis=0)

            centered_shape = indexed_shape - bbox_center

            # visualize resampled points with multiple parts in image side by side
            for cnt in range(len(annt['segmentation'])):
                polys = np.array(annt['segmentation'][cnt]).reshape((-1, 2))
                cv2.polylines(img_original, [polys.astype(np.int32)], True, (10, 10, 255), thickness=2)

            cv2.polylines(img_connect, [indexed_shape.astype(np.int32)], True, (150, 10, 255), thickness=2)

            # learned_val_codes, _ = fast_ista(centered_shape.reshape((1, -1)), dictionary,
            #                          lmbda=0.1, max_iter=60)
            # recon_contour = np.matmul(learned_val_codes, dictionary).reshape((-1, 2))
            # recon_contour = recon_contour + bbox_center
            # cv2.polylines(img_recon, [recon_contour.astype(np.int32)], True, (10, 255, 10), thickness=2)

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
            predicted_codes = []
            start_time = time.time()
            for scale in imgs:
                imgs[scale]['image'] = imgs[scale]['image'].to(cfg.device)

                output = model(imgs[scale]['image'])[-1]
                segms = ctsegm_cmm_decode(*output, K=cfg.test_topk)
                segms = segms.detach().cpu().numpy().reshape(1, -1, segms.shape[2])[0]

                top_preds = {}
                code_preds = {}
                for j in range(cfg.num_vertices):
                    segms[:, 2 * j:2 * j + 2] = transform_preds(segms[:, 2 * j:2 * j + 2],
                                                                imgs[scale]['center'],
                                                                imgs[scale]['scale'],
                                                                (imgs[scale]['fmap_w'], imgs[scale]['fmap_h']))
                segms[:, cfg.num_vertices * 2:cfg.num_vertices * 2 + 2] = transform_preds(
                    segms[:, cfg.num_vertices * 2:cfg.num_vertices * 2 + 2],
                    imgs[scale]['center'],
                    imgs[scale]['scale'],
                    (imgs[scale]['fmap_w'], imgs[scale]['fmap_h']))
                segms[:, cfg.num_vertices * 2 + 2:cfg.num_vertices * 2 + 4] = transform_preds(
                    segms[:, cfg.num_vertices * 2 + 2:cfg.num_vertices * 2 + 4],
                    imgs[scale]['center'],
                    imgs[scale]['scale'],
                    (imgs[scale]['fmap_w'], imgs[scale]['fmap_h']))

                clses = segms[:, -1]
                for j in range(num_classes):
                    inds = (clses == j)
                    top_preds[j + 1] = segms[inds, :cfg.num_vertices * 2 + 5].astype(np.float32)
                    top_preds[j + 1][:, :cfg.num_vertices * 2 + 4] /= scale
                    # code_preds[j + 1] = codes_[inds, :]

                segmentations.append(top_preds)
                predicted_codes.append(code_preds)

            segms_and_scores = {j: np.concatenate([d[j] for d in segmentations], axis=0)
                                for j in range(1, num_classes + 1)}  # a Dict label: segments
            # codes_and_scores = {j: np.concatenate([d[j] for d in predicted_codes], axis=0)
            #                     for j in range(1, num_classes + 1)}  # a Dict label: segments
            scores = np.hstack(
                [segms_and_scores[j][:, cfg.num_vertices * 2 + 4] for j in range(1, num_classes + 1)])

            if len(scores) > max_per_image:
                kth = len(scores) - max_per_image
                thresh = np.partition(scores, kth)[kth]
                for j in range(1, num_classes + 1):
                    keep_inds = (segms_and_scores[j][:, cfg.num_vertices * 2 + 4] >= thresh)
                    segms_and_scores[j] = segms_and_scores[j][keep_inds]
                    # codes_and_scores[j] = codes_and_scores[j][keep_inds]

            # Use opencv functions to output a video
            output_image = original_image

            for lab in segms_and_scores:
                for idx in range(len(segms_and_scores[lab])):
                    res = segms_and_scores[lab][idx]
                    # c_ = codes_and_scores[lab][idx]
                # for res in segms_and_scores[lab]:
                    contour, bbox, score = res[:-5], res[-5:-1], res[-1]
                    bbox[0] = np.clip(bbox[0], 0, w_img)
                    bbox[1] = np.clip(bbox[1], 0, h_img)
                    bbox[2] = np.clip(bbox[2], 0, w_img)
                    bbox[3] = np.clip(bbox[3], 0, h_img)
                    if score > cfg.detect_thres:
                        text = names[lab]  # + ' %.2f' % score
                        # label_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, thickness=2, fontScale=0.5)
                        polygon = contour.reshape((-1, 2))

                        # use bb tools to draw predictions
                        color = random.choice(COLOR_WORLD)
                        bb.add(output_image, bbox[0], bbox[1], bbox[2], bbox[3], text, color)
                        cv2.polylines(output_image, [polygon.astype(np.int32)], True, RGB_DICT[color], thickness=2)

                        # color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                        # contour_mean = np.mean(polygon, axis=0)
                        # contour_std = np.std(polygon, axis=0)
                        # center_x, center_y = np.mean(polygon, axis=0).astype(np.int32)
                        # text_location = [bbox[0] + 1, bbox[1] + 1,
                        #                  bbox[1] + label_size[0][0] + 1,
                        #                  bbox[0] + label_size[0][1] + 1]
                        # cv2.rectangle(output_image, pt1=(int(bbox[0]), int(bbox[1])),
                        #               pt2=(int(bbox[2]), int(bbox[3])),
                        #               color=color, thickness=1)
                        # cv2.rectangle(output_image, pt1=(int(np.min(polygon[:, 0])), int(np.min(polygon[:, 1]))),
                        #               pt2=(int(np.max(polygon[:, 0])), int(np.max(polygon[:, 1]))),
                        #               color=(0, 255, 0), thickness=1)
                        # cv2.polylines(output_image, [polygon.astype(np.int32)], True, color, thickness=2)
                        # cv2.putText(output_image, text, org=(int(text_location[0]), int(text_location[3])),
                        #             fontFace=cv2.FONT_HERSHEY_COMPLEX, thickness=2, fontScale=0.5,
                        #             color=(255, 0, 0))
                        # cv2.putText(output_image, text, org=(int(bbox[0]), int(bbox[1])),
                        #             fontFace=cv2.FONT_HERSHEY_COMPLEX, thickness=1, fontScale=0.5,
                        #             color=color)

                        # show the histgram for predicted codes
                        # fig = plt.figure()
                        # plt.plot(np.arange(cfg.n_codes), c_.reshape((-1,)), color='green',
                        #          marker='o', linestyle='dashed', linewidth=2, markersize=6)
                        # plt.ylabel('Value of each coefficient')
                        # plt.xlabel('All predicted {} coefficients'.format(cfg.n_codes))
                        # plt.title('Distribution of the predicted coefficients for {}'.format(text))
                        # plt.show()

            value = [255, 255, 255]
            img_original = cv2.copyMakeBorder(img_original, 0, 0, 0, 10, cv2.BORDER_CONSTANT, None, value)
            img_connect = cv2.copyMakeBorder(img_connect, 0, 0, 10, 10, cv2.BORDER_CONSTANT, None, value)
            output_image = cv2.copyMakeBorder(output_image, 0, 0, 10, 0, cv2.BORDER_CONSTANT, None, value)
            im_cat = np.concatenate((img_original, img_connect, output_image), axis=1)
            cv2.imshow('GT:Resample:Recons:Predict', im_cat)
            if cv2.waitKey() & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    main()
