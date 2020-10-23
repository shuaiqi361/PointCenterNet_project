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

from datasets.coco import COCO_MEAN, COCO_STD, COCO_NAMES, COCO_IDS
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


def encode_mask(mask):
    """Convert mask to coco rle"""
    rle = cocomask.encode(np.array(mask[:, :, np.newaxis], order='F'))[0]
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


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
    imgIds = coco.getImgIds()

    det_results = []
    seg_results = []

    for img_id in imgIds:
        img = coco.loadImgs(img_id)[0]
        image_path = '%s/images/%s/%s' % (cfg.data_dir, cfg.data_type, img['file_name'])
        w_img = int(img['width'])
        h_img = int(img['height'])
        if w_img < 1 or h_img < 1:
            continue

        image = cv2.imread(image_path)
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
            # print('In with no_grads()')
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

            # generate coco results for server eval
            # print('generate coco results for server eval ...')
            for lab in segms_and_scores:
                for res in segms_and_scores[lab]:
                    poly, score = res[:-1], res[-1]
                    recon_contour = poly.reshape((-1, 2))
                    recon_contour[:, 0] = np.clip(recon_contour[:, 0], 0, img_width - 1)
                    recon_contour[:, 1] = np.clip(recon_contour[:, 1], 0, img_height - 1)
                    category_id = int(COCO_IDS[lab - 1])
                    if score > cfg.detect_thres:
                        x1, y1, x2, y2 = int(min(recon_contour[:, 0])), int(min(recon_contour[:, 1])), \
                                         int(max(recon_contour[:, 0])), int(max(recon_contour[:, 1]))
                        bbox = [x1, y1, x2 - x1, y2 - y1]
                        det = {
                            'image_id': int(img_id),
                            'category_id': int(category_id),
                            'score': float("{:.2f}".format(score)),
                            'bbox': bbox
                        }
                        det_results.append(det)

                        # convert polygons to rle masks
                        poly = np.ndarray.flatten(recon_contour, order='C').tolist()  # row major flatten
                        rles = cocomask.frPyObjects([poly], img_height, img_width)
                        rle = cocomask.merge(rles)
                        m = cocomask.decode(rle)
                        rle_new = encode_mask(m.astype(np.uint8))

                        seg = {
                            'image_id': int(img_id),
                            'category_id': int(category_id),
                            'score': float("{:.2f}".format(score)),
                            'segmentation': rle_new
                        }
                        seg_results.append(seg)

    with open('{}/coco_result/{}_det_results_v{}.json'.format(cfg.root_dir, cfg.data_type, cfg.num_vertices),
              'w') as f_det:
        json.dump(det_results, f_det)
    with open('{}/coco_result/{}_seg_results_v{}.json'.format(cfg.root_dir, cfg.data_type, cfg.num_vertices),
              'w') as f_seg:
        json.dump(seg_results, f_seg)

    # run COCO detection evaluation
    print('Running COCO detection val17 evaluation ...')
    coco_pred = coco.loadRes(
        '{}/coco_result/{}_det_results_v{}.json'.format(cfg.root_dir, cfg.data_type, cfg.num_vertices))
    imgIds = sorted(coco.getImgIds())
    coco_eval = COCOeval(coco, coco_pred, 'bbox')
    coco_eval.params.imgIds = imgIds
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    print('---------------------------------------------------------------------------------')
    print('Running COCO segmentation val17 evaluation ...')
    coco_pred = coco.loadRes(
        '{}/coco_result/{}_seg_results_v{}.json'.format(cfg.root_dir, cfg.data_type, cfg.num_vertices))
    coco_eval = COCOeval(coco, coco_pred, 'segm')
    coco_eval.params.imgIds = imgIds
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == '__main__':
    main()
