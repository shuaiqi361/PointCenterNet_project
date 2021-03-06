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
from utils.sparse_coding import fast_ista, check_clockwise_polygon, uniformsample

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns
import torch
import torch.utils.data

from datasets.coco import COCO_MEAN, COCO_STD, COCO_NAMES
from datasets.detrac import DETRAC_MEAN, DETRAC_STD, DETRAC_NAMES
from datasets.kins_segm_cmm import KINS_NAMES

from nets.hourglass import exkp
from nets.resdcn_inmodal_scale_code_pre_act_old import get_pose_resdcn

from utils.utils import load_demo_model
from utils.image import transform_preds, get_affine_transform
from utils.post_process import ctsegm_scale_decode, ctsegm_scale_decode_debug

COCO_COLORS = sns.color_palette('hls', len(COCO_NAMES))
DETRAC_COLORS = sns.color_palette('hls', len(DETRAC_NAMES))
KINS_COLORS = sns.color_palette('hls', len(KINS_NAMES))

# Training settings
parser = argparse.ArgumentParser(description='Modal segmentation test on COCO json')
parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str,
                    default='/media/keyi/Data/Research/course_project/AdvancedCV_2020/data/COCO17')
parser.add_argument('--ckpt_dir', type=str, default='./ckpt/pascal_resdcn18_512/checkpoint.t7')
parser.add_argument('--output_video_file', type=str, default='video.mkv')
parser.add_argument('--output_text_file', type=str, default='video.txt')
parser.add_argument('--dictionary_file', type=str)
parser.add_argument('--data_type', type=str, default='val')

parser.add_argument('--arch', type=str, default='resdcn_50')
parser.add_argument('--dataset', type=str, default='Hwy7', choices=['coco', 'DETRAC', 'Hwy7', 'kins'])
parser.add_argument('--img_size', type=int, default=512)

parser.add_argument('--test_flip', action='store_false')
parser.add_argument('--test_scales', type=str, default='1')  # 0.5,0.75,1,1.25,1.5
parser.add_argument('--test_topk', type=int, default=100)
parser.add_argument('--detect_thres', type=float, default=0.3)
parser.add_argument('--num_vertices', type=int, default=32)
parser.add_argument('--n_codes', type=int, default=64)

parser.add_argument('--video_width', type=int, default=1920)
parser.add_argument('--video_height', type=int, default=1080)
parser.add_argument('--video_fps', type=int, default=30)
parser.add_argument('--detector_name', type=str, default='Resdcn_50')
parser.add_argument('--name_pattern', type=str, default='img{:05d}.png')

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


def switch_tuple(input_tuple):
    return (input_tuple[2], input_tuple[1], input_tuple[0])


nice_colors = {
    'person': switch_tuple(RGB_DICT['orange']), 'car': switch_tuple(RGB_DICT['green']),
    'bus': switch_tuple(RGB_DICT['lime']), 'truck': switch_tuple(RGB_DICT['olive']),
    'bicycle': switch_tuple(RGB_DICT['maroon']), 'motorcycle': switch_tuple(RGB_DICT['fuchsia']),
    'cyclist': switch_tuple(RGB_DICT['yellow']), 'pedestrian': switch_tuple(RGB_DICT['orange']),
    'tram': switch_tuple(RGB_DICT['purple']), 'van': switch_tuple(RGB_DICT['teal']),
    'misc': switch_tuple(RGB_DICT['navy'])
}


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
    elif 'resdcn' in cfg.arch:
        model = get_pose_resdcn(num_layers=int(cfg.arch.split('_')[-1]), head_conv=64,
                                num_classes=num_classes, num_codes=cfg.n_codes)
    else:
        raise NotImplementedError

    model = load_demo_model(model, cfg.ckpt_dir)
    model = model.to(cfg.device)
    model.eval()

    # Loading COCO validation images
    if 'train' in cfg.data_type:
        annotation_file = '{}/annotations/instances_train2017.json'.format(cfg.data_dir)
        cfg.data_type = 'train2017'
    else:
        annotation_file = '{}/annotations/instances_val2017.json'.format(cfg.data_dir)
        cfg.data_type = 'val2017'
    coco = COCO(annotation_file)

    # Load all annotations
    cats = coco.loadCats(coco.getCatIds())
    # nms = [cat['name'] for cat in cats]
    nms = ['giraffe']
    catIds = coco.getCatIds(catNms=nms)
    imgIds = coco.getImgIds(catIds=catIds)
    annIds = coco.getAnnIds(catIds=catIds)
    all_anns = coco.loadAnns(ids=annIds)

    for annotation in all_anns:
        if annotation['iscrowd'] == 1 or type(annotation['segmentation']) != list or len(
                annotation['segmentation']) > 1:
            continue

        img = coco.loadImgs(annotation['image_id'])[0]
        image_path = '%s/images/%s/%s' % (cfg.data_dir, cfg.data_type, img['file_name'])
        w_img = int(img['width'])
        h_img = int(img['height'])
        if w_img < 350 or h_img < 350:
            continue

        polygons = annotation['segmentation'][0]
        gt_bbox = annotation['bbox']
        gt_x1, gt_y1, gt_w, gt_h = gt_bbox
        contour = np.array(polygons).reshape((-1, 2))
        if cv2.contourArea(contour.astype(np.int32)) < 200:
            continue

        # Downsample the contour to fix number of vertices
        fixed_contour = resample(contour, num=cfg.num_vertices)

        clockwise_flag = check_clockwise_polygon(fixed_contour)
        if not clockwise_flag:
            fixed_contour = np.flip(fixed_contour, axis=0)
        # else:
        #     fixed_contour = indexed_shape.copy()

        # Indexing from the left-most vertex, argmin x-axis
        idx = np.argmin(fixed_contour[:, 0])
        indexed_shape = np.concatenate((fixed_contour[idx:, :], fixed_contour[:idx, :]), axis=0)

        indexed_shape[:, 0] = np.clip(indexed_shape[:, 0], gt_x1, gt_x1 + gt_w)
        indexed_shape[:, 1] = np.clip(indexed_shape[:, 1], gt_y1, gt_y1 + gt_h)

        updated_bbox = [np.min(indexed_shape[:, 0]), np.min(indexed_shape[:, 1]),
                        np.max(indexed_shape[:, 0]), np.max(indexed_shape[:, 1])]
        w, h = updated_bbox[2] - updated_bbox[0], updated_bbox[3] - updated_bbox[1]
        contour_mean = np.mean(indexed_shape, axis=0)
        # contour_std = np.std(indexed_shape, axis=0)
        # if contour_std < 1e-6 or contour_std == np.inf or contour_std == np.nan:  # invalid shapes
        #     continue

        norm_shape = (indexed_shape - contour_mean) / np.array([w / 2., h / 2.])
        gt_codes, _ = fast_ista(norm_shape.reshape((1, -1)), dictionary, lmbda=0.005, max_iter=80)

        recon_contour = np.matmul(gt_codes, dictionary).reshape((-1, 2)) * np.array([w / 2., h / 2.])
        recon_contour = recon_contour + contour_mean

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            continue

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
            mass_centers = []
            start_time = time.time()
            print('Start running model ......')
            for scale in imgs:
                imgs[scale]['image'] = imgs[scale]['image'].to(cfg.device)
                hmap, regs, w_h_, _, _, codes, offsets = model(imgs[scale]['image'])[-1]
                output = [hmap, regs, w_h_, codes, offsets]

                # segms = ctsegm_scale_decode(*output,
                #                             torch.from_numpy(dictionary.astype(np.float32)).to(cfg.device),
                #                             K=cfg.test_topk)
                # print(len(output))
                segms, pred_codes, pred_center = ctsegm_scale_decode_debug(*output,
                                                                           torch.from_numpy(dictionary.astype(np.float32)).to(cfg.device),
                                                                           K=cfg.test_topk)
                segms = segms.detach().cpu().numpy().reshape(1, -1, segms.shape[2])[0]
                pred_codes = pred_codes.detach().cpu().numpy().reshape(-1, pred_codes.shape[-1])
                pred_center = pred_center.detach().cpu().numpy().reshape(-1, 2)

                top_preds = {}
                code_preds = {}
                center_preds = {}
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
                # For mass center
                pred_center = transform_preds(pred_center,
                                              imgs[scale]['center'],
                                              imgs[scale]['scale'],
                                              (imgs[scale]['fmap_w'], imgs[scale]['fmap_h']))

                clses = segms[:, -1]
                for j in range(num_classes):
                    inds = (clses == j)
                    top_preds[j + 1] = segms[inds, :cfg.num_vertices * 2 + 5].astype(np.float32)
                    top_preds[j + 1][:, :cfg.num_vertices * 2 + 4] /= scale
                    center_preds[j + 1] = pred_center[inds, :] / scale
                    code_preds[j + 1] = pred_codes[inds, :]

                segmentations.append(top_preds)
                predicted_codes.append(code_preds)
                mass_centers.append(center_preds)

            segms_and_scores = {j: np.concatenate([d[j] for d in segmentations], axis=0)
                                for j in range(1, num_classes + 1)}  # a Dict label: segments
            segms_and_codes = {j: np.concatenate([d[j] for d in predicted_codes], axis=0)
                               for j in range(1, num_classes + 1)}
            segms_and_centers = {j: np.concatenate([d[j] for d in mass_centers], axis=0)
                                 for j in range(1, num_classes + 1)}
            scores = np.hstack(
                [segms_and_scores[j][:, cfg.num_vertices * 2 + 4] for j in range(1, num_classes + 1)])

            print('Image processing time {:.4f} sec, preparing output image ......'.format(time.time() - start_time))

            if len(scores) > max_per_image:
                kth = len(scores) - max_per_image
                thresh = np.partition(scores, kth)[kth]
                for j in range(1, num_classes + 1):
                    keep_inds = (segms_and_scores[j][:, cfg.num_vertices * 2 + 4] >= thresh)
                    segms_and_scores[j] = segms_and_scores[j][keep_inds]
                    segms_and_codes[j] = segms_and_codes[j][keep_inds]
                    segms_and_centers[j] = segms_and_centers[j][keep_inds]

            # Use opencv functions to output
            # output_image = original_image
            # blend_mask = np.zeros(shape=output_image.shape, dtype=np.uint8)

            counter = 1
            for lab in segms_and_scores:
                output_image = original_image.copy()
                # if cfg.dataset == 'coco':
                #     if names[lab] not in display_cat and cfg.dataset != 'kins':
                #         continue
                for idx in range(len(segms_and_scores[lab])):
                    res = segms_and_scores[lab][idx]
                    p_code = segms_and_codes[lab][idx]
                    p_center = segms_and_centers[lab][idx]
                    contour, bbox, score = res[:-5], res[-5:-1], res[-1]
                    bbox[0] = np.clip(bbox[0], 0, width - 1)
                    bbox[1] = np.clip(bbox[1], 0, height - 1)
                    bbox[2] = np.clip(bbox[2], 0, width - 1)
                    bbox[3] = np.clip(bbox[3], 0, height - 1)

                    polygon = contour.reshape((-1, 2))
                    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
                    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
                    if score > cfg.detect_thres:
                        # text = names[lab] + ' %.2f' % score
                        # label_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 0.3, 1)
                        # text_location = [int(bbox[0]) + 2, int(bbox[1]) + 2,
                        #                  int(bbox[0]) + 2 + label_size[0][0],
                        #                  int(bbox[1]) + 2 + label_size[0][1]]
                        # cv2.rectangle(output_image, pt1=(int(bbox[0]), int(bbox[1])),
                        #               pt2=(int(bbox[2]), int(bbox[3])),
                        #               color=colors[lab], thickness=2)
                        # cv2.rectangle(output_image, pt1=(int(bbox[0]), int(bbox[1])),
                        #               pt2=(int(bbox[2]), int(bbox[3])),
                        #               color=nice_colors[names[lab]], thickness=2)
                        # cv2.putText(output_image, text, org=(int(text_location[0]), int(text_location[3])),
                        #             fontFace=cv2.FONT_HERSHEY_COMPLEX, thickness=1, fontScale=0.3,
                        #             color=nice_colors[names[lab]])

                        # use_color_key = COLOR_WORLD[random.randint(1, len(COLOR_WORLD)) - 1]
                        # cv2.polylines(output_image, [polygon.astype(np.int32)], True,
                        #               color=switch_tuple(RGB_DICT[use_color_key]),
                        #               thickness=2)
                        # cv2.drawContours(blend_mask, [polygon.astype(np.int32)], contourIdx=-1,
                        #                  color=switch_tuple(RGB_DICT[use_color_key]),
                        #                  thickness=-1)

                        # plot the polygons/contours
                        cv2.polylines(output_image, [recon_contour.astype(np.int32)], True,
                                      color=switch_tuple(RGB_DICT['green']), thickness=2)
                        cv2.polylines(output_image, [polygon.astype(np.int32)], True,
                                      color=switch_tuple(RGB_DICT['red']), thickness=2)

                        # plot the mass center location
                        cv2.circle(output_image, tuple(contour_mean.astype(np.int32).tolist()),
                                   radius=9, color=switch_tuple(RGB_DICT['green']), thickness=-1)
                        cv2.circle(output_image, tuple(p_center.astype(np.int32).tolist()),
                                   radius=9, color=switch_tuple(RGB_DICT['red']), thickness=-1)

                        # dst_img = cv2.addWeighted(output_image, 0.4, blend_mask, 0.6, 0)
                        # dst_img[blend_mask == 0] = output_image[blend_mask == 0]
                        # output_image = dst_img

                        cv2.imshow('Frames', output_image)
                        if cv2.waitKey() & 0xFF == ord('q'):
                            break

                        counter += 1
                        # show histogram
                        fig, (ax1, ax2) = plt.subplots(1, 2)
                        # plot 1
                        bins = np.linspace(-2, 2, 30)
                        ax1.hist(gt_codes.reshape((-1,)).tolist(), bins=bins, color='g', density=False, alpha=0.5)
                        ax1.hist(p_code.reshape((-1,)).tolist(), bins=bins, color='r', density=False, alpha=0.5)
                        ax1.legend(['GT Coeffs', 'Pred Coeffs'])
                        ax1.set_xlabel('Sparse Coefficients')
                        ax1.set_ylabel('Counts')
                        ax1.set_title('Histogram of Coefficients')

                        # plot 2
                        ax2.plot(gt_codes.reshape((-1,)), 'g*-', linewidth=2, markersize=6)
                        ax2.plot(p_code.reshape((-1,)), 'ro--', linewidth=1, markersize=5)
                        ax2.legend(['GT Coeffs', 'Pred Coeffs'])
                        ax2.set_xlabel('Coefficients Index')
                        ax2.set_ylabel('Value')
                        ax2.set_title('Coefficients')

                        plt.show()
                        plt.close()

            # print('Test frame rate:', 1. / np.mean(speed_list))


if __name__ == '__main__':
    main()
