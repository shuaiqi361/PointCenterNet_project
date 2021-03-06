import os
import sys
from pathlib import Path

current_path = Path(Path(__file__).parent.absolute())
sys.path.append(str(current_path.parent))

import cv2
import argparse
import numpy as np
import time

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

import torch
import torch.utils.data

from datasets.coco import COCO_MEAN, COCO_STD, COCO_NAMES
from datasets.detrac import DETRAC_MEAN, DETRAC_STD, DETRAC_NAMES
from datasets.kins_segm_cmm import KINS_NAMES

from nets.hourglass import exkp
from nets.resdcn_inmodal_scaled_code_pre_act import get_pose_resdcn

from utils.utils import load_demo_model
from utils.image import transform_preds, get_affine_transform
from utils.post_process import ctsegm_scale_decode, ctsegm_amodal_cmm_decode

COCO_COLORS = sns.color_palette('hls', len(COCO_NAMES))
DETRAC_COLORS = sns.color_palette('hls', len(DETRAC_NAMES))
KINS_COLORS = sns.color_palette('hls', len(KINS_NAMES))

# Training settings
parser = argparse.ArgumentParser(description='Inmodal segmentation with dual iterative code loss on COCO')
parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--img_dir', type=str, default='/media/keyi/Data/Research/traffic/data/Hwy7/new/lc2_down_2')
parser.add_argument('--ckpt_dir', type=str, default='./ckpt/pascal_resdcn18_512/checkpoint.t7')
parser.add_argument('--output_video_file', type=str, default='video.mkv')
parser.add_argument('--output_text_file', type=str, default='video.txt')
parser.add_argument('--dictionary_file', type=str)

parser.add_argument('--arch', type=str, default='resdcn_50')
parser.add_argument('--dataset', type=str, default='Hwy7', choices=['coco', 'DETRAC', 'Hwy7', 'kins'])
parser.add_argument('--img_size', type=str, default='512,512')

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
cfg.img_size = tuple([int(s) for s in cfg.img_size.split(',')])
display_cat = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck']

RGB_DICT = {'navy': (0, 38, 63), 'blue': (0, 120, 210), 'aqua': (115, 221, 252), 'teal': (15, 205, 202),
            'olive': (52, 153, 114), 'green': (0, 204, 84), 'lime': (1, 255, 127), 'yellow': (255, 216, 70),
            'orange': (255, 125, 57), 'red': (255, 47, 65), 'maroon': (135, 13, 75), 'fuchsia': (246, 0, 184),
            'purple': (179, 17, 193), 'black': (24, 24, 24), 'gray': (168, 168, 168), 'silver': (220, 220, 220)}


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
    if cfg.dataset == 'coco':
        num_classes = 80
        colors = COCO_COLORS
        names = COCO_NAMES
    elif cfg.dataset == 'DETRAC':
        num_classes = 3
        colors = DETRAC_COLORS
        names = DETRAC_NAMES
    elif cfg.dataset == 'kins':
        num_classes = 7
        colors = KINS_COLORS
        names = KINS_NAMES
    else:
        print('Please specify correct dataset name.')
        raise NotImplementedError

    for j in range(len(names)):
        col_ = [c * 255 for c in colors[j]]
        colors[j] = tuple(col_)

    # Set up parameters for outputing video
    output_folder = os.path.join(cfg.root_dir, 'demo')
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    width = cfg.video_width
    height = cfg.video_height
    fps = cfg.video_fps  # output video configuration
    video_out = cv2.VideoWriter(os.path.join(output_folder, cfg.output_video_file),
                                cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, (width, height))
    text_out = open(os.path.join(output_folder, cfg.output_text_file), 'w')
    dictionary = np.load(cfg.dictionary_file)

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

    # Loading images
    speed_list = []
    frame_list = sorted(os.listdir(cfg.img_dir))
    n_frames = len(frame_list)

    for frame_id in range(n_frames):
        frame_name = frame_list[frame_id]
        image_path = os.path.join(cfg.img_dir, frame_name)

        image = cv2.imread(image_path)
        original_image = image.copy()
        height, width = image.shape[0:2]
        padding = 127 if 'hourglass' in cfg.arch else 31
        imgs = {}
        for scale in cfg.test_scales:
            new_height = int(height * scale)
            new_width = int(width * scale)

            if cfg.img_size[0] > 0 and cfg.img_size[1] > 0:
                img_height, img_width = cfg.img_size[0], cfg.img_size[1]
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
                hmap, regs, w_h_, offsets, _, _, codes = model(imgs[scale]['image'])[-1]
                output = [hmap, regs, w_h_, codes, offsets]

                segms = ctsegm_scale_decode(*output,
                                            torch.from_numpy(dictionary.astype(np.float32)).to(cfg.device),
                                            K=cfg.test_topk)
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

                segmentations.append(top_preds)
                predicted_codes.append(code_preds)

            segms_and_scores = {j: np.concatenate([d[j] for d in segmentations], axis=0)
                                for j in range(1, num_classes + 1)}  # a Dict label: segments
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
            blend_mask = np.zeros(shape=output_image.shape, dtype=np.uint8)

            counter = 1
            for lab in segms_and_scores:
                if cfg.dataset == 'coco':
                    if names[lab] not in display_cat and cfg.dataset != 'kins':
                        continue
                for idx in range(len(segms_and_scores[lab])):
                    res = segms_and_scores[lab][idx]
                    contour, bbox, score = res[:-5], res[-5:-1], res[-1]
                    bbox[0] = np.clip(bbox[0], 0, width - 1)
                    bbox[1] = np.clip(bbox[1], 0, height - 1)
                    bbox[2] = np.clip(bbox[2], 0, width - 1)
                    bbox[3] = np.clip(bbox[3], 0, height - 1)

                    polygon = contour.reshape((-1, 2))
                    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
                    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
                    if score > cfg.detect_thres:
                        text = names[lab] + ' %.2f' % score
                        label_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 0.3, 1)
                        text_location = [int(bbox[0]) + 2, int(bbox[1]) + 2,
                                         int(bbox[0]) + 2 + label_size[0][0],
                                         int(bbox[1]) + 2 + label_size[0][1]]
                        # cv2.rectangle(output_image, pt1=(int(bbox[0]), int(bbox[1])),
                        #               pt2=(int(bbox[2]), int(bbox[3])),
                        #               color=colors[lab], thickness=2)
                        # cv2.rectangle(output_image, pt1=(int(bbox[0]), int(bbox[1])),
                        #               pt2=(int(bbox[2]), int(bbox[3])),
                        #               color=nice_colors[names[lab]], thickness=2)
                        # cv2.putText(output_image, text, org=(int(text_location[0]), int(text_location[3])),
                        #             fontFace=cv2.FONT_HERSHEY_COMPLEX, thickness=1, fontScale=0.3,
                        #             color=nice_colors[names[lab]])

                        cv2.polylines(output_image, [polygon.astype(np.int32)], True, color=nice_colors[names[lab]],
                                      thickness=2)
                        cv2.drawContours(blend_mask, [polygon.astype(np.int32)], contourIdx=-1,
                                         color=nice_colors[names[lab]],
                                         thickness=-1)

                        # add to text file
                        new_line = '{0},{1},{2:.3f},{3:.3f},{4:.3f},{5:.3f},{6:.4f}\n'.format(str(frame_id + 1),
                                                                                              counter,
                                                                                              int(bbox[0]),
                                                                                              int(bbox[1]),
                                                                                              int(bbox[2]) - int(
                                                                                                  bbox[0]),
                                                                                              int(bbox[3]) - int(
                                                                                                  bbox[1]),
                                                                                              score)
                        counter += 1
                        text_out.write(new_line)

            dst_img = cv2.addWeighted(output_image, 0.4, blend_mask, 0.6, 0)
            dst_img[blend_mask == 0] = output_image[blend_mask == 0]
            output_image = dst_img

            cv2.imshow('Frames', output_image)
            video_out.write(output_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    print('Test frame rate:', 1. / np.mean(speed_list))


if __name__ == '__main__':
    main()
