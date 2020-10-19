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

from nets.hourglass import exkp

from utils.utils import load_demo_model
from utils.image import transform_preds, get_affine_transform
from utils.post_process import ctdet_decode

from lib.nms.nms import soft_nms

COCO_COLORS = sns.color_palette('hls', len(COCO_NAMES))
DETRAC_COLORS = sns.color_palette('hls', len(DETRAC_NAMES))

# Training settings
parser = argparse.ArgumentParser(description='centernet_traffic')

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--img_dir', type=str, default='./data/demo.png')
parser.add_argument('--ckpt_dir', type=str, default='./ckpt/pascal_resdcn18_512/checkpoint.t7')
parser.add_argument('--output_video_dir', type=str, default='./demo/video.mkv')
parser.add_argument('--output_text_dir', type=str, default='./demo/video.txt')

parser.add_argument('--arch', type=str, default='large_hourglass')
parser.add_argument('--dataset', type=str, default='DETRAC', choices=['coco', 'DETRAC'])
parser.add_argument('--img_size', type=int, default=512)

parser.add_argument('--test_flip', action='store_true')
parser.add_argument('--test_scales', type=str, default='1')  # 0.5,0.75,1,1.25,1.5
parser.add_argument('--test_topk', type=int, default=100)
parser.add_argument('--detect_thres', type=float, default=0.3)
parser.add_argument('--video_width', type=int, default=960)
parser.add_argument('--video_height', type=int, default=540)
parser.add_argument('--video_fps', type=int, default=30)
parser.add_argument('--detector_name', type=str, default='CenterNet')

parser.add_argument('--name_pattern', type=str, default='img{:04d}.png')

cfg = parser.parse_args()

os.chdir(cfg.root_dir)

cfg.test_scales = [float(s) for s in cfg.test_scales.split(',')]
DETRAC_compatible_names = ['car', 'bus', 'truck']


def main():
    cfg.device = torch.device('cuda')
    torch.backends.cudnn.benchmark = False

    max_per_image = 100
    num_classes = 80 if cfg.dataset == 'coco' else 4

    colors = COCO_COLORS if cfg.dataset == 'coco' else DETRAC_COLORS
    names = COCO_NAMES if cfg.dataset == 'coco' else DETRAC_NAMES
    for j in range(len(names)):
        col_ = [c * 255 for c in colors[j]]
        colors[j] = tuple(col_)

    # Set up parameters for outputing video
    output_name = 'demo/'
    width = cfg.video_width
    height = cfg.video_height
    fps = cfg.video_fps  # output video configuration
    video_out = cv2.VideoWriter(cfg.output_video_dir,
                                cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), fps, (width, height))
    text_out = open(cfg.output_text_dir, 'w')

    print('Creating model and recover from checkpoint ...')
    if 'hourglass' in cfg.arch:
        model = exkp(n=5, nstack=2, dims=[256, 256, 384, 384, 384, 512],
                     modules=[2, 2, 2, 2, 2, 4], num_classes=num_classes)
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
            detections = []
            start_time = time.time()
            for scale in imgs:
                imgs[scale]['image'] = imgs[scale]['image'].to(cfg.device)

                output = model(imgs[scale]['image'])[-1]
                dets = ctdet_decode(*output, K=cfg.test_topk)
                dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])[0]

                top_preds = {}
                dets[:, :2] = transform_preds(dets[:, 0:2],
                                              imgs[scale]['center'],
                                              imgs[scale]['scale'],
                                              (imgs[scale]['fmap_w'], imgs[scale]['fmap_h']))
                dets[:, 2:4] = transform_preds(dets[:, 2:4],
                                               imgs[scale]['center'],
                                               imgs[scale]['scale'],
                                               (imgs[scale]['fmap_w'], imgs[scale]['fmap_h']))
                cls = dets[:, -1]
                for j in range(num_classes):
                    inds = (cls == j)
                    top_preds[j + 1] = dets[inds, :5].astype(np.float32)
                    top_preds[j + 1][:, :4] /= scale

                detections.append(top_preds)

            bbox_and_scores = {}
            for j in range(1, num_classes + 1):
                bbox_and_scores[j] = np.concatenate([d[j] for d in detections], axis=0)
                if len(cfg.test_scales) > 1:
                    soft_nms(bbox_and_scores[j], Nt=0.5, method=2)
            scores = np.hstack([bbox_and_scores[j][:, 4] for j in range(1, num_classes + 1)])

            if len(scores) > max_per_image:
                kth = len(scores) - max_per_image
                thresh = np.partition(scores, kth)[kth]
                for j in range(1, num_classes + 1):
                    keep_inds = (bbox_and_scores[j][:, 4] >= thresh)
                    bbox_and_scores[j] = bbox_and_scores[j][keep_inds]

            # Use opencv functions to output a video
            # output_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            speed_list.append(time.time() - start_time)
            output_image = original_image

            counter = 1
            for lab in bbox_and_scores:
                if cfg.dataset == 'coco':
                    if names[lab] not in DETRAC_compatible_names:
                        continue
                for boxes in bbox_and_scores[lab]:
                    x1, y1, x2, y2, score = boxes
                    if score > cfg.detect_thres:
                        text = names[lab] + '%.2f' % score
                        label_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 0.3, 1)
                        text_location = [x1 + 2, y1 + 2,
                                         x1 + 2 + label_size[0][0],
                                         y1 + 2 + label_size[0][1]]
                        # cv2.rectangle(output_image, pt1=(int(x1), int(y1)),
                        #               pt2=(int(x2), int(y2)),
                        #               color=colors[lab], thickness=2)
                        cv2.rectangle(output_image, pt1=(int(x1), int(y1)),
                                      pt2=(int(x2), int(y2)),
                                      color=(0, 255, 0), thickness=2)
                        # cv2.putText(output_image, text, org=(int(text_location[0]), int(text_location[3])),
                        #             fontFace=cv2.FONT_HERSHEY_COMPLEX, thickness=1, fontScale=0.3,
                        #             color=(0, 0, 255))

                        # add to text file
                        new_line = '{0},{1},{2:.3f},{3:.3f},{4:.3f},{5:.3f},{6:.4f}\n'.format(str(frame_id + 1),
                                                                                              counter,
                                                                                              x1, y1, x2 - x1, y2 - y1,
                                                                                              score)
                        counter += 1
                        text_out.write(new_line)

            cv2.imshow('Frames'.format(frame_id), output_image)
            video_out.write(output_image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    print('Test frame rate:', 1. / np.mean(speed_list))


if __name__ == '__main__':
    main()
