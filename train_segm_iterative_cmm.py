import os
import sys
import time
import argparse
# from easydict import EasyDict
# import json
# import yaml
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import numpy as np

import torch.nn as nn
import torch.utils.data
import torch.distributed as dist

from datasets.coco_segm_cmm import COCOSEGMCMM, COCO_eval_segm_cmm
from datasets.pascal import PascalVOC, PascalVOC_eval

from nets.hourglass_segm_cmm import get_hourglass, exkp
from nets.resdcn_iterative_cmm import get_pose_resdcn

from utils.utils import _tranpose_and_gather_feature, load_model
from utils.image import transform_preds
from utils.losses import _neg_loss, _reg_loss, contour_mapping_loss, norm_reg_loss
from utils.summary import create_summary, create_logger, create_saver, DisablePrint
from utils.post_process import ctsegm_cmm_decode, ctsegm_shift_code_decode

# Training settings
parser = argparse.ArgumentParser(description='simple_centernet_segm_ncmm')

parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--device_id', type=int, default=0)  # specify device id for single GPU training
parser.add_argument('--dist', action='store_true')

parser.add_argument('--root_dir', type=str, default='./')
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--log_name', type=str, default='test')
parser.add_argument('--pretrain_checkpoint', type=str)
parser.add_argument('--dictionary_file', type=str)

parser.add_argument('--dataset', type=str, default='coco', choices=['coco', 'pascal'])
parser.add_argument('--arch', type=str, default='large_hourglass')

parser.add_argument('--img_size', type=int, default=512)
parser.add_argument('--split_ratio', type=float, default=1.0)
parser.add_argument('--n_vertices', type=int, default=32)
parser.add_argument('--n_codes', type=int, default=64)
parser.add_argument('--cmm_loss_weight', type=float, default=1)
parser.add_argument('--code_loss_weight', type=float, default=1)

parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr_step', type=str, default='90,120')
parser.add_argument('--batch_size', type=int, default=48)
parser.add_argument('--num_epochs', type=int, default=140)

parser.add_argument('--test_topk', type=int, default=100)

parser.add_argument('--log_interval', type=int, default=100)
parser.add_argument('--val_interval', type=int, default=5)
parser.add_argument('--num_workers', type=int, default=2)

cfg = parser.parse_args()

os.chdir(cfg.root_dir)

cfg.log_dir = os.path.join(cfg.root_dir, 'logs', cfg.log_name)
cfg.ckpt_dir = os.path.join(cfg.root_dir, 'ckpt', cfg.log_name)

os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)

cfg.lr_step = [int(s) for s in cfg.lr_step.split(',')]


def main():
    saver = create_saver(cfg.local_rank, save_dir=cfg.ckpt_dir)
    logger = create_logger(cfg.local_rank, save_dir=cfg.log_dir)
    summary_writer = create_summary(cfg.local_rank, log_dir=cfg.log_dir)

    print_log = logger.info
    print_log(cfg)

    torch.manual_seed(319)
    torch.backends.cudnn.benchmark = True  # disable this if OOM at beginning of training

    num_gpus = torch.cuda.device_count()
    if cfg.dist:
        cfg.device = torch.device('cuda:%d' % cfg.local_rank)
        torch.cuda.set_device(cfg.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=num_gpus, rank=cfg.local_rank)
    else:
        cfg.device = torch.device('cuda:%d' % cfg.device_id)

    print_log('Setting up data...')
    dictionary = np.load(cfg.dictionary_file)
    Dataset = COCOSEGMCMM if cfg.dataset == 'coco' else PascalVOC
    train_dataset = Dataset(cfg.data_dir, cfg.dictionary_file,
                            'train', split_ratio=cfg.split_ratio, img_size=cfg.img_size)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=num_gpus,
                                                                    rank=cfg.local_rank)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg.batch_size // num_gpus
                                               if cfg.dist else cfg.batch_size,
                                               shuffle=not cfg.dist,
                                               num_workers=cfg.num_workers,
                                               pin_memory=False,
                                               drop_last=True,
                                               sampler=train_sampler if cfg.dist else None)

    Dataset_eval = COCO_eval_segm_cmm if cfg.dataset == 'coco' else PascalVOC_eval
    val_dataset = Dataset_eval(cfg.data_dir, cfg.dictionary_file,
                               'val', test_scales=[1.], test_flip=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                             shuffle=False, num_workers=1, pin_memory=False,
                                             collate_fn=val_dataset.collate_fn)

    print_log('Creating model...')
    if 'hourglass' in cfg.arch:
        # model = get_hourglass[cfg.arch]
        model = exkp(n=5, nstack=2, dims=[256, 256, 384, 384, 384, 512],
                     modules=[2, 2, 2, 2, 2, 4],
                     dictionary=torch.from_numpy(dictionary.astype(np.float32)).to(cfg.device))
    elif 'resdcn' in cfg.arch:
        model = get_pose_resdcn(num_layers=int(cfg.arch.split('_')[-1]), head_conv=64, num_classes=train_dataset.num_classes)
    else:
        raise NotImplementedError

    if cfg.dist:
        # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.to(cfg.device)
        model = nn.parallel.DistributedDataParallel(model,
                                                    device_ids=[cfg.local_rank, ],
                                                    output_device=cfg.local_rank)
    else:
        model = nn.DataParallel(model, device_ids=[cfg.local_rank, ]).to(cfg.device)

    if cfg.pretrain_checkpoint is not None and os.path.isfile(cfg.pretrain_checkpoint):
        print_log('Load pretrain model from ' + cfg.pretrain_checkpoint)
        model = load_model(model, cfg.pretrain_checkpoint, cfg.device_id)
        torch.cuda.empty_cache()

    optimizer = torch.optim.Adam(model.parameters(), cfg.lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, cfg.lr_step, gamma=0.2)

    def train(epoch):
        print_log('\n Epoch: %d' % epoch)
        model.train()
        tic = time.perf_counter()
        for batch_idx, batch in enumerate(train_loader):
            for k in batch:
                if k != 'meta':
                    batch[k] = batch[k].to(device=cfg.device, non_blocking=True)

            dict_tensor = torch.from_numpy(dictionary.astype(np.float32)).to(cfg.device, non_blocking=True)
            dict_tensor.requires_grad = False

            outputs = model(batch['image'])
            hmap, regs, w_h_, codes_1, codes_2, codes_3 = zip(*outputs)

            regs = [_tranpose_and_gather_feature(r, batch['inds']) for r in regs]
            w_h_ = [_tranpose_and_gather_feature(r, batch['inds']) for r in w_h_]
            c_1 = [_tranpose_and_gather_feature(r, batch['inds']) for r in codes_1]
            c_2 = [_tranpose_and_gather_feature(r, batch['inds']) for r in codes_2]
            c_3 = [_tranpose_and_gather_feature(r, batch['inds']) for r in codes_3]
            # print(c_1[0].size(), dict_tensor.size())

            shapes_1 = [torch.matmul(c, dict_tensor) for c in c_1]
            shapes_2 = [torch.matmul(c, dict_tensor) for c in c_2]
            shapes_3 = [torch.matmul(c, dict_tensor) for c in c_3]

            hmap_loss = _neg_loss(hmap, batch['hmap'])
            reg_loss = _reg_loss(regs, batch['regs'], batch['ind_masks'])
            w_h_loss = _reg_loss(w_h_, batch['w_h_'], batch['ind_masks'])
            codes_loss = (norm_reg_loss(c_1, batch['codes'], batch['ind_masks'])
                          + norm_reg_loss(c_2, batch['codes'], batch['ind_masks'])
                          + norm_reg_loss(c_3, batch['codes'], batch['ind_masks'])) / 3.
            cmm_loss = (contour_mapping_loss(c_1, shapes_1, batch['shapes'], batch['ind_masks'], roll=True)
                        + contour_mapping_loss(c_2, shapes_2, batch['shapes'], batch['ind_masks'], roll=True)
                        + contour_mapping_loss(c_3, shapes_3, batch['shapes'], batch['ind_masks'], roll=True)) / 3.

            loss = 2 * hmap_loss + 1 * reg_loss + 0.1 * w_h_loss + cfg.cmm_loss_weight * cmm_loss \
                   + cfg.code_loss_weight * codes_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % cfg.log_interval == 0:
                duration = time.perf_counter() - tic
                tic = time.perf_counter()
                print_log('[%d/%d-%d/%d] ' % (epoch, cfg.num_epochs, batch_idx, len(train_loader)) +
                      ' hmap_loss= %.3f reg_loss= %.3f w_h_loss= %.3f code_loss= %.3f cmm_loss= %.3f' %
                      (hmap_loss.item(), reg_loss.item(), w_h_loss.item(), codes_loss.item(), cmm_loss.item()) +
                      ' (%d samples/sec)' % (cfg.batch_size * cfg.log_interval / duration))

                step = len(train_loader) * epoch + batch_idx
                summary_writer.add_scalar('hmap_loss', hmap_loss.item(), step)
                summary_writer.add_scalar('reg_loss', reg_loss.item(), step)
                summary_writer.add_scalar('w_h_loss', w_h_loss.item(), step)
                summary_writer.add_scalar('code_loss', codes_loss.item(), step)
                summary_writer.add_scalar('cmm_loss', cmm_loss.item(), step)
        return

    def val_map(epoch):
        print_log('\n Val@Epoch: %d' % epoch)
        model.eval()
        torch.cuda.empty_cache()
        max_per_image = 100

        results = {}
        input_scales = {}
        speed_list = []
        with torch.no_grad():
            for inputs in val_loader:
                img_id, inputs = inputs[0]
                start_image_time = time.time()
                segmentations = []
                for scale in inputs:
                    inputs[scale]['image'] = inputs[scale]['image'].to(cfg.device)
                    if scale == 1. and img_id not in input_scales.keys():  # keep track of the input image Sizes
                        _, _, input_h, input_w = inputs[scale]['image'].shape
                        input_scales[img_id] = {'h': input_h, 'w': input_w}

                    # dict_tensor = torch.from_numpy(dictionary.astype(np.float32)).to(cfg.device, non_blocking=True)
                    # dict_tensor.requires_grad = False
                    hmap, regs, w_h_, _, _, codes = model(inputs[scale]['image'])[-1]
                    output = [hmap, regs, w_h_, codes]

                    segms = ctsegm_shift_code_decode(*output,
                                                     torch.from_numpy(dictionary.astype(np.float32)).to(cfg.device),
                                                     K=cfg.test_topk)
                    segms = segms.detach().cpu().numpy().reshape(1, -1, segms.shape[2])[0]

                    top_preds = {}
                    for j in range(cfg.n_vertices):
                        segms[:, 2 * j:2 * j + 2] = transform_preds(segms[:, 2 * j:2 * j + 2],
                                                                    inputs[scale]['center'],
                                                                    inputs[scale]['scale'],
                                                                    (inputs[scale]['fmap_w'], inputs[scale]['fmap_h']))
                    segms[:, cfg.n_vertices * 2:cfg.n_vertices * 2 + 2] = transform_preds(
                        segms[:, cfg.n_vertices * 2:cfg.n_vertices * 2 + 2],
                        inputs[scale]['center'],
                        inputs[scale]['scale'],
                        (inputs[scale]['fmap_w'], inputs[scale]['fmap_h']))
                    segms[:, cfg.n_vertices * 2 + 2:cfg.n_vertices * 2 + 4] = transform_preds(
                        segms[:, cfg.n_vertices * 2 + 2:cfg.n_vertices * 2 + 4],
                        inputs[scale]['center'],
                        inputs[scale]['scale'],
                        (inputs[scale]['fmap_w'], inputs[scale]['fmap_h']))

                    clses = segms[:, -1]
                    for j in range(val_dataset.num_classes):
                        inds = (clses == j)
                        top_preds[j + 1] = segms[inds, :cfg.n_vertices * 2 + 5].astype(np.float32)
                        top_preds[j + 1][:, :cfg.n_vertices * 2 + 4] /= scale

                    segmentations.append(top_preds)

                end_image_time = time.time()
                segms_and_scores = {j: np.concatenate([d[j] for d in segmentations], axis=0)
                                    for j in range(1, val_dataset.num_classes + 1)}
                scores = np.hstack(
                    [segms_and_scores[j][:, cfg.n_vertices * 2 + 4] for j in range(1, val_dataset.num_classes + 1)])
                if len(scores) > max_per_image:
                    kth = len(scores) - max_per_image
                    thresh = np.partition(scores, kth)[kth]
                    for j in range(1, val_dataset.num_classes + 1):
                        keep_inds = (segms_and_scores[j][:, cfg.n_vertices * 2 + 4] >= thresh)
                        segms_and_scores[j] = segms_and_scores[j][keep_inds]

                results[img_id] = segms_and_scores
                speed_list.append(end_image_time - start_image_time)

        eval_results = val_dataset.run_eval(results, save_dir=cfg.ckpt_dir)
        print_log(eval_results)
        summary_writer.add_scalar('val_mAP/mAP', eval_results[0], epoch)
        print_log('Average speed on val set:{:.2f}'.format(1. / np.mean(speed_list)))

    print_log('Starting training...')
    for epoch in range(1, cfg.num_epochs + 1):
        start = time.time()
        train_sampler.set_epoch(epoch)
        train(epoch)
        if (cfg.val_interval > 0 and epoch % cfg.val_interval == 0) or epoch == 3:
            val_map(epoch)
            print_log(saver.save(model.module.state_dict(), 'checkpoint'))
        lr_scheduler.step()  # move to here after pytorch1.1.0

        epoch_time = (time.time() - start) / 3600. / 24.
        print_log('ETA:{:.2f} Days'.format((cfg.num_epochs - epoch) * epoch_time))

    summary_writer.close()


if __name__ == '__main__':
    # print(cfg.local_rank, cfg.device_id)
    with DisablePrint(local_rank=cfg.local_rank):
        main()
