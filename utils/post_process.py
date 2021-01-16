import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import _gather_feature, _tranpose_and_gather_feature, flip_tensor


def _nms(heat, kernel=3):
    hmax = F.max_pool2d(heat, kernel, stride=1, padding=(kernel - 1) // 2)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=40):
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()
    topk_inds = _gather_feature(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feature(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feature(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def ctdet_decode(hmap, regs, w_h_, K=100):
    batch, cat, height, width = hmap.shape
    hmap = torch.sigmoid(hmap)

    # if flip test
    if batch > 1:
        hmap = (hmap[0:1] + flip_tensor(hmap[1:2])) / 2
        w_h_ = (w_h_[0:1] + flip_tensor(w_h_[1:2])) / 2
        regs = regs[0:1]

    batch = 1

    hmap = _nms(hmap)  # perform nms on heatmaps

    scores, inds, clses, ys, xs = _topk(hmap, K=K)

    regs = _tranpose_and_gather_feature(regs, inds)
    regs = regs.view(batch, K, 2)
    xs = xs.view(batch, K, 1) + regs[:, :, 0:1]
    ys = ys.view(batch, K, 1) + regs[:, :, 1:2]

    w_h_ = _tranpose_and_gather_feature(w_h_, inds)
    w_h_ = w_h_.view(batch, K, 2)

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)
    bboxes = torch.cat([xs - w_h_[..., 0:1] / 2,
                        ys - w_h_[..., 1:2] / 2,
                        xs + w_h_[..., 0:1] / 2,
                        ys + w_h_[..., 1:2] / 2], dim=2)
    detections = torch.cat([bboxes, scores, clses], dim=2)
    return detections


def ctsegm_decode(hmap, regs, w_h_, codes_, dictionary, K=100):
    batch, cat, height, width = hmap.shape
    hmap = torch.sigmoid(hmap)

    # if flip test
    if batch > 1:
        hmap = (hmap[0:1] + flip_tensor(hmap[1:2])) / 2
        w_h_ = (w_h_[0:1] + flip_tensor(w_h_[1:2])) / 2
        regs = regs[0:1]
        codes_ = codes_[0:1]

    batch = 1

    hmap = _nms(hmap)  # perform nms on heatmaps

    scores, inds, clses, ys, xs = _topk(hmap, K=K)

    regs = _tranpose_and_gather_feature(regs, inds)
    regs = regs.view(batch, K, 2)
    xs = xs.view(batch, K, 1) + regs[:, :, 0:1]
    ys = ys.view(batch, K, 1) + regs[:, :, 1:2]

    w_h_ = _tranpose_and_gather_feature(w_h_, inds)
    w_h_ = w_h_.view(batch, K, 2)
    std_ = torch.sqrt(torch.sum(w_h_ ** 2., dim=2, keepdim=True))

    codes_ = _tranpose_and_gather_feature(codes_, inds)
    # codes_ = codes_.view(batch, K, 64)
    codes_ = torch.log(codes_).view(batch, K, 64)

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    segms = torch.matmul(codes_, dictionary)
    # print('Sizes:', segms.size(), std_.size(), xs.size())
    segms = (segms * std_).view(batch, K, 32, 2) + torch.cat([xs, ys], dim=2).view(batch, K, 1, 2)
    segmentations = torch.cat([segms.view(batch, K, -1), scores, clses], dim=2)

    return segmentations


def ctsegm_scale_decode(hmap, regs, w_h_, codes_, offsets_, dictionary, K=100):
    batch, cat, height, width = hmap.shape
    hmap = torch.sigmoid(hmap)

    # if flip test
    if batch > 1:
        hmap = (hmap[0:1] + flip_tensor(hmap[1:2])) / 2
        w_h_ = (w_h_[0:1] + flip_tensor(w_h_[1:2])) / 2
        regs = regs[0:1]
        codes_ = codes_[0:1]
        offsets_ = offsets_[0:1]

    batch = 1
    hmap = _nms(hmap)  # perform nms on heatmaps
    scores, inds, clses, ys, xs = _topk(hmap, K=K)

    regs = _tranpose_and_gather_feature(regs, inds)
    regs = regs.view(batch, K, 2)
    xs = xs.view(batch, K, 1) + regs[:, :, 0:1]
    ys = ys.view(batch, K, 1) + regs[:, :, 1:2]

    w_h_ = _tranpose_and_gather_feature(w_h_, inds)
    w_h_ = w_h_.view(batch, K, 2)

    codes_ = _tranpose_and_gather_feature(codes_, inds)
    codes_ = codes_.view(batch, K, -1)

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    bboxes = torch.cat([xs - w_h_[..., 0:1] / 2,
                        ys - w_h_[..., 1:2] / 2,
                        xs + w_h_[..., 0:1] / 2,
                        ys + w_h_[..., 1:2] / 2], dim=2)

    offsets_ = _tranpose_and_gather_feature(offsets_, inds)
    segms = torch.matmul(codes_, dictionary)
    segms = segms.view(batch, K, -1, 2) * w_h_.view(batch, K, 1, 2) / 2. + offsets_.view(batch, K, 1, 2) + \
            torch.cat([xs, ys], dim=2).view(batch, K, 1, 2)
    segmentations = torch.cat([segms.view(batch, K, -1), bboxes, scores, clses], dim=2)

    return segmentations


def ctsegm_cmm_decode(hmap, regs, w_h_, shapes_, K=100):
    batch, cat, height, width = hmap.shape
    hmap = torch.sigmoid(hmap)

    # if flip test
    if batch > 1:
        hmap = (hmap[0:1] + flip_tensor(hmap[1:2])) / 2
        w_h_ = (w_h_[0:1] + flip_tensor(w_h_[1:2])) / 2
        regs = regs[0:1]
        shapes_ = (shapes_[0:1] + flip_tensor(shapes_[1:2])) / 2

    batch = 1
    hmap = _nms(hmap)  # perform nms on heatmaps
    scores, inds, clses, ys, xs = _topk(hmap, K=K)

    regs = _tranpose_and_gather_feature(regs, inds)
    regs = regs.view(batch, K, 2)
    xs = xs.view(batch, K, 1) + regs[:, :, 0:1]
    ys = ys.view(batch, K, 1) + regs[:, :, 1:2]

    w_h_ = _tranpose_and_gather_feature(w_h_, inds)
    w_h_ = w_h_.view(batch, K, 2)

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    bboxes = torch.cat([xs - w_h_[..., 0:1] / 2,
                        ys - w_h_[..., 1:2] / 2,
                        xs + w_h_[..., 0:1] / 2,
                        ys + w_h_[..., 1:2] / 2], dim=2)

    shapes_ = _tranpose_and_gather_feature(shapes_, inds)
    segms = shapes_.view(batch, K, 32, 2) + torch.cat([xs, ys], dim=2).view(batch, K, 1, 2)
    segmentations = torch.cat([segms.view(batch, K, -1), bboxes, scores, clses], dim=2)

    return segmentations


def ctsegm_scale_decode_debug(hmap, regs, w_h_, codes_, dictionary, K=100):
    batch, cat, height, width = hmap.shape
    hmap = torch.sigmoid(hmap)

    # if flip test
    if batch > 1:
        hmap = (hmap[0:1] + flip_tensor(hmap[1:2])) / 2
        w_h_ = (w_h_[0:1] + flip_tensor(w_h_[1:2])) / 2
        regs = regs[0:1]
        codes_ = codes_[0:1]

    batch = 1

    hmap = _nms(hmap)  # perform nms on heatmaps

    scores, inds, clses, ys, xs = _topk(hmap, K=K)

    regs = _tranpose_and_gather_feature(regs, inds)
    regs = regs.view(batch, K, 2)
    xs = xs.view(batch, K, 1) + regs[:, :, 0:1]
    ys = ys.view(batch, K, 1) + regs[:, :, 1:2]

    w_h_ = _tranpose_and_gather_feature(w_h_, inds)
    w_h_ = w_h_.view(batch, K, 2)

    codes_ = _tranpose_and_gather_feature(codes_, inds)
    # codes_ = codes_.view(batch, K, 64)
    codes_ = torch.log(codes_).view(batch, K, 64)

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    bboxes = torch.cat([xs - w_h_[..., 0:1] / 2,
                        ys - w_h_[..., 1:2] / 2,
                        xs + w_h_[..., 0:1] / 2,
                        ys + w_h_[..., 1:2] / 2], dim=2)

    segms = torch.matmul(codes_, dictionary)
    # print('Sizes:', segms.size(), std_.size(), xs.size())
    segms = segms.view(batch, K, 32, 2) * w_h_.view(batch, K, 1, 2) / 2. \
            + torch.cat([xs, ys], dim=2).view(batch, K, 1, 2)
    segmentations = torch.cat([segms.view(batch, K, -1), bboxes, scores, clses], dim=2)

    return segmentations, codes_


def ctsegm_shift_decode(hmap, regs, w_h_, codes_, dictionary, K=100):
    batch, cat, height, width = hmap.shape
    hmap = torch.sigmoid(hmap)

    # if flip test
    if batch > 1:
        hmap = (hmap[0:1] + flip_tensor(hmap[1:2])) / 2
        w_h_ = (w_h_[0:1] + flip_tensor(w_h_[1:2])) / 2
        regs = regs[0:1]
        codes_ = codes_[0:1]

    batch = 1

    hmap = _nms(hmap)  # perform nms on heatmaps

    scores, inds, clses, ys, xs = _topk(hmap, K=K)

    regs = _tranpose_and_gather_feature(regs, inds)
    regs = regs.view(batch, K, 2)
    xs = xs.view(batch, K, 1) + regs[:, :, 0:1]
    ys = ys.view(batch, K, 1) + regs[:, :, 1:2]

    w_h_ = _tranpose_and_gather_feature(w_h_, inds)
    w_h_ = w_h_.view(batch, K, 2)

    codes_ = _tranpose_and_gather_feature(codes_, inds)
    codes_ = codes_.view(batch, K, 64)
    # codes_ = torch.log(codes_).view(batch, K, 64)

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    bboxes = torch.cat([xs - w_h_[..., 0:1] / 2,
                        ys - w_h_[..., 1:2] / 2,
                        xs + w_h_[..., 0:1] / 2,
                        ys + w_h_[..., 1:2] / 2], dim=2)

    segms = torch.matmul(codes_, dictionary)
    segms = segms.view(batch, K, 32, 2) + torch.cat([xs, ys], dim=2).view(batch, K, 1, 2)
    segmentations = torch.cat([segms.view(batch, K, -1), bboxes, scores, clses], dim=2)

    return segmentations


def ctsegm_shift_code_decode(hmap, regs, w_h_, codes_, dictionary, K=100):
    batch, cat, height, width = hmap.shape
    hmap = torch.sigmoid(hmap)

    # if flip test
    if batch > 1:
        hmap = (hmap[0:1] + flip_tensor(hmap[1:2])) / 2
        w_h_ = (w_h_[0:1] + flip_tensor(w_h_[1:2])) / 2
        regs = regs[0:1]
        codes_ = codes_[0:1]

    batch = 1

    hmap = _nms(hmap)  # perform nms on heatmaps

    scores, inds, clses, ys, xs = _topk(hmap, K=K)

    regs = _tranpose_and_gather_feature(regs, inds)
    regs = regs.view(batch, K, 2)
    xs = xs.view(batch, K, 1) + regs[:, :, 0:1]
    ys = ys.view(batch, K, 1) + regs[:, :, 1:2]

    w_h_ = _tranpose_and_gather_feature(w_h_, inds)
    w_h_ = w_h_.view(batch, K, 4)

    codes_ = _tranpose_and_gather_feature(codes_, inds)
    codes_ = codes_.view(batch, K, 64)
    # codes_ = torch.log(codes_).view(batch, K, 64)

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    bboxes = torch.cat([xs - w_h_[..., 2:3],
                        ys - w_h_[..., 0:1],
                        xs + w_h_[..., 3:4],
                        ys + w_h_[..., 1:2]], dim=2)

    segms = torch.matmul(codes_, dictionary)
    segms = segms.view(batch, K, 32, 2) + torch.cat([xs, ys], dim=2).view(batch, K, 1, 2)
    segmentations = torch.cat([segms.view(batch, K, -1), bboxes, scores, clses], dim=2)

    return segmentations


def ctsegm_amodal_cmm_decode(hmap, regs, w_h_, codes_, offsets_, dictionary, K=100):
    batch, cat, height, width = hmap.shape
    hmap = torch.sigmoid(hmap)

    # if flip test
    if batch > 1:
        hmap = (hmap[0:1] + flip_tensor(hmap[1:2])) / 2
        w_h_ = (w_h_[0:1] + flip_tensor(w_h_[1:2])) / 2
        regs = regs[0:1]
        codes_ = codes_[0:1]
        offsets_ = offsets_[0:1]

    batch = 1
    hmap = _nms(hmap)  # perform nms on heatmaps
    scores, inds, clses, ys, xs = _topk(hmap, K=K)

    regs = _tranpose_and_gather_feature(regs, inds)
    regs = regs.view(batch, K, 2)
    xs = xs.view(batch, K, 1) + regs[:, :, 0:1]
    ys = ys.view(batch, K, 1) + regs[:, :, 1:2]

    w_h_ = _tranpose_and_gather_feature(w_h_, inds)
    w_h_ = w_h_.view(batch, K, 2)

    codes_ = _tranpose_and_gather_feature(codes_, inds)
    codes_ = codes_.view(batch, K, -1)

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    bboxes = torch.cat([xs - w_h_[..., 0:1] / 2,
                        ys - w_h_[..., 1:2] / 2,
                        xs + w_h_[..., 0:1] / 2,
                        ys + w_h_[..., 1:2] / 2], dim=2)

    offsets_ = _tranpose_and_gather_feature(offsets_, inds)
    segms = torch.matmul(codes_, dictionary)
    segms = segms.view(batch, K, -1, 2) + offsets_.view(batch, K, 1, 2) + \
            torch.cat([xs, ys], dim=2).view(batch, K, 1, 2)
    segmentations = torch.cat([segms.view(batch, K, -1), bboxes, scores, clses], dim=2)

    return segmentations


# def ctsegm_amodal_cmm_whiten_decode(hmap, regs, w_h_, codes_, offsets_, dictionary, code_stat, K=100):
def ctsegm_amodal_cmm_whiten_decode(hmap, regs, w_h_, codes_, offsets_, dictionary, code_range, K=100):
    batch, cat, height, width = hmap.shape
    hmap = torch.sigmoid(hmap)

    # if flip test
    if batch > 1:
        hmap = (hmap[0:1] + flip_tensor(hmap[1:2])) / 2
        w_h_ = (w_h_[0:1] + flip_tensor(w_h_[1:2])) / 2
        regs = regs[0:1]
        codes_ = codes_[0:1]
        offsets_ = offsets_[0:1]

    batch = 1
    hmap = _nms(hmap)  # perform nms on heatmaps
    scores, inds, clses, ys, xs = _topk(hmap, K=K)

    regs = _tranpose_and_gather_feature(regs, inds)
    regs = regs.view(batch, K, 2)
    xs = xs.view(batch, K, 1) + regs[:, :, 0:1]
    ys = ys.view(batch, K, 1) + regs[:, :, 1:2]

    w_h_ = _tranpose_and_gather_feature(w_h_, inds)
    w_h_ = w_h_.view(batch, K, 2)

    codes_ = _tranpose_and_gather_feature(codes_, inds)
    codes_ = codes_.view(batch, K, 64)

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    bboxes = torch.cat([xs - w_h_[..., 0:1] / 2,
                        ys - w_h_[..., 1:2] / 2,
                        xs + w_h_[..., 0:1] / 2,
                        ys + w_h_[..., 1:2] / 2], dim=2)

    offsets_ = _tranpose_and_gather_feature(offsets_, inds)
    # codes_ = codes_ * code_stat[1].view(1, 1, -1) + code_stat[0].view(1, 1, -1)  # recover the original unnormalized codes
    codes_ = (codes_ + 1) / 2. * (code_range[1] - code_range[0]) + code_range[0]

    segms = torch.matmul(codes_, dictionary)
    segms = segms.view(batch, K, 32, 2) + offsets_.view(batch, K, 1, 2) + \
            torch.cat([xs, ys], dim=2).view(batch, K, 1, 2)
    segmentations = torch.cat([segms.view(batch, K, -1), bboxes, scores, clses], dim=2)

    return segmentations


def ctsegm_code_n_offset_decode(hmap, regs, w_h_, codes_, offsets_, dictionary, K=100):
    batch, cat, height, width = hmap.shape
    hmap = torch.sigmoid(hmap)

    # if flip test
    if batch > 1:
        hmap = (hmap[0:1] + flip_tensor(hmap[1:2])) / 2
        w_h_ = (w_h_[0:1] + flip_tensor(w_h_[1:2])) / 2
        regs = regs[0:1]
        codes_ = codes_[0:1]
        offsets_ = offsets_[0:1]

    batch = 1
    hmap = _nms(hmap)  # perform nms on heatmaps
    scores, inds, clses, ys, xs = _topk(hmap, K=K)

    regs = _tranpose_and_gather_feature(regs, inds)
    regs = regs.view(batch, K, 2)
    xs = xs.view(batch, K, 1) + regs[:, :, 0:1]
    ys = ys.view(batch, K, 1) + regs[:, :, 1:2]

    w_h_ = _tranpose_and_gather_feature(w_h_, inds)
    w_h_ = w_h_.view(batch, K, 4)

    codes_ = _tranpose_and_gather_feature(codes_, inds)
    codes_ = codes_.view(batch, K, 64)

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    bboxes = torch.cat([xs - w_h_[..., 2:3],
                        ys - w_h_[..., 0:1],
                        xs + w_h_[..., 3:4],
                        ys + w_h_[..., 1:2]], dim=2)

    offsets_ = _tranpose_and_gather_feature(offsets_, inds)
    segms = torch.matmul(codes_, dictionary)
    segms = segms.view(batch, K, 32, 2) + offsets_.view(batch, K, 32, 2) + \
            torch.cat([xs, ys], dim=2).view(batch, K, 1, 2)
    segmentations = torch.cat([segms.view(batch, K, -1), bboxes, scores, clses], dim=2)

    return segmentations


def ctsegm_fourier_decode(hmap, regs, w_h_, real_, imaginary_, K=100):
    batch, cat, height, width = hmap.shape
    hmap = torch.sigmoid(hmap)

    # if flip test
    if batch > 1:
        hmap = (hmap[0:1] + flip_tensor(hmap[1:2])) / 2
        w_h_ = (w_h_[0:1] + flip_tensor(w_h_[1:2])) / 2
        regs = regs[0:1]
        real_ = real_[0:1]
        imaginary_ = imaginary_[0:1]

    batch = 1

    hmap = _nms(hmap)  # perform nms on heatmaps

    scores, inds, clses, ys, xs = _topk(hmap, K=K)

    regs = _tranpose_and_gather_feature(regs, inds)
    regs = regs.view(batch, K, 2)
    xs = xs.view(batch, K, 1) + regs[:, :, 0:1]
    ys = ys.view(batch, K, 1) + regs[:, :, 1:2]

    w_h_ = _tranpose_and_gather_feature(w_h_, inds)
    w_h_ = w_h_.view(batch, K, 4)

    real_ = _tranpose_and_gather_feature(real_, inds)
    real_ = real_.view(batch, K, 32, 1)
    imaginary_ = _tranpose_and_gather_feature(imaginary_, inds)
    imaginary_ = imaginary_.view(batch, K, 32, 1)

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    bboxes = torch.cat([xs - w_h_[..., 2:3],
                        ys - w_h_[..., 0:1],
                        xs + w_h_[..., 3:4],
                        ys + w_h_[..., 1:2]], dim=2)

    complex_codes = torch.cat([real_, imaginary_], dim=3) * 32.
    segms = torch.ifft(complex_codes, signal_ndim=1)
    segms = segms + torch.cat([xs, ys], dim=2).view(batch, K, 1, 2)
    segmentations = torch.cat([segms.view(batch, K, -1), bboxes, scores, clses], dim=2)

    return segmentations


def ctsegm_code_shape_decode(hmap, regs, w_h_, shapes_, K=100):
    batch, cat, height, width = hmap.shape
    hmap = torch.sigmoid(hmap)

    # if flip test
    if batch > 1:
        hmap = (hmap[0:1] + flip_tensor(hmap[1:2])) / 2
        w_h_ = (w_h_[0:1] + flip_tensor(w_h_[1:2])) / 2
        regs = regs[0:1]
        shapes_ = shapes_[0:1]

    batch = 1
    hmap = _nms(hmap)  # perform nms on heatmaps
    scores, inds, clses, ys, xs = _topk(hmap, K=K)

    regs = _tranpose_and_gather_feature(regs, inds)
    regs = regs.view(batch, K, 2)
    xs = xs.view(batch, K, 1) + regs[:, :, 0:1]
    ys = ys.view(batch, K, 1) + regs[:, :, 1:2]

    w_h_ = _tranpose_and_gather_feature(w_h_, inds)
    w_h_ = w_h_.view(batch, K, 2)

    shapes_ = shapes_.view(batch, K, 64)

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    bboxes = torch.cat([xs - w_h_[..., 0:1] / 2,
                        ys - w_h_[..., 1:2] / 2,
                        xs + w_h_[..., 0:1] / 2,
                        ys + w_h_[..., 1:2] / 2], dim=2)

    segmentations = torch.cat([shapes_, bboxes, scores, clses], dim=2)

    return segmentations


def ctsegm_inmodal_norm_code_decode(hmap, regs, w_h_, codes_, offsets_, contour_std, dictionary, K=100):
    batch, cat, height, width = hmap.shape
    hmap = torch.sigmoid(hmap)

    # if flip test
    if batch > 1:
        hmap = (hmap[0:1] + flip_tensor(hmap[1:2])) / 2
        w_h_ = (w_h_[0:1] + flip_tensor(w_h_[1:2])) / 2
        regs = regs[0:1]
        codes_ = codes_[0:1]
        offsets_ = offsets_[0:1]
        contour_std = contour_std[0:1]

    batch = 1
    hmap = _nms(hmap)  # perform nms on heatmaps
    scores, inds, clses, ys, xs = _topk(hmap, K=K)

    regs = _tranpose_and_gather_feature(regs, inds)
    regs = regs.view(batch, K, 2)
    xs = xs.view(batch, K, 1) + regs[:, :, 0:1]
    ys = ys.view(batch, K, 1) + regs[:, :, 1:2]

    w_h_ = _tranpose_and_gather_feature(w_h_, inds)
    w_h_ = w_h_.view(batch, K, 2)
    contour_std = _tranpose_and_gather_feature(contour_std, inds)
    contour_std = contour_std.view(batch, K, 1)

    codes_ = _tranpose_and_gather_feature(codes_, inds)
    codes_ = codes_.view(batch, K, 64)

    clses = clses.view(batch, K, 1).float()
    scores = scores.view(batch, K, 1)

    bboxes = torch.cat([xs - w_h_[..., 0:1] / 2,
                        ys - w_h_[..., 1:2] / 2,
                        xs + w_h_[..., 0:1] / 2,
                        ys + w_h_[..., 1:2] / 2], dim=2)

    offsets_ = _tranpose_and_gather_feature(offsets_, inds)
    segms = torch.matmul(codes_, dictionary) * contour_std
    segms = segms.view(batch, K, 32, 2) + offsets_.view(batch, K, 1, 2) + \
            torch.cat([xs, ys], dim=2).view(batch, K, 1, 2)
    segmentations = torch.cat([segms.view(batch, K, -1), bboxes, scores, clses], dim=2)

    return segmentations
