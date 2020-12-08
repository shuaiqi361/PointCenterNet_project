import torch
import torch.nn as nn
import torch.nn.functional as F
from chamferdist import ChamferDistance
import math


def chamfer_distance_loss(pred_codes, pred_shapes, gt_shapes, mask, sparsity=0.1):
    chamfer = ChamferDistance()
    mask = mask[:, :, None].expand_as(gt_shapes).float()  # mask has been expanded to calculate the mean of 32 points
    loss = 0
    for r in pred_shapes:
        target_shape = (gt_shapes * mask).view(-1, 32, 2)
        shape_ = (r * mask).view(-1, 32, 2)
        loss += (chamfer(target_shape, shape_, reduction='sum')
                 + chamfer(shape_, target_shape, reduction='sum')) / (mask.sum() + 1e-4)

    loss_sparsity = sum(torch.sum(torch.abs(r * mask)) / (mask.sum() + 1e-4) for r in pred_codes)

    return loss / 2. + sparsity * loss_sparsity


def norm_contour_mapping_loss(pred_codes, pred_shapes, gt_shapes, gt_w_h, mask, sparsity=0.1):
    # gt_shape size: (bs, 128, 64)
    norm_factor = torch.sqrt(gt_w_h[:, :, 0] * gt_w_h[:, :, 1])[:, :, None] + 1e-4
    mask = mask[:, :, None].expand_as(gt_shapes).float()
    # print('Norm factor size: ', norm_factor.size())
    # print(gt_shapes.size())
    loss = sum(
        torch.sum(F.smooth_l1_loss(r * mask, gt_shapes * mask, reduction='none') / norm_factor) / (mask.sum() + 1e-4)
        for r in pred_shapes)

    loss_sparsity = sum(torch.sum(torch.abs(r * mask)) / (mask.sum() + 1e-4) for r in pred_codes)

    return loss + sparsity * loss_sparsity


def contour_mapping_loss(pred_codes, pred_shapes, gt_shapes, mask, sparsity=0.01, roll=True):
    batch_size, max_obj, n_dims = gt_shapes.size()
    mask = mask[:, :, None].expand_as(gt_shapes).float()
    scale_gt_shapes = torch.zeros(size=(batch_size, max_obj, 4), device=gt_shapes.device)
    scale_gt_shapes[:, :, 0], _ = torch.min(gt_shapes.view(batch_size, max_obj, 32, 2)[:, :, :, 0], dim=-1)
    scale_gt_shapes[:, :, 1], _ = torch.min(gt_shapes.view(batch_size, max_obj, 32, 2)[:, :, :, 1], dim=-1)
    scale_gt_shapes[:, :, 2], _ = torch.max(gt_shapes.view(batch_size, max_obj, 32, 2)[:, :, :, 0], dim=-1)
    scale_gt_shapes[:, :, 3], _ = torch.max(gt_shapes.view(batch_size, max_obj, 32, 2)[:, :, :, 1], dim=-1)
    scale_norm = torch.sqrt((scale_gt_shapes[:, :, 2] - scale_gt_shapes[:, :, 0]) ** 2 +
                            (scale_gt_shapes[:, :, 3] - scale_gt_shapes[:, :, 1]) ** 2).view(batch_size, max_obj, 1) + 1e-5
    if roll:
        cmm_gt_shapes = torch.zeros(size=gt_shapes.size(), device=gt_shapes.device)
        for bs in range(batch_size):
            for i in range(max_obj): # max loop should be 128, which is the maximum number of objects
                if mask[bs, i, 0] == 0:
                    break  # enumerated all objects of the current image
                else:
                    cmm_loss = math.inf
                    roll_index = 0
                    for j in range(n_dims // 2):
                        rolled_tensor = torch.roll(gt_shapes[bs, i, :], shifts=2 * j)
                        cmm_dist = torch.sum((rolled_tensor - pred_shapes[-1]) ** 2)
                        if cmm_dist < cmm_loss:
                            roll_index = int(j * 2)
                            cmm_loss = cmm_dist

                    cmm_gt_shapes[bs, i, :] = torch.roll(gt_shapes[bs, i, :], shifts=roll_index)

        loss_cmm = sum(torch.sum(
            F.l1_loss(r * mask, cmm_gt_shapes * mask, reduction='none') * 5 / scale_norm) / (mask.sum() + 1e-4) for r in pred_shapes)
    else:
        loss_cmm = sum(torch.sum(
            F.l1_loss(r * mask, gt_shapes * mask, reduction='none') * 5 / scale_norm) / (mask.sum() + 1e-4) for r in pred_shapes)

    loss_sparsity = sum(torch.sum(torch.abs(r * mask)) / (mask.sum() + 1e-4) for r in pred_codes)

    return loss_cmm + sparsity * loss_sparsity


# def contour_mapping_loss(pred_codes, pred_shapes, gt_shapes, mask):
#     # print('In cmm loss:')
#     # for c in pred_codes:
#     #     print('pred_codes: ', c.size())
#     # for c in pred_shapes:
#     #     print('pred_shapes: ', c.size())
#     # print('gt shape and mask size: ', gt_shapes.size(), mask.size())
#     mask = mask[:, :, None].expand_as(gt_shapes).float()
#     # print('After expand_as gt shape and mask size: ', gt_shapes.size(), mask.size())
#     loss_cmm = sum(F.smooth_l1_loss(r * mask, gt_shapes * mask, reduction='sum') / (mask.sum() + 1e-4) for r in pred_shapes)
#
#     loss_sparsity = sum(torch.sum(torch.abs(r * mask)) / (mask.sum() + 1e-4) for r in pred_codes)
#     # print('Loss items: ', loss_cmm.item() + 0.1 * loss_sparsity.item())
#     return loss_cmm + 0.1 * loss_sparsity


def _neg_loss_slow(preds, targets):
    pos_inds = targets == 1  # todo targets > 1-epsilon ?
    neg_inds = targets < 1  # todo targets < 1-epsilon ?

    neg_weights = torch.pow(1 - targets[neg_inds], 4)

    loss = 0
    for pred in preds:
        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]

        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def _neg_loss(preds, targets):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
        preds (B x c x h x w)
        gt_regr (B x c x h x w)
    '''
    pos_inds = targets.eq(1).float()
    neg_inds = targets.lt(1).float()

    neg_weights = torch.pow(1 - targets, 4)

    loss = 0
    for pred in preds:
        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss / len(preds)


def _reg_loss(regs, gt_regs, mask):
    mask = mask[:, :, None].expand_as(gt_regs).float()
    loss = sum(F.l1_loss(r * mask, gt_regs * mask, reduction='sum') / (mask.sum() + 1e-4) for r in regs)
    return loss / len(regs)


def norm_reg_loss(regs, gt_regs, mask):
    _, _, len_vec = gt_regs.shape
    mask = mask[:, :, None].expand_as(gt_regs).float()
    norm_gt_codes = torch.norm(gt_regs, dim=2, keepdim=True) + 1e-4
    loss = sum(torch.sum(F.smooth_l1_loss(r * mask, gt_regs * mask, reduction='none') * len_vec / norm_gt_codes) / (
                mask.sum() + 1e-4) for r in regs)
    return loss / len(regs)


def smooth_reg_loss(regs, gt_regs, mask):
    mask = mask[:, :, None].expand_as(gt_regs).float()
    loss = sum(F.smooth_l1_loss(r * mask, gt_regs * mask, reduction='sum') / (mask.sum() + 1e-4) for r in regs)
    return loss / len(regs)


def mse_reg_loss(regs, gt_regs, mask):
    mask = mask[:, :, None].expand_as(gt_regs).float()
    loss = sum(F.mse_loss(r * mask, gt_regs * mask, reduction='sum') / (mask.sum() + 1e-4) for r in regs)
    return loss / len(regs)
