import torch
import torch.nn as nn
import torch.nn.functional as F


def contour_mapping_loss(pred_codes, pred_shapes, gt_shapes, mask):
    # print('In cmm loss:')
    # for c in pred_codes:
    #     print('pred_codes: ', c.size())
    # for c in pred_shapes:
    #     print('pred_shapes: ', c.size())
    # print('gt shape and mask size: ', gt_shapes.size(), mask.size())
    mask = mask[:, :, None].expand_as(gt_shapes).float()
    # print('After expand_as gt shape and mask size: ', gt_shapes.size(), mask.size())
    loss_cmm = sum(F.smooth_l1_loss(r * mask, gt_shapes * mask, reduction='sum') / (mask.sum() + 1e-4) for r in pred_shapes)

    loss_sparsity = sum(torch.sum(torch.abs(r * mask)) / (mask.sum() + 1e-4) for r in pred_codes)
    # print('Loss items: ', loss_cmm.item() + 0.1 * loss_sparsity.item())
    return loss_cmm + 0.1 * loss_sparsity


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
    mask = mask[:, :, None].expand_as(gt_regs).float()
    norm_gt_codes = torch.norm(gt_regs, dim=2, keepdim=True) + 1e-4
    loss = sum(torch.sum(F.smooth_l1_loss(r * mask, gt_regs * mask, reduction='none') / norm_gt_codes) / (mask.sum() + 1e-4) for r in regs)
    return loss / len(regs)


def smooth_reg_loss(regs, gt_regs, mask):
    mask = mask[:, :, None].expand_as(gt_regs).float()
    loss = sum(F.smooth_l1_loss(r * mask, gt_regs * mask, reduction='sum') / (mask.sum() + 1e-4) for r in regs)
    return loss / len(regs)


def mse_reg_loss(regs, gt_regs, mask):
    mask = mask[:, :, None].expand_as(gt_regs).float()
    loss = sum(F.mse_loss(r * mask, gt_regs * mask, reduction='sum') / (mask.sum() + 1e-4) for r in regs)
    return loss / len(regs)
