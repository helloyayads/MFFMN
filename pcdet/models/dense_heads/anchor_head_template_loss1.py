import numpy as np
import torch
import torch.nn as nn

from ...utils import box_coder_utils, common_utils, loss_utils
from .target_assigner.anchor_generator import AnchorGenerator
from .target_assigner.atss_target_assigner import ATSSTargetAssigner
from .target_assigner.axis_aligned_target_assigner import AxisAlignedTargetAssigner


class AnchorHeadTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, class_names, grid_size, point_cloud_range, predict_boxes_when_training):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.class_names = class_names
        self.predict_boxes_when_training = predict_boxes_when_training
        self.use_multihead = self.model_cfg.get('USE_MULTIHEAD', False)

        anchor_target_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        self.box_coder = getattr(box_coder_utils, anchor_target_cfg.BOX_CODER)(
            num_dir_bins=anchor_target_cfg.get('NUM_DIR_BINS', 6),
            **anchor_target_cfg.get('BOX_CODER_CONFIG', {})
        )

        anchor_generator_cfg = self.model_cfg.ANCHOR_GENERATOR_CONFIG
        anchors, self.num_anchors_per_location = self.generate_anchors(
            anchor_generator_cfg, grid_size=grid_size, point_cloud_range=point_cloud_range,
            anchor_ndim=self.box_coder.code_size
        )
        self.anchors = [x.cuda() for x in anchors]
        self.target_assigner = self.get_target_assigner(anchor_target_cfg)

        self.forward_ret_dict = {}
        self.build_losses(self.model_cfg.LOSS_CONFIG)

    @staticmethod
    def generate_anchors(anchor_generator_cfg, grid_size, point_cloud_range, anchor_ndim=7):
        anchor_generator = AnchorGenerator(
            anchor_range=point_cloud_range,
            anchor_generator_config=anchor_generator_cfg
        )
        feature_map_size = [grid_size[:2] // config['feature_map_stride'] for config in anchor_generator_cfg]
        anchors_list, num_anchors_per_location_list = anchor_generator.generate_anchors(feature_map_size)

        if anchor_ndim != 7:
            for idx, anchors in enumerate(anchors_list):
                pad_zeros = anchors.new_zeros([*anchors.shape[0:-1], anchor_ndim - 7])
                new_anchors = torch.cat((anchors, pad_zeros), dim=-1)
                anchors_list[idx] = new_anchors

        return anchors_list, num_anchors_per_location_list

    def get_target_assigner(self, anchor_target_cfg):
        if anchor_target_cfg.NAME == 'ATSS':
            target_assigner = ATSSTargetAssigner(
                topk=anchor_target_cfg.TOPK,
                box_coder=self.box_coder,
                use_multihead=self.use_multihead,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        elif anchor_target_cfg.NAME == 'AxisAlignedTargetAssigner':
            target_assigner = AxisAlignedTargetAssigner(
                model_cfg=self.model_cfg,
                class_names=self.class_names,
                box_coder=self.box_coder,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        else:
            raise NotImplementedError
        return target_assigner

    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        reg_loss_name = 'WeightedSmoothL1Loss' if losses_cfg.get('REG_LOSS_TYPE', None) is None \
            else losses_cfg.REG_LOSS_TYPE
        self.add_module(
            'reg_loss_func',
            getattr(loss_utils, reg_loss_name)(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )
        self.add_module(
            'dir_loss_func',
            loss_utils.WeightedCrossEntropyLoss()
        )

    def assign_targets(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:

        """
        targets_dict = self.target_assigner.assign_targets(
            self.anchors, gt_boxes
        )
        return targets_dict

    def giou_loss_3d(slef,pred_boxes, target_boxes):
        """
        计算 3D 边界框的 GIoU Loss。

        Args:
            pred_boxes: 预测框，形状为 (B, N, 7)，格式为 (x, y, z, dx, dy, dz, heading)
            target_boxes: 真实框，形状为 (B, N, 7)，格式为 (x, y, z, dx, dy, dz, heading)

        Returns:
            GIoU Loss，标量值
        """
        # 提取中心点坐标和尺寸
        pred_center = pred_boxes[..., :3]  # (B, N, 3)
        pred_size = pred_boxes[..., 3:6]  # (B, N, 3)
        target_center = target_boxes[..., :3]
        target_size = target_boxes[..., 3:6]

        # 计算预测框和目标框的边界
        pred_min = pred_center - pred_size / 2
        pred_max = pred_center + pred_size / 2
        target_min = target_center - target_size / 2
        target_max = target_center + target_size / 2

        # 计算交集区域
        inter_min = torch.max(pred_min, target_min)
        inter_max = torch.min(pred_max, target_max)
        inter_size = torch.clamp(inter_max - inter_min, min=0)  # 确保大小非负
        inter_volume = inter_size[..., 0] * inter_size[..., 1] * inter_size[..., 2]

        # 计算预测框和目标框的体积
        pred_volume = pred_size[..., 0] * pred_size[..., 1] * pred_size[..., 2]
        target_volume = target_size[..., 0] * target_size[..., 1] * target_size[..., 2]
        union_volume = pred_volume + target_volume - inter_volume

        # 计算 IoU
        iou = inter_volume / (union_volume + 1e-6)

        # 计算最小包围框
        enclose_min = torch.min(pred_min, target_min)
        enclose_max = torch.max(pred_max, target_max)
        enclose_size = torch.clamp(enclose_max - enclose_min, min=0)
        enclose_volume = enclose_size[..., 0] * enclose_size[..., 1] * enclose_size[..., 2]

        # 计算 GIoU
        giou = iou - (enclose_volume - union_volume) / (enclose_volume + 1e-6)

        # GIoU Loss
        giou_loss = 1 - giou
        return giou_loss.mean()  # 返回平均损失

    def siou_loss_3d(self,pred_boxes, target_boxes):
        """
        计算 3D 边界框的 SIoU Loss，支持 [x, y, z, l, w, h, heading, cos_diff, sin_diff] 格式。

        Args:
            pred_boxes: 预测框，形状为 (B, N, 9)，格式为 [x, y, z, l, w, h, heading, cos_diff, sin_diff]
            target_boxes: 真实框，形状为 (B, N, 9)，格式为 [x, y, z, l, w, h, heading, cos_diff, sin_diff]

        Returns:
            SIoU Loss，标量值
        """
        # 提取中心点坐标和尺寸
        pred_center = pred_boxes[..., :3]  # (B, N, 3) -> [x, y, z]
        pred_size = pred_boxes[..., 3:6]  # (B, N, 3) -> [l, w, h]
        target_center = target_boxes[..., :3]
        target_size = target_boxes[..., 3:6]

        # 提取角度相关部分
        pred_heading = pred_boxes[..., 6]  # (B, N) -> heading
        target_heading = target_boxes[..., 6]  # (B, N) -> heading
        pred_cos_diff = pred_boxes[..., 7]  # (B, N) -> cos_diff
        pred_sin_diff = pred_boxes[..., 8]  # (B, N) -> sin_diff
        target_cos_diff = target_boxes[..., 7]  # (B, N) -> cos_diff
        target_sin_diff = target_boxes[..., 8]  # (B, N) -> sin_diff

        # 计算预测框和目标框的边界
        pred_min = pred_center - pred_size / 2
        pred_max = pred_center + pred_size / 2
        target_min = target_center - target_size / 2
        target_max = target_center + target_size / 2

        # 计算交集区域
        inter_min = torch.max(pred_min, target_min)
        inter_max = torch.min(pred_max, target_max)
        inter_size = torch.clamp(inter_max - inter_min, min=0)  # 确保大小非负
        inter_volume = inter_size[..., 0] * inter_size[..., 1] * inter_size[..., 2]

        # 计算预测框和目标框的体积
        pred_volume = pred_size[..., 0] * pred_size[..., 1] * pred_size[..., 2]
        target_volume = target_size[..., 0] * target_size[..., 1] * target_size[..., 2]
        union_volume = pred_volume + target_volume - inter_volume

        # 计算 IoU
        iou = inter_volume / (union_volume + 1e-6)

        # 角度惩罚项 Δr
        # 使用 cos_diff 和 sin_diff 计算角度差异
        angle_diff = torch.sqrt((pred_cos_diff - target_cos_diff) ** 2 + (pred_sin_diff - target_sin_diff) ** 2)
        delta_r = angle_diff.mean(dim=-1)

        # 距离惩罚项 Δc
        center_distance = torch.norm(pred_center - target_center, dim=-1)
        diagonal_length = torch.norm(pred_size + target_size, dim=-1)
        delta_c = (center_distance / (diagonal_length + 1e-6)).mean(dim=-1)

        # 计算 SIoU
        siou = 1 - iou + delta_r + delta_c

        # SIoU Loss
        siou_loss = siou.mean()  # 返回平均损失
        return siou_loss

    def get_cls_layer_loss(self):
        cls_preds = self.forward_ret_dict['cls_preds']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(cls_preds.shape[0])
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        reg_weights = positives.float()
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1

        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        cls_targets = cls_targets.unsqueeze(dim=-1)

        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
        )
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0)
        cls_preds = cls_preds.view(batch_size, -1, self.num_class)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / batch_size

        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        tb_dict = {
            'rpn_loss_cls': cls_loss.item()
        }
        return cls_loss, tb_dict

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        """
            改进版角度差计算，融合正弦差和余弦相似度
            输入形状：(..., box_dim)
            输出形状保持不变
            """
        assert dim != -1, "角度维度不能是最后一个维度"

        # 原始角度值
        pred_angle = boxes1[..., dim:dim + 1]
        target_angle = boxes2[..., dim:dim + 1]

        # 改进1：使用角度差的正弦和余弦组合
        angle_diff = pred_angle - target_angle
        sin_diff = torch.sin(angle_diff)
        cos_diff = torch.cos(angle_diff)

        # 改进2：保留原始角度信息的同时编码差异
        pred_sin = torch.sin(pred_angle)
        pred_cos = torch.cos(pred_angle)
        target_sin = torch.sin(target_angle)
        target_cos = torch.cos(target_angle)

        # 新编码方案：组合原始特征和差异特征
        new_pred_angle = torch.cat([
            pred_sin * target_cos,  # 保留原方法
            cos_diff,  # 新增余弦相似度项
            sin_diff  # 新增正弦差异项
        ], dim=-1)

        new_target_angle = torch.cat([
            pred_cos * target_sin,  # 保留原方法
            torch.ones_like(cos_diff),  # 余弦相似度目标
            torch.zeros_like(sin_diff)  # 正弦差异目标
        ], dim=-1)

        # 调整维度结构
        boxes1 = torch.cat([
            boxes1[..., :dim],
            new_pred_angle,
            boxes1[..., dim + 1:]
        ], dim=-1)

        boxes2 = torch.cat([
            boxes2[..., :dim],
            new_target_angle,
            boxes2[..., dim + 1:]
        ], dim=-1)

        return boxes1, boxes2

    @staticmethod
    def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0, num_bins=2):
        batch_size = reg_targets.shape[0]
        anchors = anchors.view(batch_size, -1, anchors.shape[-1])
        rot_gt = reg_targets[..., 6] + anchors[..., 6]
        offset_rot = common_utils.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)

        if one_hot:
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype,
                                      device=dir_cls_targets.device)
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    def get_box_reg_layer_loss(self):
        box_preds = self.forward_ret_dict['box_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(box_preds.shape[0])

        positives = box_cls_labels > 0
        reg_weights = positives.float()
        pos_normalizer = positives.sum(1, keepdim=True).float()
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])
        # sin(a - b) = sinacosb-cosasinb
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
        loc_loss_src = self.siou_loss_3d(box_preds_sin, reg_targets_sin)  # 直接调用 SIoU Loss
        # loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]
        print(box_preds.shape, box_reg_targets.shape)
        # loc_loss_src = self.giou_loss_3d(box_preds, box_reg_targets)  # 直接调用 GIoU Loss



        loc_loss = loc_loss_src.sum() / batch_size

        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        box_loss = loc_loss
        tb_dict = {
            'rpn_loss_loc': loc_loss.item()
        }

        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET,
                num_bins=self.model_cfg.NUM_DIR_BINS
            )

            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            weights = positives.type_as(dir_logits)
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()

        return box_loss, tb_dict

    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat([anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1])
                                     for anchor in self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
            if not isinstance(cls_preds, list) else cls_preds
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1) if not isinstance(box_preds, list) \
            else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)

        if dir_cls_preds is not None:
            dir_offset = self.model_cfg.DIR_OFFSET
            dir_limit_offset = self.model_cfg.DIR_LIMIT_OFFSET
            dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1) if not isinstance(dir_cls_preds, list) \
                else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

            period = (2 * np.pi / self.model_cfg.NUM_DIR_BINS)
            dir_rot = common_utils.limit_period(
                batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
            )
            batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

        if isinstance(self.box_coder, box_coder_utils.PreviousResidualDecoder):
            batch_box_preds[..., 6] = common_utils.limit_period(
                -(batch_box_preds[..., 6] + np.pi / 2), offset=0.5, period=np.pi * 2
            )

        return batch_cls_preds, batch_box_preds

    def forward(self, **kwargs):
        raise NotImplementedError
