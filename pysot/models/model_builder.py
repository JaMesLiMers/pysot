# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss
from pysot.models.backbone import get_backbone
from pysot.models.head import get_rpn_head, get_mask_head, get_refine_head
from pysot.models.neck import get_neck
from pysot.models.memory_network import get_key_generator, get_memory_base


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build Memory Network
        if cfg.MEMORY.MEMORY:
            self.key_generator = get_key_generator(cfg.MEMORY.KEY_GENERATOR_TYPE,
                                                   **cfg.MEMORY.KEY_GENERATOR_KWARGS)

            self.memory_base = get_memory_base(cfg.MEMORY.MEMORY_BASE_TYPE,
                                               **cfg.MEMORY.MEMORY_BASE_KWARGS)
            
        # build rpn head
        self.rpn_head = get_rpn_head(cfg.RPN.TYPE,
                                     **cfg.RPN.KWARGS)

        # build mask head
        if cfg.MASK.MASK:
            self.mask_head = get_mask_head(cfg.MASK.TYPE,
                                           **cfg.MASK.KWARGS)

            if cfg.REFINE.REFINE:
                self.refine_head = get_refine_head(cfg.REFINE.TYPE)

    def template(self, z):
        zf = self.backbone(z)
        if cfg.MASK.MASK:
            zf = zf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        if cfg.MEMORY.MEMORY:
            self.memory_base.set_f_z_value(zf)
        self.zf = zf

    def template_update(self, z):
        # feature extract
        zf = self.backbone(z)
        if cfg.MASK.MASK:
            zf = zf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        # added to memory
        zk = self.key_generator.key_vector_z(zf)
        result = self.memory_base.memory_write(zk, zf, gamma=cfg.MEMORY.GAMMA, c_init=cfg.MEMORY.C_INIT)
        return result

    def template_generate(self, xf):
        f_z_value = self.memory_base.f_z_value
        xk_list = self.key_generator.forward(self.zf, xf)
        xk = xk_list[-1]
        self.zf = self.memory_base(xk)

    def track(self, x):
        xf = self.backbone(x)
        if cfg.MASK.MASK:
            self.xf = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        if cfg.MEMORY.MEMORY:
            self.template_generate(xf[-1])
        cls, loc = self.rpn_head(self.zf, xf)
        if cfg.MASK.MASK:
            mask, self.mask_corr_feature = self.mask_head(self.zf, xf)
        return {
                'cls': cls,
                'loc': loc,
                'mask': mask if cfg.MASK.MASK else None
               }

    def mask_refine(self, pos):
        return self.refine_head(self.xf, self.mask_corr_feature, pos)

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()

        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)
        if cfg.MASK.MASK or cfg.MEMORY.MEMORY:
            zf = zf[-1]
            self.xf_refine = xf[:-1]
            xf = xf[-1]
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)
        if cfg.MEMORY.MEMORY:
            xk_vec = self.key_generator(zf, xf)[-1]
            zk_vec = self.key_generator.key_vector_z(zf)
            mem_smi = self.memory_base.cos_sim(zk_vec, xk_vec)
            mem_smi = (mem_smi + 1 ) / 2
        cls, loc = self.rpn_head(zf, xf)

        # get loss
        cls = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls, label_cls)
        loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)

        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss

        if cfg.MASK.MASK:
            # TODO
            mask, self.mask_corr_feature = self.mask_head(zf, xf)
            mask_loss = None
            outputs['total_loss'] += cfg.TRAIN.MASK_WEIGHT * mask_loss
            outputs['mask_loss'] = mask_loss
        
        if cfg.MEMORY.MEMORY:
            mem_cls_label = torch.max(label_cls.view(xf.size()[0], -1), 1)[0].float()
            mem_loss = F.binary_cross_entropy(mem_smi, mem_cls_label)
            outputs['total_loss'] += cfg.TRAIN.MEM_WEIGHT * mem_loss
            outputs['mem_loss'] = mem_loss
        return outputs
