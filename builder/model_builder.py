# -*- coding:utf-8 -*-
# author: Xinge
# @file: model_builder.py 

from network.cylinder_spconv_3d import get_model_class
from network.segmentator_3d_asymm_spconv import Asymm_3d_spconv
from network.cylinder_fea_generator import cylinder_fea
from network.unet_from_pcdet import UNetV2, UNetV2_resnet
from network.pcdet_resnet import VoxelResBackBone8x
import torch

def build(model_config, pretrained_unet = None):
    output_shape = model_config['output_shape']
    num_class = model_config['num_class']
    num_input_features = model_config['num_input_features']
    use_norm = model_config['use_norm']
    init_size = model_config['init_size']
    fea_dim = model_config['fea_dim']
    out_fea_dim = model_config['out_fea_dim']

    if model_config.get('pcdet_unet', False):
        cylinder_3d_spconv_seg = UNetV2_resnet(
            output_shape=output_shape,
            input_channels=num_input_features,
            nclasses=num_class)

    elif  model_config.get('pcdet_resnet', False):
        output_shape_ = output_shape.copy()
        output_shape_[0] *= 2
        output_shape_[1] *= 2
        output_shape_[2] *= 2

        cylinder_3d_spconv_seg = VoxelResBackBone8x(
            output_shape=output_shape_,
            input_channels=num_input_features,
            nclasses=num_class)

    else:
        cylinder_3d_spconv_seg = Asymm_3d_spconv(
            output_shape=output_shape,
            use_norm=use_norm,
            num_input_features=num_input_features,
            init_size=init_size,
            nclasses=num_class)

    if pretrained_unet is not None:

        if 'gcc3d' in pretrained_unet:
            checkpoint = torch.load(pretrained_unet, map_location='cpu')['model_state']

            model_dict = cylinder_3d_spconv_seg.conv_input.state_dict()
            pretrained_dict = {k[len('backbone_3d.conv_input.'):]: v for k, v in checkpoint.items() if
                               k.startswith('backbone_3d.conv_input.')}
            # pretrained_dict['0.weight'] = pretrained_dict['0.weight'].permute(4, 0, 1, 2, 3)
            model_dict.update(pretrained_dict)
            cylinder_3d_spconv_seg.conv_input.load_state_dict(model_dict)

            model_dict = cylinder_3d_spconv_seg.conv1.state_dict()
            pretrained_dict = {k[len('backbone_3d.conv1.'):]: v for k, v in checkpoint.items() if
                               k.startswith('backbone_3d.conv1.')}
            # pretrained_dict['0.weight'] = pretrained_dict['0.weight'].permute(4, 0, 1, 2, 3)
            model_dict.update(pretrained_dict)
            cylinder_3d_spconv_seg.conv1.load_state_dict(model_dict)

            model_dict = cylinder_3d_spconv_seg.conv2.state_dict()
            pretrained_dict = {k[len('backbone_3d.conv2.'):]: v for k, v in checkpoint.items() if
                               k.startswith('backbone_3d.conv2.')}
            # pretrained_dict['0.weight'] = pretrained_dict['0.weight'].permute(4, 0, 1, 2, 3)
            model_dict.update(pretrained_dict)
            cylinder_3d_spconv_seg.conv2.load_state_dict(model_dict)

            model_dict = cylinder_3d_spconv_seg.conv3.state_dict()
            pretrained_dict = {k[len('backbone_3d.conv3.'):]: v for k, v in checkpoint.items() if
                               k.startswith('backbone_3d.conv3.')}
            # pretrained_dict['0.weight'] = pretrained_dict['0.weight'].permute(4, 0, 1, 2, 3)
            model_dict.update(pretrained_dict)
            cylinder_3d_spconv_seg.conv3.load_state_dict(model_dict)

            model_dict = cylinder_3d_spconv_seg.conv4.state_dict()
            pretrained_dict = {k[len('backbone_3d.conv4.'):]: v for k, v in checkpoint.items() if
                               k.startswith('backbone_3d.conv4.')}
            # pretrained_dict['0.weight'] = pretrained_dict['0.weight'].permute(4, 0, 1, 2, 3)
            model_dict.update(pretrained_dict)
            cylinder_3d_spconv_seg.conv4.load_state_dict(model_dict)

        elif 'strl' in pretrained_unet:

            checkpoint = torch.load(pretrained_unet, map_location='cpu')['state_dict']

            model_dict = cylinder_3d_spconv_seg.conv_input.state_dict()
            pretrained_dict = {k[len('pts_middle_encoder.conv_input.'):]: v for k, v in checkpoint.items() if
                               k.startswith('pts_middle_encoder.conv_input.')}
            # pretrained_dict['0.weight'] = pretrained_dict['0.weight'].permute(4, 0, 1, 2, 3)
            model_dict.update(pretrained_dict)
            cylinder_3d_spconv_seg.conv_input.load_state_dict(model_dict)

            model_dict = cylinder_3d_spconv_seg.conv1.state_dict()
            pretrained_dict = {k[len('pts_middle_encoder.conv1.'):]: v for k, v in checkpoint.items() if
                               k.startswith('pts_middle_encoder.conv1.')}
            # pretrained_dict['0.weight'] = pretrained_dict['0.weight'].permute(4, 0, 1, 2, 3)
            model_dict.update(pretrained_dict)
            cylinder_3d_spconv_seg.conv1.load_state_dict(model_dict)

            model_dict = cylinder_3d_spconv_seg.conv2.state_dict()
            pretrained_dict = {k[len('pts_middle_encoder.conv2.'):]: v for k, v in checkpoint.items() if
                               k.startswith('pts_middle_encoder.conv2.')}
            # pretrained_dict['0.weight'] = pretrained_dict['0.weight'].permute(4, 0, 1, 2, 3)
            model_dict.update(pretrained_dict)
            cylinder_3d_spconv_seg.conv2.load_state_dict(model_dict)

            model_dict = cylinder_3d_spconv_seg.conv3.state_dict()
            pretrained_dict = {k[len('pts_middle_encoder.conv3.'):]: v for k, v in checkpoint.items() if
                               k.startswith('pts_middle_encoder.conv3.')}
            # pretrained_dict['0.weight'] = pretrained_dict['0.weight'].permute(4, 0, 1, 2, 3)
            model_dict.update(pretrained_dict)
            cylinder_3d_spconv_seg.conv3.load_state_dict(model_dict)

            model_dict = cylinder_3d_spconv_seg.conv4.state_dict()
            pretrained_dict = {k[len('pts_middle_encoder.conv4.'):]: v for k, v in checkpoint.items() if
                               k.startswith('pts_middle_encoder.conv4.')}
            # pretrained_dict['0.weight'] = pretrained_dict['0.weight'].permute(4, 0, 1, 2, 3)
            model_dict.update(pretrained_dict)
            cylinder_3d_spconv_seg.conv4.load_state_dict(model_dict)

        elif 'ProposalContrast' in pretrained_unet:

            checkpoint = torch.load(pretrained_unet, map_location='cpu')['state_dict']

            model_dict = cylinder_3d_spconv_seg.conv_input.state_dict()
            pretrained_dict = {k[len('backbone.conv_input.'):]: v for k, v in checkpoint.items() if
                               k.startswith('backbone.conv_input.')}
            # pretrained_dict['0.weight'] = pretrained_dict['0.weight'].permute(4, 0, 1, 2, 3)
            model_dict.update(pretrained_dict)
            cylinder_3d_spconv_seg.conv_input.load_state_dict(model_dict)

            model_dict = cylinder_3d_spconv_seg.conv1.state_dict()
            pretrained_dict = {k[len('backbone.conv1.'):]: v for k, v in checkpoint.items() if
                               k.startswith('backbone.conv1.')}
            # pretrained_dict['0.weight'] = pretrained_dict['0.weight'].permute(4, 0, 1, 2, 3)
            model_dict.update(pretrained_dict)
            cylinder_3d_spconv_seg.conv1.load_state_dict(model_dict)

            model_dict = cylinder_3d_spconv_seg.conv2.state_dict()
            pretrained_dict = {k[len('backbone.conv2.'):]: v for k, v in checkpoint.items() if
                               k.startswith('backbone.conv2.')}
            # pretrained_dict['0.weight'] = pretrained_dict['0.weight'].permute(4, 0, 1, 2, 3)
            model_dict.update(pretrained_dict)
            cylinder_3d_spconv_seg.conv2.load_state_dict(model_dict)

            model_dict = cylinder_3d_spconv_seg.conv3.state_dict()
            pretrained_dict = {k[len('backbone.conv3.'):]: v for k, v in checkpoint.items() if
                               k.startswith('backbone.conv3.')}
            # pretrained_dict['0.weight'] = pretrained_dict['0.weight'].permute(4, 0, 1, 2, 3)
            model_dict.update(pretrained_dict)
            cylinder_3d_spconv_seg.conv3.load_state_dict(model_dict)

            model_dict = cylinder_3d_spconv_seg.conv4.state_dict()
            pretrained_dict = {k[len('backbone.conv4.'):]: v for k, v in checkpoint.items() if
                               k.startswith('backbone.conv4.')}
            # pretrained_dict['0.weight'] = pretrained_dict['0.weight'].permute(4, 0, 1, 2, 3)
            model_dict.update(pretrained_dict)
            cylinder_3d_spconv_seg.conv4.load_state_dict(model_dict)

        else:

            checkpoint = torch.load(pretrained_unet, map_location='cpu')['model_state']
            model_dict = cylinder_3d_spconv_seg.state_dict()
            pretrained_dict = {k[len('backbone_3d.'):]: v for k, v in checkpoint.items() if
                               k.startswith('backbone_3d.')}
            # pretrained_dict['0.weight'] = pretrained_dict['0.weight'].permute(4, 0, 1, 2, 3)
            model_dict.update(pretrained_dict)
            cylinder_3d_spconv_seg.load_state_dict(model_dict)


    if model_config.get('pcdet_resnet', False):

        cy_fea_net = cylinder_fea(grid_size=output_shape_,
                                  fea_dim=fea_dim,
                                  out_pt_fea_dim=out_fea_dim,
                                  fea_compre=num_input_features)
    else:
        cy_fea_net = cylinder_fea(grid_size=output_shape,
                                  fea_dim=fea_dim,
                                  out_pt_fea_dim=out_fea_dim,
                                  fea_compre=num_input_features)

    model = get_model_class(model_config["model_architecture"])(
        cylin_model=cy_fea_net,
        segmentator_spconv=cylinder_3d_spconv_seg,
        sparse_shape=output_shape
    )

    return model
