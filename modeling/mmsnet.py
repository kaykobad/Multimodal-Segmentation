'''
These codes are from this Repo: https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/modeling/backbone/resnet.py
'''


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
import functools
from functools import partial


class ChannelAttentionBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttentionBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SpatialAttentionBlock(nn.Module):
    def __init__(self):
        super(SpatialAttentionBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(2, 1, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        mean_ = torch.mean(x, dim=1, keepdim=True)
        max_ = torch.max(x, dim=1, keepdim=True).values
        cat_ = torch.cat((mean_, max_), dim=1)
        w = self.fc(cat_)
        return x * w


class ChannelAndSpatialAttentionBlock(nn.Module):
    def __init__(self, channel, reduction=16, residue=True):
        super(ChannelAndSpatialAttentionBlock, self).__init__()
        self.residue = residue
        self.channel_attention = ChannelAttentionBlock(channel, reduction=reduction)
        self.spatial_attention = SpatialAttentionBlock()
    
    def forward(self, x):
        y = self.channel_attention(x)
        y = self.spatial_attention(y)
        return x+y if self.residue else y


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, BatchNorm=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, output_stride, BatchNorm, pretrained=True, input_dim=3, weight='resnet101'):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.input_dim = input_dim
        self.weight = weight

        blocks = [1, 2, 4]
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
        elif output_stride == 8:
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
        else:
            raise NotImplementedError

        # Modules
        self.conv1 = nn.Conv2d(self.input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=strides[0], dilation=dilations[0], BatchNorm=BatchNorm)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=strides[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=strides[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.layer4 = self._make_MG_unit(block, 512, blocks=blocks, stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        # self.layer4 = self._make_layer(block, 512, layers[3], stride=strides[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self._init_weight()

        if pretrained:
            self._load_pretrained_model()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation, downsample, BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def _make_MG_unit(self, block, planes, blocks, stride=1, dilation=1, BatchNorm=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                BatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=blocks[0]*dilation,
                            downsample=downsample, BatchNorm=BatchNorm))
        self.inplanes = planes * block.expansion
        for i in range(1, len(blocks)):
            layers.append(block(self.inplanes, planes, stride=1,
                                dilation=blocks[i]*dilation, BatchNorm=BatchNorm))

        return nn.Sequential(*layers)

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        low_level_feat = x
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x, low_level_feat

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _load_pretrained_model(self):
        # # pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
        # pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-cd907fc2.pth')
        # model_dict = {}
        # state_dict = self.state_dict()
        # for k, v in pretrain_dict.items():
        #     if k in state_dict:
        #         model_dict[k] = v
        # state_dict.update(model_dict)
        # self.load_state_dict(state_dict)
        if self.weight == 'resnet101' or self.weight == 'resnet':
            # pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
            pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet101-cd907fc2.pth')
        elif self.weight == 'resnet50':
            pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet50-11ad3fa6.pth')
        elif self.weight == 'resnet34':
            pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet34-b627a593.pth')
        elif self.weight == 'resnet18':
            pretrain_dict = model_zoo.load_url('https://download.pytorch.org/models/resnet18-f37072fd.pth')
        
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)


def resnet101(output_stride, BatchNorm, pretrained=True, input_dim=3):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], output_stride, BatchNorm, pretrained=pretrained, input_dim=input_dim)
    return model


def resnet50(output_stride, BatchNorm, pretrained=True, input_dim=3):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], output_stride, BatchNorm, pretrained=pretrained, input_dim=input_dim, weight='resnet50')
    return model


def build_backbone(backbone, output_stride, BatchNorm, input_dim=3, pretrained=True):
    if backbone == 'resnet' or backbone == 'resnet101':
        return resnet101(output_stride, BatchNorm, input_dim=input_dim, pretrained=pretrained)
    elif backbone == 'resnet50':
        return resnet50(output_stride, BatchNorm, input_dim=input_dim, pretrained=pretrained)
    else:
        raise NotImplementedError


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm):
        super(ASPP, self).__init__()
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             BatchNorm(256),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = BatchNorm(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_aspp(backbone, output_stride, BatchNorm):
    return ASPP(backbone, output_stride, BatchNorm)


class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, num_modalities=1):
        super(Decoder, self).__init__()
        # print("-- Backbone", backbone)
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
            last_conv_input = 48 + 256
        elif backbone == 'resnet18' or backbone == 'resnet34' or backbone == 'resnet50' or backbone == 'resnet101':
            low_level_inplanes = 256
            last_conv_input = 48 + 256
        else:
            raise NotImplementedError

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(last_conv_input, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)

        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, backbone, BatchNorm, num_modalities=1):
    return Decoder(num_classes, backbone, BatchNorm, num_modalities)


# norm = 'avg' -> Average / 'bn' -> BatchNorm / 'bnr' -> BatchNorm + ReLU
class MMSNetForMCubeS(nn.Module):
    def __init__(self, 
        backbone='resnet', 
        output_stride=16, 
        num_classes=20,      
        sync_bn=True, 
        freeze_bn=False,
        use_nir=False,
        use_pol=False,
        norm='avg',
    ): 
        super(MMSNetForMCubeS, self).__init__()
        self.use_nir = use_nir
        self.use_pol = use_pol
        self.freeze_bn = freeze_bn

        self.backbones = []
        self.decoders = []
        self.num_modalities = 1
        self.norm = norm

        self.low_level_feature_channels = 256
        self.low_level_feature_shape = (256, 128, 128)

        self.high_level_feature_channels = 256
        self.high_level_feature_shape = (256, 32, 32)

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.rgb_backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.rgb_aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.rgb_llf = ChannelAndSpatialAttentionBlock(self.low_level_feature_channels)
        self.rgb_hlf = ChannelAndSpatialAttentionBlock(self.high_level_feature_channels)
        self.backbones.extend([self.rgb_backbone, self.rgb_llf, self.rgb_hlf])
        self.decoders.append(self.rgb_aspp)

        if self.use_nir:
            self.num_modalities += 1
            self.nir_backbone = build_backbone(backbone, output_stride, BatchNorm, input_dim=3, pretrained=True)
            self.nir_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.nir_llf = ChannelAndSpatialAttentionBlock(self.low_level_feature_channels)
            self.nir_hlf = ChannelAndSpatialAttentionBlock(self.high_level_feature_channels)
            self.backbones.extend([self.nir_backbone, self.nir_llf, self.nir_hlf])
            self.decoders.append(self.nir_aspp)
        if self.use_pol:
            self.num_modalities += 1
            self.pol_backbone = build_backbone(backbone, output_stride, BatchNorm, input_dim=3, pretrained=True)
            self.pol_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.pol_llf = ChannelAndSpatialAttentionBlock(self.low_level_feature_channels)
            self.pol_hlf = ChannelAndSpatialAttentionBlock(self.high_level_feature_channels)
            self.backbones.extend([self.pol_backbone, self.pol_llf, self.pol_hlf])
            self.decoders.append(self.pol_aspp)

        if self.norm == 'bnr':
            self.hlf_norm = nn.Sequential(
                BatchNorm(self.high_level_feature_channels),
                nn.ReLU(inplace=True)
            )
            self.llf_norm = nn.Sequential(
                BatchNorm(self.low_level_feature_channels),
                nn.ReLU(inplace=True)
            )
            self.backbones.extend([self.hlf_norm, self.llf_norm])
        elif self.norm == 'bn':
            self.hlf_norm = BatchNorm(self.high_level_feature_channels)
            self.llf_norm = BatchNorm(self.low_level_feature_channels)
            self.backbones.extend([self.hlf_norm, self.llf_norm])

        self.decoder = build_decoder(num_classes, backbone, BatchNorm, num_modalities=self.num_modalities)
        self.decoders.append(self.decoder)

    def forward(self, rgb, nir=None, pol=None):
        x1, low_level_feat1 = self.rgb_backbone(rgb)
        x1 = self.rgb_aspp(x1)
        x1 = self.rgb_hlf(x1)
        low_level_feat1 = self.rgb_llf(low_level_feat1)
        # print("---------", x1.shape, low_level_feat1.shape)
        # x1 = torch.Size([8, 256, 32, 32]),  low_level_feat1 = torch.Size([8, 256, 128, 128])
        x = x1
        low_level_feat = low_level_feat1
        all_hlf = [x1]
        all_llf = [low_level_feat1]
        active_modalities = 1

        if self.use_nir and nir is not None:
            x2, low_level_feat2 = self.nir_backbone(nir)
            x2 = self.nir_aspp(x2)
            x2 = self.nir_hlf(x2)
            low_level_feat2 = self.nir_llf(low_level_feat2)
            x = torch.add(x, x2)
            low_level_feat = torch.add(low_level_feat, low_level_feat2)
            active_modalities += 1 
            all_hlf.append(x2)
            all_llf.append(low_level_feat2)
        if self.use_pol and pol is not None:
            x5, low_level_feat5 = self.pol_backbone(pol)
            x5 = self.pol_aspp(x5)
            x5 = self.pol_hlf(x5)
            low_level_feat5 = self.pol_llf(low_level_feat5)
            x = torch.add(x, x5)
            low_level_feat = torch.add(low_level_feat, low_level_feat5)
            active_modalities += 1
            all_hlf.append(x5)
            all_llf.append(low_level_feat5)

        # print("X1 and X shape:", x1.shape, x.shape)
        # print("low_level_feat1 and low_level_feat shape:", low_level_feat1.shape, low_level_feat.shape)

        if self.norm == 'avg':
            x = torch.div(x, active_modalities)
            low_level_feat = torch.div(low_level_feat, active_modalities)
        elif self.norm == 'max':
            # print("---Max")
            x = functools.reduce(torch.max, all_hlf)
            low_level_feat = functools.reduce(torch.max, all_llf)
        elif self.norm == '3dmask':
            # print("---3dMask")
            sum_e_llf = None
            sum_e_hlf = None
            llf_masks = []
            hlf_masks = []
            for i in range(len(all_hlf)):
                e_llf = torch.exp(all_llf[i])
                e_hlf = torch.exp(all_hlf[i])
                hlf_masks.append(e_hlf)
                llf_masks.append(e_llf)
                sum_e_hlf = e_hlf if sum_e_hlf is None else sum_e_hlf+e_hlf
                sum_e_llf = e_llf if sum_e_llf is None else sum_e_llf+e_llf
            
            x = None
            low_level_feat = None
            for i in range(len(all_hlf)):
                lf = all_llf[i] * (llf_masks[i] / sum_e_llf)
                hf = all_hlf[i] * (hlf_masks[i] / sum_e_hlf)
                x = hf if x is None else x+hf
                low_level_feat = lf if low_level_feat is None else low_level_feat+lf
        else:
            x = self.hlf_norm(x)
            low_level_feat = self.llf_norm(low_level_feat)

        # print("Is Leaf h:", self.rgb_hlf.mask.h.is_leaf)
        # print("Grad se fc:", self.rgb_hlf.se.fc[0].weight.grad)
        # print("Weigh se fc:", self.rgb_hlf.se.fc[0].weight)
        # print("Perameter h:", self.rgb_hlf.mask.h)
        # print("Grad:", self.rgb_hlf.mask.h.grad)
        # print("Perameter w:", self.rgb_hlf.mask.w)
        # print("Grad:", self.rgb_hlf.mask.w.grad)
        # print("Perameter c:", self.rgb_hlf.mask.c)
        # print("Grad:", self.rgb_hlf.mask.c.grad)
        # print(self.nir_hlf.mask.h)
        # print(self.pol_hlf.mask.h)
        
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=rgb.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = self.backbones
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    # print("Module M: ", m)
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], ChannelAndSpatialAttentionBlock):
                        # print("M Comes Inside", m)
                        for p in m[1].parameters():
                            if p.requires_grad:
                                # print("Yeilding M: ", p)
                                yield p

    def get_10x_lr_params(self):
        modules = self.decoders
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], ChannelAndSpatialAttentionBlock):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p


class MMSNetForPatchedRGBD(nn.Module):
    def __init__(self, 
        backbone='resnet', 
        output_stride=16, 
        num_classes=40,      
        sync_bn=True, 
        freeze_bn=False,
        use_rgb=False,
        use_depth=False,
        norm='avg',
    ): 
        super(MMSNetForPatchedRGBD, self).__init__()
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        self.freeze_bn = freeze_bn

        self.backbones = []
        self.decoders = []
        self.num_modalities = 0
        self.norm = norm

        self.low_level_feature_channels = 256
        self.low_level_feature_shape = (256, 120, 160)

        self.high_level_feature_channels = 256
        self.high_level_feature_shape = (256, 30, 40)

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        if self.use_rgb:
            self.num_modalities += 1
            self.rgb_backbone = build_backbone(backbone, output_stride, BatchNorm)
            self.rgb_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.rgb_llf = ChannelAndSpatialAttentionBlock(self.low_level_feature_channels)
            self.rgb_hlf = ChannelAndSpatialAttentionBlock(self.high_level_feature_channels)
            self.backbones.extend([self.rgb_backbone, self.rgb_llf, self.rgb_hlf])
            self.decoders.append(self.rgb_aspp)

        if self.use_depth:
            self.num_modalities += 1
            self.depth_backbone = build_backbone(backbone, output_stride, BatchNorm, input_dim=3, pretrained=True)
            self.depth_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.depth_llf = ChannelAndSpatialAttentionBlock(self.low_level_feature_channels)
            self.depth_hlf = ChannelAndSpatialAttentionBlock(self.high_level_feature_channels)
            self.backbones.extend([self.depth_backbone, self.depth_llf, self.depth_hlf])
            self.decoders.append(self.depth_aspp)

        if self.norm == 'bnr':
            self.hlf_norm = nn.Sequential(
                BatchNorm(self.high_level_feature_channels),
                nn.ReLU(inplace=True)
            )
            self.llf_norm = nn.Sequential(
                BatchNorm(self.low_level_feature_channels),
                nn.ReLU(inplace=True)
            )
            self.backbones.extend([self.hlf_norm, self.llf_norm])
        elif self.norm == 'bn':
            self.hlf_norm = BatchNorm(self.high_level_feature_channels)
            self.llf_norm = BatchNorm(self.low_level_feature_channels)
            self.backbones.extend([self.hlf_norm, self.llf_norm])

        self.decoder = build_decoder(num_classes, backbone, BatchNorm, num_modalities=self.num_modalities)
        self.decoders.append(self.decoder)

    def forward(self, rgb=None, depth=None):
        active_modalities = 0
        if self.use_rgb and rgb is not None:
            x1, low_level_feat1 = self.rgb_backbone(rgb)
            # print(x1.shape, low_level_feat1.shape)
            x1 = self.rgb_aspp(x1)
            x1 = self.rgb_hlf(x1)
            low_level_feat1 = self.rgb_llf(low_level_feat1)
            # print("---------", x1.shape, low_level_feat1.shape)
            # x1 = torch.Size([8, 256, 32, 32]),  low_level_feat1 = torch.Size([8, 256, 128, 128])
            # -------- torch.Size([4, 256, 30, 40]) torch.Size([4, 256, 120, 160])
            x = x1
            low_level_feat = low_level_feat1
            active_modalities += 1

        if self.use_depth and depth is not None:
            x2, low_level_feat2 = self.depth_backbone(depth)
            x2 = self.depth_aspp(x2)
            x2 = self.depth_hlf(x2)
            low_level_feat2 = self.depth_llf(low_level_feat2)
            if self.use_rgb and rgb is not None:
                x = torch.add(x, x2)
                low_level_feat = torch.add(low_level_feat, low_level_feat2)
            else:
                x = x2
                low_level_feat = low_level_feat2
            active_modalities += 1 

        # print("X1 and X shape:", x1.shape, x.shape)
        # print("low_level_feat1 and low_level_feat shape:", low_level_feat1.shape, low_level_feat.shape)

        if self.norm == 'avg':
            x = torch.div(x, active_modalities)
            low_level_feat = torch.div(low_level_feat, active_modalities)
        else:
            x = self.hlf_norm(x)
            low_level_feat = self.llf_norm(low_level_feat)
        
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=rgb.size()[2:] if self.use_rgb else depth.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = self.backbones
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], ChannelAndSpatialAttentionBlock):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = self.decoders
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], ChannelAndSpatialAttentionBlock):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

######################### Contrastive Larning ###############################
class DecoderForCL(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, num_modalities=1, is_teacher=False):
        super(DecoderForCL, self).__init__()
        # print("-- Backbone", backbone)
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
            last_conv_input = 48 + 256
        elif backbone == 'resnet18' or backbone == 'resnet34' or backbone == 'resnet50' or backbone == 'resnet101':
            low_level_inplanes = 256
            last_conv_input = 48 + 256
        else:
            raise NotImplementedError

        self.is_teacher = is_teacher

        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(last_conv_input, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))

        # Projection head for contrastive learning
        if not self.is_teacher:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.projhead = nn.Linear(last_conv_input, 128)

        self._init_weight()


    def forward(self, x, low_level_feat, x_ref=None):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        fused_feature = x.clone()
        
        # Feature Projection
        feat1 = None
        feat2 = None
        if (not self.is_teacher) and (x_ref is not None):
            feat1 = self.avgpool(x)
            feat1 = torch.flatten(feat1, 1)
            feat1 = F.normalize(self.projhead(feat1), dim=1)

            feat2 = self.avgpool(x_ref)
            feat2 = torch.flatten(feat2, 1)
            feat2 = F.normalize(self.projhead(feat2), dim=1)
            # print(">--> Projection shape:", feat1.shape)

        x = self.last_conv(x)

        return x, fused_feature, feat1, feat2

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder_for_CL(num_classes, backbone, BatchNorm, num_modalities=1, is_teacher=False):
    return DecoderForCL(num_classes, backbone, BatchNorm, num_modalities, is_teacher)


class MMSNetForPatchedRGBDForCL(nn.Module):
    def __init__(self, 
        backbone='resnet', 
        output_stride=16, 
        num_classes=40,      
        sync_bn=True, 
        freeze_bn=False,
        use_rgb=False,
        use_depth=False,
        norm='avg',
        is_teacher='False',
    ): 
        super(MMSNetForPatchedRGBDForCL, self).__init__()
        self.use_rgb = use_rgb
        self.use_depth = use_depth
        self.freeze_bn = freeze_bn

        self.backbones = []
        self.decoders = []
        self.num_modalities = 0
        self.norm = norm
        self.is_teacher = is_teacher

        self.low_level_feature_channels = 256
        self.low_level_feature_shape = (256, 120, 160)

        self.high_level_feature_channels = 256
        self.high_level_feature_shape = (256, 30, 40)

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        if self.use_rgb:
            self.num_modalities += 1
            self.rgb_backbone = build_backbone(backbone, output_stride, BatchNorm)
            self.rgb_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.rgb_llf = ChannelAndSpatialAttentionBlock(self.low_level_feature_channels)
            self.rgb_hlf = ChannelAndSpatialAttentionBlock(self.high_level_feature_channels)
            self.backbones.extend([self.rgb_backbone, self.rgb_llf, self.rgb_hlf])
            self.decoders.append(self.rgb_aspp)

        if self.use_depth:
            self.num_modalities += 1
            self.depth_backbone = build_backbone(backbone, output_stride, BatchNorm, input_dim=3, pretrained=True)
            self.depth_aspp = build_aspp(backbone, output_stride, BatchNorm)
            self.depth_llf = ChannelAndSpatialAttentionBlock(self.low_level_feature_channels)
            self.depth_hlf = ChannelAndSpatialAttentionBlock(self.high_level_feature_channels)
            self.backbones.extend([self.depth_backbone, self.depth_llf, self.depth_hlf])
            self.decoders.append(self.depth_aspp)

        if self.norm == 'bnr':
            self.hlf_norm = nn.Sequential(
                BatchNorm(self.high_level_feature_channels),
                nn.ReLU(inplace=True)
            )
            self.llf_norm = nn.Sequential(
                BatchNorm(self.low_level_feature_channels),
                nn.ReLU(inplace=True)
            )
            self.backbones.extend([self.hlf_norm, self.llf_norm])
        elif self.norm == 'bn':
            self.hlf_norm = BatchNorm(self.high_level_feature_channels)
            self.llf_norm = BatchNorm(self.low_level_feature_channels)
            self.backbones.extend([self.hlf_norm, self.llf_norm])

        self.decoder = build_decoder_for_CL(num_classes, backbone, BatchNorm, num_modalities=self.num_modalities, is_teacher=is_teacher)
        self.decoders.append(self.decoder)

    def forward(self, rgb=None, depth=None, x_ref=None):
        active_modalities = 0
        if self.use_rgb and rgb is not None:
            x1, low_level_feat1 = self.rgb_backbone(rgb)
            # print(x1.shape, low_level_feat1.shape)
            x1 = self.rgb_aspp(x1)
            x1 = self.rgb_hlf(x1)
            low_level_feat1 = self.rgb_llf(low_level_feat1)
            # print("---------", x1.shape, low_level_feat1.shape)
            # x1 = torch.Size([8, 256, 32, 32]),  low_level_feat1 = torch.Size([8, 256, 128, 128])
            # -------- torch.Size([4, 256, 30, 40]) torch.Size([4, 256, 120, 160])
            x = x1
            low_level_feat = low_level_feat1
            active_modalities += 1

        if self.use_depth and depth is not None:
            x2, low_level_feat2 = self.depth_backbone(depth)
            x2 = self.depth_aspp(x2)
            x2 = self.depth_hlf(x2)
            low_level_feat2 = self.depth_llf(low_level_feat2)
            if self.use_rgb and rgb is not None:
                x = torch.add(x, x2)
                low_level_feat = torch.add(low_level_feat, low_level_feat2)
            else:
                x = x2
                low_level_feat = low_level_feat2
            active_modalities += 1 

        # print("X1 and X shape:", x1.shape, x.shape)
        # print("low_level_feat1 and low_level_feat shape:", low_level_feat1.shape, low_level_feat.shape)

        if self.norm == 'avg':
            x = torch.div(x, active_modalities)
            low_level_feat = torch.div(low_level_feat, active_modalities)
        else:
            x = self.hlf_norm(x)
            low_level_feat = self.llf_norm(low_level_feat)
        
        x, fused_feature, proj1, proj2 = self.decoder(x, low_level_feat, x_ref)
        x = F.interpolate(x, size=rgb.size()[2:] if self.use_rgb else depth.size()[2:], mode='bilinear', align_corners=True)

        return x, fused_feature, proj1, proj2

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = self.backbones
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], ChannelAndSpatialAttentionBlock) or isinstance(m[1], nn.Linear):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = self.decoders
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d) or isinstance(m[1], ChannelAndSpatialAttentionBlock) or isinstance(m[1], nn.Linear):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p