## author: xin luo
## creat: 2022.4.3, modify: 2023.2.3
## des: deeplabv3plus model (with Xception65 backbone).


import torch
import torch.nn as nn
import torch.nn.functional as F


## utils
def conv1x1_bn_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 1, 1, 0),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def conv3x3_bn_relu(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, 1, 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

def deconv4x4_bn_relu(in_channels=256, out_channels=256):
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, \
                                    kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class SeparableConv2d(nn.Module):
    ''' dilated_dw_conv2d -> bn(optional) -> 1x1 conv'''
    def __init__(self, in_channels, out_channels, kernel_size=3, \
                stride=1, dilation=1, bias=False, norm_layer=None):
        super(SeparableConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, \
                        stride, 0, dilation, groups=in_channels, bias=bias)
        self.bn = norm_layer(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

    def forward(self, x):
        x = self.fix_padding(x, self.kernel_size, self.dilation)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x

    def fix_padding(self, x, ksize, dilation):
        ksize_effective = ksize + (ksize - 1) * (dilation - 1)
        pad_total = ksize_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        padded_inputs = F.pad(x, (pad_beg, pad_end, pad_beg, pad_end))
        return padded_inputs

class Block(nn.Module):
    ''' (relu(option) -> SeparableConv2d + bn)xn        
        grow (option for first or last): in_channels -> out_channels
        
        '''
    def __init__(self, in_channels, out_channels, reps, stride=1, dilation=1, norm_layer=None,
                 start_with_relu=True, grow_first=True, is_last=False):
        super(Block, self).__init__()
        if out_channels != in_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)
            self.skipbn = norm_layer(out_channels)
        else:
            self.skip = None
        self.relu = nn.ReLU(True)
        rep = list()
        filters = in_channels
        if grow_first:  #
            if start_with_relu:
                rep.append(self.relu)
            rep.append(SeparableConv2d(in_channels, out_channels, 3, 1, \
                                        dilation, norm_layer=norm_layer))
            rep.append(norm_layer(out_channels))
            filters = out_channels
        for i in range(reps - 1):
            if grow_first or start_with_relu:
                rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, 1, dilation, \
                                        norm_layer=norm_layer))
            rep.append(norm_layer(filters))
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_channels, out_channels, 3, 1, \
                                        dilation, norm_layer=norm_layer))
        if stride != 1:  # specify a new SeparableConv2d in the last
            rep.append(self.relu)
            rep.append(SeparableConv2d(out_channels, out_channels, 3, stride, \
                                        norm_layer=norm_layer))
            rep.append(norm_layer(out_channels))
        elif is_last:  # add a bn layer in the last
            rep.append(self.relu)
            rep.append(SeparableConv2d(out_channels, out_channels, 3, 1, \
                                        dilation, norm_layer=norm_layer))
            rep.append(norm_layer(out_channels))
        self.rep = nn.Sequential(*rep)

    def forward(self, x):
        out = self.rep(x)
        if self.skip is not None:
            skip = self.skipbn(self.skip(x))
        else:
            skip = x
        out = out + skip
        return out

## backbone
class Xception65(nn.Module):
    '''Modified Aligned Xception
    output_stride = 32' means output image size is 1/32'''
    def __init__(self, num_bands=4, num_classes=1000, \
                            output_stride=32, norm_layer=nn.BatchNorm2d):
        super(Xception65, self).__init__()
        ## output size:1/32
        if output_stride == 32:
            entry_block3_stride = 2
            exit_block20_stride = 2
            middle_block_dilation = 1
            exit_block_dilations = (1, 1)
        ## output size:1/16
        elif output_stride == 16:
            entry_block3_stride = 2
            exit_block20_stride = 1
            middle_block_dilation = 1
            exit_block_dilations = (1, 2)
        ## output size:1/8
        elif output_stride == 8:
            entry_block3_stride = 1
            exit_block20_stride = 1
            middle_block_dilation = 2
            exit_block_dilations = (2, 4)
        else:
            raise NotImplementedError
        self.relu = nn.ReLU(True)
        # ---------Entry flow--------- #
        self.conv1 = nn.Conv2d(num_bands, 32, 3, 2, 1, bias=False)
        self.bn1 = norm_layer(32)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn2 = norm_layer(64)
        self.block1 = Block(64, 128, reps=2, stride=2, norm_layer=norm_layer, \
                                start_with_relu=False)
        self.block2 = Block(128, 256, reps=2, stride=2, norm_layer=norm_layer, \
                                start_with_relu=False)
        self.block3 = Block(256, 728, reps=2, stride=entry_block3_stride, \
                                norm_layer=norm_layer, start_with_relu=True, \
                                is_last=True)
        # ---------Middle flow--------- #
        midflow = list()
        for i in range(4, 20):
            midflow.append(Block(728, 728, reps=3, stride=1, dilation=middle_block_dilation, norm_layer=norm_layer,
                                 start_with_relu=True, grow_first=True))
        self.midflow = nn.Sequential(*midflow)
        # ---------Exit flow--------- #
        self.block20 = Block(728, 1024, reps=2, stride=exit_block20_stride, \
                        dilation=exit_block_dilations[0],norm_layer=norm_layer,\
                        start_with_relu=True, grow_first=False, is_last=True)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, \
                        dilation=exit_block_dilations[1], norm_layer=norm_layer)
        self.bn3 = norm_layer(1536)
        self.conv4 = SeparableConv2d(1536, 1536, 3, stride=1, \
                        dilation=exit_block_dilations[1], norm_layer=norm_layer)
        self.bn4 = norm_layer(1536)
        self.conv5 = SeparableConv2d(1536, 2048, 3, 1, \
                        dilation=exit_block_dilations[1], norm_layer=norm_layer)
        self.bn5 = norm_layer(2048)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)
    def forward(self, x):
        # Entry flow
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.relu(x)
        # c1 = x
        x = self.block2(x)
        # c2 = x
        x = self.block3(x)
        # Middle flow
        x = self.midflow(x)
        # c3 = x
        # Exit flow
        x = self.block20(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x



def aspp_conv(in_channels, out_channels, dilation):
    '''aspp convolution: atrous_conv + bn + relu'''
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, \
                            padding=dilation, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
        )

class aspp_pooling(nn.Module):
    '''aspp pooling: pooling + 1x1 conv + bn + relu'''
    def __init__(self, in_channels, out_channels):
        super(aspp_pooling, self).__init__()
        self.layers = nn.Sequential(
                nn.AdaptiveAvgPool2d((1)),     ## size -> 1x1
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
                )
    def forward(self, x):
        size = x.shape[-2:]
        x = self.layers(x)
        x = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return x

class aspp(nn.Module):
    '''aspp module: consist of aspp conv, aspp pooling...'''
    def __init__(self, in_channels, atrous_rates):
        super(aspp, self).__init__()
        self.out_channels = 256   
        modules = []
        # branch 1：conv2d+bn+relu
        modules.append(nn.Sequential(
                    nn.Conv2d(in_channels, self.out_channels, 1, bias=False),
                    nn.BatchNorm2d(self.out_channels),
                    nn.ReLU()))
        rate1, rate2, rate3 = tuple(atrous_rates)  ## 
        ## branch 2-4：atrous convolution
        modules.append(aspp_conv(in_channels, self.out_channels, rate1))
        modules.append(aspp_conv(in_channels, self.out_channels, rate2))
        modules.append(aspp_conv(in_channels, self.out_channels, rate3))
        ## branch 5：pooling
        modules.append(aspp_pooling(in_channels, self.out_channels))
        self.convs = nn.ModuleList(modules) 
        ## for merged branches：conv2d+bn+relu+dropout
        self.project = nn.Sequential(
                    nn.Conv2d(5 * self.out_channels, self.out_channels, 1, bias=False),
                    nn.BatchNorm2d(self.out_channels),
                    nn.ReLU(),
                    nn.Dropout(0.5))
    # def forward(self, x):
    #     res = []
    #     for conv in self.convs:
    #         res.append(conv(x))
    #     res = torch.cat(res, dim=1)
    #     return self.project(res)
    def forward(self, x):
        res = []
        for i, conv in enumerate(self.convs):
            out = conv(x)
            res.append(out)
        
        res = torch.cat(res, dim=1)
        
        res = self.project(res)
        
        return res

class Xception65_feat(nn.Module):
    '''original encoder for deeplabv3_plus moel
      retrun:
        high-level + low-level features

    '''
    def __init__(self, num_bands=4):
        super(Xception65_feat, self).__init__()
        self.backbone = Xception65(num_bands)
    def forward(self,x):
        '''output -> low feature + high feature'''
        ## output_stride = 32
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.conv2(x)
        x = self.backbone.bn2(x)
        x = self.backbone.relu(x)
        x = self.backbone.block1(x)
        x = self.backbone.relu(x)    
        fea_low = x      # size -> 1/4, num_channel -> 128
        x = self.backbone.block2(x)
        x = self.backbone.block3(x)
        # Middle flow
        x = self.backbone.midflow(x)
        # fea_mid = x      # size -> 1/16, num_channel -> 728
        # Exit flow
        x = self.backbone.block20(x)
        x = self.backbone.relu(x)
        x = self.backbone.conv3(x)
        x = self.backbone.bn3(x)
        x = self.backbone.relu(x)
        x = self.backbone.conv4(x)
        x = self.backbone.bn4(x)
        x = self.backbone.relu(x)
        x = self.backbone.conv5(x)
        x = self.backbone.bn5(x)
        x = self.backbone.relu(x) 
        fea_high = x      # size -> 1/32, num_channels -> 2048
        return fea_low, fea_high


## deeplabv3_plus 
class deeplabv3plus(nn.Module):
    '''des: original deeplabv3_plus model'''
    def __init__(self, num_bands=4, num_classes=2, backbone=Xception65_feat):
        super(deeplabv3plus, self).__init__()
        self.name = 'deeplabv3plus'
        self.backbone = backbone(num_bands=num_bands)
        self.aspp = aspp(in_channels=2048, atrous_rates=[12, 24, 36])
        self.low_block = conv1x1_bn_relu(128, 48)
        self.high_low_block = nn.Sequential(      
                            conv3x3_bn_relu(304, 256),
                            nn.Dropout(0.5),
                            conv3x3_bn_relu(256, 256),
                            nn.Dropout(0.1),
                            )
        if num_classes == 2:
            self.outp_layer = nn.Sequential(
                    nn.Conv2d(in_channels=256, out_channels=1, kernel_size=1),
                    )
        else:
            self.outp_layer = nn.Sequential(
                    nn.Conv2d(in_channels=256,out_channels=num_classes,kernel_size=1)
                    )

    def forward(self, x):
        fea_low, fea_high = self.backbone(x) 
        ## -------high-level features------- ##
        x_fea_high = self.aspp(fea_high)    # channels: -> 256
        x_fea_high = F.interpolate(x_fea_high, fea_low.size()[-2:], \
                                    mode='bilinear', align_corners=True)

        ## -------low-level features------- ##
        x_fea_low = self.low_block(fea_low)   # channels: ->48

        ## -------features concat-------  ##
        x_fea_concat = torch.cat([x_fea_high, x_fea_low], dim=1) # 
        x_fea_concat = self.high_low_block(x_fea_concat)
        x_out_logit = self.outp_layer(x_fea_concat)
        x_out_logit = F.interpolate(x_out_logit, x.size()[-2:], \
                                        mode='bilinear', align_corners=True)
        return x_out_logit

if __name__ == "__main__":
    model = deeplabv3plus(num_bands=6, num_classes=2)
    input = torch.randn(4, 6, 512, 512)
    output = model(input)
    print(output.shape)

