# Demo Code for Paper:
# [Title]  - "Understanding Physiological and Behavioral Characteristics Separately for High-Performance Dynamic Hand Gesture Authentication"
# [Author] - Wenwei Song, Wenxiong Kang, Yufeng Zhang
# [Github] - https://github.com/SCUT-BIP-Lab/PB-Net.git

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from math import ceil


class Model_PBNet(torch.nn.Module):
    def __init__(self, frame_length, feature_dim, out_dim, sample_rate, clip_size):
        super(Model_PBNet, self).__init__()
        # there are 64 frames in each dynamic hand gesture video
        self.frame_length = frame_length
        self.out_dim = out_dim  # the feature dim of the two branches
        self.sample_rate = sample_rate  # sample rate for P-Branch in data tailoring
        self.clip_size = clip_size  # clip size for B-Branch in data tailoring
        self.frame_p = ceil(frame_length / sample_rate)  # the frame number of the derived video for P-Branch
        self.frame_b = ceil(frame_length / clip_size)  # the frame number of the derived video for B-Branch

        # load the pretrained ResNet18 for the two branch
        self.P_Branch = torchvision.models.resnet18(pretrained=True)
        self.B_Branch = torchvision.models.resnet18(pretrained=True)
        # change the last fc with the shape of 512×512
        self.P_Branch.fc = nn.Linear(in_features=feature_dim, out_features=out_dim)
        self.B_Branch.fc = nn.Linear(in_features=feature_dim, out_features=out_dim)
        # construct the TempConv for the Conv1 and the Block1 of Layer1 in the B_Branch
        self.temp_conv1 = nn.Conv3d(64, 64, kernel_size=(4, 1, 1), stride=1, padding=(0,0,0), bias=True) # for Conv1
        self.temp_layer1_0 = nn.Conv3d(64, 64, kernel_size=(4, 1, 1), stride=1, padding=(0,0,0), bias=True) # for Block1 of Layer1
        # initialize the weights and biases of the TempConv to Zero
        nn.init.constant_(self.temp_conv1.weight, 0)
        nn.init.constant_(self.temp_conv1.bias, 0)
        nn.init.constant_(self.temp_layer1_0.weight, 0)
        nn.init.constant_(self.temp_layer1_0.bias, 0)
        # importance weight generator
        self.weight_conv = nn.Linear(in_features=out_dim*2, out_features=2)
        # initialize the weights and biases of the generator to Zero
        nn.init.constant_(self.weight_conv.weight, 0)
        nn.init.constant_(self.weight_conv.bias, 0)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    # calculate the TSClips (TS-Moudle)
    def dataTailoring(self, v):
        v = v.view((-1, self.frame_length)+v.shape[-3:]) # batch, frame, c, h, w
        v = v.permute(0, 2, 1, 3, 4) # batch, c, frame, h, w
        # frame number to be padded
        pad_p = self.frame_p * self.sample_rate - self.frame_length
        pad_b = self.frame_b * self.clip_size - self.frame_length

        v_p = F.pad(v, pad=(0, 0, 0, 0, 0, pad_p), mode='replicate').permute(0, 2, 1, 3, 4).contiguous()  # batch, 66, c, h, w
        v_b = F.pad(v, pad=(0, 0, 0, 0, 0, pad_b), mode='replicate').permute(0, 2, 1, 3, 4).contiguous()  # batch, 66, c, h, w
        v_p = v_p.view((-1, self.sample_rate) + v_p.shape[-3:])  # batch*22, 3, c, h, w
        v_b = v_b.view((-1, self.clip_size) + v_b.shape[-3:])  # batch*22, 3, c, h, w

        v_p = v_p[:, 1]  # take the middle frame from every 3 frames; batch*22, c, h, w
        v_b = torch.sum(v_b, 2)  # graying by summation; batch*22, 3, h, w

        return v_p, v_b

    # TempConv add-on module
    def temp_conv_func(self, x, conv, temp_pad=(0,0)):
        x = x.reshape(-1, self.frame_b, *x.shape[-3:])
        x = x.permute(0, 2, 1, 3, 4)
        x = conv(F.pad(x, (0,0,0,0)+temp_pad, mode='constant', value=0.0)) + x  # residual connection
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(-1, *x.shape[-3:])
        return x

    def forward(self, data, label=None):
        # get tailored videos for the two branches
        v_p, v_b = self.dataTailoring(data)
        # get physiological features
        x_p = self.P_Branch(v_p)  # batch*22, 512
        x_p = x_p.view(-1, self.frame_p, self.out_dim)  # batch, 22, 512
        x_p = torch.mean(x_p, dim=1, keepdim=False) # batch, 512
        x_p_norm = torch.div(x_p, torch.norm(x_p, p=2, dim=1, keepdim=True).clamp(min=1e-12))  # normalization

        # get behavioral features
        x_b = self.B_Branch.conv1(v_b)
        x_b = self.temp_conv_func(x_b, conv=self.temp_conv1, temp_pad=(2, 1)) # the first TempConv
        x_b = self.B_Branch.bn1(x_b)
        x_b = self.B_Branch.relu(x_b)
        x_b = self.B_Branch.maxpool(x_b)
        x_b = self.B_Branch.layer1[0](x_b)  # block1 of layer1
        x_b = self.temp_conv_func(x_b, conv=self.temp_layer1_0, temp_pad=(2, 1)) # the second TempConv
        x_b = self.B_Branch.layer1[1](x_b)  # block2 of layer1
        for i in range(1, 4):
            layer_name = "layer"+str(i+1)
            layer = getattr(self.B_Branch, layer_name)
            x_b = layer(x_b)
        x_b = self.avgpool(x_b)
        x_b = torch.flatten(x_b, 1)
        x_b = self.B_Branch.fc(x_b)
        x_b = x_b.view(-1, self.frame_b, self.out_dim)  # batch, 22, 512
        x_b = torch.mean(x_b, dim=1, keepdim=False)  # batch, 512
        x_b_norm = torch.div(x_b, torch.norm(x_b, p=2, dim=1, keepdim=True).clamp(min=1e-12))  # normalization

        # generate importance weights
        x_weight = torch.cat((x_p, x_b), dim=-1).detach()  # block the gradients
        weight = F.softmax(self.weight_conv(x_weight), dim=-1)
        weight_sqrt = weight.sqrt()

        x_b_norm_d, x_p_norm_d = x_b_norm.detach(), x_p_norm.detach()
        x_b_norm_cat = x_b_norm_d * weight_sqrt[:, :1]
        x_p_norm_cat = x_p_norm_d * weight_sqrt[:, 1:]
        id_feature = torch.cat((x_b_norm_cat, x_p_norm_cat), dim=1)

        return id_feature, x_b_norm, x_p_norm, weight


