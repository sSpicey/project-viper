import torch
import numpy as np
import random
import math
from torch import nn
from util.unet import UNet


class UNetWrapper(nn.Module):
    """
    A wrapper for the U-Net model that includes input normalization and a final activation layer.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.input_batchnorm = nn.BatchNorm2d(kwargs['in_channels'])
        self.unet = UNet(**kwargs)
        self.final = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        """
        Initializes the weights of the convolutional and linear layers using Kaiming normalization.
        """
        init_set = {nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Linear}
        for m in self.modules():
            if type(m) in init_set:
                self._init_layer_weights(m)

    @staticmethod
    def _init_layer_weights(layer):
        """
        Initializes the weights of a single layer using Kaiming normalization.
        """
        nn.init.kaiming_normal_(layer.weight.data, mode='fan_out', nonlinearity='relu')
        if layer.bias is not None:
            _, fan_out = nn.init._calculate_fan_in_and_fan_out(layer.weight.data)
            bound = 1 / math.sqrt(fan_out)
            nn.init.normal_(layer.bias, -bound, bound)

    def forward(self, input_batch):
        """
        Forward pass through the UNetWrapper model.
        """
        bn_output = self.input_batchnorm(input_batch)
        un_output = self.unet(bn_output)
        fn_output = self.final(un_output)
        return fn_output


def augment3d(inp):
    """
    Applies random 3D augmentations to the input tensor.
    """
    transform_t = create_identity_matrix()
    transform_t = apply_random_flipping_and_translation(transform_t)
    transform_t = apply_random_rotation(transform_t)
    return apply_affine_transformation(inp, transform_t)


def create_identity_matrix():
    """
    Creates a 4x4 identity matrix.
    """
    return torch.eye(4, dtype=torch.float32)


def apply_random_flipping_and_translation(transform_t):
    """
    Applies random flipping and translation to the transformation matrix.
    """
    for i in range(3):
        if random.random() > 0.5:
            transform_t[i, i] *= -1
        offset_float = 0.1
        random_float = (random.random() * 2 - 1)
        transform_t[3, i] = offset_float * random_float
    return transform_t


def apply_random_rotation(transform_t):
    """
    Applies a random rotation around the z-axis to the transformation matrix.
    """
    angle_rad = random.random() * np.pi * 2
    s = np.sin(angle_rad)
    c = np.cos(angle_rad)
    rotation_t = torch.tensor([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=torch.float32)
    transform_t @= rotation_t
    return transform_t


def apply_affine_transformation(inp, transform_t):
    """
    Applies an affine transformation to the input tensor based on the transformation matrix.
    """
    affine_t = torch.nn.functional.affine_grid(
        transform_t[:3].unsqueeze(0).expand(inp.size(0), -1, -1).cuda(),
        inp.shape,
        align_corners=False,
    )
    augmented_chunk = torch.nn.functional.grid_sample(
        inp,
        affine_t,
        padding_mode='border',
        align_corners=False,
    )
    return augmented_chunk


class LunaModel(nn.Module):
    """
    A model for lung nodule analysis consisting of multiple convolutional blocks and a linear classifier.
    """

    def __init__(self, in_channels=1, conv_channels=8):
        super().__init__()
        self.tail_batchnorm = nn.BatchNorm3d(1)
        self.block1 = LunaBlock(in_channels, conv_channels)
        self.block2 = LunaBlock(conv_channels, conv_channels * 2)
        self.block3 = LunaBlock(conv_channels * 2, conv_channels * 4)
        self.block4 = LunaBlock(conv_channels * 4, conv_channels * 8)
        self.head_linear = nn.Linear(1152, 2)
        self.head_activation = nn.Softmax(dim=1)
        self._init_weights()

    def _init_weights(self):
        """
        Initializes the weights of the convolutional and linear layers using Kaiming normalization.
        """
        for m in self.modules():
            if type(m) in {nn.Linear, nn.Conv3d, nn.Conv2d, nn.ConvTranspose2d, nn.ConvTranspose3d}:
                self._init_layer_weights(m)

    @staticmethod
    def _init_layer_weights(layer):
        """
        Initializes the weights of a single layer using Kaiming normalization.
        """
        nn.init.kaiming_normal_(layer.weight.data, a=0, mode='fan_out', nonlinearity='relu')
        if layer.bias is not None:
            _, fan_out = nn.init._calculate_fan_in_and_fan_out(layer.weight.data)
            bound = 1 / math.sqrt(fan_out)
            nn.init.normal_(layer.bias, -bound, bound)

    def forward(self, input_batch):
        """
        Forward pass through the LunaModel.
        """
        bn_output = self.tail_batchnorm(input_batch)
        block_out = self.process_blocks(bn_output)
        conv_flat = block_out.view(block_out.size(0), -1)
        linear_output = self.head_linear(conv_flat)
        return linear_output, self.head_activation(linear_output)

    def process_blocks(self, bn_output):
        """
        Processes the input through the convolutional blocks of the LunaModel.
        """
        block_out = self.block1(bn_output)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
        return self.block4(block_out)


class LunaBlock(nn.Module):
    """
    A block used in the LunaModel, consisting of two convolutional layers followed by a max-pooling layer.
    """

    def __init__(self, in_channels, conv_channels):
        super(LunaBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, conv_channels, kernel_size=3, padding=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(conv_channels, conv_channels, kernel_size=3, padding=1, bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(2, 2)

    def forward(self, input_batch):
        """
        Forward pass through the LunaBlock.
        """
        block_out = self.conv1(input_batch)
        block_out = self.relu1(block_out)
        block_out = self.conv2(block_out)
        block_out = self.relu2(block_out)
        return self.maxpool(block_out)
