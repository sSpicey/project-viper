import math
import random
from torch import nn as nn
import torch.nn.functional as F
from util.logconf import logging
import torch
from util.unet import UNet
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class WrapperUNet(nn.Module):
    def __init__(self, **kwargs):
        """
        Initializes a WrapperUNet with input batch normalization, a U-Net, and a final sigmoid activation layer.
        """
        super().__init__()
        self.unet = UNet(**kwargs)
        self.input_batchnorm = nn.BatchNorm2d(kwargs['in_channels'])
        self.final = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        """
        Initializes the weights of the convolutional and linear layers using Kaiming normalization.
        """
        init_set = {
            nn.Conv2d,
            nn.Conv3d,
            nn.ConvTranspose2d,
            nn.ConvTranspose3d,
            nn.Linear,
        }
        for m in self.modules():
            if type(m) in init_set:
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu', a=0)
                if m.bias is not None:
                    _, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight.data)
                    bound = 1 / math.sqrt(fan_out)
                    nn.init.normal_(m.bias, -bound, bound)

    def forward(self, input_batch):
        """
        Forward pass through the WrapperUNet model.

        Args:
            input_batch: The input batch of images.

        Returns:
            The output of the model after applying the U-Net, batch normalization, and sigmoid activation.
        """
        bn_output = self.input_batchnorm(input_batch)
        un_output = self.unet(bn_output)
        fn_output = self.final(un_output)
        return fn_output


class SegmentationAugmentation(nn.Module):
    def __init__(self, flip=None, offset=None, scale=None, rotate=None, noise=None):
        """
        Initializes a SegmentationAugmentation module with various augmentation options.

        Args:
            flip: Whether to randomly flip the input.
            offset: The maximum offset for random translation.
            scale: The maximum scale factor for random scaling.
            rotate: Whether to randomly rotate the input.
            noise: The standard deviation of random noise to add.
        """
        super().__init__()

        self.flip = flip
        self.offset = offset
        self.scale = scale
        self.rotate = rotate
        self.noise = noise

    def forward(self, input_g, label_g):
        """
        Applies augmentation to the input and label tensors.

        Args:
            input_g: The input tensor.
            label_g: The label tensor.

        Returns:
            A tuple containing the augmented input tensor and label tensor.
        """
        transform_t = self._build_2d_transform_matrix()
        transform_t = transform_t.expand(input_g.shape[0], -1, -1)
        transform_t = transform_t.to(input_g.device, torch.float32)
        affine_t = F.affine_grid(transform_t[:,:2], input_g.size(), align_corners=False)

        augmented_input_g = F.grid_sample(input_g, affine_t, padding_mode='border', align_corners=False)
        augmented_label_g = F.grid_sample(label_g.to(torch.float32), affine_t, padding_mode='border', align_corners=False)

        if self.noise:
            noise_t = torch.randn_like(augmented_input_g)
            noise_t *= self.noise
            augmented_input_g += noise_t

        return augmented_input_g, augmented_label_g > 0.5

    def _build_2d_transform_matrix(self):
        """
        Builds a 2D transformation matrix for augmentation.

        Returns:
            The transformation matrix.
        """
        transform_t = torch.eye(3)

        for i in range(2):
            if self.flip:
                if random.random() > 0.5:
                    transform_t[i, i] *= -1

            if self.offset:
                offset_float = self.offset
                random_float = (random.random() * 2 - 1)
                transform_t[2, i] = offset_float * random_float

            if self.scale:
                scale_float = self.scale
                random_float = (random.random() * 2 - 1)
                transform_t[i, i] *= 1.0 + scale_float * random_float

        if self.rotate:
            angle_rad = random.random() * math.pi * 2
            s = math.sin(angle_rad)
            c = math.cos(angle_rad)

            rotation_t = torch.tensor([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]])

            transform_t @= rotation_t

        return transform_t