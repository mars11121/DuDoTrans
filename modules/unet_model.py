# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F
import torch.nn as nn

from .Unet_parts import *


class UNet_Slim_Fat(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_Slim_Fat, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        # self.output = ConvReLu(512, n_classes)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)

    def forward(self, x_in):
        x1 = self.inc(x_in)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = x_in+self.outc(x)
        return x
