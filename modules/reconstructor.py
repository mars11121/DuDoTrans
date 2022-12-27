import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .unet_model import UNet_Slim_Fat
from network_swinir import SwinIR

class reconstructor(nn.Module):
  # the version of reconstructor is set as FBP-DuDo-SwinIR to test the effectiveness of dual domain SwinIR in Recon
  def __init__(self, dset, img_h=512, img_w=512):
    super(reconstructor, self).__init__()
    self.Image_recon_module=SwinIR(upscale=1, img_size=(img_h, img_w), in_chans=2,
                   window_size=8, img_range=1., depths=[2, 2, 2, 2],
                   embed_dim=60, num_heads=[2, 2, 2, 2], mlp_ratio=2, upsampler='')
    self.Sinogram_recon_module=SwinIR(upscale=1, img_size=(96, 800),
                   window_size=8, img_range=1., depths=[1, 1, 1],
                   embed_dim=60, num_heads=[2, 2, 2], mlp_ratio=2, upsampler='')
    self.dset=dset
    self.fp_senet_gt, _, self.ril_odl, __ = self.dset.ril()
  def forward(self, img, img_gt, sinos):
    sinos_gt=self.radon_senet_gt(img_gt)
    sinos_enhanced=self.Sinogram_recon_module(sinos)
    img_ril=self.ril(sinos_enhanced)
    img_input=torch.cat((img, img_ril), 1)
    reconstructed_img=self.Image_recon_module(img_input)
    return sinos_gt, sinos_enhanced, img_ril, reconstructed_img

  def ril(self, img):
    # return ril results of enhanced sinograms
    if len(img.shape) == 4:
      img = img.squeeze(1)
    return self.ril_odl(img).unsqueeze(1)

  def radon_senet_gt(self, img):
    # return supervision of Sinogram_recon_module
    if len(img.shape)==4:
      img=img.squeeze(1)
    return self.fp_senet_gt(img).unsqueeze(1)

class reconstructor_loss(nn.Module):
  def __init__(self):
    super(reconstructor_loss, self).__init__()

  def forward(self, pred, gt):
    left_loss=F.mse_loss(pred, gt, reduce=False)
    return torch.mean(left_loss)


if __name__=='__main__':
  a=torch.ones([1,1,512,512])
  recon_func=reconstructor()
  b,c=recon_func(a)
  print(b.shape, c.shape)
