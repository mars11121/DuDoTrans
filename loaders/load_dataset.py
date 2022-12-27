import os
import odl
import torch
import pydicom
import random
import numpy as np
from PIL import Image
from glob import glob
from scipy import ndimage
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
from odl.contrib import torch as odl_torch

class CTSlice_Provider(Dataset):
  def __init__(self, base_path, poission_level=5e6, gaussian_level=0.05, num_view=96):
    self.base_path=base_path
    self.slices_path=glob(os.path.join(self.base_path,'*/*.dcm'))
    self.radon_full, self.iradon_full, self.fbp_full, self.op_norm_full=self._radon_transform(num_view=360)
    self.radon_curr, self.iradon_curr, self.fbp_curr, self.op_norm_curr=self._radon_transform(num_view=num_view)
    self.poission_level=poission_level
    self.gaussian_level=gaussian_level
    self.num_view=num_view
    
  def _radon_transform(self, num_view=96, start_ang=0, end_ang=2*np.pi, num_detectors=800):
    # the function is used to generate fp, bp, fbp functions
    # the physical parameters is set as MetaInvNet and EPNet
    xx=200
    space=odl.uniform_discr([-xx, -xx], [xx, xx], [512,512], dtype='float32')
    angles=np.array(num_view).astype(int)
    angle_partition=odl.uniform_partition(start_ang, end_ang, angles)
    detector_partition=odl.uniform_partition(-480, 480, num_detectors)
    geometry=odl.tomo.FanBeamGeometry(angle_partition, detector_partition, src_radius=600, det_radius=290)
    operator=odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

    op_norm=odl.operator.power_method_opnorm(operator)
    op_norm=torch.from_numpy(np.array(op_norm*2*np.pi)).double().cuda()

    op_layer=odl_torch.operator.OperatorModule(operator)
    op_layer_adjoint=odl_torch.operator.OperatorModule(operator.adjoint)
    fbp=odl.tomo.fbp_op(operator, filter_type='Ram-Lak', frequency_scaling=0.9)*np.sqrt(2)
    op_layer_fbp=odl_torch.operator.OperatorModule(fbp)

    return op_layer, op_layer_adjoint, op_layer_fbp, op_norm

  def ril(self, num_view=96, start_ang=0, end_ang=2*np.pi, num_detectors=800):
  # def ril(self, num_view=96, start_ang=-5/12*np.pi, end_ang=5/12*np.pi, num_detectors=800):
    xx=200
    space=odl.uniform_discr([-xx, -xx], [xx, xx], [512,512], dtype='float32')
    angles=np.array(num_view).astype(int)
    angle_partition=odl.uniform_partition(start_ang, end_ang, angles)
    detector_partition=odl.uniform_partition(-480, 480, num_detectors)
    geometry=odl.tomo.FanBeamGeometry(angle_partition, detector_partition, src_radius=600, det_radius=290)
    operator=odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

    op_norm=odl.operator.power_method_opnorm(operator)
    op_norm=torch.from_numpy(np.array(op_norm*2*np.pi)).double().cuda()

    op_layer=odl_torch.operator.OperatorModule(operator)
    op_layer_adjoint=odl_torch.operator.OperatorModule(operator.adjoint)
    fbp=odl.tomo.fbp_op(operator, filter_type='Ram-Lak', frequency_scaling=0.9)*np.sqrt(2)
    op_layer_fbp=odl_torch.operator.OperatorModule(fbp)

    return op_layer, op_layer_adjoint, op_layer_fbp, op_norm

  def __getitem__(self, index):
    slice_path=self.slices_path[index]
    dcm=pydicom.read_file(slice_path)
    dcm.image=dcm.pixel_array*dcm.RescaleSlope+dcm.RescaleIntercept
    data_slice=dcm.image
    data_slice=np.array(data_slice).astype(float)
    data_slice=(data_slice-np.min(data_slice))/(np.max(data_slice)-np.min(data_slice))

    # the following code is used to generate projections with noise in the way of odl package
    phantom=torch.from_numpy(data_slice).unsqueeze(0).type(torch.FloatTensor)
    sino=self.radon_curr(phantom)

    # the following part code is used to randomly choose sinograms to satisfy the sparse-view requeirement

    # add poission noise
    intensityI0=self.poission_level
    scale_value=torch.from_numpy(np.array(intensityI0).astype(np.float))
    normalized_sino=torch.exp(-sino/sino.max())
    th_data=np.random.poisson(scale_value*normalized_sino)
    sino_noisy=-torch.log(torch.from_numpy(th_data)/scale_value)
    sino_noisy = sino_noisy*sino.max()

    # add Gaussian noise
    noise_std=self.gaussian_level
    noise_std=np.array(noise_std).astype(np.float)
    nx,ny=np.array(self.num_view).astype(np.int),np.array(800).astype(np.int)
    noise = noise_std*np.random.randn(nx,ny)
    noise = torch.from_numpy(noise)
    sino_noisy = sino_noisy + noise

    fbp_u=self.fbp_curr(sino_noisy)
    phantom=phantom#.type(torch.DoubleTensor)
    fbp_u=fbp_u#.type(torch.DoubleTensor)
    sino_noisy=sino_noisy#.type(torch.DoubleTensor)

    return phantom, fbp_u, sino_noisy

  def __len__(self):
    return len(self.slices_path)
    
if __name__=='__main__':
  print('Reading CT slices Beginning')
  aapm_dataset=CTSlice_Provider('/home1/cewang/Data/NIH-AAPM-DuDo/train')
  aapm_dataloader=DataLoader(dataset=aapm_dataset, batch_size=2, shuffle=True)
  for index, (gt, fbpu, projs_noisy) in enumerate(aapm_dataloader):
    if index==1:
      img_save=sitk.GetImageFromArray(fbpu)
      print(gt.shape)
      print(fbpu.shape)
      print(projs_noisy.shape)
