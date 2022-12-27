import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import torch
import shutil
import numpy as np
import SimpleITK as sitk
from torch.utils.data import DataLoader
from modules.reconstructor import reconstructor, reconstructor_loss
from loaders.load_dataset import CTSlice_Provider
from skimage.measure import compare_psnr, compare_ssim
from sklearn import metrics

# In this version, we use FBP-ConvNet to analysize the effectiveness of active sampling
# The Reconstructor is set as ConvNet which preocesses fbp_u

class Trainer(object):
  def __init__(self, learning_rate=0.0001, is_restart=True, max_epoch=100, is_cuda=True, num_view=96):
    # Setting parameters poission_level and gaussian_level, batch_size ...
    self.poission_level=1e5
    self.gaussian_level=0.05
    self.batch_size=1
    self.lr=learning_rate
    self.is_cuda=is_cuda
    self.num_view=num_view
    self.is_restart=is_restart
    self.max_epoch=max_epoch

    # Data Flow Pipiline
    print('Reading CT slices Beginning')
    self.train_dataset=CTSlice_Provider('/home1/cewang/Data/NIH-AAPM-DuDo/train', poission_level=self.poission_level, gaussian_level=self.gaussian_level, num_view=self.num_view)
    self.train_loader=DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=True)

    self.reconstructor_func=reconstructor(self.train_dataset)

    num_param = sum(p.numel() for p in self.reconstructor_func.parameters() if p.requires_grad)
    print('Number of parameters in model: {}'.format(num_param))
    self.reconstructor_loss = reconstructor_loss()
    self.ril_loss = reconstructor_loss()
    self.sinogram_loss = reconstructor_loss()

    self.reconstructor_params=list(self.reconstructor_func.parameters())
    self.reconstructor_optimizer=torch.optim.Adam(self.reconstructor_params, self.lr)
    self.reconstructor_optimizer_20=torch.optim.Adam(self.reconstructor_params, 0.1*self.lr)

    if not os.path.exists('./results/models'):
      os.mkdir('./results/models')
    if not os.path.exists('./results/visualization'):
      os.mkdir('./results/visualization')
    if self.is_restart:
      shutil.rmtree('./results/visualization')
      os.mkdir('./results/visualization')
      self.epoch=0
      self.global_iter=0
      self.best_loss=np.inf
      print('Training process started')
    else:
      try:
        state=torch.load('./results/models/epoch_100iter1745.pth.tar')
        self.epoch=state['epoch']
        self.reconstructor_func.load_state_dict(state['reconstructor_state'])
        self.reconstructor_optimizer.load_state_dict(state['reconstructor_optimizer'])
        print('Saved ckpt is loaded successfully')
      except:
        self.epoch=0
        print('There is no saved ckpt file to load, the training process is restarted')
    print('Settings are finished')

  def train(self):
    for e in range(self.epoch, self.max_epoch):
      for num_iter, (gt, fbp_u, projs_noisy) in enumerate(self.train_loader):
        if self.is_cuda:
          gt=gt.cuda()
          fbp_u=fbp_u.cuda()
          projs_noisy=projs_noisy.float().cuda()
          self.reconstructor_func=self.reconstructor_func.cuda()

        sinos_gt, sinos_enhanced, img_ril, reconstructed_image=self.reconstructor_func(fbp_u, gt, projs_noisy)

        ## Update reconstructor
        # self.reconstructor_func.train()
        loss_recon=self.reconstructor_loss(reconstructed_image, gt)
        loss_ril=self.ril_loss(img_ril, gt)
        loss_sino=self.sinogram_loss(sinos_enhanced, sinos_gt)
        loss_reconstructor=loss_recon+loss_ril+loss_sino
        if e<20:
          self.reconstructor_optimizer.zero_grad()
          loss_reconstructor.backward()
          self.reconstructor_optimizer.step()
        else:
          self.reconstructor_optimizer_20.zero_grad()
          loss_reconstructor.backward()
          self.reconstructor_optimizer_20.step()

        # calculate the psnr/ssim of the current training images, and feedback the average vale
        if num_iter%50==0:
          curr_psnr, curr_ssim=self.calculate_metric(reconstructed_image, gt)
          print('This is the {}-th epoch {}-th iteration, the psnr and ssim is {:.2f} {:.4f}'.format(e, num_iter, curr_psnr, curr_ssim))
          print('The loss of sinos, ril, recon, sum is {:.2f} {:.2f} {:.2f} {:.2f}'.format(loss_sino, loss_ril, loss_recon, loss_reconstructor))
      if e%2==0:
        state={'epoch': e,
           'reconstructor_state': self.reconstructor_func.state_dict(),
           'reconstructor_optimizer': self.reconstructor_optimizer.state_dict(),
              }
        self.save_checkpoint(e, state, num_iter)
        to_save_image=reconstructed_image[0,0,...].detach().cpu().numpy()
        to_save_image_=sitk.GetImageFromArray(to_save_image)
        sitk.WriteImage(to_save_image_, './results/visualization/pred_epoch_{}_iter_{}_{:.2f}_{:.2f}.nii.gz'.format(e, num_iter, curr_psnr, curr_ssim))

      # self.inference()

  def save_checkpoint(self, num_epoch, state, num_iter):
    base_path='./results/models'
    save_path=base_path+'/'+'epoch_'+str(num_epoch)+'iter'+str(num_iter)+'.pth.tar'
    torch.save(state, save_path)
    print('Save model after {}-th epoch'.format(num_epoch))

  def calculate_metric(self, pred, gt):
    assert len(pred.shape)==4 and pred.shape==gt.shape
    pred=torch.clamp(pred, 0.0, 1.0).cpu().data.numpy()
    gt=torch.clamp(gt, 0.0, 1.0).cpu().data.numpy()
    curr_psnr, curr_ssim=0.0, 0.0
    for i in range(pred.shape[0]):
      for j in range(pred.shape[1]):
        curr_psnr=compare_psnr(pred[i,j,...], gt[i,j,...], data_range=1.0)
        curr_ssim=compare_ssim(pred[i,j,...], gt[i,j,...], gaussian_weights=True, win_size=11, data_range=2.0,sigma=1.5)
        curr_rmse=np.sqrt(metrics.mean_squared_error(pred[i,j,...], gt[i,j,...]))
    return curr_psnr, curr_ssim, curr_rmse

  def inference(self):
    # Data Flow Pipiline
    print('Reading CT slices Beginning')
    self.test_dataset=CTSlice_Provider('/home1/cewang/Data/NIH-AAPM-DuDo/test', poission_level=self.poission_level, gaussian_level=self.gaussian_level, num_view=self.num_view)
    self.test_loader=DataLoader(dataset=self.test_dataset, batch_size=1, shuffle=False)

    state=torch.load('./results/models/epoch_100iter1745.pth.tar')
    self.reconstructor_func.load_state_dict(state['reconstructor_state'])

    aver_psnr, aver_ssim, aver_rmse=0.0, 0.0, 0.0
    aver_time=0.0
    time_start=time.time()
    with torch.no_grad():
      for num_iter, (gt, fbp_u, projs_noisy) in enumerate(self.test_loader):
        if self.is_cuda:
          gt=gt.cuda()
          fbp_u=fbp_u.cuda()
          projs_noisy=projs_noisy.float().cuda()
          self.reconstructor_func=self.reconstructor_func.cuda()

        time_start_in=time.time()
        _, sino_enhanced, ___,reconstructed_image=self.reconstructor_func(fbp_u, gt, projs_noisy)
        time_end_in=time.time()
        curr_psnr, curr_ssim, curr_rmse=self.calculate_metric(reconstructed_image, gt)
        fbp_psnr, fbp_ssim, fbp_rmse=self.calculate_metric(fbp_u, gt)
        print('This is the {}-th infering slice, the psnr and ssim is {:.2f} {:.4f}'.format(num_iter, curr_psnr, curr_ssim))
        aver_psnr+=curr_psnr
        aver_ssim+=curr_ssim
        aver_rmse+=curr_rmse
        aver_time+=(time_end_in-time_start_in)
        # ts_recon_imgs=reconstructed_image[0,0,...].detach().cpu().numpy()
        # ts_fbp_imgs=fbp_u[0,0,...].detach().cpu().numpy()
        # ts_recon_imgs_=sitk.GetImageFromArray(ts_recon_imgs)
        # ts_fbp_imgs_=sitk.GetImageFromArray(ts_fbp_imgs)
        # sitk.WriteImage(ts_recon_imgs_, './Images/recon_imgs_noise3/Images_{}_{:.2f}_{:.2f}.nii.gz'.format(num_iter, curr_psnr, curr_ssim))
        # sitk.WriteImage(ts_fbp_imgs_, './Images/fbp_imgs_noise3/Images_{}_{:.2f}_{:.2f}.nii.gz'.format(num_iter, fbp_psnr, fbp_ssim))
      time_end=time.time()
      aver_psnr=aver_psnr/(num_iter+1)
      aver_ssim=aver_ssim/(num_iter+1)
      aver_rmse=aver_rmse/(num_iter+1)
      aver_time=aver_time/(num_iter+1)
      aver_time_out=(time_end-time_start)/(num_iter+1)
      print('the average psnr and ssim rmse is {:.2f} {:.4f} {:.4f}'.format(aver_psnr, aver_ssim, aver_rmse))
      print('the average time of inside and outside is {} and {}'.format(aver_time, aver_time_out))

if __name__=='__main__':
  trainer=Trainer()
  # trainer.train()
  # trainer.inference()
