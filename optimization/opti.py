from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.optim import Adam
import matplotlib.pyplot as plt

import utils.helpers as helper
from optimization.loss import GenLoss, DiscLoss

import os
from tqdm import tqdm

class Optimization():
  
  def __init__(self, gen, disc, hyperparams, cgan=False, n_input=784, lambda_gan_gen=1, lambda_gan_disc=1):

    self.generator = gen
    self.discriminator = disc
    # we use Adam optimizer for both Generator and Discriminator
    self.opt_gen = Adam(self.generator.parameters(), lr=hyperparams.lr)
    self.opt_disc = Adam(self.discriminator.parameters(), lr=hyperparams.lr)
    self.hyperparams = hyperparams
    self.criterionGen = GenLoss()
    self.criterionDisc = DiscLoss()
    self.cgan = cgan
    self.lambda_gan_gen = lambda_gan_gen
    self.lambda_gan_disc = lambda_gan_disc
    # self.n_channels = n_channels

    # Fixed noise vector to see the evolution of the generated images
    self.fixed_noise_vect = torch.randn(self.hyperparams.batch_size, self.hyperparams.latent_dim).to(self.hyperparams.device)

    if cgan :
      fixed_y_fake = torch.eye(self.hyperparams.n_classes)[np.argmax(torch.randn((self.hyperparams.batch_size, self.hyperparams.n_classes)) , axis=1)].to(self.hyperparams.device)
      self.fixed_noise_vect = torch.column_stack((self.fixed_noise_vect, fixed_y_fake))

  def plot_image(self, data):
    
    data = data[0][:self.hyperparams.img_dim].detach().cpu()
    dim = np.sqrt(len(data)).astype(np.int8)
    img = data.reshape(dim,dim)

    plt.imshow(img)
    plt.show()
  
  
  def optimize_network(self, disc_real, disc_fake):
    
    # run backprop on the Discriminator
    self.opt_disc.zero_grad()
    loss_D = self.backward_D(disc_real, disc_fake)
    self.opt_disc.step()

    # run backprop on the Generator
    self.opt_gen.zero_grad()
    loss_G = self.backward_G(disc_fake)
    self.opt_gen.step()

    return loss_D, loss_G
  
  def backward_G(self, disc_fake):

    loss_G = self.criterionGen(disc_fake)
    loss_G = loss_G * self.lambda_gan_gen
    with torch.autograd.set_detect_anomaly(True) :
      loss_G.backward()

    return loss_G

  def backward_D(self, disc_real, disc_fake):

    loss_D = self.criterionDisc(disc_real, disc_fake)
    loss_D = loss_D * self.lambda_gan_disc

    with torch.autograd.set_detect_anomaly(True) :
      loss_D.backward(retain_graph=True)

    return loss_D

  def train(self, dataloader, steps_train_disc=1, experiment="MNIST_FASHION", h=28, w=28):
    step = 0
    cpt = 0
    self.PATH_CKPT = "./check_points/"+experiment+"/"
    
    if self.cgan :
      print("[INFO] Started training a Conditional GAN on the MNIST Fashion dataset, using device ",self.hyperparams.device,"...")
    else :
      print("[INFO] Started training a GAN on the MNIST Fashion dataset, using device ",self.hyperparams.device,"...")

    if self.hyperparams.device != 'cpu':
      # using DataParallel tu copy the Tensors on all available GPUs
      device_ids = [i for i in range(torch.cuda.device_count())]
      print(f'[INFO] Copying tensors to all available GPUs : {device_ids}')
      # if len(device_ids) > 1 :
      self.generator = nn.DataParallel(self.generator, device_ids)
      self.discriminator = nn.DataParallel(self.discriminator, device_ids)
      self.generator.to(self.hyperparams.device)
      self.discriminator.to(self.hyperparams.device)

    for epoch in tqdm(range(self.hyperparams.n_epochs)):
      print("epoch = ",epoch," --------------------------------------------------------")

      for batch_idx, real_data in enumerate(dataloader) :        
        # print("[INFO] real_data.shape : ", real_data.shape)

        real_data = real_data.view(-1, self.hyperparams.img_dim).to(self.hyperparams.device)

        batch_size = real_data.shape[0]

        ##########################################
        #### Launch training of Discriminator ####
        ##########################################

        # we generate an image from a noise vector
        noise = torch.randn(self.hyperparams.batch_size, self.hyperparams.latent_dim).to(self.hyperparams.device)
        if self.hyperparams.cgan :
          y_fake = torch.eye(self.hyperparams.n_classes)[np.argmax(torch.randn((self.hyperparams.batch_size, self.hyperparams.n_classes)) , axis=1)].to(self.hyperparams.device)
          noise = torch.column_stack((noise, y_fake))

        # print("[INFO] noise.shape : ", noise.shape)
        fake_data = self.generator(noise)
                  
        # prediction of the discriminator on real an fake images in the batch
        disc_real = self.discriminator(real_data.float()).view(-1)
        # detach from the computational graph to not re-use the output of the Generator
        disc_fake = self.discriminator(fake_data.float().detach()).view(-1)

        # run backprop on the Discriminator
        loss_D = self.criterionDisc(disc_real, disc_fake)
        loss_D = loss_D * self.lambda_gan_disc

        self.opt_disc.zero_grad()
        with torch.autograd.set_detect_anomaly(True) :
          loss_D.backward()
        # Gradient steps for the Discriminator w.r.t its respective loss.
        self.opt_disc.step()

        ##########################################
        #####  Launch training of Generator   ####
        ##########################################

        # run backprop on the Generator
        disc_fake = self.discriminator(fake_data.float()).view(-1)

        loss_G = self.criterionGen(disc_fake)
        loss_G = loss_G * self.lambda_gan_gen

        self.opt_gen.zero_grad()
        with torch.autograd.set_detect_anomaly(True) :
          loss_G.backward()
        # Gradient steps for the Generator w.r.t its respective loss.
        self.opt_gen.step()


        print("[INFO] loss_D =  ",loss_D)
        print("[INFO] loss_G =  ",loss_G)

        if batch_idx % self.hyperparams.show_advance == 0 :

          # show advance
          print("[INFO] logging advance...")
          with torch.no_grad():
            print("[INFO] fixed_noise_vect.shape : ",self.fixed_noise_vect.shape)

            # generate images
            fake_data_ = self.generator(self.fixed_noise_vect)

            fake_data_ = fake_data_[:, :self.hyperparams.img_dim].reshape(-1, 1, h, w)
            real_data_ = real_data[:, :self.hyperparams.img_dim].reshape(-1, 1, h, w)

            img_fake = torchvision.utils.make_grid(fake_data_, normalize=True)
            img_real = torchvision.utils.make_grid(real_data_, normalize=True)

            helper.write_logs_tb(experiment, img_fake, img_real, loss_D, loss_G, step, epoch, self.hyperparams, with_print_logs=True)
            step = step + 1

        if batch_idx % self.hyperparams.save_weights == 0 and batch_idx!=0 :

          # show advance
          print("[INFO] Saving weights...")
          print(os.curdir)
          print(os.path.abspath(self.PATH_CKPT))
          torch.save(self.discriminator.state_dict(), os.path.join(self.PATH_CKPT,"D_it_"+str(step)+".pth"))
          torch.save(self.generator.state_dict(), os.path.join(self.PATH_CKPT,"G_it_"+str(step)+".pth"))

    print("[INFO] Saving weights last step...")
    print(os.curdir)
    print(os.path.abspath(self.PATH_CKPT))
    torch.save(self.discriminator.state_dict(), os.path.join(self.PATH_CKPT,"D2_it_"+str(step)+".pth"))
    torch.save(self.generator.state_dict(), os.path.join(self.PATH_CKPT,"G2_it_"+str(step)+".pth"))