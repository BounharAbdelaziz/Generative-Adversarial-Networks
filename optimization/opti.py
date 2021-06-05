from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.optim import Adam
import matplotlib.pyplot as plt

import utils.helpers as helper
from optimization.loss import GANLoss

class Optimization():
  
  def __init__(self, gen, disc, hyperparams, cgan=True, n_input=784, lambda_gan_gen=1.85, lambda_gan_disc=0.85):

    self.generator = gen
    self.discriminator = disc
    # we use Adam optimizer for both Generator and Discriminator
    self.opt_gen = Adam(self.generator.parameters(), lr=hyperparams.lr)
    self.opt_disc = Adam(self.discriminator.parameters(), lr=hyperparams.lr)
    self.hyperparams = hyperparams
    self.criterionGAN = GANLoss()
    self.cgan = cgan
    self.lambda_gan_gen = lambda_gan_gen
    self.lambda_gan_disc = lambda_gan_disc

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


  def backward_G(self, disc_real, disc_fake, disc_pred):

    _, loss_G = self.criterionGAN(disc_real, disc_fake, disc_pred)
    loss_G = loss_G * self.lambda_gan_gen
    with torch.autograd.set_detect_anomaly(True) :
      loss_G.backward()

    return loss_G

  def backward_D(self, disc_real, disc_fake, disc_pred):

    loss_D, _ = self.criterionGAN(disc_real, disc_fake, disc_pred)
    loss_D = loss_D * self.lambda_gan_disc
    with torch.autograd.set_detect_anomaly(True) :
      loss_D.backward()

    return loss_D
  
  def optimize_network(self, disc_real, disc_fake, disc_pred):

    # run backprop on the Generator
    self.opt_gen.zero_grad()
    loss_G = self.backward_G(disc_real, disc_fake, disc_pred)
    # self.opt_gen.step()
    
    # run backprop on the Discriminator
    self.opt_disc.zero_grad()
    loss_D = self.backward_D(disc_real, disc_fake, disc_pred)
    self.opt_gen.step()
    self.opt_disc.step()
    print("[INFO] loss_D =  ",loss_D)
    print("[INFO] loss_G =  ",loss_G)

    return loss_D, loss_G

  def train(self, dataloader, steps_train_disc=1, n_channels=256, experiment="MNIST_FASHION", h=28, w=28):
    step = 0
    
    if self.cgan :
      print("[INFO] Started training a Conditional GAN on the MNIST Fashion dataset, using device ",self.hyperparams.device,"...")
    else :
      print("[INFO] Started training a GAN on the MNIST Fashion dataset, using device ",self.hyperparams.device,"...")

    for epoch in range(self.hyperparams.n_epochs):
      print("epoch = ",epoch," --------------------------------------------------------")

      for batch_idx, real_data in enumerate(dataloader) :        
        real_data = real_data.view(-1, self.hyperparams.img_dim).to(self.hyperparams.device)

        batch_size = real_data.shape[0]

        ##########################################
        #####  Launch training of Generator   ####
        ##########################################

        # we generate an image from a noise vector
        noise = torch.randn(self.hyperparams.batch_size, n_channels, self.hyperparams.latent_dim, self.hyperparams.latent_dim).to(self.hyperparams.device)
        fake_data = self.generator(noise)
        disc_pred = self.discriminator(fake_data.float()).view(-1)

        ##########################################
        #### Launch training of Discriminator ####
        ##########################################
          
        if self.hyperparams.cgan :
          y_fake = torch.eye(self.hyperparams.n_classes)[np.argmax(torch.randn((self.hyperparams.batch_size, self.hyperparams.n_classes)) , axis=1)].to(self.hyperparams.device)
          noise = torch.column_stack((noise, y_fake))

        # noise = torch.randn(batch_size, self.hyperparams.input_dim_gen).to(self.hyperparams.device)
        fake_data = self.generator(noise)

        # prediction of the discriminator on real an fake images in the batch
        disc_fake = self.discriminator(fake_data.float()).view(-1)
        disc_real = self.discriminator(real_data.float()).view(-1)

        loss_D, loss_G = self.optimize_network(disc_real, disc_fake, disc_pred)
        
      # state after this epoch
      with torch.no_grad():
        fake_data_ = self.generator(self.fixed_noise_vect)
        # self.plot_image(fake_data_)
        fake_data_ = fake_data_[:, :self.hyperparams.img_dim].reshape(-1, 1, h, w)
        real_data_ = real_data[:, :self.hyperparams.img_dim].reshape(-1, 1, h, w)

        img_fake = torchvision.utils.make_grid(fake_data_, normalize=True)
        img_real = torchvision.utils.make_grid(real_data_, normalize=True)

        helper.write_logs_tb(experiment, img_fake, img_real, loss_D, loss_G, step, epoch, hyperparams, with_print_logs=True)
        step = step + 1
