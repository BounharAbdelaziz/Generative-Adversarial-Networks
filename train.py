import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms

from datasets.mnist_fashion_data import *
from optimization.hyperparameters import *
from optimization.opti import *
from models.discriminator import *
from models.generator import *


# Vanilla GAN
cgan=False

# dateset
with_normalization=True
mnist_data = MnistFashionData(path="./data/fashion-mnist_train.csv", cgan=cgan, with_normalization=with_normalization)

# Model params
latent_dim = 64
n_epochs = 200
hyperparams = Hyperparameters(n_epochs=n_epochs, cgan=cgan, latent_dim=latent_dim, lr=1e-04)

# Fixed noise vector to see the evolution of the generated images during the training
fixed_noise_vect = torch.randn((hyperparams.batch_size,hyperparams.input_dim_gen)).to(hyperparams.device)

print(hyperparams.input_dim_gen)
print(hyperparams.lr)
print(hyperparams.device)
print(fixed_noise_vect.shape)

# Models
disc = Discriminator(cgan=cgan, 
                     n_inputs=hyperparams.img_dim, 
                     n_classes=hyperparams.n_classes, 
                     n_output=hyperparams.n_output_disc, 
                     alpha_relu=hyperparams.alpha_relu,
                     norm_type='in').to(hyperparams.device)

gen = Generator(cgan=cgan, 
                n_inputs=hyperparams.latent_dim, 
                img_dim=hyperparams.img_dim, 
                n_classes=hyperparams.n_classes, 
                alpha_relu=hyperparams.alpha_relu,
                norm_type='in').to(hyperparams.device)


# Optimizers
optimization = Optimization(gen, disc, hyperparams, cgan)

# Start training
optimization.train(mnist_data.dataloader, experiment="vanilla")