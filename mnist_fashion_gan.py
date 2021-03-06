import warnings
warnings.filterwarnings('ignore')

import argparse
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
from models.mnist_fashion_discriminator import *
from models.mnist_fashion_generator import *


if __name__ == "__main__":
    # python3 mnist_fashion_gan.py --n_epochs 600 --batch_size 64 --show_advance 50
    parser = argparse.ArgumentParser()
    parser.add_argument("--cgan", type=int, default=0, help="Train a Vanilla GAN or Conditional GAN. Default to 0 (False).")
    parser.add_argument("--with_normalization", type=int, default=1, help="Normalize the dataset.")
    parser.add_argument("--latent_dim", type=int, default=64, help="Latent vector to input to the GAN.")
    parser.add_argument("--n_epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of input datas in the batch.")
    parser.add_argument("--lr", type=float, default=1e-04, help="Learning rate.")
    parser.add_argument("--save_weights", type=int, default=5000, help="Number of iterations before saving the weights")
    parser.add_argument("--show_advance", type=int, default=10, help="Number of iterations before showing advance (loss, images) in tensorboard")
    parser.add_argument("--data_path", type=str, default='./data/fashion-mnist_train.csv', help="Path to the dataset.")

    args = parser.parse_args()

    # setting random seed to have same bahaviors when we re-run the same experiment.
    np.random.seed(5)
    torch.manual_seed(5)

    # Type of GAN (Vanilla GAN or Conditional GAN)
    cgan = args.cgan

    # Model params
    latent_dim = args.latent_dim
    n_epochs = args.n_epochs
    batch_size= args.batch_size
    lr= args.lr
    save_weights= args.save_weights
    show_advance= args.show_advance
    hyperparams = Hyperparameters(batch_size=batch_size,n_epochs=n_epochs, cgan=cgan, latent_dim=latent_dim, lr=lr, show_advance=show_advance, save_weights=save_weights)

    # dateset
    with_normalization = args.with_normalization
    mnist_data = MnistFashionData(path=args.data_path, cgan=cgan, with_normalization=with_normalization, bs=batch_size)

    # Fixed noise vector to see the evolution of the generated images during the training
    fixed_noise_vect = torch.randn((hyperparams.batch_size,hyperparams.input_dim_gen)).to(hyperparams.device)

    print("## ------------------------------------------------------------------------- ##")
    print("hyperparams.input_dim_gen : ",hyperparams.input_dim_gen)
    print("latent_dim : ",latent_dim)
    print("hyperparams.lr : ", hyperparams.lr)
    print("hyperparams.device : ", hyperparams.device)
    print("fixed_noise_vect.shape : ", fixed_noise_vect.shape)
    print("n_epochs : ",n_epochs)
    print("batch_size : ",batch_size)
    print("lr : ",lr)
    print("with_normalization : ",with_normalization)
    print("save_weights : ",save_weights)
    print("show_advance : ",show_advance)
    print("## ------------------------------------------------------------------------- ##")
    print("[INFO] Started training using device : ",hyperparams.device)
    print("## ------------------------------------------------------------------------- ##")
    
    # Models
    disc = MnistFashionDiscriminator(cgan=cgan, 
                        n_inputs=hyperparams.img_dim, 
                        n_classes=hyperparams.n_classes,  
                        output_dim=hyperparams.n_output_disc, 
                        activation='lk_relu',
                        alpha_relu=hyperparams.alpha_relu,
                        norm_type='bn1d',
                        down_steps=4).to(hyperparams.device)

    gen = MnistFashionGenerator(
                cgan=hyperparams.cgan, 
                img_dim=hyperparams.img_dim, 
                n_classes=hyperparams.n_classes,
                n_inputs=hyperparams.latent_dim,
                n_output = 256,
                min_features = 64, 
                max_features=512,

                down_steps=5, 
                bottleneck_size=2, 
                up_steps=5,

                norm_type='bn1d', 
                norm_before=True, 
                activation='lk_relu', 
                alpha_relu=hyperparams.alpha_relu, 
                use_bias=True,
                ).to(hyperparams.device)
    
    # Optimizers
    optimization = Optimization(gen, disc, hyperparams, cgan, experiment="vanilla_gan_tanh_w_norma_latent_64")

    # Start training
    optimization.train(mnist_data.dataloader)