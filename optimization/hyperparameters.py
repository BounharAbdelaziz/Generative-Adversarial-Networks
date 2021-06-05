import torch

class Hyperparameters():
  
  def __init__(self, lr=0.00002, batch_size=32, n_epochs=50, latent_dim=128, img_dim=784, n_output_disc=1, n_classes=10, alpha_relu=0.15, cgan=True):

    self.device = "cuda" if torch.cuda.is_available() else "cpu"
    self.lr = lr
    self.batch_size = batch_size
    self.n_epochs = n_epochs
    self.latent_dim = latent_dim
    self.cgan = cgan
    # for the conditional GAN we include he one hot vector
    if self.cgan :
      self.input_dim_gen = latent_dim + n_classes
    else :
      self.input_dim_gen = latent_dim

    self.img_dim=img_dim
    self.n_output_disc = n_output_disc
    self.n_classes = n_classes
    self.alpha_relu=alpha_relu