import torch.nn as nn
from torch.nn import Module, BCELoss, Linear, Conv2d
from torch.nn import Softmax, ReLU, LeakyReLU, Sigmoid, Tanh
from models.block import LinearLayer, LinearResidualBlock

class MnistFashionDiscriminator(nn.Module):
  # -----------------------------------------------------------------------------#

  # 28*28 = 784
  def __init__(self, n_inputs=784, n_output=128, output_dim=1, n_classes=10,
               norm_type='bn', norm_before=True, activation='lk_relu',  use_bias=True,
                min_features = 16, max_features=256,               
                down_steps=2, alpha_relu=0.15, cgan=False):
    """
    The discriminator is in an encoder shape, we encode the features to a smaller space of features and do the decisions.
    """
    super(MnistFashionDiscriminator, self).__init__()
    
    # conditional GAN - we input also the 10-dim vector of the class of the object
    if cgan :
      n_inputs = n_inputs + n_classes

    # to do the cliping in the encoder and decoder
    features_cliping = lambda x : max(min_features, min(x, max_features))

    ##########################################
    #####             Encoder             ####
    ##########################################

    self.encoder = []

    # input layer
    self.encoder.append(
      LinearLayer(in_features=n_inputs, out_features=n_output, norm_type=norm_type, activation=activation, alpha_relu=alpha_relu, norm_before=norm_before, use_bias=use_bias)
    )

    for i in range(down_steps):
      
      if i == 0 :
        n_inputs = n_output
        n_output = features_cliping(n_output // 2)

      self.encoder.append(
        LinearResidualBlock(in_features=n_inputs, out_features=n_output, use_bias=use_bias, norm_type=norm_type, norm_before=norm_before, activation=activation, alpha_relu=alpha_relu)

      )

      if i != down_steps-1 :
        n_inputs = features_cliping(n_inputs // 2)
        n_output = features_cliping(n_output // 2)

    self.encoder = nn.Sequential(*self.encoder)

    # no normalization in the output layer
    self.out_layer = LinearLayer(in_features=n_output, out_features=output_dim, norm_type='none', activation='sigmoid', alpha_relu=alpha_relu, norm_before=norm_before, use_bias=use_bias)

  # -----------------------------------------------------------------------------#

  def forward(self, x) :
    out = self.encoder(x)
    out = self.out_layer(out)
    return out
  # -----------------------------------------------------------------------------#
  