import torch
import torch.nn as nn
from torch.nn import Module, Linear, Conv2d
from torch.nn import Softmax, ReLU, LeakyReLU, Sigmoid, Tanh
from models.block import LinearLayer, LinearResidualBlock

import functools
import operator

class MnistFashionGenerator(nn.Module):

  # -----------------------------------------------------------------------------#
  # -----------------------------------------------------------------------------#

  def __init__( self, norm_type='bn', norm_before=True, activation='lk_relu', alpha_relu=0.15, use_bias=True,
                min_features = 16, max_features=256,
                n_inputs=128, n_output = 256,
                img_dim=784, n_classes=10, cgan=False, down_steps=2, bottleneck_size=2, up_steps=2):
    """ 
    The discriminator is in an encoder-decoder shape, we start by a small lattent vector for which we increase the dimensions to the desired image dimension.
    """
    super(MnistFashionGenerator, self).__init__()

    # conditional GAN - we input also the 10-dim vector of the class of the object
    if cgan :
      # defines the latent vector dimensions
      n_inputs = n_inputs + n_classes

    # to do the cliping in the encoder and decoder
    features_cliping = lambda x : max(min_features, min(x, max_features))

    ##########################################
    #####             Encoder             ####
    ##########################################

    self.encoder = []

    # for the input layer
    n_output = features_cliping(n_inputs * 2)
    print(f'n_inputs : {n_inputs}')
    print(f'n_output : {n_output}')

    # input layer
    self.encoder.append(
      LinearLayer(in_features=n_inputs, out_features=n_output, norm_type=norm_type, activation=activation, alpha_relu=alpha_relu, norm_before=norm_before, use_bias=use_bias)
    )
    
    n_inputs = n_output
    n_output = features_cliping(n_output // 2)

    for i in range(down_steps):

      self.encoder.append(
        LinearResidualBlock(in_features=n_inputs, out_features=n_output, use_bias=use_bias, norm_type=norm_type, norm_before=norm_before, activation=activation, alpha_relu=alpha_relu)
      )

      if i != down_steps-1 :
        n_inputs = features_cliping(n_inputs // 2)
        n_output = features_cliping(n_output // 2)
      
    self.encoder = nn.Sequential(*self.encoder)

    ##########################################
    #####            Bottleneck           ####
    ##########################################

    self.bottleneck = []
    for i in range(bottleneck_size):

      self.bottleneck.append(
        LinearResidualBlock(in_features=n_output, out_features=n_output, use_bias=use_bias, norm_type=norm_type, norm_before=norm_before, activation=activation, alpha_relu=alpha_relu)
      )

    self.bottleneck = nn.Sequential(*self.bottleneck)

    ##########################################
    #####             Decoder             ####
    ##########################################

    self.decoder = []

    for i in range(up_steps):
      if i == 0 :
        n_inputs = n_output
      else :
        n_inputs = features_cliping(n_inputs * 2)
      
      n_output = features_cliping(n_output * 2)
      
      self.decoder.append(
        LinearResidualBlock(in_features=n_inputs, out_features=n_output, use_bias=use_bias, norm_type=norm_type, norm_before=norm_before, activation=activation, alpha_relu=alpha_relu)
      )

    self.decoder = nn.Sequential(*self.decoder)

    if cgan :
      out_dim = img_dim + n_classes
    else :
      out_dim = img_dim

    # output layer
    self.flatten = nn.Flatten()

    # num_features_before_fcnn = functools.reduce(operator.mul, list(self(torch.rand(1, *(self.encoder[0].in_features))).shape))

    # print("num_features_before_fcnn : ",num_features_before_fcnn)
    # no normalization in the output layer
    self.out_layer = LinearLayer(in_features=n_output, out_features=out_dim, norm_type='none', activation='tanh', alpha_relu=alpha_relu, norm_before=norm_before, use_bias=use_bias)

  # -----------------------------------------------------------------------------#
  # -----------------------------------------------------------------------------#

  def forward(self, x) :

    out = self.encoder(x)
    out = self.bottleneck(out)
    out = self.decoder(out)
    out = self.flatten(out)

    out = self.out_layer(out)

    return out

  # -----------------------------------------------------------------------------#
  # -----------------------------------------------------------------------------#