import torch.nn as nn
from torch.nn import Module, Linear, Conv2d
from torch.nn import Softmax, ReLU, LeakyReLU, Sigmoid, Tanh
from models.block import LinearLayer, ConvResidualBlock

class Generator(nn.Module):

  # -----------------------------------------------------------------------------#
  # -----------------------------------------------------------------------------#

  def __init__( self, norm_type='bn', norm_before=True, activation='lk_relu', alpha_relu=0.15, use_bias=True,
                min_features = 16, max_features=256,
                n_inputs=128, n_output = 256,
                use_pad=use_pad, interpolation_mode=interpolation_mode, kernel_size=3,
                img_dim=784, n_classes=10, cgan=True, down_steps=3, bottleneck=2, up_steps=2):
    """ 
    The discriminator is in an encoder-decoder shape, we start by a small lattent vector for which we increase the dimensions to the desired image dimension.
    """
    super(Generator, self).__init__()

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

    # input layer
    self.encoder.append(
      LinearLayer(in_features=n_inputs, out_features=n_output, norm_type=norm_type, activation=activation, alpha_relu=alpha_relu, norm_before=norm_before, use_bias=use_bias)
    )
    
    for i in range(down_steps):
      
      self.encoder.append(
        ConvResidualBlock(in_features=n_inputs, out_features=n_output, kernel_size=kernel_size, scale='down', use_pad=use_pad, use_bias=use_bias, norm_type=norm_type, norm_before=norm_before, 
                          activation=activation, alpha_relu=alpha_relu, interpolation_mode=interpolation_mode)
      )

      if i != down_steps-1 :
        n_inputs = features_cliping(n_inputs // 2)
        n_output = features_cliping(n_output // 2)
      

    ##########################################
    #####            Bottleneck           ####
    ##########################################

    self.bottleneck = []
    for i in range(bottleneck):

      self.bottleneck.append(
        ConvResidualBlock(in_features=n_output, out_features=n_output, kernel_size=kernel_size, scale='none', use_pad=use_pad, use_bias=use_bias, norm_type=norm_type, norm_before=norm_before, 
                          activation=activation, alpha_relu=alpha_relu, interpolation_mode=interpolation_mode)
      )


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

      self.bottleneck.append(
        ConvResidualBlock(in_features=n_inputs, out_features=n_output, kernel_size=kernel_size, scale='up', use_pad=use_pad, use_bias=use_bias, norm_type=norm_type, norm_before=norm_before, 
                          activation=activation, alpha_relu=alpha_relu, interpolation_mode=interpolation_mode)
      )

    if cgan :
      out_dim = img_dim + n_classes
    else :
      out_dim = img_dim

    # output layer
    self.out_layer = LinearLayer(in_features=n_output, out_features=out_dim, norm_type=norm_type, activation=activation, alpha_relu=alpha_relu, norm_before=norm_before, use_bias=use_bias)

  # -----------------------------------------------------------------------------#
  # -----------------------------------------------------------------------------#

  def forward(self, x) :

    out = self.encoder(x)
    out = self.bottleneck(out)
    out = self.decoder(out)
    out = self.out_layer(out)

    return out

  # -----------------------------------------------------------------------------#
  # -----------------------------------------------------------------------------#