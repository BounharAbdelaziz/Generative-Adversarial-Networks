import torch.nn as nn
from torch.nn import Module, BCELoss, Linear, Conv2d
from torch.nn import Softmax, ReLU, LeakyReLU, Sigmoid, Tanh
from models.block import LinearLayer, ConvResidualBlock

class Discriminator(nn.Module):
  # -----------------------------------------------------------------------------#

  # 28*28 = 784
  def __init__(self, n_inputs=784, n_output=1, n_classes=10,
               norm_type='bn', norm_before=True, activation='lk_relu',  use_bias=True,
                min_features = 16, max_features=256,
                use_pad=use_pad, interpolation_mode=interpolation_mode, kernel_size=3,
                down_steps=3, alpha_relu=0.15, cgan=True):
    """
    The discriminator is in an encoder shape, we encode the features to a smaller space of features and do the decisions.
    """
    super(Discriminator, self).__init__()
    
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
      
      self.encoder.append(
        ConvResidualBlock(in_features=n_inputs, out_features=n_output, kernel_size=kernel_size, scale='down', use_pad=use_pad, use_bias=use_bias, norm_type=norm_type, norm_before=norm_before, 
                          activation=activation, alpha_relu=alpha_relu, interpolation_mode=interpolation_mode)
      )

      if i != down_steps-1 :
        n_inputs = features_cliping(n_inputs // 2)
        n_output = features_cliping(n_output // 2)


    self.out_layer = LinearLayer(in_features=n_output, out_features=n_output, norm_type=norm_type, activation='sigmoid', alpha_relu=alpha_relu, norm_before=norm_before, use_bias=use_bias)

  # -----------------------------------------------------------------------------#

  def forward(self, x) :

    out = self.encoder(x)
    out = self.out_layer(out)

    return x
  # -----------------------------------------------------------------------------#
  