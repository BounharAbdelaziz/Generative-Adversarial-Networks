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
                use_pad=True, interpolation_mode='nearest', kernel_size=3,
                img_dim=784, n_classes=10, cgan=True, down_steps=2, bottleneck_size=2, up_steps=2):
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
    
    n_inputs = n_output
    n_output = features_cliping(n_output // 2)

    print("----------------- gen Encoder -----------------")

    for i in range(down_steps):
      print("i : ",i)
      print("n_inputs : ",n_inputs)
      print("n_output : ",n_output)
      print("-------------------------------------")

      self.encoder.append(
        ConvResidualBlock(in_features=n_inputs, out_features=n_output, kernel_size=kernel_size, scale='down', use_pad=use_pad, use_bias=use_bias, norm_type=norm_type, norm_before=norm_before, 
                          activation=activation, alpha_relu=alpha_relu, interpolation_mode=interpolation_mode)
      )

      if i != down_steps-1 :
        n_inputs = features_cliping(n_inputs // 2)
        n_output = features_cliping(n_output // 2)
      
    self.encoder = nn.Sequential(*self.encoder)

    ##########################################
    #####            Bottleneck           ####
    ##########################################
    print("----------------- gen Bottleneck -----------------")

    self.bottleneck = []
    for i in range(bottleneck_size):
      
      print("i : ",i)
      print("n_output : ",n_output)
      print("-------------------------------------")

      self.bottleneck.append(
        ConvResidualBlock(in_features=n_output, out_features=n_output, kernel_size=kernel_size, scale='none', use_pad=use_pad, use_bias=use_bias, norm_type=norm_type, norm_before=norm_before, 
                          activation=activation, alpha_relu=alpha_relu, interpolation_mode=interpolation_mode)
      )

    self.bottleneck = nn.Sequential(*self.bottleneck)

    ##########################################
    #####             Decoder             ####
    ##########################################

    self.decoder = []

    print("----------------- gen Decoder -----------------")

    for i in range(up_steps):
      if i == 0 :
        n_inputs = n_output
      else :
        n_inputs = features_cliping(n_inputs * 2)
      
      n_output = features_cliping(n_output * 2)
      
      print("i : ",i)
      print("n_inputs : ",n_inputs)
      print("n_output : ",n_output)
      print("-------------------------------------")
      self.decoder.append(
        ConvResidualBlock(in_features=n_inputs, out_features=n_output, kernel_size=kernel_size, scale='up', use_pad=use_pad, use_bias=use_bias, norm_type=norm_type, norm_before=norm_before, 
                          activation=activation, alpha_relu=alpha_relu, interpolation_mode=interpolation_mode)
      )

    self.decoder = nn.Sequential(*self.decoder)

    if cgan :
      out_dim = img_dim + n_classes
    else :
      out_dim = img_dim

    # output layer
    print("----------------- output layer -----------------")
    print("in_features : ",n_output)
    print("out_features : ",out_dim)

    self.flatten = nn.Flatten()
    n_output = 262144
    self.out_layer = LinearLayer(in_features=n_output, out_features=out_dim, norm_type='none', activation=activation, alpha_relu=alpha_relu, norm_before=norm_before, use_bias=use_bias)

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