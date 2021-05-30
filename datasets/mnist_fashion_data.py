import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision import transforms

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


class MnistFashionData(Dataset):

    # -----------------------------------------------------------------------------#

    def __init__(self, path, bs=32, cgan=True, with_normalization=True):
      """ load the dataset"""

      # read the dataframe with pandas
      self.df_train = pd.read_csv(path)
      
      # tranformations
      self.with_normalization = with_normalization # normalization helps stabilizing the training of GANs.
      if with_normalization :
        self.transforms = transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize( (0.5, ) , (0.5,) )
                          ])

        self.data =  self.transforms(self.prepare_train_data(self.df_train, cgan))
      else :
        # prepare the dataset, we don't need a transform to tensor, already handeled by the dataloader
        self.data =  self.prepare_train_data(self.df_train, cgan)

      # init dataloarder
      self.dataloader = DataLoader(self.data, batch_size=bs, shuffle=True) 

      # mapping between index and class. Used for plots only.
      self.labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
      }

    # -----------------------------------------------------------------------------#

    # number of rows in the dataset
    def __len__(self):
        return len(self.data)

    # -----------------------------------------------------------------------------#

    # get a row at an index
    def __getitem__(self, idx):
      if self.with_normalization :
        return self.data[0][idx]
      else :
        return self.data[idx]

    # -----------------------------------------------------------------------------#

    def plot_image(self, data, index, labels=None):

      if isinstance(data, pd.core.frame.DataFrame):
        # locate the data
        row = data.iloc[index]
        # get the label
        label = self.labels_map[row[0]]
        # first position is for the label
        img_data = row[1:]
        # 784 -> 28*28 image
        dim = np.sqrt(len(row)-1).astype(np.int8)
        # transform to numpy to reshape
        img = img_data.to_numpy().reshape(dim,dim)

      elif isinstance(data, np.ndarray):    
        img_data = data[index]
        label = self.labels_map[labels[index]]
        dim = np.sqrt(len(img_data)).astype(np.int8)
        img = img_data.reshape(dim,dim)

      else :
        print("Unsupported data type for arg 1. Please use Pandas dataframe or Numpy arrays.")

      plt.title("Showing image of class : " + str(label) )
      plt.imshow(img)
      plt.show()

    # -----------------------------------------------------------------------------#

    def prepare_train_data(self, df, cgan):

      if isinstance(df, pd.core.frame.DataFrame):
        X = df.to_numpy(dtype='float64')[:, 1:]
        y = df.to_numpy(dtype='float64')[:, :1]

      else :
        X = df[:, 1:]
        y = df[:, :1]

      encoder = OneHotEncoder()
      # encode class in 1-hot vector
      y = encoder.fit_transform(y).toarray()    
      
      # stack X and y data, it contains the class label so that we learn the conditional probablity
      if cgan :
        data =  np.column_stack((X, y))
      else :
        return X
      return data

    # -----------------------------------------------------------------------------#