import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc,precision_recall_curve,roc_curve,confusion_matrix
import os,sys
import pickle
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import random
import seaborn as sns
import torch
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.utils.data import DataLoader

from torch.nn import Linear, Conv2d, BatchNorm2d, MaxPool2d, Dropout2d
from torch.nn.functional import relu, elu, relu6, sigmoid, tanh, softmax
from torch.nn import Linear, GRU, Conv2d, Dropout, MaxPool2d, BatchNorm1d

import torch.optim as optim
from sklearn.metrics import accuracy_score,recall_score,f1_score
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
          nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
          if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
          nn.init.constant_(m.weight.data, 1)
          nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
          nn.init.kaiming_uniform_(m.weight.data,nonlinearity='relu')
          if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)



# hyperameters of the model
input_channels = 1
peptide_input_height = 10
hla_input_height = 34
encoding_dimension = 12


# define network
class DeepImmuno(nn.Module):

    def __init__(self):
        super(DeepImmuno, self).__init__()

        # Convelution of peptide
        self.conv1_peptide = Conv2d(in_channels=input_channels,
                            out_channels=16,
                            kernel_size=(2,encoding_dimension),
                            stride=1,
                            padding=0)
        
        self.BatchNorm_conv1_peptides = BatchNorm2d(16) # Output channels from the previous layer
        self.conv2_peptide = Conv2d(in_channels=16,
                            out_channels=32,
                            kernel_size=(2,1),
                            stride=1,
                            padding=0)
        self.BatchNorm_conv2_peptides = BatchNorm2d(32) # Output channels from the previous layer
        self.maxpool1_peptide = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1), padding=0)

        # Convelution of HLA
        self.conv1_HLA = Conv2d(in_channels=input_channels,
                            out_channels=16,
                            kernel_size=(6,encoding_dimension),
                            stride=1,
                            padding=0)
        self.BatchNorm_conv1_HLA = BatchNorm2d(16) # Output channels from the previous layer
        self.maxpool1_HLA = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1), padding=0)
        
        self.conv2_HLA = Conv2d(in_channels=16,
                            out_channels=32,
                            kernel_size=(9,1),
                            stride=1,
                            padding=0)
        self.BatchNorm_conv2_peptides = BatchNorm2d(32) # Output channels from the previous layer
        self.maxpool2_HLA = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1), padding=0)

        # Denselayer
        self.L_in = Linear(in_features=224,
                            out_features=128)
        self.drop_out = nn.Dropout(p=0.2)
        self.L_out = Linear(in_features=128,
                            out_features=1,
                            bias=False)


    def forward(self, peptide, HLA): # x.size() = [batch, channel, height, width]
        peptide = self.add_channel_dimension(peptide,peptide_input_height,encoding_dimension)
        print(peptide.shape)
        # Encoding the peptide
        peptide = self.conv1_peptide(peptide)
        peptide = self.BatchNorm_conv1_peptides(peptide)
        peptide = relu(peptide)
        peptide = self.conv2_peptide(peptide)
        peptide = self.BatchNorm_conv2_peptides(peptide)
        peptide = relu(peptide)
        peptide = self.maxpool1_peptide(peptide)
        peptide = torch.flatten(peptide,start_dim=1)

        # Encoding the HLA
        HLA = self.add_channel_dimension(HLA,hla_input_height,encoding_dimension)
        print(HLA.shape)
        HLA = self.conv1_HLA(HLA)
        HLA = self.BatchNorm_conv1_HLA(HLA)
        HLA = relu(HLA)
        HLA = self.maxpool1_HLA(HLA)
        HLA = self.conv2_HLA(HLA)
        HLA = self.BatchNorm_conv2_peptides(HLA)
        HLA = relu(HLA)
        HLA = self.maxpool2_HLA(HLA)
        HLA = torch.flatten(HLA,start_dim=1)
        

        # Combining the output
        combined_input = torch.cat((peptide, HLA), 1)
        x = self.L_in(combined_input)
        x = self.drop_out(x)
        x = relu(x)
        x = self.L_out(x)
        x = nn.Sigmoid()(x)
        return x
    

    def add_channel_dimension(self,encoded_sequence,length,encoding_dimension):
        return encoded_sequence.unsqueeze(1)
        # return encoded_sequence.reshape(-1, 1, length, encoding_dimension)



peptide_length = 10
encoding_dimensions = 12
HLA_length = 34

# define network
class Dense3layer(nn.Module):
    def __init__(self):
        super(Dense3layer, self).__init__()
        in_dimensions_L_in = peptide_length * encoding_dimensions + HLA_length * encoding_dimensions
        out_dimension_L_in = 264

        # Denselayer
        self.L_in = Linear(in_features=in_dimensions_L_in, # 528 if binding_score None, else 529
                            out_features= out_dimension_L_in)

        self.batchnorm1 = nn.BatchNorm1d(out_dimension_L_in)
        self.drop_out1 = nn.Dropout(p=0.4)


        out_dimensions_L_2 = 396
        self.L_2 = Linear(in_features = in_dimensions_L_in + out_dimension_L_in,
                            out_features = out_dimensions_L_2)

        self.batchnorm2 = nn.BatchNorm1d(out_dimensions_L_2)
        self.drop_out2 = nn.Dropout(p=0.4)


        out_dimensions_L_3 = 594 
        self.L_3 = Linear(in_features = in_dimensions_L_in + out_dimension_L_in + out_dimensions_L_2,
                            out_features = out_dimensions_L_2)

        self.batchnorm3 = nn.BatchNorm1d(out_dimensions_L_2)
        self.drop_out3 = nn.Dropout(p=0.4)

        self.L_out = Linear(in_features = out_dimensions_L_2,
                            out_features = 1)
    
    def forward(self, peptide, HLA, binding_score=None): # x.size() = [batch, channel, height, width]

        # Encoding the peptide
        peptide = torch.flatten(peptide,start_dim=1)

        # Encoding the HLA
        HLA = torch.flatten(HLA,start_dim=1)

        if binding_score is None:
            combined_input = torch.cat((peptide, HLA), 1)
      
        else:
            try:
                combined_input = torch.cat((peptide, HLA,binding_score), 1)
            except RuntimeError:
                print(binding_score.shape)
                print("peptide",peptide.shape)
                print("HLA",HLA.shape)
                sys.exit()

        x = self.L_in(combined_input)
        x = relu(x)
        x = self.batchnorm1(x)
        x = self.drop_out1(x)

        x_resnet1 = torch.cat((x, combined_input), 1)
        x = self.L_2(x_resnet1)
        x = relu(x)
        x = self.batchnorm2(x)
        x = self.drop_out2(x)

        x_resnet2 = torch.cat((x, x_resnet1), 1)
        x = self.L_3(x_resnet2)
        x = relu(x)
        x = self.batchnorm3(x)
        x = self.drop_out3(x)
        x = self.L_out(x)
        return torch.sigmoid(x)



if __name__ == "__main__":
    net = Dense3layer()
    print(net)

    peptide_random = np.random.normal(0,1, (10, 10, 12)).astype('float32')
    peptide_random = Variable(torch.from_numpy(peptide_random))
    HLA_random = np.random.normal(0,1, (10, 34, 12)).astype('float32')
    HLA_random = Variable(torch.from_numpy(HLA_random))
    binding_random = np.random.normal(0,1, (10, 1)).astype('float32')
    binding_random = Variable(torch.from_numpy(binding_random))

    output = net(peptide_random,HLA_random)
    print(output)

