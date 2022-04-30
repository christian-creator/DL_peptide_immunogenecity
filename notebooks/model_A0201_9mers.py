from unicodedata import bidirectional
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

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
    
# define network
class Dense_layer_3_highest(nn.Module):
    def __init__(self):
        super(Dense_layer_3_highest, self).__init__()

        # Denselayer
        in_dimensions_L_in = encoding_dimensions*peptide_length + encoding_dimensions*HLA_length
        out_dimension_L_in = int(in_dimensions_L_in/2)
        self.drop_out = nn.Dropout(p=0.4)

        self.L_in = Linear(in_features=in_dimensions_L_in, # 528 if binding_score None, else 529
                            out_features= out_dimension_L_in)

        self.batchnorm1 = nn.BatchNorm1d(out_dimension_L_in)

        out_dimensions_L_2 = int(out_dimension_L_in/2)
        self.L_2 = Linear(in_features =  out_dimension_L_in,
                            out_features = out_dimensions_L_2)

        self.batchnorm2 = nn.BatchNorm1d(out_dimensions_L_2)

        out_dimensions_L_3 = int(out_dimensions_L_2/2)
        self.L_3 = Linear(in_features =  out_dimensions_L_2 + out_dimension_L_in,
                            out_features = 1)
        
        # self.batchnorm3 = nn.BatchNorm1d(out_dimensions_L_3)

        # out_dimensions_L_4 = int(out_dimension_L_in/2)
        
        # self.L_4 = Linear(in_features =  out_dimensions_L_3 + out_dimensions_L_2 + out_dimension_L_in,
        #                     out_features = 1)



        # self.drop_out2 = nn.Dropout(p=0.4)
    
    def forward(self, peptide, HLA, binding_score=None): # x.size() = [batch, channel, height, width]
        peptide = torch.flatten(peptide,start_dim=1)
        HLA = torch.flatten(HLA,start_dim=1)

        # Encoding the peptide
        # peptide = torch.flatten(peptide,start_dim=1)
        # rnn_peptide, (hn_peptide, cn_peptide) = self.peptide_encoding(peptide)
        # peptide = torch.flatten(rnn_peptide,start_dim=1)

        # # Encoding the HLA
        # rnn_HLA, (hn, cn) = self.peptide_encoding(HLA)
        # HLA = torch.flatten(rnn_HLA,start_dim=1)

        if binding_score is not None: 
            combined_input = torch.cat((peptide, HLA, binding_score), 1)
        else:
            combined_input = torch.cat((peptide, HLA), 1)

        # x = self.batchnorm_rnn(combined_input)
        # x = self.drop_out(combined_input)
        l1_output = self.L_in(combined_input)
        l1_output = self.batchnorm1(l1_output)
        l1_output = self.drop_out(l1_output)
        l1_output = relu(l1_output)


        l2_output = self.L_2(l1_output)
        l2_output = relu(l2_output)
        l2_output = self.batchnorm2(l2_output)
        l2_output = self.drop_out(l2_output)

        combined_input_l3 = torch.cat((l2_output, l1_output), 1)
        x = self.L_3(combined_input_l3)
        # l3_output = relu(l3_output)
        # l3_output = self.batchnorm3(l3_output)
        # l3_output = self.drop_out(l3_output)


        # combined_input_l4 = torch.cat((l3_output, l2_output, l1_output), 1)
        # x = self.L_4(combined_input_l4)


        # x = self.L_4(x)
        # x_resnet2 = torch.cat((x, x_resnet1), 1)
        # x = self.L_3(x_resnet2)
        # x = relu(x)
        # x = self.batchnorm3(x)
        # x = self.drop_out3(x)
        # x = self.L_out(x)
        return torch.sigmoid(x)
