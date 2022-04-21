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


# define network
class FFN_3_attention(nn.Module):

    def __init__(self):
        super(FFN_3_attention, self).__init__()

        # Attention layer
        self.peptide_attention = nn.MultiheadAttention(embed_dim=12,num_heads=1,dropout=0.4)
        self.HLA_atttention = nn.MultiheadAttention(embed_dim=12,num_heads=1,dropout=0.4)

        # Denselayer
        self.L_in = Linear(in_features=529, # 528 if binding_score None, else 529
                            out_features= 264)

        self.batchnorm1 = nn.BatchNorm1d(264)
        self.drop_out1 = nn.Dropout(p=0.4)

        self.L_2 = Linear(in_features = 793,
                            out_features = 396)

        self.batchnorm2 = nn.BatchNorm1d(396)
        self.drop_out2 = nn.Dropout(p=0.4)

        self.L_3 = Linear(in_features = 1189,
                            out_features = 594)

        self.batchnorm3 = nn.BatchNorm1d(594)
        self.drop_out3 = nn.Dropout(p=0.4)

        self.L_out = Linear(in_features = 594,
                            out_features = 2)
    
    def forward(self, peptide, HLA, binding_score=None): # x.size() = [batch, channel, height, width]
         # Context embedding of peptides and HLA
        # context_embedded_peptide, attn_weights_peptide = self.peptide_attention(peptide,peptide,peptide)
        # context_embedded_hla, attn_weights_HLA = self.HLA_atttention(HLA,HLA,HLA)
        # context_embedded_peptide = context_embedded_peptide.reshape(context_embedded_peptide.shape[0],1,context_embedded_peptide.shape[1],context_embedded_peptide.shape[2])
        # context_embedded_hla = context_embedded_hla.reshape(context_embedded_hla.shape[0],1,context_embedded_hla.shape[1],context_embedded_hla.shape[2])


        # Encoding the peptide
        context_embedded_peptide = torch.flatten(peptide,start_dim=1)

        # Encoding the HLA
        context_embedded_hla = torch.flatten(HLA,start_dim=1)

        if binding_score is None:
            combined_input = torch.cat((context_embedded_peptide, context_embedded_hla ), 1)
      
        else:
            combined_input = torch.cat((context_embedded_peptide, context_embedded_hla ,binding_score), 1)

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
        x = relu(x)
        return softmax(x, dim=1)
