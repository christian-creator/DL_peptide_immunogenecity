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

# hyperameters of the model
input_channels = 1
peptide_input_height = 10
hla_input_height = 366
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

peptide_length = 10
encoding_dimensions = 12
HLA_length = 366

# define network
class ConVbig(nn.Module):

    def __init__(self):
        super(ConVbig, self).__init__()

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
                            kernel_size=(60,encoding_dimension),
                            stride=1,
                            padding=0)
        self.BatchNorm_conv1_HLA = BatchNorm2d(16) # Output channels from the previous layer
        self.maxpool1_HLA = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1), padding=0)
        
        self.conv2_HLA = Conv2d(in_channels=16,
                            out_channels=32,
                            kernel_size=(30,1),
                            stride=1,
                            padding=0)
        self.BatchNorm_conv2_peptides = BatchNorm2d(32) # Output channels from the previous layer
        self.maxpool2_HLA = nn.MaxPool2d(kernel_size=(2,1), stride=(2,1), padding=0)

        # Denselayer
        self.L_in = Linear(in_features=2112,
                            out_features=704)
        
        self.drop_out = nn.Dropout(p=0.2)

        self.L_2 = Linear(in_features=704,
                            out_features=128)

        self.L_out = Linear(in_features=128,
                            out_features=1,
                            bias=False)


    def forward(self, peptide, HLA): # x.size() = [batch, channel, height, width]
        peptide = self.add_channel_dimension(peptide,peptide_input_height,encoding_dimension)
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
        x = self.L_2(x)
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
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        RNN_encoding_dim = 10
    
        # RNN encoding
        self.peptide_encoding = nn.LSTM(encoding_dimensions,RNN_encoding_dim, batch_first=True, num_layers = 1,bidirectional=True)
        self.hla_encoding = nn.LSTM(encoding_dimensions,RNN_encoding_dim, batch_first=True, num_layers = 1,bidirectional=True)

        # Denselayer
        in_dimensions_L_in = RNN_encoding_dim*peptide_length*2 + RNN_encoding_dim*HLA_length*2
        out_dimension_L_in = int(in_dimensions_L_in/2)

        self.L_in = Linear(in_features=in_dimensions_L_in, # 528 if binding_score None, else 529
                            out_features= out_dimension_L_in)

        self.batchnorm1 = nn.BatchNorm1d(out_dimension_L_in)
        self.drop_out = nn.Dropout(p=0.4)

        out_dimensions_L_2 = int(out_dimension_L_in/2)

        self.L_2 = Linear(in_features =  out_dimension_L_in,
                            out_features = 1)

        # self.batchnorm2 = nn.BatchNorm1d(out_dimensions_L_2)

        # out_dimensions_L_3 = int(out_dimension_L_in/2)
        # self.L_3 = Linear(in_features =  out_dimensions_L_2,
        #                     out_features = 1)



        # self.drop_out2 = nn.Dropout(p=0.4)
    
    def forward(self, peptide, HLA, binding_score=None): # x.size() = [batch, channel, height, width]

        # Encoding the peptide
        # peptide = torch.flatten(peptide,start_dim=1)
        rnn_peptide, (hn_peptide, cn_peptide) = self.peptide_encoding(peptide)
        peptide = torch.flatten(rnn_peptide,start_dim=1)

        # Encoding the HLA
        rnn_HLA, (hn, cn) = self.peptide_encoding(HLA)
        HLA = torch.flatten(rnn_HLA,start_dim=1)

        if binding_score is not None: 
            combined_input = torch.cat((peptide, HLA, binding_score), 1)
        else:
            combined_input = torch.cat((peptide, HLA), 1)

        # x = self.batchnorm_rnn(combined_input)
        # x = self.drop_out(combined_input)
        x = self.L_in(combined_input)
        x = self.batchnorm1(x)
        x = self.drop_out(x)
        x = relu(x)
        x = self.L_2(x)
        # x = relu(x)
        # x = self.batchnorm2(x)
        # x = self.drop_out(x)
        # x = self.L_3(x)


        # x_resnet2 = torch.cat((x, x_resnet1), 1)
        # x = self.L_3(x_resnet2)
        # x = relu(x)
        # x = self.batchnorm3(x)
        # x = self.drop_out3(x)
        # x = self.L_out(x)
        return torch.sigmoid(x)


peptide_length = 10
encoding_dimensions = 12
HLA_length = 34

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



peptide_length = 10
encoding_dimensions = 12
HLA_length = 34

# define network
class RNN_model_best(nn.Module):
    def __init__(self):
        super(RNN_model_best, self).__init__()
        # RNN
        RNN_encoding_dim = 10
        # self.peptide_attention = torch.nn.MultiheadAttention(12, num_heads=3, dropout=0.4, batch_first=True)
        self.peptide_encoding = nn.LSTM(encoding_dimensions, RNN_encoding_dim, batch_first=True, num_layers = 1, bidirectional=False)
        self.hla_encoding = nn.LSTM(encoding_dimensions, RNN_encoding_dim, batch_first=True, num_layers = 1,bidirectional=False)

        # Attention from encoding
        # Denselayer
        in_dimensions_L_in = RNN_encoding_dim*peptide_length + RNN_encoding_dim*HLA_length
        out_dimension_L_in = int(in_dimensions_L_in/2)
        self.drop_out = nn.Dropout(p=0.4)

        self.L_in = Linear(in_features = in_dimensions_L_in, # 528 if binding_score None, else 529
                            out_features= out_dimension_L_in)

        self.batchnorm1 = nn.BatchNorm1d(out_dimension_L_in)

        out_dimensions_L_2 = int(out_dimension_L_in/2)
        self.L_2 = Linear(in_features =  out_dimension_L_in + in_dimensions_L_in,
                            out_features = out_dimensions_L_2)

        self.batchnorm2 = nn.BatchNorm1d(out_dimensions_L_2)

        out_dimensions_L_3 = int(out_dimensions_L_2/2)
        self.L_3 = Linear(in_features =  out_dimensions_L_2,
                            out_features = 1)
        
        self.batchnorm3 = nn.BatchNorm1d(out_dimensions_L_3)
        # out_dimensions_L_4 = int(out_dimension_L_in/2)
        
        # self.L_4 = Linear(in_features =  out_dimensions_L_3,
        #                     out_features = 1)

        # self.drop_out2 = nn.Dropout(p=0.4)
    
    def forward(self, peptide, HLA, binding_score=None): # x.size() = [batch, channel, height, width]
        # Encoding RNN
        # context_embedded_peptide,attn_output_weights = self.peptide_attention(peptide,peptide,peptide)
        rnn_peptide, (hn_peptide, cn_peptide) = self.peptide_encoding(peptide)
        peptide = torch.flatten(rnn_peptide,start_dim=1)
        
        # Encoding HLA
        rnn_hla, (hn_hla, cn_hla) = self.peptide_encoding(HLA)
        HLA = torch.flatten(rnn_hla,start_dim=1)

        if binding_score is not None: 
            combined_input = torch.cat((peptide, HLA, binding_score), 1)
        else:
            combined_input = torch.cat((peptide, HLA), 1)


        L_1_act = self.L_in(combined_input)
        L_1_act = self.batchnorm1(L_1_act)
        L_1_act = self.drop_out(L_1_act)
        L_1_act = relu(L_1_act)
        
        l_2_input = torch.cat((L_1_act, combined_input), 1)
        L_2_act = self.L_2(l_2_input)
        L_2_act = relu(L_2_act)
        L_2_act = self.batchnorm2(L_2_act)
        L_2_act = self.drop_out(L_2_act)

        # l_3_input = torch.cat((L_2_act, L_1_act, combined_input), 1)
        x = self.L_3(L_2_act)
        # L_3_act = relu(L_3_act)
        # L_3_act = self.batchnorm3(L_3_act)
        # L_3_act = self.drop_out(L_3_act)

        # x = self.L_4(L_3_act)
        return torch.sigmoid(x)


peptide_length = 10
encoding_dimensions = 12
HLA_length = 34
input_channels = 1

# define network
class test_model(nn.Module):
    def __init__(self):
        super(test_model, self).__init__()
        out_channels_conv1_hla = 8 
        # CNN encoding - HLA
        self.conv1_hla = Conv2d(in_channels=input_channels,
                            out_channels=out_channels_conv1_hla,
                            kernel_size=(30,encoding_dimension),
                            stride=1,
                            padding=0)
        self.maxpool1_hla = nn.MaxPool2d(kernel_size=(15,1),
                                            stride=(2,1))
        
        self.BatchNorm_conv1_hla = BatchNorm2d(out_channels_conv1_hla) # Output channels from the previous layer

        out_channels_conv2_hla = 4
        self.conv2_hla = Conv2d(in_channels=out_channels_conv1_hla,
                            out_channels=out_channels_conv2_hla,
                            kernel_size=(30,1),
                            stride=1,
                            padding=0)
        self.maxpool2_hla = nn.MaxPool2d(kernel_size=(15,1),
                                            stride=(2,1))

        self.BatchNorm_conv2_hla = BatchNorm2d(out_channels_conv2_hla) # Output channels from the previous layer
        

        # CNN encoding - peptide
        # RNN_encoding_dim = 10
        # self.peptide_encoding = nn.LSTM(encoding_dimensions, RNN_encoding_dim, batch_first=True, num_layers = 1, bidirectional=False)
        # out_channels_conv1_peptide = 8 
        # self.conv1_peptide = Conv2d(in_channels=input_channels,
        #                     out_channels=out_channels_conv1_hla,
        #                     kernel_size=(1,encoding_dimension),
        #                     stride=1,
        #                     padding=0)
        # self.BatchNorm_conv1_peptide = BatchNorm2d(out_channels_conv1_peptide) # Output channels from the previous layer
        # self.maxpool1_hla = nn.MaxPool2d(kernel_size=(15,1),
        #                                     stride=(2,1))


        # Denselayer
        in_dimensions_L_in = peptide_length*encoding_dimension + 240
        out_dimension_L_in = int(in_dimensions_L_in/2)
        self.drop_out = nn.Dropout(p=0.4)

        self.L_in = Linear(in_features = in_dimensions_L_in, # 528 if binding_score None, else 529
                            out_features= out_dimension_L_in)

        self.batchnorm1 = nn.BatchNorm1d(out_dimension_L_in)

        out_dimensions_L_2 = int(out_dimension_L_in/2)
        self.L_2 = Linear(in_features =  out_dimension_L_in,
                            out_features = 1)

        # self.batchnorm2 = nn.BatchNorm1d(out_dimensions_L_2)

        # out_dimensions_L_3 = int(out_dimensions_L_2/2)
        # self.L_3 = Linear(in_features =  out_dimension_L_in + out_dimensions_L_2,
        #                     out_features = 1)
        
        # self.batchnorm3 = nn.BatchNorm1d(out_dimensions_L_3)
        # out_dimensions_L_4 = int(out_dimension_L_in/2)
        
        # self.L_4 = Linear(in_features =  out_dimensions_L_3,
        #                     out_features = 1)

        # self.drop_out2 = nn.Dropout(p=0.4)
    
    def forward(self, peptide, HLA, binding_score=None): # x.size() = [batch, channel, height, width]
        # Encoding RNN
        # context_embedded_peptide,attn_output_weights = self.peptide_attention(peptide,peptide,peptide)
        # rnn_peptide, (hn_peptide, cn_peptide) = self.peptide_encoding(peptide)
        # peptide = self.add_channel_dimension(peptide)
        # peptide = self.conv1_peptide(peptide)
        # peptide = self.BatchNorm_conv1_peptide(peptide)
        # peptide = relu(peptide)

        peptide = torch.flatten(peptide,start_dim=1)

        # feature extraction HLA
        HLA = self.add_channel_dimension(HLA)
        HLA = self.conv1_hla(HLA)
        HLA = self.BatchNorm_conv1_hla(HLA)
        HLA = relu(HLA)
        HLA = self.maxpool1_hla(HLA)

        HLA = self.conv2_hla(HLA)
        HLA = self.BatchNorm_conv2_hla(HLA)
        HLA = relu(HLA)
        HLA = self.maxpool2_hla(HLA)
        HLA = torch.flatten(HLA,start_dim=1)

        if binding_score is not None: 
            combined_input = torch.cat((peptide, HLA, binding_score), 1)
        else:
            combined_input = torch.cat((peptide, HLA), 1)


        L_1_act = self.L_in(combined_input)
        L_1_act = self.batchnorm1(L_1_act)
        L_1_act = self.drop_out(L_1_act)
        L_1_act = relu(L_1_act)
        
        # l_2_input = torch.cat((L_1_act, combined_input), 1)
        L_2_act = self.L_2(L_1_act)
        # L_2_act = self.batchnorm2(L_2_act)
        # L_2_act = self.drop_out(L_2_act)
        # L_2_act = relu(L_2_act)
        

        # l_3_input = torch.cat((L_2_act, L_1_act), 1)
        # l_3_act = self.L_3(l_3_input)
        # L_3_act = relu(L_3_act)
        # L_3_act = self.batchnorm3(L_3_act)
        # L_3_act = self.drop_out(L_3_act)

        # x = self.L_4(L_3_act)


        return torch.sigmoid(L_2_act)

    def add_channel_dimension(self,encoded_sequence):
        return encoded_sequence.unsqueeze(1)
    
    def compute_conv_dim(dim_size,kernel_size,padding,stride):
        return int((dim_size - kernel_size + 2 * padding) / stride + 1)

if __name__ == "__main__":
    net = test_model()
    print("Number of parameters in model:", get_n_params(net))
    # sys.exit(1)
    print(net)

    peptide_random = np.random.normal(0,1, (10, 10, 12)).astype('float32')
    peptide_random = Variable(torch.from_numpy(peptide_random))
    HLA_random = np.random.normal(0,1, (10, 366, 12)).astype('float32')
    HLA_random = Variable(torch.from_numpy(HLA_random))
    binding_random = np.random.normal(0,1, (10, 1)).astype('float32')
    binding_random = Variable(torch.from_numpy(binding_random))
    output = net(peptide_random,HLA_random)
    print(output)
