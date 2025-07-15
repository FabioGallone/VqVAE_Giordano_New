import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.residual import ResidualStack


class Encoder(nn.Module):
    """
    Original 2D Encoder for images.
    This is the q_theta (z|x) network. Given a data sample x q_theta 
    maps to the latent space x -> z.

    For a VQ VAE, q_theta outputs parameters of a categorical distribution.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Encoder, self).__init__()
        kernel = 4
        stride = 2
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_dim, h_dim // 2, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim // 2, h_dim, kernel_size=kernel,
                      stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(h_dim, h_dim, kernel_size=kernel-1,
                      stride=stride-1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers)
        )

    def forward(self, x):
        return self.conv_stack(x)


class ImprovedResidualBlock(nn.Module):
    """Improved residual block with GroupNorm for 1D calcium data"""
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(ImprovedResidualBlock, self).__init__()
        self._block = nn.Sequential(
            nn.GroupNorm(min(8, in_channels), in_channels),
            nn.ReLU(True),
            nn.Conv1d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(min(8, num_residual_hiddens), num_residual_hiddens),
            nn.ReLU(True),
            nn.Conv1d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)


class CalciumEncoder(nn.Module):
    """
    1D Encoder optimized for calcium imaging data.
    
    Inputs:
    - in_channels : number of neurons (input channels)
    - num_hiddens : hidden layer dimension
    - num_residual_layers : number of residual layers
    - num_residual_hiddens : hidden dimension in residual blocks
    - dropout_rate : dropout rate for regularization
    """

    def __init__(self, in_channels, num_hiddens, num_residual_layers, 
                 num_residual_hiddens, dropout_rate=0.1):
        super(CalciumEncoder, self).__init__()
        
        # Progressive downsampling with larger receptive fields
        self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                 out_channels=num_hiddens//4,
                                 kernel_size=7,
                                 stride=1, padding=3)
        self._conv_2 = nn.Conv1d(in_channels=num_hiddens//4,
                                 out_channels=num_hiddens//2,
                                 kernel_size=5,
                                 stride=2, padding=2)
        self._conv_3 = nn.Conv1d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=2, padding=1)
        
        # Improved residual stack
        self._residual_stack = nn.ModuleList([
            ImprovedResidualBlock(num_hiddens, num_hiddens, num_residual_hiddens)
            for _ in range(num_residual_layers)
        ])
        
        self._dropout = nn.Dropout1d(dropout_rate)

    def forward(self, inputs):
        x = F.relu(self._conv_1(inputs))
        x = self._dropout(x)
        x = F.relu(self._conv_2(x))
        x = self._dropout(x)
        x = self._conv_3(x)
        
        for block in self._residual_stack:
            x = block(x)
            
        return x


if __name__ == "__main__":
    # Test original encoder
    x_2d = torch.randn(3, 3, 32, 32)  # Batch, Channels, Height, Width
    encoder_2d = Encoder(3, 128, 2, 64)
    out_2d = encoder_2d(x_2d)
    print('2D Encoder out shape:', out_2d.shape)
    
    # Test calcium encoder
    x_1d = torch.randn(3, 50, 100)  # Batch, Neurons, Time
    encoder_1d = CalciumEncoder(50, 128, 3, 64)
    out_1d = encoder_1d(x_1d)
    print('Calcium Encoder out shape:', out_1d.shape)