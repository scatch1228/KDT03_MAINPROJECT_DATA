import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, step=2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class FlowTransformer(nn.Module):
    def __init__(self, input_dim=9, d_model=64, n_head=8, num_layers=3, output_dim=15, dropout=0.1):
        super(FlowTransformer, self).__init__()
        self.input_dim = input_dim
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_head, 
            dim_feedforward=d_model*4, 
            dropout=dropout,
            batch_first=False 
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers, enable_nested_tensor=False)
        
        self.decoder = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        x = self.input_projection(x) 
        x = x.permute(1, 0, 2) 
        
        x = self.pos_encoder(x)
        
        output = self.transformer_encoder(x)
        
        last_step = output[-1, :, :] 
        
        return self.decoder(last_step)