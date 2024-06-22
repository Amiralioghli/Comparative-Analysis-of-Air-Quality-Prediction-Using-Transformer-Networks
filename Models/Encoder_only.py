import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

device = (
    "mps"
    if getattr(torch, "has_mps", False)
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

def scaled_dot_product_attention(q , k , v, mask = None):
    d_k = torch.tensor(q.shape[-1])
    scaled = torch.matmul(q , k.transpose(-1 , -2)) / math.sqrt(d_k)
    
    if mask is not None:
        scaled =scaled + mask
    attention = F.softmax(scaled , dim = -1)
    values = torch.matmul(attention , v)
    
    return values , attention

class Multihead_Attention(nn.Module):
    def __init__(self, d_model , num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)
        self.qkv_linear = nn.Linear(d_model, d_model)
        
    def forward(self , x , mask = None):
        
        batch_size , sequence_length , input_size = x.size()
        
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(batch_size , sequence_length , self.num_heads , 3 * self.head_dim)
        qkv = qkv.permute(0 , 2 , 1 , 3)
        q , k , v = qkv.chunk(3 , dim = -1)
        values , attention = scaled_dot_product_attention(q, k, v , mask)
        values = values.reshape(batch_size , sequence_length , self.num_heads * self.head_dim)
        out = self.qkv_linear(values)
        return out , attention


def get_angles(pos, i, D):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(D))
    return pos * angle_rates


def positional_encoding(D, position=20, dim=3, device=device):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(D)[np.newaxis, :],
                            D)
    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    if dim == 3:
        pos_encoding = angle_rads[np.newaxis, ...]
    elif dim == 4:
        pos_encoding = angle_rads[np.newaxis,np.newaxis,  ...]
    return torch.tensor(pos_encoding, device=device)

def create_look_ahead_mask(size1, size2, device=device):
    mask = torch.ones((size1, size2), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask  # (size, size)

class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape=parameters_shape
        self.eps=eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta =  nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std
        out = self.gamma * y  + self.beta
        return out

class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    

class Encoder_Layer(nn.Module):
    def __init__(self, D, H, hidden_mlp_dim, dropout_rate):
        super(Encoder_Layer, self).__init__()
        self.dropout_rate = dropout_rate
        
        self.ffn = PositionwiseFeedForward(d_model=D , hidden=hidden_mlp_dim)
        
        self.layernorm1 = LayerNormalization(parameters_shape=[D], eps=1e-9)
        self.layernorm2 = LayerNormalization(parameters_shape=[D], eps=1e-9)
        
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.mha = Multihead_Attention(d_model = D, num_heads = H)
    

    def forward(self, x, look_ahead_mask = None):
        
        attn, attn_weights = self.mha(x, mask = look_ahead_mask)  # (B, S, D)
        attn = self.dropout1(attn) # (B,S,D)
        attn = self.layernorm1(attn + x) # (B,S,D)
        
        mlp_act = self.ffn(attn)
        
        output = self.layernorm2(mlp_act + attn)  # (B, S, D)

        return output, attn_weights


class Encoder(nn.Module):
    
    def __init__(self, num_layers, D, H, hidden_mlp_dim, inp_features, out_features, dropout_rate):
        super(Encoder, self).__init__()
        self.sqrt_D = torch.tensor(math.sqrt(D))
        self.num_layers = num_layers
        self.input_projection = nn.Linear(inp_features, D)
        self.output_projection = nn.Linear(D, out_features)
        self.pos_encoding = positional_encoding(D)
        self.enc_layers = nn.ModuleList([Encoder_Layer(D, H, hidden_mlp_dim, 
                                        dropout_rate=dropout_rate
                                       ) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask = None):
        B, S, D = x.shape
        attention_weights = {}
        x = self.input_projection(x)
        x *= self.sqrt_D
        
        x += self.pos_encoding[:, :S, :]

        x = self.dropout(x)

        for i in range(self.num_layers):
            x, block = self.enc_layers[i](x=x,
                                          look_ahead_mask=mask)
            attention_weights['encoder_layer{}'.format(i + 1)] = block
        
        x = self.output_projection(x[:,:4,:])
        
        return x, attention_weights # (B,S,S)
    
    