import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F

import logging

# 获得logger
logger = logging.getLogger(__name__)

class TokensEncoder(nn.Module):
    def __init__(
        self,
        d_model,              # 模型维度
        nhead,                # 注意力头数
        num_layers,           # encoder层数
        dropout=0.1,          # dropout率
    ):
        super().__init__()
        
        # 创建encoder layer
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,      # 前馈神经网络总是使用d_model*4维度
            dropout=dropout,
            activation='gelu',
            batch_first=True,     # 总是使用batch_first=True更直观
            norm_first=True       # 总是使用Pre-LN
        )

        # 创建encoder
        self.transformer_encoder = TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=None,
            enable_nested_tensor=False
        )
        
    def forward(self, src, padding_mask=None):
        """
        参数:
            src: 输入序列 [batch_size, seq_len, d_model]
            padding_mask: 输入中的填充值标记 [batch_size]
            
        返回:
            output: 编码后的序列 [batch_size, seq_len, d_model]
        """        
        
        # transformer编码 
        output = self.transformer_encoder(src, src_key_padding_mask = padding_mask)
        
        return output
