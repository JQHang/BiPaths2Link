from .feat_proj import FeatureProjection
from .tokens_encoder import TokensEncoder
from .mlp import MLP

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

# 获得logger
logger = logging.getLogger(__name__)

class BiPathsNN(nn.Module):
    def __init__(self, bipathsnn_config, dataset_config):
        super().__init__()

        # 记录数据集的相关信息，之后模型运行时也需要用
        self.dataset_config = dataset_config
        
        # 获得序列化后的token对应的embed维度，方便之后设定模型和embed
        self.token_dim = bipathsnn_config["feat_proj"]["batch_output_dim"]
        
        # 创建对各类型原始特征进行映射的模块
        self.feat_proj_dict = nn.ModuleDict()
        for token_type in dataset_config["token_feat_len"]:
            feat_len = dataset_config["token_feat_len"][token_type]

            # 检查是否有特征值
            if feat_len > 0:
                # 初始化该特征映射模块
                feat_proj = nn.Linear(feat_len, self.token_dim)
    
                # 记录特征映射模块
                self.feat_proj_dict[token_type] = feat_proj
            else:
                self.feat_proj_dict[token_type] = nn.Embedding(1, self.token_dim)

        # 设定path_pad_cls_token_embed
        self.path_pad_cls_token_embed = nn.Embedding(4, self.token_dim)

        # 基于最大的token数目创建所需的pos_embed 
        self.token_pos_embed = nn.Embedding(1 + dataset_config["bipath_max_seq_len"], self.token_dim)

        # 可以对path_token_embed先进行drop out
        self.token_embed_dropout = nn.Dropout(bipathsnn_config["join_edges_encoder"]["dropout"])
        
        # 创建path encoder模块   
        self.join_edges_encoder = TokensEncoder(
                                    d_model = self.token_dim,
                                    nhead = bipathsnn_config["join_edges_encoder"]["nhead"],
                                    num_layers = bipathsnn_config["join_edges_encoder"]["num_layers"],
                                    dropout = bipathsnn_config["join_edges_encoder"]["dropout"],
                                )

        # 设定sample_pad_cls_token_embed
        self.sample_pad_cls_token_embed = nn.Embedding(2, self.token_dim)

        # 可以对path_embed先进行drop out
        self.path_embed_dropout = nn.Dropout(bipathsnn_config["join_edges_summarizer"]["dropout"])
        
        # 创建join_edges_summarizer模块
        # 针对instance数量过多的情况，可以考虑多层summarizer(这个可以等前面搜索代码都完成后再加)
        self.join_edges_cls_encoder = TokensEncoder(
                                        d_model = self.token_dim,
                                        nhead = bipathsnn_config["join_edges_summarizer"]["nhead"],
                                        num_layers = bipathsnn_config["join_edges_summarizer"]["num_layers"],
                                        dropout = bipathsnn_config["join_edges_summarizer"]["dropout"]
                                    )

        # 创建对sample_embed的归一化
        self.sample_norm = nn.LayerNorm(self.token_dim)
        
        # 创建预测结果输出模块
        self.output_proj = nn.Linear(self.token_dim, bipathsnn_config["output_proj"]["output_dim"])
        
    def forward(self, bipaths_batch):
        # 先依次对各类型token的特征进行映射
        token_embed_list = [self.path_pad_cls_token_embed.weight]
        for token_type in bipaths_batch["feats"]:
            # 检查该类型是否有对应特征
            feat_len = self.dataset_config["token_feat_len"][token_type]
            if feat_len > 0:
                # 对特征进行映射
                token_embed = self.feat_proj_dict[token_type](bipaths_batch["feats"][token_type])
                token_embed_list.append(token_embed)
            else:
                # 直接获得预设的embed
                token_embed_list.append(self.feat_proj_dict[token_type].weight)

        # 合并得到完整的token_embed表 
        token_embed = torch.vstack(token_embed_list)

        # 获得各个path对应的token_embed
        path_token_embed = token_embed[bipaths_batch["path_to_token_index"]] + self.token_pos_embed.weight.unsqueeze(0)

        # 对token_embed进行dropout
        path_token_embed = self.token_embed_dropout(path_token_embed)
        
        # 获得各个path对应的token_mask
        path_token_mask = bipaths_batch["path_to_token_index"] == 0
        
        # 进行encoder,只提取cls位的输出作为最终结果
        path_encode_embed = self.join_edges_encoder(path_token_embed, path_token_mask)
        path_encode_embed = path_encode_embed[:, 0, :]

        # 和pad_cls_token合并
        path_encode_embed = torch.vstack([self.sample_pad_cls_token_embed.weight, path_encode_embed])
        
        # 获得各个sample对应的path_embed
        sample_path_embed = path_encode_embed[bipaths_batch["sample_to_path_index"]]

        # 对path_embed进行dropout
        sample_path_embed = self.path_embed_dropout(sample_path_embed)
        
        # 获得各个sample对应的path_mask
        sample_path_mask = bipaths_batch["sample_to_path_index"] == 0
        
        # 综合各个token位的结果,只提取cls位的输出作为最终结果
        sample_embed = self.join_edges_cls_encoder(sample_path_embed, sample_path_mask)
        sample_embed = sample_embed[:, 0, :]

        # 对输出的表征进行归一化
        sample_embed = self.sample_norm(sample_embed)
        
        # 做出最终的预测
        sample_pred = self.output_proj(sample_embed)
        
        return sample_pred