import torch
import itertools
import numpy as np
import pandas as pd

def _transform_bipaths(dataset_config, batch_df):
    """将DataFrame转换为模型需要的格式"""
    assert batch_df.shape[0] > 0
    
    batch_torch = {}

    # 处理标签列，infer时require_labels为False
    if dataset_config["require_labels"]:
        label_col = dataset_config["label_column"]
        batch_torch["label"] = torch.FloatTensor(np.vstack(batch_df[label_col].to_numpy()))

    # 检查是否需要保留各个样本对应的id列
    if dataset_config["require_ids"]:
        batch_torch["ids"] = batch_df[dataset_config["id_columns"]]

    # 获得样本数量
    sample_count = batch_df.shape[0]

    # 获得各个样本对应的index
    sample_indices = np.arange(sample_count)

    # 获得sample数量对应的维度的全1数组
    sample_count_ones = np.ones((sample_count, 1))
    
    # 依次记录各token类型对应的特征
    # 先直接拼接，之后可以考虑通过id列来去重，那就是有一个id到feat_index表
    batch_torch["feats"] = {}

    # 记录各个path中的各feat对应的特征表中的index
    path_to_feat_index_list = []

    # 记录各个path中的各token对应的特征类型
    # 以后需要考虑到一个feat可能对应多个token的情况来生成对应的token_index
    # 还要连带着生成对应的token_pos_index
    path_to_token_type_list = []

    # 记录样本对应的各个类型的path的instance数目，前两列为head_node和tail_node，每个样本对应1个
    sample_to_path_count = np.ones((sample_count, 2), dtype=int)
    
    #########################################################################################
    # 依次处理pair上两个目标节点对应的信息
    pair_node_to_token_type = {}
    pair_node_to_token_index = {}
    for pair_node_type in ["head_node", "tail_node"]:
        # 获得结点类型
        token_type = dataset_config[f"{pair_node_type}_token_config"]["token_type"]
        pair_node_to_token_type[pair_node_type] = token_type
        
        # 获得结点特征
        feat_col = dataset_config[f"{pair_node_type}_token_config"]["feat_col"]
        feat_tensor = torch.FloatTensor(np.vstack(batch_df[feat_col].to_numpy()))
        
        # 记录该节点的具体特征到对应的节点类型
        if token_type not in batch_torch["feats"]:
            batch_torch["feats"][token_type] = feat_tensor
        else:
            batch_torch["feats"][token_type] = torch.vstack([batch_torch["feats"][token_type], feat_tensor])

        # 记录各个样本的该节点对应的特征的token的index
        index_start = batch_torch["feats"][token_type].shape[0] - feat_tensor.shape[0]
        index_end = batch_torch["feats"][token_type].shape[0]
        pair_node_to_token_index[pair_node_type] = torch.arange(index_start, index_end)[:, np.newaxis]

        # 获得光由该节点组成的路径对应的token_index，并记录各token类型
        if pair_node_type == "head_node":
            path_to_feat_index = sample_count_ones * 0
        else:
            path_to_feat_index = sample_count_ones * 1
        
        path_to_feat_index = np.hstack((path_to_feat_index, pair_node_to_token_index[pair_node_type]))

        path_to_feat_index_list.append(path_to_feat_index)
        path_to_token_type_list.append(["cls", token_type])

    #########################################################################################
    # 依次处理各类型路径，记录各个路径对应的token_index，以及各个sample对应的path_index
    for path_dir_type in ["forward_paths", "backward_paths", "bipaths"]:
        # 依次处理各个数据对应的具体类型
        for path_name in dataset_config[path_dir_type]:
            # 获得该path的config
            path_config = dataset_config[path_dir_type][path_name]

            # 获得该路径clt到的列的前缀
            clt_col_prefix = path_config["clt_col_prefix"]
            
            # 获得记录各行实际collect到的path的数据的列
            clt_count_col = path_config["clt_count_col"]
    
            # 获得各行实际collect到的path的数目
            clt_count_np = np.vstack(batch_df[clt_count_col].to_numpy()).ravel()
            
            # 获得有实际collect到path的sample的行mask
            valid_clt_mask = clt_count_np > 0
            
            # 获得各个路径对应的目标节点的index 
            path_to_sample_index = np.repeat(sample_indices, clt_count_np)

            # 获得实际clt到的path数目
            clt_path_count = len(path_to_sample_index)

            # 检测是否有clt到的path实例，如果没有则直接跳过
            if clt_path_count == 0:
                continue

            # 将这个路径类型的count添加到sample_to_path_count矩阵
            sample_to_path_count = np.hstack([sample_to_path_count, clt_count_np.reshape(-1, 1)])
            
            # 获得path数量对应的维度的全1数组
            path_count_ones = np.ones((clt_path_count, 1))
            
            # 先基于路径类型添加cls位的index，以及头节点的index，并开始保存各个位的token类型
            path_to_token_type = ["cls"]
            if path_dir_type == "forward_paths":
                path_to_feat_index = path_count_ones * 0
                path_to_feat_index = np.hstack((path_to_feat_index, 
                                                pair_node_to_token_index["head_node"][path_to_sample_index]))

                path_to_token_type.append(pair_node_to_token_type["head_node"])
                
            elif path_dir_type == "backward_paths":
                path_to_feat_index = path_count_ones * 1
                path_to_feat_index = np.hstack((path_to_feat_index, 
                                                pair_node_to_token_index["tail_node"][path_to_sample_index]))
                
                path_to_token_type.append(pair_node_to_token_type["tail_node"])
                
            elif path_dir_type == "bipaths":
                path_to_feat_index = path_count_ones * 2
                path_to_feat_index = np.hstack((path_to_feat_index, 
                                                pair_node_to_token_index["head_node"][path_to_sample_index]))
                
                path_to_token_type.append(pair_node_to_token_type["head_node"])
                
            # 依次处理路径包含的各个token
            for token_config in path_config["seq_tokens"]:
                # 获得对应的token类型，并记录
                token_type = token_config["token_type"]
                path_to_token_type.append(token_type)

                # 检查该token是否有对应特征
                if token_config["feat_col"] is not None:
                    token_feat_col = token_config["feat_col"]
                    clt_feat_col = f"{clt_col_prefix}_clt_{token_feat_col}"
                    
                    # 获得有效特征
                    flatten_feat = list(itertools.chain.from_iterable(batch_df[valid_clt_mask][clt_feat_col].values))
                    feat_tensor = torch.FloatTensor(np.vstack(flatten_feat))
                    
                    # 记录包含的特征到对应的token中
                    if token_type not in batch_torch["feats"]:
                        # 直接设定
                        batch_torch["feats"][token_type] = feat_tensor
                    else:
                        # 和现有的拼接
                        batch_torch["feats"][token_type] = torch.vstack([batch_torch["feats"][token_type], 
                                                                         feat_tensor])

                    # 记录对应的序号
                    index_start = batch_torch["feats"][token_type].shape[0] - feat_tensor.shape[0]
                    index_end = batch_torch["feats"][token_type].shape[0]
                    path_to_feat_index = np.hstack((path_to_feat_index, 
                                                    torch.arange(index_start, index_end)[:, np.newaxis]))
                else:
                    # 记录该token类型对应的特征为None
                    batch_torch["feats"][token_type] = None
                    
                    # 该token没特征就将index都记录为0，会为该类型的token统一创建一个embed
                    path_to_feat_index = np.hstack((path_to_feat_index, 
                                                    torch.full((clt_path_count, 1), 0, dtype=torch.long)))
            
            # 如果是bipath，再添加尾结点对应的index
            if path_dir_type == "bipaths":
                path_to_feat_index = np.hstack((path_to_feat_index, 
                                                pair_node_to_token_index["tail_node"][path_to_sample_index]))
                
                path_to_token_type.append(pair_node_to_token_type["tail_node"])

            # 记录最终的path_to_feat_index和path_to_token_type
            path_to_feat_index_list.append(path_to_feat_index)
            path_to_token_type_list.append(path_to_token_type)
            
    ######################################################################################### 
    # 修正各类型token index要累加的index值 
    feat_token_cum_index = {"cls": 1}
    feat_token_cum_count = 4
    for token_type in batch_torch["feats"]:
        feat_token_cum_index[token_type] = feat_token_cum_count
        
        if batch_torch["feats"][token_type] is not None:
            feat_token_cum_count = feat_token_cum_count + batch_torch["feats"][token_type].shape[0]
        else:
            feat_token_cum_count = feat_token_cum_count + 1

    ######################################################################################### 
    # 获得各个path拥有的最大token数量
    path_max_token_count = 1 + dataset_config["bipath_max_seq_len"]
    
    # 修正各个path对应的token_index
    path_to_token_index_list = []
    for path_i in range(len(path_to_feat_index_list)):
        # 先取原始的feat_index作为初始值
        path_to_token_index = path_to_feat_index_list[path_i]

        # 按列依次累加对应的cum_count
        for token_i in range(len(path_to_token_type_list[path_i])):
            token_type = path_to_token_type_list[path_i][token_i]

            path_to_token_index[:, token_i] = path_to_token_index[:, token_i] + feat_token_cum_index[token_type]

        # 补全path到统一长度
        padding = ((0, 0), (0, path_max_token_count - path_to_token_index.shape[1]))
        path_to_token_index = np.pad(path_to_token_index, padding, mode='constant', constant_values=0)
        
        # 记录结果
        path_to_token_index_list.append(path_to_token_index)
    
    # 合并为最终结果
    path_to_token_index = np.vstack(path_to_token_index_list)
    batch_torch["path_to_token_index"] = torch.LongTensor(path_to_token_index)

    #########################################################################################
    # 计算每个样本的总路径数
    total_paths_per_sample = np.sum(sample_to_path_count, axis=1)
    max_total_paths = np.max(total_paths_per_sample)
    
    # 一次性分配sample_to_path_index矩阵
    sample_to_path_index = np.zeros((sample_count, 1 + max_total_paths), dtype=int)

    # 添加cls位
    sample_to_path_index[:, 0] = 1
    
    # 记录每个样本当前填充到的列位置
    sample_path_start_col = sample_count_ones

    # 创建列位置矩阵：(1, max_total_paths)
    col_positions = np.arange(1 + max_total_paths)[np.newaxis, :]
    
    # 累计path计数器，一个是pad的0，一个是cls
    sample_path_cum_count = 2
    
    # 按照路径类型的顺序填充sample_to_path_index
    for col_idx in range(sample_to_path_count.shape[1]):
        # 当前路径类型各样本的path数量
        current_sample_path_count = sample_to_path_count[:, col_idx]

        # 获得各个样本对应的path的终止列
        sample_path_end_col = sample_path_start_col + current_sample_path_count[:, np.newaxis] 

        # 生成mask：判断每个位置是否在当前路径类型的范围内
        # (sample_count, max_total_paths) 的mask矩阵
        valid_positions = (col_positions >= sample_path_start_col) & (col_positions < sample_path_end_col)
        
        # 计算当前路径类型的总path数
        current_path_count = np.sum(current_sample_path_count)

        # 获得各个有效位置对应的path_index
        valid_path_index = np.arange(sample_path_cum_count, sample_path_cum_count + current_path_count)

        # 有效位置填充path_index
        sample_to_path_index[valid_positions] = valid_path_index
        
        # 更新path对应的起始列
        sample_path_start_col = sample_path_end_col

        # 更新累积的path总数
        sample_path_cum_count = sample_path_cum_count + current_path_count
    
    batch_torch["sample_to_path_index"] = torch.LongTensor(sample_to_path_index)
    
    return batch_torch

def bipaths_dataset_to_device(batch_torch, rank, device_type):
    # 获得该rank具体的device名称 
    device = f'{device_type}:{rank}'
    
    if "label" in batch_torch:
        batch_label = batch_torch["label"].to(device)
    else:
        batch_label = None

    batch_input = {}

    # 先转化特征数据
    batch_input["feats"] = {}
    for token_type in batch_torch["feats"]:
        if batch_torch["feats"][token_type] is not None:
            batch_input["feats"][token_type] = batch_torch["feats"][token_type].to(device)
        else:
            batch_input["feats"][token_type] = None

    # 转化路径到token_index的对应关系
    batch_input["path_to_token_index"] = batch_torch["path_to_token_index"].to(device)

    # 转化路径到token_index的对应关系
    batch_input["sample_to_path_index"] = batch_torch["sample_to_path_index"].to(device)
            
    return batch_input, batch_label