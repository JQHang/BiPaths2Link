import torch
import itertools
import numpy as np
import pandas as pd

def _transform_metapaths(dataset_config, batch_df):
    """将DataFrame转换为模型需要的格式"""
    # 初始化返回数据的字典
    batch_torch = {}

    # 处理标签列，infer时require_labels为False
    if dataset_config["require_labels"]:
        label_col = dataset_config["label_column"]
        batch_torch["label"] = torch.FloatTensor(np.vstack(batch_df[label_col].to_numpy()))

    # 检查是否需要保留各个样本对应的id列
    if dataset_config["require_ids"]:
        batch_torch["ids"] = batch_df[dataset_config["id_columns"]]

    # 依次记录各token类型对应的特征
    # 先直接拼接，之后可以考虑通过id列来去重，那就是有一个id到feat_index表
    batch_torch["feats"] = {}

    # 依次记录各pair_node本身对应的token以及对应的各个path对应的token
    batch_torch["pair_node"] = {}
    
    #########################################################################
    # 获得样本数量
    sample_count = batch_df.shape[0]

    # 获得各个样本对应的index
    sample_indices = np.arange(sample_count)

    #########################################################################################
    # 依次处理pair上两个目标节点对应的信息
    pair_node_to_token_type = {}
    pair_node_to_token_index = {}
    for pair_node_type in ["head_node", "tail_node"]:
        # 记录该节点对应的相关信息
        batch_torch["pair_node"][pair_node_type] = {}
        
        # 获得结点类型
        token_type = dataset_config[f"{pair_node_type}_token_config"]["token_type"]
        batch_torch["pair_node"][pair_node_type]["node_token_type"] = token_type
        
        # 获得结点特征
        feat_col = dataset_config[f"{pair_node_type}_token_config"]["feat_col"]
        feat_np = np.vstack(batch_df[feat_col].to_numpy())
        
        # 记录该节点的具体特征到对应的节点类型
        if token_type not in batch_torch["feats"]:
            batch_torch["feats"][token_type] = feat_np
        else:
            batch_torch["feats"][token_type] = np.vstack([batch_torch["feats"][token_type], feat_np])

        # 记录各个样本的该节点对应的特征的token的index
        index_start = batch_torch["feats"][token_type].shape[0] - feat_np.shape[0]
        index_end = batch_torch["feats"][token_type].shape[0]
        batch_torch["pair_node"][pair_node_type]["node_token_index"] = np.arange(index_start, index_end)[:, np.newaxis]

        # 获得要使用的路径方向
        if pair_node_type == "head_node":
            path_dir_type = "forward_paths"
        elif pair_node_type == "tail_node":
            path_dir_type = "backward_paths"
        
        # 依次处理各个path
        batch_torch["pair_node"][pair_node_type]["meta_paths"] = {}
        for path_name in dataset_config[path_dir_type]:
            # 记录该元路径相关结果
            batch_torch["pair_node"][pair_node_type]["meta_paths"][path_name] = {}
            
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

            # 获得各行最大clt到的path的数目
            max_clt_count = path_config["collect_records_count"]
            
            # 获得各个路径对应的目标节点的index 
            path_to_sample_index = np.repeat(sample_indices, clt_count_np)

            # 获得实际clt到的path数目
            clt_path_count = len(path_to_sample_index)

            # 检测是否有clt到的path实例，如果没有则直接跳过
            if clt_path_count == 0:
                batch_torch["pair_node"][pair_node_type]["meta_paths"][path_name]["path_to_node_index"] = None
                batch_torch["pair_node"][pair_node_type]["meta_paths"][path_name]["path_to_token_index"] = None
                continue

            # 记录path对应的node的index，之后path_index直接arange生成就行
            batch_torch["pair_node"][pair_node_type]["meta_paths"][path_name]["path_to_node_index"] = torch.LongTensor(path_to_sample_index)
            
            # 依次处理路径包含的剩下的各个token
            path_to_token_type = []
            path_to_feat_index = []
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
                    feat_np = np.vstack(flatten_feat)
                    
                    # 记录包含的特征到对应的token中
                    if token_type not in batch_torch["feats"]:
                        # 直接设定
                        batch_torch["feats"][token_type] = feat_np
                    else:
                        # 和现有的拼接
                        batch_torch["feats"][token_type] = np.vstack([batch_torch["feats"][token_type], feat_np])

                    # 记录对应的序号
                    index_start = batch_torch["feats"][token_type].shape[0] - feat_np.shape[0]
                    index_end = batch_torch["feats"][token_type].shape[0]
                    path_to_feat_index.append(np.arange(index_start, index_end)[:, np.newaxis])
                else:
                    # 记录该token类型对应的特征为None
                    batch_torch["feats"][token_type] = None
                    
                    # 该token没特征就将index都记录为0，会为该类型的token统一创建一个embed
                    path_to_feat_index.append(np.full((clt_path_count, 1), 0))

            # 记录最终结果
            batch_torch["pair_node"][pair_node_type]["meta_paths"][path_name]["path_to_token_type"] = path_to_token_type
            batch_torch["pair_node"][pair_node_type]["meta_paths"][path_name]["path_to_token_index"] = np.hstack(path_to_feat_index)

    ######################################################################################### 
    # 修正各类型feat_index对应的token index
    feat_token_cum_count = 0
    feat_token_cum_index = {}
    for token_type in batch_torch["feats"]:
        feat_token_cum_index[token_type] = feat_token_cum_count
        
        if batch_torch["feats"][token_type] is not None:
            # 记录特征
            batch_torch["feats"][token_type] = torch.FloatTensor(batch_torch["feats"][token_type])

            # 累加特征总数
            feat_token_cum_count = feat_token_cum_count + batch_torch["feats"][token_type].shape[0]
        else:
            # 累加特征总数
            feat_token_cum_count = feat_token_cum_count + 1

    ######################################################################################### 
    # 修正各个path对应的token_index
    for pair_node_type in batch_torch["pair_node"]:
        # 获得node对应的token_type
        token_type = batch_torch["pair_node"][pair_node_type]["node_token_type"]
        
        # 先修正节点本身的index
        node_token_index = batch_torch["pair_node"][pair_node_type]["node_token_index"] + feat_token_cum_index[token_type]
        node_token_index = torch.LongTensor(node_token_index)
        batch_torch["pair_node"][pair_node_type]["node_token_index"] = node_token_index
        
        # 再修正各个路径的index
        for path_name in batch_torch["pair_node"][pair_node_type]["meta_paths"]:
            if batch_torch["pair_node"][pair_node_type]["meta_paths"][path_name]["path_to_token_index"] is None:
                continue
            
            # 按列依次累加对应的cum_count
            for token_i in range(len(batch_torch["pair_node"][pair_node_type]["meta_paths"][path_name]["path_to_token_type"])):
                token_type = batch_torch["pair_node"][pair_node_type]["meta_paths"][path_name]["path_to_token_type"][token_i]

                path_token_i_index = batch_torch["pair_node"][pair_node_type]["meta_paths"][path_name]["path_to_token_index"][:, token_i]
                path_token_i_index= path_token_i_index + feat_token_cum_index[token_type]
                batch_torch["pair_node"][pair_node_type]["meta_paths"][path_name]["path_to_token_index"][:, token_i] = path_token_i_index

            # 转tensor
            path_to_token_index = torch.LongTensor(batch_torch["pair_node"][pair_node_type]["meta_paths"][path_name]["path_to_token_index"])
            batch_torch["pair_node"][pair_node_type]["meta_paths"][path_name]["path_to_token_index"] = path_to_token_index

    return batch_torch