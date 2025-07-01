import os
import random
import socket
import resource
import torch
import torch_npu
import torch.distributed as dist

def find_free_port(start=29000, end=29100):
    while True:
        try:
            sock = socket.socket()
            port = random.randint(start, end)
            sock.bind(('', port))
            sock.close()
            return port
        except OSError:
            continue

def setup_ddp(rank, world_size, port, device_type):
    """初始化DDP进程组"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)

    # 设置核心转储大小为0，防止中断后生成core文件
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    if device_type == "cuda":
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("hccl", rank=rank, world_size=world_size)
        
def dataset_to_device(batch_torch, rank):
    if "label" in batch_torch:
        batch_label = batch_torch["label"].cuda(rank)
    else:
        batch_label = None

    batch_feats = {}

    batch_feats["self"] = {}
    for node_type in batch_torch["feats"]["self"]:
        batch_feats["self"][node_type] = {}
        for node_table_name in batch_torch["feats"]["self"][node_type]:
            batch_feats["self"][node_type][node_table_name] = {}
            for scale_func in batch_torch["feats"]["self"][node_type][node_table_name]:
                feat_torch = batch_torch["feats"]["self"][node_type][node_table_name][scale_func].cuda(rank)
                batch_feats["self"][node_type][node_table_name][scale_func] = feat_torch

    batch_feats["join_edges"] = {}
    for join_edges_name in batch_torch["feats"]["join_edges"]:
        batch_feats["join_edges"][join_edges_name] = {}
        for seq_token_index in batch_torch["feats"]["join_edges"][join_edges_name]:
            batch_feats["join_edges"][join_edges_name][seq_token_index] = {}
            for table_name in batch_torch["feats"]["join_edges"][join_edges_name][seq_token_index]:
                batch_feats["join_edges"][join_edges_name][seq_token_index][table_name] = {}
                for scale_func in batch_torch["feats"]["join_edges"][join_edges_name][seq_token_index][table_name]:
                    feat_torch = batch_torch["feats"]["join_edges"][join_edges_name][seq_token_index][table_name][scale_func].cuda(rank)
                    batch_feats["join_edges"][join_edges_name][seq_token_index][table_name][scale_func] = feat_torch
    
    return batch_feats, batch_label