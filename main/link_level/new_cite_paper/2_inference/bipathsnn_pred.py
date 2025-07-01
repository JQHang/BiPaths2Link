#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch

device_type = "cuda"

# 获取可用的device数量
if device_type == "cuda":
    device_count = torch.cuda.device_count()
else:
    device_count = torch.npu.device_count()


# In[2]:


from joinminer.graph import TableGraph, join_edges_list_init, train_inst_config_init
from joinminer.python import mkdir, setup_logger, time_costing, read_json_file

from datetime import datetime
import random
import numpy as np


# In[3]:


# 获得项目文件夹根目录路径
from joinminer import PROJECT_ROOT

# 日志信息保存文件名
log_files_dir = PROJECT_ROOT + '/data/result_data/log_files/bipathsnn'
log_filename = log_files_dir + f'/{datetime.now().strftime("%Y-%m-%d-%H:%M")}.log'
mkdir(log_files_dir)

logger = setup_logger(log_filename, logger_name = "joinminer")

# Dataset config
dataset_config_file = PROJECT_ROOT + '/data/dataset/AMiner/bipathsnn_train/dataset_config.json'
dataset_config = read_json_file(dataset_config_file)
dataset_config["id_columns"] = ["index_0_node_Author_col_0", "index_1_node_Paper_col_0"]

# Dataset path
dataset_config["local_path"] = {}
dataset_config["local_path"]["infer"] = PROJECT_ROOT + '/data/dataset/AMiner/bipathsnn_infer/batch_id=3/sample_type=infer'


# In[4]:


# 设定BiPathsNN模型相关配置
bipathsnn_config = {}

bipathsnn_config["learning_rate"] = 3e-4
bipathsnn_config["weight_decay"] = 1e-2
bipathsnn_config["epochs"] = 100
bipathsnn_config["patience"] = 50

bipathsnn_config["data_loader"] = {}
bipathsnn_config["data_loader"]["train_batch_size"] = 128
bipathsnn_config["data_loader"]["train_epoch_size"] = 3e6
bipathsnn_config["data_loader"]["eval_batch_size"] = 256
bipathsnn_config["data_loader"]["eval_epoch_size"] = -1
bipathsnn_config["data_loader"]["num_workers"] = 8
bipathsnn_config["data_loader"]["prefetch_factor"] = 2

bipathsnn_config["feat_proj"] = {}
bipathsnn_config["feat_proj"]["batch_output_dim"] = 256

bipathsnn_config["join_edges_encoder"] = {}
bipathsnn_config["join_edges_encoder"]["nhead"] = 4
bipathsnn_config["join_edges_encoder"]["num_layers"] = 3
bipathsnn_config["join_edges_encoder"]["dropout"] = 0.1

bipathsnn_config["join_edges_summarizer"] = {}
bipathsnn_config["join_edges_summarizer"]["nhead"] = 4
bipathsnn_config["join_edges_summarizer"]["num_layers"] = 3
bipathsnn_config["join_edges_summarizer"]["dropout"] = 0.1

bipathsnn_config["output_proj"] = {}
bipathsnn_config["output_proj"]["output_dim"] = 1

import json
logger.info(f"BiPathsNN config: {json.dumps(bipathsnn_config, indent=2)}")


# In[5]:


from joinminer.python import mkdir
from datetime import datetime
import torch.multiprocessing as mp

# 设置本次实验相关配置
bipathsnn_config["experiment"] = {} 

# 设定模型参数的输出文件夹
bipathsnn_config["experiment"]["checkpoint"] = PROJECT_ROOT + f'/data/runs/AMiner/bipathsnn/4_npu_batch_128_lr_3_4_warm_decay_1_2_drop_1_dim_256_h_4_layer_3_neg_10'
os.makedirs(bipathsnn_config["experiment"]["checkpoint"], exist_ok=True)

logger.info(f"Read the experiment result from {bipathsnn_config['experiment']['checkpoint']}")


# In[7]:


from joinminer.python import read_json_file

import re

epoch_dirs = []
for epoch_dir in os.listdir(bipathsnn_config["experiment"]["checkpoint"]):
    if os.path.exists(bipathsnn_config["experiment"]["checkpoint"] + f"/{epoch_dir}/metric.json"):
        # 获得对应的epoch信息
        pattern = r'epoch_(\d+)'
        match = re.match(pattern, epoch_dir)
        epoch = int(match.group(1))

        # 获得对应的metric
        metrics = read_json_file(bipathsnn_config["experiment"]["checkpoint"] + f"/{epoch_dir}/metric.json")

        # 记录结果
        epoch_dirs.append((epoch, epoch_dir, metrics))
        
if len(epoch_dirs) > 0:
    epoch_dirs.sort(key=lambda x: x[2]["epoch_val_metrics"]["pr_auc"])

    best_epoch_dir = epoch_dirs[-1][1]
    best_model_path = bipathsnn_config["experiment"]["checkpoint"] + f"/{best_epoch_dir}/checkpoint.pt"
    logger.info(f"Best model path: {best_model_path}")
    
    # 显示其余配置
    best_epoch = epoch_dirs[-1][2]["epoch"]
    best_val_auc = epoch_dirs[-1][2]["epoch_val_metrics"]["pr_auc"]
    best_test_auc = epoch_dirs[-1][2]["epoch_test_metrics"]["pr_auc"]

    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Best validation pr_auc: {best_val_auc}")
    logger.info(f"Best test pr_auc: {best_test_auc}")


# In[9]:


from joinminer.engine import find_free_port, bipaths_inference

import torch.multiprocessing as mp
import resource

# 设置核心转储大小为0，防止中断后生成core文件
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

port = find_free_port()

# 设定进行inference的相关配置，也就是用哪个路径的模型，结果保存到哪里，多少组保存一次结果
inference_config = {}
inference_config["model_path"] = best_model_path
inference_config["infer_result_path"] = PROJECT_ROOT + '/data/dataset/AMiner/bipathsnn_infer_results/batch_id=3'
inference_config["infer_save_batch"] = 3e6

os.makedirs(inference_config["infer_result_path"], exist_ok=True)

# 获得对应的预测结果
world_size = device_count
logger.info(f"World size:{world_size}")
if world_size > 1:
    # 使用多GPU预测
    mp.spawn(
        bipaths_inference,
        args=(world_size, port, log_files_dir, dataset_config, bipathsnn_config, inference_config, device_type),
        nprocs=world_size,
        join=True
    )
else:
    # 单GPU预测
    bipaths_inference(0, 1, port, log_files_dir, dataset_config, bipathsnn_config, inference_config, device_type) 


# In[ ]:




