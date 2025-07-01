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
log_files_dir = PROJECT_ROOT + f'/data/result_data/log_files/bipathsnn/{datetime.now().strftime("%Y-%m-%d-%H:%M")}'
os.makedirs(log_files_dir, exist_ok=True)

log_filename = log_files_dir + '/main.log'
logger = setup_logger(log_filename, logger_name = "joinminer")

# Dataset config
dataset_config_file = PROJECT_ROOT + '/data/dataset/AMiner/bipathsnn_train/dataset_config.json'
dataset_config = read_json_file(dataset_config_file)
dataset_config["id_columns"] = ["index_0_node_Author_col_0", "index_1_node_Paper_col_0"]

# Dataset path
dataset_config["local_path"] = {}
dataset_config["local_path"]["train"] = PROJECT_ROOT + '/data/dataset/AMiner/bipathsnn_train/sample_type=train_neg_10'
dataset_config["local_path"]["valid"] = PROJECT_ROOT + '/data/dataset/AMiner/bipathsnn_train/sample_type=valid_neg_19'
dataset_config["local_path"]["test"] = PROJECT_ROOT + '/data/dataset/AMiner/bipathsnn_train/sample_type=test_neg_19'


# In[5]:


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


# In[6]:


from joinminer.python import mkdir
from datetime import datetime
import torch.multiprocessing as mp

# 设置本次实验相关配置
bipathsnn_config["experiment"] = {} 

# 设定本次实验相关结果对应的输出文件夹
bipathsnn_config["experiment"]["checkpoint"] = PROJECT_ROOT + f'/data/runs/AMiner/bipathsnn/4_gpu_batch_128_lr_3_4_warm_decay_1_2_drop_1_dim_256_h_4_layer_3_neg_10'
os.makedirs(bipathsnn_config["experiment"]["checkpoint"], exist_ok=True)

logger.info(f"The experiment result will be output to {bipathsnn_config['experiment']['checkpoint']}")


# In[7]:


import resource
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

from joinminer.engine import find_free_port, bipaths_trainer

# 设置核心转储大小为0，防止中断后生成core文件
port = find_free_port()

world_size = device_count
logger.info(f"World size:{world_size}")
if world_size > 1:
    # 使用多GPU训练 
    mp.spawn(
        bipaths_trainer,
        args=(world_size, port, log_files_dir, dataset_config, bipathsnn_config, device_type),
        nprocs=world_size,
        join=True
    )
else:
    # 单GPU训练
    bipaths_trainer(0, 1, port, log_files_dir, dataset_config, bipathsnn_config, device_type)


# In[ ]:




