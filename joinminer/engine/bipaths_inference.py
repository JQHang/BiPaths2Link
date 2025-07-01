import torch
import torch_npu
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .distributed import setup_ddp
from ..dataset import BiPathsDataset, bipaths_dataset_to_device
from ..model import BiPathsNN
from ..python import setup_logger, mkdir

import os
import copy
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm

# 获得logger
logger = logging.getLogger(__name__)

# 直接保存预测结果到指定路径下，要设定多少条数据保存一次结果
def bipaths_inference(rank, world_size, port, log_files_dir, dataset_config, bipathsnn_config, inference_config, device_type = "cuda"):
    # 生成一个临时logger，因为在ddp环境中启动logger并保存日志文件比较麻烦，以后再说
    global logger
    if world_size > 1:
        log_filename = log_files_dir + f'/rank_{rank}.log'
        logger = setup_logger(log_filename, logger_name = f"Rank:{rank}")

        if device_type == "npu":
            # 设置NPU设备
            torch.npu.set_device(rank)
            logger.info(f"Rank {rank}: Set NPU device to {rank}")
        
        # 设置DDP环境
        setup_ddp(rank, world_size, port, device_type)

    if device_type == "cuda":
        # torch 2.4版本专门的设置，禁用各种attention方案，只使用最传统的
        torch.backends.cuda.enable_flash_sdp(False)      # Flash Attention
        torch.backends.cuda.enable_mem_efficient_sdp(False)  # Memory Efficient Attention  
        torch.backends.cuda.enable_math_sdp(True)        # 使用传统的数学实现

    # 获得该rank具体的device名称 
    device = f'{device_type}:{rank}'
    
    # 创建推理数据集
    infer_dataset_config = copy.deepcopy(dataset_config)
    infer_dataset_config["data_dir"] = infer_dataset_config["local_path"]["infer"]
    infer_dataset_config["require_labels"] = False
    infer_dataset_config["require_ids"] = True
    infer_dataset = BiPathsDataset(
                        data_dir = infer_dataset_config["data_dir"],
                        dataset_config = infer_dataset_config,
                        num_workers = bipathsnn_config["data_loader"]["num_workers"],
                        batch_size = bipathsnn_config["data_loader"]["eval_batch_size"],
                        chunk_size = bipathsnn_config["data_loader"]["eval_batch_size"],
                        shuffle = False,
                        fill_last = False
                    )
    
    infer_datloader = DataLoader(
                        infer_dataset,
                        batch_size=None,
                        num_workers = bipathsnn_config["data_loader"]["num_workers"],
                        prefetch_factor = bipathsnn_config["data_loader"]["prefetch_factor"],
                        pin_memory=True  # 加快数据到GPU的传输，但增加内存消耗
                    )

    infer_iterator = iter(infer_datloader)

    # 获得多少个batch保存一次结果
    result_save_interval = int(inference_config["infer_save_batch"] // bipathsnn_config["data_loader"]["eval_batch_size"])
    
    # 在主进程打印各个rank分配到的推理数据 
    if rank == 0:
        for rank_id in range(world_size):
            logger.info(f"Rank {rank_id} conatins {infer_dataset.rank_total_rows[rank_id]} inference samples, "
                        f"corresponds to {infer_dataset.rank_total_batches[rank_id]} batches.")
        for worker_id in range(len(infer_dataset.workers_info)):
            logger.info(f"Worker {worker_id} conatins {len(infer_dataset.workers_info[worker_id].file_paths)} files, "
                        f"{infer_dataset.workers_info[worker_id].total_rows} samples,"
                        f"corresponds to {infer_dataset.workers_info[worker_id].target_batches} batches.")
        logger.info(f"Save inference result after each {result_save_interval} batches.")
    
    # 获得分到该rank的验证集batch总数
    infer_num_batches = infer_dataset.rank_total_batches[rank]
    
    # 创建模型
    model = BiPathsNN(bipathsnn_config, dataset_config).to(device)
    
    # 将模型包装为DDP模型
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

        # 加载用于推理的模型
        checkpoint = torch.load(inference_config["model_path"], weights_only=True)
        model.load_state_dict(checkpoint['model'])
        
    else:
        # 直接加载模型
        checkpoint = torch.load(inference_config["model_path"], weights_only=True)
    
        # 提取 model_state_dict
        model_state_dict = checkpoint['model']['model_state_dict']
    
        # 移除 'module.' 前缀
        model_state_dict = {k.replace('module.', ''): v for k, v in model_state_dict.items()}
        
        model.load_state_dict(model_state_dict)
    
    if rank == 0:
        logger.info(f"Load parameter from path: {inference_config['model_path']}")
        
    # 创建进度条，只在rank 0显示
    pbar = tqdm(total = infer_num_batches, desc = f"Inference") if rank == 0 else None
    
    # 记录各个batch的id和对应的推理结果
    ids_list = []
    outputs_list = []
    
    # 进行模型推理
    model.eval()

    # NPU 推理算子特殊处理
    for module in model.modules():
        if isinstance(module, torch.nn.TransformerEncoderLayer):
            module.train()
            for sub_module in module.modules():
                if isinstance(sub_module, torch.nn.Dropout):
                    sub_module.eval()
    
    with torch.no_grad():
        for infer_batch_count, batch_torch in enumerate(infer_iterator):
            # 将模型放入对应设备
            batch_feats, _ = bipaths_dataset_to_device(batch_torch, rank, device_type)

            # 进行推理
            outputs = model(batch_feats)

            # 通过sigmod将概率映射到0到1之间
            outputs = torch.nn.functional.sigmoid(outputs)
            
            # 记录预测的id和结果
            ids_list.append(batch_torch["ids"])
            outputs_list.append(outputs.detach().cpu().numpy().flatten())

            if rank == 0 and infer_batch_count % 100 == 0:
                pbar.update(infer_batch_count - pbar.n)
            
            # 如果达到目标数据量或是最后一份数据，则保存现有结果
            if (infer_batch_count + 1) % result_save_interval == 0 or infer_batch_count == (infer_num_batches - 1):
                file_index = (infer_batch_count + 1) // result_save_interval
                save_file_name = f"{rank}_{file_index}.parquet"

                infer_df = pd.concat(ids_list, axis=0, ignore_index=True)
                infer_df["proba"] = np.hstack(outputs_list)

                infer_df.to_parquet(inference_config['infer_result_path'] + '/' + save_file_name)

                ids_list = []
                outputs_list = []
    
    # 关闭进度条
    if rank == 0:
        pbar.update(infer_num_batches - pbar.n)
        pbar.close()

    logger.info("Finish")
    
    # 清理分布式环境
    if world_size > 1:
        # 同步所有进程
        dist.barrier()
    
        dist.destroy_process_group()
    
    return
