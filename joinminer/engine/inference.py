from .distributed import setup_ddp, dataset_to_device
from ..dataset import JoinEdgesDataset
from ..model import Join_HGNN
from ..python import setup_logger

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import os
import logging
import numpy as np
from tqdm import tqdm

# 获得logger
logger = logging.getLogger(__name__)

def model_inference(rank, world_size, port, dataset, join_hgnn_config, inference_config, shared_list):
    # 生成一个临时logger，因为在ddp环境中启动logger并保存日志文件比较麻烦，以后再说
    # if rank == 0:
    logger = setup_logger()
    
    # 设置DDP环境
    setup_ddp(rank, world_size, port)

    # 创建推理数据集
    infer_dataset = JoinEdgesDataset(
                        data_dir = inference_config["dataset_local_path"],
                        dataset_config = dataset,
                        num_workers = join_hgnn_config["data_loader"]["num_workers"],
                        batch_size = join_hgnn_config["data_loader"]["pred_batch_size"],
                        chunk_size = join_hgnn_config["data_loader"]["pred_batch_size"],
                        shuffle = False,
                        fill_last = False
                    )
    
    infer_datloader = DataLoader(
                        infer_dataset,
                        batch_size=None,
                        num_workers = join_hgnn_config["data_loader"]["num_workers"],
                        prefetch_factor = join_hgnn_config["data_loader"]["prefetch_factor"],
                        pin_memory=True  # 加快数据到GPU的传输，但增加内存消耗
                    )

    infer_iterator = iter(infer_datloader)
    
    # 获得验证集batch总数
    infer_num_batches = infer_dataset.rank_total_batches[rank]

    # 在主进程打印各个rank分配到的推理数据 
    if rank == 0:
        for rank_id in range(world_size):
            logger.info(f"Rank {rank_id} conatins {infer_dataset.rank_total_rows[rank_id]} inference samples, "
                        f"corresponds to {infer_dataset.rank_total_batches[rank_id]} batches.")

    # 读取对数据的缩放参数
    scaler_stats = np.load(dataset["local_path"] + "/_scaler.npy", allow_pickle=True).item()
    
    # 创建模型
    model = Join_HGNN(join_hgnn_config, dataset).cuda(rank)
    
    # 将模型包装为DDP模型
    model = DDP(model, device_ids=[rank])

    # 加载用于推理的模型
    model.load_state_dict(torch.load(inference_config["model_path"]))

    if rank == 0:
        logger.info(f"Load parameter from path: {inference_config['model_path']}")

    # 创建进度条，只在rank 0显示
    pbar = tqdm(total = infer_num_batches, desc = f"Inference") if rank == 0 else None

    # 记录各个batch的推理结果和真实结果
    all_outputs = []
    all_labels = []
    
    # 进行模型推理
    model.eval()
    with torch.no_grad():
        for infer_batch_count in range(infer_num_batches):
            batch_torch = next(infer_iterator)
            
            # 将模型放入对应设备
            batch_feats, batch_label = dataset_to_device(batch_torch, rank)

            # 进行推理
            outputs = model(batch_feats)

            # 将结果缩放回原始情况
            scale_std = scaler_stats["labels"]["std"]["std"].reshape(1, -1)
            scale_mean = scaler_stats["labels"]["std"]["mean"].reshape(1, -1)
            
            descaled_output = outputs.cpu().numpy()
            descaled_output = descaled_output * scale_std + scale_mean
            
            descaled_label = batch_label.cpu().numpy()
            descaled_label = descaled_label * scale_std + scale_mean
            
            # 记录结果
            all_outputs.append(descaled_output)
            all_labels.append(descaled_label)

            if rank == 0:
                pbar.update(1)

    # 同步所有进程
    dist.barrier()

    # 合并各个batch的结果
    all_outputs = np.vstack(all_outputs)
    all_labels = np.vstack(all_labels)

    # 将结果放入队列
    shared_list[rank] = (all_outputs, all_labels)
    
    # 关闭进度条
    if rank == 0:
        pbar.close()
    
    # 清理分布式环境
    dist.destroy_process_group()
    
    return
