import torch
import torch_npu
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .distributed import setup_ddp
from ..dataset import BiPathsDataset, bipaths_dataset_to_device
from ..model import BiPathsNN
from ..python import setup_logger, binary_problem_evaluate
from ..python import read_json_file, write_json_file

import os
import gc
import re
import math
import copy
import json
import logging
import itertools
import numpy as np
from tqdm import tqdm

# 获得logger
logger = logging.getLogger(__name__)

def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    num_cycles=0.5,
    last_step=-1
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    
    return LambdaLR(optimizer, lr_lambda, last_step)

def bipaths_trainer(rank, world_size, port, log_files_dir, dataset_config, bipathsnn_config, device_type = "cuda"):    
    # 如果是ddp并行训练则为主进程生成一个临时logger
    # 因为在ddp环境中得重新生成logger并设置保存日志文件，这比较麻烦，以后再说
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
    
    # 创建训练集
    train_dataset_config = copy.deepcopy(dataset_config)
    train_dataset_config["data_dir"] = train_dataset_config["local_path"]["train"]
    train_dataset_config["require_labels"] = True
    train_dataset_config["require_ids"] = False
    train_dataset = BiPathsDataset(
                        data_dir = train_dataset_config["data_dir"],
                        dataset_config = train_dataset_config,
                        num_workers = bipathsnn_config["data_loader"]["num_workers"],
                        batch_size = bipathsnn_config["data_loader"]["train_batch_size"],
                        chunk_size = bipathsnn_config["data_loader"]["train_batch_size"],
                        shuffle = True,
                        fill_last = True
                    )
    
    # 获得训练集几轮eval一次
    train_epoch_size = int(bipathsnn_config["data_loader"]["train_epoch_size"])
    train_batch_size = int(bipathsnn_config["data_loader"]["train_batch_size"] * world_size)
    train_num_batches = (train_epoch_size + train_batch_size - 1) // train_batch_size
    
    train_datloader = DataLoader(
                        train_dataset,
                        batch_size=None,
                        num_workers = bipathsnn_config["data_loader"]["num_workers"],
                        prefetch_factor = bipathsnn_config["data_loader"]["prefetch_factor"],
                        pin_memory=True,  # 加快数据到GPU的传输，但增加内存消耗
                        persistent_workers=True
                    )

    train_iterator = iter(train_datloader)

    # 在主进程打印各个rank分配到的训练数据 
    if rank == 0:
        for rank_id in range(world_size):
            logger.info(f"Rank {rank_id} conatins {train_dataset.rank_total_rows[rank_id]} train samples, "
                        f"corresponds to {train_dataset.rank_total_batches[rank_id]} batches.")
        for worker_id in range(len(train_dataset.workers_info)):
            logger.info(f"Worker {worker_id} conatins {len(train_dataset.workers_info[worker_id].file_paths)} files, "
                        f"{train_dataset.workers_info[worker_id].total_rows} samples,"
                        f"corresponds to {train_dataset.workers_info[worker_id].target_batches} batches.")
        logger.info(f"We evaluate the performance after train {train_num_batches} batches.")

    val_dataset_config = copy.deepcopy(dataset_config)
    val_dataset_config["data_dir"] = val_dataset_config["local_path"]["valid"]
    val_dataset_config["require_labels"] = True
    val_dataset_config["require_ids"] = False
    val_dataset = BiPathsDataset(
                        data_dir = val_dataset_config["data_dir"],
                        dataset_config = val_dataset_config,
                        num_workers = bipathsnn_config["data_loader"]["num_workers"],
                        batch_size = bipathsnn_config["data_loader"]["eval_batch_size"],
                        chunk_size = bipathsnn_config["data_loader"]["eval_batch_size"],
                        shuffle = False,
                        fill_last = False
                    )

    # 获得验证集验证多少个batch
    val_epoch_size = int(bipathsnn_config["data_loader"]["eval_epoch_size"])
    if val_epoch_size > 0:
        val_batch_size = int(bipathsnn_config["data_loader"]["eval_batch_size"] * world_size)
        val_num_batches = (val_epoch_size + val_batch_size - 1) // val_batch_size
        if val_num_batches > val_dataset.rank_total_batches[rank]:
            val_num_batches = val_dataset.rank_total_batches[rank]
    else:
        # 直接获得该rank分到的验证集batch总数
        val_num_batches = val_dataset.rank_total_batches[rank]
    
    val_dataloader = DataLoader(
                        val_dataset,
                        batch_size=None,
                        num_workers = bipathsnn_config["data_loader"]["num_workers"],
                        prefetch_factor = bipathsnn_config["data_loader"]["prefetch_factor"],
                        pin_memory=True  # 加快数据到GPU的传输，但增加内存消耗
                    )
    
    # 在主进程打印各个rank分配到的训练数据 
    if rank == 0:
        for rank_id in range(world_size):
            logger.info(f"Rank {rank_id} conatins {val_dataset.rank_total_rows[rank_id]} validation samples, "
                        f"corresponds to {val_dataset.rank_total_batches[rank_id]} batches.")
        for worker_id in range(len(val_dataset.workers_info)):
            logger.info(f"Worker {worker_id} conatins {len(val_dataset.workers_info[worker_id].file_paths)} files, "
                        f"{val_dataset.workers_info[worker_id].total_rows} samples,"
                        f"corresponds to {val_dataset.workers_info[worker_id].target_batches} batches.")
        logger.info(f"We evaluate the performance on the first {val_num_batches} batches of validation datasets.")

    test_dataset_config = copy.deepcopy(dataset_config)
    test_dataset_config["data_dir"] = test_dataset_config["local_path"]["test"]
    test_dataset_config["require_labels"] = True
    test_dataset_config["require_ids"] = False
    test_dataset = BiPathsDataset(
                        data_dir = test_dataset_config["data_dir"],
                        dataset_config = test_dataset_config,
                        num_workers = bipathsnn_config["data_loader"]["num_workers"],
                        batch_size = bipathsnn_config["data_loader"]["eval_batch_size"],
                        chunk_size = bipathsnn_config["data_loader"]["eval_batch_size"],
                        shuffle = False,
                        fill_last = False
                    )

    # 在主进程打印各个rank分配到的训练数据 
    if rank == 0:
        for rank_id in range(world_size):
            logger.info(f"Rank {rank_id} conatins {test_dataset.rank_total_rows[rank_id]} test samples, "
                        f"corresponds to {test_dataset.rank_total_batches[rank_id]} batches.")
        for worker_id in range(len(test_dataset.workers_info)):
            logger.info(f"Worker {worker_id} conatins {len(test_dataset.workers_info[worker_id].file_paths)} files, "
                        f"{test_dataset.workers_info[worker_id].total_rows} samples,"
                        f"corresponds to {test_dataset.workers_info[worker_id].target_batches} batches.")
    
    # 获得测试验证多少个batch
    test_epoch_size = int(bipathsnn_config["data_loader"]["eval_epoch_size"])
    if test_epoch_size > 0:
        test_batch_size = int(bipathsnn_config["data_loader"]["eval_batch_size"] * world_size)
        test_num_batches = (test_epoch_size + test_batch_size - 1) // test_batch_size
        if test_num_batches > test_dataset.rank_total_batches[rank]:
            test_num_batches = test_dataset.rank_total_batches[rank]

        if rank == 0:
            logger.info(f"We evaluate the performance on the first {test_num_batches} batches of test datasets.")
    else:
        # 直接获得该rank分到的测试集batch总数
        test_num_batches = test_dataset.rank_total_batches[rank]

        if rank == 0:
            logger.info(f"We evaluate the performance on all the {test_num_batches} batches of test datasets.")
            
    test_dataloader = DataLoader(
                        test_dataset,
                        batch_size=None,
                        num_workers = bipathsnn_config["data_loader"]["num_workers"],
                        prefetch_factor = bipathsnn_config["data_loader"]["prefetch_factor"],
                        pin_memory=True  # 加快数据到GPU的传输，但增加内存消耗
                    )
    
    # 创建模型
    model = BiPathsNN(bipathsnn_config, dataset_config).to(device)
    
    # 将模型包装为DDP模型
    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # 创建优化器
    optimizer = AdamW(model.parameters(), lr = bipathsnn_config['learning_rate'], 
                      weight_decay = bipathsnn_config['weight_decay'])
    
    # 创建损失函数
    criterion = nn.BCEWithLogitsLoss()

    # 获得现有的epoch的结果
    epoch_dirs = []
    for epoch_dir in os.listdir(bipathsnn_config["experiment"]["checkpoint"]):
        if os.path.exists(bipathsnn_config["experiment"]["checkpoint"] + f"/{epoch_dir}/metric.json"):
            # 获得对应的epoch信息
            pattern = r'epoch_(\d+)'
            match = re.match(pattern, epoch_dir)
            epoch = int(match.group(1))
    
            # 记录结果
            epoch_dirs.append((epoch, epoch_dir))

    # 初始化epoch结束时对应的各项指标
    epoch_metrics = {}
    epoch_metrics["patience_counter"] = 0
    epoch_metrics["best_val_auc"] = 0

    # 设定当前epoch
    current_epoch = 0
    
    # 读取已有的最新epoch的对应的数据
    if len(epoch_dirs) > 0:
        epoch_dirs.sort(key=lambda x: x[0])

        newest_epoch = epoch_dirs[-1][0]
        newest_epoch_dir = epoch_dirs[-1][1]
    
        # 加载最新的epoch对应的checkpoint 
        checkpoint = torch.load(bipathsnn_config["experiment"]["checkpoint"] + f"/{newest_epoch_dir}/checkpoint.pt", weights_only=True)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if rank == 0:
            logger.info(f"Load parameter from the newest epoch {newest_epoch_dir}")

        # 获得对应的metric作为当前epoch的metric
        epoch_metrics = read_json_file(bipathsnn_config["experiment"]["checkpoint"] + f"/{newest_epoch_dir}/metric.json")

        # 设定当前epoch
        current_epoch = epoch_metrics["epoch"] + 1

    # 创建学习率规划器
    total_steps = bipathsnn_config["epochs"] * train_num_batches
    last_step = current_epoch * train_num_batches - 1
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps = int(0.1 * total_steps),
        num_training_steps = total_steps,
        last_step = last_step
    )
    
    # 训练循环
    for epoch in range(current_epoch, bipathsnn_config["epochs"]):
        if rank == 0:
            logger.info(f"Processing {epoch}-th epoch")

        # 进行早停检查（每个都得检查来一起停止）
        if epoch_metrics["patience_counter"] >= bipathsnn_config["patience"]:
            if rank == 0:
                logger.info("Early stopping triggered")
            break
        
        # 训练模型
        train_epoch(model, train_datloader, train_iterator, optimizer, scheduler, criterion, 
                    train_num_batches, epoch, rank, world_size, device_type)
        
        # 获得模型验证集效果
        val_metrics = eval_epoch(model, val_dataloader, val_num_batches, epoch, rank, world_size, device_type, desc = "Validation")

        # 同步所有进程
        if world_size > 1:
            dist.barrier()
            
        # 获得模型测试集效果
        test_metrics = eval_epoch(model, test_dataloader, test_num_batches, epoch, rank, world_size, device_type, desc = "Test")
        
        # 同步所有进程
        if world_size > 1:
            dist.barrier()

        # 检查验证集效果的变化
        epoch_metrics["epoch"] = epoch
        epoch_metrics["epoch_val_metrics"] = val_metrics
        epoch_metrics["epoch_test_metrics"] = test_metrics
        if val_metrics["pr_auc"] > epoch_metrics["best_val_auc"]:
            epoch_metrics["best_epoch"] = epoch
            epoch_metrics["patience_counter"] = 0
            epoch_metrics["best_val_auc"] = val_metrics["pr_auc"]
            
            epoch_metrics["best_epoch_val_metrics"] = val_metrics
            epoch_metrics["best_epoch_test_metrics"] = test_metrics
        else:
            epoch_metrics["patience_counter"] += 1

        # 在主进程保留当前模型参数及对应的各项指标  
        if rank == 0:
            logger.info(f"Current epoch: {epoch_metrics['epoch']}.\n"
                        f"Best epoch: {epoch_metrics['best_epoch']}.\n"
                        f"No improvement for {epoch_metrics['patience_counter']} epochs.\n"
                        f"Best Val Auc: {epoch_metrics['best_val_auc']:.4f}")
            
            # 获得模型结果保留路径
            epoch_checkpoint_dir = bipathsnn_config["experiment"]["checkpoint"] + f"/epoch_{epoch}"
            os.makedirs(epoch_checkpoint_dir, exist_ok=True)
        
            # 保存该轮模型参数结果
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }
            torch.save(checkpoint, epoch_checkpoint_dir + "/checkpoint.pt")

            # 记录相关信息
            write_json_file(epoch_metrics, epoch_checkpoint_dir + "/metric.json")
    
    # 清理分布式环境
    if world_size > 1:
        dist.destroy_process_group()
    
    return

def train_epoch(model, dataloader, iterator, optimizer, scheduler, criterion, num_batches, epoch, rank, world_size, device_type):
    # 获得该rank具体的device名称 
    device = f'{device_type}:{rank}'
    
    # 训练一个epoch
    model.train()
    train_loss = 0.0
    train_batch_count = 0

    # 创建进度条，只在rank 0显示
    pbar = tqdm(total = num_batches, desc=f"Training epoch {epoch}") if rank == 0 else None

    while True:
        try:
            batch_torch = next(iterator)
            batch_feats, batch_label = bipaths_dataset_to_device(batch_torch, rank, device_type)
    
            optimizer.zero_grad()
            outputs = model(batch_feats)
            loss = criterion(outputs, batch_label)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            train_batch_count += 1
    
            if rank == 0 and train_batch_count % 10 == 0:
                pbar.update(train_batch_count - pbar.n)
                pbar.set_postfix(loss=loss.item())
            
            if train_batch_count >= num_batches:
                # 达到指定批次数，结束该epoch，并在主进程关闭进度条
                if rank == 0:
                    pbar.update(train_batch_count - pbar.n)
                    pbar.set_postfix(loss=loss.item())
                    pbar.close()
                break
                
        except StopIteration:
            iterator = iter(dataloader)
    
    # 收集所有进程的数据
    train_loss = torch.tensor(train_loss).to(device)
    train_batch_count = torch.tensor(train_batch_count).to(device)
    
    # 规约到所有进程
    if world_size > 1:
        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_batch_count, op=dist.ReduceOp.SUM)

    # 获得所有进程上的平均train_loss
    train_loss = (train_loss / train_batch_count).cpu().item()

    # 显示该轮总体训练效果
    if rank == 0:
        logger.info(f'Train Loss: {train_loss:.4f}')
    
    return

def eval_epoch(model, dataloader, num_batches, epoch, rank, world_size, device_type, desc = "Validation"):
    # 获得该rank具体的device名称 
    device = f'{device_type}:{rank}'
    
    # 验证阶段
    model.eval()

    # NPU 推理算子特殊处理
    for module in model.modules():
        if isinstance(module, torch.nn.TransformerEncoderLayer):
            module.train()
            for sub_module in module.modules():
                if isinstance(sub_module, torch.nn.Dropout):
                    sub_module.eval()

    # 创建进度条，只在rank 0显示
    pbar = tqdm(total=num_batches, desc=f"{desc} epoch {epoch}") if rank == 0 else None

    # 创建迭代器
    iterator = itertools.islice(dataloader, num_batches)
    
    # 用于保存所有输出和标签的列表
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for batch_count, batch_torch in enumerate(iterator):
            # 将模型放入对应设备
            batch_feats, batch_label = bipaths_dataset_to_device(batch_torch, rank, device_type)
            
            outputs = torch.sigmoid(model(batch_feats))
    
            # 收集结果
            all_outputs.append(outputs)
            all_labels.append(batch_label)
            
            if rank == 0 and batch_count % 10 == 0:
                pbar.update(batch_count - pbar.n)
        
    # 所有进程完成前向传播
    dist.barrier()

    # 在主进程关闭进度条
    if rank == 0:
        pbar.update(num_batches - pbar.n)
        pbar.close()
    
    # 每个进程合并自己的结果
    outputs_tensor = torch.cat(all_outputs, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)

    # 看看是gpu还是npu来决定数据同步方案
    if device_type == "cuda":
        # 收集每个进程的张量大小
        local_size = torch.tensor([outputs_tensor.shape[0]], dtype=torch.int32).to(device)
        all_sizes = [torch.zeros(1, dtype=torch.int32).to(device) for _ in range(world_size)]
        
        dist.all_gather(all_sizes, local_size)
    
        # logger.info(f"{device}")
        # if rank == 1:
        #     logger.info(f"{local_size[0].item()}")
        #     logger.info(f"{outputs_tensor.dtype}")
        #     logger.info(f"{labels_tensor.dtype}")
        
        if rank == 0:
            logger.info(f"All sizes: {[size.item() for size in all_sizes]}")
        
        # 创建接收缓冲区 
        gathered_outputs = [torch.zeros(size.item(), outputs_tensor.shape[1], 
                                        dtype=outputs_tensor.dtype).to(device)
                            for size in all_sizes]
        gathered_labels = [torch.zeros(size.item(), dtype=labels_tensor.dtype).to(device) 
                           for size in all_sizes]
        
        # 使用all_gather收集数据
        dist.all_gather(gathered_outputs, outputs_tensor)
        dist.all_gather(gathered_labels, labels_tensor)
    else:
        # 替换原来的all_gather
        gathered_outputs = [None for _ in range(world_size)]
        gathered_labels = [None for _ in range(world_size)]
        
        dist.all_gather_object(gathered_outputs, outputs_tensor.cpu().numpy())
        dist.all_gather_object(gathered_labels, labels_tensor.cpu().numpy())
        
        # 转换回tensor
        gathered_outputs = [torch.from_numpy(arr).to(device) for arr in gathered_outputs]
        gathered_labels = [torch.from_numpy(arr).to(device) for arr in gathered_labels]
    
    # 合并张量结果
    all_gathered_outputs = torch.cat(gathered_outputs, dim=0)
    all_gathered_labels = torch.cat(gathered_labels, dim=0)
    
    # 转换为numpy进行评估
    outputs_np = all_gathered_outputs.cpu().numpy().flatten()
    labels_np = all_gathered_labels.cpu().numpy().flatten()

    # 显示样本总数
    if rank == 0:
        logger.info(f"Evaluate {desc} dataset with {labels_np.shape[0]} samples from: {[x.shape[0] for x in gathered_labels]}")

    # 进行评估
    binary_metrics = binary_problem_evaluate(labels_np, outputs_np, False, 
                                             Top_K_list = [10000, 20000, 40000, 80000, 
                                                           100000, 200000, 300000, 500000])

    # 显示效果
    if rank == 0:
        logger.info(f"{desc} dataset metrics:\n%s", json.dumps(binary_metrics, indent=4))
    
    return binary_metrics
