from .distributed import setup_ddp, dataset_to_device
from .checkpoint import CheckpointNamer
from .early_stopping import EarlyStopping
from ..dataset import JoinEdgesDataset
from ..model import Join_HGNN
from ..python import setup_logger

import torch
import torch.nn as nn
from torch.optim import AdamW
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import os
import gc
import logging
import numpy as np
from tqdm import tqdm

# 获得logger
logger = logging.getLogger(__name__)

def train_model(rank, world_size, port, dataset, join_hgnn_config):    
    # 如果是ddp并行训练则为主进程生成一个临时logger
    # 因为在ddp环境中得重新生成logger并设置保存日志文件，这比较麻烦，以后再说
    global logger
    if world_size > 1:
        if rank == 0:
            logger = setup_logger()
        
        # 设置DDP环境
        setup_ddp(rank, world_size, port)

    # 创建训练集
    train_dataset = JoinEdgesDataset(
                        data_dir = dataset["local_path"] + f"/split_type=train",
                        dataset_config = dataset,
                        num_workers = join_hgnn_config["data_loader"]["num_workers"],
                        batch_size = join_hgnn_config["data_loader"]["train_batch_size"],
                        chunk_size = join_hgnn_config["data_loader"]["train_batch_size"],
                        shuffle = True,
                        fill_last = True
                    )
    
    # 获得训练集几轮eval一次
    train_epoch_size = int(join_hgnn_config["data_loader"]["train_epoch_size"])
    train_batch_size = int(join_hgnn_config["data_loader"]["train_batch_size"] * world_size)
    train_num_batches = (train_epoch_size + train_batch_size - 1) // train_batch_size
    
    train_datloader = DataLoader(
                        train_dataset,
                        batch_size=None,
                        num_workers = join_hgnn_config["data_loader"]["num_workers"],
                        prefetch_factor = join_hgnn_config["data_loader"]["prefetch_factor"],
                        pin_memory=True  # 加快数据到GPU的传输，但增加内存消耗
                    )

    train_iterator = iter(train_datloader)

    # 在主进程打印各个rank分配到的训练数据 
    if rank == 0:
        for rank_id in range(world_size):
            logger.info(f"Rank {rank_id} conatins {train_dataset.rank_total_rows[rank_id]} train samples, "
                        f"corresponds to {train_dataset.rank_total_batches[rank_id]} batches.")
        logger.info(f"We evaluate the performance after train {train_num_batches} batches.")
    
    val_dataset = JoinEdgesDataset(
                        data_dir = dataset["local_path"] + f"/split_type=valid",
                        dataset_config = dataset,
                        num_workers = join_hgnn_config["data_loader"]["num_workers"],
                        batch_size = join_hgnn_config["data_loader"]["val_batch_size"],
                        chunk_size = join_hgnn_config["data_loader"]["val_batch_size"],
                        shuffle = False,
                        fill_last = False
                    )

    # 获得验证集验证多少个batch
    val_epoch_size = int(join_hgnn_config["data_loader"]["val_epoch_size"])
    if val_epoch_size > 0:
        val_batch_size = int(join_hgnn_config["data_loader"]["val_batch_size"] * world_size)
        val_num_batches = (val_epoch_size + val_batch_size - 1) // val_batch_size
        if val_num_batches > val_dataset.rank_total_batches[rank]:
            val_num_batches = val_dataset.rank_total_batches[rank]
    else:
        # 直接获得该rank分到的验证集batch总数
        val_num_batches = val_dataset.rank_total_batches[rank]
    
    val_datloader = DataLoader(
                        val_dataset,
                        batch_size=None,
                        num_workers = join_hgnn_config["data_loader"]["num_workers"],
                        prefetch_factor = join_hgnn_config["data_loader"]["prefetch_factor"],
                        pin_memory=True  # 加快数据到GPU的传输，但增加内存消耗
                    )
    
    # 在主进程打印各个rank分配到的训练数据 
    if rank == 0:
        for rank_id in range(world_size):
            logger.info(f"Rank {rank_id} conatins {val_dataset.rank_total_rows[rank_id]} validation samples, "
                        f"corresponds to {val_dataset.rank_total_batches[rank_id]} batches.")
        logger.info(f"We evaluate the performance on the first {val_num_batches} batches.")
        
    # 读取对数据的缩放参数
    scaler_stats = np.load(dataset["local_path"] + "/_scaler.npy", allow_pickle=True).item()
    
    # 创建模型
    model = Join_HGNN(join_hgnn_config, dataset).cuda(rank)
    
    # 将模型包装为DDP模型
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    # 创建优化器
    optimizer = AdamW(model.parameters(), lr = join_hgnn_config['learning_rate'], 
                      weight_decay = join_hgnn_config['weight_decay'])
    
    # 创建损失函数
    # criterion = nn.MSELoss()
    criterion = nn.HuberLoss(delta=1.0, reduction='mean') 
    
    # 创建早停设置
    early_stopping = EarlyStopping(rank = rank, patience = join_hgnn_config["patience"], 
                                   checkpoint_dir = join_hgnn_config["experiment"]["checkpoint"])

    # 查看早停检查结果
    if early_stopping.early_stop:
        if rank == 0:
            logger.info("Early stopping triggered")
        return
    
    # 检查checkpoint文件夹是否存在，并确定哪个epoch开始
    start_epoch = 0
    if os.path.exists(join_hgnn_config["experiment"]["checkpoint"]):
        # 获取所有checkpoint文件并解析
        checkpoint_info = []
        for filename in os.listdir(join_hgnn_config["experiment"]["checkpoint"]):
            # 尝试解析文件名
            info = CheckpointNamer.parse_name(filename)
            if info is not None:
                checkpoint_info.append((filename, info))

        # 查看是否有符合条件的checkpoint文件 
        if len(checkpoint_info) > 0:
            # 获得最新的epoch对应的checkpoint文件名
            checkpoint_info.sort(key=lambda x: x[1]['epoch'])
            newest_checkpoint_name = checkpoint_info[-1][0]
            start_epoch = checkpoint_info[-1][1]['epoch']
            
            # 加载最新的epoch对应的checkpoint 
            checkpoint_path = os.path.join(join_hgnn_config["experiment"]["checkpoint"], newest_checkpoint_name)
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint)
            
            if rank == 0:
                logger.info(f"Load parameter from the newest checkpoint {newest_checkpoint_name}")
    
    # 训练循环
    for epoch in range(start_epoch, join_hgnn_config["epochs"]):
        if rank == 0:
            logger.info(f"Processing {epoch}-th epoch")

        # 训练模型
        train_loss = train_epoch(model, train_datloader, train_iterator, optimizer, criterion, 
                                 train_num_batches, epoch, rank, world_size)

        # 验证模型效果
        val_loss = eval_epoch(model, val_datloader, criterion, val_num_batches, 
                              scaler_stats, epoch, rank, world_size)

        # 同步所有进程
        if world_size > 1:
            dist.barrier()
        
        # 进行早停检查
        early_stopping(val_loss, model, epoch)

        # 在主进程显示当前最优的check_point结果
        if rank == 0:
            logger.info(f"Best checkpoint name: {early_stopping.best_checkpoint}")
            
        # 查看早停检查结果（每个都得检查来一起停止）
        if early_stopping.early_stop:
            if rank == 0:
                logger.info("Early stopping triggered")
            break
    
    # 清理分布式环境
    if world_size > 1:
        dist.destroy_process_group()
    
    return

def train_epoch(model, dataloader, iterator, optimizer, criterion, num_batches, epoch, rank, world_size):
    # 训练一个epoch
    model.train()
    train_loss = 0.0
    train_batch_count = 0

    # 创建进度条，只在rank 0显示
    pbar = tqdm(total = num_batches, desc=f"Training epoch {epoch}") if rank == 0 else None

    while True:
        try:
            batch_torch = next(iterator)
            batch_feats, batch_label = dataset_to_device(batch_torch, rank)
    
            optimizer.zero_grad()
            # outputs = model(batch_feats)
            outputs = model(batch_feats).squeeze()
            loss = criterion(outputs, batch_label)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batch_count += 1
    
            if rank == 0 and train_batch_count % 200 == 0:
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
    train_loss = torch.tensor(train_loss).cuda(rank)
    train_batch_count = torch.tensor(train_batch_count).cuda(rank)
    
    # 规约到所有进程
    if world_size > 1:
        dist.all_reduce(train_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_batch_count, op=dist.ReduceOp.SUM)

    # 获得所有进程上的平均train_loss
    train_loss = (train_loss / train_batch_count).cpu().item()

    # 显示该轮总体训练效果
    if rank == 0:
        logger.info(f'Train Loss: {train_loss:.4f}')
    
    return train_loss

def eval_epoch(model, dataloader, criterion, num_batches, scaler_stats, epoch, rank, world_size):
    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_mse = 0.0
    val_rmse = 0.0
    val_mae = 0.0
    val_smape = 0.0
    val_acc = 0.0
    val_within_3_acc = 0.0
    val_within_5_acc = 0.0
    val_batch_count = 0

    # 创建进度条，只在rank 0显示
    pbar = tqdm(total=num_batches, desc=f"Validating epoch {epoch}") if rank == 0 else None

    # 创建迭代器
    iterator = iter(dataloader)
    
    with torch.no_grad():
        while True:
            batch_torch = next(iterator)
            
            # 将模型放入对应设备
            batch_feats, batch_label = dataset_to_device(batch_torch, rank)
            
            # outputs = model(batch_feats)
            outputs = model(batch_feats).squeeze()
            loss = criterion(outputs, batch_label)
            
            val_loss += loss.item()
            val_batch_count += 1

            # 标签和预测结果缩放回原始数值
            # scale_std = scaler_stats["labels"]["std"]["std"].reshape(1, -1)
            # scale_mean = scaler_stats["labels"]["std"]["mean"].reshape(1, -1)
            scale_std = scaler_stats["labels"]["std"]["std"][21]
            scale_mean = scaler_stats["labels"]["std"]["mean"][21]
            
            descaled_output = outputs.cpu().numpy()
            descaled_output = descaled_output * scale_std + scale_mean
            
            descaled_label = batch_label.cpu().numpy()
            descaled_label = descaled_label * scale_std + scale_mean
            
            # 获得逆缩放后更细致的一些评价指标
            val_mse += np.mean((descaled_output - descaled_label) ** 2)
            val_rmse += np.sqrt(np.mean((descaled_output - descaled_label) ** 2))
            val_mae += np.mean(np.abs(descaled_output - descaled_label))
            val_smape += np.mean(2 * np.abs(descaled_output - descaled_label) / (np.abs(descaled_output) + np.abs(descaled_label))) * 100
            val_acc += np.mean(np.abs(descaled_output - descaled_label) <= 0.5)
            val_within_3_acc += np.mean(np.abs(descaled_output - descaled_label) <= 3)
            val_within_5_acc += np.mean(np.abs(descaled_output - descaled_label) <= 5)
            
            if rank == 0 and val_batch_count % 200 == 0:
                pbar.update(val_batch_count - pbar.n)
                pbar.set_postfix(loss=loss.item())

            if val_batch_count >= num_batches:
                break

        # 删除iterator，释放空间 
        del iterator
        gc.collect()
        
        # 在主进程关闭进度条
        if rank == 0:
            pbar.update(val_batch_count - pbar.n)
            pbar.set_postfix(loss=loss.item())
            pbar.close()
    
    # 收集所有进程的数据
    val_loss = torch.tensor(val_loss).cuda(rank)
    val_mse = torch.tensor(val_mse).cuda(rank)
    val_rmse = torch.tensor(val_rmse).cuda(rank)
    val_mae = torch.tensor(val_mae).cuda(rank)
    val_smape = torch.tensor(val_smape).cuda(rank)
    val_acc = torch.tensor(val_acc).cuda(rank)
    val_within_3_acc = torch.tensor(val_within_3_acc).cuda(rank)
    val_within_5_acc = torch.tensor(val_within_5_acc).cuda(rank)
    val_batch_count = torch.tensor(val_batch_count).cuda(rank)
    
    # 规约到所有进程
    if world_size > 1:
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_mse, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_rmse, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_mae, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_smape, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_within_3_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_within_5_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_batch_count, op=dist.ReduceOp.SUM)

    # 获得所有进程上的平均val_loss
    val_loss = val_loss.item() / val_batch_count.item()
    val_mse = val_mse.item() / val_batch_count.item()
    val_rmse = val_rmse.item() / val_batch_count.item()
    val_mae = val_mae.item() / val_batch_count.item()
    val_smape = val_smape.item() / val_batch_count.item()
    val_acc = val_acc.item() / val_batch_count.item()
    val_within_3_acc = val_within_3_acc.item() / val_batch_count.item()
    val_within_5_acc = val_within_5_acc.item() / val_batch_count.item()
    
    # 在主进程关闭进度条，并显示该轮总体效果
    if rank == 0:
        logger.info(f"Val Loss: {val_loss:.4f}, mse: {val_mse:.4f}, "
                    f"rmse: {val_rmse:.4f}, mae: {val_mae:.4f}, "
                    f"smape: {val_smape:.4f}, accuracy: {val_acc:.4f}, "
                    f"within_3_accuracy: {val_within_3_acc:.4f}, within_5_accuracy: {val_within_5_acc:.4f}.")
        
    return val_loss