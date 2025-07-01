from .checkpoint import CheckpointNamer
from ..python import setup_logger

import os
import torch
import logging

class EarlyStopping:
    """Early stopping类，用于防止过拟合"""
    def __init__(self, rank, mode = "max", patience=7, min_delta=0, checkpoint_dir='checkpoints'):
        self.rank = rank
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_metric = None
        self.early_stop = False
        self.checkpoint_dir = checkpoint_dir
        self.best_checkpoint = None

        # 先用一个临时logger，因为在ddp环境中启动logger并保存日志文件比较麻烦，以后再说
        self.logger = setup_logger()

        if os.path.exists(checkpoint_dir):
            # 获取所有现有checkpoint文件并解析
            checkpoint_info = []
            for filename in os.listdir(checkpoint_dir):
                # 尝试解析文件名
                info = CheckpointNamer.parse_name(filename)
                if info is not None:
                    checkpoint_info.append((filename, info))
    
            # 查看是否有符合条件的checkpoint文件 
            if len(checkpoint_info) > 0:
                # 获得最好的epoch对应的checkpoint文件名
                checkpoint_info.sort(key=lambda x: x[1]['loss'])
                best_checkpoint = checkpoint_info[0][0]
                best_epoch = checkpoint_info[0][1]['epoch']
                best_loss = checkpoint_info[0][1]['loss']
    
                # 获得最新的epoch
                checkpoint_info.sort(key=lambda x: x[1]['epoch'])
                newest_epoch = checkpoint_info[-1][1]['epoch']
                
                # 更新early stop的配置信息
                self.best_checkpoint = best_checkpoint
                self.best_loss = best_loss
                self.counter = newest_epoch - best_epoch
    
                if self.counter >= self.patience:
                    self.early_stop = True
                
                if rank == 0:
                    self.logger.info(f"Early stopping initiated with best checkpoint: {self.best_checkpoint}, "
                                     f"loss: {self.best_loss} and counter: {self.counter}")
        else:
            if self.rank == 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
            
    def __call__(self, val_loss, model, epoch):
        if self.best_loss is None or val_loss <= self.best_loss - self.min_delta:
            self.counter = 0
            self.best_loss = val_loss
            if self.rank == 0:
                self.save_checkpoint(model, val_loss, epoch)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            if self.counter % 10 == 0 and self.rank == 0:
                self.save_checkpoint(model, val_loss, epoch)
            
    def save_checkpoint(self, model, val_loss, epoch):
        """保存模型"""
        checkpoint_name = CheckpointNamer.get_name(epoch=epoch, loss=val_loss)
        fname = os.path.join(self.checkpoint_dir, checkpoint_name)
        torch.save(model.state_dict(), fname)
        
        self.best_checkpoint = checkpoint_name
        self.logger.info(f'Model saved to {fname}')
        