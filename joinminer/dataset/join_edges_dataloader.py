import os
import glob
import logging
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Iterator, Tuple, NamedTuple

import torch
import torch.distributed as dist
from torch.utils.data import IterableDataset

@dataclass
class WorkerInfo:
    global_worker_id: int
    total_workers: int
    target_batches: int
    file_paths: List[str]
    total_rows: int

class JoinEdgesDataset(IterableDataset):
    def __init__(
        self,
        data_dir: str,
        dataset_config: dict,
        num_workers: int,
        batch_size: int = 2000,
        chunk_size: int = 2000,
        shuffle: bool = True,
        fill_last: bool = False
    ):
        super().__init__()
        self.data_dir = data_dir
        self.dataset_config = dataset_config
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.shuffle = shuffle
        self.fill_last = fill_last
        self.logger = logging.getLogger(__name__)
        
        # 初始化分布式信息
        self._init_distributed_info()
        # 扫描并初始化文件信息
        self._init_file_info()
        # 初始化worker信息
        self._init_worker_info()
        
    def _init_distributed_info(self):
        """初始化分布式训练相关信息"""
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
            
    def _init_file_info(self):
        """初始化文件信息"""
        file_paths = sorted(glob.glob(os.path.join(self.data_dir, "*.parquet")))
        if not file_paths:
            raise ValueError(f"No parquet files found in {self.data_dir}")
        
        # 使用线程池并行统计文件大小
        self.file_sizes = self._parallel_get_file_sizes(file_paths)
        self.total_rows = sum(size for _, size in self.file_sizes)
        
    def _init_worker_info(self):
        """预先分配文件给所有worker并计算相关信息"""
        self.total_workers = self.world_size * self.num_workers
        
        # 预分配数组
        worker_files = [[] for _ in range(self.total_workers)]
        worker_rows = np.zeros(self.total_workers, dtype=np.int64)
        
        # 分配文件给worker
        for file_path, size in self.file_sizes:
            min_rows_worker = np.argmin(worker_rows)
            worker_files[min_rows_worker].append(file_path)
            worker_rows[min_rows_worker] += size
            
        # 存储每个worker的信息
        self.workers_info = []
        self.rank_total_rows = np.zeros(self.world_size, dtype=np.int64)
        self.rank_total_batches = np.zeros(self.world_size, dtype=np.int64)
        
        for global_worker_id in range(self.total_workers):
            worker_total_rows = worker_rows[global_worker_id]
            target_batches = (worker_total_rows + self.batch_size - 1) // self.batch_size
            
            # 计算该worker属于哪个rank
            rank_id = global_worker_id // self.num_workers
            self.rank_total_rows[rank_id] += worker_total_rows
            self.rank_total_batches[rank_id] += target_batches
            
            self.workers_info.append(WorkerInfo(
                global_worker_id=global_worker_id,
                total_workers=self.total_workers,
                target_batches=target_batches,
                file_paths=worker_files[global_worker_id],
                total_rows=worker_total_rows
            ))
            
        # 记录当前rank的总行数和batch数
        self.current_rank_rows = self.rank_total_rows[self.rank]
        self.current_rank_batches = self.rank_total_batches[self.rank]
        
    def _parallel_get_file_sizes(self, file_paths: List[str]) -> List[Tuple[str, int]]:
        """并行获取文件大小信息"""
        def get_file_size(file_path: str) -> Tuple[str, int]:
            with pq.ParquetFile(file_path) as pf:
                return file_path, pf.metadata.num_rows
                
        with ThreadPoolExecutor() as executor:
            file_sizes = list(executor.map(get_file_size, file_paths))
            
        # 按文件大小降序排序
        return sorted(file_sizes, key=lambda x: x[1], reverse=True)
        
    def _get_worker_info(self) -> WorkerInfo:
        """获取当前worker的相关信息"""
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1
        
        # 计算全局worker ID
        global_worker_id = self.rank * num_workers + worker_id
        
        # 获取预分配的worker信息
        worker_info = self.workers_info[global_worker_id]
        file_paths = worker_info.file_paths.copy()  # 创建副本以免影响原始数据
        
        # 如果需要shuffle，打乱文件顺序
        if self.shuffle and file_paths:
            rng = np.random.RandomState()
            rng.shuffle(file_paths)
            
        # 更新文件路径并返回
        worker_info.file_paths = file_paths
        return worker_info

    def _transform(self, batch_df: pd.DataFrame) -> Dict:
        """将DataFrame转换为模型需要的格式"""
        batch_torch = {}

        # 处理标签列
        if len(self.dataset_config["label_cols"]) > 0:
            batch_torch["label"] = []
            for label_col in self.dataset_config["label_cols"]:
                # batch_torch["label"].append(np.vstack(batch_df[label_col].to_numpy()))
                batch_torch["label"].append(np.vstack(batch_df[label_col].to_numpy())[:, 21].ravel())
            batch_torch["label"] = torch.FloatTensor(np.hstack(batch_torch["label"]))

        batch_torch["feats"] = {
            "self": {},
            "join_edges": {}
        }

        # 处理self特征
        for node_type in self.dataset_config["feat_cols"]["self"]:
            batch_torch["feats"]["self"][node_type] = {}
            for node_table_name in self.dataset_config["feat_cols"]["self"][node_type]:
                batch_torch["feats"]["self"][node_type][node_table_name] = {}
                for scale_func, feat_col in self.dataset_config["feat_cols"]["self"][node_type][node_table_name].items():
                    batch_torch["feats"]["self"][node_type][node_table_name][scale_func] = (
                        torch.FloatTensor(np.vstack(batch_df[feat_col].to_numpy()))
                    )

        # 处理join_edges特征
        for join_edges_name, seq_tokens in self.dataset_config["feat_cols"]["join_edges"].items():
            batch_torch["feats"]["join_edges"][join_edges_name] = {}
            for seq_token_index, tables in seq_tokens.items():
                batch_torch["feats"]["join_edges"][join_edges_name][seq_token_index] = {}
                for table_name, scale_funcs in tables.items():
                    batch_torch["feats"]["join_edges"][join_edges_name][seq_token_index][table_name] = {}
                    for scale_func, feat_col in scale_funcs.items():
                        batch_torch["feats"]["join_edges"][join_edges_name][seq_token_index][table_name][scale_func] = (
                            torch.FloatTensor(np.vstack(batch_df[feat_col].to_numpy()))
                        )

        return batch_torch

    def __iter__(self) -> Iterator[Dict]:
        """实现数据迭代器"""
        worker_info = self._get_worker_info()
        remaining_df = None
        
        # 遍历所有文件
        for file_path in worker_info.file_paths:
            parquet_file = pq.ParquetFile(file_path)
            
            for chunk in parquet_file.iter_batches(batch_size=self.chunk_size):
                chunk = chunk.to_pandas()
                if self.shuffle:
                    chunk = chunk.sample(frac=1)
                    
                # 处理上一轮剩余的数据
                if remaining_df is not None and len(remaining_df) > 0:
                    chunk = pd.concat([remaining_df, chunk], ignore_index=True)
                    remaining_df = None
                
                # 输出完整的batch
                while len(chunk) >= self.batch_size:
                    batch = chunk.iloc[:self.batch_size]
                    chunk = chunk.iloc[self.batch_size:]
                    yield self._transform(batch)
                    
                # 保存剩余数据
                if len(chunk) > 0:
                    remaining_df = chunk
        
        # 处理最后剩余的数据
        if remaining_df is not None and len(remaining_df) > 0:
            if not self.fill_last:
                yield self._transform(remaining_df)
            else:
                # 继续读取文件直到满足batch_size
                for file_path in worker_info.file_paths:
                    parquet_file = pq.ParquetFile(file_path)
                    for chunk in parquet_file.iter_batches(batch_size=self.chunk_size):
                        chunk = chunk.to_pandas()
                        if self.shuffle:
                            chunk = chunk.sample(frac=1)
                        remaining_df = pd.concat([remaining_df, chunk], ignore_index=True)
                        
                        # 一旦达到batch_size就返回
                        if len(remaining_df) >= self.batch_size:
                            yield self._transform(remaining_df.iloc[:self.batch_size])
                            return
                    