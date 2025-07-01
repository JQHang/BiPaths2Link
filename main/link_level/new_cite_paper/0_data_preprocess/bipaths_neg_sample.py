#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys

os.environ['SPARK_HOME']="/software/servers/10k/mart_scr/spark_3.0"
os.environ['PYTHONPATH']="/software/servers/10k/mart_scr/spark_3.0/python:/software/servers/10k/mart_scr/spark_3.0/python/lib/py4j-0.10.9-src.zip"
os.environ['LD_LIBRARY_PATH']="/software/servers/jdk1.8.0_121/lib:/software/servers/jdk1.8.0_121/jre/lib/amd64/server:/software/servers/hope/mart_sch/hadoop/lib/native"
os.environ['PYSPARK_PYTHON']="/usr/local/anaconda3/bin/python3.6"
os.environ['PYSPARK_DRIVER_PYTHON']="/usr/local/anaconda3/bin/python3.6"

sys.path.insert(0, '/software/servers/10k/mart_scr/spark_3.0/python/lib/py4j-0.10.9-src.zip')
sys.path.insert(0, '/software/servers/10k/mart_scr/spark_3.0/python')
sys.path.insert(0, '/software/servers/10k/mart_scr/spark_3.0/python/lib/pyspark.zip')


# In[2]:


from joinminer.pyspark import ResilientSparkRunner
from joinminer.graph import TableGraph, join_edges_list_init, train_inst_config_init
from joinminer.python import mkdir, setup_logger, time_costing, read_json_file

from datetime import datetime
import random
import numpy as np


# In[3]:


# 获得项目文件夹根目录路径
from joinminer import PROJECT_ROOT

# 日志信息保存文件名
log_files_dir = PROJECT_ROOT + '/data/result_data/log_files/bipaths_neg_sample'
log_filename = log_files_dir + f'/{datetime.now().strftime("%Y-%m-%d-%H:%M")}.log'
mkdir(log_files_dir)

logger = setup_logger(log_filename, logger_name = "joinminer")

# Dataset config
dataset_local_path = PROJECT_ROOT + '/data/dataset/AMiner/bipathsnn_train'
dataset_config_file = PROJECT_ROOT + '/data/dataset/AMiner/bipathsnn_train/dataset_config.json'
dataset_config = read_json_file(dataset_config_file)
dataset_config["local_path"] = dataset_local_path
dataset_config["id_columns"] = ["index_0_node_Author_col_0", "index_1_node_Paper_col_0"]


# In[4]:


# spark配置参数
config_dict = {
                "spark.default.parallelism": "400",
                "spark.sql.shuffle.partitions": "800",
                "spark.sql.broadcastTimeout": "3600",
                "spark.driver.memory": "20g",
                "spark.driver.cores": "4",
                "spark.driver.maxResultSize": "0",
                "spark.executor.memory": "16g",
                "spark.executor.cores": "4",
                "spark.executor.instances": "100"
            }

# 启动spark
spark_runner = ResilientSparkRunner(config_dict = config_dict)


# # 负采样函数

# In[5]:


from joinminer.hdfs import hdfs_check_file_exists
from pyspark.sql.functions import rand
from pyspark.sql.functions import col
from pyspark.sql.functions import count as count_

from pyspark.sql.functions import col, rand, floor, row_number, broadcast, lit
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType

def bipaths_neg_sample(spark, df_path, result_path, neg_sample_k):
    # 检测是否已有对应结果
    if hdfs_check_file_exists(result_path + f"/_SUCCESS"):
        return

    # 读取原始样本
    df = spark.read.parquet(df_path)
    
    # 获得全部正样本
    pos_df = df.filter(col("label") == 1)
    
    # 获得正样本数量
    pos_count = pos_df.count()
    
    # 获得要采样的负样本数量
    neg_count = pos_count * neg_sample_k

    print(pos_count, neg_count)
    
    # 获得全部负样本
    neg_df = df.filter(col("label") == 0)
    
    # 随机采样目标数量的负样本
    num_groups = 400
    neg_df = neg_df.withColumn("group_id", floor(rand() * num_groups).cast(IntegerType()))

    window = Window.partitionBy("group_id").orderBy(rand())
    neg_df = neg_df.withColumn("row_num", row_number().over(window))
    
    base_count_per_group = neg_count // num_groups  # 整除部分
    remainder = neg_count % num_groups  # 余数
    
    allocation_data = []
    for group_id in range(num_groups):
        if group_id == num_groups - 1:  # 最后一个group加上余数
            keep_count = base_count_per_group + remainder
        else:
            keep_count = base_count_per_group
        allocation_data.append((group_id, keep_count))
    
    allocation_df = spark.createDataFrame(
        allocation_data, 
        ["group_id", "keep_count"]
    )

    neg_df = neg_df.join(broadcast(allocation_df), on="group_id", how="inner")
    
    neg_df = neg_df.filter(col("row_num") <= col("keep_count")).drop("group_id", "row_num", "keep_count")
    
    # 合并正负样本
    sample_df = pos_df.unionByName(neg_df)

    # 按每个文件1万个样本分区
    partition_count = (pos_count + neg_count) // 10000
    sample_df = sample_df.repartition(partition_count)
    
    # 保存结果
    sample_df.write.mode("overwrite").parquet(result_path)

    # 最后检查下结果对不对
    sample_df = spark.read.parquet(result_path)
    sample_df.groupBy("label").agg(count_("*").alias("count")).show()
    
    return


# In[6]:


# 先按train valid test分别进行neg采样
df_path = "/user/mart_coo/mart_coo_innov/CompGraph/AMiner/task/new_citation_v1/dataset/batch_id=train_sample/batch_id=0/sample_type=train"
result_path = "/user/mart_coo/mart_coo_innov/CompGraph/AMiner/task/new_citation_v1/neg_sample/sample_type=train_neg_3"
spark_runner.run(bipaths_neg_sample, df_path, result_path, 3)

df_path = "/user/mart_coo/mart_coo_innov/CompGraph/AMiner/task/new_citation_v1/dataset/batch_id=train_sample/batch_id=0/sample_type=train"
result_path = "/user/mart_coo/mart_coo_innov/CompGraph/AMiner/task/new_citation_v1/neg_sample/sample_type=train_neg_5"
spark_runner.run(bipaths_neg_sample, df_path, result_path, 5)

df_path = "/user/mart_coo/mart_coo_innov/CompGraph/AMiner/task/new_citation_v1/dataset/batch_id=train_sample/batch_id=0/sample_type=train"
result_path = "/user/mart_coo/mart_coo_innov/CompGraph/AMiner/task/new_citation_v1/neg_sample/sample_type=train_neg_7"
spark_runner.run(bipaths_neg_sample, df_path, result_path, 7)

df_path = "/user/mart_coo/mart_coo_innov/CompGraph/AMiner/task/new_citation_v1/dataset/batch_id=train_sample/batch_id=0/sample_type=train"
result_path = "/user/mart_coo/mart_coo_innov/CompGraph/AMiner/task/new_citation_v1/neg_sample/sample_type=train_neg_10"
spark_runner.run(bipaths_neg_sample, df_path, result_path, 10)

df_path = "/user/mart_coo/mart_coo_innov/CompGraph/AMiner/task/new_citation_v1/dataset/batch_id=train_sample/batch_id=0/sample_type=valid"
result_path = "/user/mart_coo/mart_coo_innov/CompGraph/AMiner/task/new_citation_v1/neg_sample/sample_type=valid_neg_19"
spark_runner.run(bipaths_neg_sample, df_path, result_path, 19)

df_path = "/user/mart_coo/mart_coo_innov/CompGraph/AMiner/task/new_citation_v1/dataset/batch_id=train_sample/batch_id=0/sample_type=test"
result_path = "/user/mart_coo/mart_coo_innov/CompGraph/AMiner/task/new_citation_v1/neg_sample/sample_type=test_neg_19"
spark_runner.run(bipaths_neg_sample, df_path, result_path, 19)


# # 只保留特征列

# In[7]:


# 获得要保留的目标列
select_cols = ["index_0_node_Author_col_0", "index_1_node_Paper_col_0", "label"]

# 获得涉及到的count列
count_cols = []

for pair_node_type in ["head_node", "tail_node"]:
    # 获得结点特征
    feat_col = dataset_config[f"{pair_node_type}_token_config"]["feat_col"]
    select_cols.append(feat_col)

for path_dir_type in ["forward_paths", "backward_paths", "bipaths"]:
    # 依次处理各个数据对应的具体类型
    for path_name in dataset_config[path_dir_type]:
        # 获得该path的config
        path_config = dataset_config[path_dir_type][path_name]

        # 获得该路径clt到的列的前缀
        clt_col_prefix = path_config["clt_col_prefix"]
        
        # 获得记录各行实际collect到的path的数据的列
        clt_count_col = path_config["clt_count_col"]
        select_cols.append(clt_count_col)
        count_cols.append(clt_count_col)
        
        for token_config in path_config["seq_tokens"]:
            # 检查该token是否有对应特征
            if token_config["feat_col"] is not None:
                token_feat_col = token_config["feat_col"]
                clt_feat_col = f"{clt_col_prefix}_clt_{token_feat_col}"
                select_cols.append(clt_feat_col)


# In[8]:


import os
import subprocess
from pyspark.sql.functions import col

from joinminer.hdfs import hdfs_check_file_exists

# 只保留特征列给bipathsnn使用
# 均匀分割为合适的文件数目，比如每个文件两万个样本
def bipathsnn_dataset_prepare(spark, df_path, select_cols, hdfs_path, local_path):
    # 检查本地是否已有对应结果
    if os.path.isfile(local_path + "/_SUCCESS"):
        return
        
    # 检查hdfs内是否已有对应结果
    if not hdfs_check_file_exists(hdfs_path + "/_SUCCESS"):
        # 读取原始数据
        df = spark.read.parquet(df_path)
        
        # 只保留目标列
        df = df.select(*select_cols)

        # 将count列na值都补0，并转为int类型
        df = df.fillna(0, subset=count_cols)
        for count_col in count_cols:
            df = df.withColumn(count_col, col(count_col).cast("int"))
    
        # 获得样本总数
        sample_count = df.count()
        
        # 按每个文件2万个样本分区
        partition_count = sample_count // 20000
        df = df.repartition(partition_count)
        
        # 保存结果
        df.write.mode("overwrite").parquet(hdfs_path)
    
    # 将结果复制到本地
    os.makedirs(local_path, exist_ok=True)
    subprocess.run(["hdfs", "dfs", "-copyToLocal", hdfs_path + "/*", local_path], check = True)
    return


# In[9]:


df_path = "/user/mart_coo/mart_coo_innov/CompGraph/AMiner/task/new_citation_v1/neg_sample/sample_type=train_neg_3"
hdfs_path = "/user/mart_coo/mart_coo_innov/CompGraph/AMiner/task/new_citation_v1/neg_sample_select/sample_type=train_neg_3"
local_path = dataset_local_path + "/sample_type=train_neg_3"
spark_runner.run(bipathsnn_dataset_prepare, df_path, select_cols, hdfs_path, local_path)

df_path = "/user/mart_coo/mart_coo_innov/CompGraph/AMiner/task/new_citation_v1/neg_sample/sample_type=train_neg_5"
hdfs_path = "/user/mart_coo/mart_coo_innov/CompGraph/AMiner/task/new_citation_v1/neg_sample_select/sample_type=train_neg_5"
local_path = dataset_local_path + "/sample_type=train_neg_5"
spark_runner.run(bipathsnn_dataset_prepare, df_path, select_cols, hdfs_path, local_path)

df_path = "/user/mart_coo/mart_coo_innov/CompGraph/AMiner/task/new_citation_v1/neg_sample/sample_type=train_neg_7"
hdfs_path = "/user/mart_coo/mart_coo_innov/CompGraph/AMiner/task/new_citation_v1/neg_sample_select/sample_type=train_neg_7"
local_path = dataset_local_path + "/sample_type=train_neg_7"
spark_runner.run(bipathsnn_dataset_prepare, df_path, select_cols, hdfs_path, local_path)

df_path = "/user/mart_coo/mart_coo_innov/CompGraph/AMiner/task/new_citation_v1/neg_sample/sample_type=train_neg_10"
hdfs_path = "/user/mart_coo/mart_coo_innov/CompGraph/AMiner/task/new_citation_v1/neg_sample_select/sample_type=train_neg_10"
local_path = dataset_local_path + "/sample_type=train_neg_10"
spark_runner.run(bipathsnn_dataset_prepare, df_path, select_cols, hdfs_path, local_path)

df_path = "/user/mart_coo/mart_coo_innov/CompGraph/AMiner/task/new_citation_v1/neg_sample/sample_type=valid_neg_19"
hdfs_path = "/user/mart_coo/mart_coo_innov/CompGraph/AMiner/task/new_citation_v1/neg_sample_select/sample_type=valid_neg_19"
local_path = dataset_local_path + "/sample_type=valid_neg_19"
spark_runner.run(bipathsnn_dataset_prepare, df_path, select_cols, hdfs_path, local_path)

df_path = "/user/mart_coo/mart_coo_innov/CompGraph/AMiner/task/new_citation_v1/neg_sample/sample_type=test_neg_19"
hdfs_path = "/user/mart_coo/mart_coo_innov/CompGraph/AMiner/task/new_citation_v1/neg_sample_select/sample_type=test_neg_19"
local_path = dataset_local_path + "/sample_type=test_neg_19"
spark_runner.run(bipathsnn_dataset_prepare, df_path, select_cols, hdfs_path, local_path)


# # 转化为graph形式

# In[10]:


from joinminer.graph import standard_node_col_name, graph_token_node_col_name
from joinminer.pyspark import pyspark_optimal_save

from pyspark.sql.functions import arrays_zip, explode, col as col_

# 转为subgraph给普通模型使用
def subgraph_dataset_prepare(spark, df_path, subgraph_path, dataset_config):
    # 检查是否已经完成subgraph的生成
    if hdfs_check_file_exists(subgraph_path + "/_SUCCESS"):
        return

    # 读取对应的数据
    df = spark.read.parquet(df_path)
    
    # 先保留带标签的目标边
    if not hdfs_check_file_exists(subgraph_path + "/raw/labeled_edges/_SUCCESS"):
        # 只保留目标边的数据
        labeled_edges_df = df.select("index_0_node_Author_col_0", "index_1_node_Paper_col_0", "graph_year", "label")

        # 保存结果
        pyspark_optimal_save(labeled_edges_df, subgraph_path + "/raw/labeled_edges", "parquet", "overwrite", ["graph_year"])
    
    # 再保留目标边上的节点对应的特征
    for pair_node_type in ["head_node", "tail_node"]:
        # 获得结点类型
        token_type = dataset_config[f"{pair_node_type}_token_config"]["token_type"]

        # 检查对应数据是否存在
        if hdfs_check_file_exists(subgraph_path + f"/raw/nodes/{token_type}/src={pair_node_type}/_SUCCESS"):
            continue
        
        # 获得节点id列
        if pair_node_type == "head_node":
            id_col = "index_0_node_Author_col_0"
        else:
            id_col = "index_1_node_Paper_col_0"
            
        # 获得结点特征列
        feat_col = dataset_config[f"{pair_node_type}_token_config"]["feat_col"]

        # 只保留目标数据
        nodes_df = df.select(id_col, "graph_year", feat_col)
        nodes_df = nodes_df.dropDuplicates([id_col, "graph_year"])

        # 修正列名
        nodes_df = nodes_df.withColumnRenamed(id_col, graph_token_node_col_name(token_type, None, 0))
        nodes_df = nodes_df.withColumnRenamed(feat_col, "feat_vec")
        
        # 保存结果
        col_sizes = {"feat_vec": dataset_config["token_feat_len"][token_type]}
        pyspark_optimal_save(nodes_df, subgraph_path + f"/raw/nodes/{token_type}/src={pair_node_type}", "parquet", "overwrite", 
                             ["graph_year"], col_sizes = col_sizes)

    # 依次处理各类型路径
    for path_dir_type in ["forward_paths", "backward_paths"]:
        # 依次处理各个数据对应的具体类型
        for path_name in dataset_config[path_dir_type]:
            print(path_dir_type, path_name)
            
            # 获得该path的config
            path_config = dataset_config[path_dir_type][path_name]

            # 获得该路径clt到的列的前缀
            clt_col_prefix = path_config["clt_col_prefix"]
        
            # 依次处理路径包含的各个token
            for token_i, token_config in enumerate(path_config["seq_tokens"]):
                # 获得token类型
                token_type = token_config["token_type"]

                # 获得特征列
                token_feat_col = f"{clt_col_prefix}_clt_" + token_config["feat_col"]
                
                # 获得该token是节点还是特征
                if token_config['node_edge_type'] == "node":
                    # 检测是否已经完成
                    if hdfs_check_file_exists(subgraph_path + f"/raw/nodes/{token_type}/src={path_name}_{token_i}/_SUCCESS"):
                        continue
                    
                    # 获得对应的id列
                    token_id_col = f"{clt_col_prefix}_clt_" + standard_node_col_name(token_type, token_config['node_pos_index'], 0)

                    # 只保留目标数据
                    nodes_df = df.select(token_id_col, "graph_year", token_feat_col)
                    
                    # 展开list列
                    nodes_df = nodes_df.select(
                        'graph_year',
                        explode(arrays_zip(token_id_col, token_feat_col)).alias('zipped')
                    ).select(
                        'graph_year',
                        col(f'zipped.{token_id_col}').alias(token_id_col),
                        col(f'zipped.{token_feat_col}').alias('feat_vec')
                    )
                    
                    # 去重
                    nodes_df = nodes_df.dropDuplicates([token_id_col, "graph_year"])
                    
                    # 修正id列名
                    nodes_df = nodes_df.withColumnRenamed(token_id_col, 
                                                          graph_token_node_col_name(token_type, None, 0))

                    # 保存结果
                    col_sizes = {"feat_vec": dataset_config["token_feat_len"][token_type]}
                    pyspark_optimal_save(nodes_df, subgraph_path + f"/raw/nodes/{token_type}/src={path_name}_{token_i}", "parquet", 
                                         "overwrite", ["graph_year"], col_sizes = col_sizes)
        
                else:
                    # 检测是否已经完成
                    if hdfs_check_file_exists(subgraph_path + f"/raw/edges/{token_type}/src={path_name}_{token_i}/_SUCCESS"):
                        continue

                    # 检查是否是第一个token 
                    if token_i == 0:
                        # 获得对应的id列
                        if path_dir_type == "forward_paths":
                            token_id_col_0 = "index_0_node_Author_col_0"
                            token_node_type_0 = "Author"
                        else:
                            token_id_col_0 = "index_1_node_Paper_col_0"
                            token_node_type_0 = "Paper"
                            
                        token_id_col_1 = f"{clt_col_prefix}_clt_" + standard_node_col_name(path_config["seq_tokens"][1]["token_type"], 
                                                                                           token_config['edge_pos_indexes'][1], 0)
                        token_node_type_1 = path_config["seq_tokens"][1]["token_type"]
                        
                        # 只保留目标数据
                        edges_df = df.select(token_id_col_0, token_id_col_1, "graph_year", token_feat_col)

                        # 展开list列
                        edges_df = edges_df.select(
                            token_id_col_0,
                            'graph_year',
                            explode(arrays_zip(token_id_col_1, token_feat_col)).alias('zipped')
                        ).select(
                            token_id_col_0,
                            col(f'zipped.{token_id_col_1}').alias(token_id_col_1),
                            'graph_year',
                            col(f'zipped.{token_feat_col}').alias('feat_vec')
                        )

                    else:
                        # 获得对应的id列
                        token_id_col_0 = f"{clt_col_prefix}_clt_" + standard_node_col_name(path_config["seq_tokens"][token_i - 1]["token_type"], 
                                                                                           token_config['edge_pos_indexes'][0], 0)
                        token_node_type_0 = path_config["seq_tokens"][token_i - 1]["token_type"]
                            
                        token_id_col_1 = f"{clt_col_prefix}_clt_" + standard_node_col_name(path_config["seq_tokens"][token_i + 1]["token_type"], 
                                                                                           token_config['edge_pos_indexes'][1], 0)
                        token_node_type_1 = path_config["seq_tokens"][token_i + 1]["token_type"]
                        
                        # 只保留目标数据
                        edges_df = df.select(token_id_col_0, token_id_col_1, "graph_year", token_feat_col)

                        # 展开list列
                        edges_df = edges_df.select(
                            'graph_year',
                            explode(arrays_zip(token_id_col_0, token_id_col_1, token_feat_col)).alias('zipped')
                        ).select(
                            col(f'zipped.{token_id_col_0}').alias(token_id_col_0),
                            col(f'zipped.{token_id_col_1}').alias(token_id_col_1),
                            'graph_year',
                            col(f'zipped.{token_feat_col}').alias('feat_vec')
                        )

                    # 去重 
                    edges_df = edges_df.dropDuplicates([token_id_col_0, token_id_col_1, "graph_year"])

                    # 修正id列名
                    node_type_0 = token_type.split("___linked_with___")[0]
                    node_type_1 = token_type.split("___linked_with___")[1]
                    
                    if token_node_type_0 == node_type_0:
                        edges_df = edges_df.withColumnRenamed(token_id_col_0, 
                                                              graph_token_node_col_name(token_node_type_0, 0, 0))
                        edges_df = edges_df.withColumnRenamed(token_id_col_1, 
                                                              graph_token_node_col_name(token_node_type_1, 1, 0))
                    else:
                        edges_df = edges_df.withColumnRenamed(token_id_col_0, 
                                                              graph_token_node_col_name(token_node_type_0, 1, 0))
                        edges_df = edges_df.withColumnRenamed(token_id_col_1, 
                                                              graph_token_node_col_name(token_node_type_1, 0, 0))

                    # 保存结果
                    col_sizes = {"feat_vec": dataset_config["token_feat_len"][token_type]}
                    pyspark_optimal_save(edges_df, subgraph_path + f"/raw/edges/{token_type}/src={path_name}_{token_i}", "parquet", 
                                         "overwrite", ["graph_year"], col_sizes = col_sizes)

    # 可以增加全部完成的标志
    
    return


# In[11]:


df_path = "/user/mart_coo/mart_coo_innov/CompGraph/AMiner/task/new_citation_v1/neg_sample/sample_type=train_neg_10"
subgraph_path = "/user/mart_coo/mart_coo_innov/CompGraph/AMiner/task/new_citation_v1/subgraph/sample_type=train_neg_10"
spark_runner.run(subgraph_dataset_prepare, df_path, subgraph_path, dataset_config)

df_path = "/user/mart_coo/mart_coo_innov/CompGraph/AMiner/task/new_citation_v1/neg_sample/sample_type=valid_neg_19"
subgraph_path = "/user/mart_coo/mart_coo_innov/CompGraph/AMiner/task/new_citation_v1/subgraph/sample_type=valid_neg_19"
spark_runner.run(subgraph_dataset_prepare, df_path, subgraph_path, dataset_config)

df_path = "/user/mart_coo/mart_coo_innov/CompGraph/AMiner/task/new_citation_v1/neg_sample/sample_type=test_neg_19"
subgraph_path = "/user/mart_coo/mart_coo_innov/CompGraph/AMiner/task/new_citation_v1/subgraph/sample_type=test_neg_19"
spark_runner.run(subgraph_dataset_prepare, df_path, subgraph_path, dataset_config)


# # 合并为一个文件

# In[12]:


from joinminer.hdfs import hdfs_parquets_to_pandas

import gc
import joblib
from functools import reduce
from pyspark.sql.functions import lit

def subgraph_dataset_local_prepare(spark, subgraph_path, dataset_local_path, dataset_config):
    # 检查是否已有对应结果
    if os.path.isfile(dataset_local_path + "/raw_dataset.joblib"):
        return

    # 先依次对各个node的结果合并后去重
    for node_type in ["Author", "Keyword", "Org", "Paper"]:
        # 检测是否已有对应结果
        if hdfs_check_file_exists(subgraph_path + f"/union/nodes/{node_type}/_SUCCESS"):
            continue
        
        # 记录各个来源的nodes_df
        nodes_df_list = []

        # 获得id列
        node_id_col = graph_token_node_col_name(node_type, None, 0)
        
        # 依次读取各个sample_type中的结果
        for sample_type in ["train_neg_10", "valid_neg_19", "test_neg_19"]:
            nodes_df = spark.read.parquet(subgraph_path + f"/sample_type={sample_type}/raw/nodes/{node_type}")

            nodes_df = nodes_df.select(node_id_col, "graph_year", "feat_vec")

            nodes_df_list.append(nodes_df)
            
        # 合并全部的nodes_df
        nodes_df = reduce(lambda df1, df2: df1.unionByName(df2), nodes_df_list)

        # 去重
        nodes_df = nodes_df.dropDuplicates([node_id_col, "graph_year"])
        
        # 生成id号
        indexed_rdd = nodes_df.rdd.zipWithIndex()
        nodes_df = spark.createDataFrame(
            indexed_rdd.map(lambda row: (*row[0], row[1])),
            schema=nodes_df.schema.names + ["node_index"]
        )
        
        # 保存结果
        col_sizes = {"feat_vec": dataset_config["token_feat_len"][node_type]}
        pyspark_optimal_save(nodes_df, subgraph_path + f"/union/nodes/{node_type}", "parquet", 
                             "overwrite", ["graph_year"], col_sizes = col_sizes)

    # 再依次对各个边合并后去重
    for edge_type in ["Author___linked_with___Org", "Paper___linked_with___Author", "Paper___linked_with___Keyword", 
                      "Paper___linked_with___Org", "Paper___linked_with___Paper"]:
        # 检测是否已有对应结果
        if hdfs_check_file_exists(subgraph_path + f"/union/edges/{edge_type}/_SUCCESS"):
            continue

        # 记录各个来源的edges_df
        edges_df_list = []

        # 获得id列
        node_type_0 = edge_type.split("___linked_with___")[0]
        node_type_1 = edge_type.split("___linked_with___")[1]
        
        node_id_col_0 = graph_token_node_col_name(node_type_0, 0, 0)
        node_id_col_1 = graph_token_node_col_name(node_type_1, 1, 0)
        
        # 依次读取各个sample_type中的结果
        for sample_type in ["train_neg_10", "valid_neg_19", "test_neg_19"]:
            edges_df = spark.read.parquet(subgraph_path + f"/sample_type={sample_type}/raw/edges/{edge_type}")

            edges_df = edges_df.select(node_id_col_0, node_id_col_1, "graph_year", "feat_vec")

            edges_df_list.append(edges_df)
            
        # 合并全部的edges_df
        edges_df = reduce(lambda df1, df2: df1.unionByName(df2), edges_df_list)

        # 去重
        edges_df = edges_df.dropDuplicates([node_id_col_0, node_id_col_1, "graph_year"])
        
        # 生成id号
        indexed_rdd = edges_df.rdd.zipWithIndex()
        edges_df = spark.createDataFrame(
            indexed_rdd.map(lambda row: (*row[0], row[1])),
            schema=edges_df.schema.names + ["edge_index"]
        )

        # 添加节点对应的id号
        nodes_df = spark.read.parquet(subgraph_path + f"/union/nodes/{node_type_0}")
        node_id_col = graph_token_node_col_name(node_type_0, None, 0)
        nodes_df = nodes_df.select(node_id_col, "graph_year", "node_index")
        nodes_df = nodes_df.withColumnRenamed(node_id_col, node_id_col_0)
        nodes_df = nodes_df.withColumnRenamed("node_index", "head_node_index")
        edges_df = edges_df.join(nodes_df, on = [node_id_col_0, "graph_year"], how = "left")

        nodes_df = spark.read.parquet(subgraph_path + f"/union/nodes/{node_type_1}")
        node_id_col = graph_token_node_col_name(node_type_1, None, 0)
        nodes_df = nodes_df.select(node_id_col, "graph_year", "node_index")
        nodes_df = nodes_df.withColumnRenamed(node_id_col, node_id_col_1)
        nodes_df = nodes_df.withColumnRenamed("node_index", "tail_node_index")
        edges_df = edges_df.join(nodes_df, on = [node_id_col_1, "graph_year"], how = "left")
        
        # 保存结果
        col_sizes = {"feat_vec": dataset_config["token_feat_len"][edge_type]}
        pyspark_optimal_save(edges_df, subgraph_path + f"/union/edges/{edge_type}", "parquet", 
                             "overwrite", ["graph_year"], col_sizes = col_sizes)

    # 检测是否已有处理后的带标签的目标边
    if not hdfs_check_file_exists(subgraph_path + f"/union/labeled_edges/_SUCCESS"):
        # 记录各个来源的edges_df
        labeled_edges_df_list = []

        # 获得id列
        node_type_0 = "Author"
        node_type_1 = "Paper"
        
        node_id_col_0 = graph_token_node_col_name(node_type_0, 0, 0)
        node_id_col_1 = graph_token_node_col_name(node_type_1, 1, 0)
        
        # 依次读取各个sample_type中的结果
        for sample_type in ["train_neg_10", "valid_neg_19", "test_neg_19"]:
            labeled_edges_df = spark.read.parquet(subgraph_path + f"/sample_type={sample_type}/raw/labeled_edges")

            labeled_edges_df = labeled_edges_df.select(node_id_col_0, node_id_col_1, "graph_year", "label")
            
            labeled_edges_df = labeled_edges_df.withColumn("sample_type", lit(sample_type.split("_neg_")[0]))
            
            labeled_edges_df_list.append(labeled_edges_df)
            
        # 合并全部的edges_df
        labeled_edges_df = reduce(lambda df1, df2: df1.unionByName(df2), labeled_edges_df_list)

        # 生成id号
        indexed_rdd = labeled_edges_df.rdd.zipWithIndex()
        labeled_edges_df = spark.createDataFrame(
            indexed_rdd.map(lambda row: (*row[0], row[1])),
            schema=labeled_edges_df.schema.names + ["edge_index"]
        )

        # 添加节点对应的id号
        nodes_df = spark.read.parquet(subgraph_path + f"/union/nodes/{node_type_0}")
        node_id_col = graph_token_node_col_name(node_type_0, None, 0)
        nodes_df = nodes_df.select(node_id_col, "graph_year", "node_index")
        nodes_df = nodes_df.withColumnRenamed(node_id_col, node_id_col_0)
        nodes_df = nodes_df.withColumnRenamed("node_index", "head_node_index")
        labeled_edges_df = labeled_edges_df.join(nodes_df, on = [node_id_col_0, "graph_year"], how = "left")

        nodes_df = spark.read.parquet(subgraph_path + f"/union/nodes/{node_type_1}")
        node_id_col = graph_token_node_col_name(node_type_1, None, 0)
        nodes_df = nodes_df.select(node_id_col, "graph_year", "node_index")
        nodes_df = nodes_df.withColumnRenamed(node_id_col, node_id_col_1)
        nodes_df = nodes_df.withColumnRenamed("node_index", "tail_node_index")
        labeled_edges_df = labeled_edges_df.join(nodes_df, on = [node_id_col_1, "graph_year"], how = "left")
        
        # 保存结果
        pyspark_optimal_save(labeled_edges_df, subgraph_path + f"/union/labeled_edges", "parquet", 
                             "overwrite", ["graph_year"])
    
    # 获得要保存的整个dataset数据
    graph_dataset = {}
    graph_dataset["num_nodes_dict"] = {}
    graph_dataset["node_feat_dict"] = {}
    graph_dataset["edge_index_dict"] = {}
    graph_dataset["edge_feat_dict"] = {}
    graph_dataset["edge_label_dict"] = {}
    graph_dataset["edge_index_split"] = {}

    # 先记录各个node对应的节点数量，如果有特征则记录特征
    for node_type in ["Author", "Keyword", "Org", "Paper"]:
        logger.info(f"Process node {node_type}.")

        # 获得汇总后的节点信息的存储位置
        node_result_hdfs_path = subgraph_path + f"/union/nodes/{node_type}"

        # 获得要读取的列
        columns = ["node_index", "feat_vec"]

        # 读取对应的文件为pandas
        node_pd = hdfs_parquets_to_pandas(node_result_hdfs_path, columns)

        # 按node_index排序
        node_pd = node_pd.sort_values(by="node_index")

        # 记录节点总数
        graph_dataset["num_nodes_dict"][node_type] = node_pd.shape[0]

        # 记录节点特征
        graph_dataset["node_feat_dict"][node_type] = np.stack(node_pd["feat_vec"].values).astype(np.float32)

        # 释放空间
        del node_pd
        gc.collect()

    # 记录各个edge对应的头尾节点index，如果有特征则记录特征
    for edge_type in ["Author___linked_with___Org", "Paper___linked_with___Author", "Paper___linked_with___Keyword", 
                      "Paper___linked_with___Org", "Paper___linked_with___Paper"]:
        logger.info(f"Process edge {edge_type}.")
        
        # 获得汇总后的节点信息的存储位置
        edge_result_hdfs_path = subgraph_path + f"/union/edges/{edge_type}"

        # 获得要读取的列
        columns = ["edge_index", "head_node_index", "tail_node_index", "feat_vec"]

        # 读取对应的文件为pandas
        edge_pd = hdfs_parquets_to_pandas(edge_result_hdfs_path, columns)

        # 按edge_index排序
        edge_pd = edge_pd.sort_values(by="edge_index")

        # 获得该edge类型对应的三元组表示
        node_type_0 = edge_type.split("___linked_with___")[0]
        node_type_1 = edge_type.split("___linked_with___")[1]
        edge_type_tuple = (node_type_0, edge_type, node_type_1)
        
        # 获得edge对应的node index
        graph_dataset["edge_index_dict"][edge_type_tuple] = np.vstack([edge_pd['head_node_index'].astype(int).values, 
                                                                       edge_pd['tail_node_index'].astype(int).values])

        # 获得edge对应的特征
        graph_dataset["edge_feat_dict"][edge_type_tuple] = np.stack(edge_pd["feat_vec"].values).astype(np.float32)

        # 释放空间
        del edge_pd
        gc.collect()

    # 获得汇总后的带标签的边的位置
    labeled_edges_hdfs_path = subgraph_path + f"/union/labeled_edges"
    
    # 获得要读取的列
    columns = ["edge_index", "head_node_index", "tail_node_index", "label", "sample_type"]

    # 读取对应的文件为pandas
    edge_pd = hdfs_parquets_to_pandas(labeled_edges_hdfs_path, columns)

    # 按edge_index排序
    edge_pd = edge_pd.sort_values(by="edge_index")

    # 获得表示edge类型的三元组
    edge_type_tuple = ("Author", "labeled_target_edges", "Paper")
    
    # 获得edge对应的node index
    graph_dataset["edge_index_dict"][edge_type_tuple] = np.vstack([edge_pd['head_node_index'].astype(int).values, 
                                                                   edge_pd['tail_node_index'].astype(int).values])

    # 记录目标边对应的index label
    graph_dataset["edge_label_dict"][edge_type_tuple] = edge_pd['label'].values
    
    # 记录目标边对应的split_index
    for sample_type in ["train", "valid", "test"]:
        graph_dataset["edge_index_split"][sample_type] = edge_pd[edge_pd["sample_type"] == sample_type]['edge_index'].astype(int).values
    
    # 保存对应结果
    joblib.dump(graph_dataset, dataset_local_path + "/raw_dataset.joblib")
    
    return


# In[13]:


# 获得结果在本地的存储路径
subgraph_path = "/user/mart_coo/mart_coo_innov/CompGraph/AMiner/task/new_citation_v1/subgraph"

dataset_local_path = PROJECT_ROOT + "/data/dataset/AMiner/bipaths_subgraph"
os.makedirs(dataset_local_path, exist_ok=True)

spark_runner.run(subgraph_dataset_local_prepare, subgraph_path, dataset_local_path, dataset_config)


# In[ ]:




