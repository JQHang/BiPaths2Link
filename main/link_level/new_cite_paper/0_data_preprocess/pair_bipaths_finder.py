#!/usr/bin/env python
# coding: utf-8

# In[1]:


from joinminer.pyspark import ResilientSparkRunner
from joinminer.graph import TableGraph, read_labeled_samples
from joinminer.python import mkdir, setup_logger, time_costing, read_json_file

from datetime import datetime


# In[2]:


# 获得项目文件夹根目录路径
from joinminer import PROJECT_ROOT

# 日志信息保存文件名
log_files_dir = PROJECT_ROOT + '/data/result_data/log_files/intersectpaths_finder'
log_filename = log_files_dir + f'/{datetime.now().strftime("%Y-%m-%d-%H:%M")}.log'
mkdir(log_files_dir)

logger = setup_logger(log_filename, logger_name = "joinminer")

# Table_graph config
table_graph_config_file = PROJECT_ROOT + '/main/config/table_graphs/AMiner_New_Citation_V1.json'
table_graph_config = read_json_file(table_graph_config_file)

# Graph init
graph = TableGraph(table_graph_config)

# Labeled samples config
labeled_samples_config_file = PROJECT_ROOT + '/main/config/labeled_samples/new_citation_pred_V1.json'
labeled_samples_config = read_json_file(labeled_samples_config_file)

# Relevant bidirectional meta-paths file save path
join_edges_config_file = PROJECT_ROOT + '/main/config/join_edges_types/new_citation_relevant_bipaths.json'


# In[3]:


# spark配置参数
config_dict = {
                "spark.default.parallelism": "80",
                "spark.sql.shuffle.partitions": "160",
                "spark.sql.broadcastTimeout": "3600",
                "spark.driver.memory": "60g",
                "spark.driver.cores": "12",
                "spark.driver.maxResultSize": "0",
                "spark.executor.memory": "16g",
                "spark.executor.cores": "4",
                "spark.executor.instances": "20"
            }

# 启动spark
spark_runner = ResilientSparkRunner(config_dict = config_dict)


# In[4]:


from joinminer.graph import join_edge_query

# 生成全部可用的join_edge类型
join_edge_types = {}

# 遍历可选择的边
for edge_type in graph.edges:
    # 获得对应的节点类型
    linked_node_types = graph.edges[edge_type]["linked_node_types"]

    # 只针对有两个节点的边
    if len(linked_node_types) != 2:
        continue

    # 记录该边从不同node_index出发对应的配置
    join_edge_types[edge_type] = {}
    
    # 依次检查该边的两种方向对应的情况
    for head_node_i, head_node_type in enumerate(linked_node_types):
        # 获得尾结点对应的序号和类型
        tail_node_i = 1 - head_node_i
        tail_node_type = linked_node_types[tail_node_i]

        # 设定该join_edge对join节点的采样数目
        sample_count = 20
            
        # 设定该join_edge名称
        if head_node_i == 0:
            join_edge_name = f"forward_{edge_type}_sample_{sample_count}"
        else:
            join_edge_name = f"backward_{edge_type}_sample_{sample_count}"
        
        # 获得新增的边对应的join_edge配置
        join_edge_schema = {}
        join_edge_schema["join_edge_name"] = join_edge_name
        join_edge_schema["join_nodes_types"] = [head_node_type]
        join_edge_schema["join_nodes_edge_indexes"] = [head_node_i]

        join_edge_schema["edge_type"] = edge_type

        if sample_count > 0:
            edge_sample = {}
            edge_sample["sample_nodes_types"] = join_edge_schema["join_nodes_types"]
            edge_sample["sample_nodes_edge_indexes"] = join_edge_schema["join_nodes_edge_indexes"]
            edge_sample["sample_type"] = "random"
            edge_sample["sample_count"] = sample_count
            join_edge_schema["edge_samples"] = [edge_sample]

        join_edge_schema["add_nodes_types"] = [tail_node_type]
        join_edge_schema["add_nodes_edge_indexes"] = [tail_node_i]

        join_edge_types[edge_type][head_node_i] = join_edge_schema


# In[5]:


from joinminer.graph import edge_to_intra_bipaths

# paths_finder生成的join_edges的基础配置
join_edges_default_config = {}
join_edges_default_config["join_edge_root_path"] = "/user/mart_coo/mart_coo_innov/CompGraph/AMiner/join_edge"
join_edges_default_config["join_edge_table_format"] = "parquet"
join_edges_default_config["join_edges_root_path"] = "/user/mart_coo/mart_coo_innov/CompGraph/AMiner/join_edges"
join_edges_default_config["join_edges_table_format"] = "parquet"

edge_intra_bipaths = spark_runner.run(edge_to_intra_bipaths, graph, labeled_samples_config, join_edge_types, 
                                      join_edges_default_config, 4, 20)


# In[6]:


import json

# 获得relevant_paths
relevant_paths = []
for hop_k in edge_intra_bipaths:
    for tgt_path in edge_intra_bipaths[hop_k]:
        if tgt_path["match_metrics"]["matched_distinct_edge_count"] >= 0:
            relevant_paths.append(tgt_path)
        
# 按照matched_distinct_path_count的数量排序
sorted_relevant_paths = sorted(relevant_paths, key = lambda x: x["match_metrics"]['matched_distinct_edge_count'], reverse=True)

logger.info(f"All relevant path count {len(sorted_relevant_paths)}.")

# 显示排序结果，并保留所需的配置信息
sorted_relevant_paths_config = []
for relevant_path in sorted_relevant_paths:
    logger.info(f"Relevant path {relevant_path['join_edges_name']} cover {relevant_path['match_metrics']['matched_distinct_edge_count']} "
                f"distinct path count.")

    if 'forward_path' in relevant_path and "data" in relevant_path['forward_path']:
        del relevant_path['forward_path']["data"]
    if 'backward_path' in relevant_path and "data" in relevant_path['backward_path']:
        del relevant_path['backward_path']["data"]
    sorted_relevant_paths_config.append(relevant_path)

# 保存结果到对应文件
with open(join_edges_config_file, "w") as f:
    json.dump(sorted_relevant_paths_config, f, indent=4)


# In[ ]:




