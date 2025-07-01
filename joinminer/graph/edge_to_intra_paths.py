from .instances_loader import read_labeled_samples
from .join_edges_init import join_edges_init, standard_node_col_name, standard_feat_col_name
from .join_edges_query import join_edges_query
from .edge_to_intra_path_types import edge_to_intra_path_types
from joinminer.pyspark import rename_columns, pyspark_optimal_save
from joinminer.hdfs import hdfs_check_file_exists, hdfs_save_json, hdfs_read_json

import copy
import logging
from pyspark.sql.functions import col

# 获得logger
logger = logging.getLogger(__name__)

def edge_to_intra_paths(spark, graph, labeled_samples_config, join_edge_types, join_edges_default_config, max_hop_k, max_neighbor):
    # 先获得搜索基于的样本信息
    samples_df = read_labeled_samples(spark, graph, labeled_samples_config, logger = logger)
    
    # 基于train_instances获得目标边的信息(只用训练样本做目标边)
    target_edges = {}
    target_edges["head_node_type"] = labeled_samples_config["nodes_types"][0]
    target_edges["head_node_cols"] = [x[1] for x in labeled_samples_config["nodes_cols_to_aliases"][0]]
    target_edges["tail_node_type"] = labeled_samples_config["nodes_types"][1]
    target_edges["tail_node_cols"] = [x[1] for x in labeled_samples_config["nodes_cols_to_aliases"][1]]
    target_edges["result_path"] = labeled_samples_config["task_data_path"] + "/train"
    
    tgt_edges_cols = target_edges["head_node_cols"] + target_edges["tail_node_cols"] + graph.graph_time_cols_alias
    tgt_edges_df = samples_df.filter("sample_type = 'train'").select(tgt_edges_cols).distinct()
    target_edges["data"] = tgt_edges_df.persist()

    # 基于目标边设计的时间
    distinct_time_cols_rows = tgt_edges_df.select(graph.graph_time_cols_alias).distinct().collect()
    train_samples_time_cols_values = [[row[c] for c in graph.graph_time_cols_alias] for row in distinct_time_cols_rows]

    logger.info(f"Train samples related to graph times: {train_samples_time_cols_values}")
    
    # 通过dfs获得以target_edges的头结点为起点的全部k跳内的符合target_edges的路径
    k_hop_pair_paths = edge_to_intra_path_types(target_edges, join_edge_types, max_hop_k, max_neighbor)

    # 获得每跳要计算哪些路径(这样可以知道所需的运算次数)
    tgt_k_hop_paths = {}
    for hop_k in range(1, max_hop_k + 1):
        # 先记录该跳对应的k_hop_pair_paths
        if hop_k in k_hop_pair_paths:
            tgt_k_hop_paths[hop_k] = copy.deepcopy(k_hop_pair_paths[hop_k])
        else:
            tgt_k_hop_paths[hop_k] = []
            
        # 查看其中的路径的各级父路径是否存在
        for k_hop_pair_path in k_hop_pair_paths[hop_k]:
            # 依次检查各级父路径
            for parent_hop_k in range(1, hop_k):
                parent_path_schema = k_hop_pair_path[:parent_hop_k]

                # 如果没有父路径，则将其加入计算目标中
                if parent_path_schema not in tgt_k_hop_paths[parent_hop_k]:
                    tgt_k_hop_paths[parent_hop_k].append(parent_path_schema)

    for hop_k in range(1, max_hop_k + 1):
        logger.info(f"Require {len(tgt_k_hop_paths[hop_k])} paths at {hop_k}-th hop")

    # 获得目标边的全部去重后的起始节点
    head_nodes_id_col_aliases = []
    for col_i in range(len(target_edges["head_node_cols"])):
        node_col = target_edges["head_node_cols"][col_i]
        node_col_alias = standard_node_col_name(target_edges["head_node_type"], 0, col_i)
        head_nodes_id_col_aliases.append(col(node_col).alias(node_col_alias))
    for time_col in graph.graph_time_cols_alias:
        head_nodes_id_col_aliases.append(col(time_col).alias(time_col))
    head_nodes_df = target_edges["data"].select(*head_nodes_id_col_aliases).distinct()
    
    # 将目标边的起始节点作为路径的目标点的query配置
    head_query_config = {}
    head_query_config["graph_time_values"] = train_samples_time_cols_values
    head_query_config["tgt_query_nodes"] = {}
    head_query_config["tgt_query_nodes"]["result_path"] = target_edges["result_path"] + "/k_hop_paths"
    head_query_config["tgt_query_nodes"]["df"] = head_nodes_df

    # 设定这组query的query node配置
    query_nodes_config = {}
    query_nodes_config["query_nodes_types"] = [target_edges["head_node_type"]]
    query_nodes_config["query_nodes_indexes"] = [0]

    # 依次计算各个tgt_path,并获得其中的pair_path的匹配结果 
    related_k_hop_pair_paths = {}
    for hop_k in range(1, max_hop_k + 1):
        # 基于该跳对应的跳数修正target_edges的列名
        tgt_edges_node_col_to_aliases = {}
        for col_i in range(len(target_edges["head_node_cols"])):
            node_col = target_edges["head_node_cols"][col_i]
            node_col_alias = standard_node_col_name(target_edges["head_node_type"], 0, col_i)
            tgt_edges_node_col_to_aliases[node_col] = node_col_alias
            
        for col_i in range(len(target_edges["tail_node_cols"])):
            node_col = target_edges["tail_node_cols"][col_i]
            node_col_alias = standard_node_col_name(target_edges["tail_node_type"], hop_k, col_i)
            tgt_edges_node_col_to_aliases[node_col] = node_col_alias
            
        hop_k_tgt_edges = rename_columns(spark, target_edges["data"], tgt_edges_node_col_to_aliases)
        
        # 记录该hop跳数的全部路径信息
        related_k_hop_pair_paths[hop_k] = []
        
        # 依次计算各个tgt_k_hop_path
        for path_index, tgt_k_hop_path_schema in enumerate(tgt_k_hop_paths[hop_k]):
            # 补全join_edges要求的配置信息
            tgt_path_config = {}

            # 路径名
            tgt_path_config["join_edges_name"] = f"hop_{hop_k}_path_{path_index}"
            
            # 查找父路径在tgt_k_hop_paths中对应的序号
            if hop_k > 1:
                parent_path_schema = tgt_k_hop_path_schema[:-1]
                parent_path_index = tgt_k_hop_paths[hop_k - 1].index(parent_path_schema)
                
                # 额外加入父路径的信息加速运算
                tgt_path_config["parent_join_edges"] = {}
                tgt_path_config["parent_join_edges"]["join_edges_name"] = f"hop_{hop_k-1}_path_{parent_path_index}"
                tgt_path_config["parent_join_edges"]["join_edges_len"] = hop_k - 1
            
            # 记录对应的join_edges_schema
            tgt_path_config["join_edges_schema"] = copy.deepcopy(tgt_k_hop_path_schema)

            # 基于graph配置信息对join_edges信息初始化
            tgt_path = join_edges_init(graph, query_nodes_config, tgt_path_config, join_edges_default_config)

            # 路径结果存储位置  
            tgt_path_result_path = head_query_config["tgt_query_nodes"]["result_path"] + f"/join_edges/{tgt_path['name']}" 
            
            # 进行运算 
            if not hdfs_check_file_exists(tgt_path_result_path + f"/_SUCCESS"):
                tgt_path_df = join_edges_query(spark, graph, tgt_path, head_query_config)

            # 如果这个路径的首尾节点符合要求则计算匹配成功率
            if target_edges["tail_node_type"] == tgt_k_hop_path_schema[-1]["add_nodes_types"][0]:
                # 检查是否已有匹配结果 
                if not hdfs_check_file_exists(tgt_path_result_path + "/_MATCH"):
                    # 读取现有结果 
                    tgt_path_df = spark.read.parquet(tgt_path_result_path)
                    
                    # 获得该类型的路径总数
                    tgt_path_count = tgt_path_df.count()
                    
                    # 检查是否有匹配上的路径 
                    match_metrics = {}
                    if tgt_path_count > 0:
                        # 获得其中的有效路径的总数
                        matched_tgt_path_df = tgt_path_df.join(hop_k_tgt_edges, on = hop_k_tgt_edges.columns, how = "inner")
                        matched_tgt_path_count = matched_tgt_path_df.count()
                        
                        # 检查对应的匹配成功率: 取出的有效路径数目除以总数
                        match_rate = matched_tgt_path_count / tgt_path_count
            
                        # 获得该路径中包含的target_edges类型的总数
                        distinct_tgt_path_count = tgt_path_df.select(hop_k_tgt_edges.columns).distinct().count()
                        
                        # 获得其中匹配成功的target_edges类型的总数
                        matched_tgt_path_distinct_df = matched_tgt_path_df.select(hop_k_tgt_edges.columns).distinct()
                        matched_tgt_path_distinct_count = matched_tgt_path_distinct_df.count()
                        
                        # 检查对应的pair匹配成功率
                        distinct_match_rate = matched_tgt_path_distinct_count/distinct_tgt_path_count
                        
                        logger.info(f"Path {tgt_path['name']} has {tgt_path_count} instances, matched {matched_tgt_path_count} "
                                    f"target edges. The match rate is {match_rate}. Contain {distinct_tgt_path_count} distinct "
                                    f"instances, matched {matched_tgt_path_distinct_count} distinct target edges. The distinct "
                                    f"match rate is {distinct_match_rate}.")
                        
                        # 记录匹配结果
                        match_metrics["path_count"] = tgt_path_count
                        match_metrics["matched_path_count"] = matched_tgt_path_count
                        match_metrics["match_rate"] = match_rate
                        match_metrics["distinct_edge_count"] = distinct_tgt_path_count
                        match_metrics["matched_distinct_edge_count"] = matched_tgt_path_distinct_count
                        match_metrics["distinct_match_rate"] = distinct_match_rate
                        
                    else:
                        logger.info(f"Path {tgt_path['name']} has 0 instances.")
        
                        # 记录匹配结果
                        match_metrics["path_count"] = 0
                        match_metrics["matched_path_count"] = 0
                        match_metrics["match_rate"] = 0
                        match_metrics["distinct_edge_count"] = 0
                        match_metrics["matched_distinct_edge_count"] = 0
                        match_metrics["distinct_match_rate"] = 0
                    
                    # 保存相关结果  
                    hdfs_save_json(tgt_path_result_path, "_MATCH", match_metrics)

                # 读取匹配结果
                match_metrics = hdfs_read_json(tgt_path_result_path, "_MATCH")
            
                # 添加匹配结果到该路径变量 
                tgt_path["match_metrics"] = match_metrics
        
                # 记录该路径的相关信息
                related_k_hop_pair_paths[hop_k].append(tgt_path)
                
    return related_k_hop_pair_paths