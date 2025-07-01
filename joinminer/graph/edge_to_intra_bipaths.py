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

def reverse_path_schema(path_schema, join_edge_types, max_neighbor = 20):
    reversed_path_schema = []
    for hop_k in range(1, 1+ len(path_schema)):
        join_edge_schema = path_schema[-hop_k]

        # 获得该join_edge对应的边类型和起始点序号
        edge_type = join_edge_schema["edge_type"]
        head_node_index = join_edge_schema["join_nodes_edge_indexes"][0]
        
        # 获得反方向后对应的头结点index
        reverse_head_node_index = 1 - head_node_index
        
        # 记录反方向后的join_edge的配置
        reversed_join_edge_schema = copy.deepcopy(join_edge_types[edge_type][reverse_head_node_index])

        # 颠倒join_nodes和add_nodes并重新设定index
        reversed_join_edge_schema["join_nodes_indexes"] = [hop_k - 1]
        reversed_join_edge_schema["add_nodes_indexes"] = [hop_k]

        # 添加对应的join_edges_samples
        reversed_join_edge_schema["join_edges_samples"] = []

        if hop_k > 1:
            join_edges_sample = {}
            join_edges_sample["sample_nodes_types"] = reversed_path_schema[0]["join_nodes_types"] + reversed_join_edge_schema["add_nodes_types"]
            join_edges_sample["sample_nodes_indexes"] = [0] + reversed_join_edge_schema["add_nodes_indexes"]
            join_edges_sample["sample_type"] = "random"
            join_edges_sample["sample_count"] = max_neighbor
            reversed_join_edge_schema["join_edges_samples"].append(join_edges_sample)
            
        join_edges_sample = {}
        join_edges_sample["sample_nodes_types"] = reversed_join_edge_schema["add_nodes_types"]
        join_edges_sample["sample_nodes_indexes"] = reversed_join_edge_schema["add_nodes_indexes"]
        join_edges_sample["sample_type"] = "random"
        join_edges_sample["sample_count"] = max_neighbor
        reversed_join_edge_schema["join_edges_samples"].append(join_edges_sample)
        
        reversed_path_schema.append(reversed_join_edge_schema)
        
    return reversed_path_schema

def edge_to_intra_bipaths(spark, graph, labeled_samples_config, join_edge_types, join_edges_default_config, max_hop_k, max_neighbor):
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

    # 将这些k_hop_pair_paths分解为intersect_paths
    # 要获得应该计算哪些head_paths和tail_paths
    k_hop_intersect_paths = {}
    k_hop_intersect_paths["head_paths"] = {}
    k_hop_intersect_paths["tail_paths"] = {}

    # 依次处理各跳的pair_paths
    for hop_k in range(1, max_hop_k + 1):
        # 依次处理各个pair_paths
        for k_hop_pair_path in k_hop_pair_paths[hop_k]:
            # 获得该pair_path对应的head_path和tail_path的长度
            head_path_hop_k = (hop_k + 1)//2
            tail_path_hop_k = hop_k//2
            
            # 获得对应的head_path信息 
            head_path_schema = k_hop_pair_path[:head_path_hop_k]

            # 保存对应的head_path信息
            if head_path_hop_k not in k_hop_intersect_paths["head_paths"]:
                k_hop_intersect_paths["head_paths"][head_path_hop_k] = []
            if head_path_schema not in k_hop_intersect_paths["head_paths"][head_path_hop_k]:
                k_hop_intersect_paths["head_paths"][head_path_hop_k].append(head_path_schema)

            # 依次检查各级父路径
            for parent_hop_k in range(1, head_path_hop_k):
                parent_path_schema = head_path_schema[:parent_hop_k]

                # 如果没有父路径，则将其加入计算目标中
                if parent_path_schema not in k_hop_intersect_paths["head_paths"][parent_hop_k]:
                    k_hop_intersect_paths["head_paths"][parent_hop_k].append(parent_path_schema)
            
            # 检查tail_path长度是否大于0 
            if tail_path_hop_k > 0:
                # 获得对应的tail_path信息
                tail_path_schema = k_hop_pair_path[head_path_hop_k:]

                # 获得转换方向后的tail_path_schema
                tail_path_schema = reverse_path_schema(tail_path_schema, join_edge_types, max_neighbor)

                # 保存对应的tail_path信息 
                if tail_path_hop_k not in k_hop_intersect_paths["tail_paths"]:
                    k_hop_intersect_paths["tail_paths"][tail_path_hop_k] = []
                if tail_path_schema not in k_hop_intersect_paths["tail_paths"][tail_path_hop_k]:
                    k_hop_intersect_paths["tail_paths"][tail_path_hop_k].append(tail_path_schema)

                # 依次检查各级父路径
                for parent_hop_k in range(1, tail_path_hop_k):
                    parent_path_schema = tail_path_schema[:parent_hop_k]
    
                    # 如果没有父路径，则将其加入计算目标中
                    if parent_path_schema not in k_hop_intersect_paths["tail_paths"][parent_hop_k]:
                        k_hop_intersect_paths["tail_paths"][parent_hop_k].append(parent_path_schema)

    # 显示要计算的路径数目
    for hop_k in range(1, (max_hop_k + 1)//2 + 1):
        logger.info(f"Require {len(k_hop_intersect_paths['head_paths'][hop_k])} head paths at {hop_k}-th hop")
    for hop_k in range(1, max_hop_k//2 + 1):
        logger.info(f"Require {len(k_hop_intersect_paths['tail_paths'][hop_k])} tail paths at {hop_k}-th hop")

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
    head_query_config["tgt_query_nodes"]["result_path"] = target_edges["result_path"] + "/k_hop_intersect_head_paths"
    head_query_config["tgt_query_nodes"]["df"] = head_nodes_df

    # 设定这组query的query node配置
    query_nodes_config = {}
    query_nodes_config["query_nodes_types"] = [target_edges["head_node_type"]]
    query_nodes_config["query_nodes_indexes"] = [0]
    
    # 依次计算所需的head_paths路径
    related_k_hop_head_paths = {}
    for hop_k in range(1, (max_hop_k + 1)//2 + 1):
        # 记录该hop跳数的全部路径信息
        related_k_hop_head_paths[hop_k] = []
        
        # 依次计算各个tgt_k_hop_path
        for path_index, tgt_k_hop_path_schema in enumerate(k_hop_intersect_paths['head_paths'][hop_k]):
            # 补全join_edges要求的配置信息
            tgt_path_config = {}

            # 路径名
            tgt_path_config["join_edges_name"] = f"hop_{hop_k}_path_{path_index}"
            
            # 查找父路径在tgt_k_hop_paths中对应的序号
            if hop_k > 1:
                parent_path_schema = tgt_k_hop_path_schema[:-1]
                parent_path_index = k_hop_intersect_paths['head_paths'][hop_k - 1].index(parent_path_schema)
                
                # 额外加入父路径的信息加速运算
                tgt_path_config["parent_join_edges"] = {}
                tgt_path_config["parent_join_edges"]["join_edges_name"] = f"hop_{hop_k-1}_path_{parent_path_index}"
                tgt_path_config["parent_join_edges"]["join_edges_len"] = hop_k - 1
            
            # 记录对应的join_edges_schema
            tgt_path_config["join_edges_schema"] = copy.deepcopy(tgt_k_hop_path_schema)

            # 基于graph配置信息对join_edges信息初始化
            tgt_path = join_edges_init(graph, query_nodes_config, tgt_path_config, join_edges_default_config)
    
            # 进行运算
            tgt_path_df = join_edges_query(spark, graph, tgt_path, head_query_config)
            tgt_path["data"] = tgt_path_df
            
            # 获得该类型的路径总数
            tgt_path_count = tgt_path_df.count()
            tgt_path["path_count"] = tgt_path_count
            
            logger.info(f"Head path {tgt_path['name']} has {tgt_path_count} instances.")
            
            # 记录该路径的相关信息 
            related_k_hop_head_paths[hop_k].append(tgt_path)

    # 获得目标边的全部去重后的起始节点
    tail_nodes_id_col_aliases = []
    for col_i in range(len(target_edges["tail_node_cols"])):
        node_col = target_edges["tail_node_cols"][col_i]
        node_col_alias = standard_node_col_name(target_edges["tail_node_type"], 0, col_i)
        tail_nodes_id_col_aliases.append(col(node_col).alias(node_col_alias))
    for time_col in graph.graph_time_cols_alias:
        tail_nodes_id_col_aliases.append(col(time_col).alias(time_col))
    tail_nodes_df = target_edges["data"].select(*tail_nodes_id_col_aliases).distinct()

    # 将目标边的起始节点作为路径的目标点的query配置
    tail_query_config = {}
    tail_query_config["graph_time_values"] = train_samples_time_cols_values
    tail_query_config["tgt_query_nodes"] = {}
    tail_query_config["tgt_query_nodes"]["result_path"] = target_edges["result_path"] + "/k_hop_intersect_tail_paths"
    tail_query_config["tgt_query_nodes"]["df"] = tail_nodes_df

    # 设定这组query的query node配置
    query_nodes_config = {}
    query_nodes_config["query_nodes_types"] = [target_edges["tail_node_type"]]
    query_nodes_config["query_nodes_indexes"] = [0]
    
    # 依次计算所需的tail_paths路径
    related_k_hop_tail_paths = {}
    for hop_k in range(1, max_hop_k//2 + 1):
        # 记录该hop跳数的全部路径信息
        related_k_hop_tail_paths[hop_k] = []
        
        # 依次计算各个tgt_k_hop_path
        for path_index, tgt_k_hop_path_schema in enumerate(k_hop_intersect_paths['tail_paths'][hop_k]):
            # 补全join_edges要求的配置信息
            tgt_path_config = {}

            # 路径名
            tgt_path_config["join_edges_name"] = f"hop_{hop_k}_path_{path_index}"
            
            # 查找父路径在tgt_k_hop_paths中对应的序号
            if hop_k > 1:
                parent_path_schema = tgt_k_hop_path_schema[:-1]
                parent_path_index = k_hop_intersect_paths['tail_paths'][hop_k - 1].index(parent_path_schema)
                
                # 额外加入父路径的信息加速运算
                tgt_path_config["parent_join_edges"] = {}
                tgt_path_config["parent_join_edges"]["join_edges_name"] = f"hop_{hop_k-1}_path_{parent_path_index}"
                tgt_path_config["parent_join_edges"]["join_edges_len"] = hop_k - 1

            # 记录对应的join_edges_schema
            tgt_path_config["join_edges_schema"] = copy.deepcopy(tgt_k_hop_path_schema)

            # 基于graph配置信息对join_edges信息初始化
            tgt_path = join_edges_init(graph, query_nodes_config, tgt_path_config, join_edges_default_config)
    
            # 进行运算
            tgt_path_df = join_edges_query(spark, graph, tgt_path, tail_query_config)
            tgt_path["data"] = tgt_path_df
            
            # 获得该类型的路径总数
            tgt_path_count = tgt_path_df.count()
            tgt_path["path_count"] = tgt_path_count
            
            logger.info(f"Tail path {tgt_path['name']} has {tgt_path_count} instances.")
            
            # 记录该路径的相关信息 
            related_k_hop_tail_paths[hop_k].append(tgt_path)

    # 全部intersect_path的计算结果的存储路径  
    intersect_paths_result_path = target_edges["result_path"] + "/k_hop_intersect_paths"
    
    # # 依次计算全部目标路径，并获得他们的匹配成功率
    related_k_hop_intersect_paths = {}
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
        related_k_hop_intersect_paths[hop_k] = []
        
        # 依次计算各个tgt_k_hop_path
        for path_index, tgt_k_hop_path_schema in enumerate(k_hop_pair_paths[hop_k]):
            # 补全join_edges要求的配置信息
            tgt_path = {}

            # 路径名
            tgt_path["join_edges_name"] = f"hop_{hop_k}_path_{path_index}"

            # 路径shcema信息
            tgt_path["join_edges_schema"] = tgt_k_hop_path_schema

            # 要记录各个特征列对应的特征长度
            tgt_path["feat_cols_sizes"] = {}
            
            # 路径结果存储位置  
            tgt_path_result_path = f"{intersect_paths_result_path}/{tgt_path['join_edges_name']}" 

            # 获得head_path和tail_path对应的长度
            head_path_hop_k = (hop_k + 1)//2
            tail_path_hop_k = hop_k//2

            # 获得对应的head_path schema信息 
            head_path_schema = tgt_k_hop_path_schema[:head_path_hop_k]

            # 获得对应的head_path的结果信息
            head_path_index = k_hop_intersect_paths['head_paths'][head_path_hop_k].index(head_path_schema)
            head_path = related_k_hop_head_paths[head_path_hop_k][head_path_index]

            logger.info(f"Head path: {head_path['name']}")
            tgt_path["forward_path"] = head_path
            tgt_path["forward_path_schema"] = head_path_schema

            # 记录forward_path对应的特征列和维度
            for feat_col in head_path["query_config"]["feat_cols_sizes"]:
                tgt_path["feat_cols_sizes"][feat_col] = head_path["query_config"]["feat_cols_sizes"][feat_col]
        
            # 检查tail_path长度是否大于0 
            if tail_path_hop_k > 0:
                # 获得对应的tail_path信息
                tail_path_schema = tgt_k_hop_path_schema[head_path_hop_k:]

                # 获得转换方向后的tail_path_schema
                tail_path_schema = reverse_path_schema(tail_path_schema, join_edge_types, max_neighbor)

                # 获得对应的tail_path的结果信息
                tail_path_index = k_hop_intersect_paths['tail_paths'][tail_path_hop_k].index(tail_path_schema)
                tail_path = related_k_hop_tail_paths[tail_path_hop_k][tail_path_index]

                logger.info(f"Tail path: {tail_path['name']}")
                tgt_path["backward_path"] = tail_path
                tgt_path["backward_path_schema"] = tail_path_schema
                
                # 获得tail_path上的节点列的映射方式以及用于join的节点列名 
                tail_path_node_col_to_aliases = {}
                tail_path_feat_col_aliases = []
                intersect_path_node_cols = []
                for edge_index in range(tail_path_hop_k):
                    # 获得对应边的schema信息
                    edge_schema = tail_path_schema[edge_index]

                    # 获得该edge在完整的intersect_path中对应的index
                    intersect_edge_index = hop_k - 1 - edge_index
                    
                    if edge_index == 0:
                        node_type = edge_schema["join_nodes_types"][0]
                        for col_i in range(len(graph.nodes[node_type]["node_col_types"])):
                            node_col = standard_node_col_name(node_type, edge_index, col_i)
                            node_col_alias = standard_node_col_name(node_type, intersect_edge_index + 1, col_i)
                            tail_path_node_col_to_aliases[node_col] = node_col_alias

                    # 记录join_edge对应的特征和别名 
                    edge_type = edge_schema["edge_type"]
                    join_edge_feat_col = standard_feat_col_name("edge", edge_type, edge_index)
                    join_edge_feat_col_alias = standard_feat_col_name("edge", edge_type, intersect_edge_index)
                    tail_path_feat_col_aliases.append([join_edge_feat_col, join_edge_feat_col_alias])

                    # 记录边特征长度
                    tgt_path["feat_cols_sizes"][join_edge_feat_col_alias] = len(graph.edges[edge_type]["graph_token_feat_cols"])
                    
                    node_type = edge_schema["add_nodes_types"][0] 
                    for col_i in range(len(graph.nodes[node_type]["node_col_types"])):
                        node_col = standard_node_col_name(node_type, edge_index + 1, col_i)
                        node_col_alias = standard_node_col_name(node_type, intersect_edge_index, col_i)
                        tail_path_node_col_to_aliases[node_col] = node_col_alias

                        if edge_index == tail_path_hop_k - 1:
                            intersect_path_node_cols.append(node_col_alias)

                    # 记录除最后一跳的add_node对应的特征列和别名
                    if edge_index < tail_path_hop_k - 1: 
                        node_feat_col = standard_feat_col_name("node", node_type, edge_index + 1)
                        node_feat_col_alias = standard_feat_col_name("node", node_type, intersect_edge_index)
                        tail_path_feat_col_aliases.append([node_feat_col, node_feat_col_alias])

                        # 记录点特征长度
                        tgt_path["feat_cols_sizes"][node_feat_col_alias] = len(graph.nodes[node_type]["graph_token_feat_cols"])
                
                # 记录tail_path的id列的alias 
                tgt_path["backward_path_id_col_alias"] = [[k, v] for k, v in tail_path_node_col_to_aliases.items()]
                for time_col in graph.graph_time_cols_alias:
                    tgt_path["backward_path_id_col_alias"].append([time_col, time_col])

                # 记录tail_path如果有特征列的话的列和alias
                tgt_path["backward_path_feat_col_alias"] = list(tail_path_feat_col_aliases)
                
                # 记录tail_path用于join的node_cols 
                tgt_path["backward_path_join_node_cols"] = list(intersect_path_node_cols)
            
            # 检查是否已有现有结果
            if not hdfs_check_file_exists(tgt_path_result_path + f"/_SUCCESS"):
                head_path_df = head_path["data"]
                
                # 检查tail_path长度是否大于0 
                if tail_path_hop_k > 0:
                    # 修正tail_path的列名
                    tail_path_df = rename_columns(spark, tail_path["data"], tail_path_node_col_to_aliases)

                    # 获得用于join的id列名
                    intersect_path_id_cols = intersect_path_node_cols + graph.graph_time_cols_alias

                    # 联结heat_path和tail_path
                    tgt_path_df = head_path_df.join(tail_path_df, on = intersect_path_id_cols, how = "inner")
                    
                else:
                    # 否则直接以head_path的结果作为intersect_path
                    tgt_path_df = head_path_df
                    
                # 保存结果 
                pyspark_optimal_save(tgt_path_df, tgt_path_result_path, "parquet", "overwrite")

            # 检查是否已有匹配结果 
            if not hdfs_check_file_exists(tgt_path_result_path + "/_MATCH"):
                # 记录匹配结果的变量
                match_metrics = {}
                
                # 读取现有结果 
                tgt_path_df = spark.read.parquet(tgt_path_result_path)
                
                # 获得该类型的路径总数
                tgt_path_count = tgt_path_df.count()
    
                # 检查是否有匹配上的路径
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
                    
                    logger.info(f"Path {tgt_path['join_edges_name']} has {tgt_path_count} instances, matched {matched_tgt_path_count} "
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
            related_k_hop_intersect_paths[hop_k].append(tgt_path)
    
    return related_k_hop_intersect_paths