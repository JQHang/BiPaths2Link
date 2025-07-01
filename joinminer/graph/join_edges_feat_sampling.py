from .graph_loader import read_graph_table_partitions, read_graph_table
from .join_edges_sampling import join_edges_sampling
from ..python import time_costing, setup_logger
from ..hdfs import hdfs_check_file_exists, hdfs_read_json, hdfs_save_json, hdfs_save_string
from ..pyspark import pyspark_aggregate, vectorize_and_scale, rename_columns, pyspark_optimal_save

import copy
from pyspark.sql.functions import col

# 用于对join_edges中的各个最小单元(目前就是graph表维度)进行聚合
@time_costing
def join_edges_table_feat_sampling(spark, join_edges_df, token_table_schema, agg_token_config, logger = setup_logger()):
    # 获得对应聚合结果的存储表的相关schema
    agg_token_table_schema = {}
    agg_token_table_schema["table_path"] = agg_token_config["agg_token_path"]
    agg_token_table_schema["table_format"] = agg_token_config["agg_token_table_format"]
    agg_token_table_schema["partition_cols"] = list(token_table_schema["graph_time_cols"])
    agg_token_table_schema["partition_cols_values"] = list(token_table_schema["graph_time_cols_values"])

    logger.info(f"The token aggregation reulst will be output to: {agg_token_table_schema['table_path']}")
    
    # 尝试读取对应数据
    read_result = read_graph_table_partitions(spark, agg_token_table_schema)

    # 检查是否读取到全部的结果
    if read_result["success"]:
        logger.info(f"The reulst already exist.")
        
        # 如果已有全部结果，则直接返回
        return read_result["data"]
        
    else:
        # 否则基于未完成的时间点，修正token_table_schema中的目标时间
        token_table_schema["graph_time_cols_values"] = list(read_result["failed_partitions"])
    
    # 读取对应的token的graph table
    token_table_df = read_graph_table(spark, token_table_schema, logger = logger)

    # 获得该token对应的id列
    token_id_cols = list(token_table_schema["graph_time_cols"])
    for node_col_to_alias in token_table_schema["node_cols_to_aliases"]:
        token_id_cols.append(node_col_to_alias[1])
    
    # 基于id列进行join
    join_edges_token_df = join_edges_df.join(token_table_df, on = token_id_cols, how = "left")

    # 获得该token对应的全部特征列
    token_feat_cols = []
    for feat_col_to_alias in token_table_schema["feat_cols_to_aliases"]:
        token_feat_cols.append(feat_col_to_alias[1])

    # 未join上的特征列补全为0(待优化，最好之后转化为对数值转embed的方式)
    join_edges_token_df = join_edges_token_df.fillna(0, subset = token_feat_cols)

    # 获得要用于聚合的id列
    agg_id_cols = list(token_table_schema["graph_time_cols"]) + list(agg_token_config["inst_agg_nodes_cols"])
    
    # 设定聚合运算相关配置
    agg_config = []
    for agg_func in agg_token_config["inst_agg_funcs"]:
        if agg_func == 'count_*':
            agg_config.append(["*", "count", "count_all"])
        elif agg_func in ['count', 'mean', 'sum', 'max', 'min', 'first']:
            for feat_col in token_feat_cols:
                agg_config.append([feat_col, agg_func, f"{agg_func}_{feat_col}"])
                            
    # 进行聚合
    agg_token_df = pyspark_aggregate(join_edges_token_df, agg_id_cols, agg_config)
    
    # 获得聚合后的特征列名
    inst_agg_feat_cols = [x[2] for x in agg_config]
    
    # 向量化及缩放
    agg_token_df = vectorize_and_scale(agg_token_df, agg_token_config["feats_scale_funcs"], 
                                       agg_id_cols, inst_agg_feat_cols)
                
    # 保存结果
    pyspark_optimal_save(agg_token_df, agg_token_table_schema["table_path"], 
                         agg_token_table_schema["table_format"], "append",
                         agg_token_table_schema["partition_cols"], logger = logger)

    # 重新读取结果返回
    read_result = read_graph_table_partitions(spark, agg_token_table_schema)
    
    return read_result["data"]

@time_costing
def join_edges_feat_sampling(spark, graph, join_edges, target_nodes_config = None, logger = setup_logger()):
    # 可以基于是否给出inst_agg_path来决定是否保留计算结果，待优化
    logger.info(f"Complete features for the instances od join-edges {join_edges['name']}")
    logger.info(f"The reulst will be output to: {join_edges['inst_feat_path']}")

    # 获得要进行聚合的目标图时间相关信息
    graph_time_cols = list(graph.graph_time_cols_alias)
    graph_time_cols_values = list(graph.graph_time_cols_values)
    graph_time_cols_formats = list(graph.graph_time_cols_formats)
    
    # 获得agg nodes相关信息
    inst_agg_nodes_types = join_edges["inst_agg_nodes_types"]
    inst_agg_nodes_indexes = join_edges["inst_agg_nodes_indexes"]
    inst_agg_nodes_cols = join_edges["inst_agg_nodes_cols"]

    logger.info(f"The aggregating nodes are {inst_agg_nodes_types} at indexes {inst_agg_nodes_indexes} "
                f"corresponds to nodes cols {inst_agg_nodes_cols}.")
    
    # 获得这些agg_nodes分别对应哪几个join_edges中的seq_token
    inst_agg_nodes_seq_indexes = []
    for agg_node_i in range(len(inst_agg_nodes_types)):
        agg_node_type = inst_agg_nodes_types[agg_node_i]
        agg_node_index = inst_agg_nodes_indexes[agg_node_i]

        agg_node_seq_index = -1
        for seq_token_index, seq_token in enumerate(join_edges["flatten_format"]["seq"]):
            if "node_type" in seq_token:
                if seq_token["node_type"] == agg_node_type and seq_token["node_index"] == agg_node_index:
                    agg_node_seq_index = seq_token_index

        if agg_node_seq_index == -1:
            raise ValueError(f"Aggregation node {agg_node_type} index {agg_node_index} doesn't "
                             f"exist in the join-edges")

        inst_agg_nodes_seq_indexes.append(agg_node_seq_index)

    # 先通过Join edges Sampling获得该join-edges对应的instances
    join_edges_df = join_edges_sampling(spark, graph, join_edges, target_nodes_config, logger = logger)
    
    # 用该变量记录聚合后的join_edges的相关信息
    agg_join_edges = {}
    
    # 依次处理各元素信息
    for seq_token_index, seq_token in enumerate(join_edges["flatten_format"]["seq"]):
        logger.info(f"Start aggregating {seq_token_index}-th sequential token of join-edges")
        
        # 检查该seq_token是否对应到聚合节点，是就跳过
        if seq_token_index in inst_agg_nodes_seq_indexes:
            logger.info(f"Skip aggregating features from {seq_token_index}-th sequential token since "
                        f"it corresponds to the aggregation nodes")
            continue

        # 记录该seq_token对应的各个table_token的聚合后的结果
        agg_join_edges[seq_token_index] = {}
        
        # 查看该元素是节点还是边
        if "edge_type" in seq_token:
            # 获取对应的边类型和边序号
            edge_type = seq_token["edge_type"]
            edge_index = seq_token["edge_index"]

            # 获得该边在整个join_edges中对应的join_edge_schema
            join_edge_schema = join_edges["schema"][edge_index]
            
            # 获得该边连接到的节点类型，节点序号以及对应的节点列名
            edge_nodes_cols = []
            for join_node_cols in join_edge_schema["join_nodes_columns"]:
                for node_col_to_alias in join_node_cols:
                    edge_nodes_cols.append(node_col_to_alias[1])
            if "add_nodes_columns" in join_edge_schema:
                for add_node_cols in join_edge_schema["add_nodes_columns"]:
                    for node_col_to_alias in add_node_cols:
                        edge_nodes_cols.append(node_col_to_alias[1])
                        
            # 依次处理该seq_token包含的各个table(以后会把edge_table这个项改成list形式)
            for edge_table_name in [seq_token["edge_table_name"]]:
                logger.info(f"Aggregate features for {seq_token_index}-th sequential token of edge type "
                            f"{edge_type}, edge index {edge_index}, edge table {edge_table_name} and "
                            f"nodes columns {edge_nodes_cols}")
                
                # 获取该边对应的边表的schema
                edge_table_schema = graph.edges[edge_type]["edge_tables"][edge_table_name]

                # 检查该表是否有特征列，如果没有则跳过
                if len(edge_table_schema["time_aggs_feat_cols"]) == 0:
                    logger.info(f"Skip aggregating features since this table doesn't have feature columns")
                    continue
                
                # 设定聚合该token对应的edge table中的信息到agg_nodes上的相关配置
                token_table_schema = {}

                # 设定该token表对应的名称
                token_table_schema["name"] = (f"Token_edge_table_{edge_table_name}_of_{seq_token_index}-th_"
                                              f"seq_token_of_join-edge_{join_edges['name']}")
                
                # 设定该边对应到图里的目标时间
                token_table_schema["graph_time_cols"] = list(graph_time_cols)
                token_table_schema["graph_time_cols_values"] = list(graph_time_cols_values)
                token_table_schema["graph_time_cols_formats"] = list(graph_time_cols_formats)

                # 设定该edge table对应的source table的基础配置
                token_table_schema["src_table_path"] = edge_table_schema["src_table_path"]
                token_table_schema["src_table_format"] = edge_table_schema["src_table_format"]
        
                # 设定要从source table中读取的各种类型的列
                token_table_schema["src_node_cols"] = []
                for node_cols in edge_table_schema["linked_node_types_cols"]:
                    token_table_schema["src_node_cols"].extend(node_cols)
                
                token_table_schema["src_feat_cols"] = list(edge_table_schema["feat_cols"])
                token_table_schema["src_time_cols"] = list(edge_table_schema["time_cols"])
                token_table_schema["src_time_cols_formats"] = list(edge_table_schema["time_cols_formats"])
        
                # 如果有规定time aggregation结果的存储路径，则加入相关设定
                if "time_agg_table_path" in edge_table_schema:
                    token_table_schema["time_agg_table_path"] = edge_table_schema["time_agg_table_path"]
                    token_table_schema["time_agg_table_format"] = edge_table_schema["time_agg_table_format"]
                    token_table_schema["time_agg_save_interval"] = edge_table_schema["time_agg_save_interval"]
                
                # 设定source table通过time aggregation形成edge table的方案 
                token_table_schema["time_aggs"] = copy.deepcopy(edge_table_schema["time_aggs"])
        
                # 设定要获得的最终graph table的各个列的相关配置
                token_table_schema["node_cols_to_aliases"] = []
                for join_node_cols in join_edge_schema["join_nodes_columns"]:
                    token_table_schema["node_cols_to_aliases"].extend(join_node_cols)
                if "add_nodes_columns" in join_edge_schema:
                    for add_node_cols in join_edge_schema["add_nodes_columns"]:
                        token_table_schema["node_cols_to_aliases"].extend(add_node_cols)
                
                token_table_schema["feat_cols_to_aliases"] = []
                for time_agg_feat_col in edge_table_schema["time_aggs_feat_cols"]:
                    token_table_schema["feat_cols_to_aliases"].append([time_agg_feat_col, time_agg_feat_col])
                
                token_table_schema["time_cols_to_aliases"] = [[x, y] for x, y in zip(graph_time_cols, graph_time_cols)]
                    
                # # 设定对最终的graph table的限制
                # # 目前的写法，edge_limit和feat_cols的配合有问题（必须alias同一个名称才能limit，待优化，可以考虑对用于limit的列加入额外标记）
                # if "edge_limit" in join_edge_schema:
                #     token_table_schema["table_limit"] = join_edge_schema["edge_limit"]

                # 获得聚合方式的相关配置
                agg_token_config = {}
                agg_token_config["agg_token_path"] = join_edges['inst_agg_path'] + f"/{seq_token_index}/{edge_table_name}"
                agg_token_config["agg_token_table_format"] = join_edges["inst_agg_table_format"]
                agg_token_config["inst_agg_nodes_cols"] = inst_agg_nodes_cols
                agg_token_config["inst_agg_funcs"] = join_edges["inst_agg_funcs"]
                agg_token_config["feats_scale_funcs"] = join_edges["feats_scale_funcs"]

                # 基于相关配置获得对应的聚合后的结果
                agg_token_df = join_edges_table_token_aggregation(spark, join_edges_df, token_table_schema, 
                                                                  agg_token_config, logger = logger)

                # 记录对应的结果
                agg_join_edges[seq_token_index][edge_table_name] = agg_token_df

        else:
            node_type = seq_token["node_type"]
            node_index = seq_token["node_index"]
            node_cols = seq_token["node_cols"]
            
            # 依次处理该类型节点对应的各个节点表
            for node_table_name in graph.nodes[node_type]["node_tables"]:
                logger.info(f"Aggregate features for {seq_token_index}-th sequential token of node type "
                            f"{node_type}, node index {node_index}, node table {node_table_name} and "
                            f"node columns {node_cols}")

                # 获取该节点对应的节点表的schema
                node_table_schema = graph.nodes[node_type]["node_tables"][node_table_name]

                # 检查该表是否有特征列，如果没有则跳过
                if len(node_table_schema["time_aggs_feat_cols"]) == 0:
                    logger.info(f"Skip aggregating features since this table doesn't have feature columns")
                    continue
                
                # 设定聚合该token对应的edge table中的信息到agg_nodes上的相关配置
                token_table_schema = {}

                # 设定该token表对应的名称
                token_table_schema["name"] = (f"Token_node_table_{node_table_name}_of_{seq_token_index}-th_"
                                              f"seq_token_of_join-edge_{join_edges['name']}")
                
                # 设定该边对应到图里的目标时间
                token_table_schema["graph_time_cols"] = list(graph_time_cols)
                token_table_schema["graph_time_cols_values"] = list(graph_time_cols_values)
                token_table_schema["graph_time_cols_formats"] = list(graph_time_cols_formats)

                # 设定该node table对应的source table的基础配置
                token_table_schema["src_table_path"] = node_table_schema["src_table_path"]
                token_table_schema["src_table_format"] = node_table_schema["src_table_format"]
                
                token_table_schema["src_node_cols"] = list(node_table_schema["node_cols"])
                token_table_schema["src_feat_cols"] = list(node_table_schema["feat_cols"])
                token_table_schema["src_time_cols"] = list(node_table_schema["time_cols"])
                token_table_schema["src_time_cols_formats"] = list(node_table_schema["time_cols_formats"])
        
                # 如果有规定time aggregation结果的存储路径，则加入相关设定
                if "time_agg_table_path" in node_table_schema:
                    token_table_schema["time_agg_table_path"] = node_table_schema["time_agg_table_path"]
                    token_table_schema["time_agg_table_format"] = node_table_schema["time_agg_table_format"]
                    token_table_schema["time_agg_save_interval"] = node_table_schema["time_agg_save_interval"]

                # 设定source table通过time aggregation形成node table的方案 
                token_table_schema["time_aggs"] = copy.deepcopy(node_table_schema["time_aggs"])
        
                # 设定要获得的最终graph table的各个列的相关配置 node_cols
                token_table_schema["node_cols_to_aliases"] = [[x, y] for x, y in zip(node_table_schema["node_cols"], 
                                                                                     node_cols)]
                
                token_table_schema["feat_cols_to_aliases"] = []
                for time_agg_feat_col in node_table_schema["time_aggs_feat_cols"]:
                    token_table_schema["feat_cols_to_aliases"].append([time_agg_feat_col, time_agg_feat_col])
                
                token_table_schema["time_cols_to_aliases"] = [[x, y] for x, y in zip(graph_time_cols, graph_time_cols)]

                # 获得聚合方式的相关配置
                agg_token_config = {}
                agg_token_config["agg_token_path"] = join_edges['inst_agg_path'] + f"/{seq_token_index}/{node_table_name}"
                agg_token_config["agg_token_table_format"] = join_edges["inst_agg_table_format"]
                agg_token_config["inst_agg_nodes_cols"] = inst_agg_nodes_cols
                agg_token_config["inst_agg_funcs"] = join_edges["inst_agg_funcs"]
                agg_token_config["feats_scale_funcs"] = join_edges["feats_scale_funcs"]

                # 基于相关配置获得对应的聚合后的结果
                agg_token_df = join_edges_table_token_aggregation(spark, join_edges_df, token_table_schema, 
                                                                  agg_token_config, logger = logger)

                # 记录对应的结果
                agg_join_edges[seq_token_index][node_table_name] = agg_token_df
                
    return agg_join_edges