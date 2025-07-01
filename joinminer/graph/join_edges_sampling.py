from .join_edges_init import graph_token_node_col_name, standard_node_col_name
from .graph_loader import read_graph_table_partitions, read_graph_table
from ..python import time_costing, setup_logger
from ..hdfs import hdfs_delete_dir, hdfs_check_file_exists, hdfs_list_contents
from ..pyspark import random_n_sample, top_n_sample, threshold_n_sample, rename_columns
from ..pyspark import pyspark_optimal_save

import copy
from pyspark.sql.functions import col

@time_costing
def join_edges_sampling(spark, graph, join_edges, target_nodes = None, logger = setup_logger()):
    """
    针对给定的复杂路类型获取对应的全部complex-path instances

    输入：

    返回值：
        
    """
    logger.info(f"Sampling instances for join-edges {join_edges['name']}")

    # 获得目标结果的存储路径
    if target_nodes is None:
        # 如果没有目标节点，则直接设为全量结果的存储路径
        instance_path = join_edges['instance_path']
    else:
        # 若有目标节点，则设定为目标节点结果目录下对应路径文件夹
        instance_path = target_nodes["result_path"] + f"/join_edges_sampling/{join_edges['name']}"
    
    logger.info(f"The reulst will be output to: {instance_path}")
    
    # 设定采样结果对应的instance表的相关schema
    instance_table_schema = {}
    instance_table_schema["table_path"] = instance_path
    instance_table_schema["table_format"] = join_edges['instance_table_format']
    instance_table_schema["partition_cols"] = list(graph.graph_time_cols_alias)
    instance_table_schema["partition_cols_values"] = list(graph.graph_time_cols_values)

    # 尝试读取对应数据
    read_result = read_graph_table_partitions(spark, instance_table_schema)

    # 检查是否读取到全部的结果
    if read_result["success"]:
        # 有则返回现有结果
        logger.info(f"The result for all the target instances of this join-edges already exist.")
        return read_result["data"]
    else:
        # 检查是否有target_nodes
        if target_nodes is not None:
            # 获得全量结果的存储位置
            full_inst_table_schema = {}
            full_inst_table_schema["table_path"] = join_edges['instance_path']
            full_inst_table_schema["table_format"] = join_edges['instance_table_format']
            full_inst_table_schema["partition_cols"] = list(graph.graph_time_cols_alias)
            full_inst_table_schema["partition_cols_values"] = list(graph.graph_time_cols_values)
    
            # 尝试读取全量结果
            full_inst_read_result = read_graph_table_partitions(spark, full_inst_table_schema)

            # 如果读取成功
            if full_inst_read_result["success"]:
                # 获得全量结果
                full_join_edges_df = full_inst_read_result["data"]
                
                # 只保留target_nodes对应的结果
                join_edges_df = full_join_edges_df.join(target_nodes_df, on = target_nodes_df.columns, how = "inner")
                
                # 保存结果
                pyspark_optimal_save(join_edges_df, instance_table_schema["table_path"], instance_table_schema["table_format"], 
                                     "append", instance_table_schema["partition_cols"], logger = logger)
    
                # 重新读取完整结果
                read_result = read_graph_table_partitions(spark, instance_table_schema)
            
                return read_result["data"]
        
        # 获得未完成的时间点，并据此设定目标时间
        graph_time_cols = list(graph.graph_time_cols_alias)
        graph_time_cols_values = list(read_result["failed_partitions"])
        graph_time_cols_formats = list(graph.graph_time_cols_formats)
        
    # 获得当前的join edges中全部的id列
    join_edges_id_cols = list(graph_time_cols)
    for entry in join_edges["flatten_format"]["seq"]:
        if "node_cols" in entry:
            join_edges_id_cols.extend(entry["node_cols"])
    logger.info(f"The id columns in this join-edges type will be: {join_edges_id_cols}")

    # 检查是否有父路径的配置信息
    join_edges_start_k = 0
    if "parent" in join_edges:
        # 获取父路径对应的结果
        join_edges_df = join_edges_sampling(spark, graph, join_edges["parent"], target_nodes, logger = logger)

        # 获得父路径对应的跳数，从该跳接着运算 
        join_edges_start_k = len(join_edges["parent"]["schema"])
        
    # 依次处理各个join_edge
    for join_edge_k in range(join_edges_start_k, len(join_edges["schema"])):
        logger.info(f"Process the {join_edge_k}-th join_edge schema")

        # 获得对应的schema信息
        join_edge_schema = join_edges["schema"][join_edge_k]
        
        # 如果有target_nodes则设定对该edge的目标节点
        edge_tgt_nodes_df = None
        if target_nodes is not None:
            # 获得target_nodes中属于该edge的node对应的列及他们在join_edges中的alias
            target_node_col_to_aliases = {}
            for node_i in range(len(target_nodes["nodes_types"])):
                node_type = target_nodes["nodes_types"][node_i]
                node_cols = target_nodes["nodes_cols"][node_i]
                node_index = target_nodes["nodes_indexes"][node_i]

                # 如果在0跳的join_nodes_index中或是add_nodes_indexes中
                if ((node_index in join_edge_schema["add_nodes_indexes"]) or 
                    (node_index in join_edge_schema["join_nodes_indexes"] and join_edge_k == 0)):
                    # 获得该node_index的node在对应的graph_token中对应的index
                    linked_node_index = join_edge_schema["node_index_map"][node_index]
                    
                    for col_i in range(len(node_cols)):
                        node_col = node_cols[col_i]
                        node_col_alias = graph_token_node_col_name(node_type, linked_node_index, col_i)
                        target_node_col_to_aliases[node_col] = node_col_alias

            # 如果该边中有对应的节点列 
            if len(target_node_col_to_aliases.keys()) > 0:
                # 先取出target_nodes对应的数据
                edge_tgt_nodes_df = target_nodes["data"]
                
                # 设定要从target_nodes中取出的列
                tgt_cols = list(target_node_col_to_aliases.keys()) + list(graph.graph_time_cols_alias)
                
                # 检查目标点里是不是只有这些节点列
                if tgt_cols != edge_tgt_nodes_df.columns:
                    # 如果不是全部列则选取后再去重
                    edge_tgt_nodes_df = edge_tgt_nodes_df.select(tgt_cols).distinct()
                    
                # 对节点列进行重命名
                edge_tgt_nodes_df = rename_columns(spark, edge_tgt_nodes_df, target_node_col_to_aliases) 
        
        # 获取对应的边类型和表格名
        edge_type = join_edge_schema["edge_type"]
        edge_table_name = join_edge_schema["edge_table_name"]

        # 获取该边对应的边表的schema
        edge_table_schema = graph.edges[edge_type]["edge_tables"][edge_table_name]
        
        # 设定读取该edge table到graph中的相关配置
        # 要优化成graph_token到time_agg到key_agg到src的配置顺序
        graph_table_schema = {}

        # 设定该graph_token_table的名称
        # 先完全手动设定graph_token_table之类的名称
        # 之后可以保存对应的取数schema，通过查询schema信息获得已存在的结果
        if "edge_token_type" in join_edge_schema:
            graph_table_schema["name"] = join_edge_schema["edge_token_type"]
        
            # 设定该graph_token_table对应的结果表的存储位置及格式
            graph_table_schema["graph_token_table_path"] = edge_table_schema["graph_token_root_path"] + f"/{graph_table_schema['name']}"
            graph_table_schema["graph_token_table_format"] = edge_table_schema["graph_token_table_format"]
        else:
            graph_table_schema["name"] = f"{join_edge_k}-th_join-edge_of_{join_edges['name']}"
        
        # 设定该边对应到图里的目标时间
        graph_table_schema["graph_time_cols"] = list(graph_time_cols)
        graph_table_schema["graph_time_cols_values"] = list(graph_time_cols_values)
        graph_table_schema["graph_time_cols_formats"] = list(graph_time_cols_formats)
        
        # 设定该edge table对应的source table的基础配置
        graph_table_schema["src_table_path"] = edge_table_schema["src_table_path"]
        graph_table_schema["src_table_format"] = edge_table_schema["src_table_format"]

        # 设定要从source table中读取的各种类型的列
        graph_table_schema["src_node_cols"] = []
        for node_cols in edge_table_schema["linked_node_types_cols"]:
            graph_table_schema["src_node_cols"].extend(node_cols)
        
        graph_table_schema["src_feat_cols"] = list(edge_table_schema["feat_cols"])
        graph_table_schema["src_time_cols"] = list(edge_table_schema["time_cols"])
        graph_table_schema["src_time_cols_formats"] = list(edge_table_schema["time_cols_formats"])

        # 如果有规定time aggregation结果的存储路径，则加入相关设定
        if "time_agg_table_path" in edge_table_schema:
            graph_table_schema["time_agg_table_path"] = edge_table_schema["time_agg_table_path"]
            graph_table_schema["time_agg_table_format"] = edge_table_schema["time_agg_table_format"]
            graph_table_schema["time_agg_save_interval"] = edge_table_schema["time_agg_save_interval"]
        
        # 设定source table通过time aggregation形成edge table的方案 
        graph_table_schema["time_aggs"] = copy.deepcopy(edge_table_schema["time_aggs"])

        # 设定要获得的最终graph table的各个列的相关配置
        graph_table_schema["node_cols_to_aliases"] = []
        for join_node_cols in join_edge_schema["join_nodes_columns"]:
            for join_node_col in join_node_cols:
                graph_table_schema["node_cols_to_aliases"].append(join_node_col[0:2])
        if "add_nodes_columns" in join_edge_schema:
            for add_node_cols in join_edge_schema["add_nodes_columns"]:
                for add_node_col in add_node_cols:
                    graph_table_schema["node_cols_to_aliases"].append(add_node_col[0:2])
        
        graph_table_schema["feat_cols_to_aliases"] = []
        if "edge_feat_cols" in join_edge_schema:
            for edge_feat_col in join_edge_schema["edge_feat_cols"]:
                # 检查对应特征列是否在time_agg后形成的表格中
                if edge_feat_col not in edge_table_schema["time_aggs_feat_cols"]:
                    raise ValueError(f"The feature column \"{edge_feat_col}\" doesn't exist after time "
                                     f"aggregation of edge type {edge_type}.")
                    
                edge_feat_col_alias = join_edge_schema["edge_feat_cols"][edge_feat_col]
                graph_table_schema["feat_cols_to_aliases"].append([edge_feat_col, edge_feat_col_alias])
        
        graph_table_schema["time_cols_to_aliases"] = [[x, y] for x, y in zip(graph_time_cols, graph_time_cols)]
            
        # 设定对最终的graph table的限制
        if "edge_limit" in join_edge_schema:
            graph_table_schema["table_limit"] = join_edge_schema["edge_limit"]

        # 设定对最终的graph table的采样方案
        if "edge_samples" in join_edge_schema:
            graph_table_schema["table_samples"] = join_edge_schema["edge_samples"]
        
        # 读取该edge对应的具体数据
        edge_table_df = read_graph_table(spark, graph_table_schema, edge_tgt_nodes_df, logger = logger)

        # 获得edge_token_table_df到该join_edge中各列名对应的关系
        selected_columns = []
        for join_node_cols in join_edge_schema["join_nodes_columns"]:
            for join_node_col in join_node_cols:
                selected_columns.append(col(join_node_col[1]).alias(join_node_col[2]))
        if "add_nodes_columns" in join_edge_schema:
            for add_node_cols in join_edge_schema["add_nodes_columns"]:
                for add_node_col in add_node_cols:
                    selected_columns.append(col(add_node_col[1]).alias(add_node_col[2]))
        if "edge_feat_cols" in join_edge_schema:
            for edge_feat_col in join_edge_schema["edge_feat_cols"]:
                edge_feat_col_alias = join_edge_schema["edge_feat_cols"][edge_feat_col]
                selected_columns.append(col(edge_feat_col_alias).alias(edge_feat_col_alias))
        for graph_time_col in graph_time_cols:
            selected_columns.append(col(graph_time_col).alias(graph_time_col))
            
        # 选择并重命名列
        edge_table_df = edge_table_df.select(*selected_columns)
        
        # 查看是否有Nodes limits
        if "nodes_limits" in join_edge_schema:
            for node_limit in join_edge_schema["nodes_limits"]:
                # 如果有target_nodes则设定对该node_limit的目标节点
                node_tgt_nodes_df = None
                if target_nodes is not None:
                    # 获得target_nodes中属于该edge的node对应的列及他们在join_edges中的alias
                    target_node_col_to_aliases = {}
                    for node_i in range(len(target_nodes["nodes_types"])):
                        node_type = target_nodes["nodes_types"][node_i]
                        node_cols = target_nodes["nodes_cols"][node_i]
                        node_index = target_nodes["nodes_indexes"][node_i]
        
                        # 如果是node_limit对应的点
                        if node_index == node_limit["node_index"]:
                            for col_i in range(len(node_cols)):
                                node_col = node_cols[col_i]
                                node_col_alias = graph_token_node_col_name(node_type, 0, col_i)
                                target_node_col_to_aliases[node_col] = node_col_alias

                    # 如果该点在目标点中
                    if len(target_node_col_to_aliases.keys()) > 0:
                        # 先取出target_nodes对应的数据
                        node_tgt_nodes_df = target_nodes["data"]
                        
                        # 设定要从target_nodes中取出的列
                        tgt_cols = list(target_node_col_to_aliases.keys()) + list(graph.graph_time_cols_alias)
                        
                        # 检查目标点里是不是只有这些节点列
                        if tgt_cols != node_tgt_nodes_df.columns:
                            # 如果不是全部列则选取后再去重
                            node_tgt_nodes_df = node_tgt_nodes_df.select(tgt_cols).distinct()
                            
                        # 对节点列进行重命名
                        node_tgt_nodes_df = rename_columns(spark, node_tgt_nodes_df, target_node_col_to_aliases) 
                
                # 获取对应的点类型和表格名
                node_type = node_limit["node_type"]
                node_table_name = node_limit["node_table_name"]

                # 获取该节点对应的节点表的schema
                node_table_schema = graph.nodes[node_type]["node_tables"][node_table_name]

                # 设定读取该node table到graph中的相关配置
                graph_table_schema = {}

                # 设定该graph_table的名称
                if "node_token_type" in node_limit:
                    graph_table_schema["name"] = node_limit["node_token_type"]
                
                    # 设定该graph_token_table对应的结果表的存储位置及格式
                    graph_table_schema["graph_token_table_path"] = node_limit["graph_token_root_path"] + f"/{graph_table_schema['name']}"
                    graph_table_schema["graph_token_table_format"] = node_limit["graph_token_table_format"]
                else:
                    graph_table_schema["name"] = (f"Node_table_{node_table_name}_of_{join_edge_k}-th"
                                                  f"_join-edge_of_{join_edges['name']}")

                # 设定该边对应到图里的目标时间
                graph_table_schema["graph_time_cols"] = list(graph_time_cols)
                graph_table_schema["graph_time_cols_values"] = list(graph_time_cols_values)
                graph_table_schema["graph_time_cols_formats"] = list(graph_time_cols_formats)

                # 设定该node table对应的source table的基础配置
                graph_table_schema["src_table_path"] = node_table_schema["src_table_path"]
                graph_table_schema["src_table_format"] = node_table_schema["src_table_format"]

                # 设定要从source table中读取的各种类型的列
                graph_table_schema["src_node_cols"] = list(node_table_schema["node_cols"])
                graph_table_schema["src_feat_cols"] = list(node_table_schema["feat_cols"])
                graph_table_schema["src_time_cols"] = list(node_table_schema["time_cols"])
                graph_table_schema["src_time_cols_formats"] = list(node_table_schema["time_cols_formats"])

                 # 如果有规定time aggregation结果的存储路径，则加入相关设定
                if "time_agg_table_path" in node_table_schema:
                    graph_table_schema["time_agg_table_path"] = node_table_schema["time_agg_table_path"]
                    graph_table_schema["time_agg_table_format"] = node_table_schema["time_agg_table_format"]
                    graph_table_schema["time_agg_save_interval"] = node_table_schema["time_agg_save_interval"]
                
                # 设定source table通过time aggregation形成node table的方案 
                graph_table_schema["time_aggs"] = copy.deepcopy(node_table_schema["time_aggs"])
        
                # 设定要获得的最终graph table的各个列的相关配置
                graph_table_schema["node_cols_to_aliases"] = []
                for node_col in node_limit["node_columns"]:
                    graph_table_schema["node_cols_to_aliases"].append(node_col[0:2])
                
                graph_table_schema["feat_cols_to_aliases"] = []
                if "node_feat_cols" in node_limit:
                    for node_feat_col in node_limit["node_feat_cols"]:
                        node_feat_col_alias = node_limit["node_feat_cols"][node_feat_col]
                        graph_table_schema["feat_cols_to_aliases"].append([node_feat_col, node_feat_col_alias])
                
                graph_table_schema["time_cols_to_aliases"] = [[x, y] for x, y in zip(graph_time_cols, graph_time_cols)]

                # 设定对最终的graph table的限制
                if "node_limit" in node_limit:
                    graph_table_schema["table_limit"] = node_limit["node_limit"]

                # 读取该node对应的具体数据
                node_table_df = read_graph_table(spark, graph_table_schema, node_tgt_nodes_df, logger = logger)

                # 将该node在graph_token表中对应的各列在该join_edge中对应的列
                selected_columns = []
                for node_col in node_limit["node_columns"]:
                    selected_columns.append(col(node_col[1]).alias(node_col[2]))
                if "node_feat_cols" in node_limit:
                    for node_feat_col in node_limit["node_feat_cols"]:
                        node_feat_col_alias = node_limit["node_feat_cols"][node_feat_col]
                        selected_columns.append(col(node_feat_col_alias).alias(node_feat_col_alias))
                for graph_time_col in graph_time_cols:
                    selected_columns.append(col(graph_time_col).alias(graph_time_col))
                    
                # 选择并重命名列
                node_table_df = node_table_df.select(*selected_columns)
                
                # 获得该表对应的id列
                node_id_cols = list(graph_time_cols)
                for node_col in node_limit["node_columns"]:
                    node_id_cols.append(node_col[2])
                
                # 和edge table进行join来
                edge_table_df = edge_table_df.join(node_table_df, on = node_id_cols, how = node_limit["node_join_type"])

                # 检查对join后的edge是否有限制
                if "join_edge_limit" in node_limit:
                    logger.info(f"Join edge limitation in node limitation: {node_limit['join_edge_limit']}")
                    edge_table_df = edge_table_df.filter(node_limit['join_edge_limit'])
        
        # 执行完node limit后再执行join edge sample(因为node limit会过滤掉一部分不符合条件的边，可能需要再进行一次sample)
        if 'join_edge_samples' in join_edge_schema:
            for join_edge_sample in join_edge_schema['join_edge_samples']:
                edge_sample_type = join_edge_sample['sample_type']
                edge_sample_count = join_edge_sample['sample_count']

                sample_id_cols = graph_time_cols + list(join_edge_sample['sample_nodes_cols'])

                logger.info(f'Join Edge Sampling: {edge_sample_type}, {sample_id_cols}, {edge_sample_count}')

                if edge_sample_type == 'random':
                    edge_table_df = random_n_sample(spark, edge_table_df, sample_id_cols, edge_sample_count)
                elif edge_sample_type == 'threshold':
                    edge_table_df = threshold_n_sample(spark, edge_table_df, sample_id_cols, edge_sample_count)
        
        # 和之前的join_edges进行join
        if join_edge_k == 0:
            join_edges_df = edge_table_df
        else:
            # 获得该次join要使用的id列
            edge_join_id_cols = list(graph_time_cols)
            for join_node_cols in join_edge_schema["join_nodes_columns"]:
                edge_join_id_cols.extend([join_node_col[2] for join_node_col in join_node_cols])

            # 获得通过何种类型的join将该边连接到之前的join_edges中  
            edge_join_type = join_edge_schema["edge_join_type"]
            
            # 将edge_table join到之前的join_edges中
            logger.info(f'{edge_join_type} join {join_edge_k}-th edge table to join-edges on: {edge_join_id_cols}')
            join_edges_df = join_edges_df.join(edge_table_df, on = edge_join_id_cols, how = edge_join_type)
    
        # Join edges limit
        if 'join_edges_limit' in join_edge_schema and join_edge_schema['join_edges_limit'] != '':
            logger.info(f"Join edges Limitation: join_edge_schema['join_edges_limit']")
            join_edges_df = join_edges_df.filter(join_edge_schema['join_edges_limit'])
    
        # Join edges Sample
        if 'join_edges_samples' in join_edge_schema:
            for join_edges_sample in join_edge_schema['join_edges_samples']:
                edge_sample_type = join_edges_sample['sample_type']
                edge_sample_count = join_edges_sample['sample_count']

                sample_id_cols = graph_time_cols + list(join_edges_sample['sample_nodes_cols'])
                
                logger.info(f'Join edges sampling: {edge_sample_type}, {sample_id_cols}, {edge_sample_count}')

                if edge_sample_type == 'random':
                    join_edges_df = random_n_sample(spark, join_edges_df, sample_id_cols, edge_sample_count)
                elif edge_sample_type == 'threshold':
                    join_edges_df = threshold_n_sample(spark, join_edges_df, sample_id_cols, edge_sample_count)

        # 检查是否需要调整分区数量
        # 基于已存在的id列对join_edges_df重新分区防止数据倾斜
        # exist_join_edges_id_cols = list(set(join_edges_id_cols) & set(join_edges_df.columns))
        # join_edges_df = join_edges_df.repartition(*exist_join_edges_id_cols)
        # logger.info(f'Repartition the {join_edge_k}-th join-edges result with columns: {exist_join_edges_id_cols}')
        
    # 最终输出前只保留节点列和分区列
    join_edges_df = join_edges_df.select(join_edges_id_cols)
    
    # 保存结果
    pyspark_optimal_save(join_edges_df, instance_table_schema["table_path"], instance_table_schema["table_format"], 
                         "append", instance_table_schema["partition_cols"], logger = logger)
    
    # 重新读取完整结果
    read_result = read_graph_table_partitions(spark, instance_table_schema)
    
    return read_result["data"]
