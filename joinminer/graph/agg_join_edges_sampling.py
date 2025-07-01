from .join_edges_aggregation import join_edges_aggregation
from .graph_loader import read_node_table, read_edge_table, read_label_table
from .graph_loader import read_graph_table_partitions, read_graph_table
from ..python import setup_logger, time_costing
from ..hdfs import hdfs_delete_dir, hdfs_check_file_exists, hdfs_list_contents
from ..pyspark import vectorize_and_scale, rename_columns, fill_null_vectors
from ..pyspark import pyspark_optimal_save

import copy
from functools import reduce
from pyspark.sql.functions import broadcast, col

# 应该优化成没有target_inst_config也能运行的状态

@time_costing
def agg_join_edges_sampling(spark, graph, target_inst_config, join_edges_list, logger = setup_logger()):
    # 检查inst_nodes是否和agg_nodes一致
    tgt_inst_nodes_types = target_inst_config["nodes_types"]
    for join_edges in join_edges_list:
        if tgt_inst_nodes_types != join_edges["inst_agg_nodes_types"]:
            raise ValueError(f"Require same aggregation nodes types of the target instances "
                             f"for all the join_edges.")
    
    # 获得实例特征相关运算结果存储的文件夹
    feat_table_path = target_inst_config["feat_table_path"]
    
    # 获得完整的特征表的存储位置
    summary_feat_table_path = feat_table_path + "/summary"

    # 设定完整结果对应的特征表的相关schema
    summary_feat_table_schema = {}
    summary_feat_table_schema["table_path"] = summary_feat_table_path
    summary_feat_table_schema["table_format"] = target_inst_config['feat_table_format']
    summary_feat_table_schema["partition_cols"] = list(graph.graph_time_cols_alias)
    summary_feat_table_schema["partition_cols_values"] = list(graph.graph_time_cols_values)

    # 尝试读取对应数据
    read_result = read_graph_table_partitions(spark, summary_feat_table_schema)

    # 检查是否读取到全部的结果
    if read_result["success"]:
        logger.info(f"Instance summary features already exists.")
        
        # 如果已有全部结果，则直接返回
        return read_result["data"]
        
    else:
        # 否则若图有时间列则获得未完成的时间点，则据此设定目标时间
        graph_time_cols = list(graph.graph_time_cols_alias)
        graph_time_cols_values = list(read_result["failed_partitions"])
        graph_time_cols_formats = list(graph.graph_time_cols_formats)
    
    # 获得目标节点对应的数据
    target_inst_df = target_inst_config["data"]
    
    # 获得目标节点对应的id列
    target_inst_id_cols = graph_time_cols + list(target_inst_config["nodes_cols"])
    
    # 获得只有id列的数据
    target_inst_id_df = target_inst_df.select(*target_inst_id_cols).distinct()
    target_inst_id_df.persist()

    # 用该变量记录target instance的各种来源的特征
    target_inst_feats = {}
    
    # 获得特征向量化后要采用的scale函数
    feats_scale_funcs = target_inst_config["feats_scale_funcs"]
    
    # 先读取target_inst本身对应的信息(待优化，添加对有多个节点列的inst也就是边的处理方案)
    target_inst_feats["self"] = {}
    if len(target_inst_config["nodes_types"]) == 1:
        # 获得对应的节点类型
        target_node_type = target_inst_config["nodes_types"][0]
        
        # 获得目标节点对应的节点列
        target_node_cols = target_inst_config["nodes_cols"]
        
        # 依次获得各个节点表包含的特征
        for node_table_i, node_table_name in enumerate(graph.nodes[target_node_type]["node_tables"]):
            # 获得该表对应结果的存储位置
            self_feat_table_path = feat_table_path + f"/self/{node_table_name}"
            
            logger.info(f"Start sampling self features for target nodes from node table "
                        f"{node_table_name} output to {self_feat_table_path}.")

            # 设定该节点表对应的inst特征表的相关schema
            self_feat_table_schema = {}
            self_feat_table_schema["table_path"] = self_feat_table_path
            self_feat_table_schema["table_format"] = target_inst_config['feat_table_format']
            self_feat_table_schema["partition_cols"] = list(graph_time_cols)
            self_feat_table_schema["partition_cols_values"] = list(graph_time_cols_values)
            
            # 尝试读取对应数据
            read_result = read_graph_table_partitions(spark, self_feat_table_schema)
        
            # 检查是否读取到全部的结果(最好这里还能基于未完成的时间更新下目标graph时间，但目前的格式不方便写，之后优化)
            if read_result["success"]:
                # 如果已有全部结果，则直接返回
                logger.info(f"Self features from node table {node_table_name} already exists.")
                target_inst_feats["self"][node_table_name] = read_result["data"]
                continue

            # 获取该节点对应的节点表的schema
            node_table_schema = graph.nodes[target_node_type]["node_tables"][node_table_name]

            # 如果该节点表没有特征就直接跳过
            if len(node_table_schema["feat_cols"]) == 0:
                logger.info(f"Node table {node_table_name} doesn't have features, skip.")
                continue

            # 获得从图中取出该节点表对应的信息的配置
            graph_table_schema = {}

            # 设定该节点表对应的名称
            graph_table_schema["name"] = f"Node_table_{node_table_name}_in_graph"

            # 设定该边对应到图里的目标时间
            graph_table_schema["graph_time_cols"] = list(graph_time_cols)
            graph_table_schema["graph_time_cols_values"] = list(graph_time_cols_values)
            graph_table_schema["graph_time_cols_formats"] = list(graph_time_cols_formats)

            # 设定该node table对应的source table的基础配置
            graph_table_schema["src_table_path"] = node_table_schema["src_table_path"]
            graph_table_schema["src_table_format"] = node_table_schema["src_table_format"]
            
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
            graph_table_schema["node_cols_to_aliases"] = [[x, y] for x, y in zip(node_table_schema["node_cols"], 
                                                                                 target_node_cols)]
            
            graph_table_schema["feat_cols_to_aliases"] = []
            for feat_col in node_table_schema["time_aggs_feat_cols"]:
                graph_table_schema["feat_cols_to_aliases"].append([feat_col, feat_col])
            
            graph_table_schema["time_cols_to_aliases"] = [[x, y] for x, y in zip(graph_time_cols, graph_time_cols)]

            # 读取该node对应的具体数据
            node_table_df = read_graph_table(spark, graph_table_schema, logger = logger)

            # 只保留target nodes相关结果
            tgt_inst_self_feat_df = target_inst_id_df.join(node_table_df, on = target_inst_id_cols, how = "left")
            
            # 将特征向量化及缩放
            tgt_inst_self_feat_df = vectorize_and_scale(tgt_inst_self_feat_df, feats_scale_funcs, target_inst_id_cols, 
                                                        node_table_schema["time_aggs_feat_cols"])
            
            # 修正向量化后的列名
            for scale_func in feats_scale_funcs:
                raw_vector_name = f"features_{scale_func}"
                new_vector_name = f"Self_feats_{node_table_i}_scale_{scale_func}"
                tgt_inst_self_feat_df = tgt_inst_self_feat_df.withColumnRenamed(raw_vector_name, new_vector_name)

            # 保证没有重复数据
            tgt_inst_self_feat_df = tgt_inst_self_feat_df.dropDuplicates(target_inst_id_cols)
            
            # 保存结果
            pyspark_optimal_save(tgt_inst_self_feat_df, self_feat_table_schema["table_path"], 
                                 self_feat_table_schema["table_format"], "append",
                                 self_feat_table_schema["partition_cols"], logger = logger)
        
            # 重新读取全量结果，来返回，避免因为惰性计算造成的重复运算
            read_result = read_graph_table_partitions(spark, self_feat_table_schema)
            target_inst_feats["self"][node_table_name] = read_result["data"]
    
    # 再依次读取各join_edges对应的信息
    target_inst_feats["agg_join_edges"] = {}
    for join_edges in join_edges_list:
        # 获得join_edges_aggregation对应的结果
        agg_join_edges = join_edges_aggregation(spark, graph, join_edges, logger = logger)
        
        # 获得对应的join_edges_name
        join_edges_name = join_edges['name']
        
        # 获得对应的采样后的特征的存储位置
        agg_join_edges_feat_path = feat_table_path + f"/agg_join_edges/{join_edges_name}"
        
        # 记录该instance对应的agg_join-edges的特征
        inst_agg_join_edges_feats = {}

        # 依次处理该join_edges的各个seq_token对应的数据
        for seq_token_index in agg_join_edges:
            logger.info(f"Start sampling features from {seq_token_index}-th sequential token of join-edges")

            # 记录该seq_token对应的特征
            inst_agg_join_edges_feats[seq_token_index] = {}
            
            # 依次处理该seq_token对应的各个表的信息 
            for table_name in agg_join_edges[seq_token_index]:
                # 获得该表对应结果的存储位置
                table_token_feat_path = agg_join_edges_feat_path + f"/{seq_token_index}/{table_name}"
                
                logger.info(f"Start sampling edge features for target nodes from table {table_name} of "
                             f"{seq_token_index}-th token of join-edges output to {table_token_feat_path}.")

                # 设定该节点表对应的inst特征表的相关schema
                token_feat_table_schema = {}
                token_feat_table_schema["table_path"] = table_token_feat_path
                token_feat_table_schema["table_format"] = target_inst_config['feat_table_format']
                token_feat_table_schema["partition_cols"] = list(graph_time_cols)
                token_feat_table_schema["partition_cols_values"] = list(graph_time_cols_values)
                
                # 尝试读取对应数据
                read_result = read_graph_table_partitions(spark, token_feat_table_schema)

                # 检查是否读取到全部的结果(最好这里还能基于未完成的时间更新下目标graph时间，但目前的格式不方便写，之后优化)
                if read_result["success"]:
                    # 如果已有全部结果，则直接返回
                    logger.info(f"Token features from table {table_name} already exists.")
                    inst_agg_join_edges_feats[seq_token_index][table_name] = read_result["data"]
                    continue
                
                # 获得该表对应的特征
                agg_token_df = agg_join_edges[seq_token_index][table_name]

                # 通过select语句修正节点和特征列名，并只保留所需要的列
                selected_columns = []
                for graph_time_col in graph_time_cols:
                    selected_columns.append(col(graph_time_col))
                for agg_node_i, agg_nodes_col in enumerate(join_edges["inst_agg_nodes_cols"]):
                    target_node_col = target_inst_config["nodes_cols"][agg_node_i]
                    selected_columns.append(col(agg_nodes_col).alias(target_node_col))
                for scale_func in feats_scale_funcs:
                    raw_vector_name = f"features_{scale_func}"
                    new_vector_name = f"{join_edges_name}_seq_{seq_token_index}_table_{table_name}_scale_{scale_func}"
                    selected_columns.append(col(raw_vector_name).alias(new_vector_name))
                agg_token_df = agg_token_df.select(*selected_columns)

                # 只保留target_nodes对应结果
                tgt_inst_feat_df = target_inst_id_df.join(agg_token_df, on = target_inst_id_cols, how = "left")

                # 获得该表对应的特征列数
                if "edge_type" in join_edges["flatten_format"]["seq"][seq_token_index]:
                    edge_type = join_edges["flatten_format"]["seq"][seq_token_index]["edge_type"]
                    time_aggs_feat_cols = graph.edges[edge_type]["edge_tables"][table_name]["time_aggs_feat_cols"]
                else:
                    node_type = join_edges["flatten_format"]["seq"][seq_token_index]["node_type"]
                    time_aggs_feat_cols = graph.nodes[node_type]["node_tables"][table_name]["time_aggs_feat_cols"]
                    
                feat_vector_len = len(time_aggs_feat_cols)

                # 补全各个空向量列
                for scale_func in feats_scale_funcs:
                    new_vector_name = f"{join_edges_name}_seq_{seq_token_index}_table_{table_name}_scale_{scale_func}"
                    
                    # 补全因为left join造成的空向量列
                    tgt_inst_feat_df = fill_null_vectors(spark, tgt_inst_feat_df, new_vector_name, feat_vector_len)

                # 保证没有重复数据
                tgt_inst_feat_df = tgt_inst_feat_df.dropDuplicates(target_inst_id_cols)
                
                # 保存结果
                pyspark_optimal_save(tgt_inst_feat_df, token_feat_table_schema["table_path"], 
                                     token_feat_table_schema["table_format"], "append",
                                     token_feat_table_schema["partition_cols"], logger = logger)
                
                # 重新读取全量结果，来返回，避免因为惰性计算造成的重复运算
                read_result = read_graph_table_partitions(spark, token_feat_table_schema)
                inst_agg_join_edges_feats[seq_token_index][table_name] = read_result["data"]
            
        target_inst_feats["agg_join_edges"][join_edges_name] = inst_agg_join_edges_feats
        
    # 将要合并的表都放入一个list中
    target_inst_feats_list = []
    for node_table_name in target_inst_feats["self"]:
        target_inst_feats_list.append(target_inst_feats["self"][node_table_name])
    for join_edges_name in target_inst_feats["agg_join_edges"]:
        for seq_i in target_inst_feats["agg_join_edges"][join_edges_name]:
            for table_name in target_inst_feats["agg_join_edges"][join_edges_name][seq_i]:
                target_inst_feats_list.append(target_inst_feats["agg_join_edges"][join_edges_name][seq_i][table_name])
    
    # 获得中间结果的存储位置
    summary_inter_table_path = feat_table_path + "/summary_intermediate"
    
    # 设定合并特征的起始数据
    inst_summary_feat_df = target_inst_df
    summary_feat_start_index = 0

    # 检查是否有已存在的中间结果
    if hdfs_check_file_exists(summary_inter_table_path):
        # 获得其中的全部文件夹
        intermediate_index_paths = hdfs_list_contents(summary_inter_table_path, "directories")

        # 按index序号由高到低排序
        intermediate_index_paths = sorted(intermediate_index_paths, key = lambda x: int(x.split("/")[-1]), 
                                          reverse=True)

        # 由高到低检查各个index对应的结果，来查找有已存在结果的最大的index，找到第一个有结果的就可以退出
        for index_path in intermediate_index_paths:
            # 设定中间结果表的相关schema
            intermediate_index_table_schema = {}
            intermediate_index_table_schema["table_path"] = index_path
            intermediate_index_table_schema["table_format"] = target_inst_config['feat_table_format']
            intermediate_index_table_schema["partition_cols"] = list(graph_time_cols)
            intermediate_index_table_schema["partition_cols_values"] = list(graph_time_cols_values)
            
            # 尝试读取对应数据
            read_result = read_graph_table_partitions(spark, intermediate_index_table_schema)
        
            # 检查是否读取到全部的结果
            if read_result["success"]:
                # 获得对应的index序号
                success_index = int(index_path.split('/')[-1])
                logger.info(f"Index {success_index} has existing results.")

                # 记录该index对应的结果 
                inst_summary_feat_df = read_result["data"]
                
                # 更新开始计算summary_feat的index序号 
                summary_feat_start_index = success_index + 1

                break

    # 获得隔几组持久化一次数据
    feats_persist_interval = target_inst_config["feats_persist_interval"]
    
    # 获得隔几组存一次数据
    feats_save_interval = target_inst_config["feats_save_interval"]

    # 记录当前persist的中间结果
    persisted_feat_dfs = []
    
    # 合并join
    for summary_feat_index in range(summary_feat_start_index, len(target_inst_feats_list)):
        logger.info(f"Combine {summary_feat_index}-th feature vector.")
        
        tgt_inst_feat_df = target_inst_feats_list[summary_feat_index]

        # 保证没有重复数据
        tgt_inst_feat_df = tgt_inst_feat_df.dropDuplicates(target_inst_id_cols)
        
        inst_summary_feat_df = inst_summary_feat_df.join(tgt_inst_feat_df, on = target_inst_id_cols, how = "left")

        # 如果不是最后一位，则检查是否需要保留中间结果，防止join失败
        if (summary_feat_index + 1) < len(target_inst_feats_list):
            # join成功feats_save_interval个后保存一次结果
            if (summary_feat_index + 1) % feats_save_interval == 0:
                # 设定结果保存的路径
                index_path = summary_inter_table_path + f'/{summary_feat_index}'
    
                # 以最优分区配置保存结果到对应的表格 
                pyspark_optimal_save(inst_summary_feat_df, index_path, target_inst_config['feat_table_format'], 
                                     "append", graph_time_cols, logger = logger)
    
                # 释放之前persist的数据
                for persisted_feat_df in persisted_feat_dfs:
                    persisted_feat_df.unpersist()
                persisted_feat_dfs = []
                
                # 重新读取出这个保存的结果，防止之后重复运算  
                intermediate_index_table_schema = {}
                intermediate_index_table_schema["table_path"] = index_path
                intermediate_index_table_schema["table_format"] = target_inst_config['feat_table_format']
                intermediate_index_table_schema["partition_cols"] = list(graph_time_cols)
                intermediate_index_table_schema["partition_cols_values"] = list(graph_time_cols_values)
                read_result = read_graph_table_partitions(spark, intermediate_index_table_schema)
                inst_summary_feat_df = read_result["data"]
    
            elif (summary_feat_index + 1) % feats_persist_interval == 0:
                # join成功feats_persist_interval个后持久化一次结果
                inst_summary_feat_df.persist()
                persisted_feat_dfs.append(inst_summary_feat_df)
    
    # 存储结果
    pyspark_optimal_save(inst_summary_feat_df, summary_feat_table_schema["table_path"], 
                         summary_feat_table_schema['table_format'], "append", 
                         summary_feat_table_schema["partition_cols"], logger = logger)

    # 释放之前persist的数据
    for persisted_feat_df in persisted_feat_dfs:
        persisted_feat_df.unpersist()
    persisted_feat_dfs = []
    
    # 删除summary_inter_table_path下的全部文件
    if hdfs_check_file_exists(summary_inter_table_path):
        hdfs_delete_dir(summary_inter_table_path)
        
    # 重新读取全量结果，来返回，避免因为惰性计算造成的重复运算
    read_result = read_graph_table_partitions(spark, summary_feat_table_schema)
    
    return read_result["data"]