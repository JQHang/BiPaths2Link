from .graph_token_query import graph_token_query
from ..python import time_costing
from ..hdfs import hdfs_check_partitions
from ..pyspark import random_n_sample, top_n_sample, threshold_n_sample, rename_columns
from ..pyspark import pyspark_read_table, pyspark_optimal_save

import copy
import logging
from pyspark.sql.functions import col
from pyspark.ml.functions import vector_to_array

# 获得logger
logger = logging.getLogger(__name__)

@time_costing
def join_edge_query(spark, graph, join_edge_config, query_config):
    """
    针对给定的复杂路类型获取对应的全部complex-path instances

    输入：

    返回值：
        
    """
    join_edge_name = join_edge_config['name']
    logger.info(f"Query join-edge {join_edge_name}")

    # 获得目标结果的存储路径，目前对join_edge一定执行全量运算，因为会经常使用
    # 有目标点就从全量结果中取对应数据即可
    result_path = join_edge_config['result_path']

    # 获得结果对应的存储格式
    result_format = join_edge_config['result_format']

    # 获得结果对应的分区列
    partition_cols = list(graph.graph_time_cols_alias)

    # 获得结果对应的目标分区值
    partition_cols_values = query_config["graph_time_values"]
    
    logger.info(f"The reulst will be output to: {result_path} in {result_format} format "
                f"with partition cols {partition_cols} and values {partition_cols_values}.")
    
    # 检查该路径下是否已有对应结果(返回是否有全量结果以及没有全量结果的话缺失哪些分区的结果)
    is_complete, missing_values = hdfs_check_partitions(result_path, partition_cols, partition_cols_values)

    # 如果已有对应的全量结果，则直接读取对应的结果并返回
    if is_complete:
        # 读取对应结果
        join_edge_df = pyspark_read_table(spark, result_path, result_format, partition_cols, partition_cols_values)
        
        return join_edge_df

    # 以未完成的时间更新query_config,因为下面的graph_token都是全量计算，所以只要记录时间就行
    logger.info(f"Missing target partitions: {missing_values}")
    missing_query_config = {}
    missing_query_config["graph_time_values"] = missing_values

    # 先读取该edge对应的数据
    join_edge_df = graph_token_query(spark, graph, "edge", join_edge_config["edge_type"], missing_query_config)

    # 检查并处理edge_limit
    if "edge_limit" in join_edge_config:
        # 获得特征向量列对应列名
        feat_vec_col = graph.edges[join_edge_config["edge_type"]]["query_config"]["assembled_feat_col"]
        feat_array_col = feat_vec_col + "_array"
            
        # 将vector列转array
        join_edge_df = join_edge_df.withColumn(feat_array_col, vector_to_array(feat_vec_col))

        # 获得edge_limit对应的特征列
        limit_feat_cols = []
        select_cols = [c for c in join_edge_df.columns if c != feat_array_col]
        for feat_pos, feat_col in join_edge_config["edge_limit"]["feat_cols"]:
            select_cols.append(col(feat_array_col)[feat_pos].alias(feat_col))
            limit_feat_cols.append(feat_col)

        logger.info(f"Edge limit {join_edge_config['edge_limit']['limit']} for features {limit_feat_cols} "
                    f"from vector column {feat_vec_col}")
        
        # 获得对应的列
        join_edge_df = join_edge_df.select(*select_cols)

        # 进行限制
        join_edge_df = join_edge_df.filter(join_edge_config['edge_limit']['limit'])

        # 删去新增的这些特征列
        join_edge_df = join_edge_df.drop(*limit_feat_cols)
        
    # 检查并处理edge_sample
    if "edge_samples" in join_edge_config:
        for edge_sample in join_edge_config["edge_samples"]:
            edge_sample_nodes_cols = edge_sample['sample_nodes_cols']
            edge_sample_type = edge_sample['sample_type']
            edge_sample_count = edge_sample['sample_count']

            sample_id_cols = edge_sample_nodes_cols + list(graph.graph_time_cols_alias)
            
            logger.info(f'Edge Sampling: {edge_sample_type}, {sample_id_cols}, {edge_sample_count}')

            if edge_sample_type == 'random':
                join_edge_df = random_n_sample(spark, join_edge_df, sample_id_cols, edge_sample_count)
            elif edge_sample_type == 'threshold':
                join_edge_df = threshold_n_sample(spark, join_edge_df, sample_id_cols, edge_sample_count)

    # 检查并处理node_limit
    if "nodes_limit" in join_edge_config:
        for node_limit in join_edge_config["nodes_limit"]:
            # 获得该node对应的数据
            node_df = graph_token_query(spark, graph, "node", node_limit["node_type"], missing_query_config)

            # 获得特征向量列对应列名
            feat_vec_col = graph.nodes[node_limit["node_type"]]["query_config"]["assembled_feat_col"]
            feat_array_col = feat_vec_col + "_array"
            
            # 将vector列转array
            node_df = node_df.withColumn(feat_array_col, vector_to_array(feat_vec_col))
            
            # 获得node_limit对应的特征列
            limit_feat_cols = []
            select_cols = [col(column).alias(alias) for column, alias in node_limit["col_aliases"]]
            for feat_pos, feat_col in node_limit["feat_cols"]:
                select_cols.append(col(feat_array_col)[feat_pos].alias(feat_col))
                limit_feat_cols.append(feat_col)
    
            logger.info(f"Node limit {node_limit['limit']} for node {node_limit['node_type']} features "
                        f"{limit_feat_cols} from vector column {feat_vec_col}")

            # 获得对应的列
            node_df = node_df.select(*select_cols)
    
            # 进行限制
            node_df = node_df.filter(node_limit['limit'])
    
            # 删去特征列
            node_df = node_df.drop(*limit_feat_cols)

            logger.info(f"Limit node id columns {node_df.columns}")
            
            # inner join到join_edge上完成限制 
            join_edge_df = join_edge_df.join(node_df, on = node_df.columns, how = "inner")
            
    # 检查并处理join_edge_limit  
    if "join_edge_limit" in join_edge_config:
        # 依次为join_edge添加各个feat_source对应的特征
        src_feat_cols = []
        for feat_source in join_edge_config["join_edge_limit"]["feat_sources"]:
            # 如果来源于其中某个节点
            if "node_type" in feat_source:
                # 获得该node对应的数据
                node_df = graph_token_query(spark, graph, "node", feat_source["node_type"], missing_query_config)

                # 获得特征向量列对应列名
                feat_vec_col = graph.nodes[feat_source["node_type"]]["query_config"]["assembled_feat_col"]
                feat_array_col = feat_vec_col + "_array"
            
                # 将vector列转array
                node_df = node_df.withColumn(feat_array_col, vector_to_array(feat_vec_col))
            
                # 获得feat_source对应的特征列
                select_cols = [col(column).alias(alias) for column, alias in feat_source["col_aliases"]]
                for feat_pos, feat_col in feat_source["feat_cols"]:
                    select_cols.append(col(feat_array_col)[feat_pos].alias(feat_col))
                    src_feat_cols.append(feat_col)
        
                logger.info(f"Select source features {select_cols} from node {feat_source['node_type']} "
                            f"feature vector column {feat_vec_col}")

                # 获得对应的列
                node_df = node_df.select(*select_cols)
            
                # 将对应的特征列join到join_edge上
                src_id_cols = feat_source["node_cols"] + list(graph.graph_time_cols_alias)
                join_edge_df = join_edge_df.join(node_df, on = src_id_cols, how = "left")
            
            else:
                # 否则就是来源于该边，获得该边特征向量列对应列名
                feat_vec_col = graph.edges[join_edge_config["edge_type"]]["query_config"]["assembled_feat_col"]
                feat_array_col = feat_vec_col + "_array"
            
                # 将vector列转array
                join_edge_df = join_edge_df.withColumn(feat_array_col, vector_to_array(feat_vec_col))
        
                # 获得feat_source对应的特征列
                select_cols = [c for c in join_edge_df.columns if c != feat_array_col]
                for feat_pos, feat_col in feat_source["feat_cols"]:
                    select_cols.append(col(feat_array_col)[feat_pos].alias(feat_col))
                    src_feat_cols.append(feat_col)
        
                logger.info(f"Select source features {select_cols} from edge {join_edge_config['edge_type']} "
                            f"feature vector column {feat_vec_col}")
                
                # 获得对应的列
                join_edge_df = join_edge_df.select(*select_cols)
        
        logger.info(f"Join edge limit: {join_edge_config['join_edge_limit']['limit']}")

        # 对join上各个特征的join_edge完成限制 
        join_edge_df = join_edge_df.filter(join_edge_config['join_edge_limit']['limit'])

        # 删去增加的特征列 
        join_edge_df = join_edge_df.drop(*src_feat_cols)

    # 检查并处理join_edge_sample 
    if "join_edge_samples" in join_edge_config:
        for join_edge_sample in join_edge_config["join_edge_samples"]:
            edge_sample_nodes_cols = join_edge_sample['sample_nodes_cols']
            edge_sample_type = join_edge_sample['sample_type']
            edge_sample_count = join_edge_sample['sample_count']

            sample_id_cols = edge_sample_nodes_cols + list(graph.graph_time_cols_alias)
            
            logger.info(f'Join_Edge Sampling: {edge_sample_type}, {sample_id_cols}, {edge_sample_count}')

            if edge_sample_type == 'random':
                join_edge_df = random_n_sample(spark, join_edge_df, sample_id_cols, edge_sample_count)
            elif edge_sample_type == 'threshold':
                join_edge_df = threshold_n_sample(spark, join_edge_df, sample_id_cols, edge_sample_count)

    # 保存结果，带着特征列保存，因为已经带着算过了，帮助节约后续运算
    pyspark_optimal_save(join_edge_df, result_path, result_format, "overwrite", partition_cols, missing_values)
    
    # 重新读取完整结果
    join_edge_df = pyspark_read_table(spark, result_path, result_format, partition_cols, partition_cols_values)

    return join_edge_df
