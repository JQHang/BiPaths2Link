from .join_edge_query import join_edge_query
from .graph_token_query import graph_token_query
from ..python import time_costing
from ..hdfs import hdfs_check_partitions
from ..pyspark import random_n_sample, top_n_sample, threshold_n_sample
from ..pyspark import pyspark_read_table, pyspark_optimal_save, fill_null_vectors
from ..pyspark import pyspark_vector_aggregate

import copy
import logging
from pyspark.sql.functions import col
from pyspark.ml.functions import vector_to_array

# 获得logger
logger = logging.getLogger(__name__)

@time_costing
def join_edges_query(spark, graph, join_edges, query_config):
    """
    针对给定的复杂路类型获取对应的全部complex-path instances

    输入：

    返回值：
        
    """
    # 获得该join_edges的名称及对应的query_config
    join_edges_name = join_edges['name']
    logger.info(f"Query join-edges {join_edges_name}")

    # 获得query该join_edges所需的配置
    join_edges_config = join_edges["query_config"]

    # 获得目标结果的存储路径
    # 如果有设定的具体目标点就用具体目标点对应的存储路径，没有就用全量数据存储路径
    if "tgt_query_nodes" in query_config:
        # 若有目标节点，则设定为目标节点结果目录下对应路径文件夹
        result_path = query_config["tgt_query_nodes"]["result_path"] + f"/join_edges/{join_edges_name}"
    else:
        # 如果没有目标节点，则直接设为全量结果的存储路径
        result_path = join_edges_config['result_path']
    
    # 获得结果对应的存储格式
    result_format = join_edges_config['result_format']

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
        join_edges_df = pyspark_read_table(spark, result_path, result_format, partition_cols, partition_cols_values)
        
        return join_edges_df

    # 如果是取部分数据，可以额外检查下是否有对应的全量结果，内积一下就行,目前的写法也不完善，以后优化
    if "tgt_query_nodes" in query_config:
        # 检查全量结果是否存在
        is_complete, _ = hdfs_check_partitions(join_edges_config['result_path'], partition_cols, missing_values)
        
        # 如果已有对应的全量结果，则直接读取对应的结果并返回
        if is_complete:
            # 读取对应结果
            join_edges_df = pyspark_read_table(spark, join_edges_config['result_path'], result_format, partition_cols, missing_values)

            # 获得具体的query nodes
            query_nodes_df = query_config["tgt_query_nodes"]["df"]
            
            # 只保留query_config对应的结果
            join_edges_df = join_edges_df.join(query_nodes_df, on = query_nodes_df.columns, how = "inner")
            
            # 保存结果
            pyspark_optimal_save(join_edges_df, result_path, result_format, "overwrite", partition_cols)

            # 重新读取完整结果
            join_edges_df = pyspark_read_table(spark, result_path, result_format, partition_cols, partition_cols_values)
            
            return join_edges_df

    # 如果要继续计算的话，先以未完成的时间更新query_config，用于取父路径和起始join_edge
    logger.info(f"Missing target partitions: {missing_values}")
    missing_query_config = {}
    missing_query_config["graph_time_values"] = missing_values
    if "tgt_query_nodes" in query_config:
        missing_query_config["tgt_query_nodes"] = {}
        missing_query_config["tgt_query_nodes"]["result_path"] = query_config["tgt_query_nodes"]["result_path"]
        missing_query_config["tgt_query_nodes"]["df"] = query_config["tgt_query_nodes"]["df"]
        
    # 还要设定针对其他的要全量取数的部分的query_config
    join_edges_query_config = {}
    join_edges_query_config["graph_time_values"] = missing_values

    # 检查是否有父路径的配置信息
    join_edges_start_k = 0
    if "parent_join_edges" in join_edges:
        parent_join_edges = join_edges["parent_join_edges"]
        
        # 获取父路径对应的结果(目前假设设定了父路径就一定有对应结果，所以没有schema，之后优化)
        join_edges_df = join_edges_query(spark, graph, parent_join_edges, missing_query_config)

        # 获得父路径对应的跳数，从该跳接着运算 
        join_edges_start_k = parent_join_edges["join_edges_len"]

    # 依次处理各个join_edge
    for join_edge_k in range(join_edges_start_k, len(join_edges_config["join_edge_list"])):
        logger.info(f"Process the {join_edge_k}-th join_edge query config")

        # 获得对应的配置信息
        join_edges_edge_config = join_edges_config["join_edge_list"][join_edge_k]

        # 获得对应的join_edge配置信息
        join_edge_config = join_edges_edge_config["join_edge"]
        
        # 读取对应的join_edge
        join_edge_df = join_edge_query(spark, graph, join_edge_config, join_edges_query_config)
        
        # 基于join_edges内的配置只保留指定列并修正该join_edge的列名
        select_cols = [col(column).alias(alias) for column, alias in join_edges_edge_config["col_aliases"]]
        join_edge_df = join_edge_df.select(*select_cols)

        # 如果是第一条，且有规定的目标点，则只取目标点对应信息
        if join_edge_k == 0 and "tgt_query_nodes" in missing_query_config:
            tgt_query_nodes_df = missing_query_config["tgt_query_nodes"]["df"]
            join_edge_df = join_edge_df.join(tgt_query_nodes_df, on = tgt_query_nodes_df.columns, how = "inner")
        
        # 和之前的join_edges进行join
        if join_edge_k == 0:
            join_edges_df = join_edge_df
        else:
            # 获得该次join要使用的id列
            edge_join_id_cols = join_edges_edge_config["join_cols"]
            
            # 获得通过何种类型的join将该边连接到之前的join_edges中  
            edge_join_type = join_edges_edge_config["join_type"]
            
            # 将edge_table join到之前的join_edges中
            logger.info(f'{edge_join_type} join {join_edge_k}-th edge table to join-edges on: {edge_join_id_cols}')
            join_edges_df = join_edges_df.join(join_edge_df, on = edge_join_id_cols, how = edge_join_type)
    
        # Join edges limit
        if 'join_edges_limit' in join_edges_edge_config:
            # 依次为join_edge添加各个feat_source对应的特征
            src_feat_cols = []
            for feat_source in join_edges_edge_config["join_edges_limit"]["feat_sources"]:
                # 如果来源于其中某个节点
                if "node_type" in feat_source:
                    # 获得该node对应的数据
                    node_df = graph_token_query(spark, graph, "node", feat_source["node_type"], join_edges_query_config)
    
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
                
                    # 将对应的特征列join到join_edges上
                    src_id_cols = feat_source["node_cols"] + list(graph.graph_time_cols_alias)
                    join_edges_df = join_edges_df.join(node_df, on = src_id_cols, how = "left")
                else:
                    # 否则就是来源于某条join_edge，先获得该join_edge对应的配置
                    join_edge_index = feat_source["join_edge_index"]
                    join_edge_config = join_edges_config["join_edge_list"][join_edge_index]["join_edge"]

                    # 获得该join_edge对应的数据
                    join_edge_df = join_edge_query(spark, graph, join_edge_config, join_edges_query_config)
                    
                    # 获得该边特征向量列对应列名
                    feat_vec_col = graph.edges[join_edge_config["edge_type"]]["query_config"]["assembled_feat_col"]
                    feat_array_col = feat_vec_col + "_array"
                
                    # 将vector列转array
                    join_edge_df = join_edge_df.withColumn(feat_array_col, vector_to_array(feat_vec_col))
            
                    # 获得feat_source对应的特征列
                    join_edge_col_aliases = join_edges_config["join_edge_list"][join_edge_index]["col_aliases"]
                    select_cols = [col(column).alias(alias) for column, alias in join_edge_col_aliases]
                    for feat_pos, feat_col in feat_source["feat_cols"]:
                        select_cols.append(col(feat_array_col)[feat_pos].alias(feat_col))
                        src_feat_cols.append(feat_col)
            
                    logger.info(f"Select source features {select_cols} from edge {join_edge_config['edge_type']} "
                                f"feature vector column {feat_vec_col}")
                    
                    # 获得对应的列
                    join_edge_df = join_edge_df.select(*select_cols)

                    # 获得join_edge对应的id列
                    src_id_cols = [alias for column, alias in join_edge_col_aliases]
                    
                    # 将对应的特征列join到join_edges上
                    join_edges_df = join_edges_df.join(join_edge_df, on = src_id_cols, how = "left")
            
            logger.info(f"Join edges limit: {join_edges_edge_config['join_edges_limit']['limit']}")

            # 对join上各个特征的join_edge完成限制 
            join_edges_df = join_edges_df.filter(join_edges_edge_config["join_edges_limit"]['limit'])
    
            # 删去增加的特征列 
            join_edges_df = join_edges_df.drop(*src_feat_cols)
        
        # Join edges Sample
        if 'join_edges_samples' in join_edges_edge_config:
            for join_edges_sample in join_edges_edge_config["join_edges_samples"]:
                sample_nodes_cols = join_edges_sample['sample_nodes_cols']
                sample_type = join_edges_sample['sample_type']
                sample_count = join_edges_sample['sample_count']
    
                sample_id_cols = sample_nodes_cols + list(graph.graph_time_cols_alias)
                
                logger.info(f'Join_Edges Sampling: {sample_type}, {sample_id_cols}, {sample_count}')
    
                if sample_type == 'random':
                    join_edges_df = random_n_sample(spark, join_edges_df, sample_id_cols, sample_count)
                elif sample_type == 'threshold':
                    join_edges_df = threshold_n_sample(spark, join_edges_df, sample_id_cols, sample_count)

        # 检查是否要补全特征 
        if "feat_add_srcs" in join_edges_edge_config:
            # 依次处理各个来源要补全的特征
            for feat_source in join_edges_edge_config["feat_add_srcs"]:
                if "node_type" in feat_source:
                    logger.info(f"Add features from node type {feat_source['node_type']}")
                    
                    # 获得该node对应的数据
                    node_df = graph_token_query(spark, graph, "node", feat_source["node_type"], join_edges_query_config)

                    # 要使用的列及别名
                    select_cols = [col(column).alias(alias) for column, alias in feat_source["col_aliases"]]
            
                    # 获得对应的列
                    node_df = node_df.select(*select_cols)

                    # 将对应的特征列join到join_edges上
                    join_edges_df = join_edges_df.join(node_df, on = feat_source["join_cols"], how = "left")

                    # 补全空向量列
                    join_edges_df = fill_null_vectors(spark, join_edges_df, feat_source["feat_vec_col"], 
                                                      feat_source["feat_vec_len"])
                else:
                    logger.info(f"Add features from {feat_source['join_edge_index']}-th join edge.")
                    
                    # 获得对应的配置信息
                    join_edge_index = feat_source['join_edge_index']
                    add_join_edge_config = join_edges_config["join_edge_list"][join_edge_index]["join_edge"]
            
                    # 读取对应的join_edge
                    join_edge_df = join_edge_query(spark, graph, add_join_edge_config, join_edges_query_config)

                    # 要使用的列及别名
                    select_cols = [col(column).alias(alias) for column, alias in feat_source["col_aliases"]]
            
                    # 获得对应的列
                    join_edge_df = join_edge_df.select(*select_cols)

                    # 将对应的特征列join到join_edges上
                    join_edges_df = join_edges_df.join(join_edge_df, on = feat_source["join_cols"], how = "left")

                    # 补全空向量列
                    join_edges_df = fill_null_vectors(spark, join_edges_df, feat_source["feat_vec_col"], 
                                                      feat_source["feat_vec_len"])

        # 检查是否需要重新分区以及调整分区数量，以后优化      
        # exist_join_edges_id_cols = list(set(join_edges_id_cols) & set(join_edges_df.columns))
        # join_edges_df = join_edges_df.repartition(*exist_join_edges_id_cols)
        # logger.info(f'Repartition the {join_edge_k}-th join-edges result with columns: {exist_join_edges_id_cols}')

    # 检查是否需要聚合，这个以后可以优化到join_edges内部
    # 在join_edges内部进行aggregation时，补全特征就应该从aggregate结果往后补
    if "join_edges_agg" in join_edges_config:
        group_cols = join_edges_config["join_edges_agg"]["group_cols"]
        agg_config = join_edges_config["join_edges_agg"]["agg_config"]
        join_edges_df = pyspark_vector_aggregate(join_edges_df, group_cols, agg_config)

    # # 将特征列转成array列
    # join_edges_edge_config = join_edges_config["join_edge_list"][-1]
    # if "feat_add_srcs" in join_edges_edge_config:
    #     # 依次处理各个来源要补全的特征
    #     for feat_source in join_edges_edge_config["feat_add_srcs"]:
    #         join_edges_df = join_edges_df.withColumn(feat_source["feat_vec_col"], vector_to_array(col(feat_source["feat_vec_col"])))

    # 保存结果，带着特征列保存，因为已经带着算过了，帮助节约后续运算
    col_sizes = join_edges_config["feat_cols_sizes"]
    pyspark_optimal_save(join_edges_df, result_path, result_format, "overwrite", partition_cols, missing_values,
                         col_sizes = col_sizes)
    
    # 重新读取完整结果
    join_edges_df = pyspark_read_table(spark, result_path, result_format, partition_cols, partition_cols_values)

    return join_edges_df
