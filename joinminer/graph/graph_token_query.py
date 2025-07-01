from .time_agg_query import time_agg_query
from ..hdfs import hdfs_check_partitions
from ..pyspark import pyspark_read_table, pyspark_optimal_save, fill_null_vectors
from ..python import time_costing

import copy
import logging
from pyspark.sql.functions import lit
from pyspark.ml.feature import VectorSizeHint, VectorAssembler

# 获得logger
logger = logging.getLogger(__name__)

@time_costing
def graph_token_query(spark, graph, graph_token_type, graph_token_name, query_config):
    logger.info(f"Query {graph_token_type} type graph token {graph_token_name}")

    # 获得要query该graph token对应的全量数据的配置信息
    if graph_token_type == "node":
        graph_token_config = graph.nodes[graph_token_name]["query_config"]
    else:
        assert graph_token_type == "edge", f"Unknown graph token type"
        graph_token_config = graph.edges[graph_token_name]["query_config"]

    # 保证有time_agg相关配置，不然query没意义
    assert len(graph_token_config["time_agg"]) > 0, f"Useless query type {graph_token_name}"
    
    # 获得本次query的目标结果的存储路径，目前一定执行全量计算
    result_path = graph_token_config["result_path"]

    # 获得结果对应的存储格式
    result_format = graph_token_config["result_format"]
    
    # 获得结果对应的分区列
    partition_cols = list(graph.graph_time_cols_alias)

    # 获得结果对应的分区值
    partition_cols_values = query_config["graph_time_values"]
    
    logger.info(f"The reulst will be output to: {result_path} in {result_format} format "
                f"with partition cols {partition_cols} and values {partition_cols_values}.")

    # 检查该路径下是否已有对应结果(返回是否有全量结果以及没有全量结果的话缺失哪些分区的结果)
    is_complete, missing_values = hdfs_check_partitions(result_path, partition_cols, partition_cols_values)
    
    # 如果已有对应的全量结果，则直接读取对应的结果并返回
    if is_complete:
        graph_token_df = pyspark_read_table(spark, result_path, result_format, partition_cols, partition_cols_values)
        return graph_token_df
        
    # 以未完成的时间更新query_config，因为time_agg一定是全量更新，所以只需要目标时间就行
    logger.info(f"Missing target partitions: {missing_values}")
    missing_query_config = {}
    missing_query_config["graph_time_values"] = missing_values
    
    # 依次query各个所需的time_agg表
    time_agg_df_list = []
    for time_agg_config in graph_token_config["time_agg"]:
        # 读取对应的time_agg表
        time_agg_df = time_agg_query(spark, graph, time_agg_config, missing_query_config)

        # 加入null标识列
        time_agg_df = time_agg_df.withColumn(time_agg_config["null_mark_col"], lit(1))
        
        # 记录对应的time_agg表
        time_agg_df_list.append(time_agg_df)

    # 获得对应的id列，用于合并各个time_agg_df
    df_id_cols = graph_token_config["node_cols"] + partition_cols

    # 通过外连接合并全部time_agg表
    graph_token_df = time_agg_df_list[0]
    for time_agg_df in time_agg_df_list[1:]:
        # 合并聚合后的结果
        graph_token_df = graph_token_df.join(time_agg_df, on = df_id_cols, how = "outer")

    # repartition保证分布均匀，从而加速向量化计算
    graph_token_df = graph_token_df.repartition(*df_id_cols)

    # 针对各个time_agg的结果向量补全各个空的特征向量和特征列
    all_time_aggs_feat_cols = []
    for time_agg_config in graph_token_config["time_agg"]:
        # 为特征向量列补全0向量，可能需要增加向量列是sparse还是dense的检测
        graph_token_df = fill_null_vectors(spark, graph_token_df, time_agg_config["time_aggs_feat_vec"], 
                                           len(time_agg_config["time_aggs_feat_cols"]))

        # 加入对vector列长度的hint,这样可以在assembler里补全空向量
        hint = VectorSizeHint(
                    inputCol=time_agg_config["time_aggs_feat_vec"],
                    size=len(time_agg_config["time_aggs_feat_cols"]),
                    handleInvalid="error"
                )
        graph_token_df = hint.transform(graph_token_df)
        
        # 为null值标记列补全0
        graph_token_df = graph_token_df.fillna(0, subset = [time_agg_config["null_mark_col"]])

        # 记录这两个特征列
        all_time_aggs_feat_cols.append(time_agg_config["time_aggs_feat_vec"])
        all_time_aggs_feat_cols.append(time_agg_config["null_mark_col"])
        
    # 合并全部的特征向量
    assembler = VectorAssembler(
                    inputCols=all_time_aggs_feat_cols,
                    outputCol=graph_token_config["assembled_feat_col"],
                    handleInvalid="error"
                )
    graph_token_df = assembler.transform(graph_token_df).drop(*all_time_aggs_feat_cols)

    # 保存最终结果
    col_sizes = {graph_token_config["assembled_feat_col"]: 12 * len(graph_token_config["assembling_feat_cols"]) / 2}
    pyspark_optimal_save(graph_token_df, result_path, result_format, "overwrite", partition_cols, missing_values,
                         col_sizes = col_sizes)

    # 读取最终的结果
    graph_token_df = pyspark_read_table(spark, result_path, result_format, partition_cols, partition_cols_values)

    return graph_token_df