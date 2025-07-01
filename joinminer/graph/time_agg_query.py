from .src_table_query import src_table_query
from ..hdfs import hdfs_check_partitions
from ..pyspark import pyspark_read_table, pyspark_optimal_save, pyspark_aggregate
from ..python import time_values_reformat, time_costing

import copy
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pyspark.sql.functions import broadcast, col
from pyspark.ml.feature import VectorAssembler

# 获得logger
logger = logging.getLogger(__name__)

def get_agg_time_mapping(tgt_time_cols_values_list, time_cols_formats, time_ranges):
    src_time_cols_values_list = []
    agg_time_map_list = []

    # 依次处理各个目标时间
    for tgt_time_cols_values in tgt_time_cols_values_list:
        # 获得该目标时间对应的date数据
        target_date = datetime.strptime(''.join(tgt_time_cols_values), 
                                        ''.join(time_cols_formats))

        # 依次处理各个time_range
        for time_range in time_ranges:
            time_range_name = time_range["name"]
            
            # 处理每个时间点配置
            for time_point in time_range["time_points"]:
                src_time_cols_values = []
    
                # 目前只有相对时间点一种方案，剩下的以后优化
                assert time_point["time_type"] == "relative"
                
                # 计算相对时间点
                time_unit = time_point["time_unit"]
                time_interval = time_point["time_interval"]
                
                # 根据时间单位计算聚合时间点
                if time_unit == "day":
                    src_date = target_date + relativedelta(days=time_interval)
                elif time_unit == "month":
                    src_date = target_date + relativedelta(months=time_interval)
                elif time_unit == "year":
                    src_date = target_date + relativedelta(years=time_interval)
                else:
                    raise ValueError(f"Unsupported time unit: {time_unit}")
                
                # 将datetime转换为指定格式的时间列值
                src_time_cols_values = [
                    src_date.strftime(time_col_format)
                    for time_col_format in time_cols_formats
                ]
                
                # 记录聚合时间点和映射关系
                if src_time_cols_values not in src_time_cols_values_list:
                    src_time_cols_values_list.append(src_time_cols_values)
                
                agg_time_map_list.append(src_time_cols_values + tgt_time_cols_values + [time_range_name])
    
    return src_time_cols_values_list, agg_time_map_list

# 不同的source表可能有不同的时间列和格式，在这里得统一下，time_agg返回的结果应该是各个graph时间点对应的数据
def standardize_time_cols(spark, df, src_time_cols, src_time_values_list, dst_time_cols, dst_time_values_list):
    """
    Standardize time columns in Spark DataFrame to unified format and column names
    """
    assert len(src_time_values_list) == len(dst_time_values_list)
    
    # Create mapping DataFrame with original and target time values
    mapping_data = [src_values + dst_values 
                    for src_values, dst_values in zip(src_time_values_list, dst_time_values_list)]
    mapping_cols = src_time_cols + dst_time_cols
    mapping_df = spark.createDataFrame(mapping_data, mapping_cols)
    
    # Join with original DataFrame and drop old columns
    result_df = df.join(broadcast(mapping_df), src_time_cols, "left").drop(*src_time_cols)
    
    return result_df

@time_costing
def time_agg_query(spark, graph, time_agg_config, query_config):
    time_agg_name = time_agg_config['name']
    logger.info(f"Query time_aggregation table {time_agg_name}")

    # 获得对应的source表的配置
    src_table_config = time_agg_config["src_table"]
    
    # 获得本次query的目标结果的存储路径，目前一定执行全量运算
    result_path = time_agg_config["result_path"]

    # 获得结果对应的存储格式
    result_format = time_agg_config["result_format"]
    
    # 获得结果对应的分区列，和source表的分区列对应
    partition_cols = src_table_config["time_cols"]

    # 获得graph目标时间对应到的source表时间
    tgt_time_cols_values = time_values_reformat(query_config["graph_time_values"], 
                                                graph.graph_time_cols_formats,
                                                src_table_config["time_cols_formats"])
    
    # 将去重后的结果作为目标分区值
    unique_values = set()
    partition_cols_values = []
    for time_cols_values in tgt_time_cols_values:
        values_tuple = tuple(time_cols_values)
        if values_tuple not in unique_values:
            unique_values.add(values_tuple)
            partition_cols_values.append(time_cols_values)
    
    logger.info(f"The reulst will be output to: {result_path} in {result_format} format "
                f"with partition cols {partition_cols} and values {partition_cols_values}.")

    # 检查该路径下是否已有对应结果(返回是否有全量结果以及没有全量结果的话缺失哪些分区的结果)
    is_complete, missing_values = hdfs_check_partitions(result_path, partition_cols, partition_cols_values)
    
    # 如果已有对应的全量结果，则直接读取对应的结果并返回
    if is_complete:
        # 读取对应结果
        time_agg_df = pyspark_read_table(spark, result_path, result_format, partition_cols, partition_cols_values)

        # 转化为graph所需的时间形式
        time_agg_df = standardize_time_cols(spark, time_agg_df, partition_cols, tgt_time_cols_values, 
                                            graph.graph_time_cols_alias, query_config["graph_time_values"])
        
        return time_agg_df
        
    # 以未完成的时间来进行time_agg
    logger.info(f"Missing target partitions: {missing_values}")

    # 设定time_agg_df中的id列，也就是node_cols加time_cols，用于聚合及合并数据
    time_agg_id_cols = time_agg_config["src_table"]["node_cols"] + partition_cols

    # 依次获得各种agg_func对应的time_aggregation结果
    time_agg_df_list = []
    for time_agg_index in range(len(time_agg_config["time_aggs"])):
        # 获得对应的配置信息
        time_agg = time_agg_config["time_aggs"][time_agg_index]
        
        logger.info(f"Process the {time_agg_index}-th time aggregation config with funcs {time_agg['agg_funcs']}.")

        # 获得针对该time_range，各个目标时间需要获得的src时间，以及不重复的src时间
        src_time_cols_values, agg_time_map_list = get_agg_time_mapping(missing_values, 
                                                                       src_table_config["time_cols_formats"], 
                                                                       time_agg["time_ranges"])

        # query所需时间的source表
        src_query_config = {"time_cols_values": src_time_cols_values}
        src_df = src_table_query(spark, time_agg_config["src_table"], src_query_config)

        # 修正时间列名为time_agg来源时间列
        renew_cols = [col(c).alias(f"agg_src_{c}") if c in partition_cols else col(c)
                      for c in src_df.columns]
        src_df = src_df.select(*renew_cols)

        # logger.info(f"Time columns mapping list: {agg_time_map_list}")
        
        # 加入各个时间点的数据要聚合到哪个时间点 
        src_time_cols = [f"agg_src_{col}" for col in partition_cols]
        agg_time_map_df = spark.createDataFrame(agg_time_map_list, 
                                                schema = src_time_cols + partition_cols + ["time_range_name"])
        
        # 为source添加上对应的聚合的目标时间以及聚合到的组,作为要用于聚合的表 
        time_agg_df = src_df.join(broadcast(agg_time_map_df), on = src_time_cols, how = "left")
        
        # 检查是否有聚合操作的相关配置
        if len(time_agg["agg_configs"]) > 0:
            # 设定聚合对应的pivot_config
            pivot_config = {}
            pivot_config["column"] = "time_range_name"
            pivot_config["values"] = [time_range["name"] for time_range in time_agg["time_ranges"]]
            
            # 如果有相关的聚合要求则基于聚合的配置进行聚合 
            time_agg_df = pyspark_aggregate(time_agg_df, time_agg_id_cols, time_agg["agg_configs"], pivot_config)
        else:
            # 如果没有，只要对目标列去个重就行
            time_agg_df = time_agg_df.select(time_agg_id_cols).dropDuplicates()
        
        # 加入该time_range是否有对应值的not_null标志
        # 先不加，以后优化，得配合time_agg的新配置方式(也就是按time_range来分agg的配置)

        # 记录最终结果
        time_agg_df_list.append(time_agg_df)

    # 合并各个time_agg的结果 
    time_agg_df = time_agg_df_list[0]
    for add_df in time_agg_df_list[1:]:
        # 合并聚合后的结果
        time_agg_df = time_agg_df.join(add_df, on = time_agg_id_cols, how = "outer")

    # 为所有特征列补0
    time_agg_df = time_agg_df.fillna(0)

    # 向量化全部特征列
    assembler = VectorAssembler(
                    inputCols=time_agg_config["time_aggs_feat_cols"],
                    outputCol=time_agg_config["time_aggs_feat_vec"],
                    handleInvalid="keep"
                )
    time_agg_df = assembler.transform(time_agg_df).drop(*time_agg_config["time_aggs_feat_cols"])

    # 保存最终结果
    col_sizes = {time_agg_config["time_aggs_feat_vec"]: 12 * len(time_agg_config["time_aggs_feat_cols"]) / 2}
    pyspark_optimal_save(time_agg_df, result_path, result_format, "overwrite",
                         partition_cols, missing_values, col_sizes = col_sizes)

    # 读取最终的全量结果
    time_agg_df = pyspark_read_table(spark, result_path, result_format, partition_cols, partition_cols_values)

    # 转化为graph所需的格式
    time_agg_df = standardize_time_cols(spark, time_agg_df, partition_cols, tgt_time_cols_values, 
                                        graph.graph_time_cols_alias, query_config["graph_time_values"])

    return time_agg_df