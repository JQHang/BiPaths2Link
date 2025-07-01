from ..hdfs import hdfs_check_partitions
from ..pyspark import pyspark_read_table
from ..python import time_costing

import logging
from functools import reduce
from pyspark.sql.functions import broadcast, col, when

# 获得logger
logger = logging.getLogger(__name__)

@time_costing
def src_table_query(spark, src_table_config, query_config):
    # 获得source表对应的路径
    source_path = src_table_config["source_path"]

    # 获得结果对应的存储格式
    source_format = src_table_config["source_format"]
    
    # 获得结果对应的分区列
    partition_cols = src_table_config["time_cols"]

    # 获得目标结果对应的分区值
    partition_cols_values = query_config["time_cols_values"]

    # 获得要读取的列和读取出的别名
    source_col_aliases = src_table_config["col_aliases"]
    
    logger.info(f"Query source table from: {source_path} in {source_format} format "
                f"with partition cols {partition_cols} and values {partition_cols_values}.")

    # 检查该路径下是否已有对应的source table(返回是否有全量结果以及没有全量结果的话缺失哪些分区的结果)
    is_complete, missing_values = hdfs_check_partitions(source_path, partition_cols, partition_cols_values)
    
    # 如果没有全部目标分区的source table则报错
    assert is_complete, f"Missing required partitions {missing_values} from source table {source_path}"

    # 读取source表中对应的数据
    src_df = pyspark_read_table(spark, source_path, source_format, partition_cols, partition_cols_values, 
                                source_col_aliases)

    # 保证节点列里没有无意义的id值
    # 下面就是去重了，对单个nodes组合的影响不大，但之后算join_edges时会引起过多的邻居点
    # 等之后join_edges时完善了加盐操作，就可以只考虑null值
    filtered_node_cols = []
    for node_col in src_table_config["node_cols"]:
        filtered_node_cols.append(col(node_col).isNotNull())
        filtered_node_cols.append(col(node_col) != "")
        filtered_node_cols.append(col(node_col) != "NaN")
        # 专门为site_road_id的情况添加，待优化
        filtered_node_cols.append(col(node_col) != "___")
    combined_condition = reduce(lambda x, y: x & y, filtered_node_cols)
    src_df = src_df.filter(combined_condition)

    # 删去重复节点列的数据
    # 之后可以优化为nodes aggregation，这里开销很大
    src_df = src_df.dropDuplicates(src_table_config["node_cols"] + src_table_config["time_cols"])

    # 进行对全部列加入null值标注
    not_null_mark_prefix = src_table_config['null_mark_prefix']
    not_null_mark_cols = [when(col(c).isNull(), 0).otherwise(1).alias(f"{not_null_mark_prefix}_{c}") 
                          for c in src_table_config["feat_cols"]]
    src_df = src_df.select("*", *not_null_mark_cols)

    # 将所有null值都变为0
    src_df = src_df.fillna(0)

    return src_df