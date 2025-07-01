from .join_edges_query import join_edges_query
from .graph_token_query import graph_token_query
from ..python import time_costing
from ..hdfs import hdfs_check_partitions
from ..pyspark import pyspark_read_table, pyspark_optimal_save, fill_null_vectors

import copy
import logging
from pyspark.sql.functions import when, col, lit, rand, row_number
from pyspark.sql.functions import first as first_, max as max_
from pyspark.sql.window import Window

# 获得logger
logger = logging.getLogger(__name__)

@time_costing
def join_edges_list_collection(spark, graph, join_edges_list, query_config):
    # 获得该join_edges的基础信息
    query_nodes_types = join_edges_list["query_nodes_types"]
    query_nodes_indexes = join_edges_list["query_nodes_indexes"]
    query_nodes_cols = join_edges_list["query_nodes_cols"]
    query_nodes_cols_alias = join_edges_list["query_nodes_cols_alias"]
    query_nodes_join_cols = join_edges_list["query_nodes_join_cols"]
    query_nodes_feat_cols = join_edges_list["query_nodes_feat_cols"]
    join_edges_list_name = join_edges_list["join_edges_list_name"]
    
    logger.info(f"Collect instances of join_edges_list {join_edges_list_name} for query nodes types "
                f"{query_nodes_types} of indexes {query_nodes_indexes} and cols {query_nodes_cols}")

    # 获得query nodes对应的id列
    query_nodes_id_cols = query_nodes_cols + list(graph.graph_time_cols_alias)

    # 获得目标结果的存储路径
    # 如果有设定的具体目标点就用具体目标点对应的存储路径，没有就用全量数据存储路径
    if "tgt_query_nodes" in query_config:
        # 若有目标节点，则设定为目标节点结果目录下对应路径文件夹
        result_path = query_config["tgt_query_nodes"]["result_path"] + f"/join_edges_list/{join_edges_list_name}"
    else:
        # 如果没有目标节点，则直接设为全量结果的存储路径
        result_path = join_edges_list['join_edges_list_path']
        
    # 获得结果对应的存储格式
    result_format = join_edges_list["join_edges_list_table_format"]

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
        clt_join_edges_list_df = pyspark_read_table(spark, result_path, result_format, partition_cols, partition_cols_values)
        
        return clt_join_edges_list_df

    # 如果是取部分数据，可以额外检查下是否有对应的全量结果，内积一下就行,目前的写法也不完善，以后优化
    if "tgt_query_nodes" in query_config:
        # 检查全量结果是否存在
        is_complete, _ = hdfs_check_partitions(join_edges_list['join_edges_list_path'], partition_cols, missing_values)
        
        # 如果已有对应的全量结果，则直接读取对应的结果并返回
        if is_complete:
            # 读取对应结果
            clt_join_edges_list_df = pyspark_read_table(spark, join_edges_list['join_edges_list_path'], result_format, 
                                                        partition_cols, missing_values)

            # 获得具体的query nodes
            query_nodes_df = query_config["tgt_query_nodes"]["df"]
            
            # 只保留query_config对应的结果
            clt_join_edges_list_df = clt_join_edges_list_df.join(query_nodes_df, on = query_nodes_df.columns, how = "inner")
            
            # 保存结果
            pyspark_optimal_save(clt_join_edges_list_df, result_path, result_format, "overwrite", partition_cols)

            # 重新读取完整结果
            clt_join_edges_list_df = pyspark_read_table(spark, result_path, result_format, partition_cols, partition_cols_values)
            
            return clt_join_edges_list_df

    # 如果要开始计算，先以未完成的时间更新query_config
    logger.info(f"Missing target partitions: {missing_values}")
    missing_query_config = {}
    missing_query_config["graph_time_values"] = missing_values
    if "tgt_query_nodes" in query_config:
        missing_query_config["tgt_query_nodes"] = {}
        missing_query_config["tgt_query_nodes"]["result_path"] = query_config["tgt_query_nodes"]["result_path"]
        missing_query_config["tgt_query_nodes"]["df"] = query_config["tgt_query_nodes"]["df"]

    # 记录collect结果中的特征列及对应的向量维度
    clt_feat_cols_sizes = {}

    # 依次获得各个join_edges的结果
    join_edges_df_list = []
    for join_edges in join_edges_list["join_edges_list_schema"]:
        # 检测是否有对应的collection结果
        
        # 获得query_nodes_df对应的该join_edges的数据
        join_edges_df = join_edges_query(spark, graph, join_edges, missing_query_config)

        # 获得该join_edges最多的结果数
        collect_records_count = join_edges["collect_records_count"]
        
        # 为相同的query node的数据加上join_edge来源和在该来源中的序号
        # 并只保留最大的结果数以内的序号对应的数据
        window_spec = Window.partitionBy(*query_nodes_id_cols).orderBy(rand())
        join_edges_df = join_edges_df.withColumn("collect_id", row_number().over(window_spec)) \
                                     .filter(col("collect_id") <= collect_records_count)

        # 获得join_edges名称
        join_edges_name = join_edges['name']

        # 获得为collect各组数据所需的聚合配置
        agg_exprs = []

        # 先记录各个query_nodes collect的数目
        agg_expr = max_("collect_id").alias(f"{join_edges_name}_collect_count")
        agg_exprs.append(agg_expr)

        # 再依次记录各个collect的实例对应的结果
        for collect_id in range(collect_records_count):
            for collect_col in join_edges_df.columns:
                if collect_col in query_nodes_id_cols:
                    continue
                if collect_col == "collect_id":
                    continue
                    
                condition = when(col("collect_id") == (collect_id + 1), col(collect_col))
                new_col_name = f"{join_edges_name}_id_{collect_id}_{collect_col}"
                agg_expr = first_(condition).alias(new_col_name)

                agg_exprs.append(agg_expr)

                # 如果是特征列，记录新的列名和对应的向量维度
                if collect_col in join_edges["query_config"]["feat_cols_sizes"]:
                    clt_feat_cols_sizes[new_col_name] = join_edges["query_config"]["feat_cols_sizes"][collect_col]
                
        # 将相同query node的数据collect到一行，并修正列名
        join_edges_df = join_edges_df.groupBy(query_nodes_id_cols).agg(*agg_exprs)
        
        join_edges_df_list.append(join_edges_df)

    # collect同query_nodes的全部join_edges的相关信息
    clt_join_edges_list_df = join_edges_df_list[0]
    for join_edges_df in join_edges_df_list[1:]:
        clt_join_edges_list_df = clt_join_edges_list_df.join(join_edges_df, on = query_nodes_id_cols,
                                                             how = "outer")

    # 设定要读取的token的query配置
    token_query_config = {}
    token_query_config["graph_time_values"] = missing_values

    # 补全query_nodes对应的特征信息
    for query_node_i in range(len(query_nodes_types)):
        query_node_type = query_nodes_types[query_node_i]
        logger.info(f"Add features for query node type {query_node_type}")
                    
        # 获得该node对应的数据
        node_df = graph_token_query(spark, graph, "node", query_node_type, token_query_config)

        # 要使用的列及别名
        select_cols = [col(column).alias(alias) for column, alias in query_nodes_cols_alias[query_node_i]]
        
        # 修正列名
        node_df = node_df.select(*select_cols)

        # 设定用于join的列
        query_node_join_cols = query_nodes_join_cols[query_node_i]
        
        # 将特征join到join_edges上
        clt_join_edges_list_df = clt_join_edges_list_df.join(node_df, on = query_node_join_cols, how = "left")

        # 记录新增的特征列列名
        query_node_feat_col = query_nodes_feat_cols[query_node_i]
        
        # 记录新增的特征列和对应的向量长度
        clt_feat_cols_sizes[query_node_feat_col] = len(graph.nodes[query_node_type]["graph_token_feat_cols"])

        # 补全空向量 
        clt_join_edges_list_df = fill_null_vectors(spark, clt_join_edges_list_df, query_node_feat_col, 
                                                   len(graph.nodes[query_node_type]["graph_token_feat_cols"]))
    
    # 保存结果
    pyspark_optimal_save(clt_join_edges_list_df, result_path, result_format, "overwrite", partition_cols,
                         col_sizes = clt_feat_cols_sizes)
    
    # 重新读取完整结果
    clt_join_edges_list_df = pyspark_read_table(spark, result_path, result_format, partition_cols, partition_cols_values)

    return clt_join_edges_list_df