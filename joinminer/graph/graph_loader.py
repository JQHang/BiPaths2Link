from ..python import setup_logger, time_costing, ensure_logger
from ..pyspark import identify_numeric_columns, rename_columns
from ..hdfs import hdfs_delete_dir, hdfs_check_file_exists, hdfs_list_contents
from ..pyspark import random_n_sample, top_n_sample, threshold_n_sample
from ..pyspark import pyspark_optimal_save, pyspark_aggregate

import copy
import logging
from datetime import datetime
from dateutil.relativedelta import relativedelta
from functools import reduce
from pyspark.sql.functions import broadcast, col, lit

# 获得logger
logger = logging.getLogger(__name__)

# 获得表格在某个时间点对应的数据所属的分区的位置
def get_table_time_partition_path(graph, table_path, time_cols_value):
    # 获得图中对应的时间列
    graph_time_cols = graph.graph_time_cols
    
    table_time_partition_path = table_path
    for time_col_i in range(len(graph_time_cols)):
        time_col = graph_time_cols[time_col_i]
        time_col_value = time_cols_value[time_col_i]
        table_time_partition_path = table_time_partition_path + f"/{time_col}={time_col_value}"
                
    return table_time_partition_path

def check_existing_table(spark, graph, check_table_path, time_cols_values = None):
    # 检查是否有之前的_temporary文件夹，有就清空
    if hdfs_check_file_exists(check_table_path + f"/_temporary"):
        hdfs_delete_dir(check_table_path + f"/_temporary")
    
    check_result_dict = {}
    
    # 查看graph是否有时间列
    if hasattr(graph, 'graph_time_cols'):
        if time_cols_values is None:
            time_cols_values = graph.graph_time_cols_values

        # 依次查看各个分区是否有对应结果
        exist_time_cols_values = []
        miss_time_cols_values = []
        for time_cols_value in time_cols_values:
            # 获得对应的存储位置
            partition_dir = get_table_time_partition_path(graph, check_table_path, time_cols_value)
            
            if hdfs_check_file_exists(partition_dir + f"/_SUCCESS"):
                exist_time_cols_values.append(time_cols_value)
            else:
                miss_time_cols_values.append(time_cols_value)
                
                # 如果有对应文件夹但没有SUCCESS标志，则清空对应文件夹
                if hdfs_check_file_exists(partition_dir):
                    hdfs_delete_dir(partition_dir)

        # 都有结果则直接返回
        if len(miss_time_cols_values) == 0:
            # 获得全部目标路径
            check_table_paths = []
            for time_cols_value in exist_time_cols_values:
                partition_dir = get_table_time_partition_path(graph, check_table_path, time_cols_value)
                check_table_paths.append(partition_dir)
            
            exist_table_df = spark.read.option("basePath", check_table_path).parquet(*check_table_paths)
            
            check_result_dict["data"] = exist_table_df
        
        check_result_dict["exist_time_cols_values"] = exist_time_cols_values
        check_result_dict["miss_time_cols_values"] = miss_time_cols_values
    else:
        if hdfs_check_file_exists(check_table_path + "/_SUCCESS"):
            exist_table_df = spark.read.parquet(check_table_path)

            check_result_dict["data"] = exist_table_df

    return check_result_dict

def read_edge_table(spark, graph, edge_table_schema, time_cols_values = None, logger = setup_logger()):
    """
    读取关系表，并过滤出有效边
    """
    edge_type = edge_table_schema['edge_type']
    edge_table_name = edge_table_schema['edge_table_name']
    
    # 获取该表对应的路径
    edge_table_path = graph.edges[edge_type]["edge_tables"][edge_table_name]["table_path"]

    # 获取该表对应的时间列
    
    logger.info(f'Read edge table: {edge_table_name} from directory: {edge_table_path}')
    
    # 读取对应数据
    edge_table_df = spark.read.parquet(edge_table_path)
    
    # 构建 select 语句
    selected_columns = []
    for edge_node_col in edge_table_schema["edge_node_cols"]:
        selected_columns.append(col(edge_node_col[0]).alias(edge_node_col[1]))
    for edge_feat_col in edge_table_schema["edge_feat_cols"]:
        selected_columns.append(col(edge_feat_col[0]).alias(edge_feat_col[1]))
    for edge_time_col in edge_table_schema["edge_time_cols"]:
        selected_columns.append(col(edge_time_col[0]).alias(edge_time_col[1]))
        
    # 选择并重命名列
    edge_table_df = edge_table_df.select(*selected_columns)
    
    # 如果有目标时间，则根据目标时间过滤数据
    if hasattr(graph, 'graph_time_cols'):
        if time_cols_values is None:
            time_cols_values = graph.graph_time_cols_values
        
        # 只取目标时间的数据
        filtered_time_cols_values = []
        for time_cols_value in time_cols_values:
            filtered_time_cols_value = []
            for time_col_i in range(len(edge_table_schema["edge_time_cols"])):
                time_col = edge_table_schema["edge_time_cols"][time_col_i][1]
                time_col_value = time_cols_value[time_col_i]
                
                filtered_time_cols_value.append(col(time_col) == time_col_value)
            filtered_time_cols_values.append(reduce(lambda x, y: x & y, filtered_time_cols_value))
        combined_condition = reduce(lambda x, y: x | y, filtered_time_cols_values)
        edge_table_df = edge_table_df.filter(combined_condition)
    
    # 保证节点列都没有null值
    filtered_node_cols = []
    for edge_node_col in edge_table_schema["edge_node_cols"]:
        filtered_node_cols.append(col(edge_node_col[1]).isNotNull())
        filtered_node_cols.append(col(edge_node_col[1]) != "")
        filtered_node_cols.append(col(edge_node_col[1]) != "NaN")
        # 专门为site_road_id的情况添加，待优化
        filtered_node_cols.append(col(edge_node_col[1]) != "___")
    combined_condition = reduce(lambda x, y: x & y, filtered_node_cols)
    edge_table_df = edge_table_df.filter(combined_condition)
    
    # 获得其中包含的id列
    edge_id_cols = []
    for edge_node_col in edge_table_schema["edge_node_cols"]:
        edge_id_cols.append(edge_node_col[1])
    for edge_time_col in edge_table_schema["edge_time_cols"]:
        edge_id_cols.append(edge_time_col[1])
    
    # Drop duplicate edge with same id columns(之后看需求可以考虑aggregation)
    edge_table_df = edge_table_df.dropDuplicates(edge_id_cols)
    
    # Edge limit
    if 'edge_limit' in edge_table_schema and edge_table_schema['edge_limit'] != '':
        logger.info(f"Edge Limitation: {edge_table_schema['edge_limit']}")
        edge_table_df = edge_table_df.filter(edge_table_schema['edge_limit'])
    
    # Edge Sample
    if 'edge_samples' in edge_table_schema:
        for edge_sample in edge_table_schema['edge_samples']:
            groupby_nodes_cols = edge_sample['groupby_nodes_cols']
            edge_sample_type = edge_sample['sample_type']
            edge_sample_count = edge_sample['sample_count']

            groupby_id_cols = list(groupby_nodes_cols)
            for edge_time_col in edge_table_schema["edge_time_cols"]:
                groupby_id_cols.append(edge_time_col[1])

            logger.info(f'Edge Sampling: {edge_sample_type}, {groupby_id_cols}, {edge_sample_count}')

            if edge_sample_type == 'random':
                edge_table_df = random_n_sample(spark, edge_table_df, groupby_id_cols, edge_sample_count)
            elif edge_sample_type == 'threshold':
                edge_table_df = threshold_n_sample(spark, edge_table_df, groupby_id_cols, edge_sample_count)
    
    return edge_table_df

def read_node_table(spark, graph, node_table_schema, time_cols_values = None, logger = setup_logger()):
    """
    读取节点表中的目标信息
    """
    node_type = node_table_schema['node_type']
    node_table_name = node_table_schema['node_table_name']
    
    # 获取该表对应的路径
    node_table_path = graph.nodes[node_type]["node_tables"][node_table_name]["table_path"]

    logger.info(f'Read node table: {node_table_name} from directory: {node_table_path}')

    # 读取数据
    node_table_df = spark.read.parquet(node_table_path)
    
    # 构建 select 语句
    selected_columns = []
    for node_column in node_table_schema["node_columns"]:
        selected_columns.append(col(node_column[0]).alias(node_column[1]))
    for node_feat_col in node_table_schema["node_feat_cols"]:
        selected_columns.append(col(node_feat_col[0]).alias(node_feat_col[1]))
    for node_time_col in node_table_schema["node_time_cols"]:
        selected_columns.append(col(node_time_col[0]).alias(node_time_col[1]))
    
    # 选择并重命名列
    node_table_df = node_table_df.select(*selected_columns)
    
    # 如果有目标时间，则根据目标时间过滤数据
    if hasattr(graph, 'graph_time_cols'):
        if time_cols_values is None:
            time_cols_values = graph.graph_time_cols_values
            
        # 只取目标时间的数据
        filtered_time_cols_values = []
        for time_cols_value in time_cols_values:
            filtered_time_cols_value = []
            for time_col_i in range(len(node_table_schema["node_time_cols"])):
                time_col = node_table_schema["node_time_cols"][time_col_i][1]
                time_col_value = time_cols_value[time_col_i]
                
                filtered_time_cols_value.append(col(time_col) == time_col_value)
            filtered_time_cols_values.append(reduce(lambda x, y: x & y, filtered_time_cols_value))
        combined_condition = reduce(lambda x, y: x | y, filtered_time_cols_values)
        node_table_df = node_table_df.filter(combined_condition)
    
    # 保证节点列都没有null值
    filtered_node_cols = []
    for node_column in node_table_schema["node_columns"]:
        filtered_node_cols.append(col(node_column[1]).isNotNull())
        filtered_node_cols.append(col(node_column[1]) != "")
        filtered_node_cols.append(col(node_column[1]) != "NaN")
        # 专门为site_road_id的情况添加，待优化
        filtered_node_cols.append(col(node_column[1]) != "___")
    combined_condition = reduce(lambda x, y: x & y, filtered_node_cols)
    node_table_df = node_table_df.filter(combined_condition)
    
    # 获得对应的id列
    node_id_cols = []
    for node_column in node_table_schema["node_columns"]:
        node_id_cols.append(node_column[1])
    for node_time_col in node_table_schema["node_time_cols"]:
        node_id_cols.append(node_time_col[1])
    
    # Drop duplicate node with same id columns(之后看需求可以考虑aggregation)
    node_table_df = node_table_df.dropDuplicates(node_id_cols)
    
    # Node limit
    if 'node_limit' in node_table_schema and node_table_schema['node_limit'] != '':
        logger.info(f"Node Limitation: {node_table_schema['node_limit']}")
        node_table_df = node_table_df.filter(node_table_schema['node_limit'])
    
    return node_table_df

def read_label_table(spark, graph, label_table_schema, logger = setup_logger()):
    """
    读取标签表中的目标信息
    """
    # 获取该表对应的路径
    label_table_path = label_table_schema["table_path"]
    
    # 获得该表对应的文件类型
    label_table_format = "parquet"
    if "table_format" in label_table_schema:
        label_table_format = label_table_schema["table_format"]
    
    logger.info(f'Read label table from directory: {label_table_path}')
    
    # 读取数据
    label_table_df = spark.read.format(label_table_format).load(label_table_path)
    
    # 全部列的类型
    col_types = ["node_cols_to_aliases", "feat_cols_to_aliases", "label_cols_to_aliases", "time_cols_to_aliases"]
    
    # 构建 select 语句
    selected_columns = []
    for col_type in col_types:
        for col_to_alias in label_table_schema[col_type]:
            selected_columns.append(col(col_to_alias[0]).alias(col_to_alias[1]))
        
    # 选择并重命名列
    label_table_df = label_table_df.select(*selected_columns)
    
    # 保证节点列都没有null值
    filtered_node_cols = []
    for node_col_to_alias in label_table_schema["node_cols_to_aliases"]:
        filtered_node_cols.append(col(node_col_to_alias[1]).isNotNull())
        filtered_node_cols.append(col(node_col_to_alias[1]) != "")
        filtered_node_cols.append(col(node_col_to_alias[1]) != "NaN")
    combined_condition = reduce(lambda x, y: x & y, filtered_node_cols)
    label_table_df = label_table_df.filter(combined_condition)
    
    # Drop duplicate node with same id columns(之后看需求可以考虑aggregation)(补充：目前思路是在time_aggregation阶段完成)
    sample_id_cols = []
    for col_type in ["node_cols_to_aliases", "time_cols_to_aliases"]:
        for col_to_alias in label_table_schema[col_type]:
            sample_id_cols.append(col_to_alias[1])
    label_table_df = label_table_df.dropDuplicates(sample_id_cols)
    
    # Sample limit
    if 'sample_limit' in label_table_schema and label_table_schema['sample_limit'] != '':
        logger.info(f"Sample Limitation: {label_table_schema['sample_limit']}")
        label_table_df = label_table_df.filter(label_table_schema['sample_limit'])
    
    # Sample repartition
    if 'sample_repartition' in label_table_schema:
        if label_table_schema['sample_repartition']['func'] == "coalesce":
            label_table_df = label_table_df.coalesce(label_table_schema['sample_repartition']['partition_num'])
        else:
            label_table_df = label_table_df.repartition(label_table_schema['sample_repartition']['partition_num'])
            
    return label_table_df

# 获得表格在某个时间点对应的数据所属的分区的位置
def get_graph_table_time_partition_path(graph_table_path, table_time_cols, time_cols_value):
    table_time_partition_path = graph_table_path
    for time_col_i in range(len(table_time_cols)):
        time_col = table_time_cols[time_col_i]
        time_col_value = time_cols_value[time_col_i]
        table_time_partition_path = table_time_partition_path + f"/{time_col}={time_col_value}"
                
    return table_time_partition_path

def check_existing_graph_table(spark, graph_table_path, table_time_cols = [], time_cols_values = []):
    # 用一个字典保留结果
    check_result_dict = {}
    
    # 查看该表是否有时间列
    if len(table_time_cols) > 0:
        # 如果该表有时间列就必须有对应的值，否则报错
        if len(time_cols_values) == 0:
            raise ValueError(f"Require time_cols_values for time_cols in function check_existing_graph_table.")

        # 依次查看各个分区是否有对应结果
        exist_time_cols_values = []
        miss_time_cols_values = []
        for time_cols_value in time_cols_values:
            # 获得对应的存储位置
            partition_dir = get_graph_table_time_partition_path(graph_table_path, table_time_cols, time_cols_value)
            
            if hdfs_check_file_exists(partition_dir + f"/_SUCCESS"):
                exist_time_cols_values.append(time_cols_value)
            else:
                miss_time_cols_values.append(time_cols_value)
                
                # 如果有对应文件夹但没有SUCCESS标志，则清空对应文件夹
                if hdfs_check_file_exists(partition_dir):
                    hdfs_delete_dir(partition_dir)

        # 都有结果则直接返回
        if len(miss_time_cols_values) == 0:
            # 获得全部目标路径
            graph_table_partitions = []
            for time_cols_value in exist_time_cols_values:
                partition_dir = get_graph_table_time_partition_path(graph_table_path, table_time_cols, time_cols_value)
                graph_table_partitions.append(partition_dir)
            
            exist_table_df = spark.read.option("basePath", graph_table_path).parquet(*graph_table_partitions)
            
            check_result_dict["data"] = exist_table_df
        
        check_result_dict["exist_time_cols_values"] = exist_time_cols_values
        check_result_dict["miss_time_cols_values"] = miss_time_cols_values
    else:
        if hdfs_check_file_exists(graph_table_path + "/_SUCCESS"):
            exist_table_df = spark.read.parquet(graph_table_path)

            check_result_dict["data"] = exist_table_df

    # 最后检查目录下是否有_temporary文件夹，有就清空
    if hdfs_check_file_exists(graph_table_path + f"/_temporary"):
        hdfs_delete_dir(graph_table_path + f"/_temporary")
    
    return check_result_dict

def read_graph_table_partitions(spark, graph_table_schema):
    # 获取该表对应的路径
    table_path = graph_table_schema["table_path"]

    # 获得该表对应的文件类型
    table_format = graph_table_schema["table_format"]
    
    # 获取该表对应的分区列
    partition_cols = graph_table_schema["partition_cols"]

    # 获取该表的分区列可用的数值
    partition_cols_values = graph_table_schema["partition_cols_values"]
    
    # 用一个字典保留结果
    result = {
        "success": True,
        "failed_partitions": []
    }
    
    # 查看该表是否有分区列
    if len(partition_cols) > 0:
        # 如果该表有分区列就必须有对应的值，否则报错
        if len(partition_cols_values) == 0:
            raise ValueError(f"Require partition_cols_values for partition_cols in function read_graph_table_partitions.")

        # 依次查看各个分区是否有对应结果
        for partition_cols_value in partition_cols_values:
            # 获得对应的存储位置
            partition_path = table_path + "/" 
            partition_path += "/".join(f"{col}={val}" for col, val in zip(partition_cols, partition_cols_value))
            
            if not hdfs_check_file_exists(partition_path + f"/_SUCCESS"):
                result["success"] = False
                result["failed_partitions"].append(partition_cols_value)
                
                # 如果有对应文件夹但没有SUCCESS标志，则清空对应文件夹（防止干扰之后的运算）
                if hdfs_check_file_exists(partition_path):
                    hdfs_delete_dir(partition_path)

        # 如果有成功的分区则读取对应的结果
        if len(result["failed_partitions"]) < len(partition_cols_values):
            # 读取现有结果
            result_df = spark.read.format(table_format).load(table_path)
    
            # 只保留目标分区的结果
            filtered_partition_cols_values = []
            for partition_cols_value in partition_cols_values:
                if partition_cols_value in result["failed_partitions"]:
                    continue
                
                filtered_partition_cols_value = []
                for partition_col_i in range(len(partition_cols)):
                    partition_col = partition_cols[partition_col_i]
                    partition_col_value = partition_cols_value[partition_col_i]
                    
                    filtered_partition_cols_value.append(col(partition_col) == partition_col_value)
                filtered_partition_cols_values.append(reduce(lambda x, y: x & y, filtered_partition_cols_value))
            combined_condition = reduce(lambda x, y: x | y, filtered_partition_cols_values)
            result["data"] = result_df.filter(combined_condition)
    else:
        # 检查是否已有对应结果
        if not hdfs_check_file_exists(table_path + f"/_SUCCESS"):
            result["success"] = False

            # 如果有对应文件夹但没有SUCCESS标志，则清空对应文件夹（防止干扰之后的运算）
            if hdfs_check_file_exists(table_path):
                hdfs_delete_dir(table_path)

        # 如果有结果则读取对应结果返回
        if result["success"]:
            result_df = spark.read.format(table_format).load(table_path)

            result["data"] = result_df

    # 最后检查目录下是否有_temporary文件夹，有就清空（防止干扰之后的运算）
    if hdfs_check_file_exists(table_path + f"/_temporary"):
        hdfs_delete_dir(table_path + f"/_temporary")
    
    return result

# 读取source table的函数
def read_source_table(spark, source_table_schema, logger = setup_logger()):
    # 获取该表对应的基础信息
    table_path = source_table_schema["table_path"]
    table_format = source_table_schema["table_format"]
    time_cols_values = source_table_schema["time_cols_values"]
    
    logger.info(f'Read {table_format} format source table at times {time_cols_values} '
                f'from directory {table_path} ')
    
    # 读取数据
    source_table_df = spark.read.format(table_format).load(table_path)

    # 全部列的类型
    col_types = ["time_cols", "node_cols", "feat_cols"]
    
    # 构建 select 语句
    selected_columns = []
    for col_type in col_types:
        selected_columns.extend(source_table_schema[col_type])
        
    # 选择并重命名列
    source_table_df = source_table_df.select(*selected_columns)
        
    # 如果有目标时间，则根据目标时间过滤数据
    if len(source_table_schema["time_cols"]) > 0:
        filtered_time_cols_values = []
        for time_cols_value in time_cols_values:
            filtered_time_cols_value = []
            for time_col_i in range(len(source_table_schema["time_cols"])):
                time_col = source_table_schema["time_cols"][time_col_i]
                time_col_value = time_cols_value[time_col_i]
                
                filtered_time_cols_value.append(col(time_col) == time_col_value)
            filtered_time_cols_values.append(reduce(lambda x, y: x & y, filtered_time_cols_value))
        combined_condition = reduce(lambda x, y: x | y, filtered_time_cols_values)
        source_table_df = source_table_df.filter(combined_condition)

    # 保证节点列都没有null值
    filtered_node_cols = []
    for node_col in source_table_schema["node_cols"]:
        filtered_node_cols.append(col(node_col).isNotNull())
        filtered_node_cols.append(col(node_col) != "")
        filtered_node_cols.append(col(node_col) != "NaN")
        # 专门为site_road_id的情况添加，待优化
        filtered_node_cols.append(col(node_col) != "___")
    combined_condition = reduce(lambda x, y: x & y, filtered_node_cols)
    source_table_df = source_table_df.filter(combined_condition)
    
    # Drop duplicate node with same id columns(之后可以优化为nodes aggregation)
    id_cols = []
    for col_type in ["time_cols", "node_cols"]:
        id_cols.extend(source_table_schema[col_type])
    source_table_df = source_table_df.dropDuplicates(id_cols)
    
    return source_table_df

# 以后可以在source table和time agg table间加入一个nodes_aggregation函数，待优化

# 读取time aggregation表的函数
def read_time_agg_table(spark, graph_table_schema, logger = setup_logger()):
    # 获得该graph table的时间列、目标时间和对应的格式
    graph_table_name = graph_table_schema["name"]
    graph_time_cols = graph_table_schema["graph_time_cols"]
    graph_time_cols_values = graph_table_schema["graph_time_cols_values"]
    graph_time_cols_formats = graph_table_schema["graph_time_cols_formats"]

    logger.info(f"Process time aggregation result for graph table {graph_table_name} at time columns "
                f"values {graph_time_cols_values}.")
    
    # 获得source table的时间列和对应的格式
    src_time_cols = graph_table_schema["src_time_cols"]
    src_time_cols_formats = graph_table_schema["src_time_cols_formats"]

    # 获得graph table目标时间在source table的精度下对应的时间，记录为graph_src_time
    graph_to_src_times = []
    graph_src_time_cols_values = []
    for graph_time_cols_value in graph_time_cols_values:
        # 获得完整的graph时间表示，以及对应的format
        full_time_str = ' '.join(graph_time_cols_value)
        full_time_format = ' '.join(graph_time_cols_formats)

        # 转换为日期类型
        graph_date = datetime.strptime(full_time_str, full_time_format)

        # 获得对应的src_time_cols_value
        graph_src_time_cols_value = []
        for time_col_format in src_time_cols_formats:
            time_col_value = graph_date.strftime(time_col_format)
            graph_src_time_cols_value.append(time_col_value)

        # 记录各个graph_src time到graph time的映射关系
        graph_to_src_times.append(graph_time_cols_value + graph_src_time_cols_value)
        
        # 记录各个目标时间
        if graph_src_time_cols_value not in graph_src_time_cols_values:
            graph_src_time_cols_values.append(graph_src_time_cols_value)

    # 将全部的graph time到对应的graph src time的映射数据转为pyspark表  
    graph_to_src_time_df = spark.createDataFrame(graph_to_src_times, schema = graph_time_cols + src_time_cols)

    # 记录该graph table要求的graph source table全部目标时间
    tgt_src_time_cols_values = copy.deepcopy(graph_src_time_cols_values)

    # 设定从第几个time agg对应的配置开始处理
    time_agg_start_index = 0

    # 检查是否有预设的聚合后的结果存储表
    if "time_agg_table_path" in graph_table_schema:
        # 获得预设的结果的路径信息
        time_agg_table_path = graph_table_schema["time_agg_table_path"]
        time_agg_table_format = graph_table_schema["time_agg_table_format"]
        time_agg_save_interval = graph_table_schema["time_agg_save_interval"]
        
        logger.info(f"The time aggregation result will be output to: {time_agg_table_path}")
        
        # 设定time aggregation结果表的相关schema
        time_agg_table_schema = {}
        time_agg_table_schema["table_path"] = time_agg_table_path
        time_agg_table_schema["table_format"] = time_agg_table_format
        time_agg_table_schema["partition_cols"] = list(src_time_cols)
        time_agg_table_schema["partition_cols_values"] = list(graph_src_time_cols_values)
        
        # 尝试读取对应数据
        read_result = read_graph_table_partitions(spark, time_agg_table_schema)
    
        # 检查是否读取到全部的结果
        if read_result["success"]:
            logger.info(f"The reulst already exist.")

            # 添加对应的graph time列
            full_time_agg_df = read_result["data"].join(broadcast(graph_to_src_time_df), on = src_time_cols, 
                                                        how = "left")

            # 删去对应的src_time_cols
            full_time_agg_df = full_time_agg_df.drop(*src_time_cols)
            
            # 如果已有全部结果，则直接返回
            return full_time_agg_df

        # 否则若图有时间列则获得未完成的时间点，并据此更新设定的目标时间
        tgt_src_time_cols_values = list(read_result["failed_partitions"])
        
        # 设定中间结果的存储文件夹
        intermediate_table_path = time_agg_table_path + "_intermediate"

        logger.info(f"The intermediate result will be output to: {intermediate_table_path}")

        # 检查是否有已存在的中间结果
        if hdfs_check_file_exists(intermediate_table_path):
            # 获得其中的全部文件夹
            intermediate_index_paths = hdfs_list_contents(intermediate_table_path, "directories")

            # 按index序号由低到高排序
            intermediate_index_paths = sorted(intermediate_index_paths, key = lambda x: int(x.split("/")[-1]))
            
            # 由低到高检查各个index对应的结果，来查找有已存在结果的最大的index
            for index_path in intermediate_index_paths:
                # 设定中间结果表的相关schema
                intermediate_index_table_schema = {}
                intermediate_index_table_schema["table_path"] = index_path
                intermediate_index_table_schema["table_format"] = time_agg_table_format
                intermediate_index_table_schema["partition_cols"] = list(src_time_cols)
                intermediate_index_table_schema["partition_cols_values"] = list(tgt_src_time_cols_values)
                
                # 尝试读取对应数据
                read_result = read_graph_table_partitions(spark, intermediate_index_table_schema)
            
                # 检查是否读取到全部的结果
                if read_result["success"]:
                    # 获得对应的index序号
                    success_index = int(index_path.split('/')[-1])
                    logger.info(f"Index {success_index} has existing results.")
                    
                    # 更新开始计算time_agg的index序号 
                    time_agg_start_index = success_index + 1

                    # 记录该index对应的结果 
                    full_time_agg_df = read_result["data"]
                    
    else:
        # 没有预设结果路径就把对应的路径变量置空
        time_agg_table_path = None

        logger.info(f"The time aggregation result won't be stored")

    logger.info(f"Start processing time aggregation from {time_agg_start_index}-th config.")

    # 记录已处理的time_agg的聚合结果
    full_time_agg_df = None

    # 设定各个time_agg结果对应的id列
    time_agg_id_cols = src_time_cols + graph_table_schema["src_node_cols"]

    # 获得要用于聚合的来源时间列名(原始列名加上agg_前缀)
    agg_time_cols = ["agg_" + x for x in src_time_cols]

    # 记录要用到的全部聚合时间种类
    agg_time_cols_values = []

    # 记录各个agg_time的数据要聚合到哪个时间点的src time的哪组time_agg结果中
    agg_to_tgt_group = []

    # 依次处理各个time_agg配置对应的结果
    for time_agg_index in range(time_agg_start_index, len(graph_table_schema["time_aggs"])):
        logger.info(f"Process the {time_agg_index}-th time aggregation config.")
        
        # 获得对应的配置信息
        time_agg = graph_table_schema["time_aggs"][time_agg_index]

        # 记录用到的全部time_range名称
        time_range_names = []
        
        # 依次处理该种配置中包含的全部time_range
        for time_range in time_agg["time_ranges"]:
            # 获得该time_range的名称并记录
            time_range_name = time_range["name"]
            time_range_names.append(time_range_name)
            
            # 先针对每个目标时间，获得在该time_range中对应的全部要聚合的agg时间
            for src_time_cols_value in tgt_src_time_cols_values:
                # 获得目标时间对应的datetime格式的时间
                full_time_str = ' '.join(src_time_cols_value)
                full_time_format = ' '.join(src_time_cols_formats)
                target_date = datetime.strptime(full_time_str, full_time_format)
    
                # 依次获得每个目标时间点对应的具体时间
                for time_point in time_range["time_points"]:
                    # 查看目标时间点的表示格式
                    if time_point["time_type"] == "relative":
                        # 基于配置信息获得对应的目标时间点
                        time_unit = time_point["time_unit"]
                        time_interval = time_point["time_interval"]
    
                        if time_unit == "day":
                            agg_date = target_date + relativedelta(days = time_interval)
                        elif time_unit == "month":
                            agg_date = target_date + relativedelta(months = time_interval)
                        elif time_unit == "year":
                            agg_date = target_date + relativedelta(years = time_interval)
    
                        # 获得对应的日期转化为时间列后的形式
                        agg_time_cols_value = []
                        for time_col_format in src_time_cols_formats:
                            time_col_value = agg_date.strftime(time_col_format)
                            agg_time_cols_value.append(time_col_value)
    
                        # 记录这个聚合时间
                        if agg_time_cols_value not in agg_time_cols_values:
                            agg_time_cols_values.append(agg_time_cols_value)
    
                        # 记录该agg_time的数据要聚合到哪个时间点的src time的哪组time_agg结果中 
                        agg_to_tgt_group.append(agg_time_cols_value + src_time_cols_value + [time_range_name])
        
        # 设定要使用的source表的相关schema
        source_table_schema = {}
        source_table_schema["table_path"] = graph_table_schema["src_table_path"]
        source_table_schema["table_format"] = graph_table_schema["src_table_format"]
        source_table_schema["node_cols"] = list(graph_table_schema["src_node_cols"])
        source_table_schema["feat_cols"] = list(graph_table_schema["src_feat_cols"])
        source_table_schema["time_cols"] = list(src_time_cols)
        source_table_schema["time_cols_values"] = list(agg_time_cols_values)

        logger.info(f"Getting features from {len(agg_time_cols_values)} times: {agg_time_cols_values}.")
        
        # 读取全部要聚合的时间再source table内对应的数据   
        source_table_df = read_source_table(spark, source_table_schema, logger = logger)

        # 将时间列改名为聚合时间列
        for time_col in src_time_cols:
            source_table_df = source_table_df.withColumnRenamed(time_col, "agg_" + time_col)

        # 将全部的agg_date到对应的target_date的映射数据转为pyspark表   
        agg_to_tgt_group_df = spark.createDataFrame(agg_to_tgt_group, 
                                                    schema = agg_time_cols + src_time_cols + ["time_range_name"])

        # 为source添加上对应的聚合的目标时间以及聚合到的组,作为要用于聚合的表 
        src_agg_table_df = source_table_df.join(broadcast(agg_to_tgt_group_df), on = agg_time_cols, how = "left")

        # 检查是否有相关的聚合要求
        if len(time_agg["agg_configs"]) > 0:
            # 设定聚合对应的pivot_config
            pivot_config = {}
            pivot_config["column"] = "time_range_name"
            pivot_config["values"] = time_range_names
            
            # 如果有相关的聚合要求则基于聚合的配置进行聚合 
            time_agg_df = pyspark_aggregate(src_agg_table_df, time_agg_id_cols, time_agg["agg_configs"], pivot_config)
        else:
            # 有时候没有特征时不需要聚合，只要对目标列去个重就行
            time_agg_df = src_agg_table_df.select(time_agg_id_cols).dropDuplicates()
            
        # 和之前的结果合并
        if time_agg_index == 0:
            # 如果是第一个配置，则只用记录结果
            full_time_agg_df = time_agg_df
        else:
            # 合并聚合后的结果(*Outer join，但null值怎么处理，待优化)
            full_time_agg_df = full_time_agg_df.join(time_agg_df, on = time_agg_id_cols, how = "outer")
        
            # 检查是否已经达到要保存中间结果的数目 
            if time_agg_table_path is not None and (time_agg_index + 1) % time_agg_save_interval == 0:
                # 设定结果保存的路径
                index_path = intermediate_table_path + f'/{time_agg_index}'
                
                # 以最优分区配置保存结果到对应的表格 
                pyspark_optimal_save(full_time_agg_df, index_path, time_agg_table_format, "append",
                                     src_time_cols, logger = logger)

                logger.info(f"Save intermediate result at time aggregation index: {time_agg_index}")

                # 重新读取出这个保存的结果，防止之后重复运算  
                intermediate_index_table_schema = {}
                intermediate_index_table_schema["table_path"] = index_path
                intermediate_index_table_schema["table_format"] = time_agg_table_format
                intermediate_index_table_schema["partition_cols"] = list(src_time_cols)
                intermediate_index_table_schema["partition_cols_values"] = list(tgt_src_time_cols_values)
                
                read_result = read_graph_table_partitions(spark, intermediate_index_table_schema)
                full_time_agg_df = read_result["data"]
            
    # 如果有设定存储中间结果的表格，则保存结果到对应文件夹
    if time_agg_table_path is not None:
        # 以最优分区配置保存结果到对应的表格 
        pyspark_optimal_save(full_time_agg_df, time_agg_table_path, time_agg_table_format, "append",
                             src_time_cols, logger = logger)
        
        # 删除该时间点中间结果对应的文件夹
        if hdfs_check_file_exists(intermediate_table_path):
            hdfs_delete_dir(intermediate_table_path)

        # 重新读取出这个保存的结果，防止之后重复运算  
        read_result = read_graph_table_partitions(spark, time_agg_table_schema)
        full_time_agg_df = read_result["data"]

    # 添加对应的graph time列，再返回
    full_time_agg_df = full_time_agg_df.join(broadcast(graph_to_src_time_df), on = src_time_cols, 
                                             how = "left")

    # 删去对应的src_time_cols
    full_time_agg_df = full_time_agg_df.drop(*src_time_cols)

    return full_time_agg_df

# 读取graph组成表的函数
# 以后改成graph_token_table
# 这里面可以聚合同一类型的节点和边的来源于不同表的特征
# limit特征的方式应该改成规定哪个来源表的哪个特征通过哪种key_agg,time_agg得到的结果的limit
# 目前先不针对target_nodes做优化，先统一完成运算
# 以后再将target_nodes相关计算一路优化到最底层
def read_graph_table(spark, graph_table_schema, target_nodes_df = None):
    """
    读取图格式表
    """
    # 先确定要处理的graph table的名称和要读取的时间
    graph_table_name = graph_table_schema["name"]
    graph_time_cols = graph_table_schema["graph_time_cols"]
    graph_time_cols_values = graph_table_schema["graph_time_cols_values"]

    logger.info(f"Read graph token table {graph_table_name} at time columns "
                f"values {graph_time_cols_values}.")

    # 检查是否设定了结果的存储路径
    if "graph_token_table_path" in graph_table_schema:
        # 获得graph_token_table结果的存储配置信息
        graph_token_table_schema = {}
        graph_token_table_schema["table_path"] = graph_table_schema["graph_token_table_path"]
        graph_token_table_schema["table_format"] = graph_table_schema["graph_token_table_format"]
        graph_token_table_schema["partition_cols"] = graph_time_cols
        graph_token_table_schema["partition_cols_values"] = graph_time_cols_values

        logger.info(f"The result is setting to stored at {graph_table_schema['graph_token_table_path']}.")

        # 尝试读取对应数据
        read_result = read_graph_table_partitions(spark, graph_token_table_schema)

        # 检查对应数据是否存在
        if read_result["success"]:
            # 有则使用现有结果
            graph_token_df = read_result["data"]
            logger.info(f"The result of the graph token table {graph_table_name} already exist.")

            # 检查是否有target_nodes
            if target_nodes_df is not None:
                # 有则只保留对应结果
                graph_token_df = graph_token_df.join(target_nodes_df, on = target_nodes_df.columns, how = "inner")
                
            return graph_token_df
            
    # 读取该graph table对应的time aggregation数据
    graph_table_df = read_time_agg_table(spark, graph_table_schema, logger = logger)

    # 设定该graph table要保留的全部列的类型
    col_types = ["node_cols_to_aliases", "feat_cols_to_aliases", "time_cols_to_aliases"]
    
    # 构建 select 语句
    selected_columns = []
    for col_type in col_types:
        for col_to_alias in graph_table_schema[col_type]:
            selected_columns.append(col(col_to_alias[0]).alias(col_to_alias[1]))
    
    # 选择并重命名列
    graph_table_df = graph_table_df.select(*selected_columns)

    # Table limit
    if 'table_limit' in graph_table_schema and graph_table_schema['table_limit'] != '':
        logger.info(f"Table limitation: {graph_table_schema['table_limit']}")
        graph_table_df = graph_table_df.filter(graph_table_schema['table_limit'])

    # Table sample
    if 'table_samples' in graph_table_schema:
        for table_sample in graph_table_schema['table_samples']:
            sample_nodes_cols = table_sample['sample_nodes_cols']
            table_sample_type = table_sample['sample_type']
            table_sample_count = table_sample['sample_count']

            sample_id_cols = graph_time_cols + sample_nodes_cols

            logger.info(f'Table sampling: {table_sample_type}, {sample_id_cols}, {table_sample_count}')

            if table_sample_type == 'random':
                graph_table_df = random_n_sample(spark, graph_table_df, sample_id_cols, table_sample_count)
            elif table_sample_type == 'threshold':
                graph_table_df = threshold_n_sample(spark, graph_table_df, sample_id_cols, table_sample_count)

    # 检查是否要保存结果表
    if "graph_token_table_path" in graph_table_schema:
        # 保存对应结果
        pyspark_optimal_save(graph_table_df, graph_token_table_schema["table_path"], graph_token_table_schema["table_format"], 
                             "append", graph_token_table_schema["partition_cols"], logger = logger)
        
        # 重新读取保存的结果，防止重复运算
        read_result = read_graph_table_partitions(spark, graph_token_table_schema)
        graph_table_df = read_result["data"]
        
    # 如果有target_nodes则只保留对应的数据(这个应该被包含在read_time_agg_table的功能中，待优化)
    if target_nodes_df is not None:
        graph_table_df = graph_table_df.join(target_nodes_df, on = target_nodes_df.columns, how = "inner")

    return graph_table_df