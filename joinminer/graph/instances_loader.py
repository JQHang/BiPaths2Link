from .graph_loader import read_node_table, read_edge_table, read_label_table
from .graph_loader import read_graph_table_partitions, read_graph_table
from ..python import setup_logger, time_costing, python_to_spark_date_format
from ..hdfs import hdfs_check_file_exists, hdfs_read_json
from ..pyspark import pyspark_optimal_save

from functools import reduce
from pyspark.sql.functions import lit, col
from pyspark.sql import functions as F

# 初始化instances的相关配置
def train_inst_config_init(graph, train_instances_config, logger = setup_logger()):
    # 记录训练用的实例初始化后的配置
    train_instances = {}
    
    # 获得这些instance由哪些节点类型组成
    train_instances["nodes_types"] = train_instances_config["inst_nodes_types"]
    
    # 获得这些节点对应的节点列类型 
    inst_nodes_cols_types = []
    for agg_node_type in train_instances["nodes_types"]:
        inst_nodes_cols_types.extend(graph.nodes[agg_node_type]["node_col_types"])

    # 设定该组train_instances的名称
    train_instances["name"] = f"train_instances_query"
    
    # 设定这些实例对应到图里的目标时间
    train_instances["graph_time_cols"] = list(graph.graph_time_cols_alias)
    train_instances["graph_time_cols_values"] = list(graph.graph_time_cols_values)
    train_instances["graph_time_cols_formats"] = list(graph.graph_time_cols_formats)
    
    # 获得训练实例相关结果存储的路径
    task_data_path = train_instances_config["task_data_path"]
    train_instances["task_data_path"] = task_data_path
    
    # 记录实例标签信息的存储路径及对应的存储格式
    train_instances["label_table_path"] = task_data_path + "/instance_label"
    train_instances["label_table_format"] = train_instances_config["task_label_table_format"]
    
    # 记录实例特征信息的存储路径及对应的存储格式
    train_instances["feat_table_path"] = task_data_path + "/instance_feat"
    train_instances["feat_table_format"] = train_instances_config["task_feat_table_format"]

    # 记录实例数据集信息的存储路径及对应的存储格式
    train_instances["dataset_table_path"] = task_data_path + "/dataset"
    train_instances["dataset_table_format"] = train_instances_config["task_dataset_table_format"]

    # 获得对dataset进行归一化的方法
    train_instances["dataset_scale_funcs"] = train_instances_config["dataset_scale_funcs"]
    
    # 记录对dataset进行归一化的具体pipline
    train_instances["scale_pipline_path"] = task_data_path + "/scale_pipline"

    # 记录缩放后的数据集的存储路径以及对应的存储格式
    train_instances["scaled_dataset_path"] = task_data_path + "/scaled_dataset"
    train_instances["scaled_dataset_format"] = train_instances_config["task_dataset_table_format"]
    
    # 获得数据集信息在本地存储的路径
    train_instances["dataset_table_local_path"] = train_instances_config["dataset_local_path"]
    
    # 设定这些实例对应的source table的基础配置
    train_instances["src_table_path"] = train_instances_config["src_table_path"]
    train_instances["src_table_format"] = train_instances_config["src_table_format"]

    # 先检查配置里是否有src_table_schema相关配置
    if "src_table_schema" in train_instances_config:
        # 读取对应的schema
        src_table_schema = train_instances_config["src_table_schema"]
    else:
        # 检查该路径下是否有对应的table_schema
        if hdfs_check_file_exists(train_instances_config["src_table_path"] + "/_Table_Schema"):
            # 读取对应的schema
            src_table_schema = hdfs_read_json(train_instances_config["src_table_path"], 
                                              "_Table_Schema")
        else:
            # 否则报错
            raise ValueError(f"We need table schema for the source table.")
            
    # 设定要从source table中读取的各个列名
    train_instances["src_node_cols"] = []
    for node_cols in train_instances_config["inst_nodes_columns"]:
        train_instances["src_node_cols"].extend(node_cols)
        
    # 获得这些列对应的列类型
    src_node_cols_types = []
    for node_col in train_instances["src_node_cols"]:
        src_node_cols_types.append(src_table_schema["node_col_to_types"][node_col])
        
    # 确认对应的node_col_type和agg_node的一致
    if inst_nodes_cols_types != src_node_cols_types:
        raise ValueError(f"Require same source nodes columns types {src_node_cols_types} for "
                         f"the instance nodes {inst_nodes_cols_types}.")
    
    train_instances["src_feat_cols"] = list(src_table_schema["feat_cols"])
    train_instances["src_time_cols"] = list(src_table_schema["time_cols"])
    train_instances["src_time_cols_formats"] = list(src_table_schema["time_cols_formats"])

    # 如果有规定time aggregation结果的存储路径，则加入相关设定
    if "time_agg_table_path" in train_instances_config:
        train_instances["time_agg_table_path"] = train_instances_config["time_agg_table_path"]
        train_instances["time_agg_table_format"] = train_instances_config["time_agg_table_format"]
        train_instances["time_agg_save_interval"] = train_instances_config["time_agg_save_interval"]
    
    # 设定source table通过time aggregation形成edge table的方案 
    train_instances["time_aggs"] = graph.time_aggs_init(src_table_schema,
                                                        train_instances_config["time_aggs_configs"])

    # 节点列改成是第几个聚合节点对应的第几列
    inst_node_cols = []
    for inst_node_type_i, inst_node_type in enumerate(train_instances["nodes_types"]):
        for inst_node_col_i, inst_node_col_type in enumerate(graph.nodes[inst_node_type]["node_col_types"]):
            inst_node_col = f"{inst_node_type}_{inst_node_type_i}_{inst_node_col_type}_{inst_node_col_i}"
            inst_node_cols.append(inst_node_col)
            
    # 设定要获得的最终train instance table的各个列的相关配置
    # 节点列改成是第几个聚合节点对应的第几列(可以在设定里面加入一些设定来优化表示)
    train_instances["node_cols_to_aliases"] = [[x, y] for x, y in zip(train_instances["src_node_cols"], inst_node_cols)]

    # 特征列使用time aggregation后的特征列
    train_instances["feat_cols_to_aliases"] = []
    for time_agg in train_instances["time_aggs"]:
        for time_range in time_agg["time_ranges"]:
            train_instances["feat_cols_to_aliases"].extend([[x, y] for x, y in zip(time_range["agg_feat_cols"], 
                                                                                   time_range["agg_feat_cols"])])
    
    train_instances["time_cols_to_aliases"] = [[x, y] for x, y in zip(train_instances["graph_time_cols"], 
                                                                      train_instances["graph_time_cols"])]
        
    # 设定对最终的graph table的限制
    if "instance_limit" in train_instances_config:
        train_instances["table_limit"] = train_instances_config["instance_limit"]

    # 设定最终要使用的标签列及对应的别名
    train_instances["label_cols_to_aliases"] = train_instances_config["label_cols_to_aliases"]
    
    # 设定对数据的分割方式
    train_instances["train_inst_split"] = train_instances_config["train_inst_split"]
    
    return train_instances

@time_costing
def read_train_inst_table(spark, graph, train_instances, logger = setup_logger()):
    # 设定采样结果对应的instance表的相关schema
    train_inst_label_table_schema = {}
    train_inst_label_table_schema["table_path"] = train_instances["label_table_path"]
    train_inst_label_table_schema["table_format"] = train_instances['label_table_format']
    train_inst_label_table_schema["partition_cols"] = list(graph.graph_time_cols_alias)
    train_inst_label_table_schema["partition_cols_values"] = list(graph.graph_time_cols_values)

    # 尝试读取对应数据
    read_result = read_graph_table_partitions(spark, train_inst_label_table_schema)

    # 检查是否读取到全部的结果
    if read_result["success"]:
        logger.info(f"Instance basic information already exists.")
        
        # 如果已有全部结果，则直接返回
        return read_result["data"]
        
    else:
        # 否则若图有时间列则获得未完成的时间点，则据此设定目标时间
        graph_time_cols = list(graph.graph_time_cols_alias)
        graph_time_cols_values = list(read_result["failed_partitions"])
        graph_time_cols_formats = list(graph.graph_time_cols_formats)
    
    # 读取该edge对应的具体数据
    instance_table_df = read_graph_table(spark, train_instances, logger = logger)

    # 持久化该结果，方便之后进行数据分割
    instance_table_df.persist()

    # 记录由数据分割获得的全部子instances
    sub_inst_list = []
    
    # 进行数据分割 
    for split_type in train_instances["train_inst_split"]:
        # 依次获得各个分割区间内要获得的数据
        for dataset_config in train_instances["train_inst_split"][split_type]:
            # 获取该类型样本对应数据
            sub_inst_df = instance_table_df.filter(dataset_config["inst_limit"])
            
            # 只保留所需列
            selected_columns = []
            for col_type in ["time_cols_to_aliases", "node_cols_to_aliases"]:
                for col_to_alias in train_instances[col_type]:
                    selected_columns.append(col(col_to_alias[1]).alias(col_to_alias[1]))
            for col_to_alias in train_instances["label_cols_to_aliases"]:
                    selected_columns.append(col(col_to_alias[0]).alias(col_to_alias[1]))
            sub_inst_df = sub_inst_df.select(*selected_columns)

            # 获得对应的样本数目
            sub_inst_count = sub_inst_df.count()
            logger.info(f"Raw number of {split_type} instances with instance limit "
                        f"{dataset_config['inst_limit']}: {sub_inst_count}")

            # 查看是否满足行数限制
            if "max_inst_count" in dataset_config and sub_inst_count > dataset_config["max_inst_count"]:
                # 如果负样本数据量超过范围，则进行采样
                sample_fraction = dataset_config["max_inst_count"] / float(sub_inst_count)
                logger.info(f"Too much instances, will only sample {sample_fraction:.4g} percent instances.")

                # 随机抽样
                sub_inst_df = sub_inst_df.sample(False, sample_fraction)
            
            # 加入样本类型列
            sub_inst_df = sub_inst_df.withColumn("split_type", lit(split_type))
            
            # 保存结果
            sub_inst_list.append(sub_inst_df)
    
    # 合并分割结果 
    train_instance_table_df = sub_inst_list[0]
    for sub_inst_df in sub_inst_list[1:]:
        train_instance_table_df = train_instance_table_df.union(sub_inst_df)
        
    # 保存结果
    pyspark_optimal_save(train_instance_table_df, train_inst_label_table_schema["table_path"], 
                         train_inst_label_table_schema["table_format"], "append",
                         train_inst_label_table_schema["partition_cols"])
    
    # 释放persist的变量
    instance_table_df.unpersist()
    
    # 重新读取对应结果
    read_result = read_graph_table_partitions(spark, train_inst_label_table_schema)
    
    return read_result["data"]

@time_costing
def read_labeled_samples(spark, graph, labeled_samples_config, logger = setup_logger()):
    # 获得结果存储的路径
    sample_result_path = labeled_samples_config["task_data_path"] + "/Labeled_Samples"
    
    # 检查结果是否已存在
    if hdfs_check_file_exists(sample_result_path + f"/_SUCCESS"):
        logger.info(f'Sample result already exists: {sample_result_path}')
        labeled_samples_df = spark.read.parquet(sample_result_path)
        return labeled_samples_df
    
    # 获得标签表对应路径
    label_table_dir = labeled_samples_config["label_table_root_path"] + labeled_samples_config["label_table_rel_path"]
    
    # 设定对应的label_table_schema
    label_table_schema = {}
    label_table_schema["table_path"] = label_table_dir
    
    # 获得表对应的文件格式
    if "label_table_format" in labeled_samples_config:
        label_table_schema["table_format"] = labeled_samples_config["label_table_format"]
    
    # 获得节点列对应的配置
    label_table_schema["node_cols_to_aliases"] = []
    for node_cols_to_aliases in labeled_samples_config["nodes_cols_to_aliases"]:
        label_table_schema["node_cols_to_aliases"].extend(node_cols_to_aliases)
    
    label_table_schema["feat_cols_to_aliases"] = labeled_samples_config["feat_cols_to_aliases"]
    label_table_schema["label_cols_to_aliases"] = labeled_samples_config["label_cols_to_aliases"]
    label_table_schema["time_cols_to_aliases"] = labeled_samples_config["time_cols_to_aliases"]
    
    if "sample_limit" in labeled_samples_config:
        label_table_schema["sample_limit"] = labeled_samples_config["sample_limit"]
    
    # 读取对应的标签表
    label_table_df = read_label_table(spark, graph, label_table_schema, logger = logger)
    
    label_table_df.persist()
    
    # 获得不同类型的样本（训练、验证、测试）
    labeled_samples_df_list = []
    for sample_type in labeled_samples_config["sample_types_limits"]:
        # 依次处理不同部分的样本
        for sample_type_limit in labeled_samples_config["sample_types_limits"][sample_type]:
            # 获取该类型样本对应数据
            sample_type_df = label_table_df.filter(sample_type_limit["sample_limit"])
            
            # 全部所需列的类型
            col_types = ["node_cols_to_aliases", "label_cols_to_aliases", "time_cols_to_aliases"]
            
            # 只保留所需列
            selected_columns = []
            for col_type in col_types:
                for col_to_alias in label_table_schema[col_type]:
                    selected_columns.append(col_to_alias[1])
            sample_type_df = sample_type_df.select(*selected_columns)

            # 获得对应的样本数目
            sample_type_count = sample_type_df.count()
            logger.info(f"Raw number of {sample_type} samples with sample limit {sample_type_limit['sample_limit']}: {sample_type_count}")

            # 查看是否满足行数限制
            if "max_sample_count" in sample_type_limit and sample_type_count > sample_type_limit["max_sample_count"]:
                # 如果负样本数据量超过范围，则进行采样
                sample_fraction = sample_type_limit["max_sample_count"] / float(sample_type_count)
                logger.info(f"Too much samples, will only keep {sample_fraction:.4g} percent samples.")

                # 随机抽样
                sample_type_df = sample_type_df.sample(False, sample_fraction)
            
            # 加入样本类型列
            sample_type_df = sample_type_df.withColumn("sample_type", lit(sample_type))
            
            # 保存结果
            labeled_samples_df_list.append(sample_type_df)
        
    # 合并各类型的样本
    labeled_samples_df = labeled_samples_df_list[0]
    for sub_labeled_samples_df in labeled_samples_df_list[1:]:
        labeled_samples_df = labeled_samples_df.union(sub_labeled_samples_df)

    # 增加样本对应的graph时间
    sample_time_cols = [x[1] for x in labeled_samples_config["time_cols_to_aliases"]]
    labeled_samples_df = labeled_samples_df.withColumn(
                                                            'full_sample_datetime_str',
                                                            F.concat_ws(
                                                                ' ',
                                                                *sample_time_cols
                                                            )
                                                        ).drop(*sample_time_cols)
    
    full_sample_datetime_format = " ".join(labeled_samples_config["time_cols_formats"])
    full_sample_datetime_format = python_to_spark_date_format(full_sample_datetime_format)
    labeled_samples_df = labeled_samples_df.withColumn(
                                                            'full_sample_datetime',
                                                            F.to_timestamp('full_sample_datetime_str', 
                                                                           full_sample_datetime_format)
                                                        )

    for col_index in range(len(graph.graph_time_cols_alias)):
        graph_time_col_format = python_to_spark_date_format(graph.graph_time_cols_formats[col_index])
        labeled_samples_df = labeled_samples_df.withColumn(
                                                                graph.graph_time_cols_alias[col_index],
                                                                F.date_format('full_sample_datetime', 
                                                                              graph_time_col_format)
                                                            )
    labeled_samples_df = labeled_samples_df.drop('full_sample_datetime_str', 'full_sample_datetime')
    
    # 保存结果
    pyspark_optimal_save(labeled_samples_df, sample_result_path, "parquet", "overwrite")
    
    # 释放persist的变量
    label_table_df.unpersist()
    
    # 重新读取全量结果，来返回，避免因为惰性计算造成的重复运算
    labeled_samples_df = spark.read.parquet(sample_result_path)
    
    return labeled_samples_df

@time_costing
def read_unlabeled_samples(spark, graph, unlabeled_samples_config, logger = setup_logger()):
    # 获得结果存储的路径
    sample_result_path = unlabeled_samples_config["task_data_path"] + "/Unlabeled_Samples"
    
    # 检查结果是否已存在
    if hdfs_check_file_exists(sample_result_path + f"/_SUCCESS"):
        logger.info(f'Sample result already exists: {sample_result_path}')
        unlabeled_samples_df = spark.read.parquet(sample_result_path)
        return unlabeled_samples_df
    
    # 获得目标节点类型
    nodes_types = unlabeled_samples_config["nodes_types"]
    
    # 获得目标节点统一名称后的节点列
    unified_nodes_cols = unlabeled_samples_config["unified_nodes_cols"]
    
    # 依次查询各个来源表
    unlabeled_samples_df_list = []
    for sample_table_config in unlabeled_samples_config["sample_tables"]:
        # 获得来源表存储的路径
        sample_table_path = sample_table_config["table_path"]
        sample_table_format = sample_table_config["table_format"]
        
        logger.info(f'Read sample table from directory: {sample_table_path}')

        sample_table_df = spark.read.format(sample_table_format).load(sample_table_path)

        # Sample limit
        if 'sample_limit' in sample_table_config and sample_table_config['sample_limit'] != '':
            logger.info(f"Sample Limitation: {sample_table_config['sample_limit']}")
            sample_table_df = sample_table_df.filter(sample_table_config['sample_limit'])
        
        # 构建select语句
        selected_columns = []
        for node_col_i, unified_node_col in enumerate(unified_nodes_cols):
            table_node_col = sample_table_config["nodes_cols"][node_col_i]
            selected_columns.append(col(table_node_col).alias(unified_node_col))

        # 选择并重命名列
        sample_table_df = sample_table_df.select(*selected_columns)
        
        unlabeled_samples_df_list.append(sample_table_df)
        
    # 合并各个来源表中的节点并去重
    unlabeled_samples_df = unlabeled_samples_df_list[0]
    for sub_unlabeled_samples_df in unlabeled_samples_df_list[1:]:
        unlabeled_samples_df = unlabeled_samples_df.union(sub_unlabeled_samples_df)
    unlabeled_samples_df = unlabeled_samples_df.distinct()

    # 获得这些样本对应的graph_time列及对应的值
    for col_index in range(len(graph.graph_time_cols_alias)):
        graph_time_col = graph.graph_time_cols_alias[col_index]
        graph_time_col_value = unlabeled_samples_config["graph_time_cols_values"][col_index]

        # 加入对应的时间列
        unlabeled_samples_df = unlabeled_samples_df.withColumn(graph_time_col, lit(graph_time_col_value))
        
    # 保存结果
    unlabeled_samples_df.persist()

    pyspark_optimal_save(unlabeled_samples_df, sample_result_path, "parquet", "overwrite", logger = logger)
    
    # 显示结果数量
    unlabeled_samples_count = unlabeled_samples_df.count()
    logger.info(f"Number of unlabeled samples: {unlabeled_samples_count}")
    unlabeled_samples_df.unpersist()
    
    # 重新读取全量结果，来返回，避免因为惰性计算造成的重复运算
    unlabeled_samples_df = spark.read.parquet(sample_result_path)
    
    return unlabeled_samples_df