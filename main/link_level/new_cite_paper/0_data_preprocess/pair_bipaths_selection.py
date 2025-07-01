#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import sys

os.environ['SPARK_HOME']="/software/servers/10k/mart_scr/spark_3.0"
os.environ['PYTHONPATH']="/software/servers/10k/mart_scr/spark_3.0/python:/software/servers/10k/mart_scr/spark_3.0/python/lib/py4j-0.10.9-src.zip"
os.environ['LD_LIBRARY_PATH']="/software/servers/jdk1.8.0_121/lib:/software/servers/jdk1.8.0_121/jre/lib/amd64/server:/software/servers/hope/mart_sch/hadoop/lib/native"
os.environ['PYSPARK_PYTHON']="/usr/local/anaconda3/bin/python3.6"
os.environ['PYSPARK_DRIVER_PYTHON']="/usr/local/anaconda3/bin/python3.6"

sys.path.insert(0, '/software/servers/10k/mart_scr/spark_3.0/python/lib/py4j-0.10.9-src.zip')
sys.path.insert(0, '/software/servers/10k/mart_scr/spark_3.0/python')
sys.path.insert(0, '/software/servers/10k/mart_scr/spark_3.0/python/lib/pyspark.zip')


# In[2]:


from joinminer.pyspark import ResilientSparkRunner
from joinminer.graph import TableGraph, read_labeled_samples
from joinminer.python import mkdir, setup_logger, time_costing, read_json_file

from datetime import datetime


# In[3]:


# 获得项目文件夹根目录路径
from joinminer import PROJECT_ROOT

# 日志信息保存文件名
log_files_dir = PROJECT_ROOT + '/data/result_data/log_files/pair_paths_finder'
log_filename = log_files_dir + f'/{datetime.now().strftime("%Y-%m-%d-%H:%M")}.log'
mkdir(log_files_dir)

logger = setup_logger(log_filename, logger_name = "joinminer")

# Table_graph config
table_graph_config_file = PROJECT_ROOT + '/main/config/table_graphs/AMiner_New_Citation_V1.json'
table_graph_config = read_json_file(table_graph_config_file)

# Graph init
graph = TableGraph(table_graph_config)

# Relevant bidirectional meta-paths file path
join_edges_config_file = PROJECT_ROOT + '/main/config/join_edges_types/new_citation_relevant_bipaths.json'
sorted_relevant_paths_config = read_json_file(join_edges_config_file)

# Labeled samples config
labeled_samples_config_file = PROJECT_ROOT + '/main/config/labeled_samples/new_citation_pred_V1.json'
labeled_samples_config = read_json_file(labeled_samples_config_file)

# 设定取用于inference的reliable_node_pairs时的数据的时间
inference_time_cols_values = ['2019']


# In[4]:


# paths_finder生成的join_edges的基础配置
join_edges_default_config = {}
join_edges_default_config["join_edge_root_path"] = "/user/mart_coo/mart_coo_innov/CompGraph/AMiner/join_edge"
join_edges_default_config["join_edge_table_format"] = "parquet"
join_edges_default_config["join_edges_root_path"] = "/user/mart_coo/mart_coo_innov/CompGraph/AMiner/join_edges"
join_edges_default_config["join_edges_table_format"] = "parquet"


# In[5]:


# spark配置参数
config_dict = {
                "spark.default.parallelism": "1600",
                "spark.sql.shuffle.partitions": "3200",
                "spark.sql.broadcastTimeout": "3600",
                "spark.driver.memory": "20g",
                "spark.driver.cores": "4",
                "spark.driver.maxResultSize": "0",
                "spark.executor.memory": "12g",
                "spark.executor.cores": "4",
                "spark.executor.instances": "400"
            }

# 启动spark
spark_runner = ResilientSparkRunner(config_dict = config_dict)


# # 初始化node pairs选取所需配置

# In[6]:


from joinminer.graph import join_edges_init
from joinminer.graph import read_labeled_samples
from joinminer.graph import graph_token_node_col_name, standard_node_col_name

import copy

def reliable_node_pairs_config_init(spark, reliable_bipaths, join_edges_list_name, labeled_samples_config, join_edges_default_config):
    # 创建用于获得reliable_node_pairs的bipaths的配置
    # 大致对标join_edges_list_collect初始化后的配置
    main_bipaths = {}
    main_bipaths["join_edges_list_name"] = join_edges_list_name
    main_bipaths["query_nodes_types"] = labeled_samples_config["nodes_types"]
    main_bipaths["query_nodes_cols"] = []
    for query_node_i in range(len(main_bipaths["query_nodes_types"])):
        # 获得该节点类型
        query_node_type = main_bipaths["query_nodes_types"][query_node_i]
        
        for node_col_i in range(len(graph.nodes[query_node_type]["node_col_types"])):
            query_node_token_col = graph_token_node_col_name(query_node_type, query_node_i, node_col_i)
            main_bipaths["query_nodes_cols"].append(query_node_token_col)
                
    main_bipaths["join_edges_list_path"] = f"/user/mart_coo/mart_coo_innov/CompGraph/AMiner/join_edges_list/{join_edges_list_name}"
    main_bipaths["join_edges_list_table_format"] = "parquet"
    
    # 因为总体路径较少，不用分成父路径一条条算了，直接完成整个路径的计算
    main_bipaths["bipaths_list_schema"] = []
    for reliable_bipath in reliable_bipaths:
        main_path_schema = {}
        main_path_schema["join_edges_name"] = reliable_bipath["join_edges_name"]
        main_path_schema["bipath_result_path"] = join_edges_default_config["join_edges_root_path"] + f"/bipaths_v1_bipath_" + reliable_bipath["join_edges_name"]
        
        # 初始化forward_path相关配置
        exclude_keys = ["data", "path_count", "parent_join_edges"]
        main_path_schema["forward_path"] = {k: copy.deepcopy(v) for k, v in reliable_bipath["forward_path"].items() if k not in exclude_keys}

        # 修正Forward_path对应的名称和存储位置
        forward_path_name = "bipaths_v1_forward_" + main_path_schema["forward_path"]['name']
        main_path_schema["forward_path"]['name'] = forward_path_name
        main_path_schema["forward_path"]["join_edges_path"] = join_edges_default_config["join_edges_root_path"] + f"/{forward_path_name}"
        main_path_schema["forward_path"]["query_config"]["result_path"] = main_path_schema["forward_path"]["join_edges_path"]

        # 检查是否有backward_path
        if "backward_path" in reliable_bipath:
            # 初始化backward_path相关配置
            exclude_keys = ["data", "path_count", "parent_join_edges"]
            main_path_schema["backward_path"] = {k: copy.deepcopy(v) for k, v in reliable_bipath["backward_path"].items() if k not in exclude_keys}

            # 修正Backward_path对应的名称和存储位置
            backward_path_name = "bipaths_v1_backward_" + main_path_schema["backward_path"]['name']
            main_path_schema["backward_path"]['name'] = backward_path_name
            main_path_schema["backward_path"]["join_edges_path"] = join_edges_default_config["join_edges_root_path"] + f"/{backward_path_name}"
            main_path_schema["backward_path"]["query_config"]["result_path"] = main_path_schema["backward_path"]["join_edges_path"]
        
            # 加入tail_path的select_col_alias
            main_path_schema["backward_path"]["id_col_alias"] = list(reliable_bipath["backward_path_id_col_alias"])
            
            # 记录tail_path用于join的node_cols 
            main_path_schema["backward_path"]["join_node_cols"] = list(reliable_bipath["backward_path_join_node_cols"])

        # 获得query nodes对应的path_indexes
        query_nodes_path_indexes = [reliable_bipath["join_edges_schema"][0]["join_nodes_indexes"][0], 
                                    reliable_bipath["join_edges_schema"][-1]["add_nodes_indexes"][0]]
        
        # 获得最终要保留的query_node_id_col_aliases
        main_path_schema["query_nodes_id_cols_aliases"] = []
        for query_node_i in range(len(main_bipaths["query_nodes_types"])):
            # 获得该节点类型
            query_node_type = main_bipaths["query_nodes_types"][query_node_i]
            
            # 获得该node在path上对应的index
            node_path_index = query_nodes_path_indexes[query_node_i]
            
            for node_col_i in range(len(graph.nodes[query_node_type]["node_col_types"])):
                query_node_path_col = standard_node_col_name(query_node_type, node_path_index, node_col_i)
                query_node_token_col = graph_token_node_col_name(query_node_type, query_node_i, node_col_i)
                main_path_schema["query_nodes_id_cols_aliases"].append([query_node_path_col, 
                                                                        query_node_token_col])

        for time_col in graph.graph_time_cols_alias:
            main_path_schema["query_nodes_id_cols_aliases"].append([time_col, time_col])
        
        main_bipaths["bipaths_list_schema"].append(main_path_schema)

    return main_bipaths
    
# 获得已知目标边涉及到的时间点作为目标时间点
samples_df = spark_runner.run(read_labeled_samples, graph, labeled_samples_config, logger = logger)
distinct_time_cols_rows = samples_df.select(graph.graph_time_cols_alias).distinct().collect()
query_time_cols_values = [[row[c] for c in graph.graph_time_cols_alias] for row in distinct_time_cols_rows]
    
# 如果推理数据对应的时间点不在里面则加入
if inference_time_cols_values not in query_time_cols_values:
    query_time_cols_values.append(inference_time_cols_values)

# 设定query_nodes相关配置
main_bipath_query_config = {}
main_bipath_query_config["graph_time_values"] = query_time_cols_values

# 生成bipaths_union所需相关配置
main_path_count = 5
main_bipaths = spark_runner.run(reliable_node_pairs_config_init, sorted_relevant_paths_config[:main_path_count], 
                                f"top_{main_path_count}_reliable_bipaths_union",
                                labeled_samples_config, join_edges_default_config)


# # 合并多组路径获得selected node pairs

# In[7]:


from joinminer.graph import join_edges_query
from joinminer.hdfs import hdfs_check_partitions
from joinminer.hdfs import hdfs_check_file_exists
from joinminer.pyspark import pyspark_read_table, pyspark_optimal_save

from functools import reduce
from pyspark.sql.functions import col

def bipaths_union(spark, graph, bipaths, query_config):
    # 获得这些bipaths对应的基础信息
    join_edges_list_name = bipaths["join_edges_list_name"]
    query_nodes_types = bipaths["query_nodes_types"]
    query_nodes_cols = bipaths["query_nodes_cols"]
    
    logger.info(f"Union join_edges_list {join_edges_list_name} for query nodes types "
                f"{query_nodes_types} of cols {query_nodes_cols}")

    # 获得query nodes对应的id列
    query_nodes_id_cols = query_nodes_cols + list(graph.graph_time_cols_alias)

    # 获得目标结果的存储路径
    # union不用检查目标点是否设定存储路径了，因为有目标点的话就不需要union了
    result_path = bipaths['join_edges_list_path']
        
    # 获得结果对应的存储格式
    result_format = bipaths["join_edges_list_table_format"]

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
        united_bipaths_df = pyspark_read_table(spark, result_path, result_format, partition_cols, partition_cols_values)
        
        return united_bipaths_df

    # 如果要开始计算，先以未完成的时间更新query_config
    logger.info(f"Missing target partitions: {missing_values}")
    missing_query_config = {}
    missing_query_config["graph_time_values"] = missing_values

    # 依次获得各个bipath对应的数据
    bipath_df_list = []
    for bipath in bipaths["bipaths_list_schema"]:
        # 检查是否已有对应的bipath结果
        is_complete, _ = hdfs_check_partitions(bipath["bipath_result_path"], partition_cols, missing_values)
        
        # 如果已有对应的全量结果，则直接读取对应的结果并返回
        if is_complete:
            bipath_df = pyspark_read_table(spark, bipath["bipath_result_path"], result_format, partition_cols, missing_values)
        else:
            # 获得forward_path对应的该join_edges的数据，将其作为bipath_df的一部分
            bipath_df = join_edges_query(spark, graph, bipath["forward_path"], missing_query_config)
    
            # 检查是否有对应的backward_path
            if "backward_path" in bipath:
                # 获得对应的backward_path的全量数据
                backward_path_df = join_edges_query(spark, graph, bipath["backward_path"], missing_query_config)
    
                # 修正backward_path的列名
                select_cols = [col(column).alias(alias) for column, alias in bipath["backward_path"]["id_col_alias"]]
                backward_path_df = backward_path_df.select(*select_cols)
    
                # 设定要用于join的id列
                join_id_cols = bipath["backward_path"]["join_node_cols"] + graph.graph_time_cols_alias
                
                # Join forward_path和backward_path
                bipath_df = bipath_df.join(backward_path_df, on = join_id_cols, how = "inner")

            pyspark_optimal_save(bipath_df, bipath["bipath_result_path"], result_format, "overwrite", partition_cols)
            
        # 只保留要union的id列并修正列名
        select_cols = [col(column).alias(alias) for column, alias in bipath["query_nodes_id_cols_aliases"]]
        bipath_df = bipath_df.select(*select_cols).distinct()

        # 记录该bipath对应的要union的数据
        bipath_df_list.append(bipath_df)
        
    # union全部bipath的结果
    united_bipaths_df = reduce(lambda df1, df2: df1.unionByName(df2), bipath_df_list).distinct()

    # 保存结果
    pyspark_optimal_save(united_bipaths_df, result_path, result_format, "overwrite", partition_cols)
    
    # 重新读取完整结果
    united_bipaths_df = pyspark_read_table(spark, result_path, result_format, partition_cols, partition_cols_values)
    
    return united_bipaths_df


# # 要收集bipaths的节点对的对应的配置

# In[8]:


from joinminer.hdfs import hdfs_check_partitions
from joinminer.pyspark import pyspark_read_table, pyspark_optimal_save, random_n_sample

from pyspark.sql.functions import lit
from pyspark.sql.functions import max as max_, count as count_

def clt_bipaths_query_config_init(spark, graph, main_bipaths, main_bipath_query_config, labeled_samples_config, inference_time_cols_values):    
    # 设定该任务对应的文件夹
    task_data_path = labeled_samples_config["task_data_path"]
    
    # 设定reliable_node_pairs结果保存路径
    result_path = task_data_path + f"/reliable_node_pairs"
    
    # 设定reliable_node_pairs结果保存格式
    result_format = "parquet"
    
    # 检查是否已有对应结果
    is_complete, _ = hdfs_check_partitions(result_path)

    if is_complete:
        reliable_node_pairs_df = pyspark_read_table(spark, result_path, result_format)

        clt_bipaths_query_config = {}
        clt_bipaths_query_config["graph_time_values"] = list(main_bipath_query_config["graph_time_values"])
        clt_bipaths_query_config["tgt_query_nodes"] = {}
        clt_bipaths_query_config["tgt_query_nodes"]["result_path"] = task_data_path
        clt_bipaths_query_config["tgt_query_nodes"]["df"] = reliable_node_pairs_df
        
        return clt_bipaths_query_config

    # 获得node_pairs对应的id列
    node_pair_id_cols = []
    
    # 获得已知目标边要保留的列及读取后的别名
    sample_col_aliases = []
    
    head_node_type = main_bipaths["query_nodes_types"][0]
    for node_col_i in range(len(graph.nodes[head_node_type]["node_col_types"])):
        node_sample_col = labeled_samples_config["nodes_cols_to_aliases"][0][node_col_i][1]
        node_token_col = graph_token_node_col_name(head_node_type, 0, node_col_i)
        sample_col_aliases.append([node_sample_col, node_token_col])
        node_pair_id_cols.append(node_token_col)
        
    tail_node_type = main_bipaths["query_nodes_types"][1]
    for node_col_i in range(len(graph.nodes[tail_node_type]["node_col_types"])):
        node_sample_col = labeled_samples_config["nodes_cols_to_aliases"][1][node_col_i][1]
        node_token_col = graph_token_node_col_name(tail_node_type, 1, node_col_i)
        sample_col_aliases.append([node_sample_col, node_token_col])
        node_pair_id_cols.append(node_token_col)

    node_pair_id_cols = node_pair_id_cols + list(graph.graph_time_cols_alias)
    for time_col in graph.graph_time_cols_alias:
        sample_col_aliases.append([time_col, time_col])
        
    sample_col_aliases.append(["sample_type", "sample_type"])

    # 获得全部已知目标边
    samples_df = read_labeled_samples(spark, graph, labeled_samples_config, logger = logger)

    # 保留对应列
    select_cols = [col(column).alias(alias) for column, alias in sample_col_aliases]
    samples_df = samples_df.select(*select_cols).distinct().withColumn("label", lit(1))
    
    # 获得bipaths_union的结果
    united_bipaths_df = bipaths_union(spark, graph, main_bipaths, main_bipath_query_config)

    #################################################################################################### 
    # 为new paper citation准备的特殊处理,先准备所需的数据
    graph_hdfs_path = "/user/mart_coo/mart_coo_innov/CompGraph/AMiner/raw"

    author_write_paper_df = spark.read.parquet(graph_hdfs_path + "/author_write_paper")
    author_write_paper_17_df = author_write_paper_df.filter(col("year") <= "2017")
    author_write_paper_17_df = author_write_paper_17_df.select("author_id", "paper_id").distinct()
    author_write_paper_17_df = author_write_paper_17_df.withColumn("year", lit("2017"))
    author_write_paper_19_df = author_write_paper_df.filter(col("year") <= "2019")
    author_write_paper_19_df = author_write_paper_19_df.select("author_id", "paper_id").distinct()
    author_write_paper_19_df = author_write_paper_19_df.withColumn("year", lit("2019"))
    author_past_paper_df = author_write_paper_17_df.unionByName(author_write_paper_19_df)
    
    paper_cite_paper_df = spark.read.parquet(graph_hdfs_path + "/paper_cite_paper")
    paper_cite_paper_df = paper_cite_paper_df.filter(col("year") <= "2019")
    author_cite_paper_df = author_write_paper_df.join(paper_cite_paper_df, on = ["paper_id", "year"], how = "inner")
    author_cited_paper_17_df = author_cite_paper_df.filter(col("year") <= "2017")
    author_cited_paper_17_df = author_cited_paper_17_df.select("author_id", "cite_paper_id").distinct()
    author_cited_paper_17_df = author_cited_paper_17_df.withColumn("year", lit("2017"))
    author_cited_paper_19_df = author_cite_paper_df.filter(col("year") <= "2019")
    author_cited_paper_19_df = author_cited_paper_19_df.select("author_id", "cite_paper_id").distinct()
    author_cited_paper_19_df = author_cited_paper_19_df.withColumn("year", lit("2019"))
    author_cited_paper_df = author_cited_paper_17_df.unionByName(author_cited_paper_19_df)
    
    # 关联到的论文不能是作者本人写的 
    node_token_col = graph_token_node_col_name("Author", 0, 0)
    author_past_paper_df = author_past_paper_df.withColumnRenamed("author_id", node_token_col)
    node_token_col = graph_token_node_col_name("Paper", 1, 0)
    author_past_paper_df = author_past_paper_df.withColumnRenamed("paper_id", node_token_col)
    author_past_paper_df = author_past_paper_df.withColumnRenamed("year", "graph_year")
    united_bipaths_df = united_bipaths_df.join(author_past_paper_df, on = author_past_paper_df.columns, how="leftanti")
    
    # 去除之前引用过的论文
    node_token_col = graph_token_node_col_name("Author", 0, 0)
    author_cited_paper_df = author_cited_paper_df.withColumnRenamed("author_id", node_token_col)
    node_token_col = graph_token_node_col_name("Paper", 1, 0)
    author_cited_paper_df = author_cited_paper_df.withColumnRenamed("cite_paper_id", node_token_col)
    author_cited_paper_df = author_cited_paper_df.withColumnRenamed("year", "graph_year")
    united_bipaths_df = united_bipaths_df.join(author_cited_paper_df, on = author_cited_paper_df.columns, how="leftanti")

    ####################################################################################################
    # 保留bipaths_union出现过的已知目标边作为正样本
    pos_samples_df = united_bipaths_df.join(samples_df, on = node_pair_id_cols, how = "inner")
    pos_samples_df.persist()
    
    # 从潜在的reliable node pairs中去除known target edges
    known_target_edges_df = pos_samples_df.select(*node_pair_id_cols)
    unknown_node_pair_df = united_bipaths_df.join(known_target_edges_df, on = node_pair_id_cols, how = "leftanti")
    unknown_node_pair_df.persist()
    
    # 为头节点在已知目标边中出现过的行添加标记并给出对应到的样本类型
    head_node_id_cols = []
    head_node_type = main_bipaths["query_nodes_types"][0]
    for node_col_i in range(len(graph.nodes[head_node_type]["node_col_types"])):
        node_token_col = graph_token_node_col_name(head_node_type, 0, node_col_i)
        head_node_id_cols.append(node_token_col)
    head_node_id_cols = head_node_id_cols + list(graph.graph_time_cols_alias)

    head_node_df = pos_samples_df.select(head_node_id_cols + ["sample_type"]).distinct()
    head_node_df = head_node_df.withColumn("label", lit(0))
    
    # 将头节点出现过且不是正样本的行且作为负样本，并用对应的样本类型
    head_neg_sample_df = unknown_node_pair_df.join(head_node_df, on = head_node_id_cols, how = "inner")

    # 为尾节点在已知目标边中出现过的行添加标记并给出对应到的样本类型
    tail_node_id_cols = []
    tail_node_type = main_bipaths["query_nodes_types"][1]
    for node_col_i in range(len(graph.nodes[tail_node_type]["node_col_types"])):
        node_token_col = graph_token_node_col_name(tail_node_type, 1, node_col_i)
        tail_node_id_cols.append(node_token_col)
    tail_node_id_cols = tail_node_id_cols + list(graph.graph_time_cols_alias)

    tail_node_df = pos_samples_df.select(tail_node_id_cols + ["sample_type"]).distinct()
    tail_node_df = tail_node_df.withColumn("label", lit(0))
    
    # 将尾结点出现过且不是正样本的行且作为负样本，并用对应的样本类型
    tail_neg_sample_df = unknown_node_pair_df.join(tail_node_df, on = tail_node_id_cols, how = "inner")

    # 合并为负样本，去重
    neg_sample_df = head_neg_sample_df.unionByName(tail_neg_sample_df).distinct()

    # 如果同一个节点对可能有多个sample type，随机保留一个，毕竟负样本很多，之后可以考虑按比例分配
    neg_sample_df = random_n_sample(spark, neg_sample_df, node_pair_id_cols, 1)
    
    # 将剩余的目标推理时间的作为推理样本
    conditions = [(col(c) == v) for c, v in zip(graph.graph_time_cols_alias, inference_time_cols_values)]
    infer_sample_df = unknown_node_pair_df.filter(*conditions)
    
    # 将标签列置为空
    infer_sample_df = infer_sample_df.withColumn("sample_type", lit("infer"))
    infer_sample_df = infer_sample_df.withColumn("label", lit(None))
    
    # 合并全部类型的样本，不去重了，负样本里面会有和推理样本重复的node pair，因为负样本是负采样得到的，不是真的负样本
    reliable_node_pairs_df = pos_samples_df.unionByName(neg_sample_df).unionByName(infer_sample_df)
    # reliable_node_pairs_df = reliable_node_pairs_df.groupBy(node_pair_id_cols + ["sample_type"])\
    #                                                .agg(max_('label').alias('label'))
    
    # 保存结果(目前以样本类型分区：train, valid, test, inference)
    pyspark_optimal_save(reliable_node_pairs_df, result_path, result_format, "overwrite", ["sample_type"])

    # 释放变量
    pos_samples_df.unpersist()
    unknown_node_pair_df.unpersist()
    
    # 重新读取结果
    reliable_node_pairs_df = pyspark_read_table(spark, result_path, result_format)

     # 显示各个类型的样本总数及正负样本数目
    sample_count = reliable_node_pairs_df.groupBy(["sample_type", "label"])                                         .agg(count_('*').alias('count'))                                         .collect()
    for row in sample_count:
        logger.info(f"{row['sample_type']} | label={row['label']} | count={row['count']}")

    # 创建要clt的paths的query_nodes
    clt_bipaths_query_config = {}
    clt_bipaths_query_config["graph_time_values"] = list(main_bipath_query_config["graph_time_values"])
    clt_bipaths_query_config["tgt_query_nodes"] = {}
    clt_bipaths_query_config["tgt_query_nodes"]["result_path"] = task_data_path
    clt_bipaths_query_config["tgt_query_nodes"]["df"] = reliable_node_pairs_df
    
    return clt_bipaths_query_config


# # 要收集到各个节点对的bipaths对应的配置

# In[9]:


from joinminer.graph import join_edges_init
from joinminer.graph import read_labeled_samples
from joinminer.graph import graph_token_node_col_name, standard_node_col_name

import copy

def bipaths_collection_config_init(spark, reliable_bipaths, join_edges_list_name, labeled_samples_config, join_edges_default_config):
    # 创建用于获得reliable_node_pairs的bipaths的配置
    # 大致对标join_edges_list_collect初始化后的配置
    clt_bipaths = {}
    clt_bipaths["join_edges_list_name"] = join_edges_list_name
    clt_bipaths["query_nodes_types"] = labeled_samples_config["nodes_types"]
    clt_bipaths["query_nodes_cols"] = []
    for query_node_i in range(len(clt_bipaths["query_nodes_types"])):
        # 获得该节点类型
        query_node_type = clt_bipaths["query_nodes_types"][query_node_i]
        
        for node_col_i in range(len(graph.nodes[query_node_type]["node_col_types"])):
            query_node_token_col = graph_token_node_col_name(query_node_type, query_node_i, node_col_i)
            clt_bipaths["query_nodes_cols"].append(query_node_token_col)
                
    clt_bipaths["join_edges_list_path"] = f"/user/mart_coo/mart_coo_innov/CompGraph/AMiner/join_edges_list/{join_edges_list_name}"
    clt_bipaths["join_edges_list_table_format"] = "parquet"

    # 设定collect过程中多少次join后persist一次结果
    clt_bipaths["join_edges_list_persist_interval"] = 3
    
    # 设定collect过程中保存中间结果的相关配置
    clt_bipaths["join_edges_list_mid"] = {}
    clt_bipaths["join_edges_list_mid"]["persist_interval_save_round"] = 3
    clt_bipaths["join_edges_list_mid"]["result_path"] = f"/user/mart_coo/mart_coo_innov/CompGraph/AMiner/join_edges_list_mid/{join_edges_list_name}"
    clt_bipaths["join_edges_list_mid"]["result_format"] = "parquet"
    
    # 因为总体路径较少，不用分成父路径一条条算了，直接完成整个路径的计算
    clt_bipaths["forward_paths_to_schema"] = {}
    clt_bipaths["backward_paths_to_schema"] = {}
    clt_bipaths["bipaths_list_schema"] = []
    for reliable_bipath in reliable_bipaths:
        clt_path_schema = {}
        clt_path_schema["join_edges_name"] = "bipaths_v1_bipath_full_" + reliable_bipath["join_edges_name"]

        # 初始化bipath相关配置
        bipath_config = {}
        bipath_config["join_edges_name"] = "bipaths_v1_bipath_full_" + reliable_bipath["join_edges_name"]
        bipath_config["join_edges_schema"] = copy.deepcopy(reliable_bipath["join_edges_schema"])
        
        # 在最后一个join_edge加入补全全部特征的配置
        bipath_config["join_edges_schema"][-1]["feature_add"] = {}
        bipath_config["join_edges_schema"][-1]["feature_add"]["add_type"] = "full"

        # 设定bipath最多collect 20个数据，目前不起作用
        bipath_config["collect_records_count"] = 20
            
        # 设定forward path query的query node配置
        query_nodes_config = {}
        query_nodes_config["query_nodes_types"] = [labeled_samples_config["nodes_types"][0]]
        query_nodes_config["query_nodes_indexes"] = [0]

        clt_path_schema["bipath_schema"] = join_edges_init(graph, query_nodes_config, bipath_config, join_edges_default_config)
        
        # 获得对应的forward_path名称
        forward_path_name = reliable_bipath["forward_path"]['name']

        # 记录对应的forward_path名称
        clt_path_schema["forward_path_name"] = forward_path_name
        
        # 检查是否已有forward_path对应配置
        if forward_path_name not in clt_bipaths["forward_paths_to_schema"]:
            # 初始化forward_path相关配置
            forward_path_config = {}
            forward_path_config["join_edges_name"] = "bipaths_v1_forward_full_" + forward_path_name
            forward_path_config["join_edges_schema"] = copy.deepcopy(reliable_bipath["forward_path_schema"])
    
            # 在最后一个join_edge加入补全全部特征的配置
            forward_path_config["join_edges_schema"][-1]["feature_add"] = {}
            forward_path_config["join_edges_schema"][-1]["feature_add"]["add_type"] = "full"

            # 设定forward_path最多collect 20个数据
            forward_path_config["collect_records_count"] = 20
            
            # 设定forward path query的query node配置
            query_nodes_config = {}
            query_nodes_config["query_nodes_types"] = [labeled_samples_config["nodes_types"][0]]
            query_nodes_config["query_nodes_indexes"] = [0]

            clt_bipaths["forward_paths_to_schema"][forward_path_name] = join_edges_init(graph, query_nodes_config, 
                                                                                        forward_path_config, join_edges_default_config)
        
        # 检查是否有backward_path
        if "backward_path" in reliable_bipath:
            # 获得对应的forward_path名称
            backward_path_name = reliable_bipath["backward_path"]['name']

            # 记录对应的forward_path名称
            clt_path_schema["backward_path_name"] = backward_path_name

            # 检查是否已有backward_path对应配置
            if backward_path_name not in clt_bipaths["backward_paths_to_schema"]:
                # 初始化backward_path相关配置
                backward_path_config = {}
                backward_path_config["join_edges_name"] = "bipaths_v1_backward_full_" + reliable_bipath["backward_path"]['name']
                backward_path_config["join_edges_schema"] = copy.deepcopy(reliable_bipath["backward_path_schema"])
            
                # 在最后一个join_edge加入补全全部特征的配置
                backward_path_config["join_edges_schema"][-1]["feature_add"] = {}
                backward_path_config["join_edges_schema"][-1]["feature_add"]["add_type"] = "full"

                # 设定backward_path最多collect 20个数据
                backward_path_config["collect_records_count"] = 20
                
                # 设定backward path query的query node配置
                query_nodes_config = {}
                query_nodes_config["query_nodes_types"] = [labeled_samples_config["nodes_types"][1]]
                query_nodes_config["query_nodes_indexes"] = [0]
                
                clt_bipaths["backward_paths_to_schema"][backward_path_name] = join_edges_init(graph, query_nodes_config, 
                                                                                              backward_path_config, 
                                                                                              join_edges_default_config)

            # 记录要转化列名的特征列 
            clt_path_schema["backward_path_feat_col_alias"] = list(reliable_bipath["backward_path_feat_col_alias"])
            
            # 加入tail_path的select_col_alias
            clt_path_schema["backward_path_select_col_alias"] = (reliable_bipath["backward_path_id_col_alias"] + 
                                                                 reliable_bipath["backward_path_feat_col_alias"])
            
            # 记录tail_path用于join的node_cols 
            clt_path_schema["backward_path_join_node_cols"] = list(reliable_bipath["backward_path_join_node_cols"])

        # 记录联结后的路径包含的特征列对应的维度
        clt_path_schema["feat_cols_sizes"] = copy.deepcopy(reliable_bipath["feat_cols_sizes"])
        
        # 设定bipath最多collect 20个数据
        clt_path_schema["collect_records_count"] = 20
        
        # 获得query nodes对应的path_indexes
        query_nodes_path_indexes = [reliable_bipath["join_edges_schema"][0]["join_nodes_indexes"][0], 
                                    reliable_bipath["join_edges_schema"][-1]["add_nodes_indexes"][0]]
        
        # 获得最终要保留的query_node_id_col_aliases
        clt_path_schema["query_nodes_col_rename_dict"] = {}
        for query_node_i in range(len(clt_bipaths["query_nodes_types"])):
            # 获得该节点类型
            query_node_type = clt_bipaths["query_nodes_types"][query_node_i]
            
            # 获得该node在path上对应的index
            node_path_index = query_nodes_path_indexes[query_node_i]
            
            for node_col_i in range(len(graph.nodes[query_node_type]["node_col_types"])):
                query_node_path_col = standard_node_col_name(query_node_type, node_path_index, node_col_i)
                query_node_token_col = graph_token_node_col_name(query_node_type, query_node_i, node_col_i)
                clt_path_schema["query_nodes_col_rename_dict"][query_node_path_col] = query_node_token_col
        
        clt_bipaths["bipaths_list_schema"].append(clt_path_schema)

    return clt_bipaths

# 生成bipaths_union所需相关配置
clt_path_count = 30
clt_bipaths = spark_runner.run(bipaths_collection_config_init, sorted_relevant_paths_config[:clt_path_count], 
                               f"top_{clt_path_count}_reliable_bipaths_collection",
                               labeled_samples_config, join_edges_default_config)


# # 按配置收集bipaths准备为数据集

# In[10]:


from joinminer.graph import join_edges_query, graph_token_query
from joinminer.graph import standard_node_col_name, standard_feat_col_name
from joinminer.hdfs import hdfs_check_partitions, hdfs_list_contents, hdfs_delete_dir
from joinminer.hdfs import hdfs_check_file_exists, hdfs_save_string
from joinminer.pyspark import pyspark_read_table, pyspark_optimal_save, fill_null_vectors

from functools import reduce
from pyspark.sql.functions import when, col, lit, rand, row_number
from pyspark.sql.functions import first as first_, max as max_
from pyspark.sql.functions import collect_list
from pyspark.sql.window import Window
from pyspark.ml.functions import vector_to_array

def bipaths_tgt_collections(spark, graph, bipaths, query_config, tgt_collections_path):
    # 获得这些bipaths对应的基础信息
    join_edges_list_name = bipaths["join_edges_list_name"]
    query_nodes_types = bipaths["query_nodes_types"]
    query_nodes_cols = bipaths["query_nodes_cols"]
    
    logger.info(f"Collect join_edges_list {join_edges_list_name} for query nodes types "
                f"{query_nodes_types} of cols {query_nodes_cols}")

    # 获得query nodes对应的id列
    query_nodes_id_cols = query_nodes_cols + list(graph.graph_time_cols_alias)

    # 获得目标结果的存储路径
    # 一定有目标节点，设定结果路径为目标节点结果目录下对应路径文件夹
    result_path = tgt_collections_path
        
    # 获得结果对应的存储格式
    result_format = bipaths["join_edges_list_table_format"]

    logger.info(f"The reulst will be output to: {result_path} in {result_format} format.")

    # 获得结果对应的分区列
    partition_cols = list(graph.graph_time_cols_alias)

    # 获得结果对应的目标分区值
    partition_cols_values = query_config["graph_time_values"]

    logger.info(f"The graph data include time columns {partition_cols} and values {partition_cols_values}.")

    # 如果已有对应的全量结果，则直接返回
    if hdfs_check_file_exists(result_path + "/_DATASET_SUCCESS"):
        return
    
    # 记录要collect的各组id列对应的path_df
    clt_id_to_paths = []

    # 记录collect到的特征列的长度
    clt_feat_cols_sizes = {}

    # 如果要开始计算，先以未完成的时间更新query_config,直接就是partition_cols_values
    missing_values = partition_cols_values
    logger.info(f"Missing target partitions: {missing_values}")

    # 获得forward_path对应的全部目标点
    forward_path_node_id_cols = []
    forward_path_node_type = query_nodes_types[0]
    for node_col_i in range(len(graph.nodes[forward_path_node_type]["node_col_types"])):
        node_token_col = graph_token_node_col_name(forward_path_node_type, 0, node_col_i)
        forward_path_node_id_cols.append(node_token_col)
    for time_col in graph.graph_time_cols_alias:
        forward_path_node_id_cols.append(time_col)
        
    forward_path_id_df = query_config["tgt_query_nodes"]["df"].select(forward_path_node_id_cols).distinct().persist()

    # 获得forward_path的query node的query配置
    forward_path_node_query_config = {}
    forward_path_node_query_config["graph_time_values"] = missing_values

    # 获得forward_path的query_nodes本身对应的特征 
    # 之后合并的时候可以和forward_path按一样的方式join进去
    forward_path_node_df = graph_token_query(spark, graph, "node", forward_path_node_type, 
                                             forward_path_node_query_config)

    # 要使用的列及别名
    select_cols = []
    for node_col_i in range(len(graph.nodes[forward_path_node_type]["node_col_types"])):
        node_token_col = graph_token_node_col_name(forward_path_node_type, None, node_col_i)
        node_path_col = graph_token_node_col_name(forward_path_node_type, 0, node_col_i)
        select_cols.append(col(node_token_col).alias(node_path_col))
        
    query_node_token_feat_col = graph.nodes[forward_path_node_type]["query_config"]["assembled_feat_col"]
    query_node_feat_col = f"query_node_{forward_path_node_type}_index_0_feat"
    select_cols.append(col(query_node_token_feat_col).alias(query_node_feat_col))

    for time_col in graph.graph_time_cols_alias:
        select_cols.append(col(time_col).alias(time_col))
    
    # 修正列名
    forward_path_node_df = forward_path_node_df.select(*select_cols)

    # 只保留目标点对应的结果
    forward_path_node_df = forward_path_id_df.join(forward_path_node_df, on = forward_path_node_id_cols, how = "left")
    
    # 记录新增的特征列和对应的向量长度
    clt_feat_cols_sizes[query_node_feat_col] = len(graph.nodes[forward_path_node_type]["graph_token_feat_cols"])

    # 补全空向量 
    forward_path_node_df = fill_null_vectors(spark, forward_path_node_df, query_node_feat_col, 
                                             len(graph.nodes[forward_path_node_type]["graph_token_feat_cols"]))

    # 将特征向量列转化为array格式
    # 临时方案，之后还是应该改成graph_token_query结果内完成，以后只有涉及到聚合之类的操作再临时用vector向量
    forward_path_node_df = forward_path_node_df.withColumn(query_node_feat_col, vector_to_array(col(query_node_feat_col)))

    # 记录query_node最终结果
    clt_id_to_paths.append([forward_path_node_id_cols, forward_path_node_df])

    # 设定forward_path_query_nodes_df
    col_rename_dict = {}
    for node_col_i in range(len(graph.nodes[forward_path_node_type]["node_col_types"])):
        node_path_col = graph_token_node_col_name(forward_path_node_type, 0, node_col_i)
        node_token_col = standard_node_col_name(forward_path_node_type, 0, node_col_i)
        col_rename_dict[node_path_col] = node_token_col
    
    forward_path_query_nodes_df = forward_path_id_df.select([col(c).alias(col_rename_dict.get(c, c)) for c in forward_path_id_df.columns])

    # 获得forward_path的query配置
    forward_path_query_config = {}
    forward_path_query_config["graph_time_values"] = missing_values
    forward_path_query_config["tgt_query_nodes"]= {}
    forward_path_query_config["tgt_query_nodes"]["result_path"] = query_config["tgt_query_nodes"]["result_path"]
    forward_path_query_config["tgt_query_nodes"]["df"] = forward_path_query_nodes_df

    # 依次获得有关联的forward_path对应的clt数据
    for forward_path_name in bipaths["forward_paths_to_schema"]:
        # 获得forward_path对应配置
        forward_path = bipaths["forward_paths_to_schema"][forward_path_name]

        # 获得clt到的列要添加的前缀
        clt_col_prefix = forward_path['name']
        
        # 获得该forward_path的collection结果路径
        clt_forward_path_result_path = query_config["tgt_query_nodes"]["result_path"] + f"/join_edges_collection/{forward_path['name']}"

        # 获得该join_edges最多的结果数
        collect_records_count = forward_path["collect_records_count"]
        
        # 获得会collect到的特征列
        for collect_col in forward_path["query_config"]["feat_cols_sizes"]:
            new_col_name = f"{clt_col_prefix}_clt_{collect_col}"
            clt_feat_cols_sizes[new_col_name] = int(forward_path["query_config"]["feat_cols_sizes"][collect_col] * collect_records_count / 4)
        
        # 检查是否已有对应结果
        clt_is_complete, clt_missing_values = hdfs_check_partitions(clt_forward_path_result_path, partition_cols, missing_values)
        if not clt_is_complete:
            # 获得对应的数据
            forward_path_query_config["graph_time_values"] = clt_missing_values
            forward_path_df = join_edges_query(spark, graph, forward_path, forward_path_query_config)
    
            # 修正列名为collect结果所需格式
            col_rename_dict = {}
            for node_col_i in range(len(graph.nodes[forward_path_node_type]["node_col_types"])):
                node_token_col = standard_node_col_name(forward_path_node_type, 0, node_col_i)
                node_path_col = graph_token_node_col_name(forward_path_node_type, 0, node_col_i)
                col_rename_dict[node_token_col] = node_path_col
            
            forward_path_df = forward_path_df.select([col(c).alias(col_rename_dict.get(c, c)) for c in forward_path_df.columns])
            
            # 为相同的query node的数据加上join_edge来源和在该来源中的序号
            # 并只保留最大的结果数以内的序号对应的数据
            window_spec = Window.partitionBy(*forward_path_node_id_cols).orderBy(rand())
            forward_path_df = forward_path_df.withColumn("collect_id", row_number().over(window_spec))                                              .filter(col("collect_id") <= collect_records_count)

            # 将特征向量列都转化为array格式
            # 临时方案，之后还是应该改成join_edges_query结果后完成，以后只有涉及到聚合之类的操作再临时用vector向量
            for path_feat_col in forward_path["query_config"]["feat_cols_sizes"]:
                forward_path_df = forward_path_df.withColumn(path_feat_col, vector_to_array(col(path_feat_col)))
    
            # 获得为collect各组数据所需的聚合配置
            agg_exprs = []
    
            # 记录各个query_nodes collect的数目
            agg_expr = max_("collect_id").alias(f"{clt_col_prefix}_collect_count")
            agg_exprs.append(agg_expr)
    
            # 依次处理各个collect到的实例的信息
            for collect_col in forward_path_df.columns:
                # 跳过id列
                if collect_col in forward_path_node_id_cols + ["collect_id"]:
                    continue
                    
                new_col_name = f"{clt_col_prefix}_clt_{collect_col}"
                agg_expr = collect_list(col(collect_col)).alias(new_col_name)

                agg_exprs.append(agg_expr)

                # 添加剩余id列对应的空间
                if new_col_name not in clt_feat_cols_sizes:
                    clt_feat_cols_sizes[new_col_name] = int(50 * collect_records_count / 4)
                    
            # 将相同query node的数据collect到一行，并修正列名
            clt_forward_path_df = forward_path_df.groupBy(forward_path_node_id_cols).agg(*agg_exprs)

            # 对collect_count列的None值补0
            clt_forward_path_df = clt_forward_path_df.fillna(0, subset=[f"{clt_col_prefix}_collect_count"])
            
            # 保存结果  
            pyspark_optimal_save(clt_forward_path_df, clt_forward_path_result_path, "parquet", "overwrite", partition_cols,
                                 col_sizes = clt_feat_cols_sizes)
        
        # 读取对应结果
        clt_forward_path_df = pyspark_read_table(spark, clt_forward_path_result_path, "parquet", partition_cols, missing_values)
        
        # 记录forward_path_df最终结果
        clt_id_to_paths.append([forward_path_node_id_cols, clt_forward_path_df])

    # 获得backward_path对应的全部目标点
    backward_path_node_id_cols = []
    backward_path_node_type = query_nodes_types[1]
    for node_col_i in range(len(graph.nodes[backward_path_node_type]["node_col_types"])):
        node_token_col = graph_token_node_col_name(backward_path_node_type, 1, node_col_i)
        backward_path_node_id_cols.append(node_token_col)
    for time_col in graph.graph_time_cols_alias:
        backward_path_node_id_cols.append(time_col)
        
    backward_path_id_df = query_config["tgt_query_nodes"]["df"].select(*backward_path_node_id_cols).distinct().persist()

    # 获得backward_path的query node的query配置
    backward_path_node_query_config = {}
    backward_path_node_query_config["graph_time_values"] = missing_values

    # 获得backward_path的query_nodes本身对应的特征 
    # 之后合并的时候可以和backward_path按一样的方式join进去
    backward_path_node_df = graph_token_query(spark, graph, "node", backward_path_node_type, 
                                             backward_path_node_query_config)

    # 要使用的列及别名
    select_cols = []
    for node_col_i in range(len(graph.nodes[backward_path_node_type]["node_col_types"])):
        node_token_col = graph_token_node_col_name(backward_path_node_type, None, node_col_i)
        node_path_col = graph_token_node_col_name(backward_path_node_type, 1, node_col_i)
        select_cols.append(col(node_token_col).alias(node_path_col))
        
    query_node_token_feat_col = graph.nodes[backward_path_node_type]["query_config"]["assembled_feat_col"]
    query_node_feat_col = f"query_node_{backward_path_node_type}_index_1_feat"
    select_cols.append(col(query_node_token_feat_col).alias(query_node_feat_col))

    for time_col in graph.graph_time_cols_alias:
        select_cols.append(col(time_col).alias(time_col))
    
    # 修正列名
    backward_path_node_df = backward_path_node_df.select(*select_cols)

    # 只保留目标点对应的结果
    backward_path_node_df = backward_path_id_df.join(backward_path_node_df, on = backward_path_node_id_cols, how = "left")
    
    # 记录新增的特征列和对应的向量长度
    clt_feat_cols_sizes[query_node_feat_col] = len(graph.nodes[backward_path_node_type]["graph_token_feat_cols"])

    # 补全空向量 
    backward_path_node_df = fill_null_vectors(spark, backward_path_node_df, query_node_feat_col, 
                                              len(graph.nodes[backward_path_node_type]["graph_token_feat_cols"]))

    # 将特征向量列转化为array格式
    # 临时方案，之后还是应该改成graph_token_query结果内完成，以后只有涉及到聚合之类的操作再临时用vector向量
    backward_path_node_df = backward_path_node_df.withColumn(query_node_feat_col, vector_to_array(col(query_node_feat_col)))

    # 记录要collect的backward_path_node_df
    clt_id_to_paths.append([backward_path_node_id_cols, backward_path_node_df])

    # 设定backward_path_query_nodes_df
    col_rename_dict = {}
    for node_col_i in range(len(graph.nodes[backward_path_node_type]["node_col_types"])):
        node_path_col = graph_token_node_col_name(backward_path_node_type, 1, node_col_i)
        node_token_col = standard_node_col_name(backward_path_node_type, 0, node_col_i)
        col_rename_dict[node_path_col] = node_token_col
    
    backward_path_query_nodes_df = backward_path_id_df.select([col(c).alias(col_rename_dict.get(c, c)) for c in backward_path_id_df.columns])

    # 获得backward_path的query nodes
    backward_path_query_config = {}
    backward_path_query_config["graph_time_values"] = missing_values
    backward_path_query_config["tgt_query_nodes"]= {}
    backward_path_query_config["tgt_query_nodes"]["result_path"] = query_config["tgt_query_nodes"]["result_path"]
    backward_path_query_config["tgt_query_nodes"]["df"] = backward_path_query_nodes_df

    # 依次获得backward_path对应的数据
    clt_backward_paths = []
    for backward_path_name in bipaths["backward_paths_to_schema"]:
        # 获得backward_path对应配置
        backward_path = bipaths["backward_paths_to_schema"][backward_path_name]

        # 获得clt到的列要添加的前缀
        clt_col_prefix = backward_path['name']
        
        # 获得该backward_path的collection结果路径
        clt_backward_path_result_path = query_config["tgt_query_nodes"]["result_path"] + f"/join_edges_collection/{backward_path['name']}"

        # 获得该join_edges最多的结果数
        collect_records_count = backward_path["collect_records_count"]
        
        # 获得会collect到的特征列
        for collect_col in backward_path["query_config"]["feat_cols_sizes"]:
            new_col_name = f"{clt_col_prefix}_clt_{collect_col}"
            clt_feat_cols_sizes[new_col_name] = int(backward_path["query_config"]["feat_cols_sizes"][collect_col] * collect_records_count / 4)

        # 检查是否已有对应结果
        clt_is_complete, clt_missing_values = hdfs_check_partitions(clt_backward_path_result_path, partition_cols, missing_values)
        if not clt_is_complete:
            # 获得对应的数据
            backward_path_query_config["graph_time_values"] = clt_missing_values
            backward_path_df = join_edges_query(spark, graph, backward_path, backward_path_query_config)
    
            # 修正列名为collect结果所需格式
            col_rename_dict = {}
            for node_col_i in range(len(graph.nodes[backward_path_node_type]["node_col_types"])):
                node_token_col = standard_node_col_name(backward_path_node_type, 0, node_col_i)
                node_path_col = graph_token_node_col_name(backward_path_node_type, 1, node_col_i)
                col_rename_dict[node_token_col] = node_path_col
            
            backward_path_df = backward_path_df.select([col(c).alias(col_rename_dict.get(c, c)) for c in backward_path_df.columns])
            
            # 为相同的query node的数据加上join_edge来源和在该来源中的序号
            # 并只保留最大的结果数以内的序号对应的数据
            window_spec = Window.partitionBy(*backward_path_node_id_cols).orderBy(rand())
            backward_path_df = backward_path_df.withColumn("collect_id", row_number().over(window_spec))                                                .filter(col("collect_id") <= collect_records_count)

            # 将特征向量列都转化为array格式
            # 临时方案，之后还是应该改成join_edges_query结果后完成，以后只有涉及到聚合之类的操作再临时用vector向量
            for path_feat_col in backward_path["query_config"]["feat_cols_sizes"]:
                backward_path_df = backward_path_df.withColumn(path_feat_col, vector_to_array(col(path_feat_col)))
            
            # 获得为collect各组数据所需的聚合配置
            agg_exprs = []
    
            # 记录各个query_nodes collect的数目
            agg_expr = max_("collect_id").alias(f"{clt_col_prefix}_collect_count")
            agg_exprs.append(agg_expr)

            # 依次处理各个collect到的实例的信息
            for collect_col in backward_path_df.columns:
                # 跳过分组id列
                if collect_col in backward_path_node_id_cols + ["collect_id"]:
                    continue
                
                new_col_name = f"{clt_col_prefix}_clt_{collect_col}"
                agg_expr = collect_list(col(collect_col)).alias(new_col_name)

                agg_exprs.append(agg_expr)

                # 添加剩余id列对应的空间
                if new_col_name not in clt_feat_cols_sizes:
                    clt_feat_cols_sizes[new_col_name] = int(50 * collect_records_count / 4)
            
            # 将相同query node的数据collect到一行，并修正列名
            clt_backward_path_df = backward_path_df.groupBy(backward_path_node_id_cols).agg(*agg_exprs)

            # 对collect_count列的None值补0
            clt_backward_path_df = clt_backward_path_df.fillna(0, subset=[f"{clt_col_prefix}_collect_count"])
            
            # 保存结果  
            pyspark_optimal_save(clt_backward_path_df, clt_backward_path_result_path, "parquet", "overwrite", partition_cols,
                                 col_sizes = clt_feat_cols_sizes)
        
        # 读取对应结果
        clt_backward_path_df = pyspark_read_table(spark, clt_backward_path_result_path, "parquet", partition_cols, missing_values)
        
        # 记录要collect的backward_path_df
        clt_id_to_paths.append([backward_path_node_id_cols, clt_backward_path_df])

    # 释放backward_path的query nodes
    backward_path_query_config["tgt_query_nodes"]["df"].unpersist()

    # 获得bipath的目标点
    bipath_id_df = query_config["tgt_query_nodes"]["df"].select(*query_nodes_id_cols).distinct().persist()
        
    # 依次获得各个bipath经过collect获得的数据 
    for bipath_schema in bipaths["bipaths_list_schema"]:
        # 根据该bipath的对应配置获得目标点列名的对应关系
        col_rename_dict = {}
        for node_path_col in bipath_schema["query_nodes_col_rename_dict"]:
            node_token_col = bipath_schema["query_nodes_col_rename_dict"][node_path_col]
            col_rename_dict[node_token_col] = node_path_col

        # 获得bipath_query_nodes_df
        bipath_query_nodes_df = bipath_id_df.select([col(c).alias(col_rename_dict.get(c, c)) for c in bipath_id_df.columns])
        
        # 获得该bipath对应的名称
        bipath_name = bipath_schema["join_edges_name"]

        # 获得clt到的列要添加的前缀
        clt_col_prefix = bipath_name
        
        # 获得对应的forward_path名称
        forward_path_name = bipath_schema["forward_path_name"]
            
        # 获得forward_path对应配置
        forward_path = bipaths["forward_paths_to_schema"][forward_path_name]

        # 先将forward_path的feat_col对应的长度作为bipath的初始特征长度
        bipath_feat_cols_sizes = copy.deepcopy(forward_path["query_config"]["feat_cols_sizes"])
        
        # 检查是否有对应的backward_path
        if "backward_path_name" in bipath_schema:
            # 获得对应的backward_path名称
            backward_path_name = bipath_schema["backward_path_name"]
            
            # 获得backward_path对应配置
            backward_path = bipaths["backward_paths_to_schema"][backward_path_name]

            # 添加特征列对应的长度 
            for feat_col, feat_col_alias in bipath_schema["backward_path_feat_col_alias"]:
                # 添加特征列对应的长度
                bipath_feat_cols_sizes[feat_col_alias] = backward_path["query_config"]["feat_cols_sizes"][feat_col]

        # 获得该bipath的collection结果路径
        clt_bipath_result_path = query_config["tgt_query_nodes"]["result_path"] + f"/join_edges_collection/{bipath_name}"

        # 获得该join_edges最多的结果数
        collect_records_count = bipath_schema["collect_records_count"]

        # 获得会collect到的特征列
        for collect_col in bipath_feat_cols_sizes:
            new_col_name = f"{clt_col_prefix}_clt_{collect_col}"
            clt_feat_cols_sizes[new_col_name] = int(bipath_feat_cols_sizes[collect_col] * collect_records_count / 4)

        # 检查是否已有对应结果
        clt_is_complete, clt_missing_values = hdfs_check_partitions(clt_bipath_result_path, partition_cols, missing_values)
        if not clt_is_complete:
            # 获得对应结果的存储位置和存储格式
            bipath_result_path = query_config["tgt_query_nodes"]["result_path"] + f"/join_edges/{bipath_name}"
            bipath_result_format = "parquet"
            
            # 检查是否已有对应结果
            is_complete, bipath_missing_values = hdfs_check_partitions(bipath_result_path, partition_cols, clt_missing_values)
    
            # 如果未完成
            if not is_complete:
                # 没有就开始运算, 获得对应的目标数据
                forward_path_query_config["graph_time_values"] = bipath_missing_values
                bipath_df = join_edges_query(spark, graph, forward_path, forward_path_query_config)
                
                # 检查是否有对应的backward_path
                if "backward_path_name" in bipath_schema:
                    # 获得对应的数据
                    backward_path_query_config["graph_time_values"] = bipath_missing_values
                    backward_path_df = join_edges_query(spark, graph, backward_path, backward_path_query_config)
        
                    # 转换列名  
                    select_cols = [col(column).alias(alias) for column, alias in bipath_schema["backward_path_select_col_alias"]]
                    backward_path_df = backward_path_df.select(*select_cols)
                    
                    # 和forward_path联结
                    backward_path_join_id_cols = bipath_schema["backward_path_join_node_cols"] + graph.graph_time_cols_alias
                    bipath_df = bipath_df.join(backward_path_df, on = backward_path_join_id_cols, how = "inner")

                # 应该只保留终止点能匹配上目标点的数据，待优化  
                bipath_df = bipath_df.join(bipath_query_nodes_df, on = bipath_query_nodes_df.columns, how = "inner")
                
                # 可以加一个采样函数，防止过多数据
                
                # 保存结果  
                pyspark_optimal_save(bipath_df, bipath_result_path, bipath_result_format, "overwrite", partition_cols,
                                     col_sizes = bipath_feat_cols_sizes)
    
            # 重新读取完整结果 
            bipath_df = pyspark_read_table(spark, bipath_result_path, bipath_result_format, partition_cols, clt_missing_values)
            
            # 修正列名
            col_rename_dict = bipath_schema["query_nodes_col_rename_dict"]
            bipath_df = bipath_df.select([col(c).alias(col_rename_dict.get(c, c)) for c in bipath_df.columns])
            
            # 为相同的query node的数据加上join_edge来源和在该来源中的序号
            # 并只保留最大的结果数以内的序号对应的数据
            window_spec = Window.partitionBy(*query_nodes_id_cols).orderBy(rand())
            bipath_df = bipath_df.withColumn("collect_id", row_number().over(window_spec))                                  .filter(col("collect_id") <= collect_records_count)

            # 将特征向量列都转化为array格式
            # 临时方案，之后还是应该改成join_edges_query结果后完成，以后只有涉及到聚合之类的操作再临时用vector向量
            for path_feat_col in bipath_feat_cols_sizes:
                bipath_df = bipath_df.withColumn(path_feat_col, vector_to_array(col(path_feat_col)))
            
            # 获得为collect各组数据所需的聚合配置
            agg_exprs = []
    
            # 先记录总的收集的信息的数目
            agg_expr = max_("collect_id").alias(f"{clt_col_prefix}_collect_count")
            agg_exprs.append(agg_expr)

            # 依次处理各个collect到的实例的信息
            for collect_col in bipath_df.columns:
                # 跳过id列
                if collect_col in query_nodes_id_cols + ["collect_id"]:
                    continue
                
                new_col_name = f"{clt_col_prefix}_clt_{collect_col}"
                agg_expr = collect_list(col(collect_col)).alias(new_col_name)

                agg_exprs.append(agg_expr)

                # 添加剩余id列对应的空间
                if new_col_name not in clt_feat_cols_sizes:
                    clt_feat_cols_sizes[new_col_name] = int(50 * collect_records_count / 4)
            
            # 将相同query node的数据collect到一行，并修正列名
            clt_bipath_df = bipath_df.groupBy(query_nodes_id_cols).agg(*agg_exprs)

            # 对collect_count列的None值补0
            clt_bipath_df = clt_bipath_df.fillna(0, subset=[f"{clt_col_prefix}_collect_count"])
            
            # 保存结果  
            pyspark_optimal_save(clt_bipath_df, clt_bipath_result_path, "parquet", "overwrite", partition_cols,
                                 col_sizes = clt_feat_cols_sizes)
        
        # 读取对应结果
        clt_bipath_df = pyspark_read_table(spark, clt_bipath_result_path, "parquet", partition_cols, missing_values)
        
        # 记录最终的bipath_df
        clt_id_to_paths.append([query_nodes_id_cols, clt_bipath_df])

    # 设定一次性最多合并多少个目标边的join_edges_list，可以变成一个batch多大，一次性最多处理多少个batch
    clt_query_nodes_batch_count = 50000000
    
    # 获得分批处理的中间结果的存储位置
    bipaths_batch_result_path = query_config["tgt_query_nodes"]["result_path"] + f"/join_edges_list_batch/{join_edges_list_name}"

    # 获得对目标点的分批结果的存储位置
    bipaths_batch_query_nodes_result_path = bipaths_batch_result_path + "/query_nodes"
    
    # 检查是否已有对query_nodes的分批结果
    batch_query_nodes_is_complete, _ = hdfs_check_partitions(bipaths_batch_query_nodes_result_path)

    # 如果没有分组结果
    if not batch_query_nodes_is_complete:
        # 基于batch_max_rows获得要分出的batch数目
        clt_batch_count = query_config["tgt_query_nodes"]["df"].count() // clt_query_nodes_batch_count + 1
        
        # 为各行随机添加所属的batch_id
        batch_clt_query_nodes_df = query_config["tgt_query_nodes"]["df"].withColumn("batch_id", (rand() * lit(clt_batch_count)).cast("int"))

        # 保存结果，并按batch_id分区
        pyspark_optimal_save(batch_clt_query_nodes_df, bipaths_batch_query_nodes_result_path, "parquet", "overwrite", ["batch_id"])

    # 获得全部的batch_id对应的存储路径 
    clt_query_nodes_batch_ids =  [int(x.split("batch_id=")[1]) for x in hdfs_list_contents(bipaths_batch_query_nodes_result_path, "directories")]

    # 按batch_id由小到大排序
    clt_query_nodes_batch_ids.sort()
    
    # 依次处理各个batch的数据
    batch_clt_bipaths_df_list = []
    for batch_id in clt_query_nodes_batch_ids:
        logger.info(f"Start collecting path_df for {batch_id}-th batch.")
        
        # 获得该batch最终clt结果对应的路径
        bipaths_batch_id_clt_result_path = result_path + f"/batch_id={batch_id}"
        
        # 检查该batch对应的结果是否已经转化完成
        if not hdfs_check_file_exists(bipaths_batch_id_clt_result_path + f"/_SUCCESS"):
            # 设定该batch join多组join_edges时中间结果的存储位置
            bipaths_mid_result_path = bipaths_batch_result_path + f"/clt_mid_result/batch_id={batch_id}"
            bipaths_mid_result_format = bipaths["join_edges_list_mid"]["result_format"]
    
            # 获得中间结果存储文件夹内的文件夹名称，这些文件夹名对应到collect的数目
            mid_subresult_paths = []
            if hdfs_check_file_exists(bipaths_mid_result_path):
                for mid_subresult_path in hdfs_list_contents(bipaths_mid_result_path, "directories"):
                    # 依次检查各个对应的子结果路径下是否已有完成的数据
                    is_complete, _ = hdfs_check_partitions(mid_subresult_path)
                    
                    if is_complete:
                        mid_subresult_paths.append(mid_subresult_path)
        
            # 检查是否已有中间结果
            if len(mid_subresult_paths) == 0:
                # 没有结果就从第一组要collect的数据开始 
                collected_df_count = 0
        
                # 以query_df作为初始值
                clt_bipaths_df = pyspark_read_table(spark, bipaths_batch_query_nodes_result_path + f"/batch_id={batch_id}", "parquet")
            else:
                # 将中间结果按序号排列
                mid_subresult_paths = sorted(mid_subresult_paths, key = lambda x: int(x.split("/")[-1]), reverse=True)
                
                # 从最大序号的下一位开始处理
                collected_df_count = int(mid_subresult_paths[0].split('/')[-1]) + 1
                
                # 读取最大序号的结果作为计算的起点
                clt_bipaths_df = pyspark_read_table(spark, mid_subresult_paths[0], bipaths_mid_result_format)
            
            logger.info(f"Start collecting path_df from {collected_df_count}-th df.")

            # 获得join几组后persist一下结果
            persist_interval = bipaths["join_edges_list_persist_interval"]
        
            # 获得persist几组后保存一次数据
            persist_interval_save_round = bipaths["join_edges_list_mid"]["persist_interval_save_round"]
        
            # 合并全部种类collect到的数据(每计算一组数据用persist保存下)
            persisted_clt_bipaths_df = None
            for clt_df_i in range(collected_df_count, len(clt_id_to_paths)):
                # 通过join添加数据
                clt_id_cols = clt_id_to_paths[clt_df_i][0]
                clt_path_df = clt_id_to_paths[clt_df_i][1]
                clt_bipaths_df = clt_bipaths_df.join(clt_path_df, on = clt_id_cols, how = "left")

                # 得给collect_count列补0
                
                # 获得当前相对于初始状态新增了多少df
                add_df_count = clt_df_i - collected_df_count + 1
                
                # 完成一定数量后persist结果
                if clt_df_i < (len(clt_id_to_paths) - 1) and add_df_count % persist_interval == 0:
                    # 要能够完成一定数量的结果后保存中间结果
                    if add_df_count % (persist_interval * persist_interval_save_round) == 0:
                        # 获得结果保存路径
                        mid_subresult_path = bipaths_mid_result_path + f"/{clt_df_i}"
        
                        # 保存结果
                        # pyspark_optimal_save(clt_bipaths_df, mid_subresult_path, bipaths_mid_result_format, "overwrite")
                        clt_bipaths_df.write.format(bipaths_mid_result_format).mode("overwrite").save(mid_subresult_path)
                        
                        # 释放之前persist的数据
                        if persisted_clt_bipaths_df is not None:
                            persisted_clt_bipaths_df.unpersist()
                            persisted_clt_bipaths_df = None
                            
                        # 重新读取保存的结果
                        clt_bipaths_df = pyspark_read_table(spark, mid_subresult_path, bipaths_mid_result_format)
                        
                    else:
                        clt_bipaths_df.persist()

                        clt_bipaths_df_count = clt_bipaths_df.count()
                        clt_bipaths_df_rows = len(clt_bipaths_df.columns)
                        logger.info(f"{clt_df_i}-th clt_bipaths_df count {clt_bipaths_df_count} rows {clt_bipaths_df_rows}")

                        if persisted_clt_bipaths_df is not None:
                            persisted_clt_bipaths_df.unpersist()
                        persisted_clt_bipaths_df = clt_bipaths_df
                    
            # 保存全部结果   
            pyspark_optimal_save(clt_bipaths_df, bipaths_batch_id_clt_result_path, result_format, "overwrite", 
                                 ["sample_type"], col_sizes = clt_feat_cols_sizes)
            # clt_bipaths_df.coalesce(800).write.format(result_format).mode("overwrite").partitionBy(*partition_cols).save(bipaths_batch_id_clt_result_path)
        
            # 释放之前persist的数据
            if persisted_clt_bipaths_df is not None:
                persisted_clt_bipaths_df.unpersist()
                persisted_clt_bipaths_df = None
                
            # 删除中间结果文件
            if hdfs_check_file_exists(bipaths_mid_result_path):
                hdfs_delete_dir(bipaths_mid_result_path)
            
    # 删除中间结果文件
    if hdfs_check_file_exists(bipaths_batch_result_path):
        hdfs_delete_dir(bipaths_batch_result_path)

    # 添加_SUCCESS标记
    hdfs_save_string(result_path, '_DATASET_SUCCESS')

    return


# # 获得得到的dataset对应的配置信息

# In[11]:


# 设定bipaths数据集初始化相关配置
bipaths_dataset_init_config = {}
bipaths_dataset_init_config["dataset_local_rel_path"] = "/data/dataset/AMiner/2017_v2"
bipaths_dataset_init_config["dataset_hdfs_path"] = labeled_samples_config["task_data_path"] + "/dataset"
bipaths_dataset_init_config["dataset_format"] = "parquet"
bipaths_dataset_init_config["scale_hdfs_path"] = labeled_samples_config["task_data_path"] + "/scale"


# In[12]:


from joinminer.python import write_json_file

import os

def bipaths_dataset_config_init(graph, main_bipaths, main_bipath_query_config, labeled_samples_config, 
                                inference_time_cols_values, clt_bipaths, bipaths_dataset_init_config):
    # 记录全部特征向量列以及对应的向量长度用于之后做归一化
    feat_vector_col_sizes = {}

    # 还要记录要合并计算归一化配置的各组特征列
    feat_col_to_std_group = {}
    
    # 记录该数据集的具体配置信息
    bipaths_dataset = {}

    # 设定标签列列名
    bipaths_dataset["label_column"] = "label"
    
    # 设定标签列类型(binary)
    bipaths_dataset["label_type"] = "binary"

    # 获得各个graph_token类型对应的特性向量长度以及对应是node还是edge类型
    bipaths_dataset["token_feat_len"] = {}
    bipaths_dataset["token_node_edge_type"] = {}
    for node_type in graph.nodes:
        bipaths_dataset["token_feat_len"][node_type] = len(graph.nodes[node_type]["graph_token_feat_cols"])
        bipaths_dataset["token_node_edge_type"][node_type] = "node"
    for edge_type in graph.edges:
        bipaths_dataset["token_feat_len"][edge_type] = len(graph.edges[edge_type]["graph_token_feat_cols"])
        bipaths_dataset["token_node_edge_type"][edge_type] = "edge"
    
    # 获得forward_path的query node对应的特征列以及graph_token类型
    head_node_token_type = clt_bipaths["query_nodes_types"][0]

    # 获得基于head_node的token配置
    head_node_token_config = {}
    head_node_token_config["node_edge_type"] = "node"

    if len(graph.nodes[head_node_token_type]["graph_token_feat_cols"]) > 0:
        head_node_token_config["feat_col"] = f"query_node_{head_node_token_type}_index_0_feat"

        # 记录头结点特征列对应的向量长度
        feat_vector_col_sizes[head_node_token_config["feat_col"]] = len(graph.nodes[head_node_token_type]["graph_token_feat_cols"])

        # 记录头结点特征列计算归一化时所属组
        feat_col_to_std_group[head_node_token_config["feat_col"]] = head_node_token_config["feat_col"]
    else:
        head_node_token_config["feat_col"] = None

    head_node_token_config["token_type"] = head_node_token_type
    bipaths_dataset["head_node_token_config"] = head_node_token_config
    
    # 获得各forward_path类型对应的数据集所需配置
    bipaths_dataset["forward_paths"] = {}
    for forward_path_name in clt_bipaths["forward_paths_to_schema"]:
        # 记录该path所需配置
        forward_path_dataset_config = {}
        
        # 获得forward_path对应配置  
        forward_path = clt_bipaths["forward_paths_to_schema"][forward_path_name]

        # 记录该类型序列长度  
        forward_path_dataset_config["path_len"] = len(forward_path["flatten_format"]["seq"])

        # 获得clt到的列会添加的前缀
        clt_col_prefix = forward_path['name']
        forward_path_dataset_config["clt_col_prefix"] = clt_col_prefix
        
        # 记录表示collect数量的列名
        forward_path_dataset_config["clt_count_col"] = f"{clt_col_prefix}_collect_count"

        # 获得对该forward_path最大collect数量
        forward_path_dataset_config["collect_records_count"] = forward_path["collect_records_count"]
        
        # 记录序列化后的各个token的配置, 第一个token是基于head_node的配置
        forward_path_dataset_config["seq_tokens"] = []                

        # 依次加入起始点后的各个token对应的配置
        for entry in forward_path["flatten_format"]["seq"][1:]:
            # 获得该token对应的相关配置
            token_config = {}
            
            # 判断是节点还是边
            if "node_cols" in entry:
                token_config["node_edge_type"] = "node"

                # 如果对应的特征数量大于0就记录对应的特征列，否则就记为none
                if len(graph.nodes[entry["node_type"]]["graph_token_feat_cols"]) > 0:
                    token_config["feat_col"] = entry["node_feat_col"]

                    # 将对应的该特征列加入到要归一化的特征列中，并记录长度
                    clt_feat_col = f"{clt_col_prefix}_clt_{entry['node_feat_col']}"

                    # 记录头结点特征列对应的向量长度
                    feat_vector_col_sizes[clt_feat_col] = len(graph.nodes[entry["node_type"]]["graph_token_feat_cols"])
        
                    # 记录头结点特征列计算归一化时所属组
                    feat_col_to_std_group[clt_feat_col] = f"{clt_col_prefix}_{entry['node_feat_col']}"
                else:
                    token_config["feat_col"] = None
                
                token_config["node_pos_index"] = entry["node_index"]
                token_config["token_type"] = entry["node_type"]
            else:
                token_config["node_edge_type"] = "edge"

                # 如果对应的特征数量大于0就记录对应的特征列，否则就记为none
                if len(graph.edges[entry["edge_type"]]["graph_token_feat_cols"]) > 0:
                    token_config["feat_col"] = entry["edge_feat_col"]

                    # 将对应的该特征列加入到要归一化的特征列中，并记录长度
                    clt_feat_col = f"{clt_col_prefix}_clt_{entry['edge_feat_col']}"

                    # 记录头结点特征列对应的向量长度
                    feat_vector_col_sizes[clt_feat_col] = len(graph.edges[entry["edge_type"]]["graph_token_feat_cols"])
    
                    # 记录头结点特征列计算归一化时所属组
                    feat_col_to_std_group[clt_feat_col] = f"{clt_col_prefix}_{entry['edge_feat_col']}"
                else:
                    token_config["feat_col"] = None
                
                token_config["edge_pos_indexes"] = entry["linked_node_indexes"]
                token_config["token_type"] = entry["edge_type"]
                
            # 记录对应的配置
            forward_path_dataset_config["seq_tokens"].append(token_config)
            
        # 记录最终的配置
        bipaths_dataset["forward_paths"][forward_path_name] = forward_path_dataset_config

    # 获得backward_path的query node对应的graph_token类型
    tail_node_token_type = clt_bipaths["query_nodes_types"][1]

    # 获得基于tail_node的token配置
    tail_node_token_config = {}
    tail_node_token_config["node_edge_type"] = "node"

    if len(graph.nodes[tail_node_token_type]["graph_token_feat_cols"]) > 0:
        tail_node_token_config["feat_col"] = f"query_node_{tail_node_token_type}_index_1_feat"

        # 记录头结点特征列对应的向量长度
        feat_vector_col_sizes[tail_node_token_config["feat_col"]] = len(graph.nodes[tail_node_token_type]["graph_token_feat_cols"])

        # 记录头结点特征列计算归一化时所属组
        feat_col_to_std_group[tail_node_token_config["feat_col"]] = tail_node_token_config["feat_col"]
    else:
        tail_node_token_config["feat_col"] = None
    
    tail_node_token_config["token_type"] = tail_node_token_type
    bipaths_dataset["tail_node_token_config"] = tail_node_token_config
    
    # 获得各backward_path类型对应的数据集所需配置
    bipaths_dataset["backward_paths"] = {}
    for backward_path_name in clt_bipaths["backward_paths_to_schema"]:
        # 记录该path所需配置
        backward_path_dataset_config = {}
        
        # 获得backward_path对应配置  
        backward_path = clt_bipaths["backward_paths_to_schema"][backward_path_name]

        # 记录该类型序列长度  
        backward_path_dataset_config["path_len"] = len(backward_path["flatten_format"]["seq"])

        # 获得clt到的列会添加的前缀
        clt_col_prefix = backward_path['name']
        backward_path_dataset_config["clt_col_prefix"] = clt_col_prefix
        
        # 记录表示collect数量的列名
        backward_path_dataset_config["clt_count_col"] = f"{clt_col_prefix}_collect_count"

        # 获得对该backward_path最大collect数量
        backward_path_dataset_config["collect_records_count"] = backward_path["collect_records_count"]
        
        # 记录序列化后的各个token的配置, 第一个token是基于head_node的配置
        backward_path_dataset_config["seq_tokens"] = []                

        # 依次加入起始点后的各个token对应的配置
        for entry in backward_path["flatten_format"]["seq"][1:]:
            # 获得该token对应的相关配置
            token_config = {}
            
            # 判断是节点还是边
            if "node_cols" in entry:
                token_config["node_edge_type"] = "node"

                # 如果对应的特征数量大于0就记录对应的特征列，否则就记为none
                if len(graph.nodes[entry["node_type"]]["graph_token_feat_cols"]) > 0:
                    token_config["feat_col"] = entry["node_feat_col"]

                    # 将对应的该特征列加入到要归一化的特征列中，并记录长度
                    clt_feat_col = f"{clt_col_prefix}_clt_{entry['node_feat_col']}"

                    # 记录头结点特征列对应的向量长度
                    feat_vector_col_sizes[clt_feat_col] = len(graph.nodes[entry["node_type"]]["graph_token_feat_cols"])
    
                    # 记录头结点特征列计算归一化时所属组
                    feat_col_to_std_group[clt_feat_col] = f"{clt_col_prefix}_{entry['node_feat_col']}"
                else:
                    token_config["feat_col"] = None
                
                token_config["node_pos_index"] = entry["node_index"]
                token_config["token_type"] = entry["node_type"]
            else:
                token_config["node_edge_type"] = "edge"

                # 如果对应的特征数量大于0就记录对应的特征列，否则就记为none
                if len(graph.edges[entry["edge_type"]]["graph_token_feat_cols"]) > 0:
                    token_config["feat_col"] = entry["edge_feat_col"]

                    # 将各个collect id对应的该特征列加入到要归一化的特征列中，并记录长度
                    clt_feat_col = f"{clt_col_prefix}_clt_{entry['edge_feat_col']}"

                    # 记录头结点特征列对应的向量长度
                    feat_vector_col_sizes[clt_feat_col] = len(graph.edges[entry["edge_type"]]["graph_token_feat_cols"])
    
                    # 记录头结点特征列计算归一化时所属组
                    feat_col_to_std_group[clt_feat_col] = f"{clt_col_prefix}_{entry['edge_feat_col']}"
                else:
                    token_config["feat_col"] = None
                
                token_config["edge_pos_indexes"] = entry["linked_node_indexes"]
                token_config["token_type"] = entry["edge_type"]
                
            # 记录对应的配置
            backward_path_dataset_config["seq_tokens"].append(token_config)
            
        # 记录最终的配置
        bipaths_dataset["backward_paths"][backward_path_name] = backward_path_dataset_config

    # 记录各个bipath对应的seq最大节点数量，用于创建pos_embed
    bipaths_dataset["bipath_max_node_count"] = 2
    
    # 记录各个bipath对应的seq最大长度，用于创建padding_mask
    bipaths_dataset["bipath_max_seq_len"] = 1

    # 获得各bipath类型对应的数据集所需配置
    bipaths_dataset["bipaths"] = {}
    for bipath_schema in clt_bipaths["bipaths_list_schema"]:
        # 记录该bipath所需配置
        bipath_dataset_config = {}

        # 获得该bipath对应的名称
        bipath_name = bipath_schema["join_edges_name"]
        
        # 获得clt到的列会添加的前缀
        clt_col_prefix = bipath_name
        bipath_dataset_config["clt_col_prefix"] = clt_col_prefix

        # 记录表示collect数量的列名
        bipath_dataset_config["clt_count_col"] = f"{clt_col_prefix}_collect_count"

        # 获得对该backward_path最大collect数量
        bipath_dataset_config["collect_records_count"] = bipath_schema["collect_records_count"]

        # 记录序列化后的各个token的配置, 第一个token是基于head_node的配置，最后一个是tail_node，省略这两个
        bipath_dataset_config["seq_tokens"] = []      
        for entry in bipath_schema["bipath_schema"]["flatten_format"]["seq"][1:-1]:
            # 获得该token对应的相关配置
            token_config = {}
            
            # 判断是节点还是边
            if "node_cols" in entry:
                token_config["node_edge_type"] = "node"

                # 如果对应的特征数量大于0就记录对应的特征列，否则就记为none
                if len(graph.nodes[entry["node_type"]]["graph_token_feat_cols"]) > 0:
                    token_config["feat_col"] = entry["node_feat_col"]

                    # 将各个collect id对应的该特征列加入到要归一化的特征列中，并记录长度
                    clt_feat_col = f"{clt_col_prefix}_clt_{entry['node_feat_col']}"

                    # 记录头结点特征列对应的向量长度
                    feat_vector_col_sizes[clt_feat_col] = len(graph.nodes[entry["node_type"]]["graph_token_feat_cols"])
    
                    # 记录头结点特征列计算归一化时所属组
                    feat_col_to_std_group[clt_feat_col] = f"{clt_col_prefix}_{entry['node_feat_col']}"
                else:
                    token_config["feat_col"] = None
                
                token_config["node_pos_index"] = entry["node_index"]
                token_config["token_type"] = entry["node_type"]

            else:
                token_config["node_edge_type"] = "edge"

                # 如果对应的特征数量大于0就记录对应的特征列，否则就记为none
                if len(graph.edges[entry["edge_type"]]["graph_token_feat_cols"]) > 0:
                    token_config["feat_col"] = entry["edge_feat_col"]

                    # 将各个collect id对应的该特征列加入到要归一化的特征列中，并记录长度
                    clt_feat_col = f"{clt_col_prefix}_clt_{entry['edge_feat_col']}"

                    # 记录头结点特征列对应的向量长度
                    feat_vector_col_sizes[clt_feat_col] = len(graph.edges[entry["edge_type"]]["graph_token_feat_cols"])
    
                    # 记录头结点特征列计算归一化时所属组
                    feat_col_to_std_group[clt_feat_col] = f"{clt_col_prefix}_{entry['edge_feat_col']}"
                else:
                    token_config["feat_col"] = None
                
                token_config["edge_pos_indexes"] = entry["linked_node_indexes"]
                token_config["token_type"] = entry["edge_type"]
                
            # 记录对应的配置
            bipath_dataset_config["seq_tokens"].append(token_config)
            
        # 记录该类型序列长度  
        bipath_dataset_config["path_len"] = len(bipath_dataset_config["seq_tokens"]) + 2
            
        # 记录最终的配置
        bipaths_dataset["bipaths"][bipath_name] = bipath_dataset_config

        # 更新最大节点数量，这算法只能在算path时用
        bipath_node_count = (len(bipath_dataset_config["seq_tokens"]) + 3)//2
        if bipath_node_count > bipaths_dataset["bipath_max_node_count"]:
            bipaths_dataset["bipath_max_node_count"] = bipath_node_count

        # 更新最大序列长度
        if bipath_dataset_config["path_len"] > bipaths_dataset["bipath_max_seq_len"]:
            bipaths_dataset["bipath_max_seq_len"] = bipath_dataset_config["path_len"]

    # 记录完整数据集中特征列到特征类型的对应关系，方便归一化推理数据集
    bipaths_dataset["feat_col_to_std_group"] = feat_col_to_std_group

    bipaths_dataset["feat_vector_col_sizes"] = feat_vector_col_sizes
    
    # 保存该数据集对应的配置文件到数据集文件内 
    dataset_local_path = PROJECT_ROOT + bipaths_dataset_init_config["dataset_local_rel_path"]
    mkdir(dataset_local_path)
    write_json_file(bipaths_dataset, dataset_local_path + "/dataset_config.json")

    return

# 检查是否已有对应结果
dataset_config_path = PROJECT_ROOT + bipaths_dataset_init_config["dataset_local_rel_path"] + "/dataset_config.json"
if not os.path.isfile(dataset_config_path):
    bipaths_dataset_config_init(graph, main_bipaths, main_bipath_query_config, labeled_samples_config, 
                                inference_time_cols_values, clt_bipaths, bipaths_dataset_init_config)


# # 分批对目标的节点对进行bipaths_collection

# In[13]:


import subprocess
from functools import reduce
from pyspark.sql import DataFrame
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import broadcast

def reliable_node_pairs_bipaths_collect(spark, graph, main_bipaths, main_bipath_query_config, labeled_samples_config, 
                                        inference_time_cols_values, clt_bipaths, batch_max_rows = 5e8):
    # 设定clt_bipaths的相关配置
    clt_bipaths_query_config = clt_bipaths_query_config_init(spark, graph, main_bipaths, main_bipath_query_config, 
                                                             labeled_samples_config, inference_time_cols_values)

    # 获得该任务相关结果存储位置
    task_data_path = clt_bipaths_query_config["tgt_query_nodes"]["result_path"]
    
    # 设定对query_nodes的分组结果存储位置
    tgt_query_nodes_batch_path = task_data_path + f"/batch_reliable_node_pairs"

    # 检查是否已有对应的分组结果
    is_complete, _ = hdfs_check_partitions(tgt_query_nodes_batch_path)

    # 如果没有分组结果
    if not is_complete:
        # 先分别获得带标签的数据和不带标签的数据
        labeled_batch_df = clt_bipaths_query_config["tgt_query_nodes"]["df"].filter(col("sample_type") != "infer")
        unlabeled_batch_df = clt_bipaths_query_config["tgt_query_nodes"]["df"].filter(col("sample_type") == "infer")
        
        # 基于batch_max_rows获得要分出的batch数目
        labeled_batch_count = labeled_batch_df.count() // batch_max_rows + 1
        unlabeled_batch_count = unlabeled_batch_df.count() // batch_max_rows + 1
        
        # 为各行随机添加所属的batch_id
        labeled_batch_df = labeled_batch_df.withColumn("batch_id", (rand() * lit(labeled_batch_count)).cast("int"))
        unlabeled_batch_df = unlabeled_batch_df.withColumn("batch_id", 
                                                           (lit(labeled_batch_count) + rand() * lit(unlabeled_batch_count)).cast("int"))
        
        # 将label列不为None的都算为第0个batch中
        batch_query_nodes_df = labeled_batch_df.unionByName(unlabeled_batch_df)

        # 保存结果，并按batch_id分区
        pyspark_optimal_save(batch_query_nodes_df, tgt_query_nodes_batch_path, "parquet", "overwrite", ["batch_id"])

    # 检查是否有对训练数据采样后的结果 
    is_complete, _ = hdfs_check_partitions(tgt_query_nodes_batch_path + "/batch_id=train_sample")

    # 如果没有则进行采样
    if not is_complete:
        # 先获得带标签的数据
        labeled_batch_df = clt_bipaths_query_config["tgt_query_nodes"]["df"].filter(col("sample_type") != "infer")

        # 再获得正样本数据 
        positive_sample_df = labeled_batch_df.filter(col("label") == 1)
    
        # 获得各个sample_type对应的正样本数量  
        positive_counts_df = positive_sample_df.groupBy("sample_type").agg(count_("*").alias("positive_count"))
        
        # 获得各个samle_type对应的正样本数量的24倍 
        sampling_counts_df = positive_counts_df.withColumn("negative_sample_count", col("positive_count") * 24)
        
        # 获得负样本数据
        negative_sample_df = labeled_batch_df.filter(col("label") == 0)
        
        # 为每个sample_type的负样本添加行号
        window_spec = Window.partitionBy("sample_type").orderBy(rand())
        negative_sample_with_rownum = negative_sample_df.withColumn("row_number", row_number().over(window_spec))
        
        # 将采样数量信息join到负样本数据上
        negative_sample_with_limit = negative_sample_with_rownum.join(
            broadcast(sampling_counts_df.select("sample_type", "negative_sample_count")), 
            on="sample_type", 
            how="left"
        )
        
        # 基于获得的数量只保留对应数量的负样本
        sampled_negative_df = negative_sample_with_limit.filter(
            col("row_number") <= col("negative_sample_count")
        ).drop("row_number", "negative_sample_count")
        
        # 合并为完整数据
        train_sample_df = positive_sample_df.unionByName(sampled_negative_df).persist()

        train_sample_df.groupBy("sample_type", "label").agg(count_("*").alias("count")).show()
        
        # 保存结果，并按batch_id分区
        pyspark_optimal_save(train_sample_df, tgt_query_nodes_batch_path + "/batch_id=train_sample", "parquet", "overwrite")
    
    # 获得全部的batch_id对应的存储路径 
    query_nodes_batch_ids =  [int(x.split("batch_id=")[1]) for x in hdfs_list_contents(tgt_query_nodes_batch_path, "directories") if x.split("batch_id=")[1] != "train_sample"]

    # 按batch_id由小到大排序
    query_nodes_batch_ids.sort()

    # 获得collect得到的数据集的配置信息
    bipaths_dataset_config = read_json_file(dataset_config_path)
    
    # 获得query结果位置
    dataset_hdfs_path = bipaths_dataset_init_config["dataset_hdfs_path"]
    
    # 依次处理各个batch的数据
    # for batch_id in query_nodes_batch_ids:
    for batch_id in ["train_sample"]:
        # 检查是否已有对应结果
        if hdfs_check_file_exists(dataset_hdfs_path + f"/batch_id={batch_id}/_DATASET_SUCCESS"):
            continue
        
        # 获得对应的路径
        query_nodes_batch_id_path = tgt_query_nodes_batch_path + f"/batch_id={batch_id}"
        
        # 读取对应的query_nodes数据
        batch_reliable_node_pairs_df = pyspark_read_table(spark, query_nodes_batch_id_path, "parquet")

        # 获得涉及到的时间
        distinct_time_cols_rows = batch_reliable_node_pairs_df.select(graph.graph_time_cols_alias).distinct().collect()
        query_time_cols_values = [[str(row[c]) for c in graph.graph_time_cols_alias] for row in distinct_time_cols_rows]
        
        # 重新设置query_nodes配置
        batch_clt_bipaths_query_config = {}
        batch_clt_bipaths_query_config["graph_time_values"] = query_time_cols_values
        batch_clt_bipaths_query_config["tgt_query_nodes"] = {}
        batch_clt_bipaths_query_config["tgt_query_nodes"]["result_path"] = task_data_path + f"/batch_pairs_results/batch_{batch_id}"
        batch_clt_bipaths_query_config["tgt_query_nodes"]["df"] = batch_reliable_node_pairs_df

        # 获得对该batch进行clt_bipaths的具体结果(按sample_type分区保存)
        batch_clt_bipaths_df = bipaths_tgt_collections(spark, graph, clt_bipaths, batch_clt_bipaths_query_config,
                                                       dataset_hdfs_path + f"/batch_id={batch_id}")
    
    return
    
spark_runner.run(reliable_node_pairs_bipaths_collect, graph, main_bipaths, main_bipath_query_config, labeled_samples_config, 
                 inference_time_cols_values, clt_bipaths)


# In[ ]:




