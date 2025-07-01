from .table_graph import graph_token_node_col_name
from ..python import time_costing

import copy

# 每个节点列在join_edges中对应的标准名称
def standard_node_col_name(node_type, node_index, col_index):
    return f"Node_{node_index}_{node_type}_Col_{col_index}"

def standard_feat_col_name(node_edge_type, token_type, token_index):
    return f"{node_edge_type}_{token_type}_index_{token_index}_feat"

@time_costing
def join_edges_list_init(graph, join_edges_list_config):
    # 用于记录初始化后的结果
    join_edges_list = {}

    # 记录该join_edges组的名称
    join_edges_list_name = join_edges_list_config["join_edges_list_name"]
    join_edges_list["join_edges_list_name"] = join_edges_list_name
    
    # 记录query nodes相关配置
    join_edges_list["query_nodes_types"] = join_edges_list_config["query_nodes_types"]
    join_edges_list["query_nodes_indexes"] = join_edges_list_config["query_nodes_indexes"]

    # 获得要聚合的节点对应的列名称以及从对应的graph_token表中读取数据时节点列对应的列名和别名
    join_edges_list["query_nodes_cols"] = []
    join_edges_list["query_nodes_cols_alias"] = []
    join_edges_list["query_nodes_join_cols"] = []
    join_edges_list["query_nodes_feat_cols"] = []
    for query_node_i in range(len(join_edges_list["query_nodes_types"])):
        query_node_type = join_edges_list["query_nodes_types"][query_node_i]
        query_node_index = join_edges_list["query_nodes_indexes"][query_node_i]

        query_node_token_cols = []
        query_node_cols = []
        for node_col_i in range(len(graph.nodes[query_node_type]["node_col_types"])):
            query_node_token_col = graph_token_node_col_name(query_node_type, None, node_col_i)
            query_node_token_cols.append(query_node_token_col)
            
            query_node_col = standard_node_col_name(query_node_type, query_node_index, node_col_i)
            query_node_cols.append(query_node_col)
            
        join_edges_list["query_nodes_cols"].extend(query_node_cols)

        query_node_token_feat_col = graph.nodes[query_node_type]["query_config"]["assembled_feat_col"]
        query_node_feat_col = standard_feat_col_name("node", query_node_type, query_node_index)
        
        # 记录query nodes对应的node_table要取的列及对应的别名
        query_node_cols_alias = [[x, y] for x, y in zip(query_node_token_cols, query_node_cols)]
        query_node_cols_alias.append([query_node_token_feat_col, query_node_feat_col])
        for time_col in graph.graph_time_cols_alias:
            query_node_cols_alias.append([time_col, time_col])
        
        join_edges_list["query_nodes_cols_alias"].append(query_node_cols_alias)

        query_node_join_cols = query_node_cols + list(graph.graph_time_cols_alias)
        join_edges_list["query_nodes_join_cols"].append(query_node_join_cols)

        join_edges_list["query_nodes_feat_cols"].append(query_node_feat_col)
        
    # 获得结果保存位置
    join_edges_list["join_edges_list_path"] = join_edges_list_config["join_edges_list_root_path"] + f"/{join_edges_list_name}"
    
    # 获得结果保存格式
    join_edges_list["join_edges_list_table_format"] = join_edges_list_config["join_edges_list_table_format"]

    # 获得对query_nodes的基础配置来初始化各个join_edges
    query_nodes_config = {}
    query_nodes_config["query_nodes_types"] = list(join_edges_list["query_nodes_types"])
    query_nodes_config["query_nodes_indexes"] = list(join_edges_list["query_nodes_indexes"])
    
    # 获得对全部join_edges的默认配置信息
    join_edges_default_config = join_edges_list_config["join_edges_default_config"]

    # 记录各个join_edges的具体配置
    join_edges_list["join_edges_list_schema"] = []
    
    # 依次处理各个join_edges_config
    for join_edges_config in join_edges_list_config["join_edges_list_schema"]:
        # 初始化该join_edges
        join_edges = join_edges_init(graph, query_nodes_config, join_edges_config, join_edges_default_config)

        # 保存完整的信息
        join_edges_list["join_edges_list_schema"].append(join_edges)
    
    return join_edges_list

def join_edges_init(graph, query_nodes_config, join_edges_config, join_edges_default_config):
    # 创建用于记录该join_edges对应的相关信息的变量
    join_edges = {}

    # 首先是名称
    join_edges_name = join_edges_config["join_edges_name"]
    join_edges["name"] = join_edges_name

    # 设置采样结果的存储路径
    if "join_edges_root_path" in join_edges_config and join_edges_config["join_edges_root_path"] != "":
        join_edges["join_edges_path"] = join_edges_config["join_edges_root_path"] + f"/{join_edges_name}"
    else:
        join_edges["join_edges_path"] = join_edges_default_config["join_edges_root_path"] + f"/{join_edges_name}"

    # 如果未给出采样结果表格对应格式，则用默认值填充
    if "join_edges_table_format" not in join_edges_config or join_edges_config["join_edges_table_format"] == "":
        join_edges["join_edges_table_format"] = join_edges_default_config["join_edges_table_format"]
    else:
        join_edges["join_edges_table_format"] = join_edges_config["join_edges_table_format"]
    
    # 如果有父路径则记录对应配置信息 
    if "parent_join_edges" in join_edges_config:
        parent_join_edges = {}

        # 记录名称
        parent_join_edges_name = join_edges_config["parent_join_edges"]["join_edges_name"]
        parent_join_edges["name"] = parent_join_edges_name

        # 设定parent的query配置
        parent_query_config = {}
        
        # 设置采样结果的存储路径
        if "join_edges_root_path" in join_edges_config["parent_join_edges"]:
            parent_query_config["result_path"] = join_edges_config["parent_join_edges"]["join_edges_root_path"] + f"/{parent_join_edges_name}"
        else:
            parent_query_config["result_path"] = join_edges_default_config["join_edges_root_path"] + f"/{parent_join_edges_name}"
    
        # 如果未给出采样结果表格对应格式，则用默认值填充
        if "join_edges_table_format" not in join_edges_config["parent_join_edges"]:
            parent_query_config["result_format"] = join_edges_default_config["join_edges_table_format"]
        else:
            parent_query_config["result_format"] = join_edges_config["parent_join_edges"]["join_edges_table_format"]

        # 记录query配置
        parent_join_edges["query_config"] = parent_query_config
        
        # 记录parent_join_edges长度
        parent_join_edges["join_edges_len"] = join_edges_config["parent_join_edges"]["join_edges_len"]
        
        join_edges["parent_join_edges"] = parent_join_edges
    
    # 初始化对应的join_edges schema，补全一些默认信息，检查数据是否准确，并进行flatten
    join_edges_info = join_edges_schema_init(graph, query_nodes_config, join_edges_config["join_edges_schema"],
                                             join_edges_default_config)
    join_edges["schema"] = join_edges_info["join_edges_schema"]
    join_edges["flatten_format"] = join_edges_info["flatten_format"]

    # 检查是否有聚合路径相关配置 
    if "join_edges_agg" in join_edges_config:
        # 记录聚合相关配置
        join_edges["join_edges_agg"] = {}
        join_edges["join_edges_agg"]["agg_nodes_types"] = join_edges_config["join_edges_agg"]["agg_nodes_types"]
        join_edges["join_edges_agg"]["agg_nodes_indexes"] = join_edges_config["join_edges_agg"]["agg_nodes_indexes"]
        join_edges["join_edges_agg"]["agg_funcs"] = join_edges_config["join_edges_agg"]["agg_funcs"]
        
        # 获得要聚合的节点应的列名称
        join_edges["join_edges_agg"]["agg_nodes_cols"] = []
        for agg_node_i in range(len(join_edges["join_edges_agg"]["agg_nodes_types"])):
            agg_node_type = join_edges["join_edges_agg"]["agg_nodes_types"][agg_node_i]
            agg_node_index = join_edges["join_edges_agg"]["agg_nodes_indexes"][agg_node_i]
            
            agg_node_col_types = graph.nodes[agg_node_type]["node_col_types"]
            for node_col_i in range(len(agg_node_col_types)):
                agg_node_col = standard_node_col_name(agg_node_type, agg_node_index, node_col_i)
                join_edges["join_edges_agg"]["agg_nodes_cols"].append(agg_node_col)

    # 生成对应的query_config
    join_edges["query_config"] = join_edges_query_config_init(graph, join_edges)

    # 记录从query_config中要collect的记录数目
    if "collect_records_count" not in join_edges_config:
        join_edges["collect_records_count"] = 1
    else:
        join_edges["collect_records_count"] = join_edges_config["collect_records_count"]
        
    return join_edges

def join_edges_query_config_init(graph, join_edges):
    query_config = {}

    # 记录query结果的保存位置和保存格式
    query_config["result_path"] = join_edges["join_edges_path"]
    query_config["result_format"] = join_edges["join_edges_table_format"]

    # 记录要保留的全部特征列，用于collect到query nodes，目前还用于agg，但之后应该让agg自己配置
    query_config["feat_cols_sizes"] = {}
    
    # 依次获得各个join_edge的query config
    query_config["join_edge_list"] = []
    for join_edge_i, join_edge_schema in enumerate(join_edges["schema"]):
        join_edges_edge_config = {}
    
        # 先配置该join_edge的query config
        join_edge_config = {}

        # 记录该join_edge的名称
        join_edge_config["name"] = join_edge_schema["name"]
        
        # 记录query结果的保存位置和保存格式
        join_edge_config["result_path"] = join_edge_schema["join_edge_path"]
        join_edge_config["result_format"] = join_edge_schema["join_edge_table_format"]
        
        # 获得对应的edge type，直接用graph内该edge对应的query config直接取数就行
        edge_type = join_edge_schema["edge_type"]
        join_edge_config["edge_type"] = edge_type
        
        # 获得对取数后的edge进行edge_limit的配置
        if "edge_limit_schema" in join_edge_schema:
            join_edge_config["edge_limit"] = {}
            
            # 获得要使用的特征列在向量中的序号
            join_edge_config["edge_limit"]["feat_cols"] = []
            for feat_col_i, feat_col in enumerate(join_edge_schema["edge_limit_schema"]["feat_cols"]):
                assert feat_col in graph.edges[edge_type]["graph_token_feat_cols"]
                feat_vector_index = graph.edges[edge_type]["graph_token_feat_cols"].index(feat_col)
                feat_limit_col = f"feat_col_{feat_col_i}"

                join_edge_config["edge_limit"]["feat_cols"].append([feat_vector_index,
                                                                    feat_limit_col])
            # 记录要使用的限制条件 
            join_edge_config["edge_limit"]["limit"] = join_edge_schema["edge_limit_schema"]["limit"]
            
        # 获得edge_sample的配置 
        if "edge_samples" in join_edge_schema:
            join_edge_config["edge_samples"] = []
            for edge_sample in join_edge_schema["edge_samples"]:
                edge_sample_config = {}
                edge_sample_config["sample_nodes_cols"] = edge_sample["sample_nodes_cols"]
                edge_sample_config["sample_type"] = edge_sample["sample_type"]
                edge_sample_config["sample_count"] = edge_sample["sample_count"]

                join_edge_config["edge_samples"].append(edge_sample_config)
                
        # 依次获得各个node_limit对应的配置 
        if "nodes_limit_schema" in join_edge_schema:
            join_edge_config["nodes_limit"] = []

            for node_limit in join_edge_schema["nodes_limit_schema"]:
                node_limit_config = {}

                # 记录对应的节点类型
                node_type = node_limit["node_type"]
                node_limit_config["node_type"] = node_type

                # 记录和edge表join的话要选取哪些列及别名
                node_limit_config["col_aliases"] = []
                node_limit_config["col_aliases"].extend(node_limit["node_columns"])
                for time_col in graph.graph_time_cols_alias:
                    node_limit_config["col_aliases"].append([time_col, time_col])
                
                node_limit_config["feat_cols"] = []
                for feat_col_i, feat_col in enumerate(node_limit["feat_cols"]):
                    assert feat_col in graph.nodes[node_type]["graph_token_feat_cols"]
                    feat_vector_index = graph.nodes[node_type]["graph_token_feat_cols"].index(feat_col)
                    feat_limit_col = f"feat_col_{feat_col_i}"
    
                    node_limit_config["feat_cols"].append([feat_vector_index, feat_limit_col])

                # 记录要使用的限制条件 
                node_limit_config["limit"] = node_limit["limit"]
                
                join_edge_config["nodes_limit"].append(node_limit_config)
        
        # 获得join_edge_limit的配置 
        if "join_edge_limit" in join_edge_schema:
            join_edge_config["join_edge_limit"] = {}
    
            # 先获得各个来源的特征 
            join_edge_config["join_edge_limit"]["feat_sources"] = []
            for src_i, feat_source in enumerate(join_edge_schema["join_edge_limit"]["feat_sources"]):
                feat_source_config = {}

                if feat_source["token_type"] == "edge":
                    edge_type = feat_source["node_edge_type"]
                    
                    feat_source_config["feat_cols"] = []
                    for feat_col_i, feat_col in enumerate(feat_source["feat_cols"]):
                        assert feat_col in graph.edges[edge_type]["graph_token_feat_cols"]
                        feat_vector_index = graph.edges[edge_type]["graph_token_feat_cols"].index(feat_col)
                        feat_limit_col = f"src_{src_i}_feat_col_{feat_col_i}"
        
                        feat_source_config["feat_cols"].append([feat_vector_index, feat_limit_col])
                        
                elif feat_source["token_type"] == "node":
                    node_type = feat_source["node_edge_type"]

                    # 记录节点类型，之后要先读取对应表
                    feat_source_config["node_type"] = node_type

                    # 记录和edge表join的话要选取哪些列及别名
                    feat_source_config["col_aliases"] = []
                    feat_source_config["col_aliases"].extend(feat_source["node_columns"])
                    for time_col in graph.graph_time_cols_alias:
                        feat_source_config["col_aliases"].append([time_col, time_col])
                    
                    feat_source_config["feat_cols"] = []
                    for feat_col_i, feat_col in enumerate(feat_source["feat_cols"]):
                        assert feat_col in graph.nodes[node_type]["graph_token_feat_cols"]
                        feat_vector_index = graph.nodes[node_type]["graph_token_feat_cols"].index(feat_col)
                        feat_limit_col = f"src_{src_i}_feat_col_{feat_col_i}"
        
                        feat_source_config["feat_cols"].append([feat_vector_index, feat_limit_col])

                    # 记录要用于join的节点列名
                    feat_source_config["node_cols"] = []
                    for node_col_alias in feat_source["node_columns"]:
                        feat_source_config["node_cols"].append(node_col_alias[1])
                
                join_edge_config["join_edge_limit"]["feat_sources"].append(feat_source_config)
                
            # 再记录要使用的条件
            join_edge_config["join_edge_limit"]["limit"] = join_edge_schema["join_edge_limit"]["limit"]
            
        # 获得join_edge_sample的配置 
        if "join_edge_samples" in join_edge_schema:
            join_edge_config["join_edge_samples"] = []
            for join_edge_sample in join_edge_schema["join_edge_samples"]:
                join_edge_sample_config = {}
                join_edge_sample_config["sample_nodes_cols"] = join_edge_sample["sample_nodes_cols"]
                join_edge_sample_config["sample_type"] = join_edge_sample["sample_type"]
                join_edge_sample_config["sample_count"] = join_edge_sample["sample_count"]

                join_edge_config["join_edge_samples"].append(join_edge_sample_config)

        # 记录完整的join_edge配置到join_edges中
        join_edges_edge_config["join_edge"] = join_edge_config

        # 记录将该join_edge并入join_edges中要保留的列及别名 
        join_edges_edge_config["col_aliases"] = []
        for join_node_columns in join_edge_schema["join_nodes_columns"]:
            join_edges_edge_config["col_aliases"].extend(join_node_columns)
        if "add_nodes_columns" in join_edge_schema:
            for add_node_columns in join_edge_schema["add_nodes_columns"]:
                join_edges_edge_config["col_aliases"].extend(add_node_columns)
        for time_col in graph.graph_time_cols_alias:
            join_edges_edge_config["col_aliases"].append([time_col, time_col])
        
        # 记录该join_edge的join方式和join点
        join_edges_edge_config["join_cols"] = []
        for join_node_columns in join_edge_schema["join_nodes_columns"]:
            for join_node_col_alias in join_node_columns:
                join_edges_edge_config["join_cols"].append(join_node_col_alias[1])
        join_edges_edge_config["join_cols"] = join_edges_edge_config["join_cols"] + list(graph.graph_time_cols_alias)
        join_edges_edge_config["join_type"] = "inner"

        # 获得join_edges_limit的配置
        if "join_edges_limit" in join_edge_schema:
            join_edges_edge_config["join_edges_limit"] = {}

            # 先获得各个来源的特征 
            join_edges_edge_config["join_edges_limit"]["feat_sources"] = []
            for src_i, feat_source in enumerate(join_edge_schema["join_edges_limit"]["feat_sources"]):
                feat_source_config = {}

                if feat_source["token_type"] == "edge":
                    edge_type = feat_source["node_edge_type"]

                    # 记录对应的join_edge在join_edges中的序号
                    feat_source_config["join_edge_index"] = feat_source["join_edge_index"]
                    
                    feat_source_config["feat_cols"] = []
                    for feat_col_i, feat_col in enumerate(feat_source["feat_cols"]):
                        assert feat_col in graph.edges[edge_type]["graph_token_feat_cols"]
                        feat_vector_index = graph.edges[edge_type]["graph_token_feat_cols"].index(feat_col)
                        feat_limit_col = f"src_{src_i}_feat_col_{feat_col_i}"
        
                        feat_source_config["feat_cols"].append([feat_vector_index, feat_limit_col])
                        
                elif feat_source["token_type"] == "node":
                    node_type = feat_source["node_edge_type"]

                    # 记录节点类型，之后要先读取对应表
                    feat_source_config["node_type"] = node_type

                    # 记录和edge表join的话要选取哪些列及别名
                    feat_source_config["col_aliases"] = []
                    feat_source_config["col_aliases"].extend(feat_source["node_columns"])
                    for time_col in graph.graph_time_cols_alias:
                        feat_source_config["col_aliases"].append([time_col, time_col])
                    
                    feat_source_config["feat_cols"] = []
                    for feat_col_i, feat_col in enumerate(feat_source["feat_cols"]):
                        assert feat_col in graph.nodes[node_type]["graph_token_feat_cols"]
                        feat_vector_index = graph.nodes[node_type]["graph_token_feat_cols"].index(feat_col)
                        feat_limit_col = f"src_{src_i}_feat_col_{feat_col_i}"
        
                        feat_source_config["feat_cols"].append([feat_vector_index, feat_limit_col])

                    # 记录要用于join的节点列名
                    feat_source_config["node_cols"] = []
                    for node_col_alias in feat_source["node_columns"]:
                        feat_source_config["node_cols"].append(node_col_alias[1])
                        
                join_edges_edge_config["join_edges_limit"]["feat_sources"].append(feat_source_config)
                
            # 再记录要使用的条件
            join_edges_edge_config["join_edges_limit"]["limit"] = join_edge_schema["join_edges_limit"]["limit"]
        
        # 获得join_edges_sample的配置
        if "join_edges_samples" in join_edge_schema:
            join_edges_edge_config["join_edges_samples"] = []
            for join_edges_sample in join_edge_schema["join_edges_samples"]:
                join_edges_sample_config = {}
                join_edges_sample_config["sample_nodes_cols"] = join_edges_sample["sample_nodes_cols"]
                join_edges_sample_config["sample_type"] = join_edges_sample["sample_type"]
                join_edges_sample_config["sample_count"] = join_edges_sample["sample_count"]

                join_edges_edge_config["join_edges_samples"].append(join_edges_sample_config)

        # 获得feature_add的配置
        if "feature_add" in join_edge_schema:
            join_edges_edge_config["feat_add_srcs"] = []

            # 目前只有全量补全一种方案
            if join_edge_schema["feature_add"]["add_type"] == "full":
                # 补全除了query nodes外的全部特征
                # 目前认为query nodes就是第一个join_edge的join_nodes
                # 所以就是跳过第一个join_edge的join_nodes个数个entry
                query_nodes_count = len(join_edges["schema"][0]["join_nodes_types"])
                for entry in join_edges["flatten_format"]["seq"][query_nodes_count:]:
                    feat_source_config = {}

                    # 判断是节点还是边
                    if "node_cols" in entry:
                        # 检查是否有具体的特征，没有就跳过
                        if len(graph.nodes[entry["node_type"]]["graph_token_feat_cols"]) == 0:
                            continue
                            
                        # 如果是节点就记录对应的节点类型，可以直接读取对应的graph_token表
                        feat_source_config["node_type"] = entry["node_type"]
                        
                        # 获得节点表原始节点列名和特征向量列名
                        raw_node_cols = graph.nodes[entry["node_type"]]["query_config"]["node_cols"]
                        raw_feat_col = graph.nodes[entry["node_type"]]["query_config"]["assembled_feat_col"]
                        
                        # 获得对应的join_edges中的节点列名和特征向量列名
                        renamed_node_cols = entry["node_cols"]
                        renamed_feat_col = entry["node_feat_col"]
                        
                        # 记录和edge表join的话要选取哪些列及别名
                        feat_source_config["col_aliases"] = []
                        for node_col_i in range(len(raw_node_cols)):
                            feat_source_config["col_aliases"].append([raw_node_cols[node_col_i],
                                                                      renamed_node_cols[node_col_i]])
                        feat_source_config["col_aliases"].append([raw_feat_col, renamed_feat_col])
                        for time_col in graph.graph_time_cols_alias:
                            feat_source_config["col_aliases"].append([time_col, time_col])

                        # 记录用于join的id列 
                        feat_source_config["join_cols"] = renamed_node_cols + list(graph.graph_time_cols_alias)

                        # 记录最终添加的特征向量列
                        feat_source_config["feat_vec_col"] = renamed_feat_col

                        # 记录最终添加的特征向量列的维度
                        feat_source_config["feat_vec_len"] = len(graph.nodes[entry["node_type"]]["graph_token_feat_cols"])
                        
                        # 记录该join_edges目前得到的特征列
                        query_config["feat_cols_sizes"][renamed_feat_col] = feat_source_config["feat_vec_len"]
                    else:
                        # 检查是否有具体的特征，没有就跳过
                        if len(graph.edges[entry["edge_type"]]["graph_token_feat_cols"]) == 0:
                            continue
                            
                        # 如果是边就记录对应的join_edge序号，可以直接读取对应的join_edge表
                        feat_source_config["join_edge_index"] = entry["edge_index"]

                        # 获得该join_edge对应的信息
                        src_join_edge_schema = join_edges["schema"][entry["edge_index"]]

                        # 获得原始特征列名以及join_edges中对应的列名
                        raw_feat_col = graph.edges[entry["edge_type"]]["query_config"]["assembled_feat_col"]
                        renamed_feat_col = entry["edge_feat_col"]
                        
                        # 获得其中所需的列及对应的别名 
                        feat_source_config["col_aliases"] = []
                        for join_node_columns in src_join_edge_schema["join_nodes_columns"]:
                            feat_source_config["col_aliases"].extend(join_node_columns)
                        if "add_nodes_columns" in src_join_edge_schema:
                            for add_node_columns in src_join_edge_schema["add_nodes_columns"]:
                                feat_source_config["col_aliases"].extend(add_node_columns)
                        feat_source_config["col_aliases"].append([raw_feat_col, renamed_feat_col])
                        for time_col in graph.graph_time_cols_alias:
                            feat_source_config["col_aliases"].append([time_col, time_col])

                        # 记录用于join的id列 
                        feat_source_config["join_cols"] = []
                        for column, alias in feat_source_config["col_aliases"]:
                            if alias != renamed_feat_col:
                                feat_source_config["join_cols"].append(alias)

                        # 记录最终添加的特征向量列
                        feat_source_config["feat_vec_col"] = renamed_feat_col

                        # 记录最终添加的特征向量列的维度
                        feat_source_config["feat_vec_len"] = len(graph.edges[entry["edge_type"]]["graph_token_feat_cols"])
                        
                        # 记录该join_edges目前得到的特征列
                        query_config["feat_cols_sizes"][renamed_feat_col] = feat_source_config["feat_vec_len"]
                    
                    # 依次记录各个特征来源
                    join_edges_edge_config["feat_add_srcs"].append(feat_source_config)
        
        # 记录join_edges_edge_config
        query_config["join_edge_list"].append(join_edges_edge_config)
    
    # 获得join_edges_agg的配置
    if "join_edges_agg" in join_edges:
        query_config["join_edges_agg"] = {}

        agg_nodes_cols = join_edges["join_edges_agg"]["agg_nodes_cols"]
        query_config["join_edges_agg"]["group_cols"] = agg_nodes_cols + list(graph.graph_time_cols_alias)

        # 记录聚合后形成的特征列名和维度
        agg_feat_cols_sizes = {}
        
        query_config["join_edges_agg"]["agg_config"] = []
        for feat_vec_col in query_config["feat_cols_sizes"]:
            for agg_func in join_edges["join_edges_agg"]["agg_funcs"]:
                query_config["join_edges_agg"]["agg_config"].append([feat_vec_col, agg_func,
                                                                     f"{agg_func}_{feat_vec_col}"])
                agg_feat_cols_sizes[f"{agg_func}_{feat_vec_col}"] = query_config["feat_cols_sizes"][feat_vec_col]

        # 替换原始的特征列和维度
        query_config["feat_cols_sizes"] = agg_feat_cols_sizes
    
    return query_config

def join_edges_schema_init(graph, query_nodes_config, join_edges_schema, join_edges_default_config):
    # 先深拷贝一份join_edges_config作为初始的修正后的配置
    refined_schema = copy.deepcopy(join_edges_schema)

    # 记录序列化后的元素
    join_edges_seq = []

    # 将query_nodes作为第一个entry(node形式)
    for query_node_i in range(len(query_nodes_config["query_nodes_types"])):
        node_entry = {}
        node_entry["node_type"] = query_nodes_config["query_nodes_types"][query_node_i]
        node_entry["node_index"] = query_nodes_config["query_nodes_indexes"][query_node_i]

        node_col_len = len(graph.nodes[node_entry['node_type']]["node_col_types"])
        
        node_entry["node_cols"] = []
        for node_col_i in range(node_col_len):
            node_col = standard_node_col_name(node_entry['node_type'], node_entry['node_index'], node_col_i)
            node_entry["node_cols"].append(node_col)

        # 记录node对应的特征列名
        node_entry["node_feat_col"] = standard_feat_col_name("node", node_entry['node_type'], node_entry['node_index'])
        
        join_edges_seq.append(node_entry)

    # 记录每个元素间的相对关系对应的edge_list
    join_edges_adj = []
    
    for join_edge_i, join_edge_schema in enumerate(refined_schema):
        # 首先是名称
        join_edge_name = join_edge_schema["join_edge_name"]
        join_edge_schema["name"] = join_edge_name
    
        # 设置采样结果的存储路径
        if "join_edge_root_path" in join_edge_schema and join_edge_schema["join_edge_root_path"] != "":
            join_edge_schema["join_edge_path"] = join_edge_schema["join_edge_root_path"] + f"/{join_edge_name}"
        else:
            join_edge_schema["join_edge_path"] = join_edges_default_config["join_edge_root_path"] + f"/{join_edge_name}"
    
        # 如果未给出采样结果表格对应格式，则用默认值填充
        if "join_edge_table_format" not in join_edge_schema or join_edge_schema["join_edge_table_format"] == "":
            join_edge_schema["join_edge_table_format"] = join_edges_default_config["join_edge_table_format"]
        else:
            join_edge_schema["join_edge_table_format"] = join_edge_schema["join_edge_table_format"]
        
        # 获得对应的edge相关信息
        edge_type = join_edge_schema["edge_type"]

        # 记录相关信息
        edge_entry = {}
        edge_entry["edge_type"] = edge_type
        edge_entry["edge_index"] = join_edge_i
        edge_entry["edge_feat_col"] = standard_feat_col_name("edge", edge_type, join_edge_i)
        edge_entry["linked_node_indexes"] = []
        
        # 获得该边是第几个元素
        edge_entry_i = len(join_edges_seq)

        # 并获得该edge表对应的linked_node_types
        linked_node_types = graph.edges[edge_type]["linked_node_types"]

        # 得记录edge列对应的特征别名
        join_edge_schema["edge_feat_col"] = edge_entry["edge_feat_col"]
        
        # 记录node在join_edges中的index到在graph_token中的index
        join_edge_schema["node_index_map"] = {}

        # 记录要用于join的节点列的原始列名和转化后的列名
        join_edge_schema["join_nodes_columns"] = []

        # 记录该join_edge中包含的节点entry，用于node_limit的检查
        join_edge_node_entries = []
        
        # 然后添加join_nodes相关信息，并检查是否在之前的序列中
        for join_node_i in range(len(join_edge_schema["join_nodes_types"])):
            node_entry = {}
            node_entry["node_type"] = join_edge_schema["join_nodes_types"][join_node_i]
            node_entry["node_index"] = join_edge_schema["join_nodes_indexes"][join_node_i]
            
            # 获得该节点在边中的序号
            linked_node_index = join_edge_schema["join_nodes_edge_indexes"][join_node_i]

            # 获得该节点类型对应的列数目
            node_col_len = len(graph.nodes[node_entry['node_type']]["node_col_types"])

            # 查看表格中是否有对应的linked_node_index
            if linked_node_index in list(range(len(linked_node_types))):
                # 查看对应的node_type是否一致
                if linked_node_types[linked_node_index] != node_entry["node_type"]:
                    raise ValueError(f"Join node type {node_entry['node_type']} doesn't correspond to the node type "
                                     f"{linked_node_types[linked_node_index]} in edge table {edge_table_name} "
                                     f"of edge type {edge_type}")

                # 记录映射关系
                join_edge_schema["node_index_map"][node_entry["node_index"]] = linked_node_index
                
                # 获得该边对应的graph_token下该节点各列的列名
                graph_token_node_cols = []
                for node_col_i in range(node_col_len):
                    graph_token_node_col = graph_token_node_col_name(node_entry['node_type'], linked_node_index, node_col_i)
                    graph_token_node_cols.append(graph_token_node_col)

            else:
                raise ValueError(f"Join node edge index {linked_node_index} doesn't exist in edge table "
                                 f"{edge_table_name} of edge type {edge_type}")
                
            # 生成对应的节点列最终列名
            renamed_node_cols = []
            for node_col_i in range(node_col_len):
                renamed_node_col = standard_node_col_name(node_entry['node_type'], node_entry['node_index'], node_col_i)
                renamed_node_cols.append(renamed_node_col)

            # 记录graph_token中的节点列到join_edges上的映射关系(算join_edge不用改变列名)
            join_edge_schema["join_nodes_columns"].append([[x, y] for x, y in zip(graph_token_node_cols, renamed_node_cols)])

            # 记录node对应的标准化后的node_columns
            node_entry["node_cols"] = renamed_node_cols

            # 记录node对应的特征列名
            node_entry["node_feat_col"] = standard_feat_col_name("node", node_entry['node_type'], node_entry['node_index'])
            
            # 查找对应的node_entry序号
            node_entry_i = next((i for i, e in enumerate(join_edges_seq) if e == node_entry), -1)
            if node_entry_i == -1:
                raise ValueError(f"Join node {node_entry['node_type']} index {node_entry['node_index']} doesn't "
                                 f"exist in the previous join-edges")

            # 记录该node_index到边中
            edge_entry["linked_node_indexes"].append(node_entry["node_index"])
            
            # 增加对应的边
            join_edges_adj.append([node_entry_i, edge_entry_i])

            # 记录该node_entry 
            join_edge_node_entries.append(node_entry)
        
        # 再添加edge相关信息到序列中
        join_edges_seq.append(edge_entry)

        # 然后添加add_nodes相关信息
        if "add_nodes_types" in join_edge_schema:
            join_edge_schema["add_nodes_columns"] = []
            for add_node_i in range(len(join_edge_schema["add_nodes_types"])):
                # 获得该点相关信息
                node_entry = {}
                node_entry["node_type"] = join_edge_schema["add_nodes_types"][add_node_i]
                node_entry["node_index"] = join_edge_schema["add_nodes_indexes"][add_node_i]

                # 获得该节点在边中的序号
                linked_node_index = join_edge_schema["add_nodes_edge_indexes"][add_node_i]
                
                # 获得该节点类型对应的列数目
                node_col_len = len(graph.nodes[node_entry['node_type']]["node_col_types"])

                # 查看表格中是否有对应的linked_node_index
                if linked_node_index in list(range(len(linked_node_types))):
                    # 查看对应的node_type是否一致
                    if linked_node_types[linked_node_index] != node_entry["node_type"]:
                        raise ValueError(f"Add node type {node_entry['node_type']} doesn't correspond to the node type "
                                         f"{linked_node_types[linked_node_index]} in edge table {edge_table_name} "
                                         f"of edge type {edge_type}")

                    # 记录映射关系
                    join_edge_schema["node_index_map"][node_entry["node_index"]] = linked_node_index
                    
                    # 获得该边对应的graph_token下该节点各列的列名
                    graph_token_node_cols = []
                    for node_col_i in range(node_col_len):
                        graph_token_node_col = graph_token_node_col_name(node_entry['node_type'], linked_node_index, node_col_i)
                        graph_token_node_cols.append(graph_token_node_col)
                    
                else:
                    raise ValueError(f"Add node edge index {linked_node_index} doesn't exist in edge table "
                                     f"{edge_table_name} of edge type {edge_type}")

                # 生成对应的节点列最终列名
                renamed_node_cols = []
                for node_col_i in range(node_col_len):
                    renamed_node_col = standard_node_col_name(node_entry['node_type'], node_entry['node_index'], node_col_i)
                    renamed_node_cols.append(renamed_node_col)

                # 更新原始的join_nodes_columns
                join_edge_schema["add_nodes_columns"].append([[x, y] for x, y in zip(graph_token_node_cols, renamed_node_cols)])

                # 记录node对应的标准化后的node_columns
                node_entry["node_cols"] = renamed_node_cols

                # 记录node对应的特征列名
                node_entry["node_feat_col"] = standard_feat_col_name("node", node_entry['node_type'], node_entry['node_index'])
                
                # 获得该点是第几个元素
                node_entry_i = len(join_edges_seq)

                # 将该点计入序列
                join_edges_seq.append(node_entry)

                # 记录该node_index到边中
                edge_entry["linked_node_indexes"].append(node_entry["node_index"])
                
                # 增加对应的边
                join_edges_adj.append([edge_entry_i, node_entry_i])

                # 记录该node_entry 
                join_edge_node_entries.append(node_entry)

        # 检查edge_limit的相关情况 
        if "edge_limit_schema" in join_edge_schema:
            # 获得feat_col在feat_vec中对应的序号 
            for feat_col in join_edge_schema["edge_limit_schema"]["feat_cols"]:
                if feat_col not in graph.edges[edge_type]["graph_token_feat_cols"]:
                    raise ValueError(f"Limit edge feat {feat_col} doesn't exist in graph token features of "
                                     f"edge type {edge_type}")
                        
        # 如果有edge_sample，则检查相关节点，并转换对应的节点列
        if "edge_samples" in join_edge_schema:
            for edge_sample in join_edge_schema["edge_samples"]:
                edge_sample["sample_nodes_cols"] = []
                for sample_node_i in range(len(edge_sample["sample_nodes_types"])):
                    sample_node_type = edge_sample["sample_nodes_types"][sample_node_i]
                    sample_node_edge_index = edge_sample["sample_nodes_edge_indexes"][sample_node_i]

                    # 获得该node type对应的node col types，来获得该node对应几个node_col
                    node_col_target_types = graph.nodes[sample_node_type]["node_col_types"]
                    
                    # 生成对应的graph_token中的列名
                    graph_token_node_cols = []
                    for node_col_i in range(len(node_col_target_types)):
                        graph_token_node_col = graph_token_node_col_name(sample_node_type, sample_node_edge_index, node_col_i)
                        graph_token_node_cols.append(graph_token_node_col)

                    # 记录对应的列名
                    edge_sample["sample_nodes_cols"].extend(graph_token_node_cols)

        # 接着检查node_limit中的节点
        if "nodes_limit_schema" in join_edge_schema:
            for node_limit_schema in join_edge_schema["nodes_limit_schema"]:
                limit_node_type = node_limit_schema["node_type"]
                limit_node_index = node_limit_schema["node_index"]
                
                # 检查该node_type和node_index是否和该join_edges中某个node实际值对应
                node_exist_mark = False
                for node_entry in join_edge_node_entries:
                    if node_entry["node_type"] == limit_node_type and node_entry["node_index"] == limit_node_index:
                        node_exist_mark = True
                        
                if not node_exist_mark:
                    raise ValueError(f"Limit node type {limit_node_type} with index {limit_node_index} doesn't exist in "
                                     f"the current join_edge")

                # 获得feat_col在feat_vec中对应的序号 
                for feat_col in node_limit_schema["feat_cols"]:
                    if feat_col not in graph.nodes[limit_node_type]["graph_token_feat_cols"]:
                        raise ValueError(f"Limit node feat {feat_col} doesn't exist in graph token features of "
                                         f"node type {limit_node_type}")
                    
                # 获得该节点类型对应的列数目
                limit_node_col_len = len(graph.nodes[limit_node_type]["node_col_types"])

                # 生成该node在graph_token表中对应的列名(该表只有一个点，节点表的node_index设为none) 
                graph_token_node_cols = []
                for node_col_i in range(limit_node_col_len):
                    graph_token_node_col = graph_token_node_col_name(limit_node_type, None, node_col_i)
                    graph_token_node_cols.append(graph_token_node_col)

                # 获得该node在对应的edge中的graph_token中的index
                graph_token_node_index = join_edge_schema["node_index_map"][limit_node_index]
                
                # 生成对应的节点列最终列名
                renamed_node_cols = []
                for node_col_i in range(limit_node_col_len):
                    renamed_node_col = graph_token_node_col_name(limit_node_type, graph_token_node_index, node_col_i)
                    renamed_node_cols.append(renamed_node_col)

                # 更新原始的join_nodes_columns
                node_limit_schema["node_columns"] = [[x, y] for x, y in zip(graph_token_node_cols, renamed_node_cols)]

        # join_edge_limit 
        if "join_edge_limit" in join_edge_schema:
            # 为每个feat_source里的配置信息增加原始表的节点列和join_edge上对应列的列名的映射关系
            for feat_source in join_edge_schema["join_edge_limit"]["feat_sources"]:
                src_token_type = feat_source["token_type"]
                src_node_edge_type = feat_source["node_edge_type"]
                src_node_edge_index = feat_source["node_edge_index"]

                # 先检查是否有对应的entry
                node_edge_exist_mark = False
                for node_edge_entry in join_edges_seq:
                    if (src_token_type == "edge" and "edge_type" in node_edge_entry and 
                        node_edge_entry["edge_type"] == src_node_edge_type and 
                        node_edge_entry["edge_index"] == src_node_edge_index):
                        node_edge_exist_mark = True
                    elif (src_token_type == "node" and "node_type" in node_edge_entry and 
                        node_edge_entry["node_type"] == src_node_edge_type and 
                        node_edge_entry["node_index"] == src_node_edge_index):
                        node_edge_exist_mark = True

                if not node_edge_exist_mark:
                    raise ValueError(f"Limit {src_token_type} type {src_node_edge_type} with index {src_node_edge_index} "
                                     f"doesn't exist in the current join_edge")

                # 获得feat_col在feat_vec中对应的序号 
                for feat_col in feat_source["feat_cols"]:
                    if src_token_type == "edge":
                        if feat_col not in graph.edges[src_node_edge_type]["graph_token_feat_cols"]:
                            raise ValueError(f"Limit edge feat {feat_col} doesn't exist in graph token features of "
                                             f"edge type {src_node_edge_type}")
                    elif src_token_type == "node":
                        if feat_col not in graph.nodes[src_node_edge_type]["graph_token_feat_cols"]:
                            raise ValueError(f"Limit node feat {feat_col} doesn't exist in graph token features of "
                                             f"node type {src_node_edge_type}")

                # 如果来源是节点表，则获得对应的节点列到边上的节点列的映射关系，来源是边就不需要了  
                if src_token_type == "node":
                    # 获得该节点类型对应的列数目
                    limit_node_col_len = len(graph.nodes[src_node_edge_type]["node_col_types"])

                    # 获得该node在对应的edge中的graph_token中的index
                    graph_token_node_index = join_edge_schema["node_index_map"][src_node_edge_index]
                    
                    # 生成该node在graph_token表中以及join_edge中对应的列名(该表只有一个点，节点表的node_index设为none) 
                    graph_token_node_cols = []
                    renamed_node_cols = []
                    for node_col_i in range(limit_node_col_len):
                        graph_token_node_col = graph_token_node_col_name(src_node_edge_type, None, node_col_i)
                        graph_token_node_cols.append(graph_token_node_col)
                    
                        renamed_node_col = graph_token_node_col_name(src_node_edge_type, graph_token_node_index, node_col_i)
                        renamed_node_cols.append(renamed_node_col)
    
                    # 更新原始的join_nodes_columns
                    feat_source["node_columns"] = [[x, y] for x, y in zip(graph_token_node_cols, renamed_node_cols)]
        
        # 如果有join_edge_sample，则检查相关节点，并转换对应的节点列
        if "join_edge_samples" in join_edge_schema:
            for join_edge_sample in join_edge_schema["join_edge_samples"]:
                join_edge_sample["sample_nodes_cols"] = []
                for sample_node_i in range(len(join_edge_sample["sample_nodes_types"])):
                    sample_node_type = join_edge_sample["sample_nodes_types"][sample_node_i]
                    sample_node_index = join_edge_sample["sample_nodes_indexes"][sample_node_i]

                    # 获得该node type对应的node col types
                    node_col_target_types = graph.nodes[sample_node_type]["node_col_types"]

                    # 获得该node对应在edge中的index
                    graph_token_node_index = join_edge_schema["node_index_map"][sample_node_index]
                    
                    # 生成对应的节点列最终列名
                    renamed_node_cols = []
                    for node_col_i in range(len(node_col_target_types)):
                        renamed_node_col = graph_token_node_col_name(sample_node_type, graph_token_node_index, node_col_i)
                        renamed_node_cols.append(renamed_node_col)

                    # 记录对应的列名
                    join_edge_sample["sample_nodes_cols"].extend(renamed_node_cols)

        # join_edges_limit 
        if "join_edges_limit" in join_edge_schema:
            # 为每个feat_source里的配置信息增加原始表的节点列和join_edge上对应列的列名的映射关系
            for feat_source in join_edge_schema["join_edges_limit"]["feat_sources"]:
                src_token_type = feat_source["token_type"]
                src_node_edge_type = feat_source["node_edge_type"]
                src_node_edge_index = feat_source["node_edge_index"]

                # 先检查是否有对应的entry
                node_edge_exist_mark = False
                for node_edge_entry in join_edges_seq:
                    if (src_token_type == "edge" and "edge_type" in node_edge_entry and 
                        node_edge_entry["edge_type"] == src_node_edge_type and 
                        node_edge_entry["edge_index"] == src_node_edge_index):
                        node_edge_exist_mark = True
                    elif (src_token_type == "node" and "node_type" in node_edge_entry and 
                        node_edge_entry["node_type"] == src_node_edge_type and 
                        node_edge_entry["node_index"] == src_node_edge_index):
                        node_edge_exist_mark = True

                if not node_edge_exist_mark:
                    raise ValueError(f"Limit {src_token_type} type {src_node_edge_type} with index {src_node_edge_index} "
                                     f"doesn't exist in the current join_edges")

                # 获得feat_col在feat_vec中对应的序号 
                if src_token_type == "edge":
                    for feat_col in feat_source["feat_cols"]:
                        if feat_col not in graph.edges[src_node_edge_type]["graph_token_feat_cols"]:
                            raise ValueError(f"Limit edge feat {feat_col} doesn't exist in graph token features of "
                                             f"edge type {src_node_edge_type}")

                    # 记录对应的join_edge在join_edges中的序号
                    feat_source["join_edge_index"] = src_node_edge_index
                
                elif src_token_type == "node":
                    for feat_col in feat_source["feat_cols"]:
                        if feat_col not in graph.nodes[src_node_edge_type]["graph_token_feat_cols"]:
                            raise ValueError(f"Limit node feat {feat_col} doesn't exist in graph token features of "
                                             f"node type {src_node_edge_type}")

                    # 获得该节点类型对应的列数目
                    limit_node_col_len = len(graph.nodes[src_node_edge_type]["node_col_types"])

                    # 生成该node在graph_token表中以及join_edge中对应的列名(该表只有一个点，节点表的node_index设为none) 
                    graph_token_node_cols = []
                    renamed_node_cols = []
                    for node_col_i in range(limit_node_col_len):
                        graph_token_node_col = graph_token_node_col_name(src_node_edge_type, None, node_col_i)
                        graph_token_node_cols.append(graph_token_node_col)
                    
                        renamed_node_col = standard_node_col_name(src_node_edge_type, src_node_edge_index, node_col_i)
                        renamed_node_cols.append(renamed_node_col)
    
                    # 更新原始的join_nodes_columns
                    feat_source["node_columns"] = [[x, y] for x, y in zip(graph_token_node_cols, renamed_node_cols)]
                    
        # 如果有join_edges_sample，则检查相关节点，并转换对应的节点列
        if "join_edges_samples" in join_edge_schema:
            for join_edges_sample in join_edge_schema["join_edges_samples"]:
                join_edges_sample["sample_nodes_cols"] = []
                for sample_node_i in range(len(join_edges_sample["sample_nodes_types"])):
                    sample_node_type = join_edges_sample["sample_nodes_types"][sample_node_i]
                    sample_node_index = join_edges_sample["sample_nodes_indexes"][sample_node_i]

                    # 获得该node type对应的node col types
                    node_col_target_types = graph.nodes[sample_node_type]["node_col_types"]

                    # 生成对应的节点列最终列名
                    renamed_node_cols = []
                    for node_col_i in range(len(node_col_target_types)):
                        renamed_node_col = standard_node_col_name(sample_node_type, sample_node_index, node_col_i)
                        renamed_node_cols.append(renamed_node_col)

                    # 记录对应的列名
                    join_edges_sample["sample_nodes_cols"].extend(renamed_node_cols)

        # 检查是否指定了join方式 
        if "edge_join_type" not in join_edge_schema:
            # 没有则设定为inner
            join_edge_schema["edge_join_type"] = "inner"
    
        # 对重新分区的相关配置的检查 

    # 记录对应的结果 
    # seq里面的类型能用来做segment embed
    join_edges_info = {}
    join_edges_info["join_edges_schema"] = refined_schema
    join_edges_info["flatten_format"] = {}
    join_edges_info["flatten_format"]["seq"] = join_edges_seq
    join_edges_info["flatten_format"]["adj"] = join_edges_adj
    
    return join_edges_info
    
# 先进行表维度的聚合，按表维度顺序表示，再加上基于join顺序的位置编码（不行，join能连接到多个表）
# adj是否该有方向？
# 如何转化为1维向量
# 边和node的包含关系算一种segment:主要是这里面的多重覆盖问题
# 一个表算一组mask，join
# 本身的顺序算一种位置编码
# 往前join到哪个edge算是另一种position编码
# 聚合多个join_edges应该每个都作为一个segment

# 3种edge:边包含点，点归属边，边连接边（直接相加，均值，拼接都应该试一下，到时根据需求调整代码）