import copy
import logging
from pyspark.sql.functions import col, when, count

from .time_aggregation import time_aggs_init
from ..python import time_costing
from ..hdfs import hdfs_list_contents, hdfs_read_json, hdfs_check_file_exists

# 获得logger
logger = logging.getLogger(__name__)

table_schema_file_name = "_Table_Schema"

# 每个节点列在graph_token表中对应的标准名称
def graph_token_node_col_name(node_type, node_index, col_index):
    if node_index is not None:
        return f"index_{node_index}_node_{node_type}_col_{col_index}"
    else:
        return f"node_{node_type}_col_{col_index}"
        
class TableGraph():
    @time_costing
    def __init__(self, graph_config):
        # 查看该图是否有时间列
        if "graph_time_cols_alias" in graph_config:
            # 记录图对应的时间列的统一别名，以及对应的时间格式
            self.graph_time_cols_alias = graph_config["graph_time_cols_alias"]
            self.graph_time_cols_formats = graph_config["graph_time_cols_formats"]
        else:
            self.graph_time_cols_alias = []
            self.graph_time_cols_formats = []

        # 获得图中节点对应的配置
        self.nodes = self.nodes_init(graph_config["node_schemas"], graph_config["table_default_config"])
        
        # 获得图中边对应的配置
        self.edges = self.edges_init(graph_config["node_schemas"], graph_config["edge_schemas"], 
                                     graph_config["table_default_config"])
        
        # 展示一个简单的总结信息
        self.show_brief_summary()
    
    def nodes_init(self, node_schemas, table_default_config):
        # 遍历全部节点类型
        for node_type in node_schemas:
            # 生成对应的query config
            query_config = {}
            
            # 获得各个edge_table汇总后形成的graph_token表的存储位置
            if "graph_token_root_path" in node_schemas[node_type] and node_schemas[node_type]["graph_token_root_path"] != "":
                query_config["result_path"] = node_schemas[node_type]["graph_token_root_path"] + f"/{node_type}"
            else:
                query_config["result_path"] = table_default_config["graph_token_root_path"] + f"/{node_type}"
            
            # 获得对应的graph_token表的存储格式
            if "graph_token_table_format" not in node_schemas[node_type] or node_schemas[node_type]["graph_token_table_format"] == "":
                query_config["result_format"] = table_default_config["graph_token_table_format"]
            else:
                query_config["result_format"] = node_schemas[node_type]["graph_token_table_format"]

            # 记录各个由time_agg得到的表的query_config
            query_config["time_agg"] = []

            # 记录统一的节点列
            query_config["node_cols"] = []
            for node_col_i in range(len(node_schemas[node_type]["node_col_types"])):
                node_col = graph_token_node_col_name(node_type, None, node_col_i)
                query_config["node_cols"].append(node_col)
                
            # 获得该node_type对应的node_col_types
            node_col_types = node_schemas[node_type]["node_col_types"]

            # 记录来源表累计特征数量
            node_schemas[node_type]["src_feat_count"] = 0

            # 记录该节点对应的graph_token表中特征向量的各列的来源信息
            node_schemas[node_type]["graph_token_feat_cols"] = []
            
            # 检查是否设定了node_tables，没有就补一个空值
            if "node_tables" not in node_schemas[node_type]:
                node_schemas[node_type]["node_tables"] = {}
            
            # 遍历该类型对应的全部节点表
            for node_table_name in node_schemas[node_type]["node_tables"]:
                # 记录src表对应的query config
                src_table_query_config = {}
                
                # 获得对应的schema
                node_table_schema = node_schemas[node_type]["node_tables"][node_table_name]
                
                # 获得对应路径
                if "src_table_root_path" in node_table_schema and node_table_schema["src_table_root_path"] != "":
                    src_table_path = node_table_schema["src_table_root_path"] + node_table_schema["src_table_rel_path"]
                else:
                    src_table_path = table_default_config["src_table_root_path"] + node_table_schema["src_table_rel_path"]
                
                # 保存对应路径
                src_table_query_config["source_path"] = src_table_path

                # 如果未给出表格对应格式，则用默认值填充
                if "src_table_format" not in node_table_schema or node_table_schema["src_table_format"] == "":
                    src_table_query_config["source_format"] = table_default_config["src_table_format"]
                else:
                    src_table_query_config["source_format"] = node_table_schema["src_table_format"]
                
                # 读取该表对应的schema
                table_schema = hdfs_read_json(src_table_path, table_schema_file_name)
                
                # 显示对应的节点列和节点类型
                logger.info(f"Node table {node_table_name} contains {len(table_schema['node_col_to_types'])} node columns")
                for node_col, node_col_type in table_schema["node_col_to_types"].items():
                    logger.info(f"Node column \"{node_col}\" corresponds to node type \"{node_col_type}\"")
                
                # 获得该表对应的node_columns
                node_table_node_columns = node_table_schema["node_cols"]
                
                # 查看设置的node_column是否都在对应表里
                node_table_node_types = []
                for node_table_node_column in node_table_node_columns:
                    if node_table_node_column not in table_schema["node_col_to_types"]:
                        # 不在则报错
                        raise ValueError(f"The node column \"{node_table_node_column}\" for node type \"{node_type}\" in "
                                         f"node table {node_table_name} doesn't exist.")
                    else:
                        # 获得这些node_columns对应的node_col_types
                        node_table_node_types.append(table_schema["node_col_to_types"][node_table_node_column])
                
                # 查看对应类型是否一致
                if node_table_node_types != node_col_types:
                    # 不一致则报错
                    raise ValueError(f"The node column types {node_table_node_types} for node columns {node_table_node_columns} in "
                                f"node table {node_table_name} doesn't equal to the configured node column types {node_col_types} "
                                f"of node type \"{node_type}\".")
                
                # 查看是否有额外的节点列
                table_node_columns = list(table_schema["node_col_to_types"].keys())
                if set(table_node_columns) != set(node_table_node_columns):
                    # 如果有额外的节点列，理论上应该查看nodes aggregation的设定才好(*待优化)
                    logger.warning(f"The node table {node_table_name} contains extra node columns {table_node_columns} "
                                  f"than configured node columns {node_table_node_columns}")

                # 记录src特征列
                node_table_schema["src_feat_cols"] = list(table_schema["feat_cols"])
                
                # 累计src特征数量
                node_schemas[node_type]["src_feat_count"] += len(table_schema["feat_cols"])

                # 显示对应的特征数量
                logger.info(f"Node table {node_table_name} contains {len(table_schema['feat_cols'])} src feature columns")
                
                # 记录时间列
                node_table_schema["time_cols"] = list(table_schema["time_cols"])

                # 记录各个time_col对应的format
                node_table_schema["time_cols_formats"] = list(table_schema["time_cols_formats"])

                # 配置null标注列的前缀
                null_mark_prefix = "null_mark_of"
                src_table_query_config["null_mark_prefix"] = null_mark_prefix
                
                # 记录加入null标注后的特征列
                node_table_schema["null_marked_feat_cols"] = list(table_schema["feat_cols"])
                for feat_col in table_schema["feat_cols"]:
                    node_table_schema["null_marked_feat_cols"].append(f"{null_mark_prefix}_{feat_col}")

                # 配置要从src_table里读取的列名及对应的别名 
                src_table_query_config["col_aliases"] = []
                for node_col_i, node_col in enumerate(node_table_schema["node_cols"]):
                    token_node_col = graph_token_node_col_name(node_type, None, node_col_i)
                    src_table_query_config["col_aliases"].append([node_col, token_node_col])
                for feat_col in table_schema["feat_cols"]:
                    src_table_query_config["col_aliases"].append([feat_col, feat_col])
                for time_col in table_schema["time_cols"]:
                    src_table_query_config["col_aliases"].append([time_col, time_col])

                # 配置读取出的node_columns的名称
                src_table_query_config["node_cols"] = []
                for node_col_i, node_col in enumerate(node_table_schema["node_cols"]):
                    token_node_col = graph_token_node_col_name(node_type, None, node_col_i)
                    src_table_query_config["node_cols"].append(token_node_col)
                
                # 配置src_table中原始的时间列名和对应的格式，用于筛选目标分区，并转换为graph_time_format
                src_table_query_config["time_cols"] = list(table_schema["time_cols"])
                src_table_query_config["time_cols_formats"] = list(table_schema["time_cols_formats"])
                
                # 配置src_table中的原始特征列名，用于添加null标注列
                src_table_query_config["feat_cols"] = list(table_schema["feat_cols"])
                
                # 记录time_agg对应的query配置
                time_agg_query_config = {}

                # 先记录表名
                time_agg_query_config["name"] = node_table_name
                
                # 获得time_agg的结果路径
                if "time_agg_root_path" in node_table_schema and node_table_schema["time_agg_root_path"] != "":
                    time_agg_query_config["result_path"] = node_table_schema["time_agg_root_path"] + f"/{node_table_name}"
                else:
                    time_agg_query_config["result_path"] = table_default_config["time_agg_root_path"] + f"/{node_table_name}"

                # 获得time_agg的结果保存形式
                if "time_agg_table_format" not in node_table_schema or node_table_schema["time_agg_table_format"] == "":
                    time_agg_query_config["result_format"] = table_default_config["time_agg_table_format"]
                else:
                    time_agg_query_config["result_format"] = node_table_schema["time_agg_table_format"]

                # 加入用于time_agg的src_table的query_config
                time_agg_query_config["src_table"] = src_table_query_config
                
                # 检查是否有设定time aggregation的方式,没有则用默认值填充
                if "time_aggs_configs" not in node_table_schema:
                    time_aggs_configs = table_default_config["time_aggs_configs"]
                else:
                    time_aggs_configs = node_table_schema["time_aggs_configs"]
                    
                # 获得time_aggs具体的执行方式
                time_agg_query_config["time_aggs"] = time_aggs_init(node_table_schema["null_marked_feat_cols"], 
                                                                    time_aggs_configs)

                # 统计经过time_aggregation后形成的特征名称，用于将这些组合成特征向量
                time_agg_query_config["time_aggs_feat_cols"] = []
                for time_agg in time_agg_query_config["time_aggs"]:
                    for time_range in time_agg["time_ranges"]:
                        time_agg_query_config["time_aggs_feat_cols"].extend(time_range["agg_feat_cols"])

                # 获得time_agg后要向量化的特征列名
                time_agg_query_config["time_aggs_feat_vec"] = f"{node_table_name}_feat_vec"

                # 获得对该表中是否有目标数据的标记的列名
                time_agg_query_config["null_mark_col"] = f"{null_mark_prefix}_{node_table_name}"
                
                # 记录最终形成的time_agg配置
                query_config["time_agg"].append(time_agg_query_config)
                
                # 显示对应的特征数量
                logger.info(f"After time aggregation, node table {node_table_name} contains "
                            f"{len(time_agg_query_config['time_aggs_feat_cols'])} feature columns")

                # 累积形成的特征列到graph_token中对应的特征列中
                for feat_col in time_agg_query_config["time_aggs_feat_cols"]:
                    node_schemas[node_type]["graph_token_feat_cols"].append(f"{node_table_name}_{feat_col}")
                
                # 还要加入对该表中是否存在对应记录的标记（0表示不存在）
                node_schemas[node_type]["graph_token_feat_cols"].append(f"{null_mark_prefix}_{node_table_name}")

            # 记录将全部time_agg的特征列合并后全部的特征列名(用于决定存储文件大小)
            query_config["assembling_feat_cols"] = node_schemas[node_type]["graph_token_feat_cols"]
            
            # 记录将全部time_agg的特征列合并后形成的结果列名  
            query_config["assembled_feat_col"] = f"{node_type}_feat_col"
            
            # 保存最终获得的完整的query_config
            node_schemas[node_type]["query_config"] = query_config 
            
            # 显示该graph_token对应的特征总数量
            logger.info(f"Graph token of node type {node_type} contains "
                        f"{len(node_schemas[node_type]['graph_token_feat_cols'])} feature columns")
        
        return node_schemas
    
    def edges_init(self, node_schemas, edge_schemas, table_default_config):
        # 遍历全部边类型
        for edge_type in edge_schemas:
            # 生成对应的query config
            query_config = {}
            
            # 获得各个edge_table汇总后形成的graph_token表的存储位置
            if "graph_token_root_path" in edge_schemas[edge_type] and edge_schemas[edge_type]["graph_token_root_path"] != "":
                query_config["result_path"] = edge_schemas[edge_type]["graph_token_root_path"] + f"/{edge_type}"
            else:
                query_config["result_path"] = table_default_config["graph_token_root_path"] + f"/{edge_type}"
            
            # 获得对应的graph_token表的存储格式
            if "graph_token_table_format" not in edge_schemas[edge_type] or edge_schemas[edge_type]["graph_token_table_format"] == "":
                query_config["result_format"] = table_default_config["graph_token_table_format"]
            else:
                query_config["result_format"] = edge_schemas[edge_type]["graph_token_table_format"]

            # 记录各个由time_agg得到的表的query_config
            query_config["time_agg"] = []

            # 记录统一的节点列，并获得包含的node_type对应的各个node_col_types
            query_config["node_cols"] = []
            edge_schemas[edge_type]["linked_node_col_types"] = []
            for node_i, node_type in enumerate(edge_schemas[edge_type]["linked_node_types"]):
                if node_type not in node_schemas:
                    # 不在则报错
                    raise ValueError(f"The node type \"{node_type}\" in edge type \"{edge_type}\" doesn't have the "
                                     f"corresponding node_col_types.")
                    
                for node_col_i in range(len(node_schemas[node_type]["node_col_types"])):
                    token_node_col = graph_token_node_col_name(node_type, node_i, node_col_i)
                    query_config["node_cols"].append(token_node_col)

                node_col_types = node_schemas[node_type]["node_col_types"]
                edge_schemas[edge_type]["linked_node_col_types"].append(node_col_types)
                
            # 记录来源表累计特征数量
            edge_schemas[edge_type]["src_feat_count"] = 0

            # 记录该边对应的graph_token表中特征向量的各列的来源信息
            edge_schemas[edge_type]["graph_token_feat_cols"] = []
            
            # 遍历该类型对应的全部节点表
            for edge_table_name in edge_schemas[edge_type]["edge_tables"]:
                # 记录src表对应的query config
                src_table_query_config = {}
                
                # 获得对应的schema
                edge_table_schema = edge_schemas[edge_type]["edge_tables"][edge_table_name]

                # 获得对应路径
                if "src_table_root_path" in edge_table_schema and edge_table_schema["src_table_root_path"] != "":
                    src_table_path = edge_table_schema["src_table_root_path"] + edge_table_schema["src_table_rel_path"]
                else:
                    src_table_path = table_default_config["src_table_root_path"] + edge_table_schema["src_table_rel_path"]

                # 保存对应路径
                src_table_query_config["source_path"] = src_table_path

                # 如果未给出表格对应格式，则用默认值填充
                if "src_table_format" not in edge_table_schema or edge_table_schema["src_table_format"] == "":
                    src_table_query_config["source_format"] = table_default_config["src_table_format"]
                else:
                    src_table_query_config["source_format"] = edge_table_schema["src_table_format"]
                
                # 读取该表对应的schema
                table_schema = hdfs_read_json(src_table_path, table_schema_file_name)

                # 显示对应的节点列和节点类型
                logger.info(f"Edge table {edge_table_name} contains {len(table_schema['node_col_to_types'])} node columns")
                for node_col, node_col_type in table_schema["node_col_to_types"].items():
                    logger.info(f"Node column \"{node_col}\" corresponds to node column type \"{node_col_type}\"")
                
                # 一次获得该表各个node_type对应的node_columns，检查是否符合要求
                linked_node_columns = []
                for node_type_i, node_type_cols in enumerate(edge_table_schema["linked_node_types_cols"]):
                    # 记录涉及到的node_type_cols
                    linked_node_columns.extend(node_type_cols)
                    
                    # 先查看这些node_column是否都在对应表里，并记录表里设置的node_col_type
                    node_col_types = []
                    for node_col in node_type_cols:
                        if node_col not in table_schema["node_col_to_types"]:
                            # 不在则报错
                            raise ValueError(f"The node column \"{node_col}\" for edge type \"{edge_type}\" in "
                                        f"edge table {edge_table_name} doesn't exist.")
                        else:
                            # 获得这些node_columns对应的node_col_types
                            node_col_types.append(table_schema["node_col_to_types"][node_col])

                    # 查看对应类型是否一致
                    if node_col_types != edge_schemas[edge_type]["linked_node_col_types"][node_type_i]:
                        # 不一致则报错
                        raise ValueError(f"The node column types {node_col_types} for node columns {node_type_cols} "
                                    f"in edge table {edge_table_name} doesn't equal to the configured node"
                                    f"column types {edge_schemas[edge_type]['linked_node_col_types'][node_type_i]} of "
                                    f"node type \"{edge_schemas[edge_type]['linked_node_types'][node_type_i]}\".")
                
                # 查看是否有额外的节点列
                table_node_columns = list(table_schema["node_col_to_types"].keys())
                if set(table_node_columns) != set(linked_node_columns):
                    logger.warning(f"The edge table {edge_table_name} contains extra node columns {table_node_columns} "
                                  f"than configured node columns {linked_node_columns}")

                # 如果有额外的节点列，理论上应该设定nodes aggregation才好(*待优化) 

                # 统计特征数量
                edge_schemas[edge_type]["src_feat_count"] += len(table_schema["feat_cols"])
                
                # 显示对应的特征数量
                logger.info(f"Edge table {edge_table_name} contains {len(table_schema['feat_cols'])} feature columns")

                # 记录特征列
                edge_table_schema["feat_cols"] = list(table_schema["feat_cols"])
                
                # 记录时间列
                edge_table_schema["time_cols"] = list(table_schema["time_cols"])
        
                # 记录各个时间列对应的format
                edge_table_schema["time_cols_formats"] = list(table_schema["time_cols_formats"])

                # 配置null标注列的前缀
                null_mark_prefix = "null_mark_of"
                src_table_query_config["null_mark_prefix"] = null_mark_prefix
                
                # 记录加入null标注后的特征列
                edge_table_schema["null_marked_feat_cols"] = list(table_schema["feat_cols"])
                for feat_col in table_schema["feat_cols"]:
                    edge_table_schema["null_marked_feat_cols"].append(f"{null_mark_prefix}_{feat_col}")

                # 配置要从src_table里读取的列名及对应的别名 
                src_table_query_config["col_aliases"] = []
                for node_i, node_cols in enumerate(edge_table_schema["linked_node_types_cols"]):
                    node_type = edge_schemas[edge_type]["linked_node_types"][node_i]
                    for node_col_i, node_col in enumerate(node_cols):
                        token_node_col = graph_token_node_col_name(node_type, node_i, node_col_i)
                        src_table_query_config["col_aliases"].append([node_col, token_node_col])
                for feat_col in table_schema["feat_cols"]:
                    src_table_query_config["col_aliases"].append([feat_col, feat_col])
                for time_col in table_schema["time_cols"]:
                    src_table_query_config["col_aliases"].append([time_col, time_col])

                # 配置读取出的node_columns的名称
                src_table_query_config["node_cols"] = []
                for node_i, node_cols in enumerate(edge_table_schema["linked_node_types_cols"]):
                    node_type = edge_schemas[edge_type]["linked_node_types"][node_i]
                    for node_col_i, node_col in enumerate(node_cols):
                        token_node_col = graph_token_node_col_name(node_type, node_i, node_col_i)
                        src_table_query_config["node_cols"].append(token_node_col)
                
                # 配置src_table中原始的时间列名和对应的格式，用于筛选目标分区，并转换为graph_time_format
                src_table_query_config["time_cols"] = list(table_schema["time_cols"])
                src_table_query_config["time_cols_formats"] = list(table_schema["time_cols_formats"])
                
                # 配置src_table中的原始特征列名，用于添加null标注列
                src_table_query_config["feat_cols"] = list(table_schema["feat_cols"])

                # 记录time_agg对应的query配置
                time_agg_query_config = {}

                # 先记录表名
                time_agg_query_config["name"] = edge_table_name
                
                # 获得time_agg的结果路径
                if "time_agg_root_path" in edge_table_schema and edge_table_schema["time_agg_root_path"] != "":
                    time_agg_query_config["result_path"] = edge_table_schema["time_agg_root_path"] + f"/{edge_table_name}"
                else:
                    time_agg_query_config["result_path"] = table_default_config["time_agg_root_path"] + f"/{edge_table_name}"

                # 获得time_agg的结果保存形式
                if "time_agg_table_format" not in edge_table_schema or edge_table_schema["time_agg_table_format"] == "":
                    time_agg_query_config["result_format"] = table_default_config["time_agg_table_format"]
                else:
                    time_agg_query_config["result_format"] = edge_table_schema["time_agg_table_format"]

                # 加入用于time_agg的src_table的query_config
                time_agg_query_config["src_table"] = src_table_query_config

                # 检查是否有设定time aggregation的方式,没有则用默认值填充
                if "time_aggs_configs" not in edge_table_schema:
                    time_aggs_configs = table_default_config["time_aggs_configs"]
                else:
                    time_aggs_configs = edge_table_schema["time_aggs_configs"]

                # 获得time_aggs具体的执行方式
                time_agg_query_config["time_aggs"] = time_aggs_init(edge_table_schema["null_marked_feat_cols"], 
                                                                    time_aggs_configs)

                # 统计经过time_aggregation后形成的特征名称，用于将这些组合成特征向量
                time_agg_query_config["time_aggs_feat_cols"] = []
                for time_agg in time_agg_query_config["time_aggs"]:
                    for time_range in time_agg["time_ranges"]:
                        time_agg_query_config["time_aggs_feat_cols"].extend(time_range["agg_feat_cols"])

                # 获得time_agg后要向量化的特征列名
                time_agg_query_config["time_aggs_feat_vec"] = f"{edge_table_name}_feat_vec"

                # 获得对该表中是否有目标数据的标记的列名
                time_agg_query_config["null_mark_col"] = f"{null_mark_prefix}_{edge_table_name}"
                
                # 记录最终形成的time_agg配置
                query_config["time_agg"].append(time_agg_query_config)
                
                # 显示对应的特征数量
                logger.info(f"After time aggregation, edge table {edge_table_name} contains "
                            f"{len(time_agg_query_config['time_aggs_feat_cols'])} feature columns")

                # 累积形成的特征列到graph_token中对应的特征列中
                for feat_col in time_agg_query_config["time_aggs_feat_cols"]:
                    edge_schemas[edge_type]["graph_token_feat_cols"].append(f"{edge_table_name}_{feat_col}")

                # 还要加入对该表中是否存在对应记录的标记（0表示不存在）
                edge_schemas[edge_type]["graph_token_feat_cols"].append(f"{null_mark_prefix}_{edge_table_name}")

            # 记录将全部time_agg的特征列合并后全部的特征列名(用于决定存储文件大小)
            query_config["assembling_feat_cols"] = edge_schemas[edge_type]["graph_token_feat_cols"]
            
            # 记录将全部time_agg的特征列合并后形成的结果列名  
            query_config["assembled_feat_col"] = f"{edge_type}_feat_col"
            
            # 保存最终获得的完整的query_config
            edge_schemas[edge_type]["query_config"] = query_config 
            
            # 显示该graph_token对应的特征总数量
            logger.info(f"Graph token of edge type {edge_type} contains "
                        f"{len(edge_schemas[edge_type]['graph_token_feat_cols'])} feature columns")
        
        return edge_schemas
    
    # 显示图中的简单的统计信息
    def show_brief_summary(self):
        # 图是否有时间属性以及对应的目标时间
        logger.info(f"Graph with {len(self.graph_time_cols_alias)} time columns: {self.graph_time_cols_alias}.")
            
        # 有几种类型的点，各种节点有几种类型的节点表各节点表有多少个特征
        logger.info(f"Graph contains {len(self.nodes.keys())} types of nodes:")
        for node_type in self.nodes:
            node_table_names = list(self.nodes[node_type]['node_tables'].keys())
            node_src_feat_count = self.nodes[node_type]["src_feat_count"]
            node_graph_token_feat_count = len(self.nodes[node_type]["graph_token_feat_cols"])
            logger.info(f"Node type {node_type} with {len(node_table_names)} node tables: {node_table_names}, "
                        f"{node_src_feat_count} source node features and {node_graph_token_feat_count} "
                        f"graph token features")
            
        # 多少种边，每个节点有多少个特征表，分别有多少特征
        logger.info(f"Graph contains {len(self.edges.keys())} types of edges:")
        for edge_type in self.edges:
            edge_table_names = list(self.edges[edge_type]['edge_tables'].keys())
            edge_src_feat_count = self.edges[edge_type]["src_feat_count"]
            edge_graph_token_feat_count = len(self.edges[edge_type]["graph_token_feat_cols"])
            logger.info(f"Edge type {edge_type} with {len(edge_table_names)} edge tables: {edge_table_names}, "
                             f" {edge_src_feat_count} source edge features and {edge_graph_token_feat_count} "
                             f"graph token features")
        
        return
