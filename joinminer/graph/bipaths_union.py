import logging
from pyspark.sql.functions import col

# 获得logger
logger = logging.getLogger(__name__)

def bipaths_union(spark, graph, bipaths):
    # 获得这些bipaths对应的基础信息
    query_nodes_types = join_edges_list["query_nodes_types"]
    query_nodes_indexes = join_edges_list["query_nodes_indexes"]
    query_nodes_cols = join_edges_list["query_nodes_cols"]
    query_nodes_cols_alias = join_edges_list["query_nodes_cols_alias"]
    query_nodes_join_cols = join_edges_list["query_nodes_join_cols"]
    query_nodes_feat_cols = join_edges_list["query_nodes_feat_cols"]
    join_edges_list_name = join_edges_list["join_edges_list_name"]
    
    logger.info(f"Collect instances of join_edges_list {join_edges_list_name} for query nodes types "
                f"{query_nodes_types} of indexes {query_nodes_indexes} and cols {query_nodes_cols}")

    # 检查对应结果是否存在

    # 依次获得各个bipath对应的数据

        # 获得对应的forward_path的全量数据

        # 获得对应的backward_path的全量数据

        # Join forward_path和backward_path

        # 只保留要union的id列并修正列名

        # 记录该bipath对应的要union的数据

    # union全部bipath的结果

    # 保存结果

    # 读取union的结果
    
    return