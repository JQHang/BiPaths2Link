import copy
import json

from ..python import ensure_logger
from ..pyspark import identify_numeric_columns
from ..hdfs import hdfs_create_directory, hdfs_check_file_exists, hdfs_list_contents, hdfs_read_json

@ensure_logger
def ensure_complex_path_config(complex_path_config, father_path_config = None, path_result_dir=None, logger=None):
    """
    Ensure the completeness and validity of a complex path configuration.

    Args:
        complex_path_config (dict): The complex path configuration to be validated.
        father_path_config (dict, optional): The parent path configuration if it exists. Defaults to None.
        path_result_dir (str, optional): The path to the result directory. Defaults to None.
        logger (Logger, optional): The logger object for logging information. Defaults to None.

    Returns:
        dict: The validated complex path configuration.
    """

    if "path_schema" not in complex_path_config:
        raise KeyError("Require 'path_schema' for each path config")

    # Initialize sequence path and adjacency dictionary
    complex_path_config["seq_path"] = []
    complex_path_config["adj"] = {}

    # If there is a parent path configuration, ensure that tail node indexes do not overlap
    if father_path_config is not None:
        father_node_indexes = set()
        father_edge_count = 0
        for element_info in father_path_config["seq_path"]:
            if element_info[0] == "Node":
                father_node_indexes.add(element_info[2])
            elif element_info[0] == "Edge":
                father_edge_count = father_edge_count + 1
                
    # Iterate over each hop configuration
    for hop_k, hop_config in enumerate(complex_path_config["path_schema"]):
        logger.info(f"Check the {hop_k}-th hop config")

        required_keys = ["head_node_types", "head_node_indexes", "relation_list", "tail_node_types", "tail_node_indexes"]
        for required_key in required_keys:
            if required_key not in hop_config:
                raise KeyError(f"Require {required_key} for hop config")

        # Check consistency between the number of head node types and indexes
        if len(hop_config["head_node_types"]) != len(hop_config["head_node_indexes"]):
            raise ValueError("Require the same number of indexes for head nodes")

        # Check consistency between the number of tail node types and indexes
        if len(hop_config["tail_node_types"]) != len(hop_config["tail_node_indexes"]):
            raise ValueError("Require the same number of indexes for tail nodes")

        # Get existing node indexes
        existing_node_indexes = set()
        for element_info in complex_path_config["seq_path"]:
            if element_info[0] == "Node":
                existing_node_indexes.add(element_info[2])

        head_node_info_list = []
        for head_node_k in range(len(hop_config["head_node_types"])):
            head_node_info = ("Node", hop_config["head_node_types"][head_node_k], hop_config["head_node_indexes"][head_node_k])
            head_node_info_list.append(head_node_info)

            if hop_k == 0:
                complex_path_config["seq_path"].append(head_node_info)
                complex_path_config["adj"][head_node_info] = {"in": [], "out": []}

        tail_node_info_list = []
        for tail_node_k in range(len(hop_config["tail_node_indexes"])):
            tail_node_info = ("Node", hop_config["tail_node_types"][tail_node_k], hop_config["tail_node_indexes"][tail_node_k])
            tail_node_info_list.append(tail_node_info)
            complex_path_config["adj"][tail_node_info] = {"in": [], "out": []}

        # Check if head node indexes already exist
        if hop_k > 0 and not set(hop_config["head_node_indexes"]).issubset(existing_node_indexes):
            raise ValueError("Require the existing head node indexes")

        # Check if tail node indexes do not already exist
        if set(hop_config["tail_node_indexes"]) & (existing_node_indexes | set(hop_config["head_node_indexes"])):
            raise ValueError("Require the non-existing tail node indexes")

        # If there is a parent path configuration, ensure that tail node indexes do not overlap
        if father_path_config is not None and (set(hop_config["tail_node_indexes"]) & father_node_indexes):
            raise ValueError("Require the non-existing tail node indexes in father path")

        # Iterate over relations within the hop configuration
        for relation_k, relation_config in enumerate(hop_config["relation_list"]):
            if "relation_type" not in relation_config:
                raise ValueError(f"Require the type for {relation_k}-th relation")

            relation_type = relation_config["relation_type"]

            logger.info(f"Check the {relation_k}-th relation_config with type {relation_type}")

            # Count existing edges
            exist_edge_count = 0
            for element_info in complex_path_config["seq_path"]:
                if element_info[0] == "Edge":
                    exist_edge_count += 1

            # Handle edge relations
            if relation_type == "edge":
                edge_schema = relation_config["edge_schema"]
                
                edge_schema["head_node_types"] = list(hop_config["head_node_types"])
                edge_schema["head_node_indexes"] = list(hop_config["head_node_indexes"])
                edge_schema["tail_node_types"] = list(hop_config["tail_node_types"])
                edge_schema["tail_node_indexes"] = list(hop_config["tail_node_indexes"])
                
                required_keys = ["head_node_columns", "edge_table_name", "tail_node_columns"]
                for required_key in required_keys:
                    if required_key not in edge_schema:
                        raise KeyError(f"Require {required_key} for edge schema")

                # Check consistency between head node columns and types
                if len(edge_schema["head_node_columns"]) != len(hop_config["head_node_types"]):
                    raise ValueError("Require the number of columns same as the number of head nodes types")

                # Check consistency between tail node columns and types
                if len(edge_schema["tail_node_columns"]) != len(hop_config["tail_node_types"]):
                    raise ValueError("Require the same number of columns for tail nodes")

                # Assign an index to the new edge
                add_edge_index = exist_edge_count + 1
                if father_path_config is not None:
                    add_edge_index = add_edge_index + father_edge_count
                
                # Record the edge
                edge_schema["edge_index"] = add_edge_index
                edge_info = ("Edge", edge_schema["edge_table_name"], add_edge_index)
                complex_path_config["seq_path"].append(edge_info)

                # Update adjacency information
                complex_path_config["adj"][edge_info] = {"in": list(head_node_info_list), "out": list(tail_node_info_list)}

                for head_node_info in head_node_info_list:
                    complex_path_config["adj"][head_node_info]["out"].append(edge_info)
                for tail_node_info in tail_node_info_list:
                    complex_path_config["adj"][tail_node_info]["in"].append(edge_info)

                # Handle optional edge limit
                if "edge_limit" not in edge_schema:
                    edge_schema["edge_limit"] = ""

                # Handle optional edge sample
                if "edge_sample" not in edge_schema:
                    edge_schema["edge_sample"] = {}

            # Handle path relations
            elif relation_type == "path":
                path_schema = relation_config["path_schema"]

                path_schema[0]["head_node_types"] = list(hop_config["head_node_types"])
                path_schema[0]["head_node_indexes"] = list(hop_config["head_node_indexes"])
                path_schema[-1]["tail_node_types"] = list(hop_config["tail_node_types"])
                path_schema[-1]["tail_node_indexes"] = list(hop_config["tail_node_indexes"])

                # Recursively validate nested path configurations
                ensured_path_config = ensure_complex_path_config(relation_config, complex_path_config, logger=logger)

#                 # Adjust edge indexes in the nested configuration
#                 for element_index, element_info in enumerate(ensured_path_config["seq_path"]):
#                     if element_info[0] == "Edge":
#                         new_element_info = (element_info[0], element_info[1], exist_edge_count + element_info[2])
#                         ensured_path_config["seq_path"][element_index] = new_element_info

#                 # Adjust adjacency information in the nested configuration
#                 new_adj_dict = {}
#                 for element_info in ensured_path_config["adj"]:
#                     if element_info[0] == "Edge":
#                         new_key_element_info = (element_info[0], element_info[1], exist_edge_count + element_info[2])
#                         new_adj_dict[new_key_element_info] = copy.deepcopy(ensured_path_config["adj"][element_info])
#                     else:
#                         new_adj_dict[element_info] = {"in": [], "out": []}
#                         for adj_direction in ["in", "out"]:
#                             for value_element_info in ensured_path_config["adj"][element_info][adj_direction]:
#                                 new_value_element_info = (value_element_info[0], value_element_info[1], exist_edge_count + value_element_info[2])
#                                 new_adj_dict[element_info][adj_direction].append(new_value_element_info)

#                 ensured_path_config["adj"] = new_adj_dict

                hop_config["relation_list"][relation_k] = ensured_path_config

                head_node_count = len(hop_config["head_node_types"])
                tail_node_count = len(hop_config["tail_node_types"])

                complex_path_config["seq_path"].extend(ensured_path_config["seq_path"][head_node_count:-tail_node_count])

                for element_info in ensured_path_config["adj"]:
                    if element_info not in complex_path_config["adj"]:
                        complex_path_config["adj"][element_info] = copy.deepcopy(ensured_path_config["adj"][element_info])
                    else:
                        for adj_direction in ["in", "out"]:
                            add_adj_elements = ensured_path_config["adj"][element_info][adj_direction]
                            complex_path_config["adj"][element_info][adj_direction].extend(add_adj_elements)

        complex_path_config["seq_path"].extend(tail_node_info_list)

        if "node_index_to_limits" not in hop_config:
            hop_config["node_index_to_limits"] = {}
        else:
            existing_node_indexes = set()
            for element_info in complex_path_config["seq_path"]:
                if element_info[0] == "Node":
                    existing_node_indexes.add(element_info[2])

            for node_limit in hop_config["node_index_to_limits"]:
                if node_limit["node_index"] not in existing_node_indexes:
                    raise ValueError("Node limitation on unexisting node")

        if "path_limit" not in hop_config:
            hop_config["path_limit"] = ""

        if "path_sample" not in hop_config:
            hop_config["path_sample"] = {}

    # Compare with exisiting same name path config
    if path_result_dir is not None and hdfs_check_file_exists(path_result_dir):
        existed_path_config = hdfs_read_json(path_result_dir, "Config.json")
        compare_complex_path_configs(complex_path_config, existed_path_config)
    
    # Generate path name
    
    # Element info to path index["seq_adj"]
        
    # Sort adjacency keys in sequence path order and transfer the original inform into path index
    new_adj_dict = {}
    for element_info in complex_path_config["seq_path"]:
        new_adj_dict[element_info] = copy.deepcopy(complex_path_config["adj"][element_info])
    complex_path_config["adj"] = new_adj_dict

    # Remap node indexes if there is no parent path configuration
    if father_path_config is None:
        complex_path_config = remap_node_index_by_order(complex_path_config)

    return complex_path_config

def remap_node_index_by_order(complex_path_config):
    """
    Remap node indexes according to their appearance order in a complex path configuration.

    Args:
        complex_path_config (dict): The complex path configuration.

    Returns:
        dict: The complex path configuration with remapped node indexes.
    """

    node_raw_indexes = []
    node_order_indexes = []
    for element_index, element_info in enumerate(complex_path_config["seq_path"]):
        if element_info[0] == "Node":
            node_raw_indexes.append(element_info[2])
            node_order_indexes.append(len(node_raw_indexes))

    if node_raw_indexes == node_order_indexes:
        return complex_path_config

    index_map_dict = dict(zip(node_raw_indexes, node_order_indexes))

    for element_index, element_info in enumerate(complex_path_config["seq_path"]):
        if element_info[0] == "Node":
            new_element_info = (element_info[0], element_info[1], index_map_dict[element_info[2]])
            complex_path_config["seq_path"][element_index] = new_element_info

    new_adj_dict = {}
    for element_info in complex_path_config["seq_path"]:
        if element_info[0] == "Node":
            new_key_element_info = (element_info[0], element_info[1], index_map_dict[element_info[2]])
            new_adj_dict[new_key_element_info] = copy.deepcopy(complex_path_config["adj"][element_info])
        else:
            new_adj_dict[element_info] = {"in": [], "out": []}
            for adj_direction in ["in", "out"]:
                for value_element_info in complex_path_config["adj"][element_info][adj_direction]:
                    new_value_element_info = (value_element_info[0], value_element_info[1], index_map_dict[value_element_info[2]])
                    new_adj_dict[element_info][adj_direction].append(new_value_element_info)

    complex_path_config["adj"] = new_adj_dict

    return complex_path_config

def compare_complex_path_configs(config_a, config_b):
    """
    Compare two complex path configurations for equality.

    Args:
        config_a (dict): The first complex path configuration.
        config_b (dict): The second complex path configuration.

    Raises:
        ValueError: If the two configurations are different.

    Returns:
        bool: True if the configurations are identical, otherwise False.
    """

    config_a = remap_node_index_by_order(config_a)
    config_b = remap_node_index_by_order(config_b)

    if config_a != config_b:
        raise ValueError("Different with existing path config with same name")

    return True


# 比较两个config是否一致
def compare_complex_path_configs(config_a, config_b):
    # 重新复制两个字典，防止比较的过程中影响原始内容
    
    # 把index按顺序重新生成
    config_a = remap_node_index_by_order(config_a)
    config_b = remap_node_index_by_order(config_b)
    
    # 逐项对比，显示哪一项不同
    if config_a != config_b:
        raise ValueError("Different with existing path config with same name")

    return True


# (未来优化)查看现存的path表中是否有完全一样config的path
# 先检查有没有表名就一样的，再检查有没有config完全一样的
