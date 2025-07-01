import copy

# 输入seed_path，返回该seed_path能扩展出的能连接target edges的全部intra_path_types
def edge_to_intra_path_types(target_edges, join_edge_types, max_hop_k, max_neighbor = 20):
    # 记录能扩展出的全部k_hop内的pair_paths
    # key是hop数，value是该跳数的pair paths的list
    edge_intra_path_types = {}

    # 将目标边的起始节点作为第0跳的起始路径数据
    k_hop_seed_paths = {0: [[{"add_nodes_types": [target_edges["head_node_type"]]}]]}

    # 依次获得各跳对应的路径
    for seed_path_hop_k in range(max_hop_k):
        # 获得要形成的new_path对应的跳数
        new_path_hop_k = seed_path_hop_k + 1

        # 创建存储对应的intra_path和seed_path的list
        edge_intra_path_types[new_path_hop_k] = []
        k_hop_seed_paths[new_path_hop_k] = []
        
        for seed_path_schema in k_hop_seed_paths[seed_path_hop_k]:
            # 获得该seed_path需要的join_node_type
            tgt_join_node_type = seed_path_schema[-1]["add_nodes_types"][0]
            
            # 遍历可选择的边
            for edge_type in join_edge_types:
                for head_node_index in join_edge_types[edge_type]:
                    # 复制对应的配置作为初始的join_edge_schema
                    join_edge_schema = copy.deepcopy(join_edge_types[edge_type][head_node_index])
                
                    # 如果该join_edge起点不等于seed_path的终点则跳过
                    if join_edge_schema["join_nodes_types"][0] != tgt_join_node_type:
                        # 不符合则跳过该条边
                        # logger.info(f"{head_node_i}-th node type of {edge_type} edge doesn't match seed path")
                        continue
                    
                    # 获得新路径要从seed_path中继承的路径配置
                    if seed_path_hop_k == 0:
                        new_path_schema = []
                    else:    
                        new_path_schema = copy.deepcopy(seed_path_schema)
                    
                    # 获得新增的边对应的join_edge配置
                    join_edge_schema["join_nodes_indexes"] = [seed_path_hop_k]
                    join_edge_schema["add_nodes_indexes"] = [new_path_hop_k]
            
                    join_edge_schema["join_edges_samples"] = []
                    if new_path_hop_k > 1:
                        join_edges_sample = {}
                        join_edges_sample["sample_nodes_types"] = [target_edges["head_node_type"]] + join_edge_schema["add_nodes_types"]
                        join_edges_sample["sample_nodes_indexes"] = [0] + join_edge_schema["add_nodes_indexes"]
                        join_edges_sample["sample_type"] = "random"
                        join_edges_sample["sample_count"] = max_neighbor
                        join_edge_schema["join_edges_samples"].append(join_edges_sample)
                        
                    join_edges_sample = {}
                    join_edges_sample["sample_nodes_types"] = join_edge_schema["add_nodes_types"]
                    join_edges_sample["sample_nodes_indexes"] = join_edge_schema["add_nodes_indexes"]
                    join_edges_sample["sample_type"] = "random"
                    join_edges_sample["sample_count"] = max_neighbor
                    join_edge_schema["join_edges_samples"].append(join_edges_sample)
                    
                    # 将该边加入new_path_schema
                    new_path_schema.append(join_edge_schema)
    
                    # 将该路径记录到seed_path中
                    k_hop_seed_paths[new_path_hop_k].append(new_path_schema)
                    
                    # 检查终点是否符合target_edges的终点
                    if join_edge_schema["add_nodes_types"][0] == target_edges["tail_node_type"]:
                        # 符合的话就记录该路径到结果中
                        edge_intra_path_types[new_path_hop_k].append(new_path_schema)
                        
    return edge_intra_path_types