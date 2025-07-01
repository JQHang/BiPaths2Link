def dataset_init(graph, train_instances, join_edges_list):
    # 设定数据集相关配置
    dataset_config = {}
    
    # 先记录该数据集涉及到的join_edges种类
    dataset_config["join_edges_list"] = join_edges_list
    
    # 如果是训练数据，则加入对应的标签列
    dataset_config["label_cols"] = []
    dataset_config["label_len"] = 0
    for scale_func in train_instances["dataset_scale_funcs"]:
        dataset_config["label_cols"].append(f"labels_scaled_{scale_func}")
        # dataset_config["label_len"] += len(train_instances["label_cols_to_aliases"])
        dataset_config["label_len"] += 1
        
    # 加入各个token表对应的特征列
    dataset_config["feat_cols"] = {}
    
    # 记录各个表的特征长度
    dataset_config["feat_lens"] = {}
    
    # 首先是自身的特征表
    dataset_config["feat_cols"]["self"] = {}  
    
    # 依次处理自身的各个节点表
    # 如果目标instance有多个点，他们之间的边也是另一种形式的特征，可以之后考虑(参考Intersect2Pair)
    for node_type in train_instances["nodes_types"]:
        dataset_config["feat_cols"]["self"][node_type] = {}
        
        if node_type not in dataset_config["feat_lens"]:
            dataset_config["feat_lens"][node_type] = {}
        
        for node_table_i, node_table_name in enumerate(graph.nodes[node_type]["node_tables"]):
            # 获得该表对应的schema
            node_table_schema = graph.nodes[node_type]["node_tables"][node_table_name]
    
            # 如果该节点表没有特征就直接跳过
            if len(node_table_schema["time_aggs_feat_cols"]) == 0:
                continue
            
            # 记录该表对应的特征列
            dataset_config["feat_cols"]["self"][node_type][node_table_name] = {}
            
            # 获得经过缩放后对应的各个特征向量列的名称
            for scale_func in train_instances["dataset_scale_funcs"]:
                feat_col = f"Self_feats_{node_table_i}_scale_raw_scaled_{scale_func}"
                dataset_config["feat_cols"]["self"][node_type][node_table_name][scale_func] = feat_col
    
                # 记录该表对应的特征长度
                if node_table_name not in dataset_config["feat_lens"][node_type]:
                    dataset_config["feat_lens"][node_type][node_table_name] = {}
                if scale_func not in dataset_config["feat_lens"][node_type][node_table_name]:
                    dataset_config["feat_lens"][node_type][node_table_name][scale_func] = len(node_table_schema["time_aggs_feat_cols"])
    
    # 然后是join_edges对应的特征表
    dataset_config["feat_cols"]["join_edges"] = {}  
    
    # 依次处理各个join-edges
    for join_edges in join_edges_list:
        # 获得对应的join_edges_name
        join_edges_name = join_edges['name']
    
        dataset_config["feat_cols"]["join_edges"][join_edges_name] = {}
    
        # 获得agg nodes相关信息
        inst_agg_nodes_types = join_edges["inst_agg_nodes_types"]
        inst_agg_nodes_indexes = join_edges["inst_agg_nodes_indexes"]
        inst_agg_nodes_cols = join_edges["inst_agg_nodes_cols"]
        
        # 获得这些agg_nodes分别对应哪几个join_edges中的seq_token
        inst_agg_nodes_seq_indexes = []
        for agg_node_i in range(len(inst_agg_nodes_types)):
            agg_node_type = inst_agg_nodes_types[agg_node_i]
            agg_node_index = inst_agg_nodes_indexes[agg_node_i]
    
            agg_node_seq_index = -1
            for seq_token_index, seq_token in enumerate(join_edges["flatten_format"]["seq"]):
                if "node_type" in seq_token:
                    if seq_token["node_type"] == agg_node_type and seq_token["node_index"] == agg_node_index:
                        agg_node_seq_index = seq_token_index
    
            if agg_node_seq_index == -1:
                raise ValueError(f"Aggregation node {agg_node_type} index {agg_node_index} doesn't "
                                 f"exist in the join-edges")
    
            inst_agg_nodes_seq_indexes.append(agg_node_seq_index)
        
        # 依次处理该join_edges的各个seq_token对应的数据
        for seq_token_index, seq_token in enumerate(join_edges["flatten_format"]["seq"]):
            # 检查该seq_token是否对应到聚合节点，是就跳过
            if seq_token_index in inst_agg_nodes_seq_indexes:
                continue
    
            dataset_config["feat_cols"]["join_edges"][join_edges_name][seq_token_index] = {}
    
            # 查看该元素是节点还是边
            if "edge_type" in seq_token:
                # 获取对应的边类型
                edge_type = seq_token["edge_type"]

                # 创建记录各个表的特征长度的字典
                if edge_type not in dataset_config["feat_lens"]:
                    dataset_config["feat_lens"][edge_type] = {}
                
                # 依次处理该seq_token包含的各个table(以后会把edge_table这个项改成list形式)
                for edge_table_name in [seq_token["edge_table_name"]]:
                    # 获取该边对应的边表的schema
                    edge_table_schema = graph.edges[edge_type]["edge_tables"][edge_table_name]
    
                    # 检查该表是否有特征列，如果没有则跳过
                    if len(edge_table_schema["time_aggs_feat_cols"]) == 0:
                        continue
    
                    dataset_config["feat_cols"]["join_edges"][join_edges_name][seq_token_index][edge_table_name] = {}
                    
                    # 依次获得各种缩放对应的特征列
                    for scale_func in train_instances["dataset_scale_funcs"]:
                        feat_col = f"{join_edges_name}_seq_{seq_token_index}_table_{edge_table_name}_scale_raw_scaled_{scale_func}"
                        dataset_config["feat_cols"]["join_edges"][join_edges_name][seq_token_index][edge_table_name][scale_func] = feat_col
    
                        # 记录该表对应的特征长度
                        if edge_table_name not in dataset_config["feat_lens"][edge_type]:
                            dataset_config["feat_lens"][edge_type][edge_table_name] = {}
                        if scale_func not in dataset_config["feat_lens"][edge_type][edge_table_name]:
                            dataset_config["feat_lens"][edge_type][edge_table_name][scale_func] = len(edge_table_schema["time_aggs_feat_cols"])
    
            else:
                # 获取对应的点类型
                node_type = seq_token["node_type"]

                # 创建记录各个表的特征长度的字典
                if node_type not in dataset_config["feat_lens"]:
                    dataset_config["feat_lens"][node_type] = {}
                
                # 依次处理该类型节点对应的各个节点表
                for node_table_name in graph.nodes[node_type]["node_tables"]:
                    # 获取该节点对应的节点表的schema
                    node_table_schema = graph.nodes[node_type]["node_tables"][node_table_name]
    
                    # 检查该表是否有特征列，如果没有则跳过
                    if len(node_table_schema["time_aggs_feat_cols"]) == 0:
                        continue
    
                    dataset_config["feat_cols"]["join_edges"][join_edges_name][seq_token_index][node_table_name] = {}
                    
                    # 依次获得各种缩放对应的特征列
                    for scale_func in train_instances["dataset_scale_funcs"]:
                        feat_col = f"{join_edges_name}_seq_{seq_token_index}_table_{node_table_name}_scale_raw_scaled_{scale_func}"
                        dataset_config["feat_cols"]["join_edges"][join_edges_name][seq_token_index][node_table_name][scale_func] = feat_col
    
                        # 记录该表对应的特征长度
                        if node_table_name not in dataset_config["feat_lens"][node_type]:
                            dataset_config["feat_lens"][node_type][node_table_name] = {}
                        if scale_func not in dataset_config["feat_lens"][node_type][node_table_name]:
                            dataset_config["feat_lens"][node_type][node_table_name][scale_func] = len(node_table_schema["time_aggs_feat_cols"])
    
    return dataset_config