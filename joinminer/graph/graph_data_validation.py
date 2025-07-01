# 检查各节点表和边表是否合规（是否有重复数据、无效数据等）,检查过的会记录检查结果，防止重复检查
def graph_validation(self, spark):
    # 先检查各节点表
    for node_type in self.graph_schema["node_schemas"]:
        for node_table_name in self.graph_schema['node_schemas'][node_type]['node_tables']:
            # 获得对应的schema
            node_table_schema = self.graph_schema['node_schemas'][node_type]['node_tables'][node_table_name]

            # 查看是否有多个分区
            if self.partition_key != None:
                partition_locations = hdfs_list_contents(node_table_schema["table_path"], content_type = "directories")

                # 检查各分区数据是否合规
                for partition_location in partition_locations:
                    # 检查数据是否合规
                    self.table_validation(spark, partition_location, node_table_schema["table_schema"])

            else:
                # 检查数据是否合规
                self.table_validation(spark, node_table_schema["table_path"], node_table_schema["table_schema"])

    # 再检查各边表
    for edge_type in self.graph_schema["edge_schemas"]:
        for edge_table_name in self.graph_schema['edge_schemas'][edge_type]['edge_tables']:
            # 获得对应的schema
            edge_table_schema = self.graph_schema['edge_schemas'][edge_type]['edge_tables'][edge_table_name]

            # 查看是否有多个分区
            if self.partition_key != None:
                partition_locations = hdfs_list_contents(edge_table_schema["table_path"], content_type = "directories")

                # 检查各分区数据是否合规
                for partition_location in partition_locations:
                    # 检查数据是否合规
                    self.table_validation(spark, partition_location, edge_table_schema["table_schema"])

            else:
                # 检查数据是否合规
                self.table_validation(spark, edge_table_schema["table_path"], edge_table_schema["table_schema"])

    return

def table_validation(self, spark, table_path, table_schema):
    # 查看是否有之前的检查结果
    if hdfs_check_file_exists(table_path + '/' + table_quality_file_name):
        # 直接读取之前的检查结果
        table_quality = hdfs_read_json(table_path, table_quality_file_name, logger = self.logger)
    else:
        # 获得节点列
        table_node_columns = list(table_schema["node_col_to_types"].keys())

        # 获得特征列
        table_feat_columns = table_schema["feat_cols"]

        # 读取数据
        table_df = spark.read.parquet(table_path)

        # 检查数据是否合规
        table_quality = {}

        # 总行数
        table_quality["count"] = table_df.count()

        # 重复行（也同时检验了节点列是否都存在）
        grouped_table_df = table_df.groupBy(table_node_columns).agg(count("*").alias("count"))
        table_quality["duplicate_count"] = grouped_table_df.filter(col("count") > 1).count()

        # 无效特征列（也同时检验了特征列是否都存在）
        feat_summary_df = table_df.select(table_feat_columns).summary("min", "max")
        min_values = feat_summary_df.filter(col("summary") == "min").drop("summary").collect()[0].asDict()
        max_values = feat_summary_df.filter(col("summary") == "max").drop("summary").collect()[0].asDict()

        table_quality["invalid_feats"] = {col_name: min_values[col_name] for col_name in min_values.keys() 
                                         if min_values[col_name] == max_values[col_name]}

        # 保存检查结果
        hdfs_save_json(table_path, table_quality_file_name, table_quality, logger = self.logger)

    self.logger.info(f"The summary of table data at {table_path}:")

    # 总共有多少行
    if table_quality["count"] > 0:
        self.logger.info(f"Rows count: {table_quality['count']}")
    else:
        raise ValueError(f'The table is empty at {table_path}')

    # 有多少重复行
    if table_quality["duplicate_count"] == 0:
        self.logger.info(f"No duplicate rows")
    else:
        self.logger.warning(f"Duplicate rows count: {table_quality['duplicate_count']}")

    # 有多少无效特征
    if len(table_quality["invalid_feats"]) == 0:
        self.logger.info(f"No invalid features")
    else:
        self.logger.warning(f"Contains {len(table_quality['invalid_feats'])} invalid features with all same value: ")
        for col_name, value in table_quality["invalid_feats"].items():
            self.logger.warning(f"Invalid feature columns {col_name} with value {value}")

    return