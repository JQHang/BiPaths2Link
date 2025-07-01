from ..python import time_costing
from ..hdfs import hdfs_save_string, hdfs_delete_dir

from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import col, lit, rand, broadcast
from pyspark.sql.functions import count as _count, sum as _sum
import json
import logging

# 获得logger
logger = logging.getLogger(__name__)

@time_costing
def pyspark_create_hive_table(spark, table_name, df, partition_keys = [], partition_key_types = [], 
                              column_comments = None):
    """
    Create a Hive table for the specified dataframe. If the table already exists, it will be dropped.

    Parameters:
    - spark: The SparkSession object.
    - table_name: Name of the target table.
    - df: Target dataframe for the target table to store.
    - partition_key: (Optional) Name of the column to use for partitioning.
    - partition_key_type: (Optional) Data type of the partition key column. Default is "STRING".
    - column_comments: (Optional) List of comments for each column. The length of the list should match the number of columns in the dataframe.
    - logger: (Optional) Logger object for logging messages.

    Returns:
    - None
    """
    # Get the schema configuration of the dataframe
    df_json = json.loads(df.schema.json())['fields']
    
    # Construct the SQL query to create the table
    create_table_sql = f"CREATE TABLE IF NOT EXISTS {table_name} ("

    for col_i, col_info in enumerate(df_json):
        column = col_info['name']
        column_type = col_info['type'].upper()  # Ensure type is in uppercase
        
        create_table_sql += f"{column} {column_type}"
        
        if column_comments:
            column_comment = column_comments[col_i]
            create_table_sql += f" COMMENT '{column_comment}'"
        
        if col_i != len(df_json) - 1:
            create_table_sql += ', '
    
    create_table_sql += ") \n"

    if len(partition_keys) > 0:
        create_table_sql += f"PARTITIONED BY ("
        for partition_i in range(len(partition_keys)):
            partition_key = partition_keys[partition_i]
            partition_key_type = partition_key_types[partition_i]
            
            create_table_sql += f"{partition_key} {partition_key_type}"

            if partition_i != len(partition_keys) - 1:
                create_table_sql += ', '
    
    create_table_sql += f") \n"
    
    create_table_sql += "STORED AS ORC \nTBLPROPERTIES ('orc.compress' = 'SNAPPY')"
    
    logger.info(f"Creating Hive table SQL query: {create_table_sql}")
    
    # Execute the SQL query
    spark.sql(create_table_sql)

def pyspark_read_table(spark, table_path, table_format, partition_cols=None, partition_values_list=None, select_cols_aliases=None):
   """
   Read table data with optional partitions and column selection
   """
   logger.info(f"Read table from {table_path} with format {table_format}")
   
   # 如果有分区过滤，构建具体的分区路径
   if partition_cols and partition_values_list:
       assert len(partition_values_list) > 0, "Require partition values for partition columns"
       
       # 构建分区路径
       partition_paths = []
       for partition_values in partition_values_list:
           path_parts = [f"{col}={val}" for col, val in zip(partition_cols, partition_values)]
           partition_path = "/".join(path_parts)
           partition_paths.append(f"{table_path}/{partition_path}")
       
       # 读取指定分区
       df = spark.read.format(table_format) \
                     .option("basePath", table_path) \
                     .load(partition_paths)
       logger.info(f"Reading specific partitions: {partition_paths}")
   else:
       # 如果没有分区过滤，直接读取全表
       df = spark.read.format(table_format).load(table_path)
   
   # 如果需要，进行列选择
   if select_cols_aliases:
       columns = [col(column).alias(alias) for column, alias in select_cols_aliases]
       df = df.select(*columns)
       logger.info(f"Selected {len(columns)} columns")
   
   return df

# def batch_compute_num_files(df, col_type_and_sizes, partition_cols = [], batch_size = 50):
#     """
#     分批对大量列进行count聚合，计算总大小并返回优化后的DataFrame
    
#     参数:
#     df - 输入DataFrame
#     col_type_and_sizes - 字典，包含每列的预估大小信息
#     partition_cols - 分组列的列表，默认为None（不分组）
#     batch_size - 每批处理的列数
    
#     返回:
#     包含(可选)分区列、总计数和文件数量的DataFrame
#     """
#     MAX_FILE_SIZE = 1 * 1024 * 1024 * 1024  # 1 GB in bytes
    
#     # 第一批包含(可选)分区列和总计数
#     if partition_cols:
#         # 有分区列，执行groupBy
#         first_batch = df.select(*partition_cols)
#         first_batch = first_batch.groupBy(*partition_cols).agg(
#             _count("*").alias("count_total")
#         ).coalesce(1)
#     else:
#         # 无分区列，直接聚合
#         first_batch = df.select(df.columns[0])
#         first_batch = first_batch.agg(
#             _count("*").alias("count_total")
#         ).coalesce(1)
    
#     # 添加一个列来累计总大小
#     result_df = first_batch.withColumn("total_size", lit(0))
    
#     # 分批处理所有列计算总大小
#     all_columns = [c for c in df.columns if c not in partition_cols]
    
#     for i in range(0, len(all_columns), batch_size):
#         batch_columns = all_columns[i:i+batch_size]

#         required_columns = partition_cols + batch_columns
#         batch_df = df.select(*required_columns)
        
#         # 为当前批次创建聚合表达式
#         agg_exprs = [_count(col_name).alias(f"count_{col_name}") for col_name in batch_columns]
        
#         # 执行分组聚合并立即减少到1个分区
#         if partition_cols:
#             batch_result = batch_df.groupBy(*partition_cols).agg(*agg_exprs).coalesce(1)
            
#             # 将结果与主结果合并
#             result_df = result_df.join(batch_result, partition_cols, "inner").coalesce(1)
#         else:
#             batch_result = batch_df.agg(*agg_exprs).coalesce(1)
            
#             # 无分区列时，使用crossJoin合并结果
#             result_df = result_df.crossJoin(batch_result).coalesce(1)
        
#         # 为当前批次创建大小计算表达式
#         batch_size_expr = lit(0)
#         for col_name in batch_columns:
#             if col_name in col_type_and_sizes:
#                 field_size = col_type_and_sizes[col_name]["estimated_size"]
#                 batch_size_expr = batch_size_expr + (lit(field_size) * col(f"count_{col_name}"))
        
#         # 更新总大小
#         result_df = result_df.withColumn("total_size", col("total_size") + batch_size_expr)
        
#         # 删除不需要保存的count列
#         for col_name in batch_columns:
#             result_df = result_df.drop(f"count_{col_name}")
    
#     # 计算所需文件数
#     result_df = result_df.withColumn("num_files", 
#                                     (col("total_size") / lit(MAX_FILE_SIZE)).cast("int") + 1)
    
#     # 删除临时的total_size列
#     result_df = result_df.drop("total_size")
    
#     # 只保留需要的列
#     if partition_cols:
#         result_df = result_df.select(*partition_cols, "count_total", "num_files")
#     else:
#         result_df = result_df.select("count_total", "num_files")
    
#     return result_df

def compute_num_files_sql(spark, df, col_type_and_sizes, partition_cols=[]):
    """
    使用单次SQL查询对所有列进行count聚合，计算总大小并返回优化后的DataFrame
    
    参数:
    df - 输入DataFrame
    col_type_and_sizes - 字典，包含每列的预估大小信息
    partition_cols - 分组列的列表，默认为空列表（不分组）
    
    返回:
    包含(可选)分区列、总计数和文件数量的DataFrame
    """
    MAX_FILE_SIZE = 1 * 1024 * 1024 * 1024  # 1 GB in bytes
    
    # 注册临时视图
    df.createOrReplaceTempView("input_data")
    
    # 构建分区列字符串(用于SQL)
    partition_cols_str = ", ".join([f"`{col}`" for col in partition_cols]) if partition_cols else ""
    partition_group_by = f"GROUP BY {partition_cols_str}" if partition_cols else ""
    
    # 获取所有非分区列
    all_columns = [c for c in df.columns if c not in partition_cols]
    
    # 构建大小计算表达式
    size_exprs = []
    for col_name in all_columns:
        if col_name in col_type_and_sizes:
            field_size = col_type_and_sizes[col_name]["estimated_size"]
            size_exprs.append(f"{field_size} * COUNT(`{col_name}`)")
    
    size_calculation = " + ".join(size_exprs) if size_exprs else "0"
    
    # 构建完整的SQL查询 - 一次性计算所有列的count和总大小
    if partition_cols:
        query = f"""
        SELECT 
            {partition_cols_str},
            COUNT(*) as count_total,
            CAST(({size_calculation}) / {MAX_FILE_SIZE} AS INT) + 1 as num_files
        FROM input_data
        {partition_group_by}
        """
    else:
        query = f"""
        SELECT 
            COUNT(*) as count_total,
            CAST(({size_calculation}) / {MAX_FILE_SIZE} AS INT) + 1 as num_files
        FROM input_data
        """
    
    # 执行查询并返回结果
    result_df = spark.sql(query).coalesce(1)

    # 释放临时视图
    spark.catalog.dropTempView("input_data")
    
    return result_df

# 应该再加入partition_values，如果给出这个，哪怕对应分区为空也标注完成表示结果为空
@time_costing
def pyspark_optimal_save(
    df: DataFrame,
    path: str,
    format_name: str,
    mode: str,
    partition_cols: list = None,
    partition_values_list: list = None,
    col_sizes: dict = {},
    max_num_files: int = 1500
) -> None:
    """
    Save a DataFrame to disk with an optimal number of partitions based on estimated file sizes.
    Uses random distribution across optimal number of files for both partitioned and non-partitioned cases.
    
    Parameters:
    df (DataFrame): The input DataFrame to be saved.
    path (str): The storage path where the DataFrame should be saved.
    format_name (str): The format in which to save the DataFrame (e.g., 'parquet', 'orc').
    mode (str): The write mode (e.g., 'append', 'overwrite').
    partition_cols (list[str]): A list of column names to partition the DataFrame by. Default empty list for no partitioning.
    """
    # Step 1: Persist the dataframe
    df = df.persist()

    # 显示下大致结果，防止有离谱错误
    df.select(df.columns[:20]).show()
    
    # Define size estimates for different column types
    ROW_SIZE_ESTIMATES = {
        "int": 4,
        "bigint": 8,
        "double": 8,
        "float": 4,
        "string": 50,  # Assuming average 50 bytes per string
        "boolean": 1,
        "vector": 300,
        "array<double>": 300
    }

    MAX_FILE_SIZE = 1 * 1024 * 1024 * 1024  # 1 GB in bytes (Around 128M in real world test)
    
    # 显示要保存的表中包含的列、列类型及预估对应的大小
    col_type_and_sizes = {}
    for field in df.schema.fields:
        col_type_and_sizes[field.name] = {}
        col_type_and_sizes[field.name]["type"] = field.dataType.simpleString()
        
        # First check if column size is defined in col_sizes
        if field.name in col_sizes:
            col_type_and_sizes[field.name]["estimated_size"] = col_sizes[field.name] * 4
        # Then check if type has a predefined size estimate
        elif field.dataType.simpleString() in ROW_SIZE_ESTIMATES:
            col_type_and_sizes[field.name]["estimated_size"] = ROW_SIZE_ESTIMATES[field.dataType.simpleString()]
        else:
            default_estimated_size = 100
            col_type_and_sizes[field.name]["estimated_size"] = default_estimated_size
            logger.warning(f"Unexpected column type: {field.dataType.simpleString()} for column {field.name}"
                               f", use {default_estimated_size} as its size")

    # 总共有多少列，以及全部列名
    logger.info(f"Save table with {len(df.columns)} columns: {df.columns}")
    first_20_col_type_and_sizes = {k: v for k, v in list(col_type_and_sizes.items())[:20]}
    logger.info("First 20 columns information of table to save:\n%s", json.dumps(first_20_col_type_and_sizes, indent=4))

    # Common function to estimate total size based on column counts
    def total_partition_size(row):
        total_size = 0
        for field in df.schema.fields:
            total_size += col_type_and_sizes[field.name]["estimated_size"] * row[f"count_{field.name}"]
        return total_size
    
    if partition_cols is None:
        # Handle non-partitioned case
        # Calculate counts for each column in the entire DataFrame
        counts_df = df.agg(*[_count(col_name).alias(f"count_{col_name}") for col_name in df.columns],
                           _count("*").alias(f"count_total")).coalesce(1)

        # Calculate optimal number of files
        counts_df = counts_df.withColumn(
            "num_files",
            (lit(total_partition_size(counts_df)) / MAX_FILE_SIZE).cast("int") + 1
        )
        
        count_file_info = counts_df.select("count_total", "num_files").collect()[0]
        file_row_count = int(count_file_info['count_total'] / count_file_info['num_files'] * 1.05)
        num_files = count_file_info['num_files']

        logger.info(f"The {count_file_info['count_total']} rows dataframe will be stored in {num_files} files with max {file_row_count} rows.")
        
        # Add random file number column to distribute rows across files
        df_with_file_num = df.withColumn(
            "file_num", 
            (rand() * num_files).cast("int")
        )
        
        # Repartition based on file_num
        df_repartitioned = df_with_file_num.repartition(num_files, "file_num")
        
        # Drop the file_num column
        df_final = df_repartitioned.drop("file_num")
        
        # Write the dataframe
        df_final.write.option("maxRecordsPerFile", file_row_count).format(format_name).mode(mode).save(path)
        
    else:
        # Handle partitioned case
        # Calculate counts for each partition combination
        partition_counts = df.groupBy(*partition_cols).agg(
            *[_count(col_name).alias(f"count_{col_name}") for col_name in df.columns],
            _count("*").alias(f"count_total")).coalesce(1)
        
        # Calculate optimal number of files for each partition
        partition_counts = partition_counts.withColumn(
            "num_files",
            (lit(total_partition_size(partition_counts)) / MAX_FILE_SIZE).cast("int") + 1,
        )

        # 计算各分区累积num_files
        window_spec = Window.orderBy(*partition_cols)
        partition_counts = partition_counts.withColumn(
            "cumulative_num_files", 
            _sum("num_files").over(window_spec) - col("num_files")
        )
        
        partition_counts.persist()

        # 显示各个分区的总行数以及要保存的文件数
        count_total_sum = 0
        num_files_sum = 0
        partition_total_counts = partition_counts.select(*partition_cols, "count_total", "num_files", "cumulative_num_files")\
                                                 .orderBy(*partition_cols).collect()
        for row in partition_total_counts:
            partition_parts = [f"{col_name}={row[col_name]}" for col_name in partition_cols]
            partition_info = f"Partition {', '.join(partition_parts)} row count: {row['count_total']}, file count: {row['num_files']}, "
            partition_info = partition_info + f"cumulative file count: {row['cumulative_num_files']}."
            logger.info(partition_info)

            # 累计总行数
            count_total_sum = count_total_sum + row['count_total']
            
            # 累积最终的文件总数 
            num_files_sum = num_files_sum + row['num_files']

        # 设定各个输出文件最大输出行数，目前定为平均值的1.05倍 
        file_row_count = int(count_total_sum/num_files_sum * 1.05)
        logger.info(f"Output file max row count: {file_row_count}")
        
        # 获得各个分区要存储的文件数
        partition_counts = partition_counts.select(*partition_cols, "num_files", "cumulative_num_files")

        # 补全分区类型
        if partition_values_list is None:
            distinct_partition_cols_rows = partition_counts.select(*partition_cols).collect()
            partition_values_list = [[row[c] for c in partition_cols] for row in distinct_partition_cols_rows]
        
        # 如果有预设的partition_values_list，可以检查实际的是否都在范围内
        
        # Join back to original dataframe to add num_files column
        df_with_num_files = df.join(broadcast(partition_counts), on=partition_cols, how="left")
        
        # Add random file number column to distribute rows across files
        df_with_file_num = df_with_num_files.withColumn(
            "file_num", 
            (rand() * col("num_files")).cast("int") + col("cumulative_num_files")
        ).drop("num_files", "cumulative_num_files")

        # Repartition dataframe based on partition columns and file number
        df_repartitioned = df_with_file_num.repartition(num_files_sum, "file_num")

        # 检查是否能一次性写入
        if num_files_sum < max_num_files:
            # Write the dataframe to the specified path, format, and mode
            df_batch = df_repartitioned.drop("file_num")
            df_batch.write.option("maxRecordsPerFile", file_row_count).format(format_name).mode(mode).partitionBy(*partition_cols).save(path)
        else:
            df_repartitioned.persist()

            # 如果写入模式是overwrite，则清空对应的分区文件夹
            if mode == "overwrite":
                for partition_values in partition_values_list:
                    path_parts = [f"{col}={val}" for col, val in zip(partition_cols, partition_values)]
                    partition_path = f"{path}/" + "/".join(path_parts)
            
                    hdfs_delete_dir(partition_path)

            # 依次写入数据
            for num_files_start in range(0, num_files_sum, max_num_files):
                if (num_files_start + max_num_files) < num_files_sum:
                    num_files_end = num_files_start + max_num_files
                else:
                    num_files_end = num_files_sum
                    
                # Repartition dataframe based on partition columns and file number
                df_batch = df_repartitioned.filter((col("file_num") >= num_files_start) & (col("file_num") < num_files_end)).drop("file_num")
                
                # Write the dataframe to the specified path, format, and mode
                df_batch.write.option("maxRecordsPerFile", file_row_count).format(format_name).mode("append").partitionBy(*partition_cols).save(path)

            df_repartitioned.unpersist()
            
        # Unpersist the partition counts
        partition_counts.unpersist()
        
        # Add _SUCCESS mark file to each partition
        for partition_values in partition_values_list:
            path_parts = [f"{col}={val}" for col, val in zip(partition_cols, partition_values)]
            partition_path = f"{path}/" + "/".join(path_parts)
                
            hdfs_save_string(partition_path, '_SUCCESS')
    
    # Unpersist the original dataframe
    df.unpersist()