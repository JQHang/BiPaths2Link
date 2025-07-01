#!/usr/bin/env python
# coding: utf-8

# In[1]:


from joinminer.pyspark import ResilientSparkRunner
from joinminer.python import mkdir, setup_logger, time_costing, read_json_file
from joinminer.hdfs import hdfs_list_contents, hdfs_save_string
from joinminer.hdfs import hdfs_save_json

from datetime import datetime


# In[2]:


# 获得项目文件夹根目录路径
from joinminer import PROJECT_ROOT

# 日志信息保存文件名
log_files_dir = PROJECT_ROOT + '/data/result_data/log_files/aminer_graph'
log_filename = log_files_dir + f'/{datetime.now().strftime("%Y-%m-%d-%H:%M")}.log'
mkdir(log_files_dir)

logger = setup_logger(log_filename, logger_name = "joinminer")

# 获得graph在本地文件中的路径
graph_local_path = PROJECT_ROOT + "/data/dataset/AMiner/graph"

# 获得graph在hdfs文件中的路径
graph_hdfs_path = "/user/mart_coo/mart_coo_innov/CompGraph/AMiner/raw"

# 表格范式文件名
table_schema_file_name = '_Table_Schema'


# In[3]:


# spark配置参数
config_dict = {
                "spark.default.parallelism": "200",
                "spark.sql.shuffle.partitions": "300",
                "spark.sql.broadcastTimeout": "3600",
                "spark.driver.memory": "80g",
                "spark.driver.cores": "16",
                "spark.driver.maxResultSize": "0",
                "spark.executor.memory": "16g",
                "spark.executor.cores": "4",
                "spark.executor.instances": "50"
            }
spark_runner = ResilientSparkRunner(config_dict = config_dict)


# In[51]:


from pyspark.sql.functions import when, lit, col

df = spark_runner.spark_session.read.parquet("file://" + graph_local_path + "/node/paper")
df = df.toDF("paper_id", "paper_title", "lang", "doc_type", "year")
df = df.filter("year <> 0")
df = df.drop("lang")

for doc_type in ["Conference", "Journal"]:
    df = df.withColumn(f"{doc_type}_mark", 
                       when(col("doc_type") == doc_type, 1).otherwise(0))
df = df.drop("doc_type")

df.coalesce(1).write.mode("overwrite").partitionBy("year").parquet(graph_hdfs_path + "/paper")


# In[114]:


dirs = hdfs_list_contents(graph_hdfs_path + "/paper", content_type="directories")
for dir in dirs:
    hdfs_save_string(dir)


# In[105]:


table_schema = {}
table_schema["node_col_to_types"] = {"paper_id": "Paper_Node"}
table_schema["feat_cols"] = ["Conference_mark", "Journal_mark"]
table_schema["time_cols"] = ["year"]
table_schema["time_cols_formats"] = ["%Y"]

hdfs_save_json(graph_hdfs_path + "/paper", table_schema_file_name, table_schema)


# In[5]:


df = spark_runner.spark_session.read.parquet("file://" + graph_local_path + "/edge/author")
df = df.dropDuplicates(['author_id'])
df.coalesce(1).write.parquet(graph_hdfs_path + "/author")


# In[106]:


table_schema = {}
table_schema["node_col_to_types"] = {"author_id": "Author_Node"}
table_schema["feat_cols"] = []
table_schema["time_cols"] = []
table_schema["time_cols_formats"] = []

hdfs_save_json(graph_hdfs_path + "/author", table_schema_file_name, table_schema)


# In[58]:


df = spark_runner.spark_session.read.parquet("file://" + graph_local_path + "/edge/paper_cite_paper")
df = df.filter("year <> 0")
df = df.dropDuplicates(['paper_id', "cite_paper_id"])
df.coalesce(1).write.mode("overwrite").partitionBy("year").parquet(graph_hdfs_path + "/paper_cite_paper")


# In[115]:


dirs = hdfs_list_contents(graph_hdfs_path + "/paper_cite_paper", content_type="directories")
for dir in dirs:
    hdfs_save_string(dir)


# In[107]:


table_schema = {}
table_schema["node_col_to_types"] = {"paper_id": "Paper_Node", "cite_paper_id": "Paper_Node"}
table_schema["feat_cols"] = []
table_schema["time_cols"] = ["year"]
table_schema["time_cols_formats"] = ["%Y"]

hdfs_save_json(graph_hdfs_path + "/paper_cite_paper", table_schema_file_name, table_schema)


# In[65]:


df = spark_runner.spark_session.read.parquet("file://" + graph_local_path + "/edge/author_write_paper")
df = df.filter("year <> 0")
df = df.dropDuplicates(['author_id', 'paper_id'])
df.coalesce(1).write.mode("overwrite").partitionBy("year").parquet(graph_hdfs_path + "/author_write_paper")


# In[116]:


dirs = hdfs_list_contents(graph_hdfs_path + "/author_write_paper", content_type="directories")
for dir in dirs:
    hdfs_save_string(dir)


# In[108]:


table_schema = {}
table_schema["node_col_to_types"] = {"author_id": "Author_Node", "paper_id": "Paper_Node"}
table_schema["feat_cols"] = []
table_schema["time_cols"] = ["year"]
table_schema["time_cols_formats"] = ["%Y"]

hdfs_save_json(graph_hdfs_path + "/author_write_paper", table_schema_file_name, table_schema)


# In[99]:


from pyspark.sql.functions import count

df = spark_runner.spark_session.read.parquet("file://" + graph_local_path + "/edge/author_org_in_paper")
df = df.filter("year <> 0")
df = df.dropDuplicates(['author_id', 'org', 'paper_id'])
df = df.groupBy('author_id', 'org', 'year').agg(count("*").alias("author_paper_count"))
df.coalesce(1).write.mode("overwrite").partitionBy("year").parquet(graph_hdfs_path + "/author_to_org")


# In[117]:


dirs = hdfs_list_contents(graph_hdfs_path + "/author_to_org", content_type="directories")
for dir in dirs:
    hdfs_save_string(dir)


# In[109]:


table_schema = {}
table_schema["node_col_to_types"] = {"author_id": "Author_Node", "org": "Org_Node"}
table_schema["feat_cols"] = ["author_paper_count"]
table_schema["time_cols"] = ["year"]
table_schema["time_cols_formats"] = ["%Y"]

hdfs_save_json(graph_hdfs_path + "/author_to_org", table_schema_file_name, table_schema)


# In[102]:


df = spark_runner.spark_session.read.parquet("file://" + graph_local_path + "/edge/author_org_in_paper")
df = df.filter("year <> 0")
df = df.dropDuplicates(['author_id', 'org', 'paper_id'])
df = df.groupBy('paper_id', 'org', 'year').agg(count("*").alias("org_in_paper_count"))
df.coalesce(1).write.mode("overwrite").partitionBy("year").parquet(graph_hdfs_path + "/paper_to_org")


# In[118]:


dirs = hdfs_list_contents(graph_hdfs_path + "/paper_to_org", content_type="directories")
for dir in dirs:
    hdfs_save_string(dir)


# In[110]:


table_schema = {}
table_schema["node_col_to_types"] = {"paper_id": "Paper_Node", "org": "Org_Node"}
table_schema["feat_cols"] = ["org_in_paper_count"]
table_schema["time_cols"] = ["year"]
table_schema["time_cols_formats"] = ["%Y"]

hdfs_save_json(graph_hdfs_path + "/paper_to_org", table_schema_file_name, table_schema)


# In[79]:


df = spark_runner.spark_session.read.parquet("file://" + graph_local_path + "/edge/paper_to_venue")
df = df.filter("year <> 0")
df.coalesce(1).write.mode("overwrite").partitionBy("year").parquet(graph_hdfs_path + "/paper_to_venue")


# In[119]:


dirs = hdfs_list_contents(graph_hdfs_path + "/paper_to_venue", content_type="directories")
for dir in dirs:
    hdfs_save_string(dir)


# In[111]:


table_schema = {}
table_schema["node_col_to_types"] = {"paper_id": "Paper_Node", "venue": "Venue_Node"}
table_schema["feat_cols"] = []
table_schema["time_cols"] = ["year"]
table_schema["time_cols_formats"] = ["%Y"]

hdfs_save_json(graph_hdfs_path + "/paper_to_venue", table_schema_file_name, table_schema)


# In[85]:


df = spark_runner.spark_session.read.parquet("file://" + graph_local_path + "/edge/paper_to_keyword")
df = df.filter("year <> 0")
df = df.dropDuplicates(['paper_id', 'keyword'])
df.coalesce(1).write.mode("overwrite").partitionBy("year").parquet(graph_hdfs_path + "/paper_to_keyword")


# In[120]:


dirs = hdfs_list_contents(graph_hdfs_path + "/paper_to_keyword", content_type="directories")
for dir in dirs:
    hdfs_save_string(dir)


# In[112]:


table_schema = {}
table_schema["node_col_to_types"] = {"paper_id": "Paper_Node", "keyword": "Keyword_Node"}
table_schema["feat_cols"] = []
table_schema["time_cols"] = ["year"]
table_schema["time_cols_formats"] = ["%Y"]

hdfs_save_json(graph_hdfs_path + "/paper_to_keyword", table_schema_file_name, table_schema)


# In[ ]:




