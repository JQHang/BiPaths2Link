#!/usr/bin/env python
# coding: utf-8

# In[1]:


from joinminer.pyspark import ResilientSparkRunner
from joinminer.python import mkdir, setup_logger, time_costing, read_json_file
from joinminer.hdfs import hdfs_save_json

from datetime import datetime


# In[2]:


# 获得项目文件夹根目录路径
from joinminer import PROJECT_ROOT

# 日志信息保存文件名
log_files_dir = PROJECT_ROOT + '/data/result_data/log_files/new_cite_label'
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


config_dict = {
                "spark.default.parallelism": "80",
                "spark.sql.shuffle.partitions": "160",
                "spark.sql.broadcastTimeout": "3600",
                "spark.driver.memory": "50g",
                "spark.driver.cores": "8",
                "spark.driver.maxResultSize": "0",
                "spark.executor.memory": "20g",
                "spark.executor.cores": "4",
                "spark.executor.instances": "20"
            }

# 启动spark
spark_runner = ResilientSparkRunner(config_dict = config_dict)


# In[4]:


from pyspark.sql.functions import when, lit, col

# 计算author在year及之前引用过的论文，计算author在year之后引用过在year及之前发表的且不是他本人写的论文，去重
cited_paper_year = "2019"
new_cited_paper_year = "2022"

author_write_paper_df = spark_runner.spark_session.read.parquet(graph_hdfs_path + "/author_write_paper")
author_write_paper_df = author_write_paper_df.filter(col("year") <= cited_paper_year)

paper_cite_paper_df = spark_runner.spark_session.read.parquet(graph_hdfs_path + "/paper_cite_paper")
paper_cite_paper_df = paper_cite_paper_df.filter(col("year") <= cited_paper_year)

author_cited_paper_df = author_write_paper_df.join(paper_cite_paper_df, on = ["paper_id", "year"], how = "inner")
author_cited_paper_df = author_cited_paper_df.select("author_id", "cite_paper_id").distinct()

author_cited_paper_df.persist()
print(author_cited_paper_df.count())

new_author_write_paper_df = spark_runner.spark_session.read.parquet(graph_hdfs_path + "/author_write_paper")
new_author_write_paper_df = new_author_write_paper_df.filter((col("year") > cited_paper_year) & (col("year") < new_cited_paper_year))

new_paper_cite_paper_df = spark_runner.spark_session.read.parquet(graph_hdfs_path + "/paper_cite_paper")
new_paper_cite_paper_df = new_paper_cite_paper_df.filter((col("year") > cited_paper_year) & (col("year") < new_cited_paper_year))

new_author_cited_paper_df = new_author_write_paper_df.join(new_paper_cite_paper_df, on = ["paper_id", "year"], how = "inner")
new_author_cited_paper_df = new_author_cited_paper_df.select("author_id", "cite_paper_id").distinct()

# author得在之前发表过论文
author_wrote_paper = author_write_paper_df.select("author_id").distinct()
new_author_cited_paper_df = new_author_cited_paper_df.join(author_wrote_paper, on = "author_id", how = "inner")

print(new_author_cited_paper_df.count())

# 新引用的论文得在year及之前发表
old_paper_df = spark_runner.spark_session.read.parquet(graph_hdfs_path + "/paper")
old_paper_df = old_paper_df.filter(col("year") <= cited_paper_year).select("paper_id").distinct()
old_paper_df = old_paper_df.withColumnRenamed("paper_id", "cite_paper_id")
new_author_cited_paper_df = new_author_cited_paper_df.join(old_paper_df, on = "cite_paper_id", how = "inner")

# 新引用的论文不能是作者本人写的
author_old_paper_df = author_write_paper_df.select("author_id", "paper_id").distinct()
author_old_paper_df = author_old_paper_df.withColumnRenamed("paper_id", "cite_paper_id")
new_author_cited_paper_df = new_author_cited_paper_df.join(author_old_paper_df, on = ["author_id", "cite_paper_id"], how="leftanti")

# 去除之前引用过的论文
author_new_cite_paper_df = new_author_cited_paper_df.join(author_cited_paper_df, on = ["author_id", "cite_paper_id"], how="leftanti")

author_new_cite_paper_df.persist()
print(author_new_cite_paper_df.count())
print(author_new_cite_paper_df.select("author_id").distinct().count())


# In[5]:


from pyspark.sql.functions import rand, when, lit

# 按 6 2 2分为训练 验证 测试
author_new_cite_paper_df = author_new_cite_paper_df.withColumn("random_value", rand())
    
# 根据随机值分配训练集、验证集和测试集
# 0-0.6对应train(占60%)，0.6-0.8对应valid(占20%)，0.8-1.0对应test(占20%)
author_new_cite_paper_df = author_new_cite_paper_df.withColumn("sample_type", 
                                when(author_new_cite_paper_df["random_value"] < 0.6, "train")
                                .when(author_new_cite_paper_df["random_value"] < 0.8, "valid")
                                .otherwise("test")
                           ).drop("random_value")

# 保存结果
author_new_cite_paper_df.coalesce(1).write.mode("overwrite").parquet(graph_hdfs_path + f"/author_new_cite_paper_in_2_years/year={cited_paper_year}")


# In[6]:


author_new_cite_paper_df.show()


# In[ ]:




