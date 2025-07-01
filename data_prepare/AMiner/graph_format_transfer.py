#!/usr/bin/env python
# coding: utf-8

# In[1]:


from joinminer import PROJECT_ROOT

import ijson
import pandas as pd

# 节点信息
paper_info = {"cols": ["paper_id", "paper_title", "lang", "doc_type", "year"], "data": []}
author_info = {"cols": ["author_id", "author_name"], "data": []}

# 边信息
paper_cite_paper_info = {"cols": ["paper_id", "cite_paper_id", "year"], "data": []}
author_write_paper_info = {"cols": ["author_id", "paper_id", "year"], "data": []}
author_org_in_paper_info = {"cols": ["author_id", "org", "paper_id", "year"], "data": []}
paper_to_venue_info = {"cols": ["paper_id", "venue", "year"], "data": []}
paper_to_keyword_info = {"cols": ["paper_id", "keyword", "year"], "data": []}

graph_local_path = PROJECT_ROOT + "/data/dataset/AMiner/graph"

with open(PROJECT_ROOT + "/data/dataset/AMiner/dblp_v14.json", "r") as f:
    for record_i, record in enumerate(ijson.items(f, "item")):
        # id列一定得有值，不是None或""
        paper_id = record["id"] if "id" in record else None
        paper_title = record["title"] if "title" in record else None
        lang = record["lang"] if "lang" in record else None
        doc_type = record["doc_type"] if "doc_type" in record else None
        year = record["year"] if "year" in record else None

        venue = record["venue"]['raw'].strip() if "venue" in record and 'raw' in record["venue"] else None
        
        if paper_id not in [None, ""] and year not in [None, ""]:
            paper_info["data"].append([paper_id, paper_title, lang, doc_type, year]) 

            # paper本身信息有效才能查找引用论文信息
            if "references" in record and len(record["references"]) > 0:
                for cite_paper_id in record["references"]:
                    if cite_paper_id not in [None, ""]:
                        paper_cite_paper_info["data"].append([paper_id, cite_paper_id, year]) 

            # paper本身信息有效才能加入对应的venue信息
            if venue not in [None, ""]:
                paper_to_venue_info["data"].append([paper_id, venue, year]) 

            # paper本身信息有效才能查找keywords信息
            if "keywords" in record and len(record["keywords"]) > 0:
                for keyword in record["keywords"]:
                    if keyword not in [None, ""]:
                        paper_to_keyword_info["data"].append([paper_id, keyword.strip(), year]) 

        # 检查作者信息
        if "authors" in record and len(record["authors"]) > 0:
            for author in record["authors"]:
                author_id = author["id"] if "id" in author else None
                author_name = author["name"] if "name" in author else None
                org = author["org"].strip() if "org" in author else None

                if author_id not in [None, ""]:
                    author_info["data"].append([author_id, author_name]) 

                    if paper_id not in [None, ""] and year not in [None, ""]:
                        author_write_paper_info["data"].append([author_id, paper_id, year]) 

                    if org not in [None, ""] and year not in [None, ""]:
                        author_org_in_paper_info["data"].append([author_id, org, paper_id, year]) 

        # 每50万份数据保存一次结果
        if (record_i + 1) % 500000 == 0:
            print(record_i)

            paper_df = pd.DataFrame(paper_info["data"], columns = paper_info["cols"])
            paper_df.to_parquet(f"{graph_local_path}/node/paper/{record_i}.parquet")
            paper_info["data"] = []

            author_df = pd.DataFrame(author_info["data"], columns = author_info["cols"])
            author_df.to_parquet(f"{graph_local_path}/node/author/{record_i}.parquet")
            author_info["data"] = []

            paper_cite_paper_df = pd.DataFrame(paper_cite_paper_info["data"], columns = paper_cite_paper_info["cols"])
            paper_cite_paper_df.to_parquet(f"{graph_local_path}/edge/paper_cite_paper/{record_i}.parquet")
            paper_cite_paper_info["data"] = []

            author_write_paper_df = pd.DataFrame(author_write_paper_info["data"], columns = author_write_paper_info["cols"])
            author_write_paper_df.to_parquet(f"{graph_local_path}/edge/author_write_paper/{record_i}.parquet")
            author_write_paper_info["data"] = []

            author_org_in_paper_df = pd.DataFrame(author_org_in_paper_info["data"], columns = author_org_in_paper_info["cols"])
            author_org_in_paper_df.to_parquet(f"{graph_local_path}/edge/author_org_in_paper/{record_i}.parquet")
            author_org_in_paper_info["data"] = []

            paper_to_venue_df = pd.DataFrame(paper_to_venue_info["data"], columns = paper_to_venue_info["cols"])
            paper_to_venue_df.to_parquet(f"{graph_local_path}/edge/paper_to_venue/{record_i}.parquet")
            paper_to_venue_info["data"] = []

            paper_to_keyword_df = pd.DataFrame(paper_to_keyword_info["data"], columns = paper_to_keyword_info["cols"])
            paper_to_keyword_df.to_parquet(f"{graph_local_path}/edge/paper_to_keyword/{record_i}.parquet")
            paper_to_keyword_info["data"] = []
            
# 保存最后一部分的结果
print(record_i)

paper_df = pd.DataFrame(paper_info["data"], columns = paper_info["cols"])
paper_df.to_parquet(f"{graph_local_path}/node/paper/{record_i}.parquet")
paper_info["data"] = []

author_df = pd.DataFrame(author_info["data"], columns = author_info["cols"])
author_df.to_parquet(f"{graph_local_path}/node/author/{record_i}.parquet")
author_info["data"] = []

paper_cite_paper_df = pd.DataFrame(paper_cite_paper_info["data"], columns = paper_cite_paper_info["cols"])
paper_cite_paper_df.to_parquet(f"{graph_local_path}/edge/paper_cite_paper/{record_i}.parquet")
paper_cite_paper_info["data"] = []

author_write_paper_df = pd.DataFrame(author_write_paper_info["data"], columns = author_write_paper_info["cols"])
author_write_paper_df.to_parquet(f"{graph_local_path}/edge/author_write_paper/{record_i}.parquet")
author_write_paper_info["data"] = []

author_org_in_paper_df = pd.DataFrame(author_org_in_paper_info["data"], columns = author_org_in_paper_info["cols"])
author_org_in_paper_df.to_parquet(f"{graph_local_path}/edge/author_org_in_paper/{record_i}.parquet")
author_org_in_paper["data"] = []

paper_to_venue_df = pd.DataFrame(paper_to_venue_info["data"], columns = paper_to_venue_info["cols"])
paper_to_venue_df.to_parquet(f"{graph_local_path}/edge/paper_to_venue/{record_i}.parquet")
paper_to_venue_info["data"] = []

paper_to_keyword_df = pd.DataFrame(paper_to_keyword_info["data"], columns = paper_to_keyword_info["cols"])
paper_to_keyword_df.to_parquet(f"{graph_local_path}/edge/paper_to_keyword/{record_i}.parquet")
paper_to_keyword_info["data"] = []


# In[ ]:




