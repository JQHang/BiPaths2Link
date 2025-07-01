# 初始化各个表通过time aggregation聚合到graph time的方案
def time_aggs_init(src_feat_cols, time_aggs_configs):
    # 记录聚合的config信息和产生的feat_cols
    # 每一组聚合区间算一种配置，可以有多组配置分别聚合再outer join
    time_aggs = []
    
    # 依次处理各个time_agg对应的信息
    for time_aggs_config in time_aggs_configs:
        # 获得对应的聚合函数
        if "agg_funcs" in time_aggs_config:
            time_agg_funcs = time_aggs_config["agg_funcs"]
        else:
            time_agg_funcs = ["first"]
            
        # 检查之前是否已有使用同样的聚合函数的配置
        time_agg_index = 0
        for time_agg in time_aggs:
            if time_agg["agg_funcs"] == time_agg_funcs:
                break
            time_agg_index += 1
            
        # 如果没有就建立一个新的time_agg配置
        if time_agg_index >= len(time_aggs):
            # 创建一个新的time_agg配置
            time_agg = {}
            
            # 设定对应的聚合方式
            time_agg["agg_funcs"] = time_agg_funcs

            # 设定要聚合的特征列
            time_agg["src_feat_cols"] = src_feat_cols
            
            # 设定聚合的具体配置以及产生的特征列
            time_agg["agg_configs"] = []
            time_agg["agg_feat_cols"] = []
            for agg_func in time_agg["agg_funcs"]:
                if agg_func == 'count_*':
                    time_agg["agg_configs"].append(["*", "count", f"count_all"])
                    time_agg["agg_feat_cols"].append("count_all")
                elif agg_func in ['count', 'mean', 'sum', 'max', 'min', 'first']:
                    for feat_col in src_feat_cols:
                        time_agg["agg_configs"].append([feat_col, agg_func, f"{agg_func}_{feat_col}"])
                        time_agg["agg_feat_cols"].append(f"{agg_func}_{feat_col}")
            
            # 记录这组聚合方式内包含几组时间区间，各时间区间的名称以及这些时间区间会产出的特征列
            time_agg["time_ranges"] = []

            # 先向原始list中添加初始化信息，之后还会修正，用于保证time_agg_index位有数据
            time_aggs.append(time_agg)
        else:
            # 直接读取已存在的配置信息
            time_agg = time_aggs[time_agg_index]
            
        # 获得这组time_agg配置对应的时间单元
        time_unit = time_aggs_config["time_unit"]

        # 获得这组time_agg配置对应的时间区间的长度
        time_interval_len = time_aggs_config["time_interval_len"]

        # 这里还能加个步长的参数，之后优化
        
        # 依次处理该time_aggregation涉及到的各个时间区间
        for start_time_interval in range(*time_aggs_config["start_times_range"]):
            # 记录该time_range对应的全部信息
            time_range = {}

            # 首先是该组time_range的名称
            time_range["name"] = f"agg_{start_time_interval}_to_{start_time_interval + time_interval_len}_{time_unit}"
            
            # 首先是要聚合的时间点的相对位置
            time_range["time_points"] = []
            for time_interval in range(start_time_interval, start_time_interval + time_interval_len, 1):
                time_point = {}
                time_point["time_type"] = "relative"
                time_point["time_unit"] = time_unit
                time_point["time_interval"] = time_interval
                time_range["time_points"].append(time_point)

            # 记录该time_aggregation形成的全部特征列名 
            time_range["agg_feat_cols"] = [f"{time_range['name']}_{x}" for x in time_agg["agg_feat_cols"]]

            time_agg["time_ranges"].append(time_range)

        # 用更新后的配置信息替换原始的配置信息
        time_aggs[time_agg_index] = time_agg
    
    return time_aggs