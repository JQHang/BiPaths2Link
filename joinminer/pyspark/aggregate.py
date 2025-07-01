from .dataframe_manager import rename_columns

import re
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, udf, rand
from pyspark.sql.functions import count as count_, mean as mean_, sum as sum_
from pyspark.sql.functions import max as max_, min as min_, first as first_
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.ml.stat import Summarizer

# 可以考虑优化成各个聚合函数带自己的pivot config(通过select实现)，这样time_agg就不需要分别聚合再join了
def pyspark_aggregate(df, group_columns, agg_config, pivot_config=None):
    """
    Aggregates a PySpark DataFrame based on the provided configuration, with optional pivot functionality.
    
    Parameters:
    df (DataFrame): Input PySpark DataFrame to aggregate.
    group_columns (list): List of columns to group by.
    agg_config (list of tuples): Each tuple contains three values:
        - column name to aggregate on (str)
        - aggregation function (str: 'count', 'mean', 'sum', 'max', 'min', 'first')
        - output alias for the aggregated column (str)
    pivot_config (dict, optional): Configuration for pivot operation with keys:
        - column (str): Column to pivot on
        - values (list, optional): List of values to pivot. If None, all distinct values will be used
    
    Returns:
    DataFrame: Aggregated DataFrame, potentially with pivot applied.
    
    Example:
    >>> agg_config = [
    ...     ('amount', 'sum', 'total_amount'),
    ...     ('quantity', 'mean', 'avg_quantity')
    ... ]
    >>> pivot_config = {
    ...     'column': 'category',
    ...     'values': ['Electronics', 'Clothing']
    ... }
    >>> result = pyspark_aggregate(df, ['date'], agg_config, pivot_config)
    """
    # List to hold aggregation expressions
    aggregation_expressions = []
    
    # Build the aggregation expressions based on configuration
    for column, agg_function, alias in agg_config:
        if column == "*" and agg_function != "count":
            raise ValueError("The '*' column can only be used with the 'count' function.")
        
        if agg_function == 'count':
            aggregation_expressions.append(count_(column).alias(alias))
        elif agg_function == 'mean':
            aggregation_expressions.append(mean_(column).alias(alias))
        elif agg_function == 'sum':
            aggregation_expressions.append(sum_(column).alias(alias))
        elif agg_function == 'max':
            aggregation_expressions.append(max_(column).alias(alias))
        elif agg_function == 'min':
            aggregation_expressions.append(min_(column).alias(alias))
        elif agg_function == 'first':
            aggregation_expressions.append(first_(column).alias(alias))
        else:
            raise ValueError(f"Unsupported aggregation function: {agg_function}")
    
    # Handle pivot if configured
    if pivot_config:
        pivot_column = pivot_config['column']
        pivot_values = pivot_config.get('values')
        
        # Remove pivot column from group columns if it's there
        group_columns = [col for col in group_columns if col != pivot_column]
        
        # Create pivot operation
        pivot_df = df.groupBy(group_columns)
        
        if pivot_values:
            pivot_df = pivot_df.pivot(pivot_column, pivot_values)
        else:
            pivot_df = pivot_df.pivot(pivot_column)
            
        # Apply aggregations
        aggregated_df = pivot_df.agg(*aggregation_expressions)

        # 如果只有一个聚合表达式，因为这时pivot不会添加聚合别名，故重命名列以包含聚合别名
        if len(agg_config) == 1:
            _, _, alias = agg_config[0]
            # 获取所有非分组列（即pivot后生成的列）
            pivot_columns = [c for c in aggregated_df.columns if c not in group_columns]
            
            # 重命名列：从 pivot_value 改为 pivot_value_alias，以后得优化成rename_columns函数的方式
            for pivot_col in pivot_columns:
                new_name = f"{pivot_col}_{alias}"
                aggregated_df = aggregated_df.withColumnRenamed(pivot_col, new_name)
    else:
        # Standard groupBy aggregation without pivot
        aggregated_df = df.groupBy(group_columns).agg(*aggregation_expressions)
    
    return aggregated_df

def pyspark_vector_aggregate(df, group_cols, agg_config, pivot_config=None):
    """
    Aggregate vector columns in a PySpark DataFrame with optional pivoting.
    Only supports regular aggregations (e.g., mean, sum).
    
    Parameters:
    -----------
    df : pyspark.sql.DataFrame
        Input DataFrame containing vector columns to aggregate.
    group_cols : str or list
        Column(s) to group by before performing aggregation.
    agg_config : list of tuple
        List of configurations for aggregation, where each tuple contains:
        - col_name: str
            Name of the column to aggregate.
        - agg_function: str
            Aggregation method, which can be one of the following:
            'mean', 'sum', 'variance', 'std', 'count', 'numNonzeros',
            'max', 'min', 'normL2', 'normL1'
        - alias: str
            Alias for the output column name.
    pivot_config : tuple, optional
        A tuple specifying pivoting options:
        - pivot_col: str
            Name of the column to pivot on.
        - pivot_values: list
            List of values to pivot.

    Returns:
    --------
    pyspark.sql.DataFrame
        Resultant DataFrame after applying the specified aggregation and 
        (optionally) pivoting.
    """
    # Ensure group_cols is a list
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    
    # Define available aggregation functions
    summarizer_funcs = {
        'mean': Summarizer.mean,
        'sum': Summarizer.sum,
        'variance': Summarizer.variance,
        'std': Summarizer.std,
        'count': Summarizer.count,
        'numNonzeros': Summarizer.numNonZeros,
        'max': Summarizer.max,
        'min': Summarizer.min,
        'normL2': Summarizer.normL2,
        'normL1': Summarizer.normL1
    }
    
    # Build aggregation expressions
    agg_exprs = [
        summarizer_funcs[agg_function](F.col(col_name)).alias(alias)
        for col_name, agg_function, alias in agg_config
        if agg_function in summarizer_funcs
    ]
    
    if pivot_config:
        # Handle pivoted aggregation
        pivot_col, pivot_values = pivot_config
        if not pivot_values:
            raise ValueError("pivot_values must be provided")
        
        # Process regular aggregation with pivot
        pivot_df = df.groupBy(group_cols + [pivot_col]).agg(*agg_exprs)
        pivot_exprs = [
            F.max(F.when(F.col(pivot_col) == val, F.col(alias))).alias(f"{val}_{alias}")
            for val in pivot_values
            for _, _, alias in agg_config
        ]
        return pivot_df.groupBy(group_cols).agg(*pivot_exprs)
    else:
        # Handle non-pivoted aggregation
        return df.groupBy(group_cols).agg(*agg_exprs)
    