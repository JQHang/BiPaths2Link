from pyspark.sql import functions as F

def pyspark_vector_collect(df, group_cols, collect_config, pivot_config=None):
    """
    Collect vector columns in a PySpark DataFrame with optional pivoting.
    
    Parameters:
    -----------
    df : pyspark.sql.DataFrame
        Input DataFrame containing vector columns to collect.
    group_cols : str or list
        Column(s) to group by before performing collection.
    collect_config : list of tuple
        List of configurations for collection, where each tuple contains:
        - col_name: str
            Name of the column to collect.
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
        Resultant DataFrame after applying the collection operation and 
        (optionally) pivoting.
    """
    # Ensure group_cols is a list
    if isinstance(group_cols, str):
        group_cols = [group_cols]
        
    if pivot_config:
        # Handle pivoted collection
        pivot_col, pivot_values = pivot_config
        if not pivot_values:
            raise ValueError("pivot_values must be provided")
            
        collect_results = []
        for val in pivot_values:
            # Filter rows for the current pivot value
            filtered_df = df.filter(F.col(pivot_col) == val)
            
            # Collect original values for each group
            for col_name, alias in collect_config:
                collect_result = filtered_df.groupBy(group_cols).agg(
                    F.collect_list(F.col(col_name)).alias(f"{val}_{alias}")
                )
                collect_results.append(collect_result)
        
        # Combine all collection results
        if collect_results:
            final_df = collect_results[0]
            for other_df in collect_results[1:]:
                final_df = final_df.join(other_df, group_cols, "outer")
            return final_df
        else:
            return df.select(group_cols).distinct()
    else:
        # Handle non-pivoted collection
        return df.groupBy(group_cols).agg(
            *[F.collect_list(F.col(col_name)).alias(alias) 
              for col_name, alias in collect_config]
        )