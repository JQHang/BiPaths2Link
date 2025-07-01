from ..python import time_costing

import json
import logging
import pandas as pd
from pyarrow import fs as pafs
from datetime import datetime

# 获得logger
logger = logging.getLogger(__name__)

def hdfs_create_dir(hdfs_path):
    """
    Creates a directory at the specified HDFS path.

    Args:
        hdfs_path (str): The HDFS path where the directory will be created.
    """  
    hdfs = pafs.HadoopFileSystem(host="default")
    try:
        hdfs.create_dir(hdfs_path)
        logger.info(f'Created directory at {hdfs_path}')
    except Exception as e:
        raise ValueError(f'Failed to create directory: {e}')

def hdfs_delete_dir(hdfs_path):
    """
    Deletes a directory at the specified HDFS path.

    Args:
        hdfs_path (str): The HDFS path of the directory to delete.
    """
    hdfs = pafs.HadoopFileSystem(host="default")

    try:
        if hdfs.get_file_info(hdfs_path).type != pafs.FileType.NotFound:
            hdfs.delete_dir(hdfs_path)
            logger.info(f'Deleted directory at {hdfs_path}')
        else:
            logger.info(f'Directory {hdfs_path} to delete does not exist.')
    except Exception as e:
        logger.error(f'Failed to delete directory: {e}')
        raise ValueError(f'Failed to delete directory: {e}')
        
def hdfs_list_contents(hdfs_path, content_type="all", recursive=False):
    """
    Lists the contents (files and/or directories) in a specified HDFS directory.

    Args:
        hdfs_path (str): The HDFS path to list contents from.
        content_type (str): The type of contents to return. Valid values are "all", "files", or "directories".
        recursive (bool): Whether to recursively list contents in subdirectories.

    Returns:
        list: A list of file and/or directory paths under the specified path.
    """
    hdfs = pafs.HadoopFileSystem(host="default")
    file_info_list = hdfs.get_file_info(pafs.FileSelector(hdfs_path, recursive=recursive))

    contents = []

    for info in file_info_list:
        if info.type == pafs.FileType.File:
            if content_type in ["all", "files"]:
                contents.append(info.path)
        elif info.type == pafs.FileType.Directory:
            if content_type in ["all", "directories"]:
                contents.append(info.path)

    return contents

def hdfs_check_file_exists(hdfs_path):
    """
    Checks if a file or directory exists in HDFS at the specified path.

    Args:
        hdfs_path (str): The HDFS path to check.

    Returns:
        bool: True if the file or directory exists, False otherwise.
    """
    hdfs = pafs.HadoopFileSystem(host="default")
    return hdfs.get_file_info(hdfs_path).type != pafs.FileType.NotFound

def hdfs_save_string(hdfs_path, file_name = '_SUCCESS', content = None):
    """
    Creates a text file at the specified HDFS path.

    Args:
        hdfs_path (str): The base path where the text file will be created.
        file_name (str): Name of the text file. Defaults to '_SUCCESS'.
        content (str, optional): Content to write to the text file. If None, writes a timestamp.
    """
    hdfs = pafs.HadoopFileSystem(host="default")
    text_file_path = f"{hdfs_path}/{file_name}"
    
    try:
        with hdfs.open_output_stream(text_file_path) as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if content is None:
                content = f"Write operation completed successfully at {timestamp}"
            f.write(content.encode('utf-8'))
            logger.info(f'{file_name} file created at {text_file_path} with content: {content}')
    except Exception as e:
        raise ValueError(f'Failed to create text file: {e}')

def hdfs_save_json(hdfs_path, file_name, data_dict):
    """
    Converts a dictionary to a string and saves it as a text file in HDFS.

    Args:
        hdfs_path (str): The base path where the text file will be created in HDFS.
        file_name (str): Name of the text file.
        data_dict (dict): The dictionary to be converted and saved.
    """
    try:
        # Convert dictionary to JSON string
        dict_str = json.dumps(data_dict, ensure_ascii=False)

        # Create HDFS text file with the dictionary content
        hdfs_save_string(hdfs_path, file_name, content = dict_str)

    except Exception as e:
        raise ValueError(f'Failed to convert and save dictionary: {e}')

def hdfs_read_string(hdfs_path, file_name):
    """
    Reads content from a text file located in an HDFS path.

    Args:
        hdfs_path (str): The base path where the text file is located.
        file_name (str): Name of the text file to read. Defaults to '_SUCCESS'.

    Returns:
        str or None: The content of the text file, or None if an error occurs.
    """
    hdfs = pafs.HadoopFileSystem(host="default")
    text_file_path = f"{hdfs_path}/{file_name}"

    try:
        with hdfs.open_input_stream(text_file_path) as f:
            return f.read().decode('utf-8')
    except Exception as e:
        raise ValueError(f'Failed to read text file: {e}')

def hdfs_read_json(hdfs_path, file_name):
    """
    Reads a dictionary saved as a text file in HDFS and converts it back to a dictionary.

    Args:
        hdfs_path (str): The base path where the text file is located in HDFS.
        file_name (str): Name of the text file containing the dictionary.

    Returns:
        dict: The dictionary read from the text file.
    """
    try:
        # Read the content of the text file from HDFS
        dict_str = hdfs_read_string(hdfs_path, file_name)

        if dict_str is not None:
            # Convert the JSON string back to a dictionary
            data_dict = json.loads(dict_str)
            logger.info(f'Dictionary read successfully from {hdfs_path}/{file_name}')
            return data_dict
        else:
            logger.warning(f'Failed to read dictionary from {hdfs_path}/{file_name}')
            return None

    except Exception as e:
        raise ValueError(f'Failed to read and convert dictionary: {e}')

def hdfs_check_partitions(base_path, partition_cols = None, partition_values = None):
   """
   Check if _SUCCESS files exist in specified HDFS partition paths
   
   Args:
       base_path (str): Base HDFS path
       partition_cols (list): List of partition column names
       partition_values (list): List of partition value combinations
       
   Returns:
       tuple: (whether all partitions are complete, list of missing partition values)
   """
   missing_values = []
   is_complete = True
   
   # Handle empty partition_cols case
   if not partition_cols:
       return hdfs_check_file_exists(base_path + "/_SUCCESS"), []
   
   # Check each partition combination
   for values in partition_values:
       if len(partition_cols) != len(values):
           raise ValueError("Number of partition columns and values don't match")
           
       # Build partition path
       partition_path = base_path
       for col, val in zip(partition_cols, values):
           partition_path += f"/{col}={val}"
           
       # Check _SUCCESS file existence
       if not hdfs_check_file_exists(partition_path + "/_SUCCESS"):
           is_complete = False
           missing_values.append(values)
   
   return is_complete, missing_values

def hdfs_parquets_to_pandas(hdfs_path: str, columns: list) -> pd.DataFrame:
    """
    Read all parquet files from an HDFS path, keep only specified columns and merge into a single DataFrame.
    
    Args:
        hdfs_path (str): Path on HDFS
        columns (list): List of column names to keep
        
    Returns:
        pd.DataFrame: Combined DataFrame containing only the specified columns
    """
    # Get all files using the existing hdfs_list_contents function
    all_files = hdfs_list_contents(hdfs_path, content_type="files", recursive=True)
    
    # Filter for parquet files
    parquet_files = [f for f in all_files if f.endswith('.parquet')]
    
    # Assert that parquet files exist
    assert parquet_files, f"No parquet files found in path {hdfs_path}"

    # Set hdfs
    hdfs = pafs.HadoopFileSystem(host="default")
    
    # Read and merge all parquet files
    dfs = []
    for file_path in parquet_files:
        # Read parquet file from HDFS, keeping only specified columns
        df = pd.read_parquet(file_path, columns=columns, filesystem=hdfs)
        dfs.append(df)
    
    logger.info(f"Successfully read {len(parquet_files)} parquet files")
    
    return pd.concat(dfs, ignore_index=True)
