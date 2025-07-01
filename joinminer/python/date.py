from datetime import datetime

def time_values_reformat(time_values_list, src_formats, dst_formats):
   """
   Convert time values from source format to destination format
   """
   reformatted_time_values_list = []
   for time_values in time_values_list:
       dt = datetime.strptime(''.join(time_values), ''.join(src_formats))
       reformatted_time_values = [dt.strftime(fmt) for fmt in dst_formats]
       reformatted_time_values_list.append(reformatted_time_values)
           
   return reformatted_time_values_list

def python_to_spark_date_format(python_fmt):
    mappings = {
        # 年份
        '%Y': 'yyyy',
        '%y': 'yy',
        # 月份
        '%m': 'MM',
        '%b': 'MMM',
        '%B': 'MMMM',
        # 日期
        '%d': 'dd',
        # 小时
        '%H': 'HH',
        '%I': 'hh',
        # 分钟
        '%M': 'mm',
        # 秒
        '%S': 'ss',
        # 毫秒
        '%f': 'SSSSSS',
        # AM/PM
        '%p': 'a',
        # 时区
        '%z': 'Z',
        '%Z': 'z'
    }
    
    result = python_fmt
    for py_fmt, spark_fmt in mappings.items():
        result = result.replace(py_fmt, spark_fmt)
    return result