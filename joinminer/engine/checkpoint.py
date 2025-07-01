import re
from datetime import datetime

class CheckpointNamer:
    @staticmethod
    def get_name(epoch, loss, timestamp=None):
        """生成checkpoint文件名
        
        Args:
            epoch (int): 当前轮次
            loss (float): 损失值
            timestamp (str, optional): 时间戳，默认为当前时间
            
        Returns:
            str: 格式化的文件名
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        return f"checkpoint_epoch{epoch}_loss{loss:.4f}_{timestamp}.pt"
    
    @staticmethod
    def parse_name(filename):
        """从文件名解析信息"""
        pattern = r'checkpoint_epoch(\d+)_loss(\d+\.\d+)_(\d{8}_\d{6})\.pt'
        match = re.match(pattern, filename)
        if match:
            return {
                'epoch': int(match.group(1)),
                'loss': float(match.group(2)),
                'timestamp': match.group(3)
            }
        return None