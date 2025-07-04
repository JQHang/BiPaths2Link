import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

def top_k_accuracy_score(y_true, y_score, k):
    sorted_pred = np.argsort(y_score)
    sorted_pred = sorted_pred[::-1]
    sorted_pred = sorted_pred[:k]

    hits = y_true[sorted_pred]
    
    return np.sum(hits)/k

def binary_problem_evaluate(y_true, y_score, print_figure = True, Top_K_list = [5000, 10000, 20000], 
                            Top_K_percent_list = [5, 10, 20, 50, 100]):
    # 转换y_true为int类型
    y_true = np.round(y_true).astype(int)
    
    # 找出不是0或1的元素
    invalid_mask = ~((y_true == 0) | (y_true == 1))
    invalid_indices = np.where(invalid_mask)[0]
    
    # 如果有无效值，显示它们的位置和值
    if len(invalid_indices) > 0:
        invalid_values = y_true[invalid_mask]
        error_msg = f"Labels must be 0 or 1. Found invalid values: {invalid_values} at indices: {invalid_indices}"
        raise ValueError(error_msg)
    
    binary_metrics = {}

    logloss = log_loss(y_true, y_score)
    binary_metrics["log_loss"] = logloss
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    binary_metrics["roc_auc"] = roc_auc
    
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    binary_metrics["pr_auc"] = pr_auc
    
    top_k_acc_dict = {}
    for tmp_aim_k in Top_K_list:
        top_k_acc_dict[tmp_aim_k] = top_k_accuracy_score(y_true, y_score, k = tmp_aim_k)
    binary_metrics["top_k_acc"] = top_k_acc_dict
    
    top_k_percent_acc_dict = {}
    for tmp_aim_k in Top_K_percent_list:
        top_k_percent_acc_dict[tmp_aim_k] = top_k_accuracy_score(y_true, y_score, k = tmp_aim_k * y_true.shape[0] // 100)
    binary_metrics["top_k_percent_acc"] = top_k_percent_acc_dict
    
    if print_figure:
        plt.plot(fpr, tpr, 'b', label = 'Val AUC = %0.3f' % roc_auc)
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
        
        plt.plot(precision, recall)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('recall')
        plt.xlabel('precision')
        plt.show()

    return binary_metrics

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def top_percentage_hit_rate(y_true, y_pred, percentage=0.05):
    y_binary = np.where(y_true > 0, 1, 0)
    
    k = int(len(y_true) * percentage)
    
    sorted_pred = np.argsort(y_pred)
    sorted_pred = sorted_pred[::-1]
    sorted_pred = sorted_pred[:k]

    hits = y_binary[sorted_pred]
    
    return np.sum(hits)/k

def regression_evaluate(y_true, y_pred, print_figure=True, top_percent_list=[5, 10, 20, 50]):
    # RMSE and MAE
    print('RMSE:', rmse(y_true, y_pred))
    print('MAE:', mean_absolute_error(y_true, y_pred))
    
    # top_percentage_hit_rate
    top_hit_rate_dict = {}
    for percent in top_percent_list:
        top_hit_rate_dict[percent] = top_percentage_hit_rate(y_true, y_pred, percentage=percent/100)
    print('Top Percentage Hit Rates:', top_hit_rate_dict)
    
    # Residual plot
    if print_figure:
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(0, color='r', linestyle='--')
        plt.title("Residual Plot")
        plt.xlabel("Predicted values")
        plt.ylabel("Residuals")
        plt.show()
    
    return
