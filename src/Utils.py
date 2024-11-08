import csv
import os
import random
import time

import torch

import scipy.sparse as sp
import numpy as np
from loguru import logger
from sklearn.metrics import roc_auc_score
import pandas as pd

from Setting import base_path

neg_label = 1
pos_label = 0

def class_metrics(y_label, y_pred):
    logger.info(f'y_label:{y_label}')
    logger.info(f'y_pred:{y_pred}')
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    total = len(y_label)
    for i in range(total):
        if y_pred[i] == pos_label and y_label[i] == pos_label:
            TP = TP + 1
        elif y_pred[i] == pos_label and y_label[i] == neg_label:
            FP = FP + 1
        elif y_pred[i] == neg_label and y_label[i] == neg_label:
            TN = TN + 1
        elif  y_pred[i] == neg_label and y_label[i] == pos_label:
            FN = FN + 1
        else:
            print(f"error,y_pred:{y_pred[i]},y_label:{y_label[i]}")
    logger.info(f"TP:{TP},TN:{TN},FP:{FP},FN:{FN}")
    if (TP + FN) == 0:
        sensitivity= 1.0
    else:
        sensitivity  = TP / (TP + FN)
    if (FP + TN) == 0:
        specificity = 1.0
    else:
        specificity = TN / (FP + TN)
    recall = sensitivity
    if (TP + FP) == 0:
        precision = 1.0
    else:
        precision = TP / (TP + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    if (2*TP + FN + FP) == 0:
        f1 = 1.0
    else:
        f1 = 2*TP /(2*TP + FN + FP)
    return {"sensitivity":sensitivity, "specificity":specificity,
            "recall":recall, "precision":precision, "accuracy":accuracy, 
            "f1":f1}

def csv_record(path, data):
    all_header = ['epoch','batch','lr','loss','avg_loss','epoch_loss','auprc', 'auroc', 'sensitivity', 'specificity',
              'recall','precision','cindex','accuracy','f1']  
    row = []
    header = []
    for name in all_header:
        if name in data.keys():
            row.append(data[name])
            header.append(name)
            
    if os.path.exists(path):
        with open(path, 'a',newline='') as f:
            write = csv.writer(f)
            write.writerow(row)
    else:
        with open(path, 'w', newline='') as f:
            write = csv.writer(f)
            write.writerow(header)
            write.writerow(row)
            
def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"create dir: {path}")
    else:
        logger.info(f"dir exists, {path}")
        
def save_model(model, path):
    torch.save(model.state_dict(),path)
    logger.info(f"save {path} model parameters done")

def load_model(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        model.eval()
        logger.info(f"load {path} model parameters done")
        
def normalize_adjacency_matrix(A, I):
    """
    Creating a normalized adjacency matrix with self loops.
    :param A: Sparse adjacency matrix.
    :param I: Identity matrix.
    :return A_tile_hat: Normalized adjacency matrix.
    """
    A_tilde = A + 2 * I
    degrees = A_tilde.sum(axis=0)[0].tolist()
    D = sp.diags(degrees, [0])
    D = D.power(-0.5)
    A_tilde_hat = D.dot(A_tilde).dot(D)
    return A_tilde_hat

def create_propagator_matrix(A, device):
    """
    Creating a propagator matrix.
    :param graph: NetworkX graph.
    :return propagator: Dictionary of matrix indices and values.
    """
    I = sp.eye(A.shape[0])
    A_tilde_hat = normalize_adjacency_matrix(A, I)
    propagator = dict()
    A_tilde_hat = sp.coo_matrix(A_tilde_hat)
    ind = np.concatenate([A_tilde_hat.row.reshape(-1, 1), A_tilde_hat.col.reshape(-1, 1)], axis=1)
    propagator["indices"] = torch.LongTensor(ind.T).to(device)
    propagator["values"] = torch.FloatTensor(A_tilde_hat.data).to(device)
    return propagator

def features_to_sparse(features, device):
    """
    Reading the feature matrix stored as JSON from the disk.
    :param path: Path to the JSON file.
    :return out_features: Dict with index and value tensor.
    """
    index_1, index_2 = features.nonzero()
    values = [1.0]*len(index_1)
    node_count = features.shape[0]
    feature_count = features.shape[1]
    features = sp.coo_matrix((values, (index_1, index_2)),
                                 shape=(node_count, feature_count),
                                 dtype=np.float32)
    out_features = dict()
    ind = np.concatenate([features.row.reshape(-1, 1), features.col.reshape(-1, 1)], axis=1)
    out_features["indices"] = torch.LongTensor(ind.T).to(device)
    out_features["values"] = torch.FloatTensor(features.data).to(device)
    out_features["dimensions"] = features.shape
    return out_features

def setup_seed(seed):
    # https://zhuanlan.zhihu.com/p/462570775
    # torch.use_deterministic_algorithms(True) # 检查pytorch中有哪些不确定性
    torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # 大于CUDA 10.2 需要设置
    logger.info("seed: %d, random:%.4f, torch random:%.4f, np random:%.4f" %(seed, random.random(), torch.rand(1), np.random.rand(1)))

def random_slice(vec, limit):
    vlen = len(vec)
    if vlen < limit:
        padding = np.zeros((limit - vlen),dtype=np.int32).tolist()
        vec = vec + padding
    else:
        idx = random.randint(0, vlen - limit)
        vec = vec[idx:idx + limit]
    return vec

def dti_tokenizer(lines, limit=100):
    vecs = []
    for line in lines:
        vec = []
        for c in line:
            ascii = ord(c)
            vec.append(ascii)
        vec = random_slice(vec, limit)
        vecs.append(vec)
    return torch.tensor(vecs)

def calc_mse(y_ture, y_pred): # input type is numpy
    return float(np.mean((y_ture - y_pred)**2))

def calc_cindex(y_true, scores):
    cindex = roc_auc_score(y_true, scores)
    return cindex

def calc_mae(y_true, y_pred):    
    return float(np.mean(np.abs(pd.Series(y_true) - pd.Series(y_pred))))

def calc_mape(y_true, y_pred):
    return float(np.mean(np.abs((y_true - y_pred) / y_true)))

def calc_r2(y_true, y_pred):
    u = np.sum((y_true - y_pred) ** 2)
    v = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - (u / v))

def evaluate(y_ture, y_pred):
    CI = calc_cindex2(y_ture, y_pred)
    MSE = calc_mse(y_ture, y_pred)
    MAE = calc_mae(y_ture, y_pred)
    R2 = calc_r2(y_ture, y_pred)
    return MSE, CI, MAE, R2

def calc_cindex2(y_true, scores):
    """
    计算C-Index（一致性指数）。
    
    参数：
    y_true: 实际生存时间或风险事件发生的时间（如果是生存分析）。
    scores: 模型的预测得分（可以是预测的生存时间或风险分数）。
    
    返回：
    C-Index 值。
    """
    n = len(y_true)
    assert len(scores) == n

    concordant = 0
    permissible = 0
    
    # 遍历所有的样本对
    for i in range(n):
        for j in range(i + 1, n):
            # 样本对必须是可比较的
            if y_true[i] != y_true[j]:
                permissible += 1
                # 检查模型预测的一致性
                if (scores[i] > scores[j] and y_true[i] > y_true[j]) or \
                   (scores[i] < scores[j] and y_true[i] < y_true[j]):
                    concordant += 1
                elif scores[i] == scores[j]:
                    concordant += 0.5  # 如果得分相等，则算作0.5的一致性对
    
    if permissible == 0:
        return 0  # 如果没有可比较的样本对，返回0

    return concordant / permissible

def save_model(model,file_name):
    # path = f"/home/yang/sda/github/IPNET/output/pt/{file_name}.pt"
    path = f"{base_path}output/pt/{file_name}.pt"
    torch.save(model.state_dict(), path)
    logger.info("save %s model parameters done, %s" %(file_name, path))

def load_model(model, file_name):
    path = f"{base_path}output/pt/{file_name}.pt"
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        model.eval()
        logger.info("load %s model parameters done, %s." %(file_name, path))

def check_model(file_name):
    path = f"{base_path}output/pt/{file_name}.pt"
    if os.path.exists(path):
        return True
    else:
        return False

def save_metrics(metrics, file_name, columns=["Type", "MSE", "CI", "MAE", "R2"]):
    path = f"{base_path}output/csv/{file_name}_Metrics_"+str(time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime()))+".csv"
    df = pd.DataFrame(metrics)
    df.columns = columns
    df.to_csv(path)
    
def save_loss(losses, file_name, columns=['loss']):
    path = f"{base_path}output/csv/{file_name}_loss_"+str(time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime()))+".csv"
    df = pd.DataFrame(losses)
    df.columns = columns
    df.to_csv(path)
    
if __name__ == '__main__':
    # dataset_name = "DAVIS"
    # file_name = f"IPNet-Seq-{dataset_name}"
    # path = f"/home/yang/sda/github/IPNET/output/csv/{file_name}-Metrics-"+str(time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime()))+".csv"
    # print(path)
    lines=[
        "123456789abcdef",
        "123456789a",
        "12345",
        "123456789ab",
        "123456789",
    ]
    vecs = dti_tokenizer(lines, limit=10)
    print(vecs)
    
