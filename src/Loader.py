import os
import time
import datetime
import random

import torch
from torch import nn, threshold
from torch.nn import functional as F
from torch.utils import data
import torch.optim.lr_scheduler as lr_scheduler
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
from torch.utils.data import Dataset

from transformers import BertTokenizer, BertModel
import scipy.sparse as sp
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score,\
                            f1_score,accuracy_score,auc,roc_curve

from tqdm import tqdm
from loguru import logger
import pandas as pd
from tdc.multi_pred import DTI

from Setting import base_path

neg_label = 1  # false 
pos_label = 0  # true


def sample_stat(df):
    neg_samples = df[df.Y == neg_label]
    pos_samples =  df[df.Y == pos_label]
    neg_label_num = neg_samples.shape[0]
    pos_label_num = pos_samples.shape[0]
    logger.info(f'neg/pos:{neg_label_num}/{pos_label_num}, neg:{neg_label_num * 100 //(neg_label_num + pos_label_num)}%, pos:{pos_label_num * 100 //(neg_label_num + pos_label_num)}%')
    return neg_label_num, pos_label_num

def df_data_preprocess(df):
    df = df.dropna() # drop NaN rows
    df['Drug_ID'] = df['Drug_ID'].astype(str)
    return df

def dti_extract(IDs, df):
    total=len(IDs)
    features = torch.eye(total)
    return features

def df_nodes(df_raw):
    df = df_raw
    df["Drug_ID"] = df["Drug_ID"].astype(str)
    df["Target_ID"] = df["Target_ID"].astype(str)
    drugs = df["Drug_ID"].values
    drug_entities = np.unique(drugs.T.flatten()).tolist()
    targets = df['Target_ID'].values
    target_entities= np.unique(targets.T.flatten()).tolist()
    unique_entities = drug_entities + target_entities
    index = list(range(len(unique_entities)))
    idx_dict = dict(zip(unique_entities, index))
    x = torch.eye(len(unique_entities))
    return x, idx_dict

def df_encode_edges(df_s, idx_dict): # input is split from raw_data
    df = df_s
    df_pos = df[df.Y == pos_label]
    pos_edges = df_pos[["Drug_ID", "Target_ID"]].values
    pos_edge_index_list = [[idx_dict[d] for d,_ in pos_edges], [idx_dict[t] for _,t in pos_edges]]
    pos_edge_index = torch.tensor(pos_edge_index_list, dtype=torch.long)
    return pos_edge_index

def get_link_labels(pos_edge_index, neg_edge_index):
    num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
    link_labels = torch.zeros(num_links, dtype=torch.float)
    link_labels[:pos_edge_index.size(1)] = 1.
    return link_labels

def df_edge_index(df, label, idx_dict):
    df = df[df.Y == label]
    edges = df[["Drug_ID", "Target_ID"]].values
    edge_index_list = [[idx_dict[d] for d,_ in edges], [idx_dict[t] for _,t in edges]]
    edge_index = torch.tensor(edge_index_list, dtype=torch.long)
    return edge_index

def df_train_neg_edges(df_train, df_raw, idx_dict):
    pos_edge_index = df_edge_index(df_raw, pos_label, idx_dict)
    num_nodes = np.unique(pos_edge_index.cpu().numpy()).size
    num_neg_edges = df_edge_index(df_train, pos_label, idx_dict).size(1)
    neg_edge_index = negative_sampling(pos_edge_index, num_nodes,num_neg_edges)
    return neg_edge_index

def df_decode_edges(df_s, idx_dict):
    pos_edge_index = df_edge_index(df_s, pos_label, idx_dict)
    neg_edge_index = df_edge_index(df_s, neg_label, idx_dict)
    return pos_edge_index, neg_edge_index

def df_fold(df, frac=[0.7,0.2,0.1], fold_seed=10):
    train_frac, val_frac, test_frac = frac
    test = df.sample(frac=test_frac, replace=False, random_state=fold_seed)
    train_val = df[~df.index.isin(test.index)]
    val = train_val.sample(
        frac=val_frac / (1 - test_frac), replace=False, random_state=1
    )
    train = train_val[~train_val.index.isin(val.index)]
    train_df = train.reset_index(drop=True)
    valid_df = val.reset_index(drop=True)
    test_df = test.reset_index(drop=True)
    logger.info(f"train:{len(train_df)}, valid:{len(valid_df)}, test:{len(test_df)}")
    return {
        "train": train_df,
        "valid": valid_df,
        "test": test_df,
    }

def load(name="DAVIS"):
    data_dti = DTI(name = name)
    if name in "DAVIS":
        data_dti.convert_to_log(form = 'binding')
        logger.info(f"{name}:\n{data_dti.get_data()}")
        data_dti.binarize(threshold = 7, order = 'descending') # 7
        raw_data = data_dti.get_data()
        raw_data = df_data_preprocess(raw_data)
        df = data_dti.balanced(oversample = False, seed = 42)
        df = df_data_preprocess(df)
        sample_stat(df)
        df_split = df_fold(df)
        df_split['raw'] = raw_data

    elif name == "BindingDB_Kd":
        data_dti.convert_to_log(form = 'binding')
        logger.info(f"{name}:\n{data_dti.get_data()}")
        data_dti.binarize(threshold = 7.6, order = 'descending') # 7.6
        raw_data = data_dti.get_data()
        raw_data = df_data_preprocess(raw_data)
        df = data_dti.balanced(oversample = False, seed = 42)
        df = df_data_preprocess(df)
        sample_stat(df)
        df_split = df_fold(df)
        df_split['raw'] = raw_data
        
    elif name == "KIBA":
        logger.info(f"{name}:\n{data_dti.get_data()}")
        data_dti.binarize(threshold = 12.1, order = 'descending') # 12.1\
        raw_data = data_dti.get_data()
        raw_data = df_data_preprocess(raw_data)
        df = data_dti.balanced(oversample = False, seed = 42)
        df = df_data_preprocess(df)
        sample_stat(df)
        df_split = df_fold(df)
        df_split['raw'] = raw_data
    else:
        logger.error(f"dataset {name} is not supported")
        return
    logger.info(f"train df:\n{df_split['train']}")
    # logger.info(f"valid df:\n{df_split['valid']}")
    # logger.info(f"test df:\n{df_split['test']}")
    return df_split

class MyDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx].values
        Drug_ID = data[0]
        Drug = data[1]
        Target_ID = data[2]
        Target = data[3]
        Y = data[4]
        return   Drug_ID, Drug, Target_ID, Target, Y

if __name__ == '__main__':
    df_split = load(name = "DAVIS")
    # df_split['train'].to_csv(f"{base_path}/data/davis_train.csv")
    load(name = "BindingDB_Kd")
    load(name = "KIBA")
    print("done")