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

import scipy.sparse as sp
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score,\
                            f1_score,accuracy_score,auc,roc_curve
                            
import pandas as pd
from tqdm import tqdm
from loguru import logger
import pandas as pd
from tdc.multi_pred import DTI

neg_label = 1
pos_label = 0


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

def df_to_pyg(df):
    # output: {

    # 'index_to_entities': a dict map from ID in the data to node ID in the PyG object, 
    # 'split': {'train': df, 'valid': df, 'test': df}
    # }
    df["Drug_ID"] = df["Drug_ID"].astype(str)
    df["Target_ID"] = df["Target_ID"].astype(str)
    df_pos = df[df.Y == 1]
    df_neg = df[df.Y == 0]

    pos_edges = df_pos[
        ["Drug_ID", "Target_ID"]
    ].values
    neg_edges = df_neg[
        ["Drug_ID", "Target_ID"]
    ].values
    edges = df[["Drug_ID", "Target_ID"]].values
    unique_entities = np.unique(pos_edges.T.flatten()).tolist()
    index = list(range(len(unique_entities)))
    dict_ = dict(zip(unique_entities, index))
    edge_list1 = np.array([dict_[i] for i in pos_edges.T[0]])
    edge_list2 = np.array([dict_[i] for i in pos_edges.T[1]])

    edge_index = torch.tensor([edge_list1, edge_list2], dtype=torch.long)
    x = torch.tensor(np.array(index), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    # 'pyg_graph': the PyG graph object
    pyg_graph = data
    return pyg_graph

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
        data_dti.binarize(threshold = 7, order = 'descending') # 7
        df = data_dti.balanced(oversample = False, seed = 42)
        df = df_data_preprocess(df)
        sample_stat(df)
        pyg = df_to_pyg(df)
        df_split = df_fold(df)

    elif name == "BindingDB_Kd":
        data_dti.convert_to_log(form = 'binding')
        data_dti.binarize(threshold = 7.6, order = 'descending') # 7.6
        df = data_dti.balanced(oversample = False, seed = 42)
        df = df_data_preprocess(df)
        sample_stat(df)
        pyg = df_to_pyg(df)
        df_split = df_fold(df)
    elif name == "KIBA":
        data_dti.binarize(threshold = 12.1, order = 'descending') # 12.1
        df = data_dti.balanced(oversample = False, seed = 42)
        df = df_data_preprocess(df)
        sample_stat(df)
        pyg = df_to_pyg(df)
        df_split = df_fold(df)
    else:
        logger.error(f"dataset {name} is not supported")
        return
    logger.info(f"pyg: {pyg}")
    logger.info(f"train df:\n{df_split['train']}")
    logger.info(f"valid df:\n{df_split['valid']}")
    logger.info(f"test df:\n{df_split['test']}")
    return pyg,df_split

if __name__ == '__main__':
    load(name = "DAVIS")
    load(name = "BindingDB_Kd")
    load(name = "KIBA")
    print("done")