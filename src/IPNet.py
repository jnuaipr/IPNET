import datetime

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchtext
from torchtext.data import get_tokenizer
from torch.utils.data import DataLoader, Dataset

import numpy as np
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from loguru import logger
import pandas as pd
from tdc.multi_pred import DTI

from Loader import load, df_nodes, df_encode_edges, df_decode_edges,get_link_labels,df_train_neg_edges, MyDataset,neg_label,pos_label
from Utils import evaluate, setup_seed, dti_tokenizer,check_model, load_model, save_model, save_metrics
from GCN import GCN_Net, graph_train
from TransCNN import DTISeqPredictNet, seq_train


class IntelliPredictNet(nn.Module):
    def __init__(self, x, idx_dict, pos_edge_index, GCN, SeqNet, alpha = 0.9):
        super(IntelliPredictNet,self).__init__()
        self.x = x
        self.idx_dict = idx_dict
        self.pos_edge_index = pos_edge_index
        self.GraphNet = GCN
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.SeqNet = SeqNet
        self.alpha = alpha
    
    def dt_edge_index(self, drugs, targets):
        edge_index_list = [[self.idx_dict[d] for d in drugs], [self.idx_dict[t] for t in targets]]
        edge_index = torch.tensor(edge_index_list, dtype=torch.long)
        return edge_index
    
    def forward(self, drugs, targets, drug_ids, target_ids):
        n_dt = len(drugs)
        edge_index = self.dt_edge_index(drug_ids, target_ids).to(self.device)
        z = self.GraphNet.encode(self.x.to(self.device), self.pos_edge_index.to(self.device))
        g_logits = self. GraphNet.decode2(z.to(self.device), edge_index.to(self.device))
        s_logits = torch.empty(n_dt, dtype=float).to(self.device)
        for i in range(n_dt):
            drug = dti_tokenizer(drugs[i])
            target = dti_tokenizer(targets[i])
            logit = self.SeqNet(drug.to(self.device), target.to(self.device))
            s_logits[i] = logit
        logits = (1-self.alpha)*g_logits + self.alpha*s_logits
        return logits

@torch.no_grad()
def valid(model, valid_data):
    dti_dataset = MyDataset(valid_data)
    dti_data = DataLoader(dti_dataset,batch_size=64, shuffle=True, drop_last=True)
    metrics = []
    for dti in tqdm(dti_data, "valid"):
            drug_ids= dti[0]
            drugs = dti[1]
            target_ids = dti[2]
            targets = dti[3]
            labels = dti[4]
            logits = model(drugs, targets, drug_ids, target_ids)
            MSE, CI, MAE, R2 = evaluate(np.array(labels), logits.detach().cpu().numpy())
            metrics.append([MSE, CI, MAE, R2])
    metrics = np.array(metrics)
    MSE, CI, MAE, R2 = np.mean(metrics, axis=0)
    logger.info(f"[valid] MSE:{round(MSE,2)}, CI:{round(CI,2)}, MAE:{round(MAE, 2)}, R2:{round(R2, 2)}")
    return MSE, CI, MAE, R2

@torch.no_grad()
def test(model, test_data):
    model.eval()
    dti_dataset = MyDataset(test_data)
    dti_data = DataLoader(dti_dataset,batch_size=64, shuffle=True, drop_last=True)
    metrics = []
    for dti in tqdm(dti_data, "test"):
            drug_ids= dti[0]
            drugs = dti[1]
            target_ids = dti[2]
            targets = dti[3]
            labels = dti[4]
            logits = model(drugs, targets, drug_ids, target_ids)
            MSE, CI, MAE, R2 = evaluate(np.array(labels), logits.detach().cpu().numpy())
            metrics.append([MSE, CI, MAE, R2])
    metrics = np.array(metrics)
    MSE, CI, MAE, R2 = np.mean(metrics, axis=0)
    logger.info(f"[test] MSE:{round(MSE,2)}, CI:{round(CI,2)}, MAE:{round(MAE, 2)}, R2:{round(R2, 2)}")
    return MSE, CI, MAE, R2


def ipnet_train(df_split, dataset_name="DAVIS", epoch=1, batch_size=8):
    
    ## Setting and Data Preprocessing
    setup_seed(10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df_split = load(name = dataset_name)
    train_data = df_split['train']
    valid_data = df_split['valid']
    test_data = df_split['test']
    raw_data =  df_split['raw']
    x, idx_dict = df_nodes(raw_data) # nodes info
    pos_edge_index = df_encode_edges(raw_data,idx_dict)
    dti_dataset = MyDataset(train_data)
    dti_data = DataLoader(dti_dataset,batch_size=batch_size, shuffle=True, drop_last=True)
    
    ## Graph Network Train
    if check_model(f"IPNet-Graph-{dataset_name}"):
        GraphNet = GCN_Net(x.size(1)).to(device)
        load_model(GraphNet, f"IPNet-Graph-{dataset_name}")
    else:
        GraphNet = graph_train(df_split, dataset_name=dataset_name)

    ## Sequence Network Train
    if check_model(f"IPNet-Seq-{dataset_name}"):
        SeqNet = DTISeqPredictNet().to(device)
        load_model(SeqNet, f"IPNet-Seq-{dataset_name}")
    else:
        SeqNet = seq_train(df_split, dataset_name=dataset_name)
    
    ## Intelligent Prediction Network Train
    IPNet = IntelliPredictNet(x, idx_dict, pos_edge_index, GraphNet, SeqNet)
    optimizer = torch.optim.Adam(params=IPNet.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    IPNet.train()
    metrics = np.empty((epoch+1,5),dtype=float).tolist()
    for e in range(epoch):
        for dti in tqdm(dti_data, f"epoch{e}"):
            optimizer.zero_grad()  # 清空梯度
            drug_ids= dti[0]
            drugs = dti[1]
            target_ids = dti[2]
            targets = dti[3]
            labels = dti[4]
            logits = IPNet(drugs, targets, drug_ids, target_ids)
            loss = criterion(logits, torch.tensor(labels, dtype=float, device=device)) # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
        MSE, CI, MAE, R2 = valid(IPNet, valid_data)
        metrics[e] = ["valid", MSE, CI, MAE, R2]
    MSE, CI, MAE, R2 = test(IPNet, test_data)
    metrics[e+1] = ["test", MSE, CI, MAE, R2]
    save_metrics(metrics, f"IPNet-{dataset_name}")
    return IPNet
    
if __name__ == '__main__':
    dataset_name = "DAVIS"
    log_file = logger.add(f"/home/yang/sda/github/IPNET/output/log/IPNet-{dataset_name}-{str(datetime.date.today())}.log")
    df_split = load(name = dataset_name)
    model = ipnet_train(df_split,dataset_name=dataset_name)
    save_model(model, f"IPNet-{dataset_name}")
    logger.remove(log_file)
    print("done")