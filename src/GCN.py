import datetime

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

from loguru import logger
import numpy as np
from tqdm import tqdm
import pandas as pd

from Loader import load, df_nodes, df_encode_edges, df_decode_edges,get_link_labels,df_train_neg_edges
from Utils import save_metrics, setup_seed, save_model, evaluate, save_loss
from Setting import base_path

## https://blog.csdn.net/python_plus/article/details/136158335
## https://github.com/datawhalechina/team-learning-nlp/blob/master/GNN/Markdown%E7%89%88%E6%9C%AC/6-2-%E8%8A%82%E7%82%B9%E9%A2%84%E6%B5%8B%E4%B8%8E%E8%BE%B9%E9%A2%84%E6%B5%8B%E4%BB%BB%E5%8A%A1%E5%AE%9E%E8%B7%B5.md
class GCN_Net(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=4096, out_channels=128):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.conv1 = GCNConv(in_channels, hidden_channels)  # 第一层图卷积
        self.conv3 = GCNConv(hidden_channels, out_channels)  # 输出层图卷积
        logger.info(f"[GCN ] in_channels:{self.in_channels}, hidden_channels:{self.hidden_channels}, out_channels:{self.out_channels}")

    def encode(self, x, edge_index):  # 编码函数，用于节点特征的转换
        x = self.conv1(x, edge_index).relu()
        # x = self.conv2(x, edge_index).relu()
        return self.conv3(x, edge_index).relu()
  
    def decode(self, z, pos_edge_index, neg_edge_index): # z is hidden states of graph
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)
    
    def decode2(self, z, edge_index):
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)

    def decode_all(self, z):  # 解码所有节点对的函数
        prob_adj = z @ z.t()  # 计算节点特征的内积作为边的预测概率
        return (prob_adj > 0).nonzero(as_tuple=False).t()  # 返回概率大于0的边

def graph_train(df_split, dataset_name="DAVIS", epoch = 1000):
    setup_seed(10)
    train_data = df_split['train']
    valid_data = df_split['valid']
    test_data = df_split['test']
    raw_data =  df_split['raw']
    x, idx_dict = df_nodes(raw_data) # nodes info
    pos_edge_index = df_encode_edges(train_data,idx_dict)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN_Net(x.size(1)).to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    model.train()  # 设置模型为训练模式
    metrics = np.empty((epoch+1,5),dtype=float).tolist()
    losses = np.empty((epoch+1,1),dtype=float).tolist()
    for i in tqdm(range(epoch), "graph train"):
    # for i in range(epoch):
        optimizer.zero_grad()  # 清空梯度
        z = model.encode(x.to(device), pos_edge_index.to(device))  # 对训练数据进行编码
        neg_edge_index = df_train_neg_edges(train_data, raw_data, idx_dict)
        link_logits = model.decode(z, pos_edge_index.to(device), neg_edge_index.to(device))  # 解码边的存在概率
        link_labels = get_link_labels(pos_edge_index, neg_edge_index)
        loss = criterion(link_logits, link_labels.to(device))  # 计算损失
        loss.backward(retain_graph=True)  # 反向传播
        optimizer.step()  # 更新模型参数
        # logger.info(f"loss:{loss.item()}")
        MSE, CI, MAE, R2 = valid(model, valid_data, z, idx_dict, device)
        metrics[i] = ["valid",MSE, CI, MAE, R2]
        losses[i] = [loss.item()]
    MSE, CI, MAE, R2 = test(model, test_data, z, idx_dict, device)
    metrics[i+1] = ["valid",MSE, CI, MAE, R2]
    save_metrics(metrics, f"IPNet-Graph-{dataset_name}")
    save_loss(losses, f"IPNet-Graph-{dataset_name}")
    save_model(model, f"IPNet-Graph-{dataset_name}")
    return model
    
@torch.no_grad()
def valid(model, valid_data, z, idx_dict, device):
    pos_edge_index, neg_edge_index = df_decode_edges(valid_data, idx_dict)
    link_logits = model.decode(z, pos_edge_index.to(device), neg_edge_index.to(device))
    link_probs = link_logits.sigmoid()
    link_labels = get_link_labels(pos_edge_index, neg_edge_index)
    MSE, CI, MAE, R2 = evaluate(link_labels.detach().cpu().numpy(), link_probs.detach().cpu().numpy())
    # logger.info(f"[valid] MSE:{round(MSE,2)}, CI:{np.round(CI,2)}, MAE:{round(MAE, 2)}, R2:{round(R2, 2)}")
    return MSE, CI, MAE, R2

@torch.no_grad()
def test(model, test_data, z, idx_dict, device):
    model.eval()
    pos_edge_index, neg_edge_index = df_decode_edges(test_data, idx_dict)
    link_logits = model.decode(z, pos_edge_index.to(device), neg_edge_index.to(device))
    link_probs = link_logits.sigmoid()
    link_labels = get_link_labels(pos_edge_index, neg_edge_index)
    MSE, CI, MAE, R2 = evaluate(link_labels.detach().cpu().numpy(), link_probs.detach().cpu().numpy())
    logger.info(f"[GCN ] in_channels:{model.in_channels}, hidden_channels:{model.hidden_channels}, out_channels:{model.out_channels}")
    logger.info(f"[test] MSE:{round(MSE,2)}, CI:{round(CI,2)}, MAE:{round(MAE, 2)}, R2:{round(R2, 2)}")
    return MSE, CI, MAE, R2

if __name__ == '__main__':
    dataset_name = "KIBA"
    log_file = logger.add(f"{base_path}output/log/IPNet-Graph-{dataset_name}-{str(datetime.date.today())}.log")
    df_split = load(name = dataset_name)
    model = graph_train(df_split,dataset_name=dataset_name,epoch=2000)
    logger.remove(log_file)
    print('done')