import math
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
from Utils import evaluate, setup_seed, dti_tokenizer, save_model, save_metrics, load_model
from Setting import base_path

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    
class Embedder(nn.Module):
    def __init__(self,num_embeddings= 256, embedding_dim=128, dropout=0.1):
        super(Embedder,self).__init__()
        self.embedding_dim = embedding_dim  #vocab.size ascii 256
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
    
    def forward(self, x):
        batch,seq_len = x.shape
        x = x.view(seq_len, batch)
        x = self.embedding(x) * (math.sqrt(self.embedding_dim)) # seq_len* batch*   embedding_dim
        emb = self.pos_encoder(x) # seq_len* batch* embedding_dim
        return emb

class TransEncoder(nn.Module):
    def __init__(self, embedding_dim=128, num_feature_out= 32,n_head=16, num_layers=3, dropout=0.1):
        super(TransEncoder,self).__init__()
        encoder_layers = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_head, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=num_layers)
        self.fc = nn.Sequential(
                        nn.Linear(embedding_dim, embedding_dim*2),
                        nn.Linear(embedding_dim*2, num_feature_out),
                        nn.Dropout(p=0.1)
                    )
        self.bn = nn.BatchNorm2d(1)
        logger.info(f"[Trans] embedding_dim:{embedding_dim}, num_feature_out:{num_feature_out}")

    def forward(self, emb):
        output = self.encoder(emb) # seq_len* batch* embedding_dim
        output = self.fc(output)
        seq_len, batch, embedding_dim = output.shape
        output = output.view(batch, 1, seq_len, embedding_dim) # batch * 1 *seq_len * embedding_dim
        output = self.bn(output)
        output = F.softmax(output)
        output = output.view(seq_len, batch, embedding_dim)
        # output = output.view(embedding_dim, seq_len, batch)
        output = output[seq_len-1] # batch* embedding_dim
        # output = torch.sum(output,dim=0)/seq_len
        # maxpool = nn.MaxPool2d((seq_len,1))
        # output = maxpool(output).view(embedding_dim)
        return output

class CNNEncoder(nn.Module):
    def __init__(self, embedding_dim=128, num_feature_out= 32):
        super(CNNEncoder,self).__init__()
        self.out_channels = int(num_feature_out/4)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.out_channels, kernel_size=(2,embedding_dim))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=self.out_channels, kernel_size=(3,embedding_dim))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=self.out_channels, kernel_size=(4,embedding_dim))
        self.conv4 = nn.Conv2d(in_channels=1, out_channels=self.out_channels, kernel_size=(5,embedding_dim))
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(self.out_channels)
        # self.fc = nn.Sequential(
        #                 nn.Linear(self.out_channels*4, 512),
        #                 nn.Linear(512, 256),
        #                 nn.Linear(256, 64),
        #                 nn.Dropout(p=0.1)
        #             )
        logger.info(f"[CNN] embedding_dim:{embedding_dim},num_feature_out:{num_feature_out}")
    
    def forward(self, emb):
        seq_len, batch, embedding_dim = emb.shape
        emb = emb.reshape(batch,1, seq_len, embedding_dim)# batch, channel_in, seq_len, embedding_dim
        c1 = self.relu(self.bn(self.conv1(emb))) # batch, channel_out, seq_len, embedding_dim
        maxpool1 = nn.MaxPool2d((seq_len-1, 1))
        c1 = maxpool1(c1) # batch, channel_out, high, width
        c2 = self.relu(self.bn(self.conv2(emb)))
        maxpool2 = nn.MaxPool2d((seq_len-2, 1))
        c2 = maxpool2(c2)# batch, channel_out, high, width
        c3 = self.relu(self.bn(self.conv3(emb)))
        maxpool3 = nn.MaxPool2d((seq_len-3, 1))
        c3 = maxpool3(c3)# batch, channel_out, high, width
        c4 = self.relu(self.bn(self.conv4(emb)))
        maxpool4 = nn.MaxPool2d((seq_len-4, 1))
        c4 = maxpool4(c4)# batch, channel_out, high, width
        output = torch.cat([c1, c2, c3, c4], dim=1) # batch, num_feature_out
        batch,num_feature_out, hight, widht = output.shape
        output = output.view(batch, num_feature_out)
        # output = self.fc(output) # batch, num_feature_out
        # output = F.softmax(output)
        return output

class DTISeqPredictNet(nn.Module):
    def __init__(self, embedding_dim= 128, num_d_feature=32, num_t_feature=512):
        super(DTISeqPredictNet,self).__init__()
        self.num_d_feature = num_d_feature
        self.num_t_feature = num_t_feature
        self.embedding_dim = embedding_dim
        self.embedder = Embedder(embedding_dim=self.embedding_dim)
        self.drug_cnn_encoder = CNNEncoder(embedding_dim=embedding_dim, num_feature_out = num_d_feature)  # local
        self.drug_trans_encoder = TransEncoder(embedding_dim=embedding_dim, num_feature_out=num_d_feature) # local
        self.target_cnn_encoder = CNNEncoder(embedding_dim=embedding_dim,num_feature_out = num_t_feature) # global
        self.target_trans_encoder = TransEncoder(embedding_dim=embedding_dim, num_feature_out = num_t_feature) # global
        num_feature_in = self.num_d_feature+self.num_t_feature
        self.predict_unit = nn.Sequential(
            nn.Linear(num_feature_in*2, num_feature_in),
            nn.Linear(num_feature_in, 64),
            nn.Linear(64, 1),
            nn.Dropout(p=0.1)
        )
        self.sigmoid = nn.Sigmoid()
        logger.info(f"[SeqNet] num_d_feature:{num_d_feature}, num_t_feature:{num_t_feature}")

    def forward(self, drugs, targets):
        drug_emb= self.embedder(drugs)
        target_emb= self.embedder(targets)
        drug_feature = self.drug_cnn_encoder(drug_emb) # local feature
        drug_feature2 = self.drug_trans_encoder(drug_emb) # global feature
        target_feature = self.target_cnn_encoder(target_emb) # local feature
        target_feature2 = self.target_trans_encoder(target_emb) # global feature
        # batch * num_feature
        combine_emb = torch.cat((drug_feature, target_feature,drug_feature2, target_feature2),dim =1) # local + global
        output = self.predict_unit(combine_emb)
        # output = self.sigmoid(output)
        batch,_= output.shape # batch * 1
        output = output.view(batch)
        return output

def negative_sampling_train_dataset(df_split, batch_size=8):
    train_data = df_split['train']
    valid_data = df_split['valid']
    test_data = df_split['test']
    raw_data =  df_split['raw']
    # drop valid samples and test samples form raw samples
    rest_data = raw_data[~raw_data.isin(valid_data)].dropna() 
    rest_data = rest_data[~rest_data.isin(test_data)].dropna() 
    # keep train positive samples and update negative samples
    train_pos_data = train_data[train_data.Y == pos_label]
    neg_data = rest_data[rest_data.Y == neg_label]
    new_train_neg_data = neg_data.sample(n=len(train_pos_data),random_state=10)
    new_train_data = pd.concat([train_pos_data, new_train_neg_data])
    # dataset 
    dti_dataset = MyDataset(new_train_data)
    dti_data = DataLoader(dti_dataset,batch_size=batch_size, shuffle=True, drop_last=True)
    return dti_data

def seq_train(df_split, dataset_name="DAVIS", epoch = 20, batch_size=8,
              num_d_feature = 32, num_t_feature = 512, dlen_limit = 80, tlen_limit = 1500):
    setup_seed(10)
    # num_d_feature = 32
    # num_t_feature = 512
    # dlen_limit = 80
    # tlen_limit = 1500
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # df_split = load(name = dataset_name)
    train_data = df_split['train']
    valid_data = df_split['valid']
    test_data = df_split['test']
    raw_data =  df_split['raw']
    seq_net = DTISeqPredictNet(num_d_feature=num_d_feature, num_t_feature=num_t_feature).to(device)
    optimizer = torch.optim.Adam(params=seq_net.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    seq_net.train()  # 设置模型为训练模式
    dti_dataset = MyDataset(train_data)
    dti_data = DataLoader(dti_dataset,batch_size=batch_size, shuffle=True, drop_last=True)
    metrics = []
    for e in range(epoch):
        dti_data = negative_sampling_train_dataset(df_split, batch_size) 
        # for dti in dti_data:
        for dti in tqdm(dti_data, f"epoch{e}/{epoch}"):
            optimizer.zero_grad() 
            drugs = dti[1]
            targets = dti[3]
            labels = dti[4]
            # logits = torch.empty(batch_size, dtype=float,device=device)
            # for i in range(batch_size):
            drugs = dti_tokenizer(drugs,limit=dlen_limit)
            targets = dti_tokenizer(targets,limit=tlen_limit)
            logits = seq_net(drugs.to(device), targets.to(device))
            loss = criterion(logits, torch.tensor(labels, dtype=float, device=device)) # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
            # logger.info(f"loss:{loss.item()}")
        MSE, CI, MAE, R2 = valid(seq_net, valid_data, dlen_limit, tlen_limit, batch_size,device)
        metrics.append(["valid",MSE, CI, MAE, R2])
    MSE, CI, MAE, R2 = test(seq_net, test_data, dlen_limit, tlen_limit, batch_size,device)
    metrics.append(["test",MSE, CI, MAE, R2])
    save_metrics(metrics, f"IPNet-Seq-{dataset_name}")
    save_model(model, f"IPNet-Seq-{dataset_name}")
    return seq_net
        
@torch.no_grad()
def valid(model, valid_data,dlen_limit, tlen_limit, batch_size,device):
    dti_dataset = MyDataset(valid_data)
    dti_data = DataLoader(dti_dataset,batch_size=batch_size, shuffle=True, drop_last=True)
    metrics = []
    for dti in tqdm(dti_data, "valid"):
        drugs = dti[1]
        targets = dti[3]
        labels = dti[4]
        drugs = dti_tokenizer(drugs,limit=dlen_limit)
        targets = dti_tokenizer(targets,limit=tlen_limit)
        logits = model(drugs.to(device), targets.to(device))
        MSE, CI, MAE, R2 = evaluate(np.array(labels), logits.detach().cpu().numpy())
        metrics.append([MSE, CI, MAE, R2])
    MSE, CI, MAE, R2 = np.array(metrics).mean(axis=0)
    logger.info(f"[valid] MSE:{round(MSE,2)}, CI:{round(CI,2)}, MAE:{round(MAE, 2)}, R2:{round(R2, 2)}")
    return MSE, CI, MAE, R2

@torch.no_grad()
def test(model, test_data, dlen_limit, tlen_limit, batch_size,device):
    model.eval()
    dti_dataset = MyDataset(test_data)
    dti_data = DataLoader(dti_dataset,batch_size=batch_size, shuffle=True, drop_last=True)
    metrics = []
    for dti in tqdm(dti_data, "test"):
        drugs = dti[1]
        targets = dti[3]
        labels = dti[4]
        drugs = dti_tokenizer(drugs,limit=dlen_limit)
        targets = dti_tokenizer(targets,limit=tlen_limit)
        logits = model(drugs.to(device), targets.to(device))
        MSE, CI, MAE, R2 = evaluate(np.array(labels), logits.detach().cpu().numpy())
        metrics.append([MSE, CI, MAE, R2])
    MSE, CI, MAE, R2 = np.array(metrics).mean(axis=0)
    logger.info(f"[test] MSE:{round(MSE,2)}, CI:{round(CI,2)}, MAE:{round(MAE, 2)}, R2:{round(R2, 2)}")
    return MSE, CI, MAE, R2

if __name__ == '__main__':
    dataset_name = "DAVIS"
    df_split = load(name = dataset_name)
    log_file = logger.add(f"{base_path}output/log/IPNet-Seq-{dataset_name}-{str(datetime.date.today())}.log")
    model = seq_train(df_split, dataset_name=dataset_name)
    logger.remove(log_file)
