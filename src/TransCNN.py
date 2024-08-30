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
from Utils import evaluate, setup_seed, dti_tokenizer, save_model, save_metrics

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

class TransEncoder(nn.Module):
    def __init__(self,num_embeddings= 256, embedding_dim=32, n_head=16, num_layers=6, dropout=0.1):
        super(TransEncoder,self).__init__()
        self.embedding_dim = embedding_dim  #vocab.size ascii 256
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_head, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layers, num_layers=num_layers)
    def forward(self, x):
        x = x.view(-1,1)
        x = self.embedding(x) * (math.sqrt(self.embedding_dim))
        x = self.pos_encoder(x) # seq_len* batch* embedding_dim
        output = self.encoder(x) # seq_len* batch* embedding_dim
        seq_len, batch, embedding_dim = output.shape
        # output = output.view(embedding_dim, seq_len, batch)
        output = output.view(seq_len, embedding_dim)
        output = output[seq_len-1]
        # maxpool = nn.MaxPool2d((seq_len,1))
        # output = maxpool(output).view(embedding_dim)
        # output = F.softmax(output)
        return output

class CNNEncoder(nn.Module):
    def __init__(self, num_embeddings= 256, embedding_dim=32, num_feature_out= 32):
        super(CNNEncoder,self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings #vocab.size ascii 256
        self.out_channels = int(num_feature_out/4)
        self.embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=self.out_channels, kernel_size=(2,embedding_dim))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=self.out_channels, kernel_size=(3,embedding_dim))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=self.out_channels, kernel_size=(4,embedding_dim))
        self.conv4 = nn.Conv2d(in_channels=1, out_channels=self.out_channels, kernel_size=(5,embedding_dim))
        self.relu = nn.ReLU()
        # self.fc = nn.Sequential(
        #                 nn.Linear(self.out_channels*4, 512),
        #                 nn.Linear(512, 256),
        #                 nn.Linear(256, 64),
        #                 nn.Dropout(p=0.1)
        #             )
    
    def forward(self, x):
        x = x.view(-1,1) 
        x = self.embedding(x) # seq_len, batch, embedding_dim
        seq_len, batch, embedding_dim = x.shape
        x = x.reshape(batch, seq_len, embedding_dim)# batch, seq_len, embedding_dim
        c1 = self.relu(self.conv1(x)) # num_kernel, seq_len, batch
        maxpool1 = nn.MaxPool2d((seq_len-1, 1))
        c1 = maxpool1(c1)
        c2 = self.relu(self.conv2(x))
        maxpool2 = nn.MaxPool2d((seq_len-2, 1))
        c2 = maxpool2(c2)# channel, width, batch
        c3 = self.relu(self.conv3(x))
        maxpool3 = nn.MaxPool2d((seq_len-3, 1))
        c3 = maxpool3(c3)# channel, width, batch
        c4 = self.relu(self.conv4(x))
        maxpool4 = nn.MaxPool2d((seq_len-4, 1))
        c4 = maxpool4(c4)# channel, width, batch
        output = torch.cat([c1, c2, c3, c4]).view(batch,-1) # batch, feature_num
        # output = self.fc(output) # batch, feature_num
        # output = F.softmax(output)
        return output.view(-1)

class DTISeqPredictNet(nn.Module):
    def __init__(self):
        super(DTISeqPredictNet,self).__init__()
        self.num_feature_out = 32
        self.cnn_encoder = CNNEncoder(num_feature_out = 32)
        self.trans_encoder = TransEncoder()
        self.predict_unit = nn.Sequential(
            nn.Linear(self.num_feature_out*4, 256),
            nn.Linear(256, 64),
            nn.Linear(64, 1),
            nn.Dropout(p=0.1)
        )

    def forward(self, drug, target):
        drug_emb1 = self.cnn_encoder(drug) # local feature
        drug_emb2 = self.trans_encoder(drug) # global feature
        target_emb1 = self.cnn_encoder(target) # local feature
        target_emb2 = self.trans_encoder(target) # global feature
        combine_emb = torch.cat((drug_emb1, drug_emb2, target_emb1, target_emb2))
        output = self.predict_unit(combine_emb)
        # output = F.softmax(output)
        # output = F.sigmoid(output)
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

def seq_train(df_split, dataset_name="DAVIS", epoch = 20, batch_size=8):
    setup_seed(10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # df_split = load(name = dataset_name)
    train_data = df_split['train']
    valid_data = df_split['valid']
    test_data = df_split['test']
    raw_data =  df_split['raw']
    seq_net = DTISeqPredictNet().to(device)
    optimizer = torch.optim.Adam(params=seq_net.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()
    seq_net.train()  # 设置模型为训练模式
    dti_dataset = MyDataset(train_data)
    dti_data = DataLoader(dti_dataset,batch_size=batch_size, shuffle=True, drop_last=True)
    metrics = np.empty((epoch+1,5),dtype=float).tolist()
    for e in range(epoch):
        # dti_data = negative_sampling_train_dataset(df_split, batch_size) 
        loss_val = 0.0
        count = 1
        for dti in dti_data:
        # for dti in tqdm(dti_data, f"epoch{e}"):
            optimizer.zero_grad()  # 清空梯度
            drugs = dti[1]
            targets = dti[3]
            labels = dti[4]
            logits = torch.empty(batch_size, dtype=float,device=device)
            for i in range(batch_size):
                drug = dti_tokenizer(drugs[i])
                target = dti_tokenizer(targets[i])
                logit = seq_net(drug.to(device), target.to(device))
                logits[i] = logit
            loss = criterion(logits, torch.tensor(labels, dtype=float, device=device)) # 计算损失
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
            loss_val = (loss_val + loss.item())
            count = count + 1
            logger.info(f"loss:{loss_val/count}")
        MSE, CI, MAE, R2 = valid(seq_net, valid_data, device)
        metrics[e] = ["valid",MSE, CI, MAE, R2]
    MSE, CI, MAE, R2 = test(seq_net, test_data, device)
    metrics[e+1] = ["test",MSE, CI, MAE, R2]
    save_metrics(metrics, f"IPNet-Seq-{dataset_name}")
    return seq_net
        
@torch.no_grad()
def valid(model, valid_data, device):
    dti_data = valid_data[["Drug", "Target"]].values
    labels = valid_data["Y"].values
    logits = []
    for drug, target in dti_data:
        drug = dti_tokenizer(drug)
        target = dti_tokenizer(target)
        logit = model(drug.to(device), target.to(device))
        logits.append(logit.item())
    MSE, CI, MAE, R2 = evaluate(np.array(labels), np.array(logits))
    logger.info(f"[valid] MSE:{round(MSE,2)}, CI:{round(CI,2)}, MAE:{round(MAE, 2)}, R2:{round(R2, 2)}")
    return MSE, CI, MAE, R2

@torch.no_grad()
def test(model, test_data, device):
    model.eval()
    dti_data = test_data[["Drug", "Target"]].values
    labels = test_data["Y"].values
    logits = []
    for drug, target in dti_data:
        drug = dti_tokenizer(drug)
        target = dti_tokenizer(target)
        logit = model(drug.to(device), target.to(device))
        logits.append(logit.item())
    MSE, CI, MAE, R2 = evaluate(np.array(labels), np.array(logits))
    logger.info(f"[test] MSE:{round(MSE,2)}, CI:{round(CI,2)}, MAE:{round(MAE, 2)}, R2:{round(R2, 2)}")
    return MSE, CI, MAE, R2
        
if __name__ == '__main__':
    # drug = "COc1cc(Nc2ncc(F)c(Nc3ccc4c(n3)NC(=O)C(C)(C)O4)n2)cc(OC)c1OC.O=S(=O)(O)c1ccccc1"
    # target = "MSTASAASSSSSSSAGEMIEAPSQVLNFEEIDYKEIEVEEVVGRGAFGVVCKAKWRAKDVAIKQIESESERKAFIVELRQLSRVNHPNIVKLYGACLNPVCLVMEYAEGGSLYNVLHGAEPLPYYTAAHAMSWCLQCSQGVAYLHSMQPKALIHRDLKPPNLLLVAGGTVLKICDFGTACDIQTHMTNNKGSAAWMAPEVFEGSNYSEKCDVFSWGIILWEVITRRKPFDEIGGPAFRIMWAVHNGTRPPLIKNLPKPIESLMTRCWSKDPSQRPSMEEIVKIMTHLMRYFPGADEPLQYPCQYSDEGQSNSATSTGSFMDIASTNTSNKSDTNMEQVPATNDTIKRLESKLLKNQAKQQSESGRLSLGASRGSSVESLPPTSEGKRMSADMSEIEARIAATTAYSKPKRGHRKTASFGNILDVPEIVISGNGQPRRRSIQDLTVTGTEPGQVSSRSSSPSVRMITTSGPTSEKPTRSHPWTPDDSTDTNGSDNSIPMAYLTLDHQLQPLAPCPNSKESMAVFEQHCKMAQEYMKVQTEIALLLQRKQELVAELDQDEKDQQNTSRLVQEHKKLLDENKSLSTYYQQCKKQLEVIRSQQQKRQGTS"
    # drug = dti_tokenizer(drug)
    # print(drug)
    # target = dti_tokenizer(target)
    # print(target)
    # drug_encoder = TransEncoder()
    # drug_emb = drug_encoder(drug)
    # print(drug_emb)
    # target_encoder = CNNEncoder()
    # target_feature = target_encoder(target)
    # print(target_feature)
    # seq_net = DTISeqPredictNet()
    # result = seq_net(drug, target)
    # print(result)
    dataset_name = "DAVIS"
    df_split = load(name = dataset_name)
    log_file = logger.add(f"/home/yang/sda/github/IPNET/output/log/IPNet-Seq-{dataset_name}-{str(datetime.date.today())}.log")
    model = seq_train(df_split, dataset_name=dataset_name)
    save_model(model, f"IPNet-Seq-{dataset_name}")
    logger.remove(log_file)