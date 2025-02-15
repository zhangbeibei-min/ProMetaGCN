##*******************************1.特征矩阵-单位向量******************************
# -*- coding: utf-8 -*-
import numpy as np
import scipy.sparse as sp
import torch
import sys
import pickle as pkl
import networkx as nx
from normalization import fetch_normalization, row_normalize
from sklearn.metrics import f1_score
import pandas as pd

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def f1(output, labels):
    preds = output.max(1)[1]
    preds = preds.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    micro = f1_score(labels, preds, average='micro')
    return micro

def set_seed(seed, cuda):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def preprocess_citation(adj, features, normalization="FirstOrderGCN"):
    adj_normalizer = fetch_normalization(normalization)
    adj = adj_normalizer(adj)
    features = row_normalize(features)
    return adj, features

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def load_citation(dataset_str='PPI', normalization='FirstOrderGCN', cuda=True):
    #计算adj的 # 加载邻接矩阵
    adjlist_path = open(r'.\Meta-GCN\Data\string_interactions.tsv')
    adjlist_file = pd.read_table(adjlist_path, sep="\t")
    adjlist = adjlist_file[["node1_string_id", "node2_string_id", "combined_score"]]
    protein_map = pd.read_csv(open(r'.\Meta-GCN\Data\All_cytokine_node_mapping.csv'), index_col=0)
    adjlist_node1 = protein_map.loc[adjlist["node1_string_id"]]
    adjlist_node2 = protein_map.loc[adjlist["node2_string_id"]]
    adjlist["node1"] = adjlist_node1["nodes"].values
    adjlist["node2"] = adjlist_node2["nodes"].values
    # # 生成邻接矩阵
    adj_matrix1 = sp.csr_matrix((adjlist['combined_score'], (adjlist['node1'], adjlist['node2'])),shape=(len(protein_map), len(protein_map)))
    adj_matrix2 = sp.triu(adj_matrix1) + sp.tril(adj_matrix1).T
    data = []
    row = []
    col = []
    for i in range(len(protein_map)):
        row .append(i)# 行指标
    for i in range(len(protein_map)):
        col.append(i)# 列指标
    for i in range(len(protein_map)):
        data.append(1) # 在行指标列指标下的数字
    adj_unit = sp.csr_matrix((data, (row, col)), shape=(len(protein_map), len(protein_map)))
    #adj_unit = sp.eye(len(protein_map))
    adj = adj_unit + adj_matrix2 + adj_matrix2.T

    # #1.特征的单位矩阵
    features = sp.identity(len(protein_map))
    adj, features = preprocess_citation(adj, features, normalization)
    features = torch.FloatTensor(np.array(features.todense())).float()
    # 加载标签
    labels_df = pd.read_csv(r'.\Meta-GCN\Data\LabelClass\All_cytokine_class.csv', index_col=0)
    labels = torch.LongTensor(labels_df.iloc[:, 0].values)
    #print(labels)
    #labels = torch.max(labels, dim=1)[1]#有关labels的这些程序只是用来提取labels的

    # 将邻接矩阵转换为稀疏张量
    adj = sparse_mx_to_torch_sparse_tensor(adj).float()
    #print(adj)
    if cuda:
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()

    return adj, features, labels

