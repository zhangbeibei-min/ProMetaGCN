import torch
import argparse
#
#
def get_citation_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_shot_0', type=int, default=10, help='How many shot during meta-train')#20训练集，阴性细胞因子，0类，每个元图的节点个数
    parser.add_argument('--train_shot_1', type=int, default=5, help='How many shot during meta-train')#20训练集，阳性细胞因子，1类，每个元图的节点个数
    parser.add_argument('--test_shot_0', type=int, default=60, help='How many shot during meta-test')#1 测试集，阴性细胞因子，0类，每个元图的节点个数
    parser.add_argument('--test_shot_1', type=int, default=35, help='How many shot during meta-test')#1 测试集，阳性细胞因子，1类，每个元图的节点个数
    parser.add_argument('--n_way', type=int, default=2, help='Classes want to be classify')#2分两类
    parser.add_argument('--step', type=int, default=20, help='How many times to random select node to test')#50，每次训练元图的个数
    parser.add_argument('--step1', type=int, default=5, help='How many times to random select node to test')#100 测试元图的个数
    parser.add_argument('--node_num', type=int, default=1207, help='Node number (dataset)')#数据集的节点总数
    parser.add_argument('--iteration', type=int, default=20, help='Iteration each cross_validation')#50,现在取10####迭代次数，一个iteration训练一次所有数据集
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disable CUDA training.')#False
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')#42
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train.')#200要训练的epoch的数目，现在取20
    parser.add_argument('--epochs1', type=int, default=50, help='Number of epochs to train.')  # 200要训练的epoch的数目，现在取20
    parser.add_argument('--lr', type=float, default=0.0006, help='Initial learning rate.')#0.0001
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')#5e-4
    parser.add_argument('--hidden', type=int, default=4, help='Number of hidden units.')#16
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')#0.5
    parser.add_argument('--dataset', type=str, default='PPI', help='Dataset to use.')#dataset
    parser.add_argument('--model', type=str, default='GCN', help='Model to use.')#GCN  GAT  GNNLSTM  GCNGRU
    parser.add_argument('--normalization', type=str, default='FirstOrderGCN', help='Normalization method for the adjacency matrix.')#FirstOrderGCN
    parser.add_argument('--degree', type=int, default=2, help='degree of the approximation.')#2  阶数
    parser.add_argument('--t', type=int, default=1, help='degree of the t-neighbor.')  # 特征矩阵的构建考虑层数

    args, _ = parser.parse_known_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    return args
