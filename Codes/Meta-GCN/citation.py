# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
import torch.optim as optim
from itertools import combinations
import random
from cytokine_number_name import xuhao_mingcheng
from LabelClass import biaoqianfenlei
import scipy.stats as stats
from args import get_citation_args
from utils import load_citation, set_seed
from models import get_model
from metrics import accuracy,accuracy1
import pandas as pd
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import networkx as nx

pd.set_option('display.max_columns', 100000000)
pd.set_option('display.width', 100000000)
pd.set_option('display.max_colwidth', 100000000)
pd.set_option('display.max_rows', 100000000)
#Train Model
def train_regression(model, train_features, train_labels, idx_train, epochs, weight_decay, lr, adj):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loss = []  # 存储训练损失
    criterion = nn.CrossEntropyLoss()#label_smoothing=0.1
    for epoch in range(epochs):#只在训练里面有epoch
        model.train()
        optimizer.zero_grad()#把梯度置零，也就是把loss关于weight的导数变成0,对于每个batch大都执行了这样的操作
        output = model(train_features, adj)
        loss_train = F.cross_entropy(output[idx_train], train_labels[idx_train])
        #print('Step:', epoch, '\tMeta_Training_Loss:', loss_train)
        loss_train.backward()#调用backward()函数之前都要将梯度清零，反向传播求解梯度
        optimizer.step()#更新权重参数
        train_loss.append(loss_train.item())  # 将当前epoch的训练损失添加到列表中
    # # 绘制loss-epoch图
    # plt.plot(range(epochs), train_loss)
    # plt.xlabel('Epochs0')
    # plt.ylabel('Loss0')
    # plt.title('Training Loss')
    #plt.show()
    return model, loss_train, optimizer

def train_regression1(model, train_features, train_labels, idx_train, epochs1, weight_decay, lr, adj):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs1):#只在训练里面有epoch
        model.train()
        optimizer.zero_grad()#把梯度置零，也就是把loss关于weight的导数变成0,对于每个batch大都执行了这样的操作
        output = model(train_features, adj)
        loss_train = F.cross_entropy(output[idx_train], train_labels[idx_train])
        #print('Step:', epoch, '\tMeta_Training_Loss:', loss_train)
        loss_train.backward()#调用backward()函数之前都要将梯度清零，反向传播求解梯度
        optimizer.step()#更新权重参数
    return model, loss_train, optimizer
#Test Model
def test_regression(model, test_features, test_labels, idx_test, adj):
    model.eval()
    output = model(test_features, adj)
    loss_train = F.cross_entropy(output[idx_test], test_labels[idx_test])
    acc_ = accuracy(output[idx_test], test_labels[idx_test])
    return acc_, loss_train

def test_regression1(model, test_features, test_labels, idx_test, adj):
    model.eval()
    output = model(test_features, adj)
    pred_q = F.softmax(output, dim=1).detach().tolist()
    acc_ = accuracy1(output[idx_test], test_labels[idx_test])

    #print(acc_)
    return pred_q
#Clear Array
def reset_array():
    class1_train = []
    class2_train = []
    class1_test = []
    class2_test = []
    train_idx = []
    test_idx = []

def main():
    args = get_citation_args()          #get args
    n_way = args.n_way                  #how many classes
    train_shot_0 = args.train_shot_0        #train-shot
    train_shot_1 = args.train_shot_1
    test_shot_0 = args.test_shot_0          #test-shot
    test_shot_1 = args.test_shot_1
    step = args.step
    step1 = args.step1
    node_num = args.node_num
    iteration = args.iteration#迭代每个交叉验证

    accuracy_meta_test = []
    total_accuracy_meta_test = []

    set_seed(args.seed, args.cuda)
    if args.dataset == 'PPI':
        node_num = 1207
        class_label = [0, 1]
        combination = list(combinations(class_label, 2))#由两个元素组成的所有可能的组合，并将结果以列表的形式存储在 变量中。
    wai = 0
    aa = pd.read_csv(open(r'.\Meta-GCN\Data\cytokine_number_name.csv'), header=None, index_col=False)#序号和名称的对应.csv
    dic1 = {}
    for kk in aa[1]:
        dic1[kk] = 0
    dic2 = {}
    for kk1 in aa[1]:
        dic2[kk1] = 0
    list1 = []
    times=10##训练多次
    for mmh in range(times):###循环次数epochs  save100文件夹名称
        path_0 = r'.\Meta-GCN\Data\PredictResult\save' + str(wai)
        if os.path.exists(path_0):
            shutil.rmtree(path_0)
        os.makedirs(path_0)
        #随机选取test_shot_0个负标签(所有的节点减掉正标签的OthersLabel)
        with open(r'.\Meta-GCN\Data\OthersLabel.txt','r') as f:
            ss = f.readlines()
            jieguo = random.sample(ss,test_shot_0)#打开一个名为'OthersLabel.txt'的文件，然后从中随机抽取test_shot_0行内容，并将抽取的内容存储在变量jieguo中。
        #随机选取的test_shot_0（个数选取）个负标签写入，测试机的负标签节点个数
        with open(r'.\Meta-GCN\Data\label0.txt', 'w') as f:
            for iii in jieguo:
                #f.write(iii)#打开一个名为'label0.txt'的文件，以写入模式打开，然后将变量jieguo中的内容逐行写入到该文件中。
                f.write(iii.strip() + '\n')# 添加换行符来保证每个标签单独占据一行

        shu = 0
        mmm = 0
        while mmm==0:
            xuhao_mingcheng()
            biaoqianfenlei()
            #adj, features, labels = load_citation(args.dataset, args.normalization, args.cuda,args.t)
            adj, features, labels = load_citation(args.dataset, args.normalization, args.cuda)
            test_label = list(combination[0])
            train_label = [n for n in class_label if n not in test_label]
            model = get_model(args.model, features.size(1), n_way, args.hidden, args.dropout, args.cuda).cuda()
            #create model
            path = path_0+'\\'+str(shu)
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)

            ###未求均值
            train_loss0 = []
            train_loss1 = []
            val_loss = []
            train_loss0_mean = []
            val_loss_mean = []
            val_acc_mean = []

####元学习训练
            for j in range(iteration):
                labels_local = labels.clone().detach()
                select_class = [0, 1]
                print('Times {} ITERATION {} '.format(mmh+1, j+1))
                # 将节点按类别分组
                class1_idx = [k for k in range(node_num) if labels_local[k] == select_class[0]]
                class2_idx = [k for k in range(node_num) if labels_local[k] == select_class[1]]

                train_loss0_iter = []  # 用于存储当前迭代的训练损失
                val_loss_iter = []  # 用于存储当前迭代的验证损失
                val_acc_iter = []

                for m in range(step):###元图的训练（循环），一次训练step个
                    # 从D1中选择类别为C1的节点作为训练集
                    class1_train = random.sample(class1_idx,train_shot_0)
                    # 从D1中选择类别为C2的节点作为训练集
                    class2_train = random.sample(class2_idx,train_shot_1)
                    # 从D1中剩余的类别为C1的节点作为测试集
                    class1_test = [n1 for n1 in class1_idx if n1 not in class1_train]
                    # 从D1中剩余的类别为C2的节点作为测试集
                    class2_test = [n2 for n2 in class2_idx if n2 not in class2_train]
                    # 合并训练集和测试集索引
                    train_idx = class1_train + class2_train
                    random.shuffle(train_idx)
                    test_idx = class1_test + class2_test
                    random.shuffle(test_idx)
                    # 训练模型
                    model,loss_train0,optimizer= train_regression(model, features, labels_local, train_idx, args.epochs, args.weight_decay, args.lr, adj)
                    train_loss0_iter.append(loss_train0.item())
                    # 测试模型
                    acc_query,loss_val = test_regression(model, features, labels_local, test_idx, adj)#验证集的准确率
                    val_loss_iter.append(loss_val.item())
                    val_acc_iter.append(acc_query.item())
                    print('Step:', m + 1, '\tMeta_Training_Loss0:', loss_train0.item(), '\tMeta_Test_Loss0:', loss_val.item(), '\tMeta_Test_acc0:',acc_query.item())
                    reset_array()
                # 计算当前迭代的训练损失和验证损失的均值并存储
                train_loss0_mean.append(np.mean(train_loss0_iter))
                val_loss_mean.append(np.mean(val_loss_iter))
                val_acc_mean.append(np.mean(val_acc_iter))

            pd.DataFrame(val_loss_iter).to_csv(path+'\\'+'val_loss.csv',header=None)
            pd.DataFrame(train_loss0_iter).to_csv(path+'\\'+'train_loss0.csv', header=None)
            pd.DataFrame(val_acc_iter).to_csv(path + '\\' + 'val_acc.csv', header=None)
            torch.save(model.state_dict(), path+'\\'+'model.pkl')#save model as 'model.pkl'

            # 画图
            plt.plot(range(len(train_loss0_mean)), train_loss0_mean, label='Training Loss')
           #plt.plot(range(len(val_acc_mean)), val_acc_mean, label='Validation acc')
            plt.plot(range(len(val_loss_mean)), val_loss_mean, label='Validation Loss')
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.legend()
            #plt.title('Training and Validation Loss vs. Iteration')
            #plt.title('Validation Loss vs Acc')
            plt.savefig(path + '\\' + 'Validation Loss and Acc.svg')
            plt.savefig(path + '\\' + 'Validation Loss and Acc.pdf')
            plt.show()

            labels_local = labels.clone().detach()
            select_class = select_class = [0, 1]
            print('Test_Label {}: '.format( select_class))
            #print('EPOCH {} Test_Label {}: '.format(i+1, select_class))
            class1_idx1 = []
            class2_idx1 = []
            reset_array()

            s = []
            lb = list(range(node_num))
            for k in range(node_num):###所有的数据集
                if(labels_local[k] == select_class[0]):
                    class1_idx1.append(k)
                    #labels_local[k] = 0
                elif(labels_local[k] == select_class[1]):
                    class2_idx1.append(k)
                    #labels_local[k] = 1
            class_idx = class1_idx1 + class2_idx1
            class_idx_test = np.sort(class_idx)
            for m in lb:
                if m not in class_idx_test:
                    s.append(m)  # 不包括取的0和1标签的测试集

            for m in range(step1):###正标签训练测试（只有一个step元图）
                class1_train = random.sample(class1_idx1, test_shot_0-2)
                class2_train = random.sample(class2_idx1, test_shot_1)
                # class1_test = [n1 for n1 in class1_idx if n1 not in class1_train]
                # class2_test = [n2 for n2 in class2_idx if n2 not in class2_train]
                train_idx = class1_train + class2_train
                random.shuffle(train_idx)
                #test_idx = class1_test + class2_test
                test_idx = s
                random.shuffle(test_idx)
                model_meta_trained = get_model(args.model, features.size(1), n_way, args.hidden, args.dropout, args.cuda).cuda()
                model_meta_trained.load_state_dict(torch.load(path+'\\'+'model.pkl'))
                #re-load model 'model.pkl'
                model_meta_trained,loss_train1,optimizer = train_regression1(model_meta_trained, features, labels_local, train_idx, args.epochs1, args.weight_decay, args.lr, adj)
                train_loss1.append(loss_train1.item())
                pred_q = test_regression1(model_meta_trained, features, labels_local, test_idx, adj)#测试集的准确率
                #pred_q, acc_ = test_regression1(model_meta_trained, features, labels_local, test_idx, adj)  # 测试集的准确率
                print('Step:', m+1, '\tMeta_Training_Loss1:', loss_train1.item())##m，每次迭代都进行m个元图训练
                #print('Step:', m + 1, '\tMeta_Training_Loss1:', loss_train1.item(), '\tMeta_Test_acc0:', acc_.item())
                reset_array()

####元学习结束
            # 将预测结果保存到CSV文件中
            #pd.DataFrame(train_loss1).to_csv(path+'\\'+'train_loss1.csv', header=None)
            pd.DataFrame(pred_q).to_csv(path+'\\'+'pred_q.csv', header=None)
            # 读取标签编码信息
            label_num = pd.read_csv(open(r'.\Meta-GCN\Data\LabelClass\Cytokine_name_code_Idnumber.csv'), header=None,index_col=False)##细胞因子名称、编码、序号.csv
            j = label_num.loc[lb]
            j.columns = ['1','2','3']
            j.sort_values('2', inplace=True, ascending=True)
            # 将结果序号和名称保存到CSV文件中
            pd.DataFrame(j).to_csv(path+'\\'+'result_number_name.csv', header=None)#结果对应的序号和名称.csv
            # 读取预测结果，并与标签信息合并并按概率排序
            jg = pd.read_csv(open(path+'\\'+'pred_q.csv'), header=None,index_col=False)
            jg = jg.iloc[:,2]
            df = pd.concat([j, jg], axis=1)
            df.columns = ['1','2','3','4']
            df.sort_values('4', inplace=True,ascending=False)##按照概率排序
            df.index = range(len(j))
            pd.DataFrame(df).to_csv(path + '\\' + 'result_number_name.csv', header=None)
            # 获取最后100个细胞因子的编码并保存到文件中
            #dd = df['1'].iloc[len(j)-100-test_shot_0:len(j)-100].tolist()
            dd = df['1'].iloc[-test_shot_0:].tolist()
            with open(r'.\Meta-GCN\Data\label0.txt', 'w') as f:###label负.txt
                for iii in dd:
                    f.write(iii+'\n')
            # 读取结果对应的序号和名称
            a = pd.read_csv(open(path + '\\' + 'result_number_name.csv'), header=None, index_col=False)#结果对应的序号和名称.csv
            if shu==1:
                mmm=1
            print('Label',mmm)
            list1.append(shu)
            shu = shu+1
        #将最后一次的结果秩的和进行输入
        for ooo, ppp in enumerate(a[3]):
            dic1[ppp] = dic1[ppp] + ooo
        for hhh in val_loss:
            if hhh < 0.02:
                for ooo1, ppp1 in enumerate(a[3]):
                    dic2[ppp1] = dic2[ppp1] + ooo1
                break
        wai = wai+1
    pd.DataFrame.from_dict(dic1,orient='index').to_csv(r'.\Meta-GCN\Data\PredictResult\Result\结果(全部).csv', header=None)
    pd.DataFrame.from_dict(dic2,orient='index').to_csv(r'.\Meta-GCN\Data\PredictResult\Result\结果(收敛).csv', header=None)
if __name__ == '__main__':
    main()



















# # 创建一个空的无向图
# G = nx.Graph()
#
# # 添加类别1的训练样本
# class1_train = ['class1_train_' + str(i) for i in range(test_shot_0 - 2)]
# # 添加类别2的训练样本
# class2_train = ['class2_train_' + str(i) for i in range(test_shot_1)]
#
# # 将训练样本添加到图中
# G.add_nodes_from(class1_train, color='blue')
# G.add_nodes_from(class2_train, color='red')
#
# # 绘制图
# plt.figure(figsize=(8, 6))
#
# # 提取节点颜色
# node_colors = [G.nodes[node]['color'] for node in G.nodes]
#
# # 绘制图形
# nx.draw(G, with_labels=True, node_color=node_colors, node_size=800, font_size=10)
#
# # 显示图形
# plt.title('Classification Graph')
# plt.show()


# # 画图（step-loss）
# plt.figure()
# total_iterations = 10
# steps_per_iteration = 20
# total_steps = total_iterations * steps_per_iteration
# # 生成横坐标
# x = []
# for i in range(total_iterations):
#     for j in range(steps_per_iteration):
#         x.append(i * steps_per_iteration + j + 1)
# # 绘制图形
# plt.plot(x, val_loss_data, label='Validation Loss')
# plt.plot(x, train_loss0_data, label='Training Loss')
# # 如果有第二个训练过程的loss也画出来
# # plt.plot(range(len(train_loss1_data)), train_loss1_data, label='Training Loss 1')
# plt.xlabel('Steps')
# plt.ylabel('Loss')
# plt.legend()
# plt.title('Training and Validation Loss vs. Steps')
# plt.savefig(path + '\\' + 'Training and Validation.svg')
# plt.savefig(path + '\\' + 'Training and Validation.pdf')
# plt.show()




###求均值
# import numpy as np
# val_loss_avg = []
# train_loss0_avg = []
#
# for j in range(iteration):
#     labels_local = labels.clone().detach()
#     select_class = [0, 1]
#     print('EPOCH {} ITERATION {} '.format(mmh + 1, j + 1))
#     class1_idx = []
#     class2_idx = []
#
#     for k in range(node_num):
#         if (labels_local[k] == select_class[0]):
#             class1_idx.append(k)
#         elif (labels_local[k] == select_class[1]):
#             class2_idx.append(k)
#
#     train_loss0_iter = []
#     val_loss_iter = []
#
#     for m in range(step):
#         class1_train = random.sample(class1_idx, train_shot_0)
#         class2_train = random.sample(class2_idx, train_shot_1)
#         class1_test = [n1 for n1 in class1_idx if n1 not in class1_train]
#         class2_test = [n2 for n2 in class2_idx if n2 not in class2_train]
#         train_idx = class1_train + class2_train
#         random.shuffle(train_idx)
#         test_idx = class1_test + class2_test
#         random.shuffle(test_idx)
#
#         model, loss_train0, optimizer = train_regression(model, features, labels_local, train_idx,
#                                                          args.epochs, args.weight_decay, args.lr, adj)
#         train_loss0_iter.append(loss_train0.item())
#         acc_query, loss_val = test_regression(model, features, labels_local, test_idx, adj)
#         val_loss_iter.append(loss_val.item())
#         print('Step:', m + 1, '\tMeta_Training_Loss0:', loss_train0.item(), '\tMeta_Test_Loss0:',
#               loss_val.item(), '\tMeta_Test_acc0:', acc_query.item())
#         reset_array()
#
#     val_loss_avg.append(np.mean(val_loss_iter))
#     train_loss0_avg.append(np.mean(train_loss0_iter))
#
# # Save averaged losses
# pd.DataFrame(val_loss_avg).to_csv(path + '\\' + 'val_loss_avg.csv', header=None)
# pd.DataFrame(train_loss0_avg).to_csv(path + '\\' + 'train_loss0_avg.csv', header=None)
#
# # Load data for plotting
# val_loss_data_avg = pd.read_csv(path + '\\' + 'val_loss_avg.csv', usecols=[1], header=None).values.flatten()
# train_loss0_data_avg = pd.read_csv(path + '\\' + 'train_loss0_avg.csv', usecols=[1],
#                                    header=None).values.flatten()
#
# # Plot
# plt.figure()
# plt.plot(range(len(val_loss_data_avg)), val_loss_data_avg, label='Validation Loss')
# plt.plot(range(len(train_loss0_data_avg)), train_loss0_data_avg, label='Training Loss')
# plt.xlabel('Iterations')
# plt.ylabel('Loss')
# plt.legend()
# plt.title('Average Training and Validation Loss vs. Iterations')
# plt.savefig(path + '\\' + 'Training and Validation Average.svg')
# plt.savefig(path + '\\' + 'Training and Validation Average.pdf')
# plt.show()