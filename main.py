# %%
# 不挂断运行命令为：
# nohup python main.py --mode cross_subject > /dev/null 2> ./output/txt/error.log &
import os
import numpy as np
import time
import math
import scipy
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import statistics
import random
from scipy import signal
from sklearn.metrics import classification_report
from copy import deepcopy
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

import argparse
import sys
import datetime
from tqdm import tqdm

from data_processor import Data_Processor
# 旧模型model_old和改进的模型model
from model import Model,GradReverse,Feature,Predictor
# from model_old import Model,GradReverse,Feature,Predictor
from load_data import Load_Data
from prepare_data import Prepare_Data
from utils import L2Distance,reset_grad,discrepancy,create_mini_batches,create_mini_batches_T,add_gaussian_noise,Calculate_PSD_DE
from compute_clusters import Compute_Clusters

import matplotlib.font_manager as fm

# 设置matplotlib使用系统中已有的中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'AR PL UMing CN', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 确保字体正确设置 - 创建一个自定义字体属性对象
chinese_font = fm.FontProperties(fname='/usr/share/fonts/wqy-microhei/wqy-microhei.ttc')

# 预训练、聚类训练、自适应训练的轮数
PRE_EPOCHS = 50
CLU_EPOCHS = 20
ADP_EPOCHS = 30

# 预训练、聚类训练、自适应训练的学习率
PRE_LR = 0.001
CLU_LR = 0.1
ADP_LR = 0.01

# 创建记录损失值的字典
training_losses = {
    '预训练阶段': {'steps': [], 'loss': []},
    '聚类损失训练阶段': {'steps': [], 'loss': []},
    '自适应训练阶段': {'steps': [], 'loss': []}
}

# 添加命令行参数解析
parser = argparse.ArgumentParser(description='25AC Emotion Recognition')
parser.add_argument('--timestamp', type=str, default='', help='Timestamp for result files (format: M_D_H_M). If empty, current time will be used.')
parser.add_argument('--mode', type=str, default='cross_subject', choices=['cross_dataset', 'cross_subject'], 
                    help='Training mode: cross_dataset (SEED->DEAP) or cross_subject (SEED[1,2,...,14]->SEED[15])')

# args = parser.parse_args()

args, unknown = parser.parse_known_args()

# 获取模式和时间戳字符串
mode = args.mode
timestamp_str = args.timestamp
if not timestamp_str:
    # 如果没有提供时间戳参数，则自动使用当前时间作为时间戳
    now = datetime.datetime.now()
    timestamp_str = f"{now.month}_{now.day}_{now.hour}_{now.minute}"

timestamp_suffix = f"_{timestamp_str}"

# 日志文件保存路径
if mode == 'cross_dataset':
    log_filename = f"./output/txt/cd/output{timestamp_suffix}.txt"
elif mode == 'cross_subject':
    log_filename = f"./output/txt/cs/output{timestamp_suffix}.txt"

# 图片文件保存路径
if mode == 'cross_dataset':
    pic_filename = f"./output/pic/cd/"
elif mode == 'cross_subject':
    pic_filename = f"./output/pic/cs/"

# 创建一个同时写入文件和打印到控制台的类
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# 重定向标准输出
sys.stdout = Logger(log_filename)

if unknown:
    print(f"Warning: Ignoring unknown arguments: {unknown}")

print(f"以 {mode} 模式运行")
print(f"本文件运行时间为: {now.year}年{now.month}月{now.day}日 {now.hour}:{now.minute}:{now.second}")
print(f"输出已重定向到文件: {log_filename}")

# 训练结束后绘制损失曲线
def plot_training_loss(training_losses, timestamp_suffix):
    plt.figure(figsize=(15, 10))
    
    # 创建三个子图
    plt.subplot(3, 1, 1)
    plt.plot(training_losses['预训练阶段']['steps'], training_losses['预训练阶段']['loss'], 'r-', label='分类损失')
    plt.title(f'预训练阶段损失, EPOCH = {PRE_EPOCHS}, LR = {PRE_LR}', fontproperties=chinese_font)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(training_losses['聚类损失训练阶段']['steps'], training_losses['聚类损失训练阶段']['loss'], 'b-', label='分类损失')
    plt.title(f'聚类损失训练阶段损失, EPOCH = {CLU_EPOCHS}, LR = {CLU_LR}', fontproperties=chinese_font)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(training_losses['自适应训练阶段']['steps'], training_losses['自适应训练阶段']['loss'], 'g-', label='总损失')
    plt.title(f'自适应训练阶段损失, EPOCH = {ADP_EPOCHS}, LR = {ADP_LR}', fontproperties=chinese_font)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 添加包含模式和EPOCH信息的整体标题
    plt.suptitle(f'训练阶段损失统计 (训练模式：{mode})', fontproperties=chinese_font, fontsize=14, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # rect参数调整以给标题留出空间
    plt.savefig(f'{pic_filename}training_loss{timestamp_suffix}.png')
    print(f"训练损失曲线已保存至 {pic_filename}training_loss{timestamp_suffix}.png")

# 检查预处理文件是否存在
def check_preprocessed_files_exist(subjects, data_path):
    """检查预处理文件是否已存在"""
    all_exist = True
    for sub in subjects:
        files_to_check = [
            f'{data_path}PSD_sub_{sub}.npy',
            f'{data_path}DE_sub_{sub}.npy', 
            f'{data_path}labels_sub_{sub}.npy',
            f'{data_path}Data_Orig_sub_{sub}.npy'
        ]
        for file_path in files_to_check:
            if not os.path.exists(file_path):
                all_exist = False
                print(f"缺少预处理文件: {file_path}")
                break
        if not all_exist:
            break
    return all_exist

data_obj=Data_Processor()
# 跨数据集模式 - 加载DEAP数据
if mode == 'cross_dataset':
    #Load DEAP data
    #Window size 2s, step size 1s (sample frequency of DEAP 128Hz)
    window_size=256 #2s
    step_size=128  #1s
    subjectList_DEAP=['01']
    print(f'subjectList_DEAP = {subjectList_DEAP}')
    DataSeg_PSD_all_trials_DEAP, DataSeg_DE_all_trials_DEAP, labels_all_trials_DEAP=data_obj.Load_DEAP(subjectList_DEAP, window_size, step_size)

#Load SEED data
#Window size 2s, step size 1s (sample frequency of SEED 200Hz)
window_size=400
step_size=200

data_P_path_SEED = './data/SEED/preprocessed/'
if mode == 'cross_dataset':
    # 跨数据集模式 - 使用所有SEED受试者
    # subjectList_SEED=['1','2','3']
    subjectList_SEED=['1','2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
    # 每个受试者有三段
    sessionList_SEED=['1','2','3']
    print(f'subjectList_SEED = {subjectList_SEED}')
    # 检查预处理文件是否存在
    preprocessed_files_exist = check_preprocessed_files_exist(subjectList_SEED, data_P_path_SEED)
    
    # 如果预处理文件不存在，则调用Load_SEED函数
    if not preprocessed_files_exist:
        print("预处理文件不存在，开始从原始数据加载...")
        DataSeg_PSD_all_trials_SEED, DataSeg_DE_all_trials_SEED, labels_all_trials_SEED = data_obj.Load_SEED(subjectList_SEED, sessionList_SEED, window_size, step_size)
    else:
        print("已找到预处理文件，跳过原始数据加载步骤")
        # 这里设为None，之后会通过load_P_SEED函数加载
        DataSeg_PSD_all_trials_SEED, DataSeg_DE_all_trials_SEED, labels_all_trials_SEED = None, None, None
elif mode == 'cross_subject':
    # 跨用户模式 - 分别加载训练和测试用户
    subjectList_SEED_train=['1','2','3','4','5','6','7','8','9','10','11','12','13','14']
    # subjectList_SEED_train=['1']
    print(f'subjectList_SEED_train = {subjectList_SEED_train}')
    subjectList_SEED_test=['15']
    # subjectList_SEED_test=['3']
    print(f'subjectList_SEED_test = {subjectList_SEED_test}')
    sessionList_SEED=['1','2','3']

    # 检查训练数据预处理文件是否存在
    train_preprocessed_files_exist = check_preprocessed_files_exist(subjectList_SEED_train, data_P_path_SEED)

    # 如果训练预处理文件不存在，则调用Load_SEED函数
    if not train_preprocessed_files_exist:
        print("训练数据预处理文件不存在，开始从原始数据加载...")
        DataSeg_PSD_all_trials_SEED_train, DataSeg_DE_all_trials_SEED_train, labels_all_trials_SEED_train = data_obj.Load_SEED(subjectList_SEED_train, sessionList_SEED, window_size, step_size)
    else:
        print("已找到训练数据预处理文件，跳过原始数据加载步骤")
        # 这里设为None，之后会通过load_P_SEED函数加载
        DataSeg_PSD_all_trials_SEED_train, DataSeg_DE_all_trials_SEED_train, labels_all_trials_SEED_train = None, None, None
    
    # 检查测试数据预处理文件是否存在
    test_preprocessed_files_exist = check_preprocessed_files_exist(subjectList_SEED_test, data_P_path_SEED)
    
    # 如果测试预处理文件不存在，则调用Load_SEED函数
    if not test_preprocessed_files_exist:
        print("测试数据预处理文件不存在，开始从原始数据加载...")
        DataSeg_PSD_all_trials_SEED_test, DataSeg_DE_all_trials_SEED_test, labels_all_trials_SEED_test = data_obj.Load_SEED(subjectList_SEED_test, sessionList_SEED, window_size, step_size)
    else:
        print("已找到测试数据预处理文件，跳过原始数据加载步骤")
        # 这里设为None，之后会通过load_P_SEED函数加载
        DataSeg_PSD_all_trials_SEED_test, DataSeg_DE_all_trials_SEED_test, labels_all_trials_SEED_test = None, None, None
else:
    print("Invalid mode. Please choose 'cross_dataset' or 'cross_subject'.")
# %%
# #Load pre-processed data
load_data_obj=Load_Data()

if mode == 'cross_dataset':
    # 跨数据集模式
    DataSeg_PSD_all_trials_DEAP,DataSeg_DE_all_trials_DEAP,labels_all_trials_DEAP=load_data_obj.load_P_DEAP(subjectList_DEAP)
    DataSeg_PSD_all_trials_SEED,DataSeg_DE_all_trials_SEED,labels_all_trials_SEED=load_data_obj.load_P_SEED(subjectList_SEED)
elif mode == 'cross_subject':
    # 跨用户模式
    DataSeg_PSD_all_trials_SEED_train,DataSeg_DE_all_trials_SEED_train,labels_all_trials_SEED_train=load_data_obj.load_P_SEED(subjectList_SEED_train)
    DataSeg_PSD_all_trials_SEED_test,DataSeg_DE_all_trials_SEED_test,labels_all_trials_SEED_test=load_data_obj.load_P_SEED(subjectList_SEED_test)
    # for i in range(100, 105):
    #     print(f'labels_all_trials_SEED_train[{i}][0] = {labels_all_trials_SEED_train[i][0]}')
    #     print(f'labels_all_trials_SEED_test[{i}][0] = {labels_all_trials_SEED_test[i][0]}')
else:
    print("Invalid mode. Please choose 'cross_dataset' or 'cross_subject'.")

#Prepare data
num_classes=2
batch_size=64
prepare_data_obj=Prepare_Data()

if mode == 'cross_dataset':
    # 跨数据集模式
    data_batch_DEAP,label_batch_DEAP=prepare_data_obj.Prepare_DEAP(DataSeg_PSD_all_trials_DEAP, DataSeg_DE_all_trials_DEAP, labels_all_trials_DEAP, batch_size, num_classes)
    data_batch_SEED,label_batch_SEED=prepare_data_obj.Prepare_SEED(DataSeg_PSD_all_trials_SEED, DataSeg_DE_all_trials_SEED, labels_all_trials_SEED, batch_size, num_classes)
    
    train_data_batch_s=data_batch_SEED
    train_label_batch_s=label_batch_SEED
    train_data_batch_t=data_batch_DEAP
    train_label_batch_t=label_batch_DEAP
elif mode == 'cross_subject':
    # 跨用户模式
    data_batch_SEED_train,label_batch_SEED_train=prepare_data_obj.Prepare_SEED(DataSeg_PSD_all_trials_SEED_train, DataSeg_DE_all_trials_SEED_train, labels_all_trials_SEED_train, batch_size, num_classes)
    data_batch_SEED_test,label_batch_SEED_test=prepare_data_obj.Prepare_SEED(DataSeg_PSD_all_trials_SEED_test, DataSeg_DE_all_trials_SEED_test, labels_all_trials_SEED_test, batch_size, num_classes)
    
    train_data_batch_s=data_batch_SEED_train
    train_label_batch_s=label_batch_SEED_train
    train_data_batch_t=data_batch_SEED_test
    train_label_batch_t=label_batch_SEED_test
else:
    print("Invalid mode. Please choose 'cross_dataset' or 'cross_subject'.")

#Pre-training
model=Model()
G = model.Generator()
G.to('cuda')
C1 = model.Classifier(num_classes)
C1.to('cuda')
C2 = model.Classifier(num_classes)
C2.to('cuda')

opt_g = optim.Adam(G.parameters(), lr=PRE_LR, weight_decay=0.0005)
opt_c1 = optim.Adam(C1.parameters(), lr=PRE_LR, weight_decay=0.0005)
opt_c2 = optim.Adam(C2.parameters(), lr=PRE_LR, weight_decay=0.0005)
G.train()
C1.train()
C2.train()

torch.cuda.manual_seed(1)
criterion = nn.CrossEntropyLoss().cuda()
all_loss_cls=[]
all_loss_dis=[]

###预训练阶段###
# 增加轮数
# Epochs=20
start_time = time.time()
Epochs = PRE_EPOCHS
loss_count = 1
for epoch in tqdm(range(Epochs), desc="预训练阶段"):
    for i in range(train_data_batch_s.shape[0]):
        train_data_batch_tmp_s=train_data_batch_s[i]
        train_tensor_s = torch.from_numpy(train_data_batch_tmp_s)
        train_tensor_s=train_tensor_s.to('cuda')
        train_tensor_s=train_tensor_s.float()
        train_label_batch_tmp_s=train_label_batch_s[i]
        train_label_tmp_tensor_s = torch.from_numpy(train_label_batch_tmp_s)
        train_label_tmp_one_hot_s=torch.nn.functional.one_hot(train_label_tmp_tensor_s, num_classes) #No of class
        train_label_tmp_tensor_s=train_label_tmp_one_hot_s.to('cuda')
        train_label_tmp_tensor_s = train_label_tmp_tensor_s.to(torch.float)
        train_tensor_s=torch.swapaxes(train_tensor_s, 1, 2)
        reset_grad(opt_g,opt_c1,opt_c2)

        feat_s = G(train_tensor_s)
        output_C1, output_C1_prev = C1(feat_s)
        output_C2, output_C2_prev = C2(feat_s)
        loss_C1_s = F.cross_entropy(output_C1_prev, train_label_tmp_tensor_s)
        loss_C2_s = F.cross_entropy(output_C2_prev, train_label_tmp_tensor_s)
        loss_C1_C2_s=loss_C1_s+loss_C2_s

        groupdro_eta=1
        loss_cls= (groupdro_eta * loss_C1_C2_s).exp()
        Alpha_DIS=0.5
        loss_dis = discrepancy(output_C1_prev, output_C2_prev)
        loss=loss_cls + (Alpha_DIS * loss_dis)

        loss.backward()
        #Adjust parameters of Feature Extractor(G), and Classifiers(C1, C2)
        opt_g.step()
        opt_c1.step()
        opt_c2.step()
        reset_grad(opt_g,opt_c1,opt_c2)

        loss_cls_tmp=loss_cls.detach().cpu().data.numpy()
        loss_dis_tmp=loss_dis.detach().cpu().data.numpy()
        if i%100==0:
            all_loss_cls.append(loss_cls_tmp)
            all_loss_dis.append(loss_dis_tmp)

        #print('Epoch: '+str(epoch+1))
        #print('Iteration:  '+str(i+1))
        #print('Loss')
        #print(loss)
        training_losses['预训练阶段']['steps'].append(loss_count)
        loss_count=loss_count+1
        training_losses['预训练阶段']['loss'].append(loss.detach().cpu().numpy())
# 记录预训练阶段耗时
pre_train_time = time.time() - start_time
###预训练阶段结束###

# 简化
compute_clusters_obj=Compute_Clusters()
#Compute source clusters
all_centers_s= compute_clusters_obj.get_source_centers(G, C1, C2, train_data_batch_s, train_label_batch_s, num_classes)

# 增加轮数
Epochs=CLU_EPOCHS
# Epochs=int(0.5*EPOCHS)
all_loss_cls=[]
all_loss=[]

###聚类训练阶段###
start_time = time.time()
loss_count = 1
for epoch in tqdm(range(Epochs), desc="聚类损失训练阶段"):
  for i in range(train_data_batch_s.shape[0]):
      train_data_batch_tmp_s=train_data_batch_s[i]
      train_tensor_s = torch.from_numpy(train_data_batch_tmp_s)
      train_tensor_s=train_tensor_s.to('cuda')
      train_tensor_s=train_tensor_s.float()
      train_label_batch_tmp_s=train_label_batch_s[i]
      train_label_tmp_tensor_s = torch.from_numpy(train_label_batch_tmp_s)
      train_label_tmp_one_hot_s=torch.nn.functional.one_hot(train_label_tmp_tensor_s, num_classes)
      train_label_tmp_tensor_s=train_label_tmp_one_hot_s.to('cuda')
      train_label_tmp_tensor_s = train_label_tmp_tensor_s.to(torch.float)

      centers_tmp_s=all_centers_s
      centers_tmp_s = torch.from_numpy(centers_tmp_s)
      centers_tmp_s=centers_tmp_s.to('cuda')
      centers_tensor_s = centers_tmp_s.to(torch.float)
      train_tensor_s=torch.swapaxes(train_tensor_s, 1, 2)
      reset_grad(opt_g,opt_c1,opt_c2)

      feat_s = G(train_tensor_s)
      output_C1_s, output_C1_prev_s = C1(feat_s)
      output_C2_s, output_C2_prev_s = C2(feat_s)
      output_C1_C2_prev_s=(output_C1_prev_s+output_C2_prev_s)/2
      loss_C1_C2_s=F.cross_entropy(output_C1_C2_prev_s, train_label_tmp_tensor_s)

      #loss_intra: cluster-compacting loss
      #loss_inter: cluster-separating loss
      loss_cs = 0
      loss_intra = 0
      loss_inter = 0
      lr_cs=CLU_LR
      

      for l in range(num_classes):
          label_batch_s=train_label_tmp_tensor_s.data.max(1)[1]
          _idx_s=torch.where(label_batch_s==l)
          _feat_s=feat_s[_idx_s]
          if _feat_s.shape[0]!=0:
              m_feat_s = torch.mean(_feat_s, dim=0)
              m_feat_s=m_feat_s.to('cuda')
              delta_cs_l = centers_tensor_s[l] - m_feat_s
              delta_cs_l_np = delta_cs_l.cpu().detach().numpy()
              all_centers_s[l] = all_centers_s[l] - lr_cs * delta_cs_l_np
              loss_cs_l = L2Distance(m_feat_s, centers_tensor_s[l])
              loss_cs += loss_cs_l

              bs_ = _feat_s.shape[0]
              cl_feat_s = centers_tensor_s[l].repeat((bs_, 1))
              cl_feat_s=cl_feat_s.to('cuda')
              loss_intra_l = L2Distance(_feat_s, cl_feat_s, dim=1) / bs_
              loss_intra += loss_intra_l

      THR_M= 50
      c_inter=0
      for m in range(num_classes - 1):
          for n in range(m + 1, num_classes):
              c_m=torch.count_nonzero(centers_tensor_s[m])
              c_n=torch.count_nonzero(centers_tensor_s[n])
              if c_m!=0 and c_n!=0:
                    loss_inter_mn = torch.max(THR_M - L2Distance(centers_tensor_s[m], centers_tensor_s[n]),
                                              torch.FloatTensor([0]).cuda()).squeeze()
                    loss_inter += loss_inter_mn
                    c_inter=c_inter+1
      if c_inter!=0:
          loss_inter = loss_inter / c_inter

      loss_intra = loss_intra / num_classes
      Gamma_INTER= 0.1
      Gamma_INTRA= 0.1
      loss = loss_C1_C2_s +  (loss_intra * Gamma_INTRA) + (loss_inter * Gamma_INTER)

      reset_grad(opt_g,opt_c1,opt_c2)
      loss.backward()
      opt_g.step()
      opt_c1.step()
      opt_c2.step()

      loss_cls_tmp=loss_C1_C2_s.detach().cpu().data.numpy()
      loss_tmp=loss.detach().cpu().data.numpy()
      if i%100==0:
          all_loss_cls.append(loss_cls_tmp)
          all_loss.append(loss_tmp)

      #print('Epoch: '+str(epoch+1))
      #print('Iteration:  '+str(i+1))
      #print(loss)
      training_losses['聚类损失训练阶段']['steps'].append(loss_count)
      loss_count=loss_count+1
      training_losses['聚类损失训练阶段']['loss'].append(loss.detach().cpu().numpy())
# 记录聚类训练耗时
cluster_train_time = time.time() - start_time
### Training With Cluster Loss Ends ###

#Compute source clusters
all_centers_s= compute_clusters_obj.get_source_centers(G, C1, C2, train_data_batch_s, train_label_batch_s, num_classes)

#Compute intra-cluster distance (mean and std) and classifier discrepancy (mean and std)
mean_dist_classifier_s, std_dist_classifier_s, mean_dist_feat_s, all_mean_dist_feat_s, all_std_dist_feat_s= compute_clusters_obj.get_mean_dist(G, C1, C2, train_data_batch_s, train_label_batch_s, all_centers_s,num_classes)

#Compute target clusters
all_centers_t = compute_clusters_obj.get_target_centers(G, C1, C2, train_data_batch_t, all_centers_s, mean_dist_classifier_s, mean_dist_feat_s, all_mean_dist_feat_s,num_classes)

#Compute average clusters
centers=np.empty((all_centers_s.shape[0],all_centers_s.shape[1]))
for l in range(num_classes):
    centers[l]=(all_centers_s[l]+all_centers_t[l])/2

###自适应训练阶段###
# 增加轮数
Epochs=ADP_EPOCHS
# Epochs=int(0.5*EPOCHS)
mean_dist_classifier_s=torch.from_numpy(mean_dist_classifier_s)
mean_dist_classifier_s=mean_dist_classifier_s.to('cuda')
mean_dist_classifier_s = mean_dist_classifier_s.to(torch.float)
mean_dist_classifier_s=mean_dist_classifier_s
mean_dist_feat_s = torch.from_numpy(mean_dist_feat_s)
mean_dist_feat_s=mean_dist_feat_s.to('cuda')
mean_dist_feat_s = mean_dist_feat_s.to(torch.float)
mean_dist_feat_s=mean_dist_feat_s+1

all_mean_dist_feat_s = torch.from_numpy(all_mean_dist_feat_s)
all_mean_dist_feat_s=all_mean_dist_feat_s.to('cuda')
all_mean_dist_feat_s = all_mean_dist_feat_s.to(torch.float)
all_std_dist_feat_s = torch.from_numpy(all_std_dist_feat_s)
all_std_dist_feat_s=all_std_dist_feat_s.to('cuda')
all_std_dist_feat_s = all_std_dist_feat_s.to(torch.float)
all_centers_s_tmp = torch.from_numpy(all_centers_s)
all_centers_s_tmp=all_centers_s_tmp.to('cuda')
all_centers_s_tmp = all_centers_s_tmp.to(torch.float)

all_mean_dist_feat_s[0]=all_mean_dist_feat_s[0]+1
all_mean_dist_feat_s[1]=all_mean_dist_feat_s[1]+1
all_loss_cls=[]
all_loss=[]
train_data_not_selected_batch_t=train_data_batch_t
count_selection=5  #Run target selection for training for max 5 times

#Target selection for training based on similarity with source
#Gradual Proximity-guided Target Data Selection(GPTDS)
loss_count = 1
start_time = time.time()
for cc in tqdm(range(count_selection), desc="自适应训练目标选择"):
    if (train_data_not_selected_batch_t.shape[0]>0):
        train_selected_t=[]
        train_not_selected_t=[]
        for i in range(int(train_data_not_selected_batch_t.shape[0])):
            train_tmp_t=train_data_not_selected_batch_t[i]
            train_tmp_t = torch.from_numpy(train_tmp_t)
            train_tmp_t=train_tmp_t.to('cuda')
            train_tmp_t=train_tmp_t.float()

            feat_t = G(train_tmp_t)
            output_C1_t, output_C1_prev_t = C1(feat_t)
            output_C2_t, output_C2_prev_t = C2(feat_t)
            output_C1_C2_t= (output_C1_t+output_C2_t)/2
            max_pred_val_t,max_pred_idx_t=torch.max(output_C1_C2_t,1)

            for l in range(num_classes):
                  _idx_t=torch.where(max_pred_idx_t==l)
                  _feat_t=feat_t[_idx_t]
                  _train_tmp_t=train_tmp_t[_idx_t]
                  jj=0
                  for ii in range(_feat_t.shape[0]):
                        #Take target samples that have distance < (mean distance + (std/2))
                        if (L2Distance(all_centers_s_tmp[l], _feat_t[ii]) < (all_mean_dist_feat_s[l]+(all_std_dist_feat_s[l]/2))):
                            train_selected_t.append(_train_tmp_t[ii].detach().cpu().data.numpy())
                            jj=jj+1
                        else:
                          train_not_selected_t.append(_train_tmp_t[ii].detach().cpu().data.numpy())

        train_selected_t_np = np.array(train_selected_t)
        train_not_selected_t_np=np.array(train_not_selected_t)

        if train_selected_t_np.shape[0]>0:
            data_size=train_selected_t_np.shape[0]
            itr_len=int(data_size/batch_size)
            data=train_selected_t_np[0:itr_len*batch_size]
            train_data_selected_batch_t= np.empty([itr_len, batch_size, data.shape[1], data.shape[2]])
            jj=0

            for ii in range(itr_len):
              train_data_selected_batch_t[ii]=data[jj:jj+batch_size]
              jj=jj+batch_size
        else:
            train_data_selected_batch_t=train_selected_t_np

        if train_not_selected_t_np.shape[0]>0:
            data_size=train_not_selected_t_np.shape[0]
            itr_len=int(data_size/batch_size)
            data=train_not_selected_t_np[0:itr_len*batch_size]
            train_data_not_selected_batch_t= np.empty([itr_len, batch_size, data.shape[1], data.shape[2]])

            jj=0
            for ii in range(itr_len):
              train_data_not_selected_batch_t[ii]=data[jj:jj+batch_size]
              jj=jj+batch_size

        else:
            train_data_not_selected_batch_t=train_not_selected_t_np

        #print('selected size')
        #print(train_data_selected_batch_t.shape)
        #print('not selected size')
        #print(train_data_not_selected_batch_t.shape)
        train_n=min(train_data_batch_s.shape[0],train_data_selected_batch_t.shape[0])

        #### Training with final loss function #####
        for epoch in tqdm(range(Epochs), desc="自适应训练阶段"):
            for i in range(int(train_n)):
                train_data_batch_tmp_s=train_data_batch_s[i]
                train_data_batch_tmp_t=train_data_selected_batch_t[i]
                train_tensor_s = torch.from_numpy(train_data_batch_tmp_s)
                train_tensor_t = torch.from_numpy(train_data_batch_tmp_t)
                train_tensor_s=train_tensor_s.to('cuda')
                train_tensor_t=train_tensor_t.to('cuda')
                train_tensor_s=train_tensor_s.float()
                train_tensor_t=train_tensor_t.float()
                train_label_batch_tmp_s=train_label_batch_s[i]
                train_label_tmp_tensor_s = torch.from_numpy(train_label_batch_tmp_s)
                train_label_tmp_one_hot_s=torch.nn.functional.one_hot(train_label_tmp_tensor_s, num_classes)
                train_label_tmp_tensor_s=train_label_tmp_one_hot_s.to('cuda')
                train_label_tmp_tensor_s = train_label_tmp_tensor_s.to(torch.float)

                centers_tmp_s=all_centers_s
                centers_tmp_t=all_centers_t
                centers_tmp=centers
                centers_tmp_s = torch.from_numpy(centers_tmp_s)
                centers_tmp_s=centers_tmp_s.to('cuda')
                centers_tmp_t = torch.from_numpy(centers_tmp_t)
                centers_tmp_t=centers_tmp_t.to('cuda')
                centers_tmp = torch.from_numpy(centers_tmp)
                centers_tmp=centers_tmp.to('cuda')
                centers_tensor_s = centers_tmp_s.to(torch.float)
                centers_tensor_t = centers_tmp_t.to(torch.float)
                centers_tensor = centers_tmp.to(torch.float)
                train_tensor_s=torch.swapaxes(train_tensor_s, 1, 2)
                train_tensor_t=torch.swapaxes(train_tensor_t, 1, 2)
                reset_grad(opt_g,opt_c1,opt_c2)

                feat_s = G(train_tensor_s)
                output_C1_s, output_C1_prev_s = C1(feat_s)
                output_C2_s, output_C2_prev_s = C2(feat_s)
                output_C1_C2_prev_s=(output_C1_prev_s+output_C2_prev_s)/2
                loss_Cls = F.cross_entropy(output_C1_C2_prev_s, train_label_tmp_tensor_s)  #Classification Loss

                feat_t = G(train_tensor_t)
                output_C1_t, output_C1_prev_t = C1(feat_t)
                output_C2_t, output_C2_prev_t = C2(feat_t)
                output_C1_C2_t= (output_C1_t+output_C2_t)/2

                max_pred_val_t,max_pred_idx_t=torch.max(output_C1_C2_t,1)
                confident_pred_idx_t=torch.where(max_pred_val_t>0.99)
                confident_pred_class_t=max_pred_idx_t[confident_pred_idx_t]
                confident_pred_output_t=max_pred_val_t[confident_pred_idx_t]
                confident_feat_t=feat_t[confident_pred_idx_t]
                confident_C1_t=output_C1_prev_t[confident_pred_idx_t]
                confident_C2_t=output_C2_prev_t[confident_pred_idx_t]
                jj=0

                for ii in range(confident_feat_t.shape[0]):
                    if ((L2Distance(confident_C1_t[ii], confident_C2_t[ii])<mean_dist_classifier_s)):
                        confident_pred_class_t[jj]=confident_pred_class_t[ii]
                        confident_pred_output_t[jj]=confident_pred_output_t[ii]
                        confident_feat_t[jj]=confident_feat_t[ii]
                        confident_C1_t[jj]=confident_C1_t[ii]
                        confident_C2_t[jj]=confident_C2_t[ii]
                        jj=jj+1
                confident_pred_class_t=confident_pred_class_t[0:jj]
                confident_pred_output_t=confident_pred_output_t[0:jj]
                confident_feat_t=confident_feat_t[0:jj]
                confident_C1_t=confident_C1_t[0:jj]
                confident_C2_t=confident_C2_t[0:jj]

                loss_intra_t = 0
                loss_inter_t = 0
                loss_intra_s = 0
                loss_inter_s = 0
                loss_cd=0
                loss_cmd = 0
                lr_cs=ADP_LR
                lr_c=ADP_LR
                pesudo_label_nums=torch.zeros(num_classes)
                tmp_centers_t=torch.zeros(num_classes, feat_t.shape[1])
                pesudo_label_nums=pesudo_label_nums.to('cuda')
                tmp_centers_t=tmp_centers_t.to('cuda')

                #Calculate cluster-compacting loss for target
                for l in range(num_classes):
                    _idx_t=torch.where(confident_pred_class_t==l)
                    _feat_t=confident_feat_t[_idx_t]
                    '''
                    jj=0
                    for ii in range(_feat_t.shape[0]):
                        if ((L2Distance(centers_tensor[l], _feat_t[ii]) < all_mean_dist_feat_s[l])):
                            _feat_t[jj]=_feat_t[ii]
                            jj=jj+1
                    _feat_t=_feat_t[0:jj]
                    '''
                    pesudo_label_nums[l]=_feat_t.shape[0]

                    if _feat_t.shape[0]!=0:
                        m_feat_t = torch.mean(_feat_t, dim=0)
                        tmp_centers_t[l] = m_feat_t
                        m_feat_t=m_feat_t.to('cuda')
                        delta_cs_l_t = centers_tensor_t[l] - m_feat_t
                        delta_cs_l_t = delta_cs_l_t.cpu().detach().numpy()
                        all_centers_t[l] = all_centers_t[l] - lr_cs * delta_cs_l_t
                        all_centers_tensor_t = torch.from_numpy(all_centers_t)
                        all_centers_tensor_t=all_centers_tensor_t.to('cuda')
                        all_centers_tensor_t = all_centers_tensor_t.to(torch.float)

                        bs_ = _feat_t.shape[0]
                        cl_feat_t = all_centers_tensor_t[l].repeat((bs_, 1))
                        cl_feat_t=cl_feat_t.to('cuda')
                        loss_intra_l_t = L2Distance(_feat_t, cl_feat_t, dim=1) / bs_
                        loss_intra_t += loss_intra_l_t
                loss_intra_t = loss_intra_t / num_classes

                true_label_nums=torch.zeros(num_classes)
                tmp_centers_s=torch.zeros(num_classes, feat_s.shape[1])
                true_label_nums=true_label_nums.to('cuda')
                tmp_centers_s=tmp_centers_s.to('cuda')

                #Calculate cluster-compacting loss for source
                for l in range(num_classes):
                    label_batch_s=train_label_tmp_tensor_s.data.max(1)[1]
                    _idx_s=torch.where(label_batch_s==l)
                    _feat_s=feat_s[_idx_s]
                    true_label_nums[l]=_feat_s.shape[0]
                    if _feat_s.shape[0]!=0:
                        m_feat_s = torch.mean(_feat_s, dim=0)
                        tmp_centers_s[l] = m_feat_s
                        m_feat_s=m_feat_s.to('cuda')
                        delta_cs_l = centers_tensor_s[l] - m_feat_s
                        delta_cs_l_np = delta_cs_l.cpu().detach().numpy()
                        all_centers_s[l] = all_centers_s[l] - lr_cs * delta_cs_l_np
                        all_centers_tensor_s = torch.from_numpy(all_centers_s)
                        all_centers_tensor_s=all_centers_tensor_s.to('cuda')
                        all_centers_tensor_s = all_centers_tensor_s.to(torch.float)

                        bs_ = _feat_s.shape[0]
                        cl_feat_s = all_centers_tensor_s[l].repeat((bs_, 1))
                        cl_feat_s=cl_feat_s.to('cuda')
                        loss_intra_l_s = L2Distance(_feat_s, cl_feat_s, dim=1) / bs_
                        loss_intra_s += loss_intra_l_s
                loss_intra_s = loss_intra_s / num_classes


                all_centers_tensor_t = torch.from_numpy(all_centers_t)
                all_centers_tensor_t=all_centers_tensor_t.to('cuda')
                all_centers_tensor_t = all_centers_tensor_t.to(torch.float)

                THR_M= 50
                c_inter=0

                #Calculate cluster-separating loss for target
                for m in range(num_classes - 1):
                    for n in range(m + 1, num_classes):
                        c_m=torch.count_nonzero(all_centers_tensor_t[m])
                        c_n=torch.count_nonzero(all_centers_tensor_t[n])
                        if c_m!=0 and c_n!=0:
                            loss_inter_mn_t = torch.max(THR_M - L2Distance(all_centers_tensor_t[m], all_centers_tensor_t[n]),
                                                      torch.FloatTensor([0]).cuda()).squeeze()
                            loss_inter_t += loss_inter_mn_t
                            c_inter=c_inter+1

                if c_inter!=0:
                  loss_inter_t = loss_inter_t / c_inter

                #Calculate cluster-separating loss for source
                for m in range(num_classes - 1):
                    for n in range(m + 1, num_classes):
                        c_m=torch.count_nonzero(all_centers_tensor_s[m])
                        c_n=torch.count_nonzero(all_centers_tensor_s[n])
                        if c_m!=0 and c_n!=0:
                              loss_inter_mn = torch.max(THR_M - L2Distance(all_centers_tensor_s[m], all_centers_tensor_s[n]),
                                                        torch.FloatTensor([0]).cuda()).squeeze()
                              loss_inter_s += loss_inter_mn
                              c_inter=c_inter+1
                if c_inter!=0:
                    loss_inter_s = loss_inter_s / c_inter

                #Calculate inter-domain cluster discrepancy loss
                for l in range(num_classes):
                    loss_cd_l = L2Distance(all_centers_tensor_s[l], all_centers_tensor_t[l])
                    loss_cd += loss_cd_l
                loss_cd=loss_cd/num_classes

                centers_tensor = torch.from_numpy(centers)
                centers_tensor=centers_tensor.to('cuda')
                centers_tensor = centers_tensor.to(torch.float)

                #Calculate running combined loss
                for l in range(num_classes):
                        tmp_centers_sl = tmp_centers_s[l]
                        tmp_centers_tl = tmp_centers_t[l]
                        if pesudo_label_nums[l]>0 and true_label_nums[l]>0:
                            m_centers_stl = (pesudo_label_nums[l] * tmp_centers_tl + true_label_nums[l] * tmp_centers_sl) / (pesudo_label_nums[l] + true_label_nums[l])
                        else:
                            m_centers_stl= (tmp_centers_tl+tmp_centers_sl)/2
                        delta_l = centers_tensor[l] - m_centers_stl
                        delta_l_np = delta_l.cpu().detach().numpy()
                        centers[l] = centers[l] - lr_c * delta_l_np
                        centers_tensor = torch.from_numpy(centers)
                        centers_tensor=centers_tensor.to('cuda')
                        centers_tensor = centers_tensor.to(torch.float)

                        loss_cl = L2Distance(m_centers_stl, centers_tensor[l])
                        loss_cmd += loss_cl
                loss_cmd=loss_cmd/num_classes

                BETA_INTER= 0.1
                BETA_INTRA= 0.1
                BETA_CD= 0.5
                BETA_CMD= 0.1

                loss_intra_f_s=(loss_intra_s * BETA_INTRA) #cluster-compacting loss for source
                loss_inter_f_s=(loss_inter_s * BETA_INTER) #cluster-separating loss for source
                loss_intra_f_t=(loss_intra_t * BETA_INTRA)  #cluster-compacting loss for target
                loss_inter_f_t=(loss_inter_t * BETA_INTER) #cluster-separating loss for target
                loss_cd_f= (loss_cd * BETA_CD)  # inter-domain cluster discrepancy loss
                loss_cmd_f= (loss_cmd * BETA_CMD)  #running combined loss
                loss = loss_Cls + loss_intra_f_t + loss_inter_f_t +  loss_intra_f_s + loss_inter_f_s + loss_cd_f + loss_cmd_f

                loss_cls_tmp=loss_Cls.detach().cpu().data.numpy()
                loss_tmp=loss.detach().cpu().data.numpy()
                if i%100==0:
                    all_loss_cls.append(loss_cls_tmp)
                    all_loss.append(loss_tmp)

                reset_grad(opt_g,opt_c1,opt_c2)
                loss.backward()
                opt_g.step()
                opt_c1.step()
                #print('Epoch: '+str(epoch+1))
                #print('Iteration:  '+str(i+1))
                #print(loss)
                training_losses['自适应训练阶段']['steps'].append(loss_count)
                loss_count=loss_count+1
                training_losses['自适应训练阶段']['loss'].append(loss.detach().cpu().numpy())
# 记录自适应训练耗时
adapt_train_time = time.time() - start_time
###Domain Adaptation Ends###

print('Training ends')
plot_training_loss(training_losses, timestamp_suffix)

#Inference
#Prediction Confidence-aware Test-Time Augmentation (PC-TTA)
if mode == 'cross_dataset':
    print('Test on DEAP')
elif mode == 'cross_subject':
    print('Test on SEED')
else:
    print("Invalid test data. Please choose 'cross_dataset' or 'cross_subject'.")

G.eval()
C1.eval()
C2.eval()

acc=0
all_acc=[]
all_preds=[]
all_true_labels=[]
all_acc_Sub=[]
all_preds_Sub=[]
all_true_labels_Sub=[]
criterion = nn.CrossEntropyLoss()

start_time = time.time()
if mode == 'cross_dataset':
    #Test on DEAP
    subjectList_test = subjectList_DEAP
    data_P_path = './data/DEAP/preprocessed/'

    for sub in subjectList_test:
        DE_tmp=np.load(data_P_path+'DE_sub_'+sub+'.npy')
        PSD_tmp=np.load(data_P_path+'PSD_sub_'+sub+'.npy')
        data_orig_tmp=np.load(data_P_path+'Data_Orig_sub_'+sub+'.npy')
        labels_tmp=np.load(data_P_path+'labels_sub_'+sub+'.npy')
        labels_dis_all = deepcopy(labels_tmp)

        for i in range(labels_dis_all.shape[0]):
            if labels_tmp[i][0]<=4.5:
                labels_dis_all[i][0]=0
            else:
                labels_dis_all[i][0]=1
        
        tmp=np.zeros((PSD_tmp.shape[0],PSD_tmp.shape[1]))
        for i in range (PSD_tmp.shape[0]):
            tmp[i]=(((PSD_tmp[i]-np.amin(PSD_tmp[i]))/(np.amax(PSD_tmp[i])-np.amin(PSD_tmp[i])))*2)-1
        PSD_tmp=tmp
        PSD_tmp=np.reshape(PSD_tmp,(PSD_tmp.shape[0],PSD_tmp.shape[1],1))
        labels_dis_all = labels_dis_all.ravel()
        test_data_batch_t, test_data_orig_batch_t, test_label_batch_t= create_mini_batches_T(PSD_tmp, data_orig_tmp, labels_dis_all, batch_size)

        acc_Each_Sub=[]
        preds_Each_Sub=[]
        true_labels_Each_Sub=[]

        for i in range(int(test_data_batch_t.shape[0])):
            test_data_tensor = torch.from_numpy(test_data_batch_t[i])
            test_data_tensor=test_data_tensor.to('cuda')
            test_data_tensor=test_data_tensor.float()
            test_data_orig_tensor = torch.from_numpy(test_data_orig_batch_t[i])
            test_data_orig_tensor=test_data_orig_tensor.to('cuda')
            test_data_orig_tensor=test_data_orig_tensor.float()
            test_data_tensor=torch.swapaxes(test_data_tensor, 1, 2)
            test_labels_tensor = torch.from_numpy(test_label_batch_t[i])
            test_labels_tensor=torch.nn.functional.one_hot(test_labels_tensor, num_classes)
            test_labels_tensor=test_labels_tensor.to('cuda')

            feat_test = G(test_data_tensor)
            output_test_C1, output_test_prev_C1= C1(feat_test)
            output_test_C2, output_test_prev_C2= C2(feat_test)
            output_test= (output_test_C1 + output_test_C2)/2

            #Check entropy
            entropy = -torch.sum(output_test * torch.log2(output_test), dim=1)
            _idx_to_monitor=torch.where(entropy>=0.9)

            #Take samples to monitor
            entropy_to_monitor=entropy[_idx_to_monitor]
            test_data_tensor_to_monitor=test_data_tensor[_idx_to_monitor]
            test_data_orig_tensor_to_monitor=test_data_orig_tensor[_idx_to_monitor]
            test_labels_tensor_to_monitor=test_labels_tensor[_idx_to_monitor]
            output_test_prev_C1_to_monitor=output_test_prev_C1[_idx_to_monitor]
            output_test_prev_C2_to_monitor=output_test_prev_C2[_idx_to_monitor]
            output_test_to_monitor=output_test[_idx_to_monitor]

            #Take samples no monitoring needed
            mask = torch.ones_like(entropy, dtype=torch.bool)
            mask[_idx_to_monitor] = False
            entropy_monitor_no = entropy[mask]
            test_data_tensor_monitor_no=test_data_tensor[mask]
            test_labels_tensor_monitor_no=test_labels_tensor[mask]
            output_test_monitor_no=output_test[mask]

            #Augmentation
            count_aug=0
            if test_data_tensor_to_monitor.shape[0]>0:  #Augmentation may require
                #print('Samples to Monitor Step-1')
                #print(test_data_tensor_to_monitor.shape)

                #Check classifier discrepancy
                _idx_to_monitor_cd=[]
                for ii in range(output_test_prev_C1_to_monitor.shape[0]):
                    # # 简化
                    # if L2Distance(output_test_prev_C1_to_monitor[ii],output_test_prev_C2_to_monitor[ii]) > 0:
                    if L2Distance(output_test_prev_C1_to_monitor[ii],output_test_prev_C2_to_monitor[ii]) > (mean_dist_classifier_s):
                        _idx_to_monitor_cd.append(ii)

                #Update samples to monitor
                entropy_to_monitor_final=entropy_to_monitor[_idx_to_monitor_cd]
                test_data_tensor_to_monitor_final=test_data_tensor_to_monitor[_idx_to_monitor_cd]
                test_data_orig_tensor_to_monitor=test_data_orig_tensor_to_monitor[_idx_to_monitor_cd]
                output_test_to_monitor_final=output_test_to_monitor[_idx_to_monitor_cd]
                test_labels_tensor_to_monitor_final=test_labels_tensor_to_monitor[_idx_to_monitor_cd]
                pred_to_monitor = output_test_to_monitor_final.data.max(1)[1]

                #Update samples no monitoring needed
                mask = torch.ones_like(entropy_to_monitor, dtype=torch.bool)
                mask[_idx_to_monitor_cd] = False
                entropy_cd_monitor_no = entropy_to_monitor[mask]
                test_data_tensor_cd_monitor_no=test_data_tensor_to_monitor[mask]
                test_labels_tensor_cd_monitor_no=test_labels_tensor_to_monitor[mask]
                output_test_cd_monitor_no=output_test_to_monitor[mask]

                entropy_monitor_no=torch.concatenate((entropy_monitor_no,entropy_cd_monitor_no), axis=0)
                test_data_tensor_monitor_no=torch.concatenate((test_data_tensor_monitor_no,test_data_tensor_cd_monitor_no), axis=0)
                test_labels_tensor_monitor_no=torch.concatenate((test_labels_tensor_monitor_no,test_labels_tensor_cd_monitor_no), axis=0)
                output_test_monitor_no=torch.concatenate((output_test_monitor_no,output_test_cd_monitor_no), axis=0)

                if test_data_tensor_to_monitor_final.shape[0]>0:  #Augmentation required
                    #print('Samples to Monitor Step-2')
                    #print(test_data_tensor_to_monitor.shape)
                    _idx_to_monitor=_idx_to_monitor_cd

                    #Augment Test Data
                    sampling_rate_DEAP=128
                    fs_DEAP=128
                    #5 frequency bands
                    fStart = [1,4,8,14,31]
                    fEnd = [4,8,14,31,50]

                    #Augmentation (Gaussian noise addition)
                    std_curr=0.1
                    test_data_orig_tensor_noisy_1=add_gaussian_noise(test_data_orig_tensor_to_monitor,mean=0,std=std_curr)
                    std_curr=0.5
                    test_data_orig_tensor_noisy_2=add_gaussian_noise(test_data_orig_tensor_to_monitor,mean=0,std=std_curr)
                    std_curr=0.9
                    test_data_orig_tensor_noisy_3=add_gaussian_noise(test_data_orig_tensor_to_monitor,mean=0,std=std_curr)

                    #Augmentation (resampling)
                    new_fs_DEAP=256
                    test_data_orig_tensor_noisy_4=signal.resample_poly(test_data_orig_tensor_to_monitor.detach().cpu().data.numpy(), new_fs_DEAP, fs_DEAP, axis=2)
                    test_data_orig_tensor_noisy_4 = torch.from_numpy(test_data_orig_tensor_noisy_4)
                    test_data_orig_tensor_noisy_4=test_data_orig_tensor_noisy_4.to('cuda')
                    new_fs_DEAP=384
                    test_data_orig_tensor_noisy_5=signal.resample_poly(test_data_orig_tensor_to_monitor.detach().cpu().data.numpy(), new_fs_DEAP, fs_DEAP, axis=2)
                    test_data_orig_tensor_noisy_5 = torch.from_numpy(test_data_orig_tensor_noisy_5)
                    test_data_orig_tensor_noisy_5=test_data_orig_tensor_noisy_5.to('cuda')
                    count_=0

                    for jj in range(test_data_orig_tensor_to_monitor.shape[0]):
                        tmp_PSD_1,tmp_DE_1=Calculate_PSD_DE(test_data_orig_tensor_noisy_1[jj],sampling_rate_DEAP,fs_DEAP,fStart,fEnd)
                        tmp_PSD_1=tmp_PSD_1.reshape(1,tmp_PSD_1.shape[0])
                        tmp_PSD_2,tmp_DE_2=Calculate_PSD_DE(test_data_orig_tensor_noisy_2[jj],sampling_rate_DEAP,fs_DEAP,fStart,fEnd)
                        tmp_PSD_2=tmp_PSD_2.reshape(1,tmp_PSD_2.shape[0])
                        tmp_PSD_3,tmp_DE_3=Calculate_PSD_DE(test_data_orig_tensor_noisy_3[jj],sampling_rate_DEAP,fs_DEAP,fStart,fEnd)
                        tmp_PSD_3=tmp_PSD_3.reshape(1,tmp_PSD_3.shape[0])

                        new_fs_DEAP=256
                        tmp_PSD_4,tmp_DE_4=Calculate_PSD_DE(test_data_orig_tensor_noisy_4[jj],new_fs_DEAP,new_fs_DEAP,fStart,fEnd)
                        tmp_PSD_4=tmp_PSD_4.reshape(1,tmp_PSD_4.shape[0])
                        new_fs_DEAP=384
                        tmp_PSD_5,tmp_DE_5=Calculate_PSD_DE(test_data_orig_tensor_noisy_5[jj],new_fs_DEAP,new_fs_DEAP,fStart,fEnd)
                        tmp_PSD_5=tmp_PSD_5.reshape(1,tmp_PSD_5.shape[0])

                        if count_==0:
                            all_PSD_noisy_1=tmp_PSD_1
                            all_PSD_noisy_2=tmp_PSD_2
                            all_PSD_noisy_3=tmp_PSD_3
                            all_PSD_noisy_4=tmp_PSD_4
                            all_PSD_noisy_5=tmp_PSD_5
                        else:
                            all_PSD_noisy_1=torch.concatenate((all_PSD_noisy_1,tmp_PSD_1), axis=0)
                            all_PSD_noisy_2=torch.concatenate((all_PSD_noisy_2,tmp_PSD_2), axis=0)
                            all_PSD_noisy_3=torch.concatenate((all_PSD_noisy_3,tmp_PSD_3), axis=0)
                            all_PSD_noisy_4=torch.concatenate((all_PSD_noisy_4,tmp_PSD_4), axis=0)
                            all_PSD_noisy_5=torch.concatenate((all_PSD_noisy_5,tmp_PSD_5), axis=0)

                        count_=count_+1

                    tmp_noisy_1=torch.zeros((all_PSD_noisy_1.shape[0],all_PSD_noisy_1.shape[1]))
                    tmp_noisy_2=torch.zeros((all_PSD_noisy_2.shape[0],all_PSD_noisy_2.shape[1]))
                    tmp_noisy_3=torch.zeros((all_PSD_noisy_3.shape[0],all_PSD_noisy_3.shape[1]))
                    tmp_noisy_4=torch.zeros((all_PSD_noisy_4.shape[0],all_PSD_noisy_4.shape[1]))
                    tmp_noisy_5=torch.zeros((all_PSD_noisy_5.shape[0],all_PSD_noisy_5.shape[1]))

                    for i in range (all_PSD_noisy_1.shape[0]):
                        tmp_noisy_1[i]=(((all_PSD_noisy_1[i]-torch.min(all_PSD_noisy_1[i]))/(torch.max(all_PSD_noisy_1[i])-torch.min(all_PSD_noisy_1[i])))*2)-1
                        tmp_noisy_2[i]=(((all_PSD_noisy_2[i]-torch.min(all_PSD_noisy_2[i]))/(torch.max(all_PSD_noisy_2[i])-torch.min(all_PSD_noisy_2[i])))*2)-1
                        tmp_noisy_3[i]=(((all_PSD_noisy_3[i]-torch.min(all_PSD_noisy_3[i]))/(torch.max(all_PSD_noisy_3[i])-torch.min(all_PSD_noisy_3[i])))*2)-1
                        tmp_noisy_4[i]=(((all_PSD_noisy_4[i]-torch.min(all_PSD_noisy_4[i]))/(torch.max(all_PSD_noisy_4[i])-torch.min(all_PSD_noisy_4[i])))*2)-1
                        tmp_noisy_5[i]=(((all_PSD_noisy_5[i]-torch.min(all_PSD_noisy_5[i]))/(torch.max(all_PSD_noisy_5[i])-torch.min(all_PSD_noisy_5[i])))*2)-1

                    all_PSD_noisy_1=tmp_noisy_1
                    all_PSD_noisy_2=tmp_noisy_2
                    all_PSD_noisy_3=tmp_noisy_3
                    all_PSD_noisy_4=tmp_noisy_4
                    all_PSD_noisy_5=tmp_noisy_5

                    test_data_tensor_noisy_1=torch.reshape(all_PSD_noisy_1,(all_PSD_noisy_1.shape[0],all_PSD_noisy_1.shape[1],1))
                    test_data_tensor_noisy_2=torch.reshape(all_PSD_noisy_2,(all_PSD_noisy_2.shape[0],all_PSD_noisy_2.shape[1],1))
                    test_data_tensor_noisy_3=torch.reshape(all_PSD_noisy_3,(all_PSD_noisy_3.shape[0],all_PSD_noisy_3.shape[1],1))
                    test_data_tensor_noisy_4=torch.reshape(all_PSD_noisy_4,(all_PSD_noisy_4.shape[0],all_PSD_noisy_4.shape[1],1))
                    test_data_tensor_noisy_5=torch.reshape(all_PSD_noisy_5,(all_PSD_noisy_5.shape[0],all_PSD_noisy_5.shape[1],1))
                    test_data_tensor_noisy_1=torch.swapaxes(test_data_tensor_noisy_1, 1, 2)
                    test_data_tensor_noisy_2=torch.swapaxes(test_data_tensor_noisy_2, 1, 2)
                    test_data_tensor_noisy_3=torch.swapaxes(test_data_tensor_noisy_3, 1, 2)
                    test_data_tensor_noisy_4=torch.swapaxes(test_data_tensor_noisy_4, 1, 2)
                    test_data_tensor_noisy_5=torch.swapaxes(test_data_tensor_noisy_5, 1, 2)
                    test_data_tensor_noisy_1=test_data_tensor_noisy_1.to('cuda')
                    test_data_tensor_noisy_2=test_data_tensor_noisy_2.to('cuda')
                    test_data_tensor_noisy_3=test_data_tensor_noisy_3.to('cuda')
                    test_data_tensor_noisy_4=test_data_tensor_noisy_4.to('cuda')
                    test_data_tensor_noisy_5=test_data_tensor_noisy_5.to('cuda')

                    feat_test_noisy_1 = G(test_data_tensor_noisy_1)
                    output_test_C1_noisy_1, output_test_prev_C1_noisy_1= C1(feat_test_noisy_1)
                    output_test_C2_noisy_1, output_test_prev_C2_noisy_1= C2(feat_test_noisy_1)
                    output_test_noisy_1= (output_test_C1_noisy_1 + output_test_C2_noisy_1)/2

                    feat_test_noisy_2 = G(test_data_tensor_noisy_2)
                    output_test_C1_noisy_2, output_test_prev_C1_noisy_2= C1(feat_test_noisy_2)
                    output_test_C2_noisy_2, output_test_prev_C2_noisy_2= C2(feat_test_noisy_2)
                    output_test_noisy_2= (output_test_C1_noisy_2 + output_test_C2_noisy_2)/2

                    feat_test_noisy_3 = G(test_data_tensor_noisy_3)
                    output_test_C1_noisy_3, output_test_prev_C1_noisy_3= C1(feat_test_noisy_3)
                    output_test_C2_noisy_3, output_test_prev_C2_noisy_3= C2(feat_test_noisy_3)
                    output_test_noisy_3= (output_test_C1_noisy_3 + output_test_C2_noisy_3)/2

                    feat_test_noisy_4 = G(test_data_tensor_noisy_4)
                    output_test_C1_noisy_4, output_test_prev_C1_noisy_4= C1(feat_test_noisy_4)
                    output_test_C2_noisy_4, output_test_prev_C2_noisy_4= C2(feat_test_noisy_4)
                    output_test_noisy_4= (output_test_C1_noisy_4 + output_test_C2_noisy_4)/2

                    feat_test_noisy_5 = G(test_data_tensor_noisy_5)
                    output_test_C1_noisy_5, output_test_prev_C1_noisy_5= C1(feat_test_noisy_5)
                    output_test_C2_noisy_5, output_test_prev_C2_noisy_5= C2(feat_test_noisy_5)
                    output_test_noisy_5= (output_test_C1_noisy_5 + output_test_C2_noisy_5)/2


                    pred_noisy_1 = output_test_noisy_1.data.max(1)[1]
                    pred_noisy_2 = output_test_noisy_2.data.max(1)[1]
                    pred_noisy_3 = output_test_noisy_3.data.max(1)[1]
                    pred_noisy_4 = output_test_noisy_4.data.max(1)[1]
                    pred_noisy_5 = output_test_noisy_5.data.max(1)[1]

                    #Voting
                    all_tensors = torch.stack((pred_to_monitor, pred_noisy_1, pred_noisy_2, pred_noisy_3, pred_noisy_4, pred_noisy_5), dim=0)

                    pred_final_to_monitor, _ = torch.mode(all_tensors, dim=0)
                    test_labels_tensor_to_monitor_final = test_labels_tensor_to_monitor_final.float()
                    pred_final_monitor_no = output_test_monitor_no.data.max(1)[1]
                    label_to_monitor = test_labels_tensor_to_monitor_final.data.max(1)[1]
                    label_monitor_no = test_labels_tensor_monitor_no.data.max(1)[1]
                    pred_cmbd = torch.cat([pred_final_monitor_no, pred_final_to_monitor])
                    label_cmbd = torch.cat([label_monitor_no, label_to_monitor])
                    pred_cmbd_np=pred_cmbd.detach().cpu().data.numpy()
                    label_cmbd_np=label_cmbd.detach().cpu().data.numpy()

                    all_preds.append(pred_cmbd_np)
                    all_true_labels.append(label_cmbd_np)
                    preds_Each_Sub.append(pred_cmbd_np)
                    true_labels_Each_Sub.append(label_cmbd_np)

                    correct1= pred_cmbd.eq(label_cmbd.data).cpu().sum()
                    size=pred_cmbd.shape[0]
                    curr_acc=correct1/size*100
                    all_acc.append(curr_acc)
                    acc_Each_Sub.append(curr_acc)
                    count_aug=count_aug+1
                    #print('Augmentation Done')

            if count_aug==0:  #If no candidates for augmentation, take results for the original samples
                test_labels_tensor = test_labels_tensor.float()
                test_loss_curr = criterion(output_test, test_labels_tensor)

                pred1 = output_test.data.max(1)[1]
                label1 = test_labels_tensor.data.max(1)[1]
                preds_np=pred1.detach().cpu().data.numpy()
                ture_labels_np=label1.detach().cpu().data.numpy()
                all_preds.append(preds_np)
                all_true_labels.append(ture_labels_np)
                preds_Each_Sub.append(preds_np)
                true_labels_Each_Sub.append(ture_labels_np)
                correct1= pred1.eq(label1.data).cpu().sum()
                size=pred1.shape[0]
                curr_acc=correct1/size*100
                all_acc.append(curr_acc)
                acc_Each_Sub.append(curr_acc)
                #print(curr_acc)

        all_acc_Sub.append(acc_Each_Sub)
        all_preds_Sub.append(preds_Each_Sub)
        all_true_labels_Sub.append(true_labels_Each_Sub)
        #print('End of Sub-'+sub)
elif mode == 'cross_subject':
    #Test on SEED
    subjectList_test = subjectList_SEED_test
    data_P_path = './data/SEED/preprocessed/'
    
    for sub in subjectList_test:
        DE_tmp=np.load(data_P_path+'DE_sub_'+sub+'.npy')
        PSD_tmp=np.load(data_P_path+'PSD_sub_'+sub+'.npy')
        data_orig_tmp=np.load(data_P_path+'Data_Orig_sub_'+sub+'.npy')
        labels_tmp=np.load(data_P_path+'labels_sub_'+sub+'.npy')
        labels_dis_all = deepcopy(labels_tmp)

        # SEED标签处理方案1
        # 按load_P_SEED相同方法处理标签
        # condition1 = labels_tmp == 0
        # condition2 = labels_tmp == 1
        # indices = np.where(np.logical_or(condition1, condition2))
        # labels_dis_all = labels_dis_all[indices[0],:]
        # DE_tmp = DE_tmp[indices[0],:]
        # PSD_tmp = PSD_tmp[indices[0],:]
        # data_orig_tmp = data_orig_tmp[indices[0],:]

        # SEED标签处理方案2
        # 去掉中立
        condition1 = labels_tmp == 0
        condition2 = labels_tmp == 2
        indices = np.where(np.logical_or(condition1, condition2))
        labels_dis_all = labels_dis_all[indices[0],:]
        DE_tmp = DE_tmp[indices[0],:]
        PSD_tmp = PSD_tmp[indices[0],:]
        data_orig_tmp = data_orig_tmp[indices[0],:]
        for i in range(labels_dis_all.shape[0]):
            if labels_dis_all[i][0]==2:
                labels_dis_all[i][0]=1
            elif labels_dis_all[i][0]==0:
                labels_dis_all[i][0]=0
            else:
                print(f'Error in labels_dis_all[{i}][0] processing, except 0 or 2, get {labels_dis_all[i][0]}')

        tmp=np.zeros((PSD_tmp.shape[0],PSD_tmp.shape[1]))
        for i in range (PSD_tmp.shape[0]):
            tmp[i]=(((PSD_tmp[i]-np.amin(PSD_tmp[i]))/(np.amax(PSD_tmp[i])-np.amin(PSD_tmp[i])))*2)-1
        PSD_tmp=tmp
        PSD_tmp=np.reshape(PSD_tmp,(PSD_tmp.shape[0],PSD_tmp.shape[1],1))
        labels_dis_all = labels_dis_all.ravel()
        test_data_batch_t, test_data_orig_batch_t, test_label_batch_t= create_mini_batches_T(PSD_tmp, data_orig_tmp, labels_dis_all, batch_size)

        acc_Each_Sub=[]
        preds_Each_Sub=[]
        true_labels_Each_Sub=[]

        for i in tqdm(range(int(test_data_batch_t.shape[0])), desc='测试阶段：', unit='batch'):
            test_data_tensor = torch.from_numpy(test_data_batch_t[i])
            test_data_tensor=test_data_tensor.to('cuda')
            test_data_tensor=test_data_tensor.float()
            test_data_orig_tensor = torch.from_numpy(test_data_orig_batch_t[i])
            test_data_orig_tensor=test_data_orig_tensor.to('cuda')
            test_data_orig_tensor=test_data_orig_tensor.float()
            test_data_tensor=torch.swapaxes(test_data_tensor, 1, 2)
            test_labels_tensor = torch.from_numpy(test_label_batch_t[i])
            test_labels_tensor=torch.nn.functional.one_hot(test_labels_tensor, num_classes)
            test_labels_tensor=test_labels_tensor.to('cuda')

            feat_test = G(test_data_tensor)
            output_test_C1, output_test_prev_C1= C1(feat_test)
            output_test_C2, output_test_prev_C2= C2(feat_test)
            output_test= (output_test_C1 + output_test_C2)/2

            #Check entropy
            entropy = -torch.sum(output_test * torch.log2(output_test), dim=1)
            _idx_to_monitor=torch.where(entropy>=0.9)

            #Take samples to monitor
            entropy_to_monitor=entropy[_idx_to_monitor]
            test_data_tensor_to_monitor=test_data_tensor[_idx_to_monitor]
            test_data_orig_tensor_to_monitor=test_data_orig_tensor[_idx_to_monitor]
            test_labels_tensor_to_monitor=test_labels_tensor[_idx_to_monitor]
            output_test_prev_C1_to_monitor=output_test_prev_C1[_idx_to_monitor]
            output_test_prev_C2_to_monitor=output_test_prev_C2[_idx_to_monitor]
            output_test_to_monitor=output_test[_idx_to_monitor]

            #Take samples no monitoring needed
            mask = torch.ones_like(entropy, dtype=torch.bool)
            mask[_idx_to_monitor] = False
            entropy_monitor_no = entropy[mask]
            test_data_tensor_monitor_no=test_data_tensor[mask]
            test_labels_tensor_monitor_no=test_labels_tensor[mask]
            output_test_monitor_no=output_test[mask]

            #Augmentation
            count_aug=0
            if test_data_tensor_to_monitor.shape[0]>0:  #Augmentation may require
                #print('Samples to Monitor Step-1')
                #print(test_data_tensor_to_monitor.shape)

                #Check classifier discrepancy
                _idx_to_monitor_cd=[]
                for ii in range(output_test_prev_C1_to_monitor.shape[0]):
                    # # 简化
                    # if L2Distance(output_test_prev_C1_to_monitor[ii],output_test_prev_C2_to_monitor[ii]) > 0:
                    if L2Distance(output_test_prev_C1_to_monitor[ii],output_test_prev_C2_to_monitor[ii]) > (mean_dist_classifier_s):
                        _idx_to_monitor_cd.append(ii)

                #Update samples to monitor
                entropy_to_monitor_final=entropy_to_monitor[_idx_to_monitor_cd]
                test_data_tensor_to_monitor_final=test_data_tensor_to_monitor[_idx_to_monitor_cd]
                test_data_orig_tensor_to_monitor=test_data_orig_tensor_to_monitor[_idx_to_monitor_cd]
                output_test_to_monitor_final=output_test_to_monitor[_idx_to_monitor_cd]
                test_labels_tensor_to_monitor_final=test_labels_tensor_to_monitor[_idx_to_monitor_cd]
                pred_to_monitor = output_test_to_monitor_final.data.max(1)[1]

                #Update samples no monitoring needed
                mask = torch.ones_like(entropy_to_monitor, dtype=torch.bool)
                mask[_idx_to_monitor_cd] = False
                entropy_cd_monitor_no = entropy_to_monitor[mask]
                test_data_tensor_cd_monitor_no=test_data_tensor_to_monitor[mask]
                test_labels_tensor_cd_monitor_no=test_labels_tensor_to_monitor[mask]
                output_test_cd_monitor_no=output_test_to_monitor[mask]

                entropy_monitor_no=torch.concatenate((entropy_monitor_no,entropy_cd_monitor_no), axis=0)
                test_data_tensor_monitor_no=torch.concatenate((test_data_tensor_monitor_no,test_data_tensor_cd_monitor_no), axis=0)
                test_labels_tensor_monitor_no=torch.concatenate((test_labels_tensor_monitor_no,test_labels_tensor_cd_monitor_no), axis=0)
                output_test_monitor_no=torch.concatenate((output_test_monitor_no,output_test_cd_monitor_no), axis=0)

                if test_data_tensor_to_monitor_final.shape[0]>0:  #Augmentation required
                    #print('Samples to Monitor Step-2')
                    #print(test_data_tensor_to_monitor.shape)
                    _idx_to_monitor=_idx_to_monitor_cd

                    #Augment Test Data
                    sampling_rate_DEAP=200
                    fs_DEAP=200
                    #5 frequency bands
                    fStart = [1,4,8,14,31]
                    fEnd = [4,8,14,31,50]

                    #Augmentation (Gaussian noise addition)
                    std_curr=0.1
                    test_data_orig_tensor_noisy_1=add_gaussian_noise(test_data_orig_tensor_to_monitor,mean=0,std=std_curr)
                    std_curr=0.5
                    test_data_orig_tensor_noisy_2=add_gaussian_noise(test_data_orig_tensor_to_monitor,mean=0,std=std_curr)
                    std_curr=0.9
                    test_data_orig_tensor_noisy_3=add_gaussian_noise(test_data_orig_tensor_to_monitor,mean=0,std=std_curr)

                    #Augmentation (resampling)
                    new_fs_DEAP=256
                    test_data_orig_tensor_noisy_4=signal.resample_poly(test_data_orig_tensor_to_monitor.detach().cpu().data.numpy(), new_fs_DEAP, fs_DEAP, axis=2)
                    test_data_orig_tensor_noisy_4 = torch.from_numpy(test_data_orig_tensor_noisy_4)
                    test_data_orig_tensor_noisy_4=test_data_orig_tensor_noisy_4.to('cuda')
                    new_fs_DEAP=384
                    test_data_orig_tensor_noisy_5=signal.resample_poly(test_data_orig_tensor_to_monitor.detach().cpu().data.numpy(), new_fs_DEAP, fs_DEAP, axis=2)
                    test_data_orig_tensor_noisy_5 = torch.from_numpy(test_data_orig_tensor_noisy_5)
                    test_data_orig_tensor_noisy_5=test_data_orig_tensor_noisy_5.to('cuda')
                    count_=0

                    for jj in range(test_data_orig_tensor_to_monitor.shape[0]):
                        tmp_PSD_1,tmp_DE_1=Calculate_PSD_DE(test_data_orig_tensor_noisy_1[jj],sampling_rate_DEAP,fs_DEAP,fStart,fEnd)
                        tmp_PSD_1=tmp_PSD_1.reshape(1,tmp_PSD_1.shape[0])
                        tmp_PSD_2,tmp_DE_2=Calculate_PSD_DE(test_data_orig_tensor_noisy_2[jj],sampling_rate_DEAP,fs_DEAP,fStart,fEnd)
                        tmp_PSD_2=tmp_PSD_2.reshape(1,tmp_PSD_2.shape[0])
                        tmp_PSD_3,tmp_DE_3=Calculate_PSD_DE(test_data_orig_tensor_noisy_3[jj],sampling_rate_DEAP,fs_DEAP,fStart,fEnd)
                        tmp_PSD_3=tmp_PSD_3.reshape(1,tmp_PSD_3.shape[0])

                        new_fs_DEAP=256
                        tmp_PSD_4,tmp_DE_4=Calculate_PSD_DE(test_data_orig_tensor_noisy_4[jj],new_fs_DEAP,new_fs_DEAP,fStart,fEnd)
                        tmp_PSD_4=tmp_PSD_4.reshape(1,tmp_PSD_4.shape[0])
                        new_fs_DEAP=384
                        tmp_PSD_5,tmp_DE_5=Calculate_PSD_DE(test_data_orig_tensor_noisy_5[jj],new_fs_DEAP,new_fs_DEAP,fStart,fEnd)
                        tmp_PSD_5=tmp_PSD_5.reshape(1,tmp_PSD_5.shape[0])

                        if count_==0:
                            all_PSD_noisy_1=tmp_PSD_1
                            all_PSD_noisy_2=tmp_PSD_2
                            all_PSD_noisy_3=tmp_PSD_3
                            all_PSD_noisy_4=tmp_PSD_4
                            all_PSD_noisy_5=tmp_PSD_5
                        else:
                            all_PSD_noisy_1=torch.concatenate((all_PSD_noisy_1,tmp_PSD_1), axis=0)
                            all_PSD_noisy_2=torch.concatenate((all_PSD_noisy_2,tmp_PSD_2), axis=0)
                            all_PSD_noisy_3=torch.concatenate((all_PSD_noisy_3,tmp_PSD_3), axis=0)
                            all_PSD_noisy_4=torch.concatenate((all_PSD_noisy_4,tmp_PSD_4), axis=0)
                            all_PSD_noisy_5=torch.concatenate((all_PSD_noisy_5,tmp_PSD_5), axis=0)

                        count_=count_+1

                    tmp_noisy_1=torch.zeros((all_PSD_noisy_1.shape[0],all_PSD_noisy_1.shape[1]))
                    tmp_noisy_2=torch.zeros((all_PSD_noisy_2.shape[0],all_PSD_noisy_2.shape[1]))
                    tmp_noisy_3=torch.zeros((all_PSD_noisy_3.shape[0],all_PSD_noisy_3.shape[1]))
                    tmp_noisy_4=torch.zeros((all_PSD_noisy_4.shape[0],all_PSD_noisy_4.shape[1]))
                    tmp_noisy_5=torch.zeros((all_PSD_noisy_5.shape[0],all_PSD_noisy_5.shape[1]))

                    for i in range (all_PSD_noisy_1.shape[0]):
                        tmp_noisy_1[i]=(((all_PSD_noisy_1[i]-torch.min(all_PSD_noisy_1[i]))/(torch.max(all_PSD_noisy_1[i])-torch.min(all_PSD_noisy_1[i])))*2)-1
                        tmp_noisy_2[i]=(((all_PSD_noisy_2[i]-torch.min(all_PSD_noisy_2[i]))/(torch.max(all_PSD_noisy_2[i])-torch.min(all_PSD_noisy_2[i])))*2)-1
                        tmp_noisy_3[i]=(((all_PSD_noisy_3[i]-torch.min(all_PSD_noisy_3[i]))/(torch.max(all_PSD_noisy_3[i])-torch.min(all_PSD_noisy_3[i])))*2)-1
                        tmp_noisy_4[i]=(((all_PSD_noisy_4[i]-torch.min(all_PSD_noisy_4[i]))/(torch.max(all_PSD_noisy_4[i])-torch.min(all_PSD_noisy_4[i])))*2)-1
                        tmp_noisy_5[i]=(((all_PSD_noisy_5[i]-torch.min(all_PSD_noisy_5[i]))/(torch.max(all_PSD_noisy_5[i])-torch.min(all_PSD_noisy_5[i])))*2)-1

                    all_PSD_noisy_1=tmp_noisy_1
                    all_PSD_noisy_2=tmp_noisy_2
                    all_PSD_noisy_3=tmp_noisy_3
                    all_PSD_noisy_4=tmp_noisy_4
                    all_PSD_noisy_5=tmp_noisy_5

                    test_data_tensor_noisy_1=torch.reshape(all_PSD_noisy_1,(all_PSD_noisy_1.shape[0],all_PSD_noisy_1.shape[1],1))
                    test_data_tensor_noisy_2=torch.reshape(all_PSD_noisy_2,(all_PSD_noisy_2.shape[0],all_PSD_noisy_2.shape[1],1))
                    test_data_tensor_noisy_3=torch.reshape(all_PSD_noisy_3,(all_PSD_noisy_3.shape[0],all_PSD_noisy_3.shape[1],1))
                    test_data_tensor_noisy_4=torch.reshape(all_PSD_noisy_4,(all_PSD_noisy_4.shape[0],all_PSD_noisy_4.shape[1],1))
                    test_data_tensor_noisy_5=torch.reshape(all_PSD_noisy_5,(all_PSD_noisy_5.shape[0],all_PSD_noisy_5.shape[1],1))
                    test_data_tensor_noisy_1=torch.swapaxes(test_data_tensor_noisy_1, 1, 2)
                    test_data_tensor_noisy_2=torch.swapaxes(test_data_tensor_noisy_2, 1, 2)
                    test_data_tensor_noisy_3=torch.swapaxes(test_data_tensor_noisy_3, 1, 2)
                    test_data_tensor_noisy_4=torch.swapaxes(test_data_tensor_noisy_4, 1, 2)
                    test_data_tensor_noisy_5=torch.swapaxes(test_data_tensor_noisy_5, 1, 2)
                    test_data_tensor_noisy_1=test_data_tensor_noisy_1.to('cuda')
                    test_data_tensor_noisy_2=test_data_tensor_noisy_2.to('cuda')
                    test_data_tensor_noisy_3=test_data_tensor_noisy_3.to('cuda')
                    test_data_tensor_noisy_4=test_data_tensor_noisy_4.to('cuda')
                    test_data_tensor_noisy_5=test_data_tensor_noisy_5.to('cuda')

                    feat_test_noisy_1 = G(test_data_tensor_noisy_1)
                    output_test_C1_noisy_1, output_test_prev_C1_noisy_1= C1(feat_test_noisy_1)
                    output_test_C2_noisy_1, output_test_prev_C2_noisy_1= C2(feat_test_noisy_1)
                    output_test_noisy_1= (output_test_C1_noisy_1 + output_test_C2_noisy_1)/2

                    feat_test_noisy_2 = G(test_data_tensor_noisy_2)
                    output_test_C1_noisy_2, output_test_prev_C1_noisy_2= C1(feat_test_noisy_2)
                    output_test_C2_noisy_2, output_test_prev_C2_noisy_2= C2(feat_test_noisy_2)
                    output_test_noisy_2= (output_test_C1_noisy_2 + output_test_C2_noisy_2)/2

                    feat_test_noisy_3 = G(test_data_tensor_noisy_3)
                    output_test_C1_noisy_3, output_test_prev_C1_noisy_3= C1(feat_test_noisy_3)
                    output_test_C2_noisy_3, output_test_prev_C2_noisy_3= C2(feat_test_noisy_3)
                    output_test_noisy_3= (output_test_C1_noisy_3 + output_test_C2_noisy_3)/2

                    feat_test_noisy_4 = G(test_data_tensor_noisy_4)
                    output_test_C1_noisy_4, output_test_prev_C1_noisy_4= C1(feat_test_noisy_4)
                    output_test_C2_noisy_4, output_test_prev_C2_noisy_4= C2(feat_test_noisy_4)
                    output_test_noisy_4= (output_test_C1_noisy_4 + output_test_C2_noisy_4)/2

                    feat_test_noisy_5 = G(test_data_tensor_noisy_5)
                    output_test_C1_noisy_5, output_test_prev_C1_noisy_5= C1(feat_test_noisy_5)
                    output_test_C2_noisy_5, output_test_prev_C2_noisy_5= C2(feat_test_noisy_5)
                    output_test_noisy_5= (output_test_C1_noisy_5 + output_test_C2_noisy_5)/2


                    pred_noisy_1 = output_test_noisy_1.data.max(1)[1]
                    pred_noisy_2 = output_test_noisy_2.data.max(1)[1]
                    pred_noisy_3 = output_test_noisy_3.data.max(1)[1]
                    pred_noisy_4 = output_test_noisy_4.data.max(1)[1]
                    pred_noisy_5 = output_test_noisy_5.data.max(1)[1]

                    #Voting
                    all_tensors = torch.stack((pred_to_monitor, pred_noisy_1, pred_noisy_2, pred_noisy_3, pred_noisy_4, pred_noisy_5), dim=0)

                    pred_final_to_monitor, _ = torch.mode(all_tensors, dim=0)
                    test_labels_tensor_to_monitor_final = test_labels_tensor_to_monitor_final.float()
                    pred_final_monitor_no = output_test_monitor_no.data.max(1)[1]
                    label_to_monitor = test_labels_tensor_to_monitor_final.data.max(1)[1]
                    label_monitor_no = test_labels_tensor_monitor_no.data.max(1)[1]
                    pred_cmbd = torch.cat([pred_final_monitor_no, pred_final_to_monitor])
                    label_cmbd = torch.cat([label_monitor_no, label_to_monitor])
                    pred_cmbd_np=pred_cmbd.detach().cpu().data.numpy()
                    label_cmbd_np=label_cmbd.detach().cpu().data.numpy()

                    all_preds.append(pred_cmbd_np)
                    all_true_labels.append(label_cmbd_np)
                    preds_Each_Sub.append(pred_cmbd_np)
                    true_labels_Each_Sub.append(label_cmbd_np)

                    correct1= pred_cmbd.eq(label_cmbd.data).cpu().sum()
                    size=pred_cmbd.shape[0]
                    curr_acc=correct1/size*100
                    all_acc.append(curr_acc)
                    acc_Each_Sub.append(curr_acc)
                    count_aug=count_aug+1
                    #print('Augmentation Done')

            if count_aug==0:  #If no candidates for augmentation, take results for the original samples
                test_labels_tensor = test_labels_tensor.float()
                test_loss_curr = criterion(output_test, test_labels_tensor)

                pred1 = output_test.data.max(1)[1]
                label1 = test_labels_tensor.data.max(1)[1]
                preds_np=pred1.detach().cpu().data.numpy()
                ture_labels_np=label1.detach().cpu().data.numpy()
                all_preds.append(preds_np)
                all_true_labels.append(ture_labels_np)
                preds_Each_Sub.append(preds_np)
                true_labels_Each_Sub.append(ture_labels_np)
                correct1= pred1.eq(label1.data).cpu().sum()
                size=pred1.shape[0]
                curr_acc=correct1/size*100
                all_acc.append(curr_acc)
                acc_Each_Sub.append(curr_acc)
                #print(curr_acc)

        all_acc_Sub.append(acc_Each_Sub)
        all_preds_Sub.append(preds_Each_Sub)
        all_true_labels_Sub.append(true_labels_Each_Sub)
        #print('End of Sub-'+sub)
# 记录训练阶段耗时
test_time = time.time() - start_time

#Calculate test accuracy
test_acc_all=np.zeros((len(all_acc_Sub)))
c1=0
for i in range(len(all_acc_Sub)):
      test_acc_Sub=0
      c2=0
      for j in range(len(all_acc_Sub[i])):
          test_acc_Sub = test_acc_Sub + all_acc_Sub[i][j]
          c2=c2+1
      avg_acc_Sub = test_acc_Sub / c2
      test_acc_all[i]=round(avg_acc_Sub.item(),2)
      c1=c1+1
      print('Accuracy Sub-'+str(i+1))
      print(round(avg_acc_Sub.item(),2))

avg_test_acc=round(np.mean(test_acc_all),2)
print('Average Test Accuracy')
print(avg_test_acc)
print(f"PRE_EPOCHS = {PRE_EPOCHS}, CLU_EPOCHS = {CLU_EPOCHS}, ADP_EPOCHS = {ADP_EPOCHS}")
print(f"PRE_LR = {PRE_LR}, CLU_LR = {CLU_LR}, ADP_LR = {ADP_LR}")

#Classification Report
all_preds = np.array(all_preds)
all_true_labels = np.array(all_true_labels)
all_preds=np.reshape(all_preds,-1)
all_true_labels=np.reshape(all_true_labels,-1)
cf_report= classification_report(all_true_labels, all_preds, digits=4)
print(cf_report)

print("\n时间统计汇总:")
total_train_time = pre_train_time + cluster_train_time + adapt_train_time
total_time = pre_train_time + cluster_train_time + adapt_train_time + test_time
print(f"全过程耗时:{total_time:.2f}秒")
print(f"+---训练阶段总耗时:{total_train_time:.2f}秒")
print(f"|   +---预训练阶段:{pre_train_time:.2f}秒")
print(f"|   +---聚类损失训练阶段:{cluster_train_time:.2f}秒")
print(f"|   +---自适应训练阶段:{adapt_train_time:.2f}秒")
print(f"+---测试阶段总耗时:{test_time:.2f}秒")

cf_matrix_Proposed_PSD = confusion_matrix(all_true_labels, all_preds)
classes=['Negative', 'Positive']
df_cm_Proposed_PSD = pd.DataFrame(cf_matrix_Proposed_PSD, index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (5,5), dpi = 300)
# sn.set(font_scale=1.5)
ax=sn.heatmap(df_cm_Proposed_PSD, annot=True, fmt='.5g', cmap="crest")
ax.set(xlabel="Predicted Label", ylabel="True Label")

# 保存confusion_matrix图片
# 添加包含模式和EPOCH信息的整体标题
plt.title(f'混淆矩阵 (训练模式: {mode})', fontproperties=chinese_font)

plt.savefig(f'{pic_filename}cm_res{timestamp_suffix}.png')  # 保存为PNG格式
# plt.savefig(f'./output/pic/cm_res{timestamp_suffix}.pdf')  # 同时保存为PDF格式(矢量图，适合论文使用)

