# 模式切换功能说明

## 功能概述

本项目实现了两种不同的迁移学习模式：跨数据集迁移（cross_dataset）和跨用户迁移（cross_subject）。通过命令行参数可以灵活切换这两种模式，实现不同场景下的情绪识别迁移学习。

## 模式说明

### 跨数据集模式 (cross_dataset)
- **源域**: SEED数据集
- **目标域**: DEAP数据集
- **应用场景**: 将模型从一个EEG情绪数据集迁移到另一个具有不同记录设备和实验范式的数据集
- **特点**: 解决跨数据集因采集设备、电极位置、实验任务不同带来的域漂移问题

### 跨用户模式 (cross_subject)
- **源域**: SEED数据集的部分用户数据
- **目标域**: SEED数据集未参与训练的其他用户数据
- **应用场景**: 适用于将模型训练在某些人的EEG数据上，然后应用到新用户身上
- **特点**: 解决不同人之间脑电信号模式差异导致的个体化问题

## 数据处理差异

两种模式下的数据处理存在一些关键差异：

1. **标签处理**:
   - SEED数据集: 原始标签为三分类（消极，中性，积极），通过筛选只保留标签为消极情绪和积极情绪的样本，转换为二分类（中性情绪难以定义）
   - DEAP数据集: 原始标签为连续值，根据4.5阈值将连续评分转换为二分类

2. **测试阶段**:
   - 跨数据集模式: 在DEAP数据集上测试，需要将连续标签转换为二分类
   - 跨用户模式: 在未参与训练的SEED用户数据上测试

## 使用方法

通过命令行参数`--mode`指定运行模式:

```bash
# 跨数据集模式 (默认)
python main.py --mode cross_dataset

# 跨用户模式
python main.py --mode cross_subject

# 带时间戳的运行
python main.py --mode cross_subject --timestamp 5_9_15_30

# 不挂断运行
nohup python main.py --mode cross_subject > /dev/null 2> ./output/txt/error.log &
```
