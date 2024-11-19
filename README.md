This repository contains the code accompanying our paper Enhanced Cross-Dataset Electroencephalogram-based Emotion Recognition using Unsupervised Domain Adaptation published in Computers in Biology and Medicine. Our work presents a domain-adaptive deep network for EEG-based emotion classification, aiming to improve cross-domain model performance by addressing feature distribution discrepancies. We introduce a sample selection technique to reduce negative transfer and propose a cost-effective test-time augmentation method to enhance test performance.

### Setup Environment

The experiments were conducted using the following environment and packages:

- PyTorch version == 2.1.0+cu121<br />
- Python version == 3.10.12<br />
- NumPy version == 1.26.4<br />
- pandas version == 2.2.2<br />
- scikit-learn version == 1.5.2<br />
- SciPy version == 1.13.1

### Data Preparation

1.	Download the following EEG Datasets:<br />
 [DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html)<br />
 [SEED](https://bcmi.sjtu.edu.cn/home/seed/)<br />
2.	Unzip the data and organize it according to the following directory structure.<br />

![Screenshot 2024-11-18 202854](https://github.com/user-attachments/assets/708aa4fd-2070-46bd-b82b-fa11333a210f)

The preprocessed directory stores the data generated during the preprocessing stage in the code.

### Execute Code
Run main.py to reproduce the results. This script will automatically link and execute the other .py files as needed.


### Citation
If you find this code useful, please consider citing our work:

\cite{IMTIAZ2025109394}
[Uploading S0010482524014793.bib…]()@article{IMTIAZ2025109394,
title = {Enhanced cross-dataset electroencephalogram-based emotion recognition using unsupervised domain adaptation},
journal = {Computers in Biology and Medicine},
volume = {184},
pages = {109394},
year = {2025},
issn = {0010-4825},
doi = {https://doi.org/10.1016/j.compbiomed.2024.109394},
url = {https://www.sciencedirect.com/science/article/pii/S0010482524014793},
author = {Md Niaz Imtiaz and Naimul Khan},
keywords = {Electroencephalogram (EEG), Emotion recognition, Unsupervised domain adaptation, Brain–computer interface (BCI), Test-time augmentation (TTA)},
abstract = {Emotion recognition holds great promise in healthcare and in the development of affect-sensitive systems such as brain–computer interfaces (BCIs). However, the high cost of labeled data and significant differences in electroencephalogram (EEG) signals among individuals limit the cross-domain application of EEG-based emotion recognition models. Addressing cross-dataset scenarios poses greater challenges due to changes in subject demographics, recording devices, and stimuli presented. To tackle these challenges, we propose an improved method for classifying EEG-based emotions across domains with different distributions. We propose a Gradual Proximity-guided Target Data Selection (GPTDS) technique, which gradually selects reliable target domain samples for training based on their proximity to the source clusters and the model’s confidence in predicting them. This approach avoids negative transfer caused by diverse and unreliable samples. Additionally, we introduce a cost-effective test-time augmentation (TTA) technique named Prediction Confidence-aware Test-Time Augmentation (PC-TTA). Traditional TTA methods often face substantial computational burden, limiting their practical utility. By applying TTA only when necessary, based on the model’s predictive confidence, our approach improves the model’s performance during inference while minimizing computational costs compared to traditional TTA approaches. Experiments on the DEAP and SEED datasets demonstrate that our method outperforms state-of-the-art approaches, achieving accuracies of 67.44% when trained on DEAP and tested on SEED, and 59.68% vice versa, with improvements of 7.09% and 6.07% over the baseline. It excels in detecting both positive and negative emotions, highlighting its effectiveness for practical emotion recognition in healthcare applications. Moreover, our proposed PC-TTA technique reduces computational time by a factor of 15 compared to traditional full TTA approaches.}
}


