# Enhanced cross-dataset electroencephalogram-based emotion recognition using unsupervised domain adaptation
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
[@imtiaz2025enhanced]

@article{imtiaz2025enhanced,
  title={Enhanced cross-dataset electroencephalogram-based emotion recognition using unsupervised domain adaptation},
  author={Imtiaz, Md Niaz and Khan, Naimul},
  journal={Computers in Biology and Medicine},
  volume={184},
  pages={109394},
  year={2025},
  publisher={Elsevier}
}


