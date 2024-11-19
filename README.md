# Enhanced cross-dataset electroencephalogram-based emotion recognition using unsupervised domain adaptation
This repository contains the code accompanying our paper [Enhanced Cross-Dataset Electroencephalogram-based Emotion Recognition using Unsupervised Domain Adaptation](https://www.sciencedirect.com/science/article/pii/S0010482524014793) published in Computers in Biology and Medicine. Our work presents a domain-adaptive deep network for EEG-based emotion classification, aiming to improve cross-domain model performance by addressing feature distribution discrepancies. We introduce a sample selection technique to reduce negative transfer and propose a cost-effective test-time augmentation method to enhance test performance.

### Setup Environment

The experiments were conducted using the following environment and packages:

- PyTorch == 2.1.0+cu121<br />
- Python == 3.10.12<br />
- NumPy == 1.26.4<br />
- pandas == 2.2.2<br />
- scikit-learn == 1.5.2<br />
- SciPy == 1.13.1

### Data Preparation

1.	Download the following EEG Datasets:<br />
 [DEAP](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html)<br />
 [SEED](https://bcmi.sjtu.edu.cn/home/seed/)<br />
2.	Unzip the data and organize it according to the following directory structure:<br />
```
    data
    +---DEAP
    ¦   ¦   sXX.dat
    ¦   +---preprocessed
    ¦           Data_Orig_sub_XX.npy
    ¦           DE_sub_XX.npy
    ¦           labels_sub_XX.npy
    ¦           PSD_sub_XX.npy       
    +---SEED
        ¦   label.mat
        ¦   X_1.mat
        ¦   X_2.mat
        ¦   X_3.mat 
        +---preprocessed
                Data_Orig_sub_X.npy
                DE_sub_X.npy
                labels_sub_X.npy
                PSD_sub_X.npy
 ```               

The preprocessed directory stores the data generated during the preprocessing stage in the code.

### Execute Code
Run main.py to reproduce the results. This script will automatically link and execute the other .py files as needed.


### Citation
If you find this code useful, please consider citing our work:

```bibtex
@article{imtiaz2025enhanced,
  title={Enhanced cross-dataset electroencephalogram-based emotion recognition using unsupervised domain adaptation},
  author={Imtiaz, Md Niaz and Khan, Naimul},
  journal={Computers in Biology and Medicine},
  volume={184},
  pages={109394},
  year={2025},
  publisher={Elsevier}
}


