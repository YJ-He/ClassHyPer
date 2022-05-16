# **ClassHyPer: ClassMix-Based Hybrid Perturbations for Deep Semi-Supervised Semantic Segmentation of  Remote Sensing Imagery**

The PyTorch implementation of semi-supervised learning method—ClassHyPer.
The manuscript can be visited via https://www.mdpi.com/2072-4292/14/4/879

## 1. Datasets
### (1) Links
* [DeepGlobe Road](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset)
* [Massachusetts Building](https://www.cs.toronto.edu/~vmnih/data)
* [WHU Aerial Building](http://study.rsgis.whu.edu.cn/pages/download/building_dataset.html)
* [ISPRS 2D Semantic Labeling (Potsdam and Vaihingen)](https://www.isprs.org/education/benchmarks/UrbanSemLab/default.aspx)
### (2) Directory Structure    
After obtain the datasets, you need to process first and generate lists of image/label files and place as the structure shown below. Every txt file contains the full absolute path of the files, each image/label per line. Example files can be found  in `./examples`.
```
/root
    /train_image.txt
    /train_label.txt
    /test_image.txt
    /test_label.txt
    /val_image.txt
    /val_label.txt
    /train_unsup_image.txt
``` 

## 2. Usage
### 2.1 Installation
The code is developed using Python 3.8 with PyTorch 1.9.0 on Windows 10. The code is developed and tested using single RTX 2080 Ti GPU.

**(1) Clone this repo.**
```
git clone https://github.com/YJ-He/ClassHyPer.git
```

**(2) Create a conda environment.**  
```
conda env create -f environment.yaml
conda activate class_hyper
```

### 2.2 Training
1. set `root_dir` and hyper-parameters configuration in `./configs/config.cfg`.
2. run `python train.py`.

### 2.3 Evaludation
1. set `root_dir` and hyper-parameters configuration in `./configs/config.cfg`.
2. set `pathCkpt` in `test.py` to indicate the model checkpoint file.
3. run `python test.py`.

## 3. Structure of ClassHyPer
<img src="figs/ClassHyPer.jpg" width="800px" hight="1400px" />

---
## 4. Citation
If this repo is useful in your research, please kindly consider citing our paper as follow.
```
@article{he2022classhyper,
  title={ClassHyPer: ClassMix-Based Hybrid Perturbations for Deep Semi-Supervised Semantic Segmentation of Remote Sensing Imagery},
  author={He, Yongjun and Wang, Jinfei and Liao, Chunhua and Shan, Bo and Zhou, Xin},
  journal={Remote Sensing},
  volume={14},
  number={4},
  pages={879},
  year={2022},
  publisher={MDPI}
}
```

##  5. References
[1] [Semi-Supervised Semantic Segmentation with Cross Pseudo Supervision](https://arxiv.org/abs/2106.01226)  
[2] [Semi-supervised semantic segmentation needs strong, varied perturbations](https://arxiv.org/abs/1906.01916)  
[3] [ClassMix: Segmentation-Based Data Augmentation for Semi-Supervised Learning](https://arxiv.org/abs/2007.07936)  
...  

**If our work give you some insights and hints, star me please! Thank you~**


