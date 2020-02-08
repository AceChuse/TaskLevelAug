# TaskLevelAug
A data-augmentation method for meta-learning

### Abstract

Data augmentation is one of the most effective approaches for improving the accuracy of modern machine learning models, and it is also indispensable to train a deep model for meta-learning. In this paper, we introduce a task augmentation method by rotating, which increases the number of classes by rotating the original images 90, 180 and 270 degrees, different from traditional augmentation methods which increase the number of images. With a larger amount of classes, we can sample more diverse task instances during training. Therefore, task augmentation by rotating allows us to train a deep network by meta-learning methods with little over-fitting. Experimental results show that our approach is better than the rotation for increasing the number of images and achieves state-of-the-art performance on miniImageNet, CIFAR-FS, and FC100 few-shot learning benchmarks. 

### Dependencies
This code requires the following:
* python 3.\*
* Pytorch 1.1.0+

## Usage

### Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/AceChuse/TaskLevelAug.git
    cd TaskLevelAug
    ```
2. Download and decompress dataset files: [**miniImageNet**](https://drive.google.com/file/d/1fJAK5WZTjerW7EWHHQAR9pRJVNg1T1Y7/view?usp=sharing) (courtesy of [**Spyros Gidaris**](https://github.com/gidariss/FewShotWithoutForgetting)), [**FC100**](https://drive.google.com/file/d/1_ZsLyqI487NRDQhwvI7rg86FK3YAZvz1/view?usp=sharing), [**CIFAR-FS**](https://drive.google.com/file/d/1GjGMI0q3bgcpcB_CjI40fX54WgLPuTpS/view?usp=sharing)

3. For each dataset loader, specify the path to the directory. For example, in TaskLevelAugGithub/metadatas/mini_imagenet.py line 9:
    ```python
    _MINI_IMAGENET_DATASET_DIR = filepath + '/DataSets/miniImageNet_numpy'
    ```

### Automated training and testing
#### CIFAR-FS
Automated training R2D2 on 5-way 5-shot 6-query and testing on 5-way 5-shot and 5-way 1-shot.
```bash
$ python CIFAR_FS_auto.py
```

#### FC100
Automated training R2D2 on 5-way 5-shot 6-query and testing on 5-way 5-shot and 5-way 1-shot.
```bash
$ python FC100_auto.py
```

#### MiniImageNet
Automated training R2D2 on 5-way 5-shot 6-query and testing on 5-way 5-shot and 5-way 1-shot.
```bash
$ python mini_auto.py
```

#### Result
The average accuracies (\%) with 95\% confidence intervals on CIFAR-FS. 

| |  Method | 1-shot| 5-shot |
| -----   | -----  | ----  |
|MAML |  Base. | +RE |
|R2-D2 |  7.21 | 6.73 |
|ProtoNets |  6.41 | 5.66 |
|M-SVM |  72.8 | 85.0 |
|M-SVM (best) (our) |  5.53 | 5.13 |
|R2-D2 (best) (our) |  5.31 | 4.89|


### Manual Operation
#### CIFAR-FS
Training:
To train R2-D2 on 5-way 5-shot 6-query train set, and to test on 5-way 5-shot 15-query validation set.
```bash
python meta_e.py --dataset=CIFAR_FS --mode=train --epochs=60 --task_aug Rot90 --rot90_p=0.5 \ 
--feat_aug=norm --start_epoch=0 -es=8000 -b=8 -pn=8 --lossf=cross_entropy --eps=0.0 --optim=SGD \ 
--lr_sche=lambda_epoch --lr=0.1 --wd=0.0005 --embedding=ResNet12 --head=R2D2 --kway=5 --trshot=5 \
--trquery=6 --vshot=5 --vquery=15 --teshot=1 --tequery=15 --seed=0
```

Testing:
To test R2-D2 on 5-way 5-shot 15-query test set.
```bash
python meta_e.py --dataset=CIFAR_FS --mode=ens_test --epochs=60 --task_aug Rot90 --rot90_p=0.5 \
--feat_aug=norm --start_epoch=0 -es=8000 -b=8 -pn=8 --lossf=cross_entropy --eps=0.0 --optim=SGD \
--lr_sche=lambda_epoch --lr=0.1 --wd=0.0005 --embedding=ResNet12 --head=R2D2 --kway=5 --trshot=5 \
--trquery=6 --vshot=5 --vquery=15 --teshot=5 --tequery=15 --seed=0
```

To test R2-D2 on 5-way 1-shot 15-query test set.
```bash
python meta_e.py --dataset=CIFAR_FS --mode=ens_test --epochs=60 --task_aug Rot90 --rot90_p=0.5 \
--feat_aug=norm --start_epoch=0 -es=8000 -b=8 -pn=8 --lossf=cross_entropy --eps=0.0 --optim=SGD \
--lr_sche=lambda_epoch --lr=0.1 --wd=0.0005 --embedding=ResNet12 --head=R2D2 --kway=5 --trshot=5 \
--trquery=6 --vshot=1 --vquery=15 --teshot=1 --tequery=15 --seed=0
```

Retraining:
To train R2-D2 on 5-way 5-shot 6-query train set and validation set.
```bash
python meta_e.py --dataset=CIFAR_FS --mode=final --epochs=60 --task_aug Rot90 --rot90_p=0.5 \
--feat_aug=norm --start_epoch=0 -es=8000 -b=8 -pn=8 --lossf=cross_entropy --eps=0.0 --optim=SGD \
--lr_sche=lambda_epoch --lr=0.1 --wd=0.0005 --embedding=ResNet12 --head=R2D2 --kway=5 --trshot=5 \
--trquery=6 --vshot=5 --vquery=15 --teshot=1 --tequery=15 --seed=0
```

Retesting:
To test R2-D2 on 5-way 5-shot 15-query test set.
```bash
python meta_e.py --dataset=CIFAR_FS --mode=ens_testac --epochs=60 --task_aug Rot90 --rot90_p=0.5 \
--feat_aug=norm --start_epoch=0 -es=8000 -b=8 -pn=8 --lossf=cross_entropy --eps=0.0 --optim=SGD \
--lr_sche=lambda_epoch --lr=0.1 --wd=0.0005 --embedding=ResNet12 --head=R2D2 --kway=5 --trshot=5 \
--trquery=6 --vshot=5 --vquery=15 --teshot=5 --tequery=15 --seed=0
```

To test R2D2 on 5-way 1-shot 15-query test set.
```bash
python meta_e.py --dataset=CIFAR_FS --mode=ens_testac --epochs=60 --task_aug Rot90 --rot90_p=0.5 \
--feat_aug=norm --start_epoch=0 -es=8000 -b=8 -pn=8 --lossf=cross_entropy --eps=0.0 --optim=SGD \
--lr_sche=lambda_epoch --lr=0.1 --wd=0.0005 --embedding=ResNet12 --head=R2D2 --kway=5 --trshot=5 \
--trquery=6 --vshot=1 --vquery=15 --teshot=1 --tequery=15 --seed=0
```

#### FC100
Training:
To train R2-D2 on 5-way 5-shot 6-query train set, and to test on 5-way 5-shot 15-query validation set.
```bash
python meta_e.py --dataset=FC100 --mode=train --epochs=60 --task_aug Rot90 --rot90_p=0.25 \
--feat_aug=norm --start_epoch=0 -es=8000 -b=8 -pn=8 --lossf=cross_entropy --eps=0.0 --optim=SGD \
--lr_sche=lambda_epoch --lr=0.1 --wd=0.0005 --embedding=ResNet12 --head=R2D2 --kway=5 --trshot=15 \
--trquery=6 --vshot=5 --vquery=15 --teshot=1 --tequery=15 --seed=0
```

Testing: 
To test R2-D2 on 5-way 5-shot 15-query test set.
```bash
python meta_e.py --dataset=FC100 --mode=ens_test --epochs=60 --task_aug Rot90 --rot90_p=0.25 \
--feat_aug=norm --start_epoch=0 -es=8000 -b=8 -pn=8 --lossf=cross_entropy --eps=0.0 --optim=SGD \
--lr_sche=lambda_epoch --lr=0.1 --wd=0.0005 --embedding=ResNet12 --head=R2D2 --kway=5 --trshot=15 \
--trquery=6 --vshot=5 --vquery=15 --teshot=5 --tequery=15 --seed=0
```

To test R2-D2 on 5-way 1-shot 15-query test set.
```bash
python meta_e.py --dataset=FC100 --mode=ens_test --epochs=60 --task_aug Rot90 --rot90_p=0.25 \
--feat_aug=norm --start_epoch=0 -es=8000 -b=8 -pn=8 --lossf=cross_entropy --eps=0.0 --optim=SGD \
--lr_sche=lambda_epoch --lr=0.1 --wd=0.0005 --embedding=ResNet12 --head=R2D2 --kway=5 --trshot=15 \
--trquery=6 --vshot=1 --vquery=15 --teshot=1 --tequery=15 --seed=0
```

Retraining:
To train R2-D2 on 5-way 5-shot 6-query train set and validation set.
```bash
python meta_e.py --dataset=FC100 --mode=final --epochs=60 --task_aug Rot90 --rot90_p=0.25 \
--feat_aug=norm --start_epoch=0 -es=8000 -b=8 -pn=8 --lossf=cross_entropy --eps=0.0 --optim=SGD \
--lr_sche=lambda_epoch --lr=0.1 --wd=0.0005 --embedding=ResNet12 --head=R2D2 --kway=5 --trshot=15 \
--trquery=6 --vshot=5 --vquery=15 --teshot=1 --tequery=15 --seed=0
```

Retesting:
To test R2-D2 on 5-way 5-shot 15-query test set.
```bash
python meta_e.py --dataset=FC100 --mode=ens_testac --epochs=60 --task_aug Rot90 --rot90_p=0.25 \
--feat_aug=norm --start_epoch=0 -es=8000 -b=8 -pn=8 --lossf=cross_entropy --eps=0.0 --optim=SGD \
--lr_sche=lambda_epoch --lr=0.1 --wd=0.0005 --embedding=ResNet12 --head=R2D2 --kway=5 --trshot=15 \
--trquery=6 --vshot=5 --vquery=15 --teshot=5 --tequery=15 --seed=0
```

To test R2-D2 on 5-way 1-shot 15-query test set.
```bash
python meta_e.py --dataset=FC100 --mode=ens_testac --epochs=60 --task_aug Rot90 --rot90_p=0.25 \
--feat_aug=norm --start_epoch=0 -es=8000 -b=8 -pn=8 --lossf=cross_entropy --eps=0.0 --optim=SGD \
--lr_sche=lambda_epoch --lr=0.1 --wd=0.0005 --embedding=ResNet12 --head=R2D2 --kway=5 --trshot=15 \
--trquery=6 --vshot=1 --vquery=15 --teshot=1 --tequery=15 --seed=0
```

#### MiniImageNet
Training:
To train R2-D2 on 5-way 5-shot 6-query train set, and to test on 5-way 5-shot 15-query validation set.
```bash
python meta_e.py --dataset=miniImageNet --mode=train --epochs=60 --task_aug Rot90 --rot90_p=0.25 \
--feat_aug=norm --start_epoch=0 -es=8000 -b=8 -pn=2 --lossf=cross_entropy --eps=0.0 --optim=SGD \
--lr_sche=lambda_epoch --lr=0.1 --wd=0.0005 --embedding=ResNet12 --head=R2D2 --kway=5 --trshot=15 \
--trquery=6 --vshot=1 --vquery=15 --teshot=1 --tequery=15 --seed=0
```

Testing:
To test R2-D2 on 5-way 5-shot 15-query test set.
```bash
python meta_e.py --dataset=miniImageNet --mode=ens_test --epochs=60 --task_aug Rot90 --rot90_p=0.25 \
--feat_aug=norm --start_epoch=0 -es=8000 -b=8 -pn=2 --lossf=cross_entropy --eps=0.0 --optim=SGD \
--lr_sche=lambda_epoch --lr=0.1 --wd=0.0005 --embedding=ResNet12 --head=R2D2 --kway=5 --trshot=15 \
--trquery=6 --vshot=5 --vquery=15 --teshot=5 --tequery=15 --seed=0
```

To test R2-D2 on 5-way 1-shot 15-query test set.
```bash
python meta_e.py --dataset=miniImageNet --mode=ens_test --epochs=60 --task_aug Rot90 --rot90_p=0.25 \
--feat_aug=norm --start_epoch=0 -es=8000 -b=8 -pn=2 --lossf=cross_entropy --eps=0.0 --optim=SGD \
--lr_sche=lambda_epoch --lr=0.1 --wd=0.0005 --embedding=ResNet12 --head=R2D2 --kway=5 --trshot=15 \
--trquery=6 --vshot=1 --vquery=15 --teshot=1 --tequery=15 --seed=0
```

Retraining:
To train R2-D2 on 5-way 5-shot 6-query train set and validation set.
```bash
python meta_e.py --dataset=miniImageNet --mode=final --epochs=60 --task_aug Rot90 --rot90_p=0.25 \
--feat_aug=norm --start_epoch=0 -es=8000 -b=8 -pn=2 --lossf=cross_entropy --eps=0.0 --optim=SGD \
--lr_sche=lambda_epoch --lr=0.1 --wd=0.0005 --embedding=ResNet12 --head=R2D2 --kway=5 --trshot=15 \
--trquery=6 --vshot=5 --vquery=15 --teshot=1 --tequery=15 --seed=0
```

Retesting:
To test R2-D2 on 5-way 5-shot 15-query test set.
```bash
python meta_e.py --dataset=miniImageNet --mode=ens_testac --epochs=60 --task_aug Rot90 --rot90_p=0.25 \
--feat_aug=norm --start_epoch=0 -es=8000 -b=8 -pn=2 --lossf=cross_entropy --eps=0.0 --optim=SGD \
--lr_sche=lambda_epoch --lr=0.1 --wd=0.0005 --embedding=ResNet12 --head=R2D2 --kway=5 --trshot=15 \
--trquery=6 --vshot=5 --vquery=15 --teshot=5 --tequery=15 --seed=0
```

To test R2-D2 on 5-way 1-shot 15-query test set.
```bash
python meta_e.py --dataset=miniImageNet --mode=ens_testac --epochs=60 --task_aug Rot90 --rot90_p=0.25 \
 --feat_aug=norm --start_epoch=0 -es=8000 -b=8 -pn=2 --lossf=cross_entropy --eps=0.0 --optim=SGD \
 --lr_sche=lambda_epoch --lr=0.1 --wd=0.0005 --embedding=ResNet12 --head=R2D2 --kway=5 --trshot=15 \
 --trquery=6 --vshot=1 --vquery=15 --teshot=1 --tequery=15 --seed=0
```

## Acknowledgments

This code is based on the implementations of [**Prototypical Networks**](https://github.com/cyvius96/prototypical-network-pytorch),  [**Dynamic Few-Shot Visual Learning without Forgetting**](https://github.com/gidariss/FewShotWithoutForgetting), [**DropBlock**](https://github.com/miguelvr/dropblock), [**qpth**](https://github.com/locuslab/qpth), and [**MetaOptNet**](https://github.com/kjunelee/MetaOptNet).
