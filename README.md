# TaskLevelAug
A data-augmentation method for meta-learning

### Abstract

Data augmentation is one of the most effective approaches for improving the accuracy of modern machine learning models, and it is also indispensable to train a deep model for meta-learning. In this paper, we introduce a task augmentation method by rotating, which increases the number of classes by rotating the original images 90, 180 and 270 degrees, different from traditional augmentation methods which increase the number of images. With a larger amount of classes, we can sample more diverse task instances during training. Therefore, task augmentation by rotating allows us to train a deep network by meta-learning methods with little over-fitting. Experimental results show that our approach is better than the rotation for increasing the number of images and achieves state-of-the-art performance on miniImageNet, CIFAR-FS, and FC100 few-shot learning benchmarks. The code is available on \url{www.github.com/AceChuse/TaskLevelAug}.

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



## Acknowledgments

This code is based on the implementations of [**Prototypical Networks**](https://github.com/cyvius96/prototypical-network-pytorch),  [**Dynamic Few-Shot Visual Learning without Forgetting**](https://github.com/gidariss/FewShotWithoutForgetting), [**DropBlock**](https://github.com/miguelvr/dropblock), [**qpth**](https://github.com/locuslab/qpth), and [**MetaOptNet**](https://github.com/kjunelee/MetaOptNet).
