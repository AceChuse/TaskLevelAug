from __future__ import print_function

import numpy as np
import random
import math
import multiprocessing
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms.functional as TranF

from .utils import ProtoData


class DualCategories(data.Dataset):
    def __init__(self, dataset, p=0.5, std=0.1, batch_size_down=4e4):
        self.dataset = dataset
        self.std = std
        self.batch_num = multiprocessing.Value("d", -1.)
        self.batch_size_down = batch_size_down

        self.phase = self.dataset.phase
        self.num_cats_new = self.dataset.num_cats * (self.dataset.num_cats - 1) // 2
        self.num_cats = self.dataset.num_cats + self.num_cats_new

        if p == -1:
            self.p = float(self.num_cats_new) / self.num_cats
        else:
            self.p = p

    def sampleCategories(self, sample_size):
        self.batch_num.value += 1
        p = self.p * (self.batch_size_down - self.batch_num.value) / self.batch_size_down
        sample1_size = np.sum(np.random.rand(sample_size) > p)
        sample2_size = sample_size - sample1_size
        sample1 = np.random.choice(self.dataset.num_cats, sample1_size, replace=False)
        sample2 = np.random.choice(self.num_cats_new, sample2_size, replace=False) + self.dataset.num_cats
        return list(sample1) + list(sample2)

    def sampleImageIdsFrom(self, cat_id, sample_size=1):
        """
        Samples `sample_size` number of unique image ids picked from the
        category `cat_id` (i.e., self.dataset.label2ind[cat_id]).

        Args:
            cat_id: a scalar with the id of the category from which images will
                be sampled.
            sample_size: number of images that will be sampled.

        Returns:
            image_ids: a list of length `sample_size` with unique image ids.
                Each id is a 2-elements tuples. The 1st element of each tuple
                is the augment process mark and the 2nd element is the image
                loading related information..
        """
        if cat_id < self.dataset.num_cats:
            return [(False, d_id) for d_id in self.dataset.sampleImageIdsFrom(
                self.dataset.labelIds[cat_id], sample_size)]
        else:
            for cat_id in range(self.dataset.num_cats, self.dataset.num_cats + self.num_cats_new):
                cat_id = cat_id - self.dataset.num_cats
                cat1_id = int((-1 + math.sqrt(1 + 8 * cat_id)) / 2)
                cat2_id = int(cat_id - cat1_id * (cat1_id + 1) / 2)
                cat1_id += 1
            ids1 = self.dataset.sampleImageIdsFrom(self.dataset.labelIds[cat1_id], sample_size)
            ids2 = self.dataset.sampleImageIdsFrom(self.dataset.labelIds[cat2_id], sample_size)
            return [(True, d_id) for d_id in zip(ids1, ids2)]

    def createExamplesTensorData(self, examples):
        """
        Creates the examples image and label tensor data.

        Args:
            examples: a list of 2-element tuples, each representing a
                train or test example. The 1st element of each tuple
                is the image id of the example and 2nd element is the
                category label of the example, which is in the range
                [0, nK - 1], where nK is the total number of categories
                (both novel and base).

        Returns:
            images: a tensor of shape [nExamples, Height, Width, 3] with the
                example images, where nExamples is the number of examples
                (i.e., nExamples = len(examples)).
            labels: a tensor of shape [nExamples] with the category label
                of each example.
        """

        dataset = self.dataset
        def get_image(img_idx):
            if img_idx[0]:
                img_idx = img_idx[1]
                image1 = dataset[img_idx[0]][0]
                image2 = dataset[img_idx[1]][0]
                shift = np.random.normal(loc=0, scale=self.std)
                return image1 * (shift / 2.) + image2 * ((1 - shift) / 2.)
            else:
                return dataset[img_idx[1]][0]

        images = torch.stack(
            [get_image(img_idx) for img_idx, _ in examples], dim=0)
        labels = torch.LongTensor([label for _, label in examples])
        return images, labels

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'p=' + str(self.p) + ', ' \
               + 'std=' + str(self.std) + ', ' \
               + 'phase=' + str(self.phase) + ', ' \
               + 'num_cats_new=' + str(self.num_cats_new) + ', ' \
               + 'num_cats=' + str(self.num_cats) + ', ' \
               + 'batch_size_down=' + str(self.batch_size_down) + ')'


class PermuteChannels(data.Dataset):
    def __init__(self, dataset, p=-1, ):
        self.dataset = dataset

        self.phase = self.dataset.phase
        self.num_cats_new = self.dataset.num_cats * 5
        self.num_cats = self.dataset.num_cats + self.num_cats_new

        self.orders = (torch.LongTensor([0, 1, 2]), torch.LongTensor([1, 2, 0]),
                       torch.LongTensor([2, 0, 1]), torch.LongTensor([1, 0, 2]),
                       torch.LongTensor([0, 2, 1]), torch.LongTensor([2, 1, 0]))

        if p == -1:
            self.p = 5./6.
        else:
            self.p = p

    def sampleCategories(self, sample_size):
        sample = np.random.choice(self.dataset.num_cats, sample_size, replace=False)
        if random.random() < self.p:
            sample += (np.random.choice(5, 1, replace=False)[0] + 1) * self.dataset.num_cats
        return sample

    # def sampleCategories(self, sample_size):
    #     sample1_size = np.sum(np.random.rand(sample_size) > self.p)
    #     sample2_size = sample_size - sample1_size
    #     sample1 = np.random.choice(self.dataset.num_cats, sample1_size, replace=False)
    #     sample2 = np.random.choice(self.num_cats_new, sample2_size, replace=False) + self.dataset.num_cats
    #     return list(sample1) + list(sample2)

    def sampleImageIdsFrom(self, cat_id, sample_size=1):
        """
        Samples `sample_size` number of unique image ids picked from the
        category `cat_id` (i.e., self.dataset.label2ind[cat_id]).

        Args:
            cat_id: a scalar with the id of the category from which images will
                be sampled.
            sample_size: number of images that will be sampled.

        Returns:
            image_ids: a list of length `sample_size` with unique image ids.
                Each id is a 2-elements tuples. The 1st element of each tuple
                is the augment process mark and the 2nd element is the image
                loading related information..
        """
        perm_id = cat_id // self.dataset.num_cats
        cat_id = cat_id % self.dataset.num_cats
        return [(perm_id, d_id) for d_id in self.dataset.sampleImageIdsFrom(
            self.dataset.labelIds[cat_id], sample_size)]

    def createExamplesTensorData(self, examples):
        """
        Creates the examples image and label tensor data.

        Args:
            examples: a list of 2-element tuples, each representing a
                train or test example. The 1st element of each tuple
                is the image id of the example and 2nd element is the
                category label of the example, which is in the range
                [0, nK - 1], where nK is the total number of categories
                (both novel and base).

        Returns:
            images: a tensor of shape [nExamples, Height, Width, 3] with the
                example images, where nExamples is the number of examples
                (i.e., nExamples = len(examples)).
            labels: a tensor of shape [nExamples] with the category label
                of each example.
        """

        dataset = self.dataset

        images = torch.stack(
            [dataset[img_idx[1]][0][self.orders[img_idx[0]]] for img_idx, _ in examples], dim=0)
        labels = torch.LongTensor([label for _, label in examples])
        return images, labels

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'p=' + str(self.p) + ', ' \
               + 'phase=' + str(self.phase) + ', ' \
               + 'num_cats_new=' + str(self.num_cats_new) + ', ' \
               + 'num_cats=' + str(self.num_cats) + ')'


class Rot90(data.Dataset):
    def __init__(self, dataset, p=-1, batch_size_down=8e4):
        self.dataset = dataset
        self.batch_num = multiprocessing.Value("d", -1.)
        self.batch_size_down = batch_size_down
        #self.rate = (lambda x, y: 1.) if batch_size_down == 0 else (lambda x, y: min(1., x / y))

        self.phase = self.dataset.phase
        self.num_cats_new = self.dataset.num_cats * 3
        self.num_cats = self.dataset.num_cats + self.num_cats_new

        if p == -1:
            self.p = float(self.num_cats_new) / self.num_cats
        else:
            self.p = p

    # def sampleCategories(self, sample_size):
    #     sample = np.random.choice(self.dataset.num_cats, sample_size, replace=False)
    #     if random.random() < self.p:
    #         sample += (np.random.choice(3, 1, replace=False)[0] + 1) * self.dataset.num_cats
    #     return sample

    def sampleCategories(self, sample_size):
        self.batch_num.value += 1.
        p = self.p * min(1., self.batch_num.value / self.batch_size_down)#self.rate(self.batch_num.value, self.batch_size_down)
        sample1_size = np.sum(np.random.rand(sample_size) > p)
        sample2_size = sample_size - sample1_size
        sample1 = np.random.choice(self.dataset.num_cats, sample1_size, replace=False)
        sample2 = np.random.choice(self.num_cats_new, sample2_size, replace=False) + self.dataset.num_cats
        return list(sample1) + list(sample2)

    # def sampleCategories(self, sample_size):
    #     sample = np.random.choice(self.dataset.num_cats, sample_size, replace=False)
    #     sample1_size = np.sum(np.random.rand(sample_size) > self.p)
    #
    #     sample1 = sample[:sample1_size]
    #     sample2 = sample[sample1_size:] + (np.random.choice(3, 1, replace=False)[0] + 1) * self.dataset.num_cats
    #
    #     return list(sample1) + list(sample2)

    def sampleImageIdsFrom(self, cat_id, sample_size=1):
        """
        Samples `sample_size` number of unique image ids picked from the
        category `cat_id` (i.e., self.dataset.label2ind[cat_id]).

        Args:
            cat_id: a scalar with the id of the category from which images will
                be sampled.
            sample_size: number of images that will be sampled.

        Returns:
            image_ids: a list of length `sample_size` with unique image ids.
                Each id is a 2-elements tuples. The 1st element of each tuple
                is the augment process mark and the 2nd element is the image
                loading related information..
        """
        rot90_id = int(cat_id // self.dataset.num_cats)
        cat_id = cat_id % self.dataset.num_cats
        return [(rot90_id, d_id) for d_id in self.dataset.sampleImageIdsFrom(
            self.dataset.labelIds[cat_id], sample_size)]

    def createExamplesTensorData(self, examples):
        """
        Creates the examples image and label tensor data.

        Args:
            examples: a list of 2-element tuples, each representing a
                train or test example. The 1st element of each tuple
                is the image id of the example and 2nd element is the
                category label of the example, which is in the range
                [0, nK - 1], where nK is the total number of categories
                (both novel and base).

        Returns:
            images: a tensor of shape [nExamples, Height, Width, 3] with the
                example images, where nExamples is the number of examples
                (i.e., nExamples = len(examples)).
            labels: a tensor of shape [nExamples] with the category label
                of each example.
        """

        dataset = self.dataset

        images = torch.stack(
            [torch.rot90(dataset[img_idx[1]][0], img_idx[0], [1, 2]) for img_idx, _ in examples], dim=0)
        labels = torch.LongTensor([label for _, label in examples])
        return images, labels

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'p=' + str(self.p) + ', ' \
               + 'phase=' + str(self.phase) + ', ' \
               + 'num_cats_new=' + str(self.num_cats_new) + ', ' \
               + 'num_cats=' + str(self.num_cats) + ', ' \
               + 'batch_size_down=' + str(self.batch_size_down) + ')'


class AspectRatio(data.Dataset):
    def __init__(self, dataset, ratiomm, p=-1, batch_size_down=8e4, interpolation=Image.BILINEAR):
        assert(isinstance(dataset, ProtoData))
        self.dataset = dataset
        self.batch_num = multiprocessing.Value("d", -1.)
        self.batch_size_down = batch_size_down

        if ratiomm[0] <= 1. or ratiomm[1] <= 1.:
            raise ValueError('The range of ratio should be greater than 1!')
        self.ratiomm = ratiomm
        self.interpolation = interpolation
        self.img_size = self.dataset.img_size

        self.transform = self.dataset.transform
        self.dataset.transform = None

        self.phase = self.dataset.phase
        self.num_cats_new = self.dataset.num_cats * 2
        self.num_cats = self.dataset.num_cats + self.num_cats_new

        if p == -1:
            self.p = float(self.num_cats_new) / self.num_cats
        else:
            self.p = p

    @staticmethod
    def get_params(img_size, ratio, width=True):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image): Image to be cropped.
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped
            width (bool): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
        aspect_ratio = math.exp(random.uniform(*log_ratio))

        if width:
            w = img_size[0]
            h = int(round(img_size[1] / aspect_ratio))
        else:
            w = int(round(img_size[0] / aspect_ratio))
            h = img_size[1]

        if w <= img_size[0] and h <= img_size[1]:
            i = random.randint(0, img_size[1] - h)
            j = random.randint(0, img_size[0] - w)
            return i, j, h, w

    def trans_ratio(self, img, ar_id):
        if ar_id == 0:
            return img
        elif ar_id == 1:
            i, j, h, w = self.get_params(self.img_size, self.ratiomm, width=True)
        elif ar_id == 2:
            i, j, h, w = self.get_params(self.img_size, self.ratiomm, width=False)
        return TranF.resized_crop(img, i, j, h, w, self.img_size, self.interpolation)

    def sampleCategories(self, sample_size):
        self.batch_num.value += 1.
        p = self.p * min(1., self.batch_num.value / self.batch_size_down)
        sample1_size = np.sum(np.random.rand(sample_size) > p)
        sample2_size = sample_size - sample1_size
        sample1 = np.random.choice(self.dataset.num_cats, sample1_size, replace=False)
        sample2 = np.random.choice(self.num_cats_new, sample2_size, replace=False) + self.dataset.num_cats
        return list(sample1) + list(sample2)

    def sampleImageIdsFrom(self, cat_id, sample_size=1):
        """
        Samples `sample_size` number of unique image ids picked from the
        category `cat_id` (i.e., self.dataset.label2ind[cat_id]).

        Args:
            cat_id: a scalar with the id of the category from which images will
                be sampled.
            sample_size: number of images that will be sampled.

        Returns:
            image_ids: a list of length `sample_size` with unique image ids.
                Each id is a 2-elements tuples. The 1st element of each tuple
                is the augment process mark and the 2nd element is the image
                loading related information..
        """
        ar_id = int(cat_id // self.dataset.num_cats)
        cat_id = cat_id % self.dataset.num_cats
        return [(ar_id, d_id) for d_id in self.dataset.sampleImageIdsFrom(
            self.dataset.labelIds[cat_id], sample_size)]

    def createExamplesTensorData(self, examples):
        """
        Creates the examples image and label tensor data.

        Args:
            examples: a list of 2-element tuples, each representing a
                train or test example. The 1st element of each tuple
                is the image id of the example and 2nd element is the
                category label of the example, which is in the range
                [0, nK - 1], where nK is the total number of categories
                (both novel and base).

        Returns:
            images: a tensor of shape [nExamples, Height, Width, 3] with the
                example images, where nExamples is the number of examples
                (i.e., nExamples = len(examples)).
            labels: a tensor of shape [nExamples] with the category label
                of each example.
        """

        dataset = self.dataset

        images = [self.trans_ratio(dataset[img_idx[1]][0], img_idx[0]) for img_idx, _ in examples]
        if self.transform is not None:
            images = [self.transform(img) for img in images]
        images = torch.stack(images, dim=0)
        labels = torch.LongTensor([label for _, label in examples])
        return images, labels

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'p=' + str(self.p) + ', ' \
               + 'phase=' + str(self.phase) + ', ' \
               + 'ratiomm=' + str(self.ratiomm) + ', ' \
               + 'num_cats_new=' + str(self.num_cats_new) + ', ' \
               + 'num_cats=' + str(self.num_cats) + ', ' \
               + 'batch_size_down=' + str(self.batch_size_down) + ')'
