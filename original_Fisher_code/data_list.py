#from __future__ import print_function, division

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
from PIL import Image
import torch.utils.data as data
import os
import os.path
from collections import Counter
from sklearn.model_selection import train_test_split


def stratify_sampling(image_list, ratio=1.0):
    '''stratify sampling a subset from the input dataset with a given ratio
    Args: 
        image_list: [list], each element is a str, "image_file_path label"
        num_labels_per_class: [int], number of labeled sample per class, if -1, then return a labeled dataset
    Returns:
        sampled_list: [list], the same structure as `image_list`
    '''
    assert(ratio > 0. and ratio <= 1.0)
    if ratio == 1.:
        return image_list
    images = [val.strip().split()[0] for val in image_list]
    labels = [int(val.strip().split()[1]) for val in image_list]
    assert(len(images) == len(labels))
    # print('image size={}, label size={}'.format(len(images),len(labels)))
    num_classes = len(np.unique(labels))
    labeled_images, _, labeled_y, _ = train_test_split(images, labels, 
        train_size=ratio, stratify=labels, random_state=1)
    return [image_name + " " + str(image_label) for image_name, image_label in zip(labeled_images, labeled_y)]


def make_dataset(image_list, labels):
    if labels is not None:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def make_class_balanced_labeled_dataset(image_list, num_labels_per_class=-1):
    '''split the dataset into labeled and unlabeled sets
    Args: 
        image_list: [list], each element is a str, "image_file_path label"
        num_labels_per_class: [int], number of labeled sample per class, if -1, then return a labeled dataset
    Returns:
        labeled_images: list of str, path to labeled images
        labeled_y: np.array with shape (num_samples,1)
        unlabeled_images
        unlabeled_y
    '''
    if num_labels_per_class == -1:
        return [val.split()[0] for val in image_list], np.array([int(val.split()[1]) for val in image_list]).reshape(-1,1), None, None
    elif num_labels_per_class == 0:
        return None, None, [val.split()[0] for val in image_list], np.array([int(val.split()[1]) for val in image_list]).reshape(-1,1)
    else:
        images = [val.split()[0] for val in image_list]
        labels = [int(val.split()[1]) for val in image_list]
        assert(len(images) == len(labels))
        # print('image size={}, label size={}'.format(len(images),len(labels)))
        num_classes = len(np.unique(labels))
        labeled_images, unlabeled_images, labeled_y, unlabeled_y = train_test_split(images, labels, 
            train_size=num_labels_per_class*num_classes, stratify=labels, random_state=1)
        return labeled_images, np.array(labeled_y).reshape(-1,1), unlabeled_images, np.array(unlabeled_y).reshape(-1,1)


def make_triplet_dataset(image_list):
    '''
    Args: 
        image_list: file object. (image_filename, label) in each row
    Returns: 
        triplets: a list with (x1, y1, sim_label) triplets n*n elements
    '''
    ds = []
    for line in image_list:
        image_name, label = line.strip().split()
        label = int(label)
        ds.append([image_name, label])

    triplets = []
    c = Counter()
    for x1, y1 in ds:
        list_to_match = list(ds)
        list_to_match.remove([x1, y1])
        for x2, y2 in list_to_match:
            sim_label = 1-int(y1 == y2)
            triplets.append([x1, x2, sim_label])
            c.update([sim_label])
    num_samples = float(sum(c.values()))
    weight_per_class = {class_label:num_samples/cnt for class_label, cnt in c.items()}
    weights = [weight_per_class[t[-1]] for t in triplets]
    return triplets, weights


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    #from torchvision import get_image_backend
    #if get_image_backend() == 'accimage':
    #    return accimage_loader(path)
    #else:
        return pil_loader(path)


class ImageList(object):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, image_list, labels=None, transform=None, target_transform=None,
                 loader=default_loader):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)


class SiameseImageList(object):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, image_list, transform=None, target_transform=None,
                 loader=default_loader):
        imgs, sample_weights = make_triplet_dataset(image_list)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.sample_weights = sample_weights
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image_a, image_b, target) where target is sim_label.
        """
        x1_path, x2_path, target = self.imgs[index]
        img_a = self.loader(x1_path)
        img_b = self.loader(x2_path)
        if self.transform is not None:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img_a, img_b, target

    def __len__(self):
        return len(self.imgs)


class ImageValueList(object):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, image_list, labels=None, transform=None, target_transform=None,
                 loader=default_loader):
        imgs = make_dataset(image_list, labels)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs
        self.values = [1.0] * len(imgs)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def set_values(self, values):
        self.values = values

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

