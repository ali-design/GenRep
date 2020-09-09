from __future__ import print_function


import numpy as np
import torch
from PIL import Image
from skimage import color
from torchvision import datasets
from torchvision import transforms


class STL10Instance(datasets.STL10):
    """STL10 instance dataset"""
    def __init__(self, root, split='train',
                 transform=None, target_transform=None, download=False, separate=False):
        super(STL10Instance, self).__init__(root=root, split=split, transform=transform,
                                            target_transform=target_transform, download=download)
        self.separate = separate

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            if not self.separate:
                img = self.transform(img)
            else:
                img1 = self.transform(img)
                img2 = self.transform(img)
                l = img1[0, :].unsqueeze(0)
                ab = img2[1:, :]
                img = torch.cat([l, ab], dim=0)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class STL10MINE(datasets.STL10):
    """STL10 dataset for mutual information neural estimation"""
    def __init__(self, root, split='train',
                 transform=None, target_transform=None, download=False, two_crop=False):
        super(STL10MINE, self).__init__(root=root, split=split, transform=transform,
                                        target_transform=target_transform, download=download)
        self.two_crop = two_crop

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            image = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.two_crop:
            image2 = self.transform(img)
            image = torch.cat([image, image2], dim=0)

        return image, target, index


class STL10Supervised(datasets.STL10):
    """STL10 supervised dataset"""
    def __init__(self, root, split='train',
                 transform=None, target_transform=None, download=False, two_crop=False):
        super(STL10Supervised, self).__init__(root=root, split=split, transform=transform,
                                              target_transform=target_transform, download=download)
        assert split == 'train' or split == 'test', 'wrong split: {}'.format(split)
        self.two_crop = two_crop
        self.num_data = len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        index = index % self.num_data
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            image = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.two_crop:
            image2 = self.transform(img)
            image = torch.cat([image, image2], dim=0)

        return image, target

    def __len__(self):
        """hand-crafted number for longer loading"""
        return 105000 * 1000


class RGB2RGB(object):
    """Convert RGB PIL image to RGB."""
    def __call__(self, img):
        return img


class RGB2Lab(object):
    """Convert RGB PIL image to ndarray Lab."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2lab(img)
        return img


class RGB2YDbDr(object):
    """Convert RGB PIL image to ndarray YDbDr."""
    def __call__(self, img):
        img = np.asarray(img, np.uint8)
        img = color.rgb2ydbdr(img)
        return img


class RandomTranslateWithReflect:
    """
    Translate image randomly
    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].
    Fill the uncovered blank area with reflect padding.
    """

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))
        return new_image


class CIFAR10Instance(datasets.CIFAR10):
    """
    CIFAR10Instance+Sample Dataset
    """
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, k=4096, mode='NoSample', two_crop=False):
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)
        self.k = k
        self.mode = mode
        self.two_crop = two_crop
        self.num = self.__len__()

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.two_crop:
            img2 = self.transform(image)
            img = torch.cat([img, img2], dim=0)

        if self.mode == 'NoSample':
            return img, target, index
        elif self.mode == 'Replace':
            neg_idx = np.random.choice(self.num, self.k, replace=True)
            shift = self.get_shift(index, 0, self.num - 1)
            neg_idx[neg_idx == index] = index + shift
            sample_idx = np.hstack((np.asarray([index]), neg_idx))
            return img, target, index, sample_idx
        elif self.mode == 'NoReplace':
            neg_idx = np.random.choice(self.num, self.k, replace=False)
            shift = self.get_shift(index, 0, self.num - 1)
            neg_idx[neg_idx == index] = index + shift
            sample_idx = np.hstack((np.asarray([index]), neg_idx))
            return img, target, index, sample_idx
        else:
            raise NotImplementedError('cifar mode not supported: {}'.format(self.mode))

    @staticmethod
    def get_shift(index, min, max):
        if index == min:
            shift = 1
        elif index == max:
            shift = -1
        else:
            shift = 1 if np.random.rand() > 0.5 else -1
        return shift


class CIFAR100Instance(datasets.CIFAR100):
    """
    CIFAR100Instance+Sample Dataset
    """
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, k=4096, mode='NoSample', two_crop=False):
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)
        self.k = k
        self.mode = mode
        self.two_crop = two_crop
        self.num = self.__len__()

    def __getitem__(self, index):
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        image = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.two_crop:
            img2 = self.transform(image)
            img = torch.cat([img, img2], dim=0)

        if self.mode == 'NoSample':
            return img, target, index
        elif self.mode == 'Random':
            neg_idx = np.random.choice(self.num, self.k, replace=True)
            sample_idx = np.hstack((np.asarray([index]), neg_idx))
            return img, target, index, sample_idx
        elif self.mode == 'Replace':
            neg_idx = np.random.choice(self.num, self.k, replace=True)
            shift = self.get_shift(index, 0, self.num - 1)
            neg_idx[neg_idx == index] = index + shift
            sample_idx = np.hstack((np.asarray([index]), neg_idx))
            return img, target, index, sample_idx
        elif self.mode == 'NoReplace':
            neg_idx = np.random.choice(self.num, self.k, replace=False)
            shift = self.get_shift(index, 0, self.num - 1)
            neg_idx[neg_idx == index] = index + shift
            sample_idx = np.hstack((np.asarray([index]), neg_idx))
            return img, target, index, sample_idx
        else:
            raise NotImplementedError('cifar mode not supported: {}'.format(self.mode))

    @staticmethod
    def get_shift(index, min, max):
        if index == min:
            shift = 1
        elif index == max:
            shift = -1
        else:
            shift = 1 if np.random.rand() > 0.5 else -1
        return shift


if __name__ == '__main__':
    # data_folder = '/data/vision/billf/scratch/yltian/datasets'
    # train_transform = transforms.Compose([
    #     transforms.RandomCrop(64),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.43, 0.42, 0.39), std=(0.27, 0.26, 0.27)),
    # ])
    # train_dataset = STL10MINE(root=data_folder,
    #                           download=True,
    #                           split='train+unlabeled',
    #                           transform=train_transform,
    #                           two_crop=True)
    #
    # image, target, index = train_dataset[50000]
    # print(image.shape)
    # print(target)
    # print(index)

    data_folder = '/data/vision/billf/scratch/yltian/datasets'
    train_transform = transforms.Compose([
        transforms.RandomCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.43, 0.42, 0.39), std=(0.27, 0.26, 0.27)),
    ])
    train_dataset = STL10Supervised(root=data_folder,
                                    transform=train_transform)

    from torch.utils.data.dataloader import DataLoader, _MultiProcessingDataLoaderIter

    train_loader_1 = DataLoader(train_dataset,
                                batch_size=16,
                                shuffle=True,
                                num_workers=4)

    train_loader_2 = _MultiProcessingDataLoaderIter(train_loader_1)

    # for idx, (data, target) in enumerate(train_loader_1):
    #     print(data.shape)
    #     print(target)
    #     break
    #
    # for idx, (data, target) in enumerate(train_loader_1):
    #     print(data.shape)
    #     print(target)
    #     break

    # for i in range(2):
    #     (data, target) = next(train_loader_2)
    #     print(target)
    #     print('done')
    #
    # for i in range(2):
    #     (data, target) = next(train_loader_2)
    #     print(target)
    #     print('done')

    print(len(train_dataset))
    for i in range(1000):
        (data, target) = next(train_loader_2)
        if i % 100 == 0:
            print('{} done'.format(i))

