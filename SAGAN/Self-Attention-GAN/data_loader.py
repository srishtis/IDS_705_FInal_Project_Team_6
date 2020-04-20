import torch
import torchvision.datasets as dsets
from torchvision import transforms
import numpy as np
import os
from torchvision.utils import save_image

class Data_Loader():
    def __init__(self, train, dataset, image_path, image_size, batch_size, shuf=True):
        self.dataset = dataset
        self.path    = image_path
        self.imsize  = image_size
        self.batch   = batch_size
        self.shuf    = shuf
        self.train   = train

    def transform(self, resize, totensor, grey2rgb, normalize, centercrop):
        options = []
        if centercrop:
            options.append(transforms.CenterCrop(64))
        if resize:
            options.append(transforms.Resize((self.imsize, self.imsize)))
        if totensor:
            options.append(transforms.ToTensor())
        if grey2rgb:
            options.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
        if normalize:
            options.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

        transform = transforms.Compose(options)
        return transform

    def load_lsun(self, classes='church_outdoor_train'):
        transforms = self.transform(True, True, False, True, False)
        dataset = dsets.LSUN(self.path, classes=[classes], transform=transforms)
        return dataset

    def load_celeb(self):
        transforms = self.transform(True, True, False, True, True)
        dataset = dsets.ImageFolder(self.path+'/CelebA', transform=transforms)
        return dataset

    def load_cifar10(self):
        # Crop(32)
        transforms = self.transform(True, True, False, True, False)
        dataset = dsets.CIFAR10('~/data', train=True, transform=transforms, target_transform=None, download=True)
        idx = dataset.class_to_idx['horse']
        dataset = [(data[0], data[1]) for data in dataset if data[1] == idx]
        data = torch.stack([x[0] for x in dataset])
        # print(data.shape)
        for i, img in enumerate(data):
            save_image(img, os.path.join('./data/CIFAR-10/horse', "{}.png".format(i)))
        return dataset

    def load_pokemon(self):
        transforms = self.transform(True, True, False, True, False)
        dataset = dsets.ImageFolder(self.path + '/pokemon', transform=transforms)
        return dataset

    def load_fashion_mnist(self):
        # Crop(28)
        transforms = self.transform(True, True, True, False, True)
        dataset = dsets.FashionMNIST('~/data', train=True, transform=transforms, target_transform=None, download=True)
        return dataset

    def loader(self):
        if self.dataset == 'lsun':
            dataset = self.load_lsun()
        elif self.dataset == 'celeb':
            dataset = self.load_celeb()
        elif self.dataset == 'cifar-10':
            dataset = self.load_cifar10()
        elif self.dataset == 'pokemon':
            dataset = self.load_pokemon()
        elif self.dataset == 'fashion-mnist':
            dataset = self.load_fashion_mnist()

        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=self.batch,
                                             shuffle=self.shuf,
                                             num_workers=8,
                                             drop_last=True)
        return loader


if __name__ == '__main__':
    print('hello')
